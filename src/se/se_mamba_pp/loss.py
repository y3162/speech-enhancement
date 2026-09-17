from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from src.se.common.phase_loss import phase_loss_gradient_matrix
from src.se.se_mamba_pp.discriminator import DiscriminatorOutputs
from src.se.common.stft import Spec


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Sum of log10-mel L1 over several window lengths. Input is waveform [B, 1, T]."""

    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_mels = [40, 80, 160, 320]
        self.window_lengths = [256, 512, 1024, 2048]
        self.clamp_eps = 1e-5

    @staticmethod
    @lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(self, wav: torch.Tensor, n_mels: int, window_length: int) -> torch.Tensor:
        batch, channels, time = wav.shape
        stft = torch.stft(
            wav.reshape(-1, time),
            n_fft=window_length,
            hop_length=window_length // 4,
            window=torch.hann_window(window_length, device=wav.device),
            return_complex=True,
            center=True,
        )
        _, n_freq, n_time = stft.shape
        magnitude = torch.abs(stft.reshape(batch, channels, n_freq, n_time))
        mel_basis = torch.from_numpy(
            self.get_mel_filters(self.sampling_rate, 2 * (n_freq - 1), n_mels, 0, None)
        ).to(wav.device)
        return (magnitude.transpose(2, -1) @ mel_basis.T).transpose(-1, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        log10 = torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))
        for n_mels, window_length in zip(self.n_mels, self.window_lengths):
            x_logmels = torch.log(self.mel_spectrogram(x, n_mels, window_length).clamp(min=self.clamp_eps)) / log10
            y_logmels = torch.log(self.mel_spectrogram(y, n_mels, window_length).clamp(min=self.clamp_eps)) / log10
            loss = loss + F.l1_loss(x_logmels, y_logmels)
        return loss


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    loss = 0
    for real_maps, fake_maps in zip(fmap_r, fmap_g):
        for real, fake in zip(real_maps, fake_maps):
            loss += torch.mean(torch.abs(real - fake))
    return loss * 2


def lsgan_discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> torch.Tensor:
    loss = 0
    for real, fake in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - real) ** 2) + torch.mean(fake ** 2)
    return loss


def lsgan_generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for fake in disc_outputs:
        loss = loss + torch.mean((1 - fake) ** 2)
    return loss


def discriminator_loss(d: DiscriminatorOutputs) -> dict[str, torch.Tensor]:
    """LSGAN: real -> 1, fake -> 0. d is the discriminator applied to detached generated audio."""
    losses = {
        "cqt": lsgan_discriminator_loss(d.cqt_real, d.cqt_fake),
        "mrd": lsgan_discriminator_loss(d.mrd_real, d.mrd_fake),
    }
    losses["total"] = losses["cqt"] + losses["mrd"]
    return losses


def generator_loss(
    clean: Spec,
    gen: Spec,
    gen_hat: Spec,
    d: DiscriminatorOutputs,
    mel: torch.Tensor,
    w,
    n_fft: int,
) -> dict[str, torch.Tensor]:
    """SEMamba++ generator loss: adversarial, feature matching, mel, and spectral terms. No time term."""
    ip_loss, gd_loss, iaf_loss = phase_loss_gradient_matrix(clean.pha, gen.pha, n_fft)
    losses = {
        "magnitude": F.mse_loss(clean.mag, gen.mag),
        "phase": ip_loss + gd_loss + iaf_loss,
        "complex": F.mse_loss(clean.com, gen.com) * 2,
        "consistency": F.mse_loss(gen.com, gen_hat.com) * 2,
        "adv_g": lsgan_generator_loss(d.cqt_fake) + lsgan_generator_loss(d.mrd_fake),
        "fm_g": feature_loss(d.cqt_fmap_real, d.cqt_fmap_fake) + feature_loss(d.mrd_fmap_real, d.mrd_fmap_fake),
        "mel": mel,
    }
    losses["total"] = (
        w.magnitude * losses["magnitude"]
        + w.phase * losses["phase"]
        + w.complex * losses["complex"]
        + w.consistency * losses["consistency"]
        + w.adv_g * losses["adv_g"]
        + w.fm_g * losses["fm_g"]
        + w.mel * losses["mel"]
    )
    return losses
