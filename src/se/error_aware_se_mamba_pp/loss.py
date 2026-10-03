from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from src.se.common.phase_loss import _anti_wrapping, phase_loss_gradient_matrix
from src.se.common.stft import Spec
from src.se.error_aware_se_mamba_pp.discriminator import DiscriminatorOutputs


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
    def get_mel_filters(
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        fmin: float,
        fmax: float | None,
    ) -> np.ndarray:
        return librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

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
        mel_basis = torch.from_numpy(self.get_mel_filters(self.sampling_rate, 2 * (n_freq - 1), n_mels, 0, None)).to(
            wav.device
        )
        return (magnitude.transpose(2, -1) @ mel_basis.T).transpose(-1, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=x.device, dtype=x.dtype)
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
    loss = torch.tensor(0.0)
    for real_maps, fake_maps in zip(fmap_r, fmap_g):
        for real, fake in zip(real_maps, fake_maps):
            loss = loss + torch.mean(torch.abs(real - fake))
    return loss * 2


def lsgan_discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> torch.Tensor:
    loss = torch.tensor(0.0)
    for real, fake in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - real) ** 2) + torch.mean(fake**2)
    return loss


def lsgan_generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0)
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
    tdt: torch.Tensor,
    weights: SimpleNamespace,
    n_fft: int,
) -> dict[str, torch.Tensor]:
    """SEMamba++ generator loss: adversarial, feature matching, mel, spectral, and TDT terms. No time term."""
    ip_loss, gd_loss, iaf_loss = phase_loss_gradient_matrix(clean.pha, gen.pha, n_fft)
    losses = {
        "magnitude": F.mse_loss(clean.mag, gen.mag),
        "phase": ip_loss + gd_loss + iaf_loss,
        "complex": F.mse_loss(clean.com, gen.com) * 2,
        "consistency": F.mse_loss(gen.com, gen_hat.com) * 2,
        "adv_g": lsgan_generator_loss(d.cqt_fake) + lsgan_generator_loss(d.mrd_fake),
        "fm_g": feature_loss(d.cqt_fmap_real, d.cqt_fmap_fake) + feature_loss(d.mrd_fmap_real, d.mrd_fmap_fake),
        "mel": mel,
        "tdt": tdt,
    }
    losses["total"] = (
        weights.magnitude * losses["magnitude"]
        + weights.phase * losses["phase"]
        + weights.complex * losses["complex"]
        + weights.consistency * losses["consistency"]
        + weights.adv_g * losses["adv_g"]
        + weights.fm_g * losses["fm_g"]
        + weights.mel * losses["mel"]
        + weights.tdt * losses["tdt"]
    )
    return losses


def phase_maps(
    phase_r: torch.Tensor,
    phase_g: torch.Tensor,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unreduced maps from phase_loss_gradient_matrix. ip/iaf are [B, F, T], gd is [B, T, F]."""
    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)
    if phase_r.size(1) != dim_freq or phase_g.shape != phase_r.shape:
        raise ValueError(f"phase shape {tuple(phase_r.shape)} does not match n_fft={n_fft} or the pair")
    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq, device=phase_g.device), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq, device=phase_g.device), diagonal=2)
        - torch.eye(dim_freq, device=phase_g.device)
    )
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time, device=phase_g.device), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time, device=phase_g.device), diagonal=2)
        - torch.eye(dim_time, device=phase_g.device)
    )
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    ip = _anti_wrapping(phase_r - phase_g)
    gd = _anti_wrapping(gd_r - gd_g)
    iaf = _anti_wrapping(iaf_r - iaf_g)
    return ip, gd, iaf


def frame_weight_and_valid(
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    n_frames: int,
    hop: int,
    alpha: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """w = 1 + alpha * m. m is 1 when frame center j*hop lies in the error samples.

    A frame is valid when it belongs to center=True analysis of the unpadded length.
    """
    if sample_error.ndim != 2:
        raise ValueError(f"sample_error must be [B, T], got {tuple(sample_error.shape)}")
    if content_length.ndim != 1 or content_length.size(0) != sample_error.size(0):
        raise ValueError("content_length must be [B] aligned with sample_error")
    if n_frames < 1 or hop <= 0:
        raise ValueError(f"n_frames={n_frames} hop={hop}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    device = sample_error.device
    content = content_length.to(device=device, dtype=torch.long)
    centers = torch.arange(n_frames, device=device) * hop
    n_valid = torch.where(content > 0, torch.div(content, hop, rounding_mode="floor") + 1, torch.zeros_like(content))
    frame_index = torch.arange(n_frames, device=device).view(1, -1)
    valid = frame_index < n_valid.view(-1, 1)
    inside = centers.view(1, -1) < content.view(-1, 1)
    in_tensor = centers < sample_error.size(1)
    if sample_error.size(1) == 0:
        belongs = torch.zeros(sample_error.size(0), n_frames, dtype=torch.bool, device=device)
    else:
        index = centers.clamp(max=sample_error.size(1) - 1)
        belongs = sample_error[:, index] & inside & in_tensor.view(1, -1) & valid
    weight = belongs.to(dtype=dtype) * float(alpha) + 1.0
    return weight, valid


def weighted_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor,
    time_dim: int,
) -> torch.Tensor:
    """sum(valid * w * l) / sum(valid * w). Padding is absent from both sums."""
    if values.size(0) != weight.size(0) or values.size(time_dim) != weight.size(1):
        raise ValueError(f"time axis mismatch values {tuple(values.shape)} dim {time_dim} weight {tuple(weight.shape)}")
    gate = weight.to(dtype=values.dtype) * valid.to(device=values.device, dtype=values.dtype)
    view = [1] * values.ndim
    view[0] = gate.size(0)
    view[time_dim] = gate.size(1)
    broadcast = gate.view(*view).expand_as(values)
    present = broadcast > 0
    masked = torch.where(present, values, torch.zeros_like(values))
    numer = (masked * broadcast).sum()
    denom = broadcast.sum()
    zero = numer * 0
    return torch.where(denom > 0, numer / denom.clamp_min(torch.finfo(values.dtype).tiny), zero)


def _weighted_mel(
    mel_module: MultiScaleMelSpectrogramLoss,
    clean_audio: torch.Tensor,
    enhanced_audio: torch.Tensor,
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    loss = clean_audio.new_zeros(())
    log10 = torch.log(clean_audio.new_tensor(10.0))
    clean = clean_audio.unsqueeze(1)
    enhanced = enhanced_audio.unsqueeze(1)
    for n_mels, window_length in zip(mel_module.n_mels, mel_module.window_lengths):
        hop = window_length // 4
        clean_log = torch.log(mel_module.mel_spectrogram(clean, n_mels, window_length).clamp(min=mel_module.clamp_eps))
        enhanced_log = torch.log(
            mel_module.mel_spectrogram(enhanced, n_mels, window_length).clamp(min=mel_module.clamp_eps)
        )
        diff = (clean_log / log10 - enhanced_log / log10).abs()
        weight, valid = frame_weight_and_valid(sample_error, content_length, diff.size(-1), hop, alpha, diff.dtype)
        loss = loss + weighted_mean(diff, weight, valid, -1)
    return loss


def _weighted_total(losses: dict[str, torch.Tensor], weights: SimpleNamespace) -> torch.Tensor:
    return (
        weights.magnitude * losses["magnitude"]
        + weights.phase * losses["phase"]
        + weights.complex * losses["complex"]
        + weights.consistency * losses["consistency"]
        + weights.adv_g * losses["adv_g"]
        + weights.fm_g * losses["fm_g"]
        + weights.mel * losses["mel"]
        + weights.tdt * losses["tdt"]
    )


def error_aware_generator_loss(
    clean: Spec,
    gen: Spec,
    gen_hat: Spec,
    d: DiscriminatorOutputs,
    tdt: torch.Tensor,
    weights: SimpleNamespace,
    n_fft: int,
    mel_module: MultiScaleMelSpectrogramLoss,
    clean_audio: torch.Tensor,
    enhanced_audio: torch.Tensor,
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    stft_hop: int,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Local spectral and mel terms use center weights. adv_g, fm_g, and tdt are unchanged."""
    ip, gd, iaf = phase_maps(clean.pha, gen.pha, n_fft)
    n_time = clean.mag.size(-1)
    weight, valid = frame_weight_and_valid(sample_error, content_length, n_time, stft_hop, alpha, clean.mag.dtype)
    losses = {
        "magnitude": weighted_mean((clean.mag - gen.mag).square(), weight, valid, 2),
        "phase": (
            weighted_mean(ip, weight, valid, 2)
            + weighted_mean(gd, weight, valid, 1)
            + weighted_mean(iaf, weight, valid, 2)
        ),
        "complex": weighted_mean((clean.com - gen.com).square(), weight, valid, 2) * 2,
        "consistency": weighted_mean((gen.com - gen_hat.com).square(), weight, valid, 2) * 2,
        "adv_g": lsgan_generator_loss(d.cqt_fake) + lsgan_generator_loss(d.mrd_fake),
        "fm_g": feature_loss(d.cqt_fmap_real, d.cqt_fmap_fake) + feature_loss(d.mrd_fmap_real, d.mrd_fmap_fake),
        "mel": _weighted_mel(mel_module, clean_audio, enhanced_audio, sample_error, content_length, alpha),
        "tdt": tdt,
    }
    losses["total"] = _weighted_total(losses, weights)
    return losses


def phase_maps_match_reference(phase_r: torch.Tensor, phase_g: torch.Tensor, n_fft: int) -> None:
    ip, gd, iaf = phase_maps(phase_r, phase_g, n_fft)
    ip_ref, gd_ref, iaf_ref = phase_loss_gradient_matrix(phase_r, phase_g, n_fft)
    if not torch.allclose(ip.mean(), ip_ref) or not torch.allclose(gd.mean(), gd_ref):
        raise RuntimeError("phase maps do not match phase_loss_gradient_matrix")
    if not torch.allclose(iaf.mean(), iaf_ref):
        raise RuntimeError("IAF map does not match phase_loss_gradient_matrix")
