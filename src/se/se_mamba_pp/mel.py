from functools import lru_cache

import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_mels = [40, 80, 160, 320]
        self.window_lengths = [256, 512, 1024, 2048]
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = 1e-5

    @staticmethod
    @lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav: torch.Tensor,
        n_mels: int,
        window_length: int,
    ) -> torch.Tensor:
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
        stft = stft.reshape(batch, channels, n_freq, n_time)
        magnitude = torch.abs(stft)
        mel_basis = torch.from_numpy(
            self.get_mel_filters(
                self.sampling_rate,
                2 * (n_freq - 1),
                n_mels,
                0,
                None,
            )
        ).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        return mel_spectrogram.transpose(-1, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_mels, window_length in zip(self.n_mels, self.window_lengths):
            x_mels = self.mel_spectrogram(x, n_mels, window_length)
            y_mels = self.mel_spectrogram(y, n_mels, window_length)
            x_logmels = torch.log(x_mels.clamp(min=self.clamp_eps)) / torch.log(
                torch.tensor(10.0, device=x.device, dtype=x.dtype)
            )
            y_logmels = torch.log(y_mels.clamp(min=self.clamp_eps)) / torch.log(
                torch.tensor(10.0, device=y.device, dtype=y.dtype)
            )
            loss = loss + self.loss_fn(x_logmels, y_logmels)
        return loss
