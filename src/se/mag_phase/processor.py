from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.se.mag_phase.layout import complex_from_mag_pha


class MagPhaseProcessor:
    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_size: int,
        compress_factor: float,
        stft_kind: str,
        reconstruct_add_eps: bool,
    ) -> None:
        if stft_kind not in ("mp_senet", "abs_angle"):
            raise ValueError(
                f"stft_kind must be 'mp_senet' or 'abs_angle', got {stft_kind!r}"
            )
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.stft_kind = stft_kind
        self.reconstruct_add_eps = reconstruct_add_eps

    def encode(
        self,
        waveform: torch.Tensor,
        add_eps: bool = False,
    ) -> SimpleNamespace:
        if waveform.ndim != 2:
            raise ValueError(
                f"waveform must be [B, T], got shape {tuple(waveform.shape)}"
            )
        hann_window = torch.hann_window(self.win_size).to(waveform.device)
        stft_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        if self.stft_kind == "mp_senet":
            spec_real = torch.view_as_real(stft_spec)
            mag = torch.sqrt(spec_real.pow(2).sum(-1) + 1e-9)
            pha = torch.atan2(spec_real[..., 1] + 1e-10, spec_real[..., 0] + 1e-5)
        elif add_eps:
            mag = torch.sqrt(stft_spec.real.pow(2) + stft_spec.imag.pow(2) + 1e-10)
            pha = torch.atan2(stft_spec.imag + 1e-10, stft_spec.real + 1e-10)
        else:
            mag = torch.abs(stft_spec)
            pha = torch.angle(stft_spec)
        mag = torch.pow(mag, self.compress_factor)
        com = complex_from_mag_pha(mag, pha)
        return SimpleNamespace(mag=mag, pha=pha, com=com)

    def decode(
        self,
        features: SimpleNamespace,
        length: int | None = None,
    ) -> torch.Tensor:
        mag = torch.pow(features.mag, 1.0 / self.compress_factor)
        com = torch.complex(
            mag * torch.cos(features.pha),
            mag * torch.sin(features.pha),
        )
        hann_window = torch.hann_window(self.win_size).to(com.device)
        audio = torch.istft(
            com,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=True,
        )
        if length is None:
            return audio
        if audio.size(1) > length:
            return audio[:, :length]
        if audio.size(1) < length:
            return F.pad(audio, (0, length - audio.size(1)))
        return audio
