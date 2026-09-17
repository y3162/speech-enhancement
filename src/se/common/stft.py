from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn.functional as F


class Spec(NamedTuple):
    mag: torch.Tensor
    pha: torch.Tensor
    com: torch.Tensor


def complex_from_mag_pha(mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
    return torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)


def stack_mag_pha(mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (mag.transpose(1, 2).unsqueeze(1), pha.transpose(1, 2).unsqueeze(1)),
        dim=1,
    )


def mag_pha_stft(waveform: torch.Tensor, stft: SimpleNamespace, eps: float | None = None) -> Spec:
    """Compressed magnitude, phase, and complex spec from a [B, T] waveform.

    If eps is set, it is added inside sqrt / atan2 so zero bins do not produce NaN gradients.
    Only the re-analysis of generated audio uses eps=1e-10.
    """
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be [B, T], got shape {tuple(waveform.shape)}")
    spec = torch.stft(
        waveform,
        n_fft=stft.n_fft,
        hop_length=stft.hop_size,
        win_length=stft.win_size,
        window=torch.hann_window(stft.win_size, device=waveform.device),
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    if eps is None:
        mag = torch.abs(spec)
        pha = torch.angle(spec)
    else:
        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + eps)
        pha = torch.atan2(spec.imag + eps, spec.real + eps)
    mag = torch.pow(mag, stft.compress_factor)
    return Spec(mag, pha, complex_from_mag_pha(mag, pha))


def mag_pha_istft(
    mag: torch.Tensor,
    pha: torch.Tensor,
    stft: SimpleNamespace,
    length: int | None = None,
) -> torch.Tensor:
    mag = torch.pow(mag, 1.0 / stft.compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    audio = torch.istft(
        com,
        n_fft=stft.n_fft,
        hop_length=stft.hop_size,
        win_length=stft.win_size,
        window=torch.hann_window(stft.win_size, device=com.device),
        center=True,
    )
    if length is None:
        return audio
    if audio.size(1) > length:
        return audio[:, :length]
    return F.pad(audio, (0, length - audio.size(1)))
