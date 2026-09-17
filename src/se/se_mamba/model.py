from types import SimpleNamespace

import torch
import torch.nn as nn

from src.se.common.activations import LearnableSigmoid2d
from src.se.common.dense import DenseEncoder, MagDecoder, PhaseDecoder
from src.se.common.mamba import MambaBlock
from src.se.common.stft import Spec, complex_from_mag_pha, stack_mag_pha


class TFMambaBlock(nn.Module):
    """Bidirectional Mamba along time then frequency. I/O [B, C, T, F]."""

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        hid_feature = cfg.hid_feature
        self.time_mamba = MambaBlock(hid_feature, cfg)
        self.freq_mamba = MambaBlock(hid_feature, cfg)
        self.tlinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class SEMamba(nn.Module):
    """SEMamba generator. Multiplies the mask by noisy magnitude and estimates phase."""

    def __init__(self, cfg: SimpleNamespace, n_fft: int) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(cfg.hid_feature)
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(cfg.num_tfmamba)])
        self.mask_decoder = MagDecoder(
            cfg.hid_feature,
            activation=LearnableSigmoid2d(n_fft // 2 + 1, beta=cfg.beta),
        )
        self.phase_decoder = PhaseDecoder(cfg.hid_feature)

    def forward(self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor) -> Spec:
        x = stack_mag_pha(noisy_mag, noisy_pha)
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        mag = self.mask_decoder(x) * noisy_mag
        pha = self.phase_decoder(x)
        return Spec(mag, pha, complex_from_mag_pha(mag, pha))
