import torch
import torch.nn as nn

from src.se.common.activations import LearnableSoftplus
from src.se.common.dense import DenseEncoder, MagDecoder, PhaseDecoder
from src.se.se_mamba_pp.bottleneck import SEMambaPPBottleneck
from src.se.common.stft import Spec, complex_from_mag_pha, stack_mag_pha


class SEMambaPP(nn.Module):
    """SEMamba++ generator. Unlike SEMamba, decoder output is the magnitude (not multiplied by noisy mag)."""

    def __init__(self, cfg, n_fft: int) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(cfg.hid_feature)
        self.TSMamba = nn.ModuleList([SEMambaPPBottleneck(cfg) for _ in range(cfg.num_tfmamba)])
        self.mask_decoder = MagDecoder(
            cfg.hid_feature,
            activation=LearnableSoftplus(n_fft // 2 + 1),
        )
        self.phase_decoder = PhaseDecoder(cfg.hid_feature)

    def forward(self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor) -> Spec:
        x = stack_mag_pha(noisy_mag, noisy_pha)
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        mag = self.mask_decoder(x)
        pha = self.phase_decoder(x)
        return Spec(mag, pha, complex_from_mag_pha(mag, pha))
