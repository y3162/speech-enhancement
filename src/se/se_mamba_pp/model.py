from types import SimpleNamespace

import torch
import torch.nn as nn

from src.se.common.activations import LearnableSoftplus
from src.se.common.dense import DenseEncoder, MagDecoder, PhaseDecoder
from src.se.common.stft import Spec, complex_from_mag_pha, stack_mag_pha
from src.se.se_mamba_pp.asr_guidance import ASR_DIM, fuse_asr_features, project_asr_features
from src.se.se_mamba_pp.bottleneck import SEMambaPPBottleneck


def require_asr_guidance_dim(cfg: SimpleNamespace) -> int:
    """0 disables ASR feature input. A positive int is the projection width."""
    value = cfg.asr_guidance_dim
    if type(value) is not int or value < 0:
        raise ValueError(f"asr_guidance_dim must be a non-negative int, got {value!r}")
    return value


class SEMambaPP(nn.Module):
    """SEMamba++ generator. Unlike SEMamba, decoder output is the magnitude (not multiplied by noisy mag)."""

    def __init__(self, cfg: SimpleNamespace, n_fft: int) -> None:
        super().__init__()
        guidance_dim = require_asr_guidance_dim(cfg)
        self.dense_encoder = DenseEncoder(cfg.hid_feature)
        self.TSMamba = nn.ModuleList([SEMambaPPBottleneck(cfg) for _ in range(cfg.num_tfmamba)])
        self.mask_decoder = MagDecoder(
            cfg.hid_feature,
            activation=LearnableSoftplus(n_fft // 2 + 1),
        )
        self.phase_decoder = PhaseDecoder(cfg.hid_feature)
        if guidance_dim:
            self.asr_proj: nn.Linear | None = nn.Linear(ASR_DIM, guidance_dim)
            self.asr_fuse: nn.Conv2d | None = nn.Conv2d(cfg.hid_feature + guidance_dim, cfg.hid_feature, kernel_size=1)
        else:
            self.asr_proj = None
            self.asr_fuse = None

    def forward(
        self,
        noisy_mag: torch.Tensor,
        noisy_pha: torch.Tensor,
        asr_hidden: torch.Tensor | None = None,
        asr_lengths: torch.Tensor | None = None,
    ) -> Spec:
        x = stack_mag_pha(noisy_mag, noisy_pha)
        x = self.dense_encoder(x)
        asr_proj = self.asr_proj
        asr_fuse = self.asr_fuse
        if asr_proj is not None and asr_fuse is not None:
            if asr_hidden is None:
                raise ValueError("asr_hidden is required when ASR guidance is enabled")
            if asr_hidden.size(1) != ASR_DIM:
                raise ValueError(f"asr_hidden channel dim must be {ASR_DIM}, got {tuple(asr_hidden.shape)}")
            aligned = project_asr_features(asr_proj, asr_hidden, x.size(2), asr_lengths)
            x = fuse_asr_features(asr_fuse, x, aligned)
        elif asr_hidden is not None:
            raise ValueError("asr_hidden was passed but ASR guidance is disabled")
        for block in self.TSMamba:
            x = block(x)
        mag = self.mask_decoder(x)
        pha = self.phase_decoder(x)
        return Spec(mag, pha, complex_from_mag_pha(mag, pha))
