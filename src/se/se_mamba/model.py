from types import SimpleNamespace

import torch.nn as nn

from src.se.mag_phase.dense import DenseEncoder, MagDecoder, PhaseDecoder
from src.se.mag_phase.layout import complex_from_mag_pha, stack_mag_pha
from src.se.mag_phase.network import MagPhaseNetwork
from src.se.se_mamba.tf_mamba import TFMambaBlock


class SEMamba(MagPhaseNetwork):
    def __init__(self, cfg, n_fft: int) -> None:
        super().__init__()
        num_blocks = cfg.num_tfmamba
        self.dense_encoder = DenseEncoder(cfg.input_channel, cfg.hid_feature)
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(num_blocks)])
        self.mask_decoder = MagDecoder(
            cfg.hid_feature,
            cfg.output_channel,
            n_fft,
            activation="learnable_sigmoid",
            beta=cfg.beta,
        )
        self.phase_decoder = PhaseDecoder(cfg.hid_feature, cfg.output_channel)

    def forward(self, features: SimpleNamespace) -> SimpleNamespace:
        noisy_mag = features.mag.transpose(1, 2).unsqueeze(1)
        x = stack_mag_pha(features.mag, features.pha)
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        denoised_mag = (self.mask_decoder(x) * noisy_mag).squeeze(1).transpose(1, 2)
        denoised_pha = self.phase_decoder(x).squeeze(1).transpose(1, 2)
        denoised_com = complex_from_mag_pha(denoised_mag, denoised_pha)
        return SimpleNamespace(mag=denoised_mag, pha=denoised_pha, com=denoised_com)
