import torch
import torch.nn as nn

from src.se.mag_phase.mamba import mamba_block_from_cfg


class TFMambaBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        hid_feature = cfg.hid_feature
        self.time_mamba = mamba_block_from_cfg(hid_feature, cfg)
        self.freq_mamba = mamba_block_from_cfg(hid_feature, cfg)
        self.tlinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
