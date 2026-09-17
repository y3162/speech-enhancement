import torch
import torch.nn as nn

from src.se.mag_phase.mamba import mamba_block_from_cfg
from src.se.se_mamba_pp.fan import FANFFNGateChannel
from src.se.se_mamba_pp.frequency_glp import FrequencyGLP


class ChannelsFirstLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _down_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            (3, 4),
            stride=(1, 2),
            padding=(1, 1),
        ),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.PReLU(out_ch),
    )


def _up_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch,
            out_ch,
            (3, 4),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 0),
        ),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.PReLU(out_ch),
    )


def _gate(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels * 2, channels, kernel_size=1),
        nn.GELU(),
        ChannelsFirstLayerNorm(channels),
    )


class SEMambaPPBottleneck(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        hid_feature = cfg.hid_feature
        unet_expansion = cfg.unet_expansion
        self.features = [
            hid_feature,
            int(hid_feature * unet_expansion),
            int(hid_feature * unet_expansion ** 2),
        ]
        self.downsamples = nn.ModuleList(
            [
                nn.Identity(),
                _down_conv(self.features[0], self.features[1]),
                _down_conv(self.features[1], self.features[2]),
            ]
        )
        self.time_mambas = nn.ModuleList(
            [mamba_block_from_cfg(self.features[i], cfg) for i in range(3)]
        )
        self.freq_ffns = nn.ModuleList(
            [
                FrequencyGLP(freq_bins, channels, 2)
                for freq_bins, channels in zip((100, 50, 25), self.features)
            ]
        )
        self.freq_layernorm = nn.ModuleList(
            [nn.LayerNorm(self.features[i]) for i in range(3)]
        )
        self.channel_ffns = nn.ModuleList(
            [FANFFNGateChannel(self.features[i], 2) for i in range(3)]
        )
        self.tlinears = nn.ModuleList(
            [
                nn.ConvTranspose1d(self.features[i] * 2, self.features[i], 1, stride=1)
                for i in range(3)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                _up_conv(self.features[2], self.features[1]),
                _up_conv(self.features[1], self.features[0]),
            ]
        )
        self.gates = nn.ModuleList(
            [
                _gate(self.features[1]),
                _gate(self.features[0]),
            ]
        )

    def _time_freq_block(self, x: torch.Tensor, level: int) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinears[level](self.time_mambas[level](x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_layernorm[level](x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        x = self.freq_ffns[level](x) + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_level1 = self.downsamples[0](x)
        x_level2 = self.downsamples[1](x_level1)
        x_level3 = self.downsamples[2](x_level2)

        x_level1 = self._time_freq_block(x_level1, 0)
        x_level1 = self.channel_ffns[0](x_level1) + x_level1
        residual = x_level1

        x_level2 = self._time_freq_block(x_level2, 1)
        x_level2 = self.channel_ffns[1](x_level2) + x_level2

        x_level3 = self._time_freq_block(x_level3, 2)
        x_level3 = self.channel_ffns[2](x_level3) + x_level3

        x_us_2 = self.upsamples[0](x_level3)
        x_ds = self.gates[0](torch.cat([x_level2, x_us_2], dim=1))
        x_us = self.upsamples[1](x_ds)
        x = self.gates[1](torch.cat([x_level1, x_us], dim=1))
        return residual + x
