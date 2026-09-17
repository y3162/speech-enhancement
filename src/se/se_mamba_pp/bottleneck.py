import torch
import torch.nn as nn
import torch.nn.functional as F

from src.se.common.mamba import MambaBlock


class FANLayer(nn.Module):
    """Fourier Analysis Network layer. Half of the output is cos/sin; the rest is a GELU path."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        p_output_dim = int(output_dim * 0.25)
        g_output_dim = output_dim - p_output_dim * 2
        self.input_linear_p = nn.Linear(input_dim, p_output_dim)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        g = F.gelu(self.input_linear_g(src))
        p = self.input_linear_p(src)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)


class FANFFNGateFreq(nn.Module):
    """Gated FAN-FFN along the last (frequency) axis."""

    def __init__(self, input_dim: int, expansion: int) -> None:
        super().__init__()
        expansion_dim = int(input_dim * expansion)
        self.FAN1 = FANLayer(input_dim, expansion_dim)
        self.FAN2 = FANLayer(expansion_dim, expansion_dim)
        self.Linear = nn.Linear(expansion_dim, input_dim * 2)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = F.gelu(src)
        x = self.FAN1(x)
        x = self.FAN2(x)
        output1, output2 = self.Linear(x).chunk(2, dim=-1)
        return output1 * torch.sigmoid(output2)


class FANFFNGateChannel(nn.Module):
    """Gated FAN-FFN along the channel axis. I/O [B, C, T, F]."""

    def __init__(self, input_dim: int, expansion: int) -> None:
        super().__init__()
        expansion_dim = input_dim * expansion
        self.layernorm = nn.LayerNorm(input_dim)
        self.FAN1 = FANLayer(input_dim, expansion_dim)
        self.FAN2 = FANLayer(expansion_dim, expansion_dim)
        self.Linear = nn.Linear(expansion_dim, input_dim * 2)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = F.gelu(src)
        x = self.layernorm(src.permute(0, 2, 3, 1))
        x = self.FAN1(x)
        x = self.FAN2(x)
        output1, output2 = self.Linear(x).chunk(2, dim=-1)
        return (output1 * torch.sigmoid(output2)).permute(0, 3, 1, 2)


class LocalFrequencyMix(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b * t, c, f)
        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = residual + x
        return x.view(b, t, c, f).permute(0, 2, 1, 3)


class FrequencyGLP(nn.Module):
    """Mix of global (FAN) and local (Conv1d) along frequency. input_dim is fixed to the number of frequency bins."""

    def __init__(self, input_dim: int, channel: int, expansion: int) -> None:
        super().__init__()
        self.global_branch = FANFFNGateFreq(input_dim, expansion)
        self.local_branch = LocalFrequencyMix(channel)
        self.linear = nn.Conv2d(channel * 2, channel, kernel_size=1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = torch.cat([self.global_branch(src), self.local_branch(src)], dim=1)
        return self.linear(output)


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
        nn.Conv2d(in_ch, out_ch, (3, 4), stride=(1, 2), padding=(1, 1)),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.PReLU(out_ch),
    )


def _up_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, (3, 4), stride=(1, 2), padding=(1, 1), output_padding=(0, 0)),
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
    """3-stage U-Net. Each stage: time Mamba, frequency GLP, channel FAN-FFN. Frequency widths (100, 50, 25) assume n_fft=400."""

    def __init__(self, cfg) -> None:
        super().__init__()
        hid_feature = cfg.hid_feature
        unet_expansion = cfg.unet_expansion
        features = [
            hid_feature,
            int(hid_feature * unet_expansion),
            int(hid_feature * unet_expansion ** 2),
        ]
        self.downsamples = nn.ModuleList(
            [_down_conv(features[0], features[1]), _down_conv(features[1], features[2])]
        )
        self.time_mambas = nn.ModuleList([MambaBlock(features[i], cfg) for i in range(3)])
        self.freq_ffns = nn.ModuleList(
            [
                FrequencyGLP(freq_bins, channels, 2)
                for freq_bins, channels in zip((100, 50, 25), features)
            ]
        )
        self.freq_layernorm = nn.ModuleList([nn.LayerNorm(features[i]) for i in range(3)])
        self.channel_ffns = nn.ModuleList([FANFFNGateChannel(features[i], 2) for i in range(3)])
        self.tlinears = nn.ModuleList(
            [nn.ConvTranspose1d(features[i] * 2, features[i], 1, stride=1) for i in range(3)]
        )
        self.upsamples = nn.ModuleList(
            [_up_conv(features[2], features[1]), _up_conv(features[1], features[0])]
        )
        self.gates = nn.ModuleList([_gate(features[1]), _gate(features[0])])

    def _time_freq_block(self, x: torch.Tensor, level: int) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinears[level](self.time_mambas[level](x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_layernorm[level](x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        x = self.freq_ffns[level](x) + x
        return self.channel_ffns[level](x) + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_level1 = x
        x_level2 = self.downsamples[0](x_level1)
        x_level3 = self.downsamples[1](x_level2)

        x_level1 = self._time_freq_block(x_level1, 0)
        x_level2 = self._time_freq_block(x_level2, 1)
        x_level3 = self._time_freq_block(x_level3, 2)

        x_up2 = self.upsamples[0](x_level3)
        x_mid = self.gates[0](torch.cat([x_level2, x_up2], dim=1))
        x_up1 = self.upsamples[1](x_mid)
        x_out = self.gates[1](torch.cat([x_level1, x_up1], dim=1))
        return x_level1 + x_out
