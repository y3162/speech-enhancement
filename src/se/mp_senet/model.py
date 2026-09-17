from types import SimpleNamespace

import torch
import torch.nn as nn

from src.se.mag_phase.activations import LearnableSigmoid2d
from src.se.mag_phase.layout import complex_from_mag_pha, stack_mag_pha
from src.se.mag_phase.network import MagPhaseNetwork
from src.se.mp_senet.transformer import TransformerBlock


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int = 1) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
        )
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad1(x)
        x = self.conv(x)
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, self.r, n_channels // self.r, height, width)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.contiguous().view(batch_size, n_channels // self.r, height, -1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size: tuple[int, int] = (2, 3), depth: int = 4) -> None:
        super().__init__()
        self.dense_block = nn.ModuleList(
            [self._make_dense_layer(h, kernel_size, i) for i in range(depth)]
        )

    @staticmethod
    def _make_dense_layer(h, kernel_size, layer_idx: int) -> nn.Sequential:
        dilation = 2 ** layer_idx
        pad_length = dilation
        return nn.Sequential(
            nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            nn.Conv2d(
                h.dense_channel * (layer_idx + 1),
                h.dense_channel,
                kernel_size,
                dilation=(dilation, 1),
            ),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for dense_layer in self.dense_block:
            x = dense_layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.dense_block = DenseBlock(h, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(
                h.dense_channel,
                h.dense_channel,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, n_fft: int, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft // 2 + 1, beta=h.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, kernel_size=(1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, kernel_size=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=h.n_heads)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=h.n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(MagPhaseNetwork):
    def __init__(self, h, n_fft: int) -> None:
        super().__init__()
        num_tsblocks = h.num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.TSTransformer = nn.ModuleList(
            [TSTransformerBlock(h) for _ in range(num_tsblocks)]
        )
        self.mask_decoder = MaskDecoder(h, n_fft, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, features: SimpleNamespace) -> SimpleNamespace:
        x = stack_mag_pha(features.mag, features.pha)
        x = self.dense_encoder(x)
        for ts_block in self.TSTransformer:
            x = ts_block(x)
        denoised_amp = features.mag * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = complex_from_mag_pha(denoised_amp, denoised_pha)
        return SimpleNamespace(mag=denoised_amp, pha=denoised_pha, com=denoised_com)
