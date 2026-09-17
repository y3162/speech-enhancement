import torch
import torch.nn as nn
import torch.nn.functional as F

from src.se.common.activations import LearnableSigmoid2d
from src.se.common.stft import Spec, complex_from_mag_pha, stack_mag_pha


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution that upsamples the frequency axis by r."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int = 1) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
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
    """MP-SENet DenseBlock. Kernel (2, 3) and causal time padding differ from the SEMamba version."""

    def __init__(self, dense_channel: int, kernel_size: tuple[int, int] = (2, 3), depth: int = 4) -> None:
        super().__init__()
        self.dense_block = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.dense_block.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, dilation, 0), value=0.0),
                    nn.Conv2d(
                        dense_channel * (i + 1),
                        dense_channel,
                        kernel_size,
                        dilation=(dilation, 1),
                    ),
                    nn.InstanceNorm2d(dense_channel, affine=True),
                    nn.PReLU(dense_channel),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for dense_layer in self.dense_block:
            x = dense_layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, dense_channel: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(2, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, dense_channel: int, n_fft: int, beta: float) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
            nn.Conv2d(dense_channel, 1, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft // 2 + 1, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.lsigmoid(x)


class PhaseDecoder(nn.Module):
    def __init__(self, dense_channel: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, 1, kernel_size=(1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, 1, kernel_size=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x = torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return x.permute(0, 3, 2, 1).squeeze(-1)


class FFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_size = d_model * 2
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt)
        x = x + xt
        xt = self.norm2(x)
        x = x + self.ffn(xt)
        return self.norm3(x)


class TSTransformerBlock(nn.Module):
    """Two-stage Transformer: time then frequency. I/O [B, C, T, F]."""

    def __init__(self, dense_channel: int, n_heads: int) -> None:
        super().__init__()
        self.time_transformer = TransformerBlock(d_model=dense_channel, n_heads=n_heads)
        self.freq_transformer = TransformerBlock(d_model=dense_channel, n_heads=n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class MPNet(nn.Module):
    """MP-SENet generator. Takes compressed mag/phase [B, F, T] and returns masked mag and estimated phase."""

    def __init__(self, cfg, n_fft: int) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(cfg.dense_channel)
        self.TSTransformer = nn.ModuleList(
            [TSTransformerBlock(cfg.dense_channel, cfg.n_heads) for _ in range(cfg.num_tsblocks)]
        )
        self.mask_decoder = MaskDecoder(cfg.dense_channel, n_fft, cfg.beta)
        self.phase_decoder = PhaseDecoder(cfg.dense_channel)

    def forward(self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor) -> Spec:
        x = stack_mag_pha(noisy_mag, noisy_pha)
        x = self.dense_encoder(x)
        for block in self.TSTransformer:
            x = block(x)
        mag = noisy_mag * self.mask_decoder(x)
        pha = self.phase_decoder(x)
        return Spec(mag, pha, complex_from_mag_pha(mag, pha))
