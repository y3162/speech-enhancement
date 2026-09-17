import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, hid_feature: int, kernel_size: tuple[int, int] = (3, 3), depth: int = 4) -> None:
        super().__init__()
        self.dense_block = nn.ModuleList()
        for i in range(depth):
            dilation = 2**i
            self.dense_block.append(
                nn.Sequential(
                    nn.Conv2d(
                        hid_feature * (i + 1),
                        hid_feature,
                        kernel_size,
                        dilation=(dilation, 1),
                        padding=(
                            int((kernel_size[0] * dilation - dilation) / 2),
                            int((kernel_size[1] - 1) / 2),
                        ),
                    ),
                    nn.InstanceNorm2d(hid_feature, affine=True),
                    nn.PReLU(hid_feature),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for dense_layer in self.dense_block:
            x = dense_layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, hid_feature: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(2, hid_feature, (1, 1)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MagDecoder(nn.Module):
    """[B, C, T, F] -> mask [B, F, T]. activation has a learnable parameter per frequency bin."""

    def __init__(self, hid_feature: int, activation: nn.Module) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(hid_feature, 1, (1, 1)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1),
            nn.Conv2d(1, 1, (1, 1)),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.activation(x)


class PhaseDecoder(nn.Module):
    def __init__(self, hid_feature: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )
        self.phase_conv_r = nn.Conv2d(hid_feature, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid_feature, 1, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        return torch.atan2(x_i, x_r).squeeze(1).transpose(1, 2)
