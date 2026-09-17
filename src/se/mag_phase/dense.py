import torch
import torch.nn as nn

from src.se.mag_phase.activations import LearnableSigmoid2d, LearnableSoftplus


def padding_2d(kernel_size: tuple[int, int], dilation: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class DenseBlock(nn.Module):
    def __init__(self, hid_feature: int, kernel_size: tuple[int, int] = (3, 3), depth: int = 4) -> None:
        super().__init__()
        self.dense_block = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.dense_block.append(
                nn.Sequential(
                    nn.Conv2d(
                        hid_feature * (i + 1),
                        hid_feature,
                        kernel_size,
                        dilation=(dilation, 1),
                        padding=padding_2d(kernel_size, (dilation, 1)),
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
    def __init__(self, input_channel: int, hid_feature: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, hid_feature, (1, 1)),
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
    def __init__(self, hid_feature: int, output_channel: int, n_fft: int, activation: str, beta: float = 2.0) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(hid_feature, output_channel, (1, 1)),
            nn.InstanceNorm2d(output_channel, affine=True),
            nn.PReLU(output_channel),
            nn.Conv2d(output_channel, output_channel, (1, 1)),
        )
        if activation == "learnable_sigmoid":
            self.lsigmoid = LearnableSigmoid2d(n_fft // 2 + 1, beta=beta)
        elif activation == "learnable_softplus":
            self.softplus = LearnableSoftplus(n_fft // 2 + 1)
        else:
            raise ValueError(
                "activation must be 'learnable_sigmoid' or 'learnable_softplus', "
                f"got {activation!r}"
            )
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        if self._activation == "learnable_sigmoid":
            x = self.lsigmoid(x)
        else:
            x = self.softplus(x)
        return x.permute(0, 2, 1).unsqueeze(1)


class PhaseDecoder(nn.Module):
    def __init__(self, hid_feature: int, output_channel: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )
        self.phase_conv_r = nn.Conv2d(hid_feature, output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid_feature, output_channel, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        return torch.atan2(x_i, x_r)
