import torch
import torch.nn as nn

from src.se.se_mamba_pp.fan import FANFFNGateFreq


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
    def __init__(self, input_dim: int, channel: int, expansion: int) -> None:
        super().__init__()
        self.global_branch = FANFFNGateFreq(input_dim, expansion)
        self.local_branch = LocalFrequencyMix(channel)
        self.linear = nn.Conv2d(channel * 2, channel, kernel_size=1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = torch.cat([self.global_branch(src), self.local_branch(src)], dim=1)
        return self.linear(output)
