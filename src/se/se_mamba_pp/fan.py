import torch
import torch.nn as nn
import torch.nn.functional as F


class FANLayer(nn.Module):
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
        output = self.Linear(x)
        output1, output2 = output.chunk(2, dim=-1)
        return output1 * torch.sigmoid(output2)


class FANFFNGateChannel(nn.Module):
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
        output = self.Linear(x)
        output1, output2 = output.chunk(2, dim=-1)
        output = output1 * torch.sigmoid(output2)
        return output.permute(0, 3, 1, 2)
