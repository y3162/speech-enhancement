import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSoftplus(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.log(torch.ones(in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.exp(self.beta.clamp(max=40.0)).view(1, -1, 1)
        return (1.0 / beta + 1e-6) * F.softplus(beta * x)
