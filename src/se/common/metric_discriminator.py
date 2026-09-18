import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.se.common.activations import LearnableSigmoid1d


class MetricDiscriminator(nn.Module):
    """Regresses PESQ in [0, 1] from a pair of clean and evaluated magnitudes (MetricGAN)."""

    def __init__(self) -> None:
        super().__init__()
        dim = 16
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(2, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1d(1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.stack((x, y), dim=1))
