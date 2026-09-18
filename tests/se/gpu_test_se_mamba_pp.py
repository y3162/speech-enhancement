import unittest

import torch
import torch.nn as nn

from src.se.common.stft import Spec
from src.se.common.training import load_config
from src.se.se_mamba_pp.model import SEMambaPP
from tests.helpers import se_config


class _ConstantMask(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return self.mask


class SeMambaPPGpuTest(unittest.TestCase):
    def test_forward_and_mask_is_not_multiplied_by_noisy_mag(self) -> None:
        device = torch.device("cuda")
        cfg = load_config(se_config("se_mamba_pp"))
        model = SEMambaPP(cfg.model, cfg.data.stft.n_fft).to(device).eval()
        mag = torch.rand(1, 201, 9, device=device)
        pha = torch.rand(1, 201, 9, device=device)
        with torch.no_grad():
            generated = model(mag, pha)
        self.assertIsInstance(generated, Spec)
        self.assertEqual(tuple(generated.mag.shape), (1, 201, 9))
        self.assertEqual(tuple(generated.pha.shape), (1, 201, 9))
        self.assertEqual(tuple(generated.com.shape), (1, 201, 9, 2))

        known = torch.full((1, 201, 9), 0.5, device=device)
        model.mask_decoder = _ConstantMask(known).to(device)
        with torch.no_grad():
            masked = model(mag, pha)
        self.assertTrue(torch.equal(masked.mag, known))
        self.assertFalse(torch.equal(masked.mag, known * mag))
