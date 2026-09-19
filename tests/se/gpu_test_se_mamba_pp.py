import unittest

import torch
import torch.nn as nn

from src.se.common.stft import Spec
from src.se.common.training import load_config
from src.se.se_mamba_pp.asr_guidance import ASR_DIM
from src.se.se_mamba_pp.model import SEMambaPP
from tests.helpers import se_config


class _ConstantMask(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return self.mask


def _cfg(guidance_dim: int):
    cfg = load_config(se_config("se_mamba_pp"))
    cfg.model.asr_guidance_dim = guidance_dim
    return cfg


class SeMambaPPGpuTest(unittest.TestCase):
    def test_forward_and_mask_is_not_multiplied_by_noisy_mag(self) -> None:
        device = torch.device("cuda")
        cfg = _cfg(0)
        model = SEMambaPP(cfg.model, cfg.data.stft.n_fft).to(device).eval()
        self.assertIsNone(model.asr_proj)
        self.assertIsNone(model.asr_fuse)
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

    def test_guidance_forward_shapes_and_hidden_is_required(self) -> None:
        device = torch.device("cuda")
        cfg = _cfg(128)
        model = SEMambaPP(cfg.model, cfg.data.stft.n_fft).to(device).eval()
        self.assertIsNotNone(model.asr_proj)
        self.assertIsNotNone(model.asr_fuse)
        mag = torch.rand(1, 201, 9, device=device)
        pha = torch.rand(1, 201, 9, device=device)
        asr_hidden = torch.randn(1, ASR_DIM, 3, device=device)
        asr_lengths = torch.tensor([3], device=device, dtype=torch.int64)
        with torch.no_grad():
            generated = model(mag, pha, asr_hidden, asr_lengths)
        self.assertEqual(tuple(generated.mag.shape), (1, 201, 9))
        self.assertEqual(tuple(generated.pha.shape), (1, 201, 9))
        with self.assertRaises(ValueError):
            model(mag, pha)
