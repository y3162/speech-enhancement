import unittest

import torch
import torch.nn as nn

from src.se.common.metric_discriminator import MetricDiscriminator
from src.se.common.stft import Spec
from src.se.common.training import load_config
from tests.helpers import se_config


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


class _ConstantMask(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return self.mask


class MpSenetTest(unittest.TestCase):
    def test_forward_shapes_and_mask_times_noisy_mag(self) -> None:
        from src.se.mp_senet.model import MPNet

        cfg = load_config(se_config("mp_senet"))
        torch.manual_seed(0)
        model = MPNet(cfg.model, cfg.data.stft.n_fft).eval()
        self.assertTrue(hasattr(model, "TSTransformer"))
        self.assertTrue(hasattr(model, "mask_decoder"))
        mag, pha = torch.rand(1, 201, 9), torch.rand(1, 201, 9)
        with torch.no_grad():
            gen = model(mag, pha)
        self.assertIsInstance(gen, Spec)
        self.assertEqual(tuple(gen.mag.shape), (1, 201, 9))
        self.assertEqual(tuple(gen.pha.shape), (1, 201, 9))
        self.assertEqual(tuple(gen.com.shape), (1, 201, 9, 2))

        known = torch.full((1, 201, 9), 0.5)
        model.mask_decoder = _ConstantMask(known)
        with torch.no_grad():
            masked = model(mag, pha)
        self.assertTrue(torch.equal(masked.mag, known * mag))


class MetricDiscriminatorTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        disc = MetricDiscriminator()
        out = disc(torch.rand(2, 201, 17), torch.rand(2, 201, 17))
        self.assertEqual(tuple(out.shape), (2, 1))


@unittest.skipUnless(_has_module("nnAudio"), "nnAudio is not installed")
class SeMambaPPDiscriminatorTest(unittest.TestCase):
    def test_three_cqt_and_three_mrd_branches(self) -> None:
        from src.se.se_mamba_pp.discriminator import DiscriminatorOutputs, SEMambaPPDiscriminator

        disc = SEMambaPPDiscriminator().eval()
        with torch.no_grad():
            outputs = disc(torch.rand(1, 1, 8000), torch.rand(1, 1, 8000))
        self.assertIsInstance(outputs, DiscriminatorOutputs)
        self.assertEqual(len(outputs.cqt_real), 3)
        self.assertEqual(len(outputs.mrd_real), 3)
        self.assertEqual(len(outputs.cqt_fmap_real), 3)
        self.assertEqual(len(outputs.mrd_fmap_fake), 3)
