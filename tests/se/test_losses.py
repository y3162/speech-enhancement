import unittest
from types import SimpleNamespace

import torch

from src.se.common.phase_loss import phase_loss_difference, phase_loss_gradient_matrix
from src.se.common.stft import Spec
from src.se.common.training import load_config
from tests.helpers import se_config

N_FREQ = 201
N_FRAMES = 17
METRICGAN_KEYS = {"magnitude", "phase", "complex", "consistency", "time", "metric", "total"}
SEMAMBA_PP_KEYS = {"magnitude", "phase", "complex", "consistency", "adv_g", "fm_g", "mel", "tdt", "total"}


def _spec() -> Spec:
    return Spec(torch.rand(2, N_FREQ, N_FRAMES), torch.rand(2, N_FREQ, N_FRAMES), torch.rand(2, N_FREQ, N_FRAMES, 2))


def _weighted_total(losses: dict[str, torch.Tensor], weights: SimpleNamespace) -> torch.Tensor:
    total = torch.zeros((), dtype=losses["total"].dtype)
    for name, value in losses.items():
        if name == "total":
            continue
        total = total + getattr(weights, name) * value
    return total


class PhaseLossTest(unittest.TestCase):
    def test_phase_losses_are_finite_and_zero_when_identical(self) -> None:
        torch.manual_seed(0)
        phase_r = torch.rand(2, N_FREQ, N_FRAMES)
        phase_g = torch.rand(2, N_FREQ, N_FRAMES)
        for losses in (
            phase_loss_difference(phase_r, phase_g),
            phase_loss_gradient_matrix(phase_r, phase_g, 400),
        ):
            self.assertEqual(len(losses), 3)
            for value in losses:
                self.assertTrue(torch.isfinite(value))
        identical = phase_loss_difference(phase_r, phase_r)
        for value in identical:
            self.assertTrue(torch.allclose(value, torch.zeros_like(value), atol=1e-6))


class MetricGanGeneratorLossTest(unittest.TestCase):
    def _run(self, generator_loss, weights):
        torch.manual_seed(0)
        clean, gen, gen_hat = _spec(), _spec(), _spec()
        clean_audio, enhanced_audio = torch.rand(2, 1600), torch.rand(2, 1600)
        metric_g = torch.rand(2, 1)
        return generator_loss(clean, clean_audio, gen, enhanced_audio, gen_hat, metric_g, weights)

    def test_mp_senet_keys_and_weighted_total(self) -> None:
        from src.se.mp_senet.loss import generator_loss

        weights = load_config(se_config("mp_senet")).train.loss
        losses = self._run(generator_loss, weights)
        self.assertEqual(set(losses), METRICGAN_KEYS)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.allclose(losses["total"], _weighted_total(losses, weights)))

    def test_se_mamba_keys_and_weighted_total(self) -> None:
        from src.se.se_mamba.loss import generator_loss

        weights = load_config(se_config("se_mamba")).train.loss

        def se_mamba_loss(*args):
            return generator_loss(*args, 400)

        losses = self._run(se_mamba_loss, weights)
        self.assertEqual(set(losses), METRICGAN_KEYS)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.allclose(losses["total"], _weighted_total(losses, weights)))

    def test_missing_weight_is_an_error(self) -> None:
        from src.se.mp_senet.loss import generator_loss

        weights = SimpleNamespace(magnitude=0.9, phase=0.3, complex=0.1, consistency=0.1, time=0.2)
        with self.assertRaises(AttributeError):
            self._run(generator_loss, weights)


class SeMambaPPLossTest(unittest.TestCase):
    def test_generator_has_no_time_term_and_weighted_total(self) -> None:
        from src.se.se_mamba_pp.discriminator import DiscriminatorOutputs
        from src.se.se_mamba_pp.loss import discriminator_loss, generator_loss

        torch.manual_seed(0)
        scores = [torch.rand(2, 5) for _ in range(3)]
        fmaps = [[torch.rand(2, 4, 3) for _ in range(2)] for _ in range(3)]
        d = DiscriminatorOutputs(scores, scores, fmaps, fmaps, scores, scores, fmaps, fmaps)
        weights = load_config(se_config("se_mamba_pp")).train.loss
        losses = generator_loss(_spec(), _spec(), _spec(), d, torch.tensor(0.5), torch.tensor(1.2), weights, 400)
        self.assertEqual(set(losses), SEMAMBA_PP_KEYS)
        self.assertNotIn("time", losses)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.allclose(losses["total"], _weighted_total(losses, weights)))
        d_losses = discriminator_loss(d)
        self.assertEqual(set(d_losses), {"cqt", "mrd", "total"})
        self.assertTrue(torch.allclose(d_losses["total"], d_losses["cqt"] + d_losses["mrd"]))
