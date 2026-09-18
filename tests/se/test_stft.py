import unittest
from types import SimpleNamespace

import torch

from src.se.common.stft import Spec, complex_from_mag_pha, mag_pha_istft, mag_pha_stft, stack_mag_pha

# src/se/*/configs/default.json: n_fft=400, hop=100, T=1600, center=True → F=201, frames=17
STFT = SimpleNamespace(n_fft=400, hop_size=100, win_size=400, compress_factor=0.3)
N_FREQ = STFT.n_fft // 2 + 1
N_SAMPLES = 1600
N_FRAMES = N_SAMPLES // STFT.hop_size + 1


def _waveform() -> torch.Tensor:
    wave = torch.linspace(-0.5, 0.5, N_SAMPLES).unsqueeze(0).repeat(2, 1)
    # Leading silence makes eps=1e-10 change the magnitude (not a no-op).
    wave[:, :300] = 0.0
    return wave


class MagPhaStftTest(unittest.TestCase):
    def test_waveform_must_be_2d(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            mag_pha_stft(torch.zeros(N_SAMPLES), STFT)
        self.assertIn("waveform must be [B, T]", str(ctx.exception))

    def test_returns_spec_with_expected_shapes(self) -> None:
        spec = mag_pha_stft(_waveform(), STFT)
        self.assertIsInstance(spec, Spec)
        self.assertEqual(tuple(spec.mag.shape), (2, N_FREQ, N_FRAMES))
        self.assertEqual(tuple(spec.pha.shape), (2, N_FREQ, N_FRAMES))
        self.assertEqual(tuple(spec.com.shape), (2, N_FREQ, N_FRAMES, 2))
        self.assertTrue(torch.equal(spec.com, complex_from_mag_pha(spec.mag, spec.pha)))

    def test_eps_variant_is_finite_and_differs_on_silence(self) -> None:
        wave = _waveform()
        plain = mag_pha_stft(wave, STFT)
        safe = mag_pha_stft(wave, STFT, eps=1e-10)
        for value in (safe.mag, safe.pha, safe.com):
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.allclose(plain.mag, safe.mag, atol=5e-2, rtol=5e-2))
        self.assertFalse(torch.equal(plain.mag, safe.mag))

    def test_eps_variant_has_finite_gradient_on_zero_input(self) -> None:
        wave = torch.zeros(1, N_SAMPLES, requires_grad=True)
        spec = mag_pha_stft(wave, STFT, eps=1e-10)
        (spec.mag.sum() + spec.pha.sum()).backward()
        self.assertTrue(torch.isfinite(wave.grad).all())

    def test_istft_roundtrip_and_length(self) -> None:
        wave = _waveform()
        spec = mag_pha_stft(wave, STFT)
        audio = mag_pha_istft(spec.mag, spec.pha, STFT)
        self.assertEqual(tuple(audio.shape), (2, N_SAMPLES))
        self.assertTrue(torch.allclose(audio, wave, atol=1e-4))
        self.assertEqual(tuple(mag_pha_istft(spec.mag, spec.pha, STFT, length=1500).shape), (2, 1500))
        self.assertEqual(tuple(mag_pha_istft(spec.mag, spec.pha, STFT, length=1700).shape), (2, 1700))


class LayoutTest(unittest.TestCase):
    def test_stack_and_complex_shapes(self) -> None:
        torch.manual_seed(0)
        mag = torch.rand(2, N_FREQ, N_FRAMES)
        pha = torch.rand(2, N_FREQ, N_FRAMES) * 2 - 1
        self.assertEqual(tuple(stack_mag_pha(mag, pha).shape), (2, 2, N_FRAMES, N_FREQ))
        com = complex_from_mag_pha(mag, pha)
        self.assertEqual(tuple(com.shape), (2, N_FREQ, N_FRAMES, 2))
        self.assertTrue(torch.allclose(com[..., 0], mag * torch.cos(pha)))
        self.assertTrue(torch.allclose(com[..., 1], mag * torch.sin(pha)))
