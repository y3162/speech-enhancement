import unittest

import torch

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
from src.se.common.stft import mag_pha_istft, mag_pha_stft
from src.se.common.training import load_config
from src.se.se_mamba_pp.asr_guidance import ASR_DIM, ENCODER_LAYER
from src.se.se_mamba_pp.model import SEMambaPP
from src.se.se_mamba_pp.train import _guidance_features, _wav_lengths
from tests.helpers import se_config

DEVICE: torch.device
ASR: ParakeetTDT06BV2


def setUpModule() -> None:
    global DEVICE, ASR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SEMamba++ ASR guidance tests")
    DEVICE = torch.device("cuda")
    ASR = ParakeetTDT06BV2().to(DEVICE)
    ASR.eval()
    for parameter in ASR.parameters():
        parameter.requires_grad_(False)


class _AsrGuidanceCase(unittest.TestCase):
    def tearDown(self) -> None:
        ASR.zero_grad(set_to_none=True)


class AsrGuidanceTdtGradientTest(_AsrGuidanceCase):
    def test_tdt_backward_reaches_semamba_not_asr(self) -> None:
        cfg = load_config(se_config("se_mamba_pp"))
        generator = SEMambaPP(cfg.model, cfg.data.stft.n_fft).to(DEVICE)
        generator.train()
        torch.manual_seed(0)
        samples = 16000
        clean = (0.05 * torch.randn(1, samples, device=DEVICE)).to(dtype=torch.float32)
        noisy = clean + 0.01 * torch.randn_like(clean)
        lengths = _wav_lengths(clean)
        with torch.no_grad():
            texts = [recognition.text for recognition in ASR.recognize(clean, lengths)]
            asr_hidden, asr_lengths = _guidance_features(ASR, noisy, lengths)
        self.assertEqual(ASR.samples_per_encoder_frame, 1280)
        self.assertTrue(all(isinstance(text, str) and text != "" for text in texts), texts)
        self.assertEqual(asr_hidden.size(1), ASR_DIM)
        self.assertFalse(asr_hidden.requires_grad)

        noisy_spec = mag_pha_stft(noisy, cfg.data.stft)
        gen = generator(noisy_spec.mag, noisy_spec.pha, asr_hidden, asr_lengths)
        enhanced = mag_pha_istft(gen.mag, gen.pha, cfg.data.stft)
        enhanced.retain_grad()
        tdt = ASR.loss(enhanced, lengths, texts).mean()
        self.assertTrue(tdt.requires_grad)
        self.assertTrue(torch.isfinite(tdt).all())

        generator.zero_grad(set_to_none=True)
        ASR.zero_grad(set_to_none=True)
        tdt.backward()

        self.assertIsNotNone(enhanced.grad)
        self.assertTrue(torch.isfinite(enhanced.grad).all())
        self.assertGreater(float(enhanced.grad.abs().sum()), 0.0)
        self.assertFalse(asr_hidden.requires_grad)
        self.assertIsNone(asr_hidden.grad)

        self.assertIsNotNone(generator.asr_proj)
        self.assertIsNotNone(generator.asr_fuse)
        proj_grad = generator.asr_proj.weight.grad
        fuse_grad = generator.asr_fuse.weight.grad
        self.assertIsNotNone(proj_grad)
        self.assertIsNotNone(fuse_grad)
        self.assertTrue(torch.isfinite(proj_grad).all())
        self.assertTrue(torch.isfinite(fuse_grad).all())
        self.assertGreater(float(proj_grad.abs().sum()), 0.0)
        self.assertGreater(float(fuse_grad.abs().sum()), 0.0)

        se_grads = [
            parameter.grad
            for parameter in generator.parameters()
            if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        ]
        self.assertGreater(len(se_grads), 0)
        self.assertTrue(all(parameter.grad is None for parameter in ASR.parameters()))
        self.assertEqual(ENCODER_LAYER, 15)
