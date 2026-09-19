import os
import unittest

import torch
import torchaudio

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2

DEVICE: torch.device
MODEL: ParakeetTDT06BV2


def setUpModule() -> None:
    global DEVICE, MODEL
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Parakeet TDT tests")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    DEVICE = torch.device("cuda")
    MODEL = ParakeetTDT06BV2().to(DEVICE)


def _waveforms(batch: int = 2, samples: int = 16000, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    wav = (0.05 * torch.randn(batch, samples, device=DEVICE)).to(dtype=torch.float32)
    if requires_grad:
        wav = wav.requires_grad_(True)
    lengths = torch.full((batch,), samples, device=DEVICE, dtype=torch.int64)
    return wav, lengths


def _has_nonzero_param_grad(model: ParakeetTDT06BV2) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0:
            return True
    return False


def _speech_env_complete() -> bool:
    return all(
        os.environ.get(name)
        for name in ("PARAKEET_TEST_AUDIO", "PARAKEET_TEST_TRANSCRIPT", "PARAKEET_TEST_UTTERANCE_ID")
    )


class _ParakeetCase(unittest.TestCase):
    def tearDown(self) -> None:
        MODEL.eval()
        MODEL.zero_grad(set_to_none=True)


class ParakeetLossTest(_ParakeetCase):
    def test_loss_shape_and_empty_text_is_zero(self) -> None:
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["hello", ""])
        self.assertEqual(tuple(nll.shape), (2,))
        self.assertTrue(nll.requires_grad)
        self.assertEqual(float(nll[1].detach()), 0.0)
        mixed_valid = nll[0].detach()
        solo_wav = wav[0:1].detach().clone().requires_grad_(True)
        solo = MODEL.loss(solo_wav, lengths[:1], ["hello"])
        self.assertTrue(torch.allclose(mixed_valid, solo[0].detach(), atol=1e-4, rtol=1e-4))

    def test_zero_length_nll_is_zero(self) -> None:
        wav, lengths = _waveforms(requires_grad=True)
        lengths = lengths.clone()
        lengths[1] = 0
        nll = MODEL.loss(wav, lengths, ["hello", "hello"])
        self.assertEqual(float(nll[1].detach()), 0.0)

    def test_all_empty_text_is_zeros(self) -> None:
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["", ""])
        self.assertTrue(torch.equal(nll.detach(), torch.zeros(2, device=DEVICE)))


class ParakeetGradientTest(_ParakeetCase):
    def test_eval_populates_waveform_grad_not_parameter_grad(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["hello", "world"])
        nll.sum().backward()
        self.assertIsNotNone(wav.grad)
        self.assertTrue(torch.isfinite(wav.grad).all())
        self.assertFalse(any(parameter.grad is not None for parameter in MODEL.parameters()))

    def test_train_populates_parameter_grad(self) -> None:
        MODEL.train()
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["hello", "world"])
        nll.sum().backward()
        self.assertTrue(_has_nonzero_param_grad(MODEL))

    def test_recognize_does_not_block_later_train_grads(self) -> None:
        MODEL.train()
        wav, lengths = _waveforms()
        MODEL.recognize(wav, lengths)
        MODEL.zero_grad(set_to_none=True)
        wav_grad, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav_grad, lengths, ["hello", "world"])
        nll.sum().backward()
        self.assertTrue(_has_nonzero_param_grad(MODEL))

    def test_empty_and_zero_length_items_have_zero_waveform_grad(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["hello", ""])
        nll.sum().backward()
        self.assertIsNotNone(wav.grad)
        self.assertEqual(float(wav.grad[1].abs().sum()), 0.0)

        MODEL.zero_grad(set_to_none=True)
        wav, lengths = _waveforms(requires_grad=True)
        lengths = lengths.clone()
        lengths[1] = 0
        nll = MODEL.loss(wav, lengths, ["hello", "hello"])
        nll.sum().backward()
        self.assertEqual(float(wav.grad[1].abs().sum()), 0.0)

        MODEL.zero_grad(set_to_none=True)
        wav, lengths = _waveforms(requires_grad=True)
        nll = MODEL.loss(wav, lengths, ["", ""])
        nll.sum().backward()
        self.assertEqual(float(wav.grad.abs().sum()), 0.0)


class ParakeetEncodeTest(_ParakeetCase):
    def test_encode_shapes_layers_and_out_of_range(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms(requires_grad=True)
        encoded = MODEL.encode(wav, lengths, layers=(0, -1))
        self.assertEqual(encoded.encoded.ndim, 3)
        self.assertEqual(encoded.encoded.shape[0], 2)
        self.assertGreater(encoded.encoded.shape[1], 0)
        self.assertGreater(encoded.encoded.shape[2], 0)
        self.assertEqual(tuple(encoded.encoded_length.shape), (2,))
        self.assertEqual(encoded.encoded_length.dtype, torch.int64)
        self.assertTrue(encoded.encoded.requires_grad)
        self.assertIn(0, encoded.layer_outputs)
        self.assertEqual(len(encoded.layer_outputs), 2)
        last = max(encoded.layer_outputs)
        self.assertEqual(tuple(encoded.layer_outputs[last].shape), tuple(encoded.encoded.shape))
        self.assertEqual(tuple(encoded.layer_outputs[0].shape), tuple(encoded.encoded.shape))
        self.assertEqual(encoded.encoded.size(-1), int(encoded.encoded_length.max()))

        recognized = MODEL.recognize_encoded(encoded.encoded, encoded.encoded_length)
        self.assertEqual(recognized[0].encoded_length, int(encoded.encoded_length[0]))

        encoded.encoded.sum().backward()
        self.assertIsNotNone(wav.grad)
        self.assertTrue(torch.isfinite(wav.grad).all())
        self.assertFalse(any(parameter.grad is not None for parameter in MODEL.parameters()))

        MODEL.zero_grad(set_to_none=True)
        plain = MODEL.encode(wav.detach(), lengths)
        self.assertEqual(plain.layer_outputs, {})

        with self.assertRaises(ValueError):
            MODEL.encode(wav.detach(), lengths, layers=(10**6,))

    def test_clip_encoded_time_trims_extra_frame(self) -> None:
        from src.asr.parakeet_tdt_0_6b_v2 import clip_encoded_time

        clipped, clipped_length = clip_encoded_time(torch.zeros(1, 4, 164), torch.tensor([163]))
        self.assertEqual(clipped.size(-1), 163)
        self.assertEqual(int(clipped_length[0]), 163)
        clipped, clipped_length = clip_encoded_time(torch.zeros(1, 4, 163), torch.tensor([164]))
        self.assertEqual(clipped.size(-1), 163)
        self.assertEqual(int(clipped_length[0]), 163)

    def test_encode_time_matches_reported_length(self) -> None:
        MODEL.eval()
        for samples in (16000, 48000, 208640):
            wav, lengths = _waveforms(batch=1, samples=samples)
            encoded = MODEL.encode(wav, lengths, layers=(-1,))
            self.assertEqual(encoded.encoded.size(-1), int(encoded.encoded_length.max()))
            last = max(encoded.layer_outputs)
            self.assertEqual(encoded.layer_outputs[last].size(-1), encoded.encoded.size(-1))
            nll = MODEL.loss(wav, lengths, ["hello"])
            self.assertTrue(torch.isfinite(nll).all())

    def test_zero_length_encode_scatters_and_backward(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms(requires_grad=True)
        lengths = lengths.clone()
        lengths[1] = 0
        encoded = MODEL.encode(wav, lengths, layers=(-1,))
        self.assertEqual(int(encoded.encoded_length[1]), 0)
        self.assertEqual(float(encoded.encoded[1].abs().sum()), 0.0)
        last = max(encoded.layer_outputs)
        self.assertEqual(tuple(encoded.layer_outputs[last].shape), tuple(encoded.encoded.shape))
        encoded.encoded.sum().backward()
        self.assertEqual(float(wav.grad[1].abs().sum()), 0.0)

        MODEL.zero_grad(set_to_none=True)
        wav, _ = _waveforms(requires_grad=True)
        zeros = torch.zeros(2, device=DEVICE, dtype=torch.int64)
        empty = MODEL.encode(wav, zeros, layers=(0,))
        self.assertEqual(empty.encoded.shape[0], 2)
        self.assertGreater(empty.encoded.shape[1], 0)
        self.assertEqual(empty.encoded.shape[2], 0)
        self.assertTrue(torch.equal(empty.encoded_length.cpu(), torch.zeros(2, dtype=torch.int64)))
        empty.encoded.sum().backward()
        self.assertTrue(torch.isfinite(wav.grad).all())


class ParakeetRecognizeTest(_ParakeetCase):
    def test_recognition_timestamps_match_encoder_frames(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms()
        results = MODEL.recognize(wav, lengths)
        self.assertEqual(len(results), 2)
        for rec in results:
            self.assertIsInstance(rec.text, str)
            self.assertEqual(len(rec.frame_token_index), rec.encoded_length)
            self.assertEqual(rec.samples_per_encoder_frame, MODEL.samples_per_encoder_frame)
            for token in rec.tokens:
                self.assertGreaterEqual(token.start_offset, 0)
                self.assertLess(token.start_offset, token.end_offset)
                self.assertLessEqual(token.end_offset, rec.encoded_length)
            for index in rec.frame_token_index:
                self.assertGreaterEqual(index, -1)
                self.assertLess(index, len(rec.tokens))

    def test_zero_length_recognize_is_empty(self) -> None:
        MODEL.eval()
        wav, lengths = _waveforms()
        lengths = lengths.clone()
        lengths[1] = 0
        results = MODEL.recognize(wav, lengths)
        empty = results[1]
        self.assertEqual(empty.text, "")
        self.assertEqual(empty.encoded_length, 0)
        self.assertEqual(empty.tokens, [])
        self.assertEqual(empty.frame_token_index, [])


@unittest.skipUnless(
    _speech_env_complete(),
    "PARAKEET_TEST_AUDIO/TRANSCRIPT/UTTERANCE_ID are not set",
)
class ParakeetTranscriptTest(_ParakeetCase):
    def _load_speech(self) -> tuple[torch.Tensor, str]:
        utterance_id = os.environ["PARAKEET_TEST_UTTERANCE_ID"]
        text_gt = None
        with open(os.environ["PARAKEET_TEST_TRANSCRIPT"], encoding="utf-8") as trans_file:
            for line in trans_file:
                utterance, _, transcript = line.rstrip("\n").partition(" ")
                if utterance == utterance_id:
                    text_gt = transcript
                    break
        if text_gt is None:
            self.fail(f"ground-truth transcript not found for {utterance_id}")
        wav, sample_rate = torchaudio.load(os.environ["PARAKEET_TEST_AUDIO"])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        wav = wav.to(device=DEVICE, dtype=torch.float32)
        return wav, text_gt

    def test_non_empty_text_and_closer_transcript_has_lower_nll(self) -> None:
        wav, text_gt = self._load_speech()
        batch = wav.repeat(2, 1)
        lengths = torch.full((2,), batch.shape[-1], device=DEVICE, dtype=torch.int64)
        nll_close = MODEL.loss(batch, lengths, [text_gt, text_gt])
        with torch.no_grad():
            nll_far = MODEL.loss(batch, lengths, ["The weather is nice today in Tokyo."] * 2)
        self.assertTrue(torch.all(nll_far > nll_close.detach()))
        recognized = MODEL.recognize(batch.detach(), lengths)
        self.assertIsInstance(recognized[0].text, str)
        self.assertNotEqual(recognized[0].text, "")
        self.assertTrue(any(index >= 0 for index in recognized[0].frame_token_index))
