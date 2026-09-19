import tempfile
import unittest
from pathlib import Path

import torch

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
from src.asr.timestamp_cache import (
    crop_token_ids,
    load_timestamp_cache,
    require_timestamp_record,
)
from src.asr.timestamp_crop import slice_tokens_for_crop
from src.data.db import import_rows
from src.data.schema import ASR_TIMESTAMPS_TABLE, AsrTimestamp, AsrToken
from src.se.se_mamba_pp.train import _tdt_targets_from_cache, _wav_lengths
from tests.helpers import speech_env

DEVICE: torch.device
ASR: ParakeetTDT06BV2


def setUpModule() -> None:
    global DEVICE, ASR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for timestamp cache GPU tests")
    DEVICE = torch.device("cuda")
    ASR = ParakeetTDT06BV2().to(DEVICE)
    ASR.eval()
    for parameter in ASR.parameters():
        parameter.requires_grad_(False)


class TimestampCacheGpuTest(unittest.TestCase):
    def test_full_recognize_cache_slice_and_tdt_target(self) -> None:
        torch.manual_seed(0)
        samples = 32000
        wav = (0.05 * torch.randn(1, samples, device=DEVICE)).to(dtype=torch.float32)
        lengths = _wav_lengths(wav)
        first = ASR.recognize(wav, lengths)[0]
        second = ASR.recognize(wav, lengths)[0]
        self.assertEqual(first.text, second.text)
        self.assertEqual(
            [(token.token_id, token.start_offset, token.end_offset) for token in first.tokens],
            [(token.token_id, token.start_offset, token.end_offset) for token in second.tokens],
        )
        record = AsrTimestamp(
            utterance_key="1234-56789-0000",
            text=first.text,
            samples_per_encoder_frame=first.samples_per_encoder_frame,
            tokens=[
                AsrToken(token.token_id, token.token, token.start_offset, token.end_offset) for token in first.tokens
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with speech_env(Path(tmp)) as (db_dir, _):
                import_rows(db_dir / "metadata.duckdb", ASR_TIMESTAMPS_TABLE, [record])
                cache = load_timestamp_cache()
                loaded = require_timestamp_record(cache, "1234-56789-0000")
                self.assertEqual(loaded.utterance_key, record.utterance_key)
                self.assertEqual(loaded.text, record.text)
                self.assertEqual(
                    [(token.token_id, token.start_offset, token.end_offset) for token in loaded.tokens],
                    [(token.token_id, token.start_offset, token.end_offset) for token in record.tokens],
                )
                crop_start, crop_end = 0, 16000
                crop_ids = crop_token_ids(loaded, crop_start, crop_end)
                selected = slice_tokens_for_crop(record.tokens, crop_start, crop_end, record.samples_per_encoder_frame)
                self.assertEqual(crop_ids, [token.token_id for token in selected])
                token_ids, empty = _tdt_targets_from_cache(
                    cache,
                    ["1234-56789-0000"],
                    torch.tensor([crop_start]),
                    torch.tensor([crop_end]),
                )
                self.assertEqual(token_ids[0], crop_ids)
                other_start, other_end = 16000, samples
                other_ids = crop_token_ids(record, other_start, other_end)
                if crop_ids or other_ids:
                    self.assertNotEqual(crop_ids, other_ids)
                nll = ASR.loss_from_ids(
                    wav[:, crop_start:crop_end],
                    torch.tensor([crop_end - crop_start], device=DEVICE),
                    token_ids,
                )
                self.assertTrue(torch.isfinite(nll).all())
                self.assertEqual(empty, int(len(token_ids[0]) == 0))

    def test_ids_to_text_roundtrip_empty_and_tokens(self) -> None:
        self.assertEqual(ASR.ids_to_text([]), "")
        ids = ASR.model.tokenizer.text_to_ids("hello world")
        text = ASR.ids_to_text(ids)
        self.assertIsInstance(text, str)
        self.assertNotEqual(text, "")
        self.assertEqual(ASR.model.tokenizer.text_to_ids(text), ids)
