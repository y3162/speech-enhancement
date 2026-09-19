import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.asr.timestamp_cache import (
    CachedToken,
    TimestampRecord,
    crop_token_ids,
    load_timestamp_cache,
    missing_timestamp_keys,
    require_timestamp_record,
    timestamp_record_from_json,
    timestamp_record_to_json,
)
from src.asr.timestamp_crop import slice_tokens_for_crop
from src.data.db import import_rows
from src.data.schema import UTTERANCES_TABLE, Utterance
from tests.helpers import speech_env, write_audio


def _record(key: str, tokens: list[CachedToken], text: str = "hello") -> TimestampRecord:
    return TimestampRecord(utterance_key=key, text=text, samples_per_encoder_frame=1280, tokens=tokens)


def _write_jsonl(path: Path, records: list[TimestampRecord]) -> None:
    path.write_text(
        "".join(json.dumps(timestamp_record_to_json(record)) + "\n" for record in records),
        encoding="utf-8",
    )


class TimestampRecordJsonTest(unittest.TestCase):
    def test_roundtrip_maps_frames_to_offsets(self) -> None:
        record = _record(
            "6078-54013-0037",
            [
                CachedToken(token_id=11, token="\u2581hello", start_offset=0, end_offset=2),
                CachedToken(token_id=22, token="\u2581world", start_offset=4, end_offset=6),
            ],
        )
        payload = timestamp_record_to_json(record)
        self.assertEqual(payload["utterance_key"], "6078-54013-0037")
        self.assertEqual(payload["tokens"][0]["start_frame"], 0)
        self.assertEqual(payload["tokens"][0]["end_frame"], 2)
        restored = timestamp_record_from_json(payload)
        self.assertEqual(restored, record)


class TimestampCacheLoadTest(unittest.TestCase):
    def test_load_lookup_and_missing_key(self) -> None:
        record = _record("1234-56789-0000", [CachedToken(1, "a", 0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            _write_jsonl(path, [record])
            cache = load_timestamp_cache(path)
            self.assertEqual(list(cache), ["1234-56789-0000"])
            self.assertEqual(require_timestamp_record(cache, "1234-56789-0000"), record)
            with self.assertRaises(KeyError) as ctx:
                require_timestamp_record(cache, "1234-56789-0001")
            self.assertIn("timestamp cache miss", str(ctx.exception))
            self.assertEqual(missing_timestamp_keys(cache, ["1234-56789-0000", "missing"]), ["missing"])

    def test_duplicate_keys_raise(self) -> None:
        record = _record("1234-56789-0000", [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            _write_jsonl(path, [record, record])
            with self.assertRaises(ValueError) as ctx:
                load_timestamp_cache(path)
            self.assertIn("duplicate timestamp cache keys", str(ctx.exception))


class CropTokenIdsTest(unittest.TestCase):
    def test_crop_token_ids_match_slice_order_and_empty(self) -> None:
        tokens = [
            CachedToken(3, "c", 6, 7),
            CachedToken(1, "a", 0, 1),
            CachedToken(2, "b", 3, 4),
        ]
        record = _record("k", tokens)
        selected = slice_tokens_for_crop(tokens, 0, 10_000)
        self.assertEqual(crop_token_ids(record, 0, 10_000), [token.token_id for token in selected])
        self.assertEqual(crop_token_ids(record, 3000, 4000), [])


class TimestampCacheCliTest(unittest.TestCase):
    def test_overwrite_and_skip_existing_conflict(self) -> None:
        from src.asr.timestamp_cache import main

        with mock.patch.object(
            sys,
            "argv",
            [
                "timestamp_cache",
                "--splits",
                "train-clean-100",
                "--output",
                "cache.jsonl",
                "--overwrite",
                "--skip-existing",
            ],
        ):
            with self.assertRaises(SystemExit):
                main()

    def test_check_reports_missing_and_succeeds_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = speech_env(root)
            db_dir, corpora = env.__enter__()
            try:
                audio = corpora / "LibriSpeech" / "train-clean-100" / "1234" / "56789" / "1234-56789-0000.flac"
                write_audio(audio, np.zeros((64, 1), dtype=np.float32), 16000)
                import_rows(
                    db_dir / "metadata.duckdb",
                    UTTERANCES_TABLE,
                    [
                        Utterance(
                            corpus="LibriSpeech",
                            subset="train-clean-100",
                            speaker_id="1234",
                            chapter_id="56789",
                            utterance_id="0000",
                            audio_path=audio,
                            sample_rate=16000,
                            frames=64,
                            channels=1,
                            text="train",
                        )
                    ],
                )
                cache_path = root / "cache.jsonl"
                argv_missing = [
                    "timestamp_cache",
                    "--splits",
                    "train-clean-100",
                    "--output",
                    str(cache_path),
                    "--check",
                ]
                cache_path.write_text("", encoding="utf-8")
                with mock.patch.object(sys, "argv", argv_missing):
                    with self.assertRaises(SystemExit) as ctx:
                        from src.asr.timestamp_cache import main

                        main()
                self.assertEqual(ctx.exception.code, 1)
                _write_jsonl(cache_path, [_record("1234-56789-0000", [])])
                with mock.patch.object(sys, "argv", argv_missing):
                    with self.assertRaises(SystemExit) as ctx_ok:
                        main()
                self.assertEqual(ctx_ok.exception.code, 0)
            finally:
                env.__exit__(None, None, None)
