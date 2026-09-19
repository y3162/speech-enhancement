import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import numpy as np

from src.asr.timestamp_cache import (
    crop_batch_token_ids,
    crop_token_ids,
    load_timestamp_cache,
    missing_timestamp_keys,
    require_timestamp_record,
)
from src.asr.timestamp_crop import slice_tokens_for_crop
from src.data.db import Connection, import_rows
from src.data.schema import ASR_TIMESTAMPS_TABLE, AsrTimestamp, AsrToken, UTTERANCES_TABLE, Utterance
from tests.helpers import speech_env, write_audio


def _record(key: str, tokens: list[AsrToken], text: str = "hello") -> AsrTimestamp:
    return AsrTimestamp(utterance_key=key, text=text, samples_per_encoder_frame=1280, tokens=tokens)


def _core(record: AsrTimestamp) -> tuple[object, ...]:
    return (record.utterance_key, record.text, record.samples_per_encoder_frame, record.tokens)


class TimestampCacheLoadTest(unittest.TestCase):
    def test_load_lookup_and_missing_key(self) -> None:
        record = _record("1234-56789-0000", [AsrToken(1, "a", 0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            with speech_env(Path(tmp)) as (db_dir, _):
                import_rows(db_dir / "metadata.duckdb", ASR_TIMESTAMPS_TABLE, [record])
                cache = load_timestamp_cache()
                self.assertEqual(list(cache), ["1234-56789-0000"])
                loaded = require_timestamp_record(cache, "1234-56789-0000")
                self.assertEqual(_core(loaded), _core(record))
                with self.assertRaises(KeyError) as ctx:
                    require_timestamp_record(cache, "1234-56789-0001")
                self.assertIn("timestamp cache miss", str(ctx.exception))
                self.assertEqual(missing_timestamp_keys(cache, ["1234-56789-0000", "missing"]), ["missing"])

    def test_empty_text_roundtrips(self) -> None:
        record = _record("1234-56789-0000", [], text="")
        with tempfile.TemporaryDirectory() as tmp:
            with speech_env(Path(tmp)) as (db_dir, _):
                import_rows(db_dir / "metadata.duckdb", ASR_TIMESTAMPS_TABLE, [record])
                loaded = require_timestamp_record(load_timestamp_cache(), "1234-56789-0000")
                self.assertEqual(loaded.text, "")
                self.assertEqual(loaded.tokens, [])

    def test_duplicate_keys_raise_on_insert(self) -> None:
        record = _record("1234-56789-0000", [])
        with tempfile.TemporaryDirectory() as tmp:
            with speech_env(Path(tmp)) as (db_dir, _):
                db_path = db_dir / "metadata.duckdb"
                import_rows(db_path, ASR_TIMESTAMPS_TABLE, [record])
                with self.assertRaises(duckdb.Error):
                    import_rows(db_path, ASR_TIMESTAMPS_TABLE, [record])

    def test_tokens_column_is_json_and_rejects_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with speech_env(Path(tmp)) as (db_dir, _):
                db_path = db_dir / "metadata.duckdb"
                import_rows(db_path, ASR_TIMESTAMPS_TABLE, [_record("1234-56789-0000", [])])
                with Connection(db_path) as connection:
                    column_type = connection.fetchall(
                        "SELECT data_type FROM information_schema.columns "
                        "WHERE table_name = 'asr_timestamps' AND column_name = 'tokens'"
                    )
                    self.assertEqual(column_type, [("JSON",)])
                    with self.assertRaises(duckdb.Error) as ctx:
                        connection.fetchall(
                            "INSERT INTO asr_timestamps (utterance_key, text, samples_per_encoder_frame, tokens) "
                            "VALUES (?, ?, ?, ?)",
                            ("bad", "t", 1280, "not json"),
                        )
                    self.assertIn("JSON", str(ctx.exception).upper())


class CropTokenIdsTest(unittest.TestCase):
    def test_crop_token_ids_match_slice_order_and_empty(self) -> None:
        tokens = [
            AsrToken(3, "c", 6, 7),
            AsrToken(1, "a", 0, 1),
            AsrToken(2, "b", 3, 4),
        ]
        record = _record("k", tokens)
        selected = slice_tokens_for_crop(tokens, 0, 10_000)
        self.assertEqual(crop_token_ids(record, 0, 10_000), [token.token_id for token in selected])
        self.assertEqual(crop_token_ids(record, 3000, 4000), [])

    def test_crop_batch_token_ids_looks_up_and_checks_lengths(self) -> None:
        left = _record("a", [AsrToken(1, "x", 0, 1)])
        right = _record("b", [AsrToken(2, "y", 0, 2), AsrToken(3, "z", 2, 3)])
        cache = {"a": left, "b": right}
        self.assertEqual(
            crop_batch_token_ids(cache, ["a", "b"], [0, 0], [2000, 10_000]),
            [crop_token_ids(left, 0, 2000), crop_token_ids(right, 0, 10_000)],
        )
        with self.assertRaises(ValueError):
            crop_batch_token_ids(cache, ["a"], [0, 1], [10])
        with self.assertRaises(KeyError):
            crop_batch_token_ids(cache, ["missing"], [0], [10])


class TimestampCacheCliTest(unittest.TestCase):
    def test_check_and_replace_conflict(self) -> None:
        from src.asr.timestamp_cache import main

        with mock.patch.object(
            sys,
            "argv",
            ["timestamp_cache", "--splits", "train-clean-100", "--check", "--replace"],
        ):
            with self.assertRaises(SystemExit):
                main()

    def test_generate_skips_existing_without_asr(self) -> None:
        from src.asr.timestamp_cache import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with speech_env(root) as (db_dir, corpora):
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
                record = _record("1234-56789-0000", [])
                import_rows(db_dir / "metadata.duckdb", ASR_TIMESTAMPS_TABLE, [record])
                with mock.patch.object(
                    sys,
                    "argv",
                    ["timestamp_cache", "--splits", "train-clean-100"],
                ):
                    main()
                cache = load_timestamp_cache()
                self.assertEqual(list(cache), ["1234-56789-0000"])
                self.assertEqual(_core(cache["1234-56789-0000"]), _core(record))

    def test_check_reports_missing_and_succeeds_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with speech_env(root) as (db_dir, corpora):
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
                argv_check = ["timestamp_cache", "--splits", "train-clean-100", "--check"]
                with mock.patch.object(sys, "argv", argv_check):
                    with self.assertRaises(SystemExit) as ctx:
                        from src.asr.timestamp_cache import main

                        main()
                self.assertEqual(ctx.exception.code, 1)
                import_rows(
                    db_dir / "metadata.duckdb",
                    ASR_TIMESTAMPS_TABLE,
                    [_record("1234-56789-0000", [])],
                )
                with mock.patch.object(sys, "argv", argv_check):
                    with self.assertRaises(SystemExit) as ctx_ok:
                        main()
                self.assertEqual(ctx_ok.exception.code, 0)
