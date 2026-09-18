import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import numpy as np

from src.data.db import (
    Connection,
    corpora_root_dir,
    fetch_noise_by_id,
    fetch_noise_configs,
    fetch_noises,
    fetch_utterances,
    import_rows,
    metadata_path,
)
from src.data.schema import UTTERANCES_TABLE, Utterance
from tests.helpers import speech_env, write_audio

SAMPLE_RATE = 16000
N_SAMPLES = 64


def _tone(frames: int, freq: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(-1, 1)


def _run_cli(module: str, args: list[str]) -> None:
    with mock.patch.object(sys, "argv", args):
        runpy.run_module(module, run_name="__main__")


def _write_librispeech(corpora: Path) -> None:
    chapter = corpora / "LibriSpeech" / "train-clean-100" / "1234" / "56789"
    write_audio(chapter / "1234-56789-0000.flac", _tone(N_SAMPLES, 220.0), SAMPLE_RATE)
    write_audio(chapter / "1234-56789-0001.flac", _tone(N_SAMPLES, 330.0), SAMPLE_RATE)
    (chapter / "1234-56789.trans.txt").write_text(
        "1234-56789-0000 hello world\n1234-56789-0001 second line\n",
        encoding="utf-8",
    )


def _write_libritts(corpora: Path) -> None:
    chapter = corpora / "LibriTTS" / "train-clean-100" / "14" / "000"
    write_audio(chapter / "14_000_000.wav", _tone(N_SAMPLES, 440.0), SAMPLE_RATE)
    (chapter / "14_000.trans.tsv").write_text("14_000_000\tignored\ta libritts line\n", encoding="utf-8")


def _write_vctk(corpora: Path) -> None:
    wav = corpora / "VCTK" / "wav48" / "p225" / "p225_001.wav"
    txt = corpora / "VCTK" / "txt" / "p225" / "p225_001.txt"
    write_audio(wav, _tone(N_SAMPLES, 110.0), SAMPLE_RATE)
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("Please call Stella.\n", encoding="utf-8")


def _write_demand(corpora: Path) -> None:
    env_dir = corpora / "DEMAND" / "OMEETING"
    write_audio(env_dir / "ch01.wav", _tone(N_SAMPLES, 80.0), SAMPLE_RATE)
    write_audio(env_dir / "ch02.wav", _tone(N_SAMPLES, 90.0), SAMPLE_RATE)


class EnvUnsetTest(unittest.TestCase):
    def test_missing_db_root_raises(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("SPEECH_DB_ROOT_DIR", None)
            with self.assertRaises(RuntimeError) as ctx:
                metadata_path()
            self.assertEqual(str(ctx.exception), "SPEECH_DB_ROOT_DIR is not set")

    def test_missing_corpora_root_raises(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("SPEECH_CORPORA_ROOT_DIR", None)
            with self.assertRaises(RuntimeError) as ctx:
                corpora_root_dir()
            self.assertEqual(str(ctx.exception), "SPEECH_CORPORA_ROOT_DIR is not set")


class MetadataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env = speech_env(Path(self.tmp.name))
        self.db_dir, self.corpora = self.env.__enter__()
        _write_librispeech(self.corpora)
        _write_libritts(self.corpora)
        _write_vctk(self.corpora)
        _write_demand(self.corpora)

    def tearDown(self) -> None:
        self.env.__exit__(None, None, None)
        self.tmp.cleanup()

    def _db(self) -> Path:
        return metadata_path()

    def test_unknown_corpus_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            _run_cli("src.data.utterances", ["utterances", "--corpus", "nope"])

    def test_cli_import_fetch_replace_and_ids(self) -> None:
        _run_cli("src.data.utterances", ["utterances", "--corpus", "librispeech"])
        _run_cli("src.data.utterances", ["utterances", "--corpus", "libritts"])
        _run_cli("src.data.utterances", ["utterances", "--corpus", "vctk"])
        _run_cli("src.data.noises", ["noises", "--corpus", "demand"])
        _run_cli("src.data.noise_configs", ["noise_configs"])

        with Connection(self._db(), read_only=True) as connection:
            libri = fetch_utterances(connection, "LibriSpeech", ["train-clean-100"])
            tts = fetch_utterances(connection, "LibriTTS", ["train-clean-100"])
            vctk = fetch_utterances(connection, "VCTK", ["missing"])
            noises = fetch_noises(connection)
            train_cfgs = fetch_noise_configs(connection, "train", None)
            dev_cfgs = fetch_noise_configs(connection, "dev", None)
            test_cfgs = fetch_noise_configs(connection, "test", None)

        self.assertEqual(len(libri), 2)
        self.assertEqual([row.utterance_id for row in libri], ["0000", "0001"])
        self.assertTrue(all(isinstance(row.audio_path, Path) for row in libri))
        self.assertEqual(libri[0].text, "hello world")
        self.assertEqual(tts[0].corpus, "LibriTTS")
        self.assertEqual(tts[0].text, "a libritts line")
        self.assertEqual(vctk, [])

        with Connection(self._db(), read_only=True) as connection:
            vctk_all = connection.fetchall(
                "SELECT utterances.corpus, utterances.subset, utterances.text FROM utterances WHERE corpus = 'VCTK'"
            )
        self.assertEqual(len(vctk_all), 1)
        self.assertEqual(vctk_all[0][1], None)
        self.assertEqual(vctk_all[0][2], "Please call Stella.")

        first_ids = {row.audio_path.as_posix(): row.id for row in libri}
        self.assertEqual(list(first_ids.values()), [1, 2])

        self.assertEqual(len(noises), 2)
        self.assertEqual([row.audio_path.name for row in noises], ["ch01.wav", "ch02.wav"])
        ch01 = next(row for row in noises if row.audio_path.name == "ch01.wav")
        self.assertEqual(len(train_cfgs) + len(dev_cfgs) + len(test_cfgs), 210)
        self.assertEqual(len(train_cfgs), 168)
        self.assertEqual(len(dev_cfgs), 21)
        self.assertEqual(len(test_cfgs), 21)
        payload = train_cfgs[0].json
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["version"], "1.0")
        self.assertEqual(payload["pipeline"][0]["method"], "additive")
        self.assertEqual(payload["pipeline"][0]["params"]["noise_id"], str(ch01.id))
        self.assertIsInstance(payload["pipeline"][0]["params"]["noise_id"], str)

        with self.assertRaises(duckdb.Error):
            _run_cli("src.data.utterances", ["utterances", "--corpus", "librispeech"])

        _run_cli("src.data.utterances", ["utterances", "--corpus", "librispeech", "--replace"])
        with Connection(self._db(), read_only=True) as connection:
            after = fetch_utterances(connection, "LibriSpeech", ["train-clean-100"])
            tts_after = fetch_utterances(connection, "LibriTTS", ["train-clean-100"])
            still_noises = fetch_noises(connection)
        self.assertEqual({row.audio_path.as_posix(): row.id for row in after}, first_ids)
        self.assertEqual(tts_after, [])
        self.assertEqual(len(still_noises), 2)

        _run_cli("src.data.noise_configs", ["noise_configs", "--replace"])
        with Connection(self._db(), read_only=True) as connection:
            self.assertEqual(len(fetch_noises(connection)), 2)
            self.assertEqual(len(fetch_noise_configs(connection, "train", None)), 168)

    def test_fetch_utterances_requires_subsets_and_exact_corpus(self) -> None:
        import_rows(
            self._db(),
            UTTERANCES_TABLE,
            [
                Utterance(
                    corpus="LibriSpeech",
                    subset="train-clean-100",
                    audio_path=self.corpora / "a.flac",
                    sample_rate=SAMPLE_RATE,
                    frames=N_SAMPLES,
                    channels=1,
                ),
                Utterance(
                    corpus="librispeech",
                    subset="train-clean-100",
                    audio_path=self.corpora / "b.flac",
                    sample_rate=SAMPLE_RATE,
                    frames=N_SAMPLES,
                    channels=1,
                ),
            ],
        )
        with Connection(self._db(), read_only=True) as connection:
            with self.assertRaises(ValueError):
                fetch_utterances(connection, "LibriSpeech", [])
            rows = fetch_utterances(connection, "LibriSpeech", ["train-clean-100"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].corpus, "LibriSpeech")

    def test_fetch_noise_by_id_missing(self) -> None:
        _run_cli("src.data.noises", ["noises", "--corpus", "demand"])
        with Connection(self._db(), read_only=True) as connection:
            with self.assertRaises(ValueError):
                fetch_noise_by_id(connection, 999999)
