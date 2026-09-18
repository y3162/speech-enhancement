import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.audio import read_audio_segment
from src.data.db import Connection, fetch_noise_configs, fetch_noises, fetch_utterances, import_rows, metadata_path
from src.data.noise import generate
from src.data.schema import (
    NOISE_CONFIGS_TABLE,
    NOISES_TABLE,
    UTTERANCES_TABLE,
    Noise,
    NoiseConfig,
    Utterance,
)
from src.se.common.dataset import (
    AdditiveNoiseDataset,
    _crop_or_pad,
    _noise_split,
    _normalize_pair,
    build_datasets,
    pad_collate,
    worker_init_fn,
)
from src.se.common.training import load_config
from tests.helpers import se_config, speech_env, write_audio

SAMPLE_RATE = 16000
N_SAMPLES = 1600


def _tone(frames: int, freq: float) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(-1, 1)


def _noise_json(noise_id: int, split: str, snr_db: float) -> NoiseConfig:
    return NoiseConfig(
        json={
            "version": "1.0",
            "pipeline": [
                {
                    "method": "additive",
                    "params": {
                        "seed": 0,
                        "noise_id": str(noise_id),
                        "target_range": {"type": "all"},
                        "noise_valid_range": {"start_ratio": 0.0, "end_ratio": 1.0},
                        "snr_db": snr_db,
                    },
                }
            ],
        },
        split=split,
    )


def _write_pair(corpora: Path, subset: str, utterance_id: str, freq: float) -> Path:
    speaker = "1234"
    chapter = "56789"
    stem = f"{speaker}-{chapter}-{utterance_id}"
    path = corpora / "LibriSpeech" / subset / speaker / chapter / f"{stem}.flac"
    write_audio(path, _tone(N_SAMPLES, freq), SAMPLE_RATE)
    return path


def _seed_database(corpora: Path, db_path: Path) -> None:
    train_path = _write_pair(corpora, "train-clean-100", "0000", 220.0)
    valid_path = _write_pair(corpora, "dev-clean", "0001", 330.0)
    noise_path = corpora / "DEMAND" / "OMEETING" / "ch01.wav"
    write_audio(noise_path, _tone(N_SAMPLES, 80.0), SAMPLE_RATE)
    import_rows(
        db_path,
        UTTERANCES_TABLE,
        [
            Utterance(
                corpus="LibriSpeech",
                subset="train-clean-100",
                speaker_id="1234",
                chapter_id="56789",
                utterance_id="0000",
                audio_path=train_path,
                sample_rate=SAMPLE_RATE,
                frames=N_SAMPLES,
                channels=1,
                text="train",
            ),
            Utterance(
                corpus="LibriSpeech",
                subset="dev-clean",
                speaker_id="1234",
                chapter_id="56789",
                utterance_id="0001",
                audio_path=valid_path,
                sample_rate=SAMPLE_RATE,
                frames=N_SAMPLES,
                channels=1,
                text="valid",
            ),
        ],
    )
    import_rows(
        db_path,
        NOISES_TABLE,
        [
            Noise(
                corpus="DEMAND",
                subset="OMEETING",
                noise_id="ch01",
                audio_path=noise_path,
                sample_rate=SAMPLE_RATE,
                frames=N_SAMPLES,
                channels=1,
            )
        ],
    )
    with Connection(db_path) as connection:
        noise_id = fetch_noises(connection)[0].id
    import_rows(
        db_path,
        NOISE_CONFIGS_TABLE,
        [_noise_json(int(noise_id), "train", 5.0), _noise_json(int(noise_id), "dev", 0.0)],
    )


class NoiseSplitTest(unittest.TestCase):
    def test_train_dev_test_prefixes(self) -> None:
        self.assertEqual(_noise_split(["train-clean-100", "train-clean-360"]), "train")
        self.assertEqual(_noise_split(["dev-clean"]), "dev")
        self.assertEqual(_noise_split(["test-clean"]), "test")

    def test_mixed_or_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            _noise_split(["train-clean-100", "dev-clean"])
        with self.assertRaises(ValueError):
            _noise_split(["foo"])


class NormalizePairTest(unittest.TestCase):
    def test_energy_and_peak_invariants(self) -> None:
        clean = torch.linspace(0.1, 1.0, 20)
        noisy = torch.linspace(1.0, 2.0, 20)
        energy_clean, energy_noisy = _normalize_pair(clean, noisy, "energy")
        self.assertTrue(torch.allclose(energy_noisy.pow(2).mean(), torch.tensor(1.0), atol=1e-5))
        self.assertTrue(torch.allclose(energy_clean / clean, energy_noisy / noisy))
        peak_clean, peak_noisy = _normalize_pair(clean, noisy, "peak")
        self.assertTrue(torch.allclose(peak_noisy.abs().amax(), torch.tensor(1.0), atol=1e-5))
        self.assertTrue(torch.allclose(peak_clean / clean, peak_noisy / noisy))

    def test_unsupported_normalize_on_dataset(self) -> None:
        with self.assertRaises(ValueError):
            AdditiveNoiseDataset(
                utterances=[Utterance(corpus="LibriSpeech", audio_path=Path("x.flac"))],
                noise_configs=[NoiseConfig(json={}, split="train")],
                database_path=Path("."),
                sampling_rate=SAMPLE_RATE,
                segment_size=32,
                normalize="nope",
                crop=True,
            )


class CropOrPadTest(unittest.TestCase):
    def test_crop_pad_and_identity_length(self) -> None:
        rng = random.Random(0)
        cropped_clean, cropped_noisy = _crop_or_pad(torch.arange(10.0), torch.arange(10.0) + 1, 6, rng)
        self.assertEqual(tuple(cropped_clean.shape), (6,))
        self.assertEqual(tuple(cropped_noisy.shape), (6,))
        padded_clean, padded_noisy = _crop_or_pad(torch.arange(4.0), torch.arange(4.0) + 1, 6, random.Random(0))
        self.assertEqual(tuple(padded_clean.shape), (6,))
        self.assertTrue(torch.equal(padded_clean[:4], torch.arange(4.0)))
        self.assertTrue(torch.equal(padded_noisy[4:], torch.zeros(2)))
        equal_clean, equal_noisy = _crop_or_pad(torch.arange(6.0), torch.arange(6.0) + 1, 6, random.Random(0))
        self.assertTrue(torch.equal(equal_clean, torch.arange(6.0)))
        self.assertTrue(torch.equal(equal_noisy, torch.arange(6.0) + 1))


class PadCollateTest(unittest.TestCase):
    def test_pads_to_max_length(self) -> None:
        batch = [
            (torch.arange(3.0), torch.arange(3.0) + 10),
            (torch.arange(5.0), torch.arange(5.0) + 10),
        ]
        clean, noisy, lengths = pad_collate(batch)
        self.assertEqual(tuple(clean.shape), (2, 5))
        self.assertEqual(tuple(noisy.shape), (2, 5))
        self.assertTrue(torch.equal(lengths, torch.tensor([3, 5])))
        self.assertTrue(torch.equal(clean[0, 3:], torch.zeros(2)))


class BuildDatasetsErrorsTest(unittest.TestCase):
    def test_empty_splits_are_required(self) -> None:
        cfg = load_config(se_config("mp_senet"))
        cfg.data.train_splits = []
        with self.assertRaises(ValueError) as ctx:
            build_datasets(cfg)
        self.assertEqual(str(ctx.exception), "data.train_splits is required")
        cfg = load_config(se_config("mp_senet"))
        cfg.data.validation_splits = []
        with self.assertRaises(ValueError) as ctx:
            build_datasets(cfg)
        self.assertEqual(str(ctx.exception), "data.validation_splits is required")


class AdditiveNoiseDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env = speech_env(Path(self.tmp.name))
        self.db_dir, self.corpora = self.env.__enter__()
        self.db_path = self.db_dir / "metadata.duckdb"
        _seed_database(self.corpora, self.db_path)

    def tearDown(self) -> None:
        worker_init_fn(0)
        self.env.__exit__(None, None, None)
        self.tmp.cleanup()

    def _dataset(self, **kwargs: object) -> AdditiveNoiseDataset:
        with Connection(self.db_path, read_only=True) as connection:
            utts = fetch_utterances(connection, "LibriSpeech", ["train-clean-100"])
            noises = fetch_noise_configs(connection, "train", None)
        defaults: dict[str, object] = {
            "utterances": utts,
            "noise_configs": noises,
            "database_path": self.db_path,
            "sampling_rate": SAMPLE_RATE,
            "segment_size": N_SAMPLES,
            "normalize": "energy",
            "crop": False,
            "seed": 0,
        }
        defaults.update(kwargs)
        return AdditiveNoiseDataset(**defaults)

    def test_getitem_is_mono_and_noisy_is_normalized_clean_plus_noise(self) -> None:
        dataset = self._dataset()
        clean, noisy = dataset[0]
        self.assertEqual(tuple(clean.shape), (N_SAMPLES,))
        self.assertEqual(tuple(noisy.shape), (N_SAMPLES,))
        utterance = dataset.utterances[0]
        clean_2d = read_audio_segment(utterance.audio_path, 0, int(utterance.frames))
        with Connection(self.db_path, read_only=True) as connection:
            noise_2d = generate(clean_2d, SAMPLE_RATE, dataset.noise_configs[0], connection)
        raw_clean = torch.from_numpy(np.ascontiguousarray(clean_2d[:, 0]))
        raw_noisy = torch.from_numpy(np.ascontiguousarray((clean_2d + noise_2d)[:, 0]))
        expected_clean, expected_noisy = _normalize_pair(raw_clean, raw_noisy, "energy")
        self.assertTrue(torch.allclose(clean, expected_clean, atol=1e-5))
        self.assertTrue(torch.allclose(noisy, expected_noisy, atol=1e-5))
        self.assertTrue(torch.allclose(noisy.pow(2).mean(), torch.tensor(1.0), atol=1e-5))

    def test_seed_makes_getitem_deterministic(self) -> None:
        first = self._dataset(seed=42)[0]
        second = self._dataset(seed=42)[0]
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_build_datasets_rejects_mixed_noise_splits(self) -> None:
        cfg = load_config(se_config("mp_senet"))
        cfg.data.train_splits = ["train-clean-100", "dev-clean"]
        cfg.data.validation_splits = ["dev-clean"]
        with self.assertRaises(ValueError):
            build_datasets(cfg)

    def test_pcs400_applies_to_train_clean_only(self) -> None:
        cfg = load_config(se_config("se_mamba", "pcs.json"))
        cfg.data.train_splits = ["train-clean-100"]
        cfg.data.validation_splits = ["dev-clean"]
        self.assertEqual(metadata_path(), self.db_path)
        train, valid = build_datasets(cfg)
        self.assertTrue(train.pcs400)
        self.assertFalse(valid.pcs400)

    def test_cal_pcs_keeps_length_and_peak_normalizes(self) -> None:
        from src.se.common.pcs import cal_pcs

        wave = np.linspace(-0.3, 0.3, N_SAMPLES, dtype=np.float32)
        out = cal_pcs(wave)
        self.assertEqual(out.shape, wave.shape)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, places=5)
