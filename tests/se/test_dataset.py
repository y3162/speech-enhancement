import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.audio import read_audio_segment
from src.data.corpora.librispeech import utterance_key
from src.data.db import Connection, fetch_noise_configs, fetch_noises, fetch_utterances, import_rows, metadata_path
from src.data.noise import generate, parse_pipeline
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
    AudioPair,
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


class UtteranceKeyTest(unittest.TestCase):
    def test_uses_speaker_chapter_utterance_not_short_id(self) -> None:
        first = Utterance(
            corpus="LibriSpeech",
            audio_path=Path("a.flac"),
            speaker_id="6078",
            chapter_id="54013",
            utterance_id="0037",
        )
        second = Utterance(
            corpus="LibriSpeech",
            audio_path=Path("b.flac"),
            speaker_id="1234",
            chapter_id="56789",
            utterance_id="0037",
        )
        self.assertEqual(utterance_key(first), "6078-54013-0037")
        self.assertNotEqual(utterance_key(first), utterance_key(second))
        self.assertNotEqual(utterance_key(first), "0037")


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
                noise_pipelines=[()],
                sampling_rate=SAMPLE_RATE,
                segment_size=32,
                normalize="nope",
                crop=True,
            )


class CropOrPadTest(unittest.TestCase):
    def test_crop_pad_and_identity_length(self) -> None:
        rng = random.Random(0)
        cropped_clean, cropped_noisy, crop_start, crop_end = _crop_or_pad(
            torch.arange(10.0), torch.arange(10.0) + 1, 6, rng
        )
        expected_start = random.Random(0).randint(0, 4)
        self.assertEqual(tuple(cropped_clean.shape), (6,))
        self.assertEqual(tuple(cropped_noisy.shape), (6,))
        self.assertEqual(crop_start, expected_start)
        self.assertEqual(crop_end, expected_start + 6)
        self.assertTrue(torch.equal(cropped_clean, torch.arange(10.0)[crop_start:crop_end]))
        padded_clean, padded_noisy, pad_start, pad_end = _crop_or_pad(
            torch.arange(4.0), torch.arange(4.0) + 1, 6, random.Random(0)
        )
        self.assertEqual(tuple(padded_clean.shape), (6,))
        self.assertTrue(torch.equal(padded_clean[:4], torch.arange(4.0)))
        self.assertTrue(torch.equal(padded_noisy[4:], torch.zeros(2)))
        self.assertEqual(pad_start, 0)
        self.assertEqual(pad_end, 4)
        equal_clean, equal_noisy, equal_start, equal_end = _crop_or_pad(
            torch.arange(6.0), torch.arange(6.0) + 1, 6, random.Random(0)
        )
        self.assertTrue(torch.equal(equal_clean, torch.arange(6.0)))
        self.assertTrue(torch.equal(equal_noisy, torch.arange(6.0) + 1))
        self.assertEqual(equal_start, 0)
        self.assertEqual(equal_end, 6)


class PadCollateTest(unittest.TestCase):
    def test_pads_to_max_length_and_keeps_crop_metadata(self) -> None:
        batch = [
            AudioPair(torch.arange(3.0), torch.arange(3.0) + 10, "1234-56789-0000", 0, 3),
            AudioPair(torch.arange(5.0), torch.arange(5.0) + 10, "1234-56789-0001", 2, 7),
        ]
        clean, noisy, lengths, keys, crop_starts, crop_ends = pad_collate(batch)
        self.assertEqual(tuple(clean.shape), (2, 5))
        self.assertEqual(tuple(noisy.shape), (2, 5))
        self.assertTrue(torch.equal(lengths, torch.tensor([3, 5])))
        self.assertTrue(torch.equal(clean[0, 3:], torch.zeros(2)))
        self.assertEqual(keys, ("1234-56789-0000", "1234-56789-0001"))
        self.assertTrue(torch.equal(crop_starts, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(crop_ends, torch.tensor([3, 7])))


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
            configs = fetch_noise_configs(connection, "train", None)
            noises_by_id = {int(row.id): row for row in fetch_noises(connection) if row.id is not None}
        defaults: dict[str, object] = {
            "utterances": utts,
            "noise_pipelines": [parse_pipeline(config, noises_by_id) for config in configs],
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
        sample = dataset[0]
        clean, noisy = sample.clean, sample.noisy
        self.assertEqual(tuple(clean.shape), (N_SAMPLES,))
        self.assertEqual(tuple(noisy.shape), (N_SAMPLES,))
        utterance = dataset.utterances[0]
        self.assertEqual(sample.utterance_key, utterance_key(utterance))
        self.assertEqual(sample.utterance_key, "1234-56789-0000")
        self.assertNotEqual(sample.utterance_key, utterance.utterance_id)
        self.assertEqual(sample.crop_start, 0)
        self.assertEqual(sample.crop_end, N_SAMPLES)
        clean_2d = read_audio_segment(utterance.audio_path, 0, int(utterance.frames))
        noise_2d = generate(clean_2d, SAMPLE_RATE, dataset.noise_pipelines[0])
        raw_clean = torch.from_numpy(np.ascontiguousarray(clean_2d[:, 0]))
        raw_noisy = torch.from_numpy(np.ascontiguousarray((clean_2d + noise_2d)[:, 0]))
        expected_clean, expected_noisy = _normalize_pair(raw_clean, raw_noisy, "energy")
        self.assertTrue(torch.allclose(clean, expected_clean, atol=1e-5))
        self.assertTrue(torch.allclose(noisy, expected_noisy, atol=1e-5))
        self.assertTrue(torch.allclose(noisy.pow(2).mean(), torch.tensor(1.0), atol=1e-5))

    def test_seed_makes_getitem_deterministic(self) -> None:
        first = self._dataset(seed=42)[0]
        second = self._dataset(seed=42)[0]
        self.assertTrue(torch.equal(first.clean, second.clean))
        self.assertTrue(torch.equal(first.noisy, second.noisy))
        self.assertEqual(first.utterance_key, second.utterance_key)
        self.assertEqual(first.crop_start, second.crop_start)
        self.assertEqual(first.crop_end, second.crop_end)

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

    def test_random_crop_uses_source_sample_coordinates(self) -> None:
        long_frames = 8000
        path = self.corpora / "LibriSpeech" / "train-clean-100" / "1234" / "56789" / "1234-56789-0099.flac"
        write_audio(path, _tone(long_frames, 220.0), SAMPLE_RATE)
        utterance = Utterance(
            corpus="LibriSpeech",
            subset="train-clean-100",
            speaker_id="1234",
            chapter_id="56789",
            utterance_id="0099",
            audio_path=path,
            sample_rate=SAMPLE_RATE,
            frames=long_frames,
            channels=1,
            text="long",
        )
        first = self._dataset(utterances=[utterance], crop=True, segment_size=1600, seed=1)[0]
        second = self._dataset(utterances=[utterance], crop=True, segment_size=1600, seed=2)[0]
        self.assertEqual(first.utterance_key, "1234-56789-0099")
        self.assertEqual(first.utterance_key, second.utterance_key)
        self.assertEqual(tuple(first.clean.shape), (1600,))
        self.assertEqual(first.crop_end, first.crop_start + 1600)
        self.assertEqual(second.crop_end, second.crop_start + 1600)
        self.assertNotEqual(first.crop_start, second.crop_start)

    def test_short_utterance_padding_crop_end_is_real_audio(self) -> None:
        sample = self._dataset(crop=True, segment_size=48000, seed=0)[0]
        self.assertEqual(tuple(sample.clean.shape), (48000,))
        self.assertEqual(sample.crop_start, 0)
        self.assertEqual(sample.crop_end, N_SAMPLES)
        self.assertTrue(torch.equal(sample.clean[N_SAMPLES:], torch.zeros(48000 - N_SAMPLES)))

    def test_default_collate_keeps_crop_metadata(self) -> None:
        from torch.utils.data._utils.collate import default_collate

        sample = self._dataset(crop=True, segment_size=800, seed=0)[0]
        batched = default_collate([sample])
        self.assertEqual(list(batched.utterance_key), ["1234-56789-0000"])
        self.assertEqual(int(batched.crop_start[0]), sample.crop_start)
        self.assertEqual(int(batched.crop_end[0]), sample.crop_end)
