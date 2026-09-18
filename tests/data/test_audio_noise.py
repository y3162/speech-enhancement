import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.audio import read_audio_segment, read_audio_stream_info
from src.data.db import Connection, fetch_noises, import_rows
from src.data.noise import generate
from src.data.schema import NOISES_TABLE, Noise, NoiseConfig
from tests.helpers import speech_env, write_audio

SAMPLE_RATE = 16000


def _ramp(frames: int, channels: int = 1) -> np.ndarray:
    values = np.linspace(0.05, 0.9, frames, dtype=np.float32)
    return np.tile(values.reshape(-1, 1), (1, channels))


class AudioSegmentTest(unittest.TestCase):
    def test_reads_shape_and_wraps_inside_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            samples = (np.arange(8, dtype=np.float32) / 10.0).reshape(8, 1)
            write_audio(path, samples, SAMPLE_RATE)
            got = read_audio_segment(path, 0, 8)
            self.assertEqual(got.shape, (8, 1))
            wrapped = read_audio_segment(path, 6, 4)
            expected = np.concatenate([got[6:], got[:2]], axis=0)
            self.assertTrue(np.allclose(wrapped, expected, atol=1e-6))

    def test_empty_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_audio(path, _ramp(8), SAMPLE_RATE)
            with self.assertRaises(ValueError):
                read_audio_segment(path, 0, 4, 3, 3)


class StreamInfoTest(unittest.TestCase):
    def test_wav_and_flac_match_written_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            flac = Path(tmp) / "a.flac"
            samples = _ramp(32, channels=1)
            write_audio(wav, samples, SAMPLE_RATE)
            write_audio(flac, samples, SAMPLE_RATE)
            self.assertEqual(read_audio_stream_info(wav), (SAMPLE_RATE, 32, 1))
            self.assertEqual(read_audio_stream_info(flac), (SAMPLE_RATE, 32, 1))

    def test_unsupported_suffix_raises(self) -> None:
        with self.assertRaises(ValueError):
            read_audio_stream_info(Path("clip.mp3"))


class GenerateNoiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env = speech_env(Path(self.tmp.name))
        self.db_dir, self.corpora = self.env.__enter__()
        self.noise_path = self.corpora / "noise.wav"
        write_audio(self.noise_path, _ramp(64), SAMPLE_RATE)
        self.db_path = self.db_dir / "metadata.duckdb"
        import_rows(
            self.db_path,
            NOISES_TABLE,
            [
                Noise(
                    corpus="DEMAND",
                    subset="OMEETING",
                    noise_id="ch01",
                    audio_path=self.noise_path,
                    sample_rate=SAMPLE_RATE,
                    frames=64,
                    channels=1,
                )
            ],
        )
        with Connection(self.db_path) as connection:
            self.noise_id = int(fetch_noises(connection)[0].id)

    def tearDown(self) -> None:
        self.env.__exit__(None, None, None)
        self.tmp.cleanup()

    def _config(self, seed: int, snr_db: float) -> NoiseConfig:
        return NoiseConfig(
            json={
                "version": "1.0",
                "pipeline": [
                    {
                        "method": "additive",
                        "params": {
                            "seed": seed,
                            "noise_id": str(self.noise_id),
                            "target_range": {"type": "all"},
                            "noise_valid_range": {"start_ratio": 0.0, "end_ratio": 1.0},
                            "snr_db": snr_db,
                        },
                    }
                ],
            },
            split="train",
        )

    def test_generate_matches_snr_formula_and_is_seed_deterministic(self) -> None:
        clean = _ramp(40)
        config = self._config(seed=3, snr_db=5.0)
        with Connection(self.db_path, read_only=True) as connection:
            noise = generate(clean, SAMPLE_RATE, config, connection)
            again = generate(clean, SAMPLE_RATE, config, connection)
        self.assertEqual(noise.shape, clean.shape)
        self.assertEqual(noise.dtype, np.float32)
        self.assertTrue(np.array_equal(noise, again))

        rng = np.random.default_rng(3)
        start = int(rng.integers(0, 64))
        raw = read_audio_segment(self.noise_path, start, clean.shape[0], 0, 64)
        clean_power = np.mean(np.square(clean.astype(np.float64)))
        noise_power = np.mean(np.square(raw.astype(np.float64)))
        scale = np.sqrt(clean_power / (noise_power * (10 ** (5.0 / 10.0))))
        expected = raw.astype(np.float64) * scale
        self.assertTrue(np.allclose(noise, expected.astype(np.float32), atol=1e-6))

        mixed = clean + noise
        self.assertFalse(np.allclose(mixed, noise))
        self.assertFalse(np.allclose(mixed, clean))

    def test_unsupported_version_raises(self) -> None:
        config = NoiseConfig(json={"version": "2.0", "pipeline": []}, split="train")
        with Connection(self.db_path, read_only=True) as connection:
            with self.assertRaises(ValueError):
                generate(_ramp(8), SAMPLE_RATE, config, connection)
