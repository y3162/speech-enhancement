import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.audio import read_audio_segment, read_audio_stream_info
from src.data.noise import AdditiveStep, generate, parse_pipeline
from src.data.schema import NoiseConfig
from tests.helpers import write_audio

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
        self.noise_path = Path(self.tmp.name) / "noise.wav"
        write_audio(self.noise_path, _ramp(64), SAMPLE_RATE)
        self.step = AdditiveStep(
            seed=3,
            snr_db=5.0,
            start_ratio=0.0,
            end_ratio=1.0,
            audio_path=self.noise_path,
            sample_rate=SAMPLE_RATE,
            frames=64,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_generate_matches_snr_formula_and_is_seed_deterministic(self) -> None:
        clean = _ramp(40)
        noise = generate(clean, SAMPLE_RATE, (self.step,))
        again = generate(clean, SAMPLE_RATE, (self.step,))
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
        with self.assertRaises(ValueError):
            parse_pipeline(config, {})
