"""CPU checks for the error-aware SEMamba++ schedule, masks, loss, and alignment cache."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from src.data.db import Connection, fetch_noises, import_rows
from src.data.schema import (
    NOISE_CONFIGS_TABLE,
    NOISES_TABLE,
    UTTERANCES_TABLE,
    Noise,
    NoiseConfig,
    Utterance,
)
from src.se.common.stft import Spec
from src.se.common.training import load_config, seed_everything
from src.se.error_aware_se_mamba_pp.alignment_cache import (
    AlignmentRow,
    AlignmentStore,
    AlignmentWriter,
    fingerprint_mismatches,
    merge_shard_databases,
    state_dict_sha256,
)
from src.se.error_aware_se_mamba_pp.dataset import ScheduledNoiseDataset, datasets_from_cfg
from src.se.error_aware_se_mamba_pp.discriminator import DiscriminatorOutputs
from src.se.error_aware_se_mamba_pp.error_mask import (
    align_token_ids,
    alignment_counts,
    boundary_sd_counts,
    error_frames_from_alignment,
    fill_sample_error,
    sample_intervals_from_frames,
    stft_frame_count,
)
from src.se.error_aware_se_mamba_pp.loss import (
    MultiScaleMelSpectrogramLoss,
    error_aware_generator_loss,
    frame_weight_and_valid,
    generator_loss,
    phase_maps_match_reference,
    weighted_mean,
)
from src.se.error_aware_se_mamba_pp.schedule import (
    TRAIN_SPLIT_CODE,
    NoiseMeta,
    augmentation_spec,
    variant_for_epoch,
)
from tests.helpers import se_config, write_audio

SAMPLE_RATE = 16000
N_SAMPLES = 1600


def _metas(count: int) -> list[NoiseMeta]:
    return [NoiseMeta(noise_id=index + 1, snr_db=float(index - 10), noise_offset=index * 3) for index in range(count)]


def _spec(variant: int, source_frames: int = 100_000, crop: bool = True) -> object:
    return augmentation_spec(42, TRAIN_SPLIT_CODE, 7, variant, 20, source_frames, 48_000, crop, _metas(20))


def _disc() -> DiscriminatorOutputs:
    zero = torch.zeros(())
    return DiscriminatorOutputs([zero], [zero], [[zero]], [[zero]], [zero], [zero], [[zero]], [[zero]])


def _weights() -> SimpleNamespace:
    return SimpleNamespace(
        magnitude=0.9,
        phase=0.3,
        complex=0.1,
        consistency=0.1,
        adv_g=1.0,
        fm_g=1.0,
        mel=0.1,
        tdt=1.0,
    )


def _fingerprint() -> dict[str, str]:
    return {"schema": "2", "schedule_version": "seed_sequence_v1"}


def _row(index: int, variant: int = 0) -> AlignmentRow:
    tokens = [{"token_id": 1, "token": "a", "start_offset": 0, "end_offset": 2}]
    alignment = [("S", 0, 0)]
    return AlignmentRow(
        split="train",
        utterance_index=index,
        variant=variant,
        utterance_key=f"key-{index}",
        crop_start=0,
        crop_end=4,
        source_frames=4,
        pipeline_index=0,
        noise_id=1,
        noise_offset=0,
        snr_db=0.0,
        content_samples=4,
        entropy=f"0,1,{index},{variant}",
        clean_encoded_length=2,
        noisy_encoded_length=2,
        clean_text="a",
        noisy_text="b",
        clean_tokens=tokens,
        noisy_tokens=[{"token_id": 2, "token": "b", "start_offset": 0, "end_offset": 2}],
        alignment=alignment,
        error_frames=error_frames_from_alignment(tokens, alignment),
    )


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


class ScheduleTest(unittest.TestCase):
    def test_seed_sequence_is_stable_across_processes(self) -> None:
        code = (
            "import numpy as np; "
            "rng=np.random.default_rng(np.random.SeedSequence([42,1,7,3])); "
            "print(int(rng.integers(0, 100000)))"
        )
        first = subprocess.check_output([sys.executable, "-c", code], text=True)
        second = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(first, second)
        rng = np.random.default_rng(np.random.SeedSequence([42, 1, 7, 3]))
        self.assertEqual(int(rng.integers(0, 100000)), int(first.strip()))

    def test_crop_bounds_and_repeatability(self) -> None:
        first = _spec(0)
        second = _spec(0)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.crop_start, 0)
        self.assertLessEqual(first.crop_start, 100_000 - 48_000)
        self.assertEqual(first.crop_end, first.crop_start + 48_000)
        varied = {_spec(variant).pipeline_index for variant in range(8)}
        varied.update(_spec(variant).crop_start for variant in range(8))
        self.assertGreater(len(varied), 1)

    def test_short_and_validation_do_not_crop(self) -> None:
        short = _spec(1, source_frames=1000, crop=True)
        self.assertEqual((short.crop_start, short.crop_end), (0, 1000))
        valid = _spec(3, source_frames=80_000, crop=False)
        self.assertEqual((valid.crop_start, valid.crop_end), (0, 80_000))
        self.assertEqual(variant_for_epoch(5, 4, crop=True), 1)
        self.assertEqual(variant_for_epoch(5, None, crop=True), 5)
        self.assertEqual(variant_for_epoch(5, 4, crop=False), 0)

    def test_alignment_tie_break(self) -> None:
        ops = align_token_ids([1, 2], [2, 1])
        self.assertEqual([op for op, _i, _j in ops], ["S", "S"])
        self.assertEqual(alignment_counts(ops), {"C": 0, "S": 2, "D": 0, "I": 0})


class MaskTest(unittest.TestCase):
    def test_overlapping_windows_are_a_union(self) -> None:
        tokens = [
            {"token_id": 1, "token": "a", "start_offset": 0, "end_offset": 5},
            {"token_id": 2, "token": "b", "start_offset": 3, "end_offset": 7},
        ]
        frames = error_frames_from_alignment(tokens, [("S", 0, 0), ("S", 1, 1)])
        self.assertEqual(frames, list(range(7)))
        numeric = [0] * 10
        for start, end in sample_intervals_from_frames(frames, 10, hop=1):
            for index in range(start, end):
                numeric[index] += 1
        self.assertEqual(max(numeric), 1)
        self.assertTrue(all(value <= 1 for value in fill_sample_error(10, [(0, 5), (3, 7)])))

    def test_deletion_uses_clean_token_window(self) -> None:
        tokens = [{"token_id": 9, "token": "x", "start_offset": 2, "end_offset": 5}]
        self.assertEqual(error_frames_from_alignment(tokens, [("D", 0, None)]), [2, 3, 4])

    def test_insertion_is_not_a_reference_window(self) -> None:
        tokens = [{"token_id": 9, "token": "x", "start_offset": 0, "end_offset": 3}]
        self.assertEqual(error_frames_from_alignment(tokens, [("I", None, 0), ("C", 0, 1)]), [])

    def test_windows_clip_to_the_crop(self) -> None:
        self.assertEqual(sample_intervals_from_frames([0, 5], n_samples=100, hop=1280), [(0, 100)])

    def test_center_rule_does_not_dilate(self) -> None:
        sample_error = torch.zeros(1, 1000, dtype=torch.bool)
        sample_error[0, 100] = True
        weight, valid = frame_weight_and_valid(
            sample_error, torch.tensor([1000]), 11, 100, alpha=1.0, dtype=torch.float32
        )
        mask = weight[0] - 1
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(int(mask[1]), 1)
        self.assertEqual(int(valid.sum()), stft_frame_count(1000, 100))

    def test_stft_frame_count_matches_torch(self) -> None:
        for n_samples, n_fft, hop in ((48000, 400, 100), (17040, 400, 100), (4000, 400, 100)):
            spec = torch.stft(
                torch.zeros(1, n_samples),
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=torch.hann_window(n_fft),
                center=True,
                return_complex=True,
            )
            self.assertEqual(spec.size(-1), stft_frame_count(n_samples, hop))

    def test_boundary_count_includes_first_and_last_frame(self) -> None:
        tokens = [
            {"token_id": 1, "token": "a", "start_offset": 0, "end_offset": 1},
            {"token_id": 2, "token": "b", "start_offset": 4, "end_offset": 6},
        ]
        touched, total = boundary_sd_counts(tokens, [("S", 0, 0), ("D", 1, None)], encoded_length=6)
        self.assertEqual((touched, total), (2, 2))


class LossTest(unittest.TestCase):
    def test_zero_errors_match_the_unweighted_mean(self) -> None:
        values = torch.randn(2, 3, 5)
        weight = torch.ones(2, 5)
        valid = torch.ones(2, 5, dtype=torch.bool)
        self.assertTrue(torch.allclose(weighted_mean(values, weight, valid, 2), values.mean()))

    def test_constant_error_weight_does_not_scale_the_mean(self) -> None:
        values = torch.ones(2, 3, 5)
        weight = torch.full((2, 5), 5.0)
        valid = torch.ones(2, 5, dtype=torch.bool)
        loss = weighted_mean(values, weight, valid, 2)
        self.assertTrue(torch.allclose(loss, values.mean()))
        self.assertFalse(torch.allclose(loss, values.mean() * 5))

    def test_padding_is_excluded_from_both_sums(self) -> None:
        values = torch.ones(1, 1, 6)
        values[0, 0, 4:] = 1000
        weight = torch.ones(1, 6)
        valid = torch.tensor([[True, True, True, True, False, False]])
        loss = weighted_mean(values, weight, valid, 2)
        self.assertTrue(torch.allclose(loss, torch.ones(())))
        self.assertFalse(torch.isnan(loss))

    def test_zero_valid_frames_are_finite(self) -> None:
        values = torch.randn(1, 2, 4)
        weight, valid = frame_weight_and_valid(
            torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            4,
            10,
            1.0,
            torch.float32,
        )
        loss = weighted_mean(values, weight, valid, 2)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_alpha_zero_matches_generator_loss_and_gradients(self) -> None:
        torch.manual_seed(0)
        batch, freq, time, wave = 2, 5, 17, 1600
        clean_audio = torch.randn(batch, wave)
        enhanced = torch.randn(batch, wave, requires_grad=True)
        mag = torch.randn(batch, freq, time, requires_grad=True)
        pha = torch.randn(batch, freq, time, requires_grad=True)
        com = torch.randn(batch, freq, time, 2, requires_grad=True)
        hat_mag = torch.randn(batch, freq, time)
        hat_pha = torch.randn(batch, freq, time)
        hat_com = torch.randn(batch, freq, time, 2, requires_grad=True)
        clean = Spec(torch.randn_like(mag), torch.randn_like(pha), torch.randn_like(com))
        gen = Spec(mag, pha, com)
        gen_hat = Spec(hat_mag, hat_pha, hat_com)
        phase_maps_match_reference(clean.pha, gen.pha, 8)
        content = torch.full((batch,), wave)
        sample_error = torch.zeros(batch, wave, dtype=torch.bool)
        mel = MultiScaleMelSpectrogramLoss(16000)
        disc = _disc()
        tdt = torch.tensor(0.25)
        weights = _weights()
        reference = generator_loss(
            clean, gen, gen_hat, disc, mel(clean_audio.unsqueeze(1), enhanced.unsqueeze(1)), tdt, weights, 8
        )
        aware = error_aware_generator_loss(
            clean,
            gen,
            gen_hat,
            disc,
            tdt,
            weights,
            8,
            mel,
            clean_audio,
            enhanced,
            sample_error,
            content,
            100,
            0.0,
        )
        for key, value in reference.items():
            self.assertLess(float((value - aware[key]).abs().max()), 1e-5, key)
        params = [mag, pha, com, hat_com, enhanced]
        reference["total"].backward(retain_graph=True)
        ref_grads = [param.grad.detach().clone() for param in params]
        for param in params:
            param.grad = None
        aware["total"].backward()
        for ref_grad, param in zip(ref_grads, params):
            self.assertLess(float((ref_grad - param.grad).abs().max()), 1e-4)


class FingerprintTest(unittest.TestCase):
    def test_mismatch_lists_changed_keys(self) -> None:
        self.assertEqual(
            fingerprint_mismatches({"a": "1", "b": "2"}, {"a": "1", "b": "9"}),
            ["b: cache='2' current='9'"],
        )

    def test_store_does_not_open_on_init(self) -> None:
        store = AlignmentStore(Path("/tmp/does-not-need-to-exist.sqlite"), {})
        self.assertIsNone(store._connection)

    def test_initial_hashes_repeat(self) -> None:
        def linear_hash() -> str:
            seed_everything(42)
            return state_dict_sha256(nn.Linear(8, 8))

        self.assertEqual(linear_hash(), linear_hash())

    def test_roundtrip_and_shard_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "cache.shard0.sqlite"
            right = root / "cache.shard1.sqlite"
            merged = root / "cache.sqlite"
            for path, row in ((left, _row(0)), (right, _row(1))):
                writer = AlignmentWriter(path, _fingerprint(), rebuild=False)
                writer.insert(row)
                writer.close()
            count = merge_shard_databases(merged, 2, rebuild=False)
            self.assertEqual(count, 2)
            store = AlignmentStore(merged, _fingerprint())
            self.assertEqual(store.get("train", 1, 0).utterance_key, "key-1")
            store.close()
            with self.assertRaises(RuntimeError):
                AlignmentStore(merged, {"schema": "2", "schedule_version": "other"}).get("train", 0, 0)


class DatasetTest(unittest.TestCase):
    def test_materialize_repeats_the_same_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir, corpora = speech_pair(Path(tmp))
            with speech_env_at(db_dir, corpora):
                cfg = SimpleNamespace(
                    data=SimpleNamespace(
                        sampling_rate=SAMPLE_RATE,
                        segment_size=800,
                        normalize="peak",
                        train_splits=["train-clean-100"],
                        validation_splits=["dev-clean"],
                        noise_config_ids=None,
                        variants_per_utterance=2,
                    ),
                    train=SimpleNamespace(seed=42),
                )
                trainset, validset = datasets_from_cfg(cfg)
                first = trainset.materialize(0, 0)
                second = trainset.materialize(0, 0)
                self.assertTrue(torch.equal(first.clean, second.clean))
                self.assertTrue(torch.equal(first.noisy, second.noisy))
                self.assertEqual((first.crop_start, first.crop_end), (second.crop_start, second.crop_end))
                self.assertEqual(len(first.clean), 800)
                self.assertEqual(validset.variant_for_epoch(3), 0)
                self.assertEqual(trainset.variant_for_epoch(3), 1)
                self.assertIsInstance(trainset, ScheduledNoiseDataset)


class ConfigFileTest(unittest.TestCase):
    def test_default_and_baseline_configs(self) -> None:
        proposed = load_config(se_config("error_aware_se_mamba_pp"))
        baseline = load_config(se_config("error_aware_se_mamba_pp", "baseline.json"))
        self.assertTrue(proposed.error_aware.enabled)
        self.assertEqual(proposed.error_aware.alpha, 1.0)
        self.assertEqual(proposed.data.variants_per_utterance, 9)
        self.assertFalse(hasattr(proposed.data, "pcs400"))
        self.assertFalse(baseline.error_aware.enabled)
        self.assertFalse(hasattr(baseline.error_aware, "alpha"))
        self.assertEqual(baseline.data.variants_per_utterance, 9)
        for key in ("magnitude", "phase", "complex", "consistency", "adv_g", "fm_g", "mel", "tdt"):
            self.assertIn(key, vars(proposed.train.loss))


def _import_parse_args():
    try:
        from src.se.error_aware_se_mamba_pp.train import parse_args
    except Exception as exc:
        raise unittest.SkipTest(f"train entry is not importable here: {exc}") from exc
    return parse_args


class ParseArgsTest(unittest.TestCase):
    def test_new_run_requires_cache_only_when_enabled(self) -> None:
        parse_args = _import_parse_args()
        proposed = se_config("error_aware_se_mamba_pp")
        baseline = se_config("error_aware_se_mamba_pp", "baseline.json")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = str(Path(tmp) / "run")
            cache = str(Path(tmp) / "cache.sqlite")
            with mock.patch.object(sys, "argv", ["train", "--run_dir", run_dir, "--config", str(baseline)]):
                cfg, parsed, parsed_cache, max_steps = parse_args()
            self.assertFalse(cfg.error_aware.enabled)
            self.assertEqual(parsed, Path(run_dir))
            self.assertIsNone(parsed_cache)
            self.assertIsNone(max_steps)
            with mock.patch.object(
                sys, "argv", ["train", "--run_dir", run_dir, "--config", str(baseline), "--alignment-cache", cache]
            ):
                with self.assertRaises(SystemExit):
                    parse_args()
            with mock.patch.object(sys, "argv", ["train", "--run_dir", run_dir, "--config", str(proposed)]):
                with self.assertRaises(SystemExit):
                    parse_args()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "train",
                    "--run_dir",
                    run_dir,
                    "--config",
                    str(proposed),
                    "--alignment-cache",
                    cache,
                    "--max-steps",
                    "2",
                ],
            ):
                _cfg, _dir, parsed_cache, max_steps = parse_args()
            self.assertEqual(parsed_cache, Path(cache))
            self.assertEqual(max_steps, 2)

    def test_resume_rejects_config_and_max_steps(self) -> None:
        parse_args = _import_parse_args()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            saved = {
                "data": {"variants_per_utterance": 9},
                "error_aware": {"enabled": False},
            }
            (run_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["train", "--run_dir", str(run_dir)]):
                cfg, _, cache, max_steps = parse_args()
            self.assertFalse(cfg.error_aware.enabled)
            self.assertIsNone(cache)
            self.assertIsNone(max_steps)
            with mock.patch.object(
                sys, "argv", ["train", "--run_dir", str(run_dir), "--config", str(se_config("error_aware_se_mamba_pp"))]
            ):
                with self.assertRaises(SystemExit):
                    parse_args()
            with mock.patch.object(sys, "argv", ["train", "--run_dir", str(run_dir), "--max-steps", "2"]):
                with self.assertRaises(SystemExit):
                    parse_args()


def speech_pair(root: Path) -> tuple[Path, Path]:
    db_dir = root / "db"
    corpora = root / "corpora"
    db_dir.mkdir()
    corpora.mkdir()
    train_path = corpora / "LibriSpeech" / "train-clean-100" / "1234" / "56789" / "1234-56789-0000.flac"
    valid_path = corpora / "LibriSpeech" / "dev-clean" / "1234" / "56789" / "1234-56789-0001.flac"
    noise_path = corpora / "DEMAND" / "OMEETING" / "ch01.wav"
    write_audio(train_path, _tone(N_SAMPLES, 220.0), SAMPLE_RATE)
    write_audio(valid_path, _tone(N_SAMPLES, 330.0), SAMPLE_RATE)
    write_audio(noise_path, _tone(N_SAMPLES, 80.0), SAMPLE_RATE)
    db_path = db_dir / "metadata.duckdb"
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
        noise_id = int(fetch_noises(connection)[0].id)
    import_rows(
        db_path,
        NOISE_CONFIGS_TABLE,
        [_noise_json(noise_id, "train", 5.0), _noise_json(noise_id, "dev", 0.0)],
    )
    return db_dir, corpora


def speech_env_at(db_dir: Path, corpora: Path):
    return mock.patch.dict(
        __import__("os").environ,
        {"SPEECH_DB_ROOT_DIR": str(db_dir), "SPEECH_CORPORA_ROOT_DIR": str(corpora)},
    )
