import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.se.common.training import load_config, parse_args, to_dict, unpadded
from tests.helpers import se_config

CONFIGS = [
    se_config("mp_senet"),
    se_config("se_mamba"),
    se_config("se_mamba", "pcs.json"),
    se_config("se_mamba_pp"),
]


class ConfigFileTest(unittest.TestCase):
    def test_load_config_roundtrip(self) -> None:
        for path in CONFIGS:
            with self.subTest(path=path):
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(loaded, to_dict(load_config(path)))

    def test_configs_have_flat_train_section(self) -> None:
        for path in CONFIGS:
            with self.subTest(path=path):
                cfg = load_config(path)
                for key in ("batch_size", "epochs", "seed", "log_interval", "optim", "loss"):
                    self.assertTrue(hasattr(cfg.train, key), key)
                # Old nested train.env / top-level checkpoint_root must stay gone (resume writes flat train).
                self.assertFalse(hasattr(cfg.train, "env"))
                self.assertFalse(hasattr(cfg, "checkpoint_root"))
                self.assertIn("consistency", vars(cfg.train.loss))


class ParseArgsTest(unittest.TestCase):
    def test_new_run_uses_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            with mock.patch.object(sys, "argv", ["train", "--run_dir", str(run_dir)]):
                cfg, parsed_dir = parse_args(CONFIGS[0], "test")
            self.assertEqual(parsed_dir, run_dir)
            self.assertEqual(to_dict(cfg), json.loads(CONFIGS[0].read_text(encoding="utf-8")))

    def test_resume_reads_saved_config_and_rejects_config_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            saved = {"train": {"seed": 7}}
            (run_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["train", "--run_dir", str(run_dir)]):
                cfg, _ = parse_args(CONFIGS[0], "test")
            self.assertEqual(cfg.train.seed, 7)
            with mock.patch.object(sys, "argv", ["train", "--run_dir", str(run_dir), "--config", str(CONFIGS[0])]):
                with self.assertRaises(SystemExit):
                    parse_args(CONFIGS[0], "test")


class UnpaddedTest(unittest.TestCase):
    def test_trims_to_min_of_lengths_and_enhanced(self) -> None:
        clean = torch.arange(10.0).view(2, 5)
        enhanced = torch.arange(8.0).view(2, 4)
        clean_list, enhanced_list = unpadded(clean, enhanced, torch.tensor([3, 5]))
        self.assertEqual([len(row) for row in clean_list], [3, 4])
        self.assertEqual([len(row) for row in enhanced_list], [3, 4])
        self.assertTrue(np.array_equal(clean_list[0], np.array([0.0, 1.0, 2.0])))
