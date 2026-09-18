import unittest

import torch

from src.se.common.pesq import pesq_batch_target, pesq_sum


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_module("pesq"), "pesq is not installed")
class PesqContractTest(unittest.TestCase):
    def test_failed_utterance_makes_batch_target_none(self) -> None:
        # Wideband PESQ rejects inputs that are too short.
        clean = torch.zeros(1, 16)
        enhanced = torch.zeros(1, 16)
        self.assertIsNone(pesq_batch_target(clean, enhanced, 16000))
        total, count = pesq_sum([clean[0].numpy()], [enhanced[0].numpy()], 16000, num_workers=1)
        self.assertEqual(total, 0.0)
        self.assertEqual(count, 0)
