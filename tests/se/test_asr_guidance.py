import unittest

import torch
import torch.nn as nn

from src.se.common.dense import DenseEncoder
from src.se.common.stft import stack_mag_pha
from src.se.se_mamba_pp.asr_guidance import (
    ASR_DIM,
    ASR_HOP,
    STFT_HOP,
    broadcast_frequency,
    fuse_asr_features,
    nearest_align,
    project_asr_features,
)


class NearestAlignTest(unittest.TestCase):
    def test_selects_nearest_center_100j_vs_1280i(self) -> None:
        n_stft, n_asr, dim = 20, 4, 3
        hidden = torch.arange(n_asr, dtype=torch.float32).view(1, n_asr, 1).expand(1, n_asr, dim).contiguous()
        aligned = nearest_align(hidden, n_stft)
        stft_centers = torch.arange(n_stft, dtype=torch.float32) * STFT_HOP
        asr_centers = torch.arange(n_asr, dtype=torch.float32) * ASR_HOP
        expected = (stft_centers[:, None] - asr_centers[None, :]).abs().argmin(dim=1)
        self.assertEqual(tuple(aligned.shape), (1, n_stft, dim))
        self.assertTrue(torch.equal(aligned[0, :, 0], expected.float()))
        self.assertEqual(int(expected[0]), 0)
        self.assertEqual(int(expected[6]), 0)
        self.assertEqual(int(expected[7]), 1)

    def test_asr_lengths_mask_invalid_frames(self) -> None:
        hidden = torch.tensor([[[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]])
        aligned = nearest_align(hidden, n_stft=5, asr_lengths=torch.tensor([1]))
        self.assertEqual(tuple(aligned.shape), (1, 5, 2))
        self.assertTrue(torch.equal(aligned[0], hidden[0, 0].expand(5, 2)))


class BroadcastAndFusionTest(unittest.TestCase):
    def test_frequency_broadcast_copies_along_freq(self) -> None:
        projected = torch.randn(2, 11, 128)
        broadcast = broadcast_frequency(projected, 100)
        self.assertEqual(tuple(broadcast.shape), (2, 128, 11, 100))
        self.assertTrue(torch.equal(broadcast[:, :, 3, 0], projected[:, 3, :]))
        self.assertTrue(torch.equal(broadcast[:, :, 3, 50], projected[:, 3, :]))

    def test_fusion_shapes_after_dense_encoder(self) -> None:
        mag, pha = torch.rand(2, 201, 17), torch.rand(2, 201, 17)
        se_feat = DenseEncoder(48)(stack_mag_pha(mag, pha))
        self.assertEqual(tuple(se_feat.shape), (2, 48, 17, 100))
        aligned = torch.randn(2, 17, 128)
        fused = fuse_asr_features(nn.Conv2d(176, 48, kernel_size=1), se_feat, aligned)
        self.assertEqual(tuple(fused.shape), (2, 48, 17, 100))


class ProjectionGradientTest(unittest.TestCase):
    def test_projection_shape_and_detached_feature_has_no_grad(self) -> None:
        hidden = torch.randn(2, ASR_DIM, 5, requires_grad=True)
        detached = hidden.detach()
        proj = nn.Linear(ASR_DIM, 128)
        fuse = nn.Conv2d(176, 48, kernel_size=1)
        se_feat = torch.randn(2, 48, 11, 7, requires_grad=True)
        aligned = project_asr_features(proj, detached, n_stft=11)
        self.assertEqual(tuple(proj(detached.transpose(1, 2)).shape), (2, 5, 128))
        self.assertEqual(tuple(aligned.shape), (2, 11, 128))
        fused = fuse_asr_features(fuse, se_feat, aligned)
        self.assertEqual(tuple(fused.shape), (2, 48, 11, 7))
        fused.sum().backward()
        self.assertIsNone(hidden.grad)
        self.assertIsNone(detached.grad)
        self.assertIsNotNone(proj.weight.grad)
        self.assertIsNotNone(fuse.weight.grad)
        self.assertTrue(torch.isfinite(proj.weight.grad).all())
        self.assertTrue(torch.isfinite(fuse.weight.grad).all())
        self.assertGreater(float(proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(fuse.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(se_feat.grad)
