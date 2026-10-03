"""Deterministic augmentation from numpy SeedSequence. Python hash() is not used."""

import math
from dataclasses import dataclass

import numpy as np

from src.data.noise import AdditiveStep

TRAIN_SPLIT_CODE = 1
VALID_SPLIT_CODE = 2


@dataclass(frozen=True)
class NoiseMeta:
    noise_id: int
    snr_db: float
    noise_offset: int


@dataclass(frozen=True)
class AugmentationSpec:
    entropy: str
    variant: int
    pipeline_index: int
    crop_start: int
    crop_end: int
    noise_id: int
    noise_offset: int
    snr_db: float
    source_frames: int

    @property
    def content_samples(self) -> int:
        return self.crop_end - self.crop_start


def noise_start_frame(step: AdditiveStep) -> int:
    """Same offset as src.data.noise.generate: Generator(step.seed) over the valid range."""
    range_start = math.floor(step.start_ratio * step.frames)
    range_end = math.floor(step.end_ratio * step.frames)
    range_len = range_end - range_start
    if range_len < 1:
        raise ValueError(f"{step.audio_path} has empty valid range [{range_start}, {range_end})")
    rng = np.random.default_rng(step.seed)
    return int(rng.integers(0, range_len))


def variant_for_epoch(epoch: int, variants_per_utterance: int | None, crop: bool) -> int:
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if not crop:
        return 0
    if variants_per_utterance is None:
        return epoch
    if variants_per_utterance < 1:
        raise ValueError(f"variants_per_utterance must be >= 1, got {variants_per_utterance}")
    return epoch % variants_per_utterance


def augmentation_spec(
    base_seed: int,
    split_code: int,
    index: int,
    variant: int,
    n_pipelines: int,
    source_frames: int,
    segment_size: int,
    crop: bool,
    noise_meta: list[NoiseMeta],
) -> AugmentationSpec:
    """Replay one (utterance, variant) mix. Crop draws use inclusive bounds, matching random.randint."""
    if n_pipelines < 1:
        raise ValueError("n_pipelines must be >= 1")
    if len(noise_meta) != n_pipelines:
        raise ValueError(f"noise_meta length {len(noise_meta)} != n_pipelines {n_pipelines}")
    if source_frames < 1:
        raise ValueError(f"source_frames must be >= 1, got {source_frames}")
    if variant < 0 or index < 0:
        raise ValueError(f"index and variant must be >= 0, got {index}, {variant}")
    entropy = f"{int(base_seed)},{int(split_code)},{int(index)},{int(variant)}"
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(split_code), int(index), int(variant)]))
    pipeline_index = int(rng.integers(0, n_pipelines))
    if crop and source_frames > segment_size:
        crop_start = int(rng.integers(0, source_frames - segment_size + 1))
        crop_end = crop_start + segment_size
    else:
        crop_start = 0
        crop_end = source_frames
    meta = noise_meta[pipeline_index]
    return AugmentationSpec(
        entropy=entropy,
        variant=variant,
        pipeline_index=pipeline_index,
        crop_start=crop_start,
        crop_end=crop_end,
        noise_id=meta.noise_id,
        noise_offset=meta.noise_offset,
        snr_db=meta.snr_db,
        source_frames=source_frames,
    )
