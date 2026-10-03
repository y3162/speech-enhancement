"""LibriSpeech mix that replays SeedSequence crops."""

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data.audio import read_audio_segment
from src.data.corpora.librispeech import utterance_key
from src.data.db import Connection, fetch_noise_configs, fetch_noises, fetch_utterances, metadata_path
from src.data.noise import generate, parse_pipeline
from src.data.schema import Utterance
from src.se.common.dataset import (
    AudioPair,
    _noise_split,
    _normalize_pair,
    _to_mono_waveform,
    pad_collate,
    worker_init_fn,
)
from src.se.error_aware_se_mamba_pp.schedule import (
    TRAIN_SPLIT_CODE,
    VALID_SPLIT_CODE,
    AugmentationSpec,
    NoiseMeta,
    augmentation_spec,
    noise_start_frame,
    variant_for_epoch,
)

_LIBRISPEECH_CORPUS = "LibriSpeech"


@dataclass(frozen=True)
class ScheduledSample:
    clean: torch.Tensor
    noisy: torch.Tensor
    utterance_key: str
    crop_start: int
    crop_end: int
    index: int


def scheduled_collate(
    batch: list[ScheduledSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = [
        AudioPair(sample.clean, sample.noisy, sample.utterance_key, sample.crop_start, sample.crop_end)
        for sample in batch
    ]
    clean, noisy, lengths, keys, starts, ends = pad_collate(pairs)
    indices = torch.tensor([sample.index for sample in batch], dtype=torch.long)
    return clean, noisy, lengths, keys, starts, ends, indices


def content_lengths(crop_starts: torch.Tensor, crop_ends: torch.Tensor) -> torch.Tensor:
    return crop_ends - crop_starts


class ScheduledNoiseDataset(Dataset):
    def __init__(
        self,
        utterances: list[Utterance],
        pipelines: list[tuple],
        noise_meta: list[NoiseMeta],
        sampling_rate: int,
        segment_size: int,
        normalize: str,
        crop: bool,
        split_code: int,
        base_seed: int,
        variants_per_utterance: int | None,
        max_frames: int | None = None,
    ) -> None:
        if not utterances:
            raise ValueError("utterances must be non-empty")
        if not pipelines:
            raise ValueError("noise pipelines must be non-empty")
        if len(noise_meta) != len(pipelines):
            raise ValueError("noise_meta and pipelines must be the same length")
        self.utterances = utterances
        self.pipelines = pipelines
        self.noise_meta = noise_meta
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.normalize = normalize
        self.crop = crop
        self.split_code = split_code
        self.base_seed = base_seed
        self.variants_per_utterance = variants_per_utterance
        self.max_frames = max_frames
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.utterances)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def source_frames(self, index: int) -> int:
        frames = self.utterances[index].frames
        if frames is None or frames < 1:
            raise ValueError(f"utterance index {index} has invalid frames")
        if self.max_frames is not None:
            return min(int(frames), int(self.max_frames))
        return int(frames)

    def variant_for_epoch(self, epoch: int) -> int:
        return variant_for_epoch(epoch, self.variants_per_utterance, self.crop)

    def spec_for(self, index: int, variant: int) -> AugmentationSpec:
        return augmentation_spec(
            self.base_seed,
            self.split_code,
            index,
            variant,
            len(self.pipelines),
            self.source_frames(index),
            self.segment_size,
            self.crop,
            self.noise_meta,
        )

    def materialize(self, index: int, variant: int) -> ScheduledSample:
        utterance = self.utterances[index]
        if utterance.sample_rate != self.sampling_rate:
            raise ValueError(
                f"{utterance.audio_path} sample_rate {utterance.sample_rate} does not match {self.sampling_rate}"
            )
        source_frames = self.source_frames(index)
        spec = self.spec_for(index, variant)
        clean_2d = read_audio_segment(utterance.audio_path, 0, source_frames)
        noise_2d = generate(clean_2d, int(utterance.sample_rate), self.pipelines[spec.pipeline_index])
        clean = torch.from_numpy(_to_mono_waveform(clean_2d, utterance.audio_path))
        noisy = torch.from_numpy(_to_mono_waveform(clean_2d + noise_2d, utterance.audio_path))
        if clean.size(0) != source_frames:
            raise RuntimeError(f"read {clean.size(0)} frames, expected {source_frames}")
        clean, noisy = _normalize_pair(clean, noisy, self.normalize)
        if self.crop and source_frames > self.segment_size:
            clean = clean[spec.crop_start : spec.crop_end]
            noisy = noisy[spec.crop_start : spec.crop_end]
        elif self.crop and source_frames < self.segment_size:
            pad = (0, self.segment_size - source_frames)
            clean = F.pad(clean, pad)
            noisy = F.pad(noisy, pad)
        return ScheduledSample(clean, noisy, utterance_key(utterance), spec.crop_start, spec.crop_end, index)

    def __getitem__(self, index: int) -> ScheduledSample:
        return self.materialize(index, self.variant_for_epoch(self.epoch))


def _load_split(
    subsets: list[str],
    noise_ids: list[int] | None,
) -> tuple[list[Utterance], list[tuple], list[NoiseMeta]]:
    with Connection(metadata_path(), read_only=True) as connection:
        noises_by_id = {int(noise.id): noise for noise in fetch_noises(connection) if noise.id is not None}
        utterances = fetch_utterances(connection, _LIBRISPEECH_CORPUS, subsets)
        configs = fetch_noise_configs(connection, _noise_split(subsets), noise_ids)
    if not utterances:
        raise ValueError("no LibriSpeech utterances for splits: " + ", ".join(subsets))
    if not configs:
        raise ValueError(f"no noise configs for splits {subsets}")
    pipelines = []
    metas = []
    for config in configs:
        steps = parse_pipeline(config, noises_by_id)
        if len(steps) != 1:
            raise ValueError(f"expected one additive step, got {len(steps)}")
        step = steps[0]
        params = config.json["pipeline"][0]["params"]
        pipelines.append(steps)
        metas.append(
            NoiseMeta(
                noise_id=int(params["noise_id"]),
                snr_db=float(step.snr_db),
                noise_offset=noise_start_frame(step),
            )
        )
    return utterances, pipelines, metas


def datasets_from_cfg(cfg: SimpleNamespace) -> tuple[ScheduledNoiseDataset, ScheduledNoiseDataset]:
    ids = cfg.data.noise_config_ids
    if ids is not None and not isinstance(ids, list):
        ids = list(ids)
    variants = cfg.data.variants_per_utterance
    return load_scheduled_datasets(
        int(cfg.data.sampling_rate),
        int(cfg.data.segment_size),
        str(cfg.data.normalize),
        list(cfg.data.train_splits),
        list(cfg.data.validation_splits),
        ids,
        int(cfg.train.seed),
        None if variants is None else int(variants),
    )


def load_scheduled_datasets(
    sampling_rate: int,
    segment_size: int,
    normalize: str,
    train_splits: list[str],
    valid_splits: list[str],
    noise_config_ids: list[int] | None,
    base_seed: int,
    variants_per_utterance: int | None,
) -> tuple[ScheduledNoiseDataset, ScheduledNoiseDataset]:
    train_utterances, train_pipelines, train_meta = _load_split(train_splits, noise_config_ids)
    valid_utterances, valid_pipelines, valid_meta = _load_split(valid_splits, noise_config_ids)
    frame_counts = [int(utt.frames) for utt in train_utterances if utt.frames]
    if not frame_counts:
        raise ValueError("train utterances are missing frames")
    max_frames = max(frame_counts)
    trainset = ScheduledNoiseDataset(
        train_utterances,
        train_pipelines,
        train_meta,
        sampling_rate,
        segment_size,
        normalize,
        crop=True,
        split_code=TRAIN_SPLIT_CODE,
        base_seed=base_seed,
        variants_per_utterance=variants_per_utterance,
    )
    validset = ScheduledNoiseDataset(
        valid_utterances,
        valid_pipelines,
        valid_meta,
        sampling_rate,
        segment_size,
        normalize,
        crop=False,
        split_code=VALID_SPLIT_CODE,
        base_seed=base_seed,
        variants_per_utterance=variants_per_utterance,
        max_frames=max_frames,
    )
    return trainset, validset


def noise_list_sha256(metas: list[NoiseMeta]) -> str:
    digest = hashlib.sha256()
    for meta in metas:
        digest.update(f"{meta.noise_id}:{meta.snr_db}:{meta.noise_offset}".encode())
        digest.update(b"\0")
    return digest.hexdigest()


def utterance_key_sha256(utterances: list[Utterance]) -> str:
    digest = hashlib.sha256()
    for utterance in utterances:
        digest.update(utterance_key(utterance).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def build_scheduled_loaders(
    trainset: ScheduledNoiseDataset,
    validset: ScheduledNoiseDataset,
    train_cfg: SimpleNamespace,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_kwargs: dict = {}
    if train_cfg.num_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = train_cfg.prefetch_factor
        train_kwargs["worker_init_fn"] = worker_init_fn
    train_loader = DataLoader(
        trainset,
        batch_size=train_cfg.batch_size,
        sampler=DistributedSampler(trainset, seed=seed, shuffle=True),
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=scheduled_collate,
        **train_kwargs,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=train_cfg.val_batch_size,
        sampler=DistributedSampler(validset, seed=seed, shuffle=False, drop_last=False),
        num_workers=0,
        pin_memory=True,
        collate_fn=scheduled_collate,
    )
    return train_loader, valid_loader
