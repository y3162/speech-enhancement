import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.audio import read_audio_segment
from src.data.db import Connection, fetch_noise_configs, fetch_utterances, metadata_path
from src.data.noise import generate
from src.data.schema import NoiseConfig, Utterance

_CONNECTIONS: dict[tuple[int, str], Connection] = {}
_LIBRISPEECH_CORPUS = "LibriSpeech"


def worker_init_fn(_worker_id: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _CONNECTIONS.clear()


def _connection(database_path: Path) -> Connection:
    key = (os.getpid(), str(database_path))
    connection = _CONNECTIONS.get(key)
    if connection is None:
        connection = Connection(database_path, read_only=True)
        _CONNECTIONS[key] = connection
    return connection


def _noise_split(subsets: list[str]) -> str:
    kinds = set()
    for name in subsets:
        if name.startswith("train"):
            kinds.add("train")
        elif name.startswith("dev"):
            kinds.add("dev")
        elif name.startswith("test"):
            kinds.add("test")
        else:
            raise ValueError(f"cannot map subset {name!r} to a noise split")
    if len(kinds) != 1:
        raise ValueError(f"utterance subsets must map to one noise split, got {subsets} -> {kinds}")
    return next(iter(kinds))


def _fetch_split(
    connection: Connection,
    splits: list[str],
    noise_config_ids: list[int] | None,
) -> tuple[list[Utterance], list[NoiseConfig]]:
    utterances = fetch_utterances(connection, _LIBRISPEECH_CORPUS, splits)
    if not utterances:
        raise ValueError("no LibriSpeech utterances found for splits: " + ", ".join(splits))
    noise_split = _noise_split(splits)
    noise_configs = fetch_noise_configs(connection, noise_split, noise_config_ids)
    if not noise_configs:
        raise ValueError(f"no noise_configs found for split={noise_split!r} ids={noise_config_ids}")
    return utterances, noise_configs


def _sample_seed(seed: int, index: int) -> int:
    return (int(seed) * 1_000_003 + int(index)) & 0xFFFFFFFFFFFFFFFF


def _to_mono_waveform(audio: np.ndarray, path: Path) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError(f"{path} has invalid shape {audio.shape}")
    if audio.shape[1] != 1:
        raise ValueError(f"{path} must be mono, got channels={audio.shape[1]}")
    return np.ascontiguousarray(audio[:, 0])


def _normalize_pair(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "energy":
        scale = torch.sqrt(noisy.numel() / noisy.pow(2).sum().clamp_min(1e-12))
        return clean * scale, noisy * scale
    if kind == "peak":
        scale = noisy.abs().amax() + 1e-9
        return clean / scale, noisy / scale
    raise ValueError(f"unsupported normalize {kind!r}; expected 'energy' or 'peak'")


def _crop_or_pad(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    segment_size: int,
    rng: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = clean.size(0)
    if length == segment_size:
        return clean, noisy
    if length > segment_size:
        start = rng.randint(0, length - segment_size)
        end = start + segment_size
        return clean[start:end], noisy[start:end]
    pad = (0, segment_size - length)
    return F.pad(clean, pad), F.pad(noisy, pad)


def pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_list, noisy_list = zip(*batch)
    lengths = torch.tensor([int(clean.size(0)) for clean in clean_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    clean_batch = clean_list[0].new_zeros((len(batch), max_len))
    noisy_batch = noisy_list[0].new_zeros((len(batch), max_len))
    for i, (clean, noisy) in enumerate(zip(clean_list, noisy_list)):
        clean_batch[i, : clean.size(0)] = clean
        noisy_batch[i, : noisy.size(0)] = noisy
    return clean_batch, noisy_batch, lengths


class AdditiveNoiseDataset(Dataset):
    def __init__(
        self,
        utterances: list[Utterance],
        noise_configs: list[NoiseConfig],
        database_path: Path,
        sampling_rate: int,
        segment_size: int,
        normalize: str,
        crop: bool,
        max_frames: int | None = None,
        seed: int | None = None,
        pcs400: bool = False,
    ) -> None:
        if not utterances:
            raise ValueError("utterances must be non-empty")
        if not noise_configs:
            raise ValueError("noise_configs must be non-empty")
        if normalize not in ("energy", "peak"):
            raise ValueError(f"unsupported normalize {normalize!r}; expected 'energy' or 'peak'")
        self.utterances = utterances
        self.noise_configs = noise_configs
        self.database_path = Path(database_path)
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.normalize = normalize
        self.crop = crop
        self.max_frames = max_frames
        self.seed = seed
        self.pcs400 = pcs400

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        utterance = self.utterances[index]
        if utterance.sample_rate is None or utterance.sample_rate != self.sampling_rate:
            raise ValueError(
                f"{utterance.audio_path} sample_rate {utterance.sample_rate} does not match {self.sampling_rate}"
            )
        if utterance.frames is None or utterance.frames < 1:
            raise ValueError(f"{utterance.audio_path} has invalid frames")

        num_frames = utterance.frames
        if self.max_frames is not None:
            num_frames = min(num_frames, self.max_frames)

        rng = random if self.seed is None else random.Random(_sample_seed(self.seed, index))
        noise_config = rng.choice(self.noise_configs)

        clean_2d = read_audio_segment(utterance.audio_path, 0, num_frames)
        noise_2d = generate(
            clean_2d,
            utterance.sample_rate,
            noise_config,
            _connection(self.database_path),
        )
        clean = torch.from_numpy(_to_mono_waveform(clean_2d, utterance.audio_path))
        noisy = torch.from_numpy(_to_mono_waveform(clean_2d + noise_2d, utterance.audio_path))
        clean, noisy = _normalize_pair(clean, noisy, self.normalize)
        if self.crop:
            clean, noisy = _crop_or_pad(clean, noisy, self.segment_size, rng)
        if self.pcs400:
            from src.se.common.pcs import cal_pcs

            clean = torch.from_numpy(np.asarray(cal_pcs(clean.numpy()), dtype=np.float32))
        return clean, noisy


def build_datasets(cfg: SimpleNamespace) -> tuple[AdditiveNoiseDataset, AdditiveNoiseDataset]:
    """LibriSpeech plus additive noise. PCS400 is applied to training clean speech only."""
    data = cfg.data
    if not data.train_splits:
        raise ValueError("data.train_splits is required")
    if not data.validation_splits:
        raise ValueError("data.validation_splits is required")

    database_path = metadata_path()
    with Connection(database_path, read_only=True) as connection:
        train_utterances, train_noises = _fetch_split(connection, data.train_splits, data.noise_config_ids)
        valid_utterances, valid_noises = _fetch_split(connection, data.validation_splits, data.noise_config_ids)

    frame_counts = [int(utt.frames) for utt in train_utterances if utt.frames]
    if not frame_counts:
        raise ValueError("train utterances are missing frames")
    max_frames = max(frame_counts)
    trainset = AdditiveNoiseDataset(
        utterances=train_utterances,
        noise_configs=train_noises,
        database_path=database_path,
        sampling_rate=data.sampling_rate,
        segment_size=data.segment_size,
        normalize=data.normalize,
        crop=True,
        pcs400=getattr(data, "pcs400", False),
    )
    validset = AdditiveNoiseDataset(
        utterances=valid_utterances,
        noise_configs=valid_noises,
        database_path=database_path,
        sampling_rate=data.sampling_rate,
        segment_size=data.segment_size,
        normalize=data.normalize,
        crop=False,
        max_frames=max_frames,
        seed=cfg.train.seed,
    )
    return trainset, validset
