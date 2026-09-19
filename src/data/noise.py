import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio import read_audio_segment
from .schema import Noise, NoiseConfig


@dataclass(frozen=True)
class AdditiveStep:
    seed: int
    snr_db: float
    start_ratio: float
    end_ratio: float
    audio_path: Path
    sample_rate: int
    frames: int


def parse_pipeline(
    config: NoiseConfig,
    noises_by_id: Mapping[int, Noise],
) -> tuple[AdditiveStep, ...]:
    version = config.json["version"]
    if version != "1.0":
        raise ValueError(f"Unsupported noise config version: {version!r}")
    steps = []
    for step in config.json["pipeline"]:
        method = step["method"]
        if method != "additive":
            raise ValueError(f"Unsupported method: {method!r}")
        params = step["params"]
        range_type = params["target_range"]["type"]
        if range_type != "all":
            raise ValueError(f"Unsupported target_range type: {range_type!r}")
        seed = params["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Invalid seed: {seed!r}")
        noise_id = params["noise_id"]
        if isinstance(noise_id, bool):
            raise ValueError(f"Invalid noise_id: {noise_id!r}")
        if isinstance(noise_id, str):
            noise_id = int(noise_id)
        elif not isinstance(noise_id, int):
            raise ValueError(f"Invalid noise_id: {noise_id!r}")
        snr_db = params["snr_db"]
        if isinstance(snr_db, bool) or not isinstance(snr_db, (int, float)):
            raise ValueError(f"Invalid snr_db: {snr_db!r}")
        valid_range = params["noise_valid_range"]
        start_ratio = valid_range["start_ratio"]
        end_ratio = valid_range["end_ratio"]
        if isinstance(start_ratio, bool) or not isinstance(start_ratio, (int, float)):
            raise ValueError(f"Invalid start_ratio: {start_ratio!r}")
        if isinstance(end_ratio, bool) or not isinstance(end_ratio, (int, float)):
            raise ValueError(f"Invalid end_ratio: {end_ratio!r}")
        start_ratio = float(start_ratio)
        end_ratio = float(end_ratio)
        if not 0.0 <= start_ratio < end_ratio <= 1.0:
            raise ValueError(f"Invalid noise_valid_range: start_ratio={start_ratio}, end_ratio={end_ratio}")
        noise = noises_by_id.get(noise_id)
        if noise is None:
            raise ValueError(f"Noise id {noise_id} not found")
        if noise.sample_rate is None or noise.frames is None:
            raise ValueError(f"Noise id {noise_id} is missing sample_rate or frames")
        steps.append(
            AdditiveStep(
                seed=seed,
                snr_db=float(snr_db),
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                audio_path=noise.audio_path,
                sample_rate=noise.sample_rate,
                frames=noise.frames,
            )
        )
    return tuple(steps)


def _native_frame_count(n_out: int, orig_sr: int, target_sr: int) -> int:
    if orig_sr == target_sr:
        return n_out
    return max(int(round(n_out * orig_sr / target_sr)), 1)


def _resample_to_length(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
    n_out: int,
) -> np.ndarray:
    if orig_sr < 1 or target_sr < 1:
        raise ValueError(f"invalid sample rates orig_sr={orig_sr} target_sr={target_sr}")
    if orig_sr == target_sr:
        if waveform.shape[0] != n_out:
            raise ValueError(f"waveform length {waveform.shape[0]} does not match {n_out}")
        return waveform
    gcd = math.gcd(orig_sr, target_sr)
    resampled = resample_poly(
        waveform.astype(np.float64, copy=False),
        target_sr // gcd,
        orig_sr // gcd,
        axis=0,
    )
    resampled = np.asarray(resampled, dtype=np.float32)
    if resampled.ndim == 1:
        resampled = resampled.reshape(-1, 1)
    if resampled.shape[0] == n_out:
        return resampled
    if resampled.shape[0] > n_out:
        return resampled[:n_out]
    pad = np.zeros((n_out - resampled.shape[0], resampled.shape[1]), dtype=np.float32)
    return np.concatenate([resampled, pad], axis=0)


def generate(
    clean: np.ndarray,
    sample_rate: int,
    steps: Sequence[AdditiveStep],
) -> np.ndarray:
    if clean.ndim != 2 or clean.shape[0] < 1 or clean.shape[1] < 1:
        raise ValueError(f"clean must be a non-empty 2D array (frames, channels), got shape {clean.shape}")
    if sample_rate < 1:
        raise ValueError(f"invalid sample_rate {sample_rate}")
    out = np.zeros(clean.shape, dtype=np.float64)
    for step in steps:
        if step.sample_rate < 1:
            raise ValueError(f"{step.audio_path} has invalid sample_rate {step.sample_rate}")
        if step.frames < 1:
            raise ValueError(f"{step.audio_path} has invalid frames: {step.frames}")
        range_start = math.floor(step.start_ratio * step.frames)
        range_end = math.floor(step.end_ratio * step.frames)
        range_len = range_end - range_start
        if range_len < 1:
            raise ValueError(f"{step.audio_path} has empty valid range [{range_start}, {range_end})")
        rng = np.random.default_rng(step.seed)
        start_frame = int(rng.integers(0, range_len))
        native_frames = _native_frame_count(clean.shape[0], step.sample_rate, sample_rate)
        noise = read_audio_segment(
            step.audio_path,
            start_frame,
            native_frames,
            range_start,
            range_end,
        )
        if noise.shape[1] != clean.shape[1]:
            raise ValueError(
                f"{step.audio_path} channels {noise.shape[1]} do not match clean channels {clean.shape[1]}"
            )
        noise = _resample_to_length(noise, step.sample_rate, sample_rate, clean.shape[0])
        out += _scale_to_snr(clean, noise, step.snr_db)
    return out.astype(np.float32)


def _scale_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    clean_power = np.mean(np.square(clean.astype(np.float64)))
    noise_power = np.mean(np.square(noise.astype(np.float64)))
    if noise_power == 0.0:
        raise ValueError("Noise has zero power")
    scale = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10.0))))
    return noise.astype(np.float64) * scale
