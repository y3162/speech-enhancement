from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ...audio.io import read_audio_segment
from ...database.connection import Connection
from ...database.corpora.dto import NOISES_TABLE, Noise
from ...database.schema import Query


@dataclass(frozen=True)
class AdditiveStep:
    seed: int
    noise_id: int
    snr_db: float
    start_ratio: float
    end_ratio: float


@dataclass(frozen=True)
class Pipeline:
    steps: tuple[AdditiveStep, ...]


def parse(data: dict[str, Any]) -> Pipeline:
    steps = tuple(_parse_step(step) for step in data["pipeline"])
    return Pipeline(steps=steps)


def generate(
    clean: np.ndarray,
    sample_rate: int,
    pipeline: Pipeline,
    connection: Connection,
) -> np.ndarray:
    out = np.zeros(clean.shape, dtype=np.float64)
    for step in pipeline.steps:
        out += _generate_additive(clean, sample_rate, step, connection)
    return out.astype(np.float32)


def _parse_step(step: dict[str, Any]) -> AdditiveStep:
    method = step["method"]
    if method != "additive":
        raise ValueError(f"Unsupported method: {method!r}")

    params = step["params"]
    target_range = params["target_range"]
    range_type = target_range["type"]
    if range_type != "all":
        raise ValueError(f"Unsupported target_range type: {range_type!r}")

    start_ratio, end_ratio = _parse_noise_valid_range(params["noise_valid_range"])
    return AdditiveStep(
        seed=_parse_seed(params["seed"]),
        noise_id=_parse_noise_id(params["noise_id"]),
        snr_db=_parse_snr_db(params["snr_db"]),
        start_ratio=start_ratio,
        end_ratio=end_ratio,
    )


def _parse_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid seed: {value!r}")
    return value


def _parse_noise_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid noise_id: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Invalid noise_id: {value!r}")


def _parse_snr_db(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid snr_db: {value!r}")
    return float(value)


def _parse_ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def _parse_noise_valid_range(value: dict[str, Any]) -> tuple[float, float]:
    start_ratio = _parse_ratio(value["start_ratio"], "start_ratio")
    end_ratio = _parse_ratio(value["end_ratio"], "end_ratio")
    if not 0.0 <= start_ratio < end_ratio <= 1.0:
        raise ValueError(
            f"Invalid noise_valid_range: start_ratio={start_ratio}, "
            f"end_ratio={end_ratio}"
        )
    return start_ratio, end_ratio


def _generate_additive(
    clean: np.ndarray,
    sample_rate: int,
    step: AdditiveStep,
    connection: Connection,
) -> np.ndarray:
    noise_row = _fetch_noise(connection, step.noise_id)
    if noise_row.sample_rate != sample_rate:
        raise ValueError(
            f"Noise id {step.noise_id} sample_rate {noise_row.sample_rate} "
            f"does not match {sample_rate}"
        )
    if noise_row.frames is None or noise_row.frames < 1:
        raise ValueError(
            f"Noise id {step.noise_id} has invalid frames: {noise_row.frames}"
        )

    range_start = math.floor(step.start_ratio * noise_row.frames)
    range_end = math.floor(step.end_ratio * noise_row.frames)
    range_len = range_end - range_start
    if range_len < 1:
        raise ValueError(
            f"Noise id {step.noise_id} has empty valid range "
            f"[{range_start}, {range_end})"
        )

    rng = np.random.default_rng(step.seed)
    start_frame = int(rng.integers(0, range_len))
    noise = read_audio_segment(
        noise_row.audio_path,
        start_frame,
        clean.shape[0],
        range_start,
        range_end,
    )
    if noise.shape[1] != clean.shape[1]:
        raise ValueError(
            f"Noise id {step.noise_id} channels {noise.shape[1]} "
            f"do not match clean channels {clean.shape[1]}"
        )
    return _scale_to_snr(clean, noise, step.snr_db)


def _fetch_noise(connection: Connection, noise_id: int) -> Noise:
    rows = connection.fetch(
        Query(NOISES_TABLE).where("id = ?", noise_id).limit(1),
        Noise,
    )
    if not rows:
        raise ValueError(f"Noise id {noise_id} not found")
    return rows[0]


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
