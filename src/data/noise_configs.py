import argparse
import hashlib
from pathlib import Path

from .db import ensure_table, fetch_noises, metadata_path
from .schema import NOISE_CONFIGS_TABLE, Noise, NoiseConfig

SNR_MIN = -10
SNR_MAX = 10
SPLIT_RANGES = (
    ("train", 0.0, 0.1),
    ("train", 0.1, 0.2),
    ("train", 0.2, 0.3),
    ("train", 0.3, 0.4),
    ("train", 0.4, 0.5),
    ("train", 0.5, 0.6),
    ("train", 0.6, 0.7),
    ("train", 0.7, 0.8),
    ("dev", 0.8, 0.9),
    ("test", 0.9, 1.0),
)


def config_seed(
    audio_path: Path,
    snr_db: int,
    start_ratio: float,
    end_ratio: float,
) -> int:
    payload = f"{audio_path.as_posix()}:{snr_db}:{start_ratio}:{end_ratio}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_noise_config_json(
    noise_id: int,
    audio_path: Path,
    snr_db: int,
    start_ratio: float,
    end_ratio: float,
) -> dict:
    return {
        "version": "1.0",
        "pipeline": [
            {
                "method": "additive",
                "params": {
                    "seed": config_seed(
                        audio_path,
                        snr_db,
                        start_ratio,
                        end_ratio,
                    ),
                    "noise_id": str(noise_id),
                    "target_range": {"type": "all"},
                    "noise_valid_range": {
                        "start_ratio": start_ratio,
                        "end_ratio": end_ratio,
                    },
                    "snr_db": float(snr_db),
                },
            }
        ],
    }


def build_noise_configs(noises: list[Noise]) -> list[NoiseConfig]:
    selected = [noise for noise in noises if noise.corpus == "DEMAND" and noise.audio_path.name == "ch01.wav"]
    if not selected:
        raise ValueError("no DEMAND ch01.wav rows in noises")
    selected = sorted(selected, key=lambda noise: noise.audio_path.as_posix())

    configs: list[NoiseConfig] = []
    for noise in selected:
        if noise.id is None:
            raise ValueError(f"Noise is missing id: {noise}")
        for split, start_ratio, end_ratio in SPLIT_RANGES:
            for snr_db in range(SNR_MIN, SNR_MAX + 1):
                configs.append(
                    NoiseConfig(
                        json=build_noise_config_json(
                            noise.id,
                            noise.audio_path,
                            snr_db,
                            start_ratio,
                            end_ratio,
                        ),
                        split=split,
                    )
                )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    with ensure_table(metadata_path(), NOISE_CONFIGS_TABLE, replace=args.replace) as con:
        noises = fetch_noises(con)
        if not noises:
            raise ValueError("noises table is empty")
        con.insert(NOISE_CONFIGS_TABLE, build_noise_configs(noises))
        con.commit()
