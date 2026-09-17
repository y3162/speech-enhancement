import os
from collections.abc import Iterator
from pathlib import Path

from ..audio import read_audio_stream_info
from ..db import corpora_root_dir
from ..schema import Noise
from .scan import iter_in_threads


def noise_iterator() -> Iterator[Noise]:
    environment_dirs = []
    with os.scandir(corpora_root_dir() / "DEMAND") as entries:
        for entry in entries:
            if entry.is_dir():
                environment_dirs.append(Path(entry.path))
    return iter_in_threads(environment_dirs, parse_environment_dir)


"""
DEMAND
└── <environment>
    ├── ch01.wav
    ├── ch02.wav
    ...
    └── ch16.wav
"""


def parse_environment_dir(
    environment_dir: Path,
) -> list[Noise]:
    subset_name = environment_dir.name
    results = []
    with os.scandir(environment_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".wav"):
                continue
            audio_path = Path(entry.path)
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            results.append(
                Noise(
                    corpus="DEMAND",
                    subset=subset_name,
                    noise_id=audio_path.stem,
                    audio_path=audio_path,
                    sample_rate=sample_rate,
                    frames=frames,
                    channels=channels,
                )
            )
    return results
