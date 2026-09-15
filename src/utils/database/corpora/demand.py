import os
from pathlib import Path
from typing import Iterator

from ..constants import SPEECH_UTILS_CORPORA_DEMAND_DIR
from .audio import read_audio_stream_info
from .dto import Noise
from .scan import iter_in_threads


def noise_iterator() -> Iterator[Noise]:
    return iter_in_threads(
        iter_environment_dirs(SPEECH_UTILS_CORPORA_DEMAND_DIR),
        parse_environment_dir,
    )


"""
DEMAND
└── <environment>
    ├── ch01.wav
    ├── ch02.wav
    ...
    └── ch16.wav
"""


def iter_environment_dirs(root: Path) -> Iterator[Path]:
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir():
                yield Path(entry.path)


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
