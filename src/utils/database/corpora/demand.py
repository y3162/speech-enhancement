import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import (
    Iterator,
    List,
)

from ..config import SPEECH_UTILS_CORPORA_DEMAND_DIR
from .audio import read_audio_stream_info
from .dto import Noise


_IO_WORKERS = min(os.cpu_count() or 1, 64)


def noise_iterator() -> Iterator[Noise]:
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as executor:
        pending = set()
        paths = iter_environment_dirs(SPEECH_UTILS_CORPORA_DEMAND_DIR)
        listing_done = False
        in_flight_limit = _IO_WORKERS * 2
        while pending or not listing_done:
            while not listing_done and len(pending) < in_flight_limit:
                environment_dir = next(paths, None)
                if environment_dir is None:
                    listing_done = True
                    break
                pending.add(executor.submit(parse_environment_dir, environment_dir))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield from future.result()


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
) -> List[Noise]:
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
