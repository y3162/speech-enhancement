import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]


def se_config(model: str, name: str = "default.json") -> Path:
    return REPO_ROOT / "src" / "se" / model / "configs" / name


@contextmanager
def speech_env(tmp: Path) -> Iterator[tuple[Path, Path]]:
    db_dir = tmp / "db"
    corpora_dir = tmp / "corpora"
    db_dir.mkdir(parents=True, exist_ok=True)
    corpora_dir.mkdir(parents=True, exist_ok=True)
    with mock.patch.dict(
        os.environ,
        {
            "SPEECH_DB_ROOT_DIR": str(db_dir),
            "SPEECH_CORPORA_ROOT_DIR": str(corpora_dir),
        },
    ):
        yield db_dir, corpora_dir


def write_audio(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)
