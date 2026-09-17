import os
from pathlib import Path


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        raise TypeError(f"{name} is not set")
    return Path(value)


SPEECH_UTILS_DB_ROOT_DIR = _required_env_path("SPEECH_UTILS_DB_ROOT_DIR")
SPEECH_UTILS_DB_METADATA_PATH = SPEECH_UTILS_DB_ROOT_DIR / "metadata.duckdb"

SPEECH_UTILS_CORPORA_ROOT_DIR = _required_env_path("SPEECH_UTILS_CORPORA_ROOT_DIR")
SPEECH_UTILS_CORPORA_LIBRISPEECH_DIR = SPEECH_UTILS_CORPORA_ROOT_DIR / "LibriSpeech"
SPEECH_UTILS_CORPORA_LIBRITTS_DIR = SPEECH_UTILS_CORPORA_ROOT_DIR / "LibriTTS"
SPEECH_UTILS_CORPORA_VCTK_DIR = SPEECH_UTILS_CORPORA_ROOT_DIR / "VCTK"
SPEECH_UTILS_CORPORA_DEMAND_DIR = SPEECH_UTILS_CORPORA_ROOT_DIR / "DEMAND"
