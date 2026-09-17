from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..schema import (
    Column,
    Row,
    Table,
)

UTTERANCES_TABLE = Table(
    name="utterances",
    columns=(
        Column(name="id", type=int, auto_increment=True, primary=True),
        Column(name="corpus", type=str, nullable=False),
        Column(name="subset", type=str, nullable=True),
        Column(name="speaker_id", type=str, nullable=True),
        Column(name="chapter_id", type=str, nullable=True),
        Column(name="utterance_id", type=str, nullable=True),
        Column(name="audio_path", type=Path, nullable=False, unique=True),
        Column(name="sample_rate", type=int, nullable=True),
        Column(name="frames", type=int, nullable=True),
        Column(name="channels", type=int, nullable=True),
        Column(name="text", type=str, nullable=True),
        Column(name="created_at", type=datetime, default="CURRENT_TIMESTAMP"),
        Column(name="updated_at", type=datetime, default="CURRENT_TIMESTAMP"),
    ),
)


@dataclass(frozen=True, kw_only=True)
class Utterance(Row):
    corpus: str
    audio_path: Path
    subset: str | None = None
    speaker_id: str | None = None
    chapter_id: str | None = None
    utterance_id: str | None = None
    sample_rate: int | None = None
    frames: int | None = None
    channels: int | None = None
    text: str | None = None


NOISES_TABLE = Table(
    name="noises",
    columns=(
        Column(name="id", type=int, auto_increment=True, primary=True),
        Column(name="corpus", type=str, nullable=False),
        Column(name="subset", type=str, nullable=True),
        Column(name="noise_id", type=str, nullable=True),
        Column(name="audio_path", type=Path, nullable=False, unique=True),
        Column(name="sample_rate", type=int, nullable=True),
        Column(name="frames", type=int, nullable=True),
        Column(name="channels", type=int, nullable=True),
        Column(name="created_at", type=datetime, default="CURRENT_TIMESTAMP"),
        Column(name="updated_at", type=datetime, default="CURRENT_TIMESTAMP"),
    ),
)


@dataclass(frozen=True, kw_only=True)
class Noise(Row):
    corpus: str
    audio_path: Path
    subset: str | None = None
    noise_id: str | None = None
    sample_rate: int | None = None
    frames: int | None = None
    channels: int | None = None
