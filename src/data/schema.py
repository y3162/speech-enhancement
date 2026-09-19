import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Table:
    name: str
    create_sql: str
    sequence: str
    insert_columns: tuple[str, ...]
    select_sql: str


UTTERANCES_TABLE = Table(
    name="utterances",
    sequence="utterances_id_seq",
    insert_columns=(
        "corpus",
        "subset",
        "speaker_id",
        "chapter_id",
        "utterance_id",
        "audio_path",
        "sample_rate",
        "frames",
        "channels",
        "text",
    ),
    select_sql=(
        "utterances.id, utterances.corpus, utterances.subset, utterances.speaker_id, "
        "utterances.chapter_id, utterances.utterance_id, utterances.audio_path, utterances.sample_rate, "
        "utterances.frames, utterances.channels, utterances.text, utterances.created_at"
    ),
    create_sql="""CREATE SEQUENCE IF NOT EXISTS utterances_id_seq START 1;
CREATE TABLE IF NOT EXISTS utterances (
    id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('utterances_id_seq'),
    corpus TEXT NOT NULL,
    subset TEXT,
    speaker_id TEXT,
    chapter_id TEXT,
    utterance_id TEXT,
    audio_path TEXT NOT NULL UNIQUE,
    sample_rate INTEGER,
    frames INTEGER,
    channels INTEGER,
    text TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);""",
)


@dataclass(frozen=True, kw_only=True)
class Utterance:
    corpus: str
    audio_path: Path
    id: int | None = None
    subset: str | None = None
    speaker_id: str | None = None
    chapter_id: str | None = None
    utterance_id: str | None = None
    sample_rate: int | None = None
    frames: int | None = None
    channels: int | None = None
    text: str | None = None
    created_at: datetime | None = None

    def insert_values(self) -> tuple[Any, ...]:
        return (
            self.corpus,
            self.subset,
            self.speaker_id,
            self.chapter_id,
            self.utterance_id,
            str(self.audio_path),
            self.sample_rate,
            self.frames,
            self.channels,
            self.text,
        )

    @classmethod
    def from_sql(cls, row: Sequence[Any]) -> "Utterance":
        (
            id_,
            corpus,
            subset,
            speaker_id,
            chapter_id,
            utterance_id,
            audio_path,
            sample_rate,
            frames,
            channels,
            text,
            created_at,
        ) = row
        return cls(
            id=id_,
            corpus=corpus,
            subset=subset,
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            utterance_id=utterance_id,
            audio_path=audio_path if isinstance(audio_path, Path) else Path(audio_path),
            sample_rate=sample_rate,
            frames=frames,
            channels=channels,
            text=text,
            created_at=created_at,
        )


NOISES_TABLE = Table(
    name="noises",
    sequence="noises_id_seq",
    insert_columns=("corpus", "subset", "noise_id", "audio_path", "sample_rate", "frames", "channels"),
    select_sql=(
        "noises.id, noises.corpus, noises.subset, noises.noise_id, noises.audio_path, "
        "noises.sample_rate, noises.frames, noises.channels, noises.created_at"
    ),
    create_sql="""CREATE SEQUENCE IF NOT EXISTS noises_id_seq START 1;
CREATE TABLE IF NOT EXISTS noises (
    id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('noises_id_seq'),
    corpus TEXT NOT NULL,
    subset TEXT,
    noise_id TEXT,
    audio_path TEXT NOT NULL UNIQUE,
    sample_rate INTEGER,
    frames INTEGER,
    channels INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);""",
)


@dataclass(frozen=True, kw_only=True)
class Noise:
    corpus: str
    audio_path: Path
    id: int | None = None
    subset: str | None = None
    noise_id: str | None = None
    sample_rate: int | None = None
    frames: int | None = None
    channels: int | None = None
    created_at: datetime | None = None

    def insert_values(self) -> tuple[Any, ...]:
        return (
            self.corpus,
            self.subset,
            self.noise_id,
            str(self.audio_path),
            self.sample_rate,
            self.frames,
            self.channels,
        )

    @classmethod
    def from_sql(cls, row: Sequence[Any]) -> "Noise":
        id_, corpus, subset, noise_id, audio_path, sample_rate, frames, channels, created_at = row
        return cls(
            id=id_,
            corpus=corpus,
            subset=subset,
            noise_id=noise_id,
            audio_path=audio_path if isinstance(audio_path, Path) else Path(audio_path),
            sample_rate=sample_rate,
            frames=frames,
            channels=channels,
            created_at=created_at,
        )


NOISE_CONFIGS_TABLE = Table(
    name="noise_configs",
    sequence="noise_configs_id_seq",
    insert_columns=("json", "split"),
    select_sql="noise_configs.id, noise_configs.json, noise_configs.split, noise_configs.created_at",
    create_sql="""CREATE SEQUENCE IF NOT EXISTS noise_configs_id_seq START 1;
CREATE TABLE IF NOT EXISTS noise_configs (
    id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('noise_configs_id_seq'),
    json TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);""",
)


@dataclass(frozen=True, kw_only=True)
class NoiseConfig:
    json: dict[str, Any]
    split: str
    id: int | None = None
    created_at: datetime | None = None

    def insert_values(self) -> tuple[Any, ...]:
        return (json.dumps(self.json, sort_keys=True), self.split)

    @classmethod
    def from_sql(cls, row: Sequence[Any]) -> "NoiseConfig":
        id_, payload, split, created_at = row
        return cls(
            id=id_,
            json=json.loads(payload) if isinstance(payload, str) else payload,
            split=split,
            created_at=created_at,
        )


ASR_TIMESTAMPS_TABLE = Table(
    name="asr_timestamps",
    sequence="asr_timestamps_id_seq",
    insert_columns=("utterance_key", "text", "samples_per_encoder_frame", "tokens"),
    select_sql=(
        "asr_timestamps.id, asr_timestamps.utterance_key, asr_timestamps.text, "
        "asr_timestamps.samples_per_encoder_frame, asr_timestamps.tokens, asr_timestamps.created_at"
    ),
    create_sql="""CREATE SEQUENCE IF NOT EXISTS asr_timestamps_id_seq START 1;
CREATE TABLE IF NOT EXISTS asr_timestamps (
    id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('asr_timestamps_id_seq'),
    utterance_key TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    samples_per_encoder_frame INTEGER NOT NULL,
    tokens JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);""",
)


@dataclass(frozen=True)
class AsrToken:
    token_id: int
    token: str
    start_offset: int
    end_offset: int

    def to_json(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "token": self.token,
            "start_frame": self.start_offset,
            "end_frame": self.end_offset,
        }

    @classmethod
    def from_json(cls, payload: object) -> "AsrToken":
        if not isinstance(payload, dict):
            raise ValueError("timestamp token must be an object")
        return cls(
            token_id=int(payload["token_id"]),
            token=str(payload["token"]),
            start_offset=int(payload["start_frame"]),
            end_offset=int(payload["end_frame"]),
        )


def _asr_tokens_from_payload(payload: object) -> list[AsrToken]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, list):
        raise ValueError("timestamp record is missing tokens")
    return [AsrToken.from_json(item) for item in payload]


@dataclass(frozen=True, kw_only=True)
class AsrTimestamp:
    utterance_key: str
    text: str
    samples_per_encoder_frame: int
    tokens: list[AsrToken]
    id: int | None = None
    created_at: datetime | None = None

    def insert_values(self) -> tuple[Any, ...]:
        return (
            self.utterance_key,
            self.text,
            self.samples_per_encoder_frame,
            json.dumps([token.to_json() for token in self.tokens], ensure_ascii=False),
        )

    @classmethod
    def from_sql(cls, row: Sequence[Any]) -> "AsrTimestamp":
        id_, utterance_key, text, samples_per_encoder_frame, tokens, created_at = row
        return cls(
            id=id_,
            utterance_key=utterance_key,
            text=text,
            samples_per_encoder_frame=int(samples_per_encoder_frame),
            tokens=_asr_tokens_from_payload(tokens),
            created_at=created_at,
        )
