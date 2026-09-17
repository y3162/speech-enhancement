import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

TYPE_MAP = {
    int: "INTEGER",
    str: "TEXT",
    bool: "BOOLEAN",
    Path: "TEXT",
    datetime: "TIMESTAMP",
    dict: "TEXT",
}

_RAW_SQL_DEFAULTS = {
    "CURRENT_TIMESTAMP",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "NULL",
}


@dataclass(frozen=True)
class Column:
    name: str
    type: type

    auto_increment: bool = False
    nullable: bool = False
    primary: bool = False
    unique: bool = False
    default: Any = None

    @property
    def sql_type(self) -> str:
        return TYPE_MAP[self.type]

    @property
    def insertable(self) -> bool:
        return not self.auto_increment and self.default is None

    def sequence_name(self, table_name: str) -> str:
        return f"{table_name}_{self.name}_seq"

    def create_sequence_sql(self, table_name: str) -> str | None:
        if not self.auto_increment:
            return None
        return f"CREATE SEQUENCE IF NOT EXISTS {self.sequence_name(table_name)} START 1;"

    def drop_sequence_sql(self, table_name: str) -> str | None:
        if not self.auto_increment:
            return None
        return f"DROP SEQUENCE IF EXISTS {self.sequence_name(table_name)};"

    def definition(self, table_name: str) -> str:
        parts = [self.name, self.sql_type]
        if self.primary:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.auto_increment:
            parts.append(f"DEFAULT nextval('{self.sequence_name(table_name)}')")
        elif self.default is not None:
            parts.append(f"DEFAULT {self.format_default()}")
        if self.unique:
            parts.append("UNIQUE")
        return " ".join(parts)

    def format_default(self) -> str:
        value = self.default
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(value)
        if isinstance(value, Path):
            return "'" + str(value).replace("'", "''") + "'"
        if isinstance(value, str):
            if value.upper() in _RAW_SQL_DEFAULTS:
                return value.upper()
            return "'" + value.replace("'", "''") + "'"
        raise TypeError(f"Unsupported default value for {self.name}: {value!r}")

    def to_sql_value(self, value: Any) -> Any:
        if value is None:
            if not self.nullable:
                raise ValueError(f"{self.name} is NOT NULL")
            return None
        if self.type is Path:
            return str(value)
        if self.type is dict:
            return json.dumps(value, sort_keys=True)
        return value

    def from_sql_value(self, value: Any) -> Any:
        if value is None:
            return None
        if self.type is Path:
            return value if isinstance(value, Path) else Path(value)
        if self.type is dict:
            if isinstance(value, str):
                return json.loads(value)
            return value
        return value


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    @property
    def insertable_columns(self) -> tuple[Column, ...]:
        return tuple(column for column in self.columns if column.insertable)

    @property
    def select_list(self) -> str:
        return ", ".join(f"{self.name}.{column.name}" for column in self.columns)

    def get_create_table_sql(self) -> str:
        statements = [sql for column in self.columns if (sql := column.create_sequence_sql(self.name)) is not None]
        definitions = [column.definition(self.name) for column in self.columns]
        body = ",\n".join(f"    {definition}" for definition in definitions)
        statements.append(f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n);")
        return "\n".join(statements)


RowT = TypeVar("RowT", bound="Row")


@dataclass(frozen=True, kw_only=True)
class Row:
    id: int | None = None
    created_at: datetime | None = None

    def value_for(self, column: Column) -> Any:
        if not hasattr(self, column.name):
            raise ValueError(f"{type(self).__name__} is missing column {column.name}")
        return column.to_sql_value(getattr(self, column.name))

    def insert_params(self, columns: tuple[Column, ...]) -> tuple[Any, ...]:
        return tuple(self.value_for(column) for column in columns)

    @classmethod
    def from_sql(
        cls: type[RowT],
        columns: tuple[Column, ...],
        values: Sequence[Any],
    ) -> RowT:
        kwargs = {column.name: column.from_sql_value(value) for column, value in zip(columns, values, strict=True)}
        return cls(**kwargs)


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


NOISE_CONFIGS_TABLE = Table(
    name="noise_configs",
    columns=(
        Column(name="id", type=int, auto_increment=True, primary=True),
        Column(name="json", type=dict, nullable=False, unique=True),
        Column(name="split", type=str, nullable=False),
        Column(name="created_at", type=datetime, default="CURRENT_TIMESTAMP"),
    ),
)


@dataclass(frozen=True, kw_only=True)
class NoiseConfig(Row):
    json: dict[str, Any]
    split: str
