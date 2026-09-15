from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


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
