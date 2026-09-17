import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import duckdb

from .schema import (
    NOISE_CONFIGS_TABLE,
    NOISES_TABLE,
    UTTERANCES_TABLE,
    Noise,
    NoiseConfig,
    Table,
    Utterance,
)

Record = Utterance | Noise | NoiseConfig


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return Path(value)


def metadata_path() -> Path:
    return _required_env_path("SPEECH_DB_ROOT_DIR") / "metadata.duckdb"


def corpora_root_dir() -> Path:
    return _required_env_path("SPEECH_CORPORA_ROOT_DIR")


class Connection:
    def __init__(self, database_path: Path, read_only: bool = False) -> None:
        self._conn = duckdb.connect(database_path, read_only=read_only)
        self._pending: dict[str, tuple[Table, list[Record]]] = {}

    def insert(self, table: Table, rows: Sequence[Record]) -> None:
        records = list(rows)
        if not records:
            return
        if table.name not in self._pending:
            self._pending[table.name] = (table, [])
        self._pending[table.name][1].extend(records)

    def fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str) -> None:
        for statement in sql.strip().rstrip(";").split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    def commit(self) -> None:
        for table, rows in self._pending.values():
            self._copy_insert(table, rows)
        self._conn.commit()
        self._pending.clear()

    def _copy_insert(self, table: Table, rows: list[Record]) -> None:
        names = list(table.insert_columns)
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file, lineterminator="\n")
                writer.writerow(names)
                for row in rows:
                    writer.writerow("" if value is None else value for value in row.insert_values())
            column_list = ", ".join(names)
            escaped_path = csv_path.replace("'", "''")
            self._conn.execute(
                f"COPY {table.name} ({column_list}) FROM '{escaped_path}' "
                "(HEADER TRUE, DELIMITER ',', QUOTE '\"', ESCAPE '\"', NULL '')"
            )
        finally:
            os.unlink(csv_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def ensure_table(database_path: Path, table: Table, *, replace: bool = False) -> Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = Connection(database_path)
    if replace:
        connection.execute(f"DROP TABLE IF EXISTS {table.name}")
        connection.execute(f"DROP SEQUENCE IF EXISTS {table.sequence}")
    connection.execute(table.create_sql)
    return connection


def import_rows(
    database_path: Path,
    table: Table,
    rows: Sequence[Record],
    *,
    replace: bool = False,
) -> None:
    with ensure_table(database_path, table, replace=replace) as con:
        con.insert(table, rows)
        con.commit()


def fetch_utterances(connection: Connection, corpus: str, subsets: Sequence[str]) -> list[Utterance]:
    if not subsets:
        raise ValueError("utterance splits must be a non-empty list")
    placeholders = ", ".join("?" for _ in subsets)
    sql = (
        f"SELECT {UTTERANCES_TABLE.select_sql} FROM {UTTERANCES_TABLE.name} "
        f"WHERE corpus = ? AND subset IN ({placeholders}) ORDER BY id"
    )
    return [Utterance.from_sql(row) for row in connection.fetchall(sql, (corpus, *subsets))]


def fetch_noise_configs(
    connection: Connection,
    split: str,
    ids: Sequence[int] | None,
) -> list[NoiseConfig]:
    sql = f"SELECT {NOISE_CONFIGS_TABLE.select_sql} FROM {NOISE_CONFIGS_TABLE.name} WHERE split = ?"
    params: list[object] = [split]
    if ids:
        placeholders = ", ".join("?" for _ in ids)
        sql += f" AND id IN ({placeholders})"
        params.extend(ids)
    sql += " ORDER BY id"
    return [NoiseConfig.from_sql(row) for row in connection.fetchall(sql, tuple(params))]


def fetch_noises(connection: Connection) -> list[Noise]:
    sql = f"SELECT {NOISES_TABLE.select_sql} FROM {NOISES_TABLE.name} ORDER BY audio_path"
    return [Noise.from_sql(row) for row in connection.fetchall(sql)]


def fetch_noise_by_id(connection: Connection, noise_id: int) -> Noise:
    sql = f"SELECT {NOISES_TABLE.select_sql} FROM {NOISES_TABLE.name} WHERE id = ? LIMIT 1"
    rows = connection.fetchall(sql, (noise_id,))
    if not rows:
        raise ValueError(f"Noise id {noise_id} not found")
    return Noise.from_sql(rows[0])
