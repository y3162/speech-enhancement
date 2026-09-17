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
    Row,
    RowT,
    Table,
    Utterance,
)


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return Path(value)


def db_root_dir() -> Path:
    return _required_env_path("SPEECH_DB_ROOT_DIR")


def metadata_path() -> Path:
    return db_root_dir() / "metadata.duckdb"


def corpora_root_dir() -> Path:
    return _required_env_path("SPEECH_CORPORA_ROOT_DIR")


def librispeech_dir() -> Path:
    return corpora_root_dir() / "LibriSpeech"


def libritts_dir() -> Path:
    return corpora_root_dir() / "LibriTTS"


def vctk_dir() -> Path:
    return corpora_root_dir() / "VCTK"


def demand_dir() -> Path:
    return corpora_root_dir() / "DEMAND"


class Connection:
    def __init__(self, database_path: Path, read_only: bool = False) -> None:
        self._conn = duckdb.connect(database_path, read_only=read_only)
        self._pending: dict[str, tuple[Table, list[Row]]] = {}

    def insert(self, table: Table, rows: Row | Sequence[Row]) -> None:
        if isinstance(rows, Row):
            rows = [rows]
        else:
            rows = list(rows)
        if not rows:
            return
        for row in rows:
            if not isinstance(row, Row):
                raise TypeError(f"Expected Row, got {type(row).__name__}")
        if table.name not in self._pending:
            self._pending[table.name] = (table, [])
        self._pending[table.name][1].extend(rows)

    def fetch_rows(
        self,
        sql: str,
        params: tuple[object, ...],
        table: Table,
        row_type: type[RowT],
    ) -> list[RowT]:
        result = self._conn.execute(sql, params).fetchall()
        return [row_type.from_sql(table.columns, row) for row in result]

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

    def _copy_insert(self, table: Table, rows: list[Row]) -> None:
        columns = table.insertable_columns
        names = [column.name for column in columns]
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file, lineterminator="\n")
                writer.writerow(names)
                for row in rows:
                    writer.writerow("" if value is None else value for value in row.insert_params(columns))
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
        for column in table.columns:
            drop_sql = column.drop_sequence_sql(table.name)
            if drop_sql is not None:
                connection.execute(drop_sql)
    connection.execute(table.get_create_table_sql())
    return connection


def import_rows(
    database_path: Path,
    table: Table,
    rows: Sequence[Row],
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
        f"SELECT {UTTERANCES_TABLE.select_list} FROM {UTTERANCES_TABLE.name} "
        f"WHERE corpus = ? AND subset IN ({placeholders}) ORDER BY id"
    )
    return connection.fetch_rows(sql, (corpus, *subsets), UTTERANCES_TABLE, Utterance)


def fetch_noise_configs(
    connection: Connection,
    split: str,
    ids: Sequence[int] | None,
) -> list[NoiseConfig]:
    sql = f"SELECT {NOISE_CONFIGS_TABLE.select_list} FROM {NOISE_CONFIGS_TABLE.name} WHERE split = ?"
    params: list[object] = [split]
    if ids:
        placeholders = ", ".join("?" for _ in ids)
        sql += f" AND id IN ({placeholders})"
        params.extend(ids)
    sql += " ORDER BY id"
    return connection.fetch_rows(sql, tuple(params), NOISE_CONFIGS_TABLE, NoiseConfig)


def fetch_noises(connection: Connection) -> list[Noise]:
    sql = f"SELECT {NOISES_TABLE.select_list} FROM {NOISES_TABLE.name} ORDER BY audio_path"
    return connection.fetch_rows(sql, (), NOISES_TABLE, Noise)


def fetch_noise_by_id(connection: Connection, noise_id: int) -> Noise:
    sql = f"SELECT {NOISES_TABLE.select_list} FROM {NOISES_TABLE.name} WHERE id = ? LIMIT 1"
    rows = connection.fetch_rows(sql, (noise_id,), NOISES_TABLE, Noise)
    if not rows:
        raise ValueError(f"Noise id {noise_id} not found")
    return rows[0]
