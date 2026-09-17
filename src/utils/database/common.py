from pathlib import Path

import duckdb

from .connection import Connection
from .schema import Row, Table


def create_database(
    database_path: Path,
    force: bool = False,
) -> None:
    if database_path.exists() and not force:
        raise FileExistsError(f"Database {database_path} already exists.")

    database_path.parent.mkdir(parents=True, exist_ok=True)


def create_table(
    database_path: Path,
    table: Table,
) -> None:
    with duckdb.connect(database_path) as conn:
        conn.execute(table.get_create_table_sql())


def open_metadata_db(
    database_path: Path,
    table: Table,
    force: bool = False,
) -> Connection:
    create_database(database_path, force=force)
    create_table(database_path, table)
    return Connection(database_path)


def import_rows(
    database_path: Path,
    table: Table,
    rows: list[Row],
    force: bool = False,
) -> None:
    with open_metadata_db(database_path, table, force=force) as con:
        con.insert(table, rows)
        con.commit()
