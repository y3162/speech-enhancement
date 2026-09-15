from pathlib import Path
import duckdb

from .schema import Table


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
