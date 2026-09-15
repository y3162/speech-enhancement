from pathlib import Path
import duckdb

from ..confirm import confirm
from .schema import Table


def create_database(database_path: Path) -> None:
    if database_path.exists():
        if confirm(f"Database {database_path} already exists. Do you want to overwrite it?"):
            database_path.unlink()
            
    database_path.parent.mkdir(parents=True, exist_ok=True)
    # database_path.touch()


def create_table(
    database_path: Path,
    table: Table,
) -> None:
    with duckdb.connect(database_path) as conn:
        conn.execute(table.get_create_table_sql())
