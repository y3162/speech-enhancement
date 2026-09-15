import csv
import os
import tempfile
from pathlib import Path
import duckdb

from .schema import Query, Row, Table


class Connection:
    def __init__(self, database_path: Path) -> None:
        self._conn = duckdb.connect(database_path)
        self._pending: dict[str, tuple[Table, list[Row]]] = {}

    def insert(self, table: Table, rows: Row | list[Row]) -> None:
        if isinstance(rows, Row):
            rows = [rows]
        elif not isinstance(rows, list):
            raise TypeError(f"Expected Row or list[Row], got {type(rows).__name__}")
        if not rows:
            return
        for row in rows:
            if not isinstance(row, Row):
                raise TypeError(f"Expected Row, got {type(row).__name__}")
        if table.name not in self._pending:
            self._pending[table.name] = (table, [])
        self._pending[table.name][1].extend(rows)

    def fetch(self, query: Query, row_type: type[Row]) -> list[Row]:
        sql, params = query.build()
        result = self._conn.execute(sql, params).fetchall()
        columns = query.row_columns
        return [row_type.from_sql(columns, row) for row in result]

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
                    writer.writerow(
                        "" if value is None else value
                        for value in row.insert_params(columns)
                    )
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
