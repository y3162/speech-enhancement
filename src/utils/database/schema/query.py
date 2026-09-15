from dataclasses import dataclass, replace
from typing import Any

from .column import Column
from .table import Table


@dataclass(frozen=True)
class Query:
    table: Table
    conditions: tuple[str, ...] = ()
    parameters: tuple[Any, ...] = ()
    row_limit: int | None = None

    def where(self, sql: str, *params: Any) -> "Query":
        if sql.count("?") != len(params):
            raise ValueError(
                f"Placeholder count ({sql.count('?')}) does not match "
                f"parameter count ({len(params)})"
            )
        return replace(
            self,
            conditions=self.conditions + (sql,),
            parameters=self.parameters + params,
        )

    def limit(self, n: int) -> "Query":
        return replace(self, row_limit=n)

    @property
    def row_columns(self) -> tuple[Column, ...]:
        return self.table.columns

    def build(self) -> tuple[str, tuple[Any, ...]]:
        columns = ", ".join(
            f"{self.table.name}.{column.name}" for column in self.table.columns
        )
        parts = [f"SELECT {columns} FROM {self.table.name}"]
        if self.conditions:
            parts.append("WHERE " + " AND ".join(f"({c})" for c in self.conditions))
        if self.row_limit is not None:
            parts.append(f"LIMIT {self.row_limit}")
        return " ".join(parts), self.parameters
