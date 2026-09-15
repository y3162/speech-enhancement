from dataclasses import dataclass, replace
from typing import Any, Sequence

from .column import Column
from .row import Row
from .table import Table


@dataclass(frozen=True)
class Join:
    table: Table
    on: str
    kind: str = "INNER"


@dataclass(frozen=True)
class SelectedColumn:
    table: Table
    column: Column

    @property
    def sql(self) -> str:
        return f"{self.table.name}.{self.column.name}"


@dataclass(frozen=True)
class RowMapper:
    row_type: type[Row]
    columns: tuple[Column, ...]

    def map_row(self, values: Sequence[Any]) -> Row:
        return self.row_type.from_sql(self.columns, values)


@dataclass(frozen=True)
class Query:
    table: Table
    joins: tuple[Join, ...] = ()
    selected: tuple[SelectedColumn, ...] = ()
    conditions: tuple[str, ...] = ()
    parameters: tuple[Any, ...] = ()
    orderings: tuple[str, ...] = ()
    row_limit: int | None = None

    def _tables(self) -> tuple[Table, ...]:
        return (self.table, *(join.table for join in self.joins))

    def _all_columns(self) -> tuple[SelectedColumn, ...]:
        return tuple(
            SelectedColumn(table, column)
            for table in self._tables()
            for column in table.columns
        )

    def _resolve(self, column: str | Column) -> SelectedColumn:
        items = self._all_columns()
        if isinstance(column, Column):
            matches = tuple(item for item in items if item.column is column)
            if len(matches) == 1:
                return matches[0]
            column = column.name
        if "." in column:
            matches = tuple(item for item in items if item.sql == column)
        else:
            matches = tuple(item for item in items if item.column.name == column)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous column {column}: "
                + ", ".join(item.sql for item in matches)
            )
        raise ValueError(f"Unknown column {column}")

    def select(self, *columns: str | Column) -> "Query":
        return replace(self, selected=tuple(self._resolve(column) for column in columns))

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

    def order_by(self, *items: str) -> "Query":
        return replace(self, orderings=self.orderings + items)

    def join(self, table: Table, on: str, kind: str = "INNER") -> "Query":
        return replace(self, joins=self.joins + (Join(table, on, kind),))

    def limit(self, n: int) -> "Query":
        return replace(self, row_limit=n)

    @property
    def selected_columns(self) -> tuple[SelectedColumn, ...]:
        return self.selected or tuple(
            SelectedColumn(self.table, column) for column in self.table.columns
        )

    @property
    def row_columns(self) -> tuple[Column, ...]:
        return tuple(item.column for item in self.selected_columns)

    def mapper(self, row_type: type[Row]) -> RowMapper:
        return RowMapper(row_type, self.row_columns)

    def build(self) -> tuple[str, tuple[Any, ...]]:
        parts = [
            f"SELECT {', '.join(item.sql for item in self.selected_columns)} "
            f"FROM {self.table.name}"
        ]
        parts.extend(
            f"{join.kind} JOIN {join.table.name} ON {join.on}" for join in self.joins
        )
        if self.conditions:
            parts.append("WHERE " + " AND ".join(f"({c})" for c in self.conditions))
        if self.orderings:
            parts.append("ORDER BY " + ", ".join(self.orderings))
        if self.row_limit is not None:
            parts.append(f"LIMIT {self.row_limit}")
        return " ".join(parts), self.parameters
