from dataclasses import dataclass

from .column import Column


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    @property
    def insertable_columns(self) -> tuple[Column, ...]:
        return tuple(column for column in self.columns if column.insertable)

    def get_create_table_sql(self) -> str:
        statements = [
            sql
            for column in self.columns
            if (sql := column.create_sequence_sql(self.name)) is not None
        ]
        definitions = [column.definition(self.name) for column in self.columns]
        body = ",\n".join(f"    {definition}" for definition in definitions)
        statements.append(f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n);")
        return "\n".join(statements)
