from dataclasses import dataclass

from .column import Column


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    @property
    def insertable_columns(self) -> tuple[Column, ...]:
        return tuple(column for column in self.columns if column.insertable)

    @property
    def insert_sql(self) -> str:
        columns = self.insertable_columns
        if not columns:
            raise ValueError(f"Table {self.name} has no insertable columns")
        names = ", ".join(column.name for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        return f"INSERT INTO {self.name} ({names}) VALUES ({placeholders})"

    def get_create_table_sql(self) -> str:
        inline_primary = sum(column.primary for column in self.columns) == 1
        statements = [
            sql
            for column in self.columns
            if (sql := column.create_sequence_sql(self.name)) is not None
        ]
        definitions = [
            column.definition(table_name=self.name, inline_primary=inline_primary)
            for column in self.columns
        ]
        primary = [column.name for column in self.columns if column.primary]
        if len(primary) > 1:
            definitions.append(f"PRIMARY KEY ({', '.join(primary)})")
        definitions.extend(Column.unique_group_constraints(self.columns))
        definitions.extend(
            sql
            for column in self.columns
            if (sql := column.foreign_key_constraint()) is not None
        )
        body = ",\n".join(f"    {definition}" for definition in definitions)
        statements.append(f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n);")
        return "\n".join(statements)
