from __future__ import annotations

from pydantic import BaseModel, Field


class Column(BaseModel):
    name: str
    type: str = "text"
    is_primary_key: bool = False


class Table(BaseModel):
    name: str
    columns: list[Column] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class ForeignKey(BaseModel):
    table: str
    column: str
    ref_table: str
    ref_column: str


class DBSchema(BaseModel):
    db_id: str
    tables: list[Table] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    def table_map(self) -> dict[str, Table]:
        return {t.name.lower(): t for t in self.tables}

    def to_sqlglot_mapping(self) -> dict[str, dict[str, str]]:
        type_map = {
            "text": "TEXT",
            "number": "NUMERIC",
            "time": "TIMESTAMP",
            "boolean": "BOOLEAN",
            "others": "TEXT",
        }
        return {
            t.name: {c.name: type_map.get(c.type, "TEXT") for c in t.columns}
            for t in self.tables
        }

    def prune(
        self,
        used_columns: dict[str, set[str]],
        *,
        include_primary_keys: bool = True,
        include_foreign_keys: bool = True,
    ) -> "DBSchema":
        table_map = self.table_map()
        used = {t: {c for c in cols} for t, cols in used_columns.items() if t in table_map}

        if include_foreign_keys:
            for fk in self.foreign_keys:
                t, rt = fk.table.lower(), fk.ref_table.lower()
                if t in used and rt in used:
                    used[t].add(fk.column.lower())
                    used[rt].add(fk.ref_column.lower())

        pruned_tables: list[Table] = []
        for table in self.tables:
            tname = table.name.lower()
            if tname not in used:
                continue
            keep = used[tname]
            keep_all = "*" in keep  # SELECT * was used on this table
            columns = [
                col
                for col in table.columns
                if keep_all
                or col.name.lower() in keep
                or (include_primary_keys and col.is_primary_key)
            ]
            pruned_tables.append(Table(name=table.name, columns=columns))

        kept_cols = {
            t.name.lower(): {c.name.lower() for c in t.columns} for t in pruned_tables
        }
        pruned_fks = [
            fk
            for fk in self.foreign_keys
            if fk.column.lower() in kept_cols.get(fk.table.lower(), set())
            and fk.ref_column.lower() in kept_cols.get(fk.ref_table.lower(), set())
        ]
        return DBSchema(db_id=self.db_id, tables=pruned_tables, foreign_keys=pruned_fks)


def serialize_schema(
    schema: DBSchema, *, style: str = "verbose", include_types: bool = True
) -> str:
    # compact: `table(col1, col2, ...)` | verbose: CREATE TABLE-like blocks with types, PKs, FKs
    if style == "compact":
        parts = [f"{t.name}({', '.join(t.column_names())})" for t in schema.tables]
        return " | ".join(parts)

    if style != "verbose":
        raise ValueError(f"unknown serialization style: {style!r}")

    lines: list[str] = []
    for table in schema.tables:
        cols = []
        for col in table.columns:
            piece = col.name
            if include_types:
                piece += f" {col.type}"
            if col.is_primary_key:
                piece += " primary key"
            cols.append(piece)
        lines.append(f"table {table.name} ({', '.join(cols)})")
    for fk in schema.foreign_keys:
        lines.append(f"foreign key {fk.table}.{fk.column} references {fk.ref_table}.{fk.ref_column}")
    return "\n".join(lines)


def schema_from_spider_entry(entry: dict) -> DBSchema:
    table_names: list[str] = entry["table_names_original"]
    column_names: list[list] = entry["column_names_original"]  # [table_idx, name]
    column_types: list[str] = entry["column_types"]
    primary_keys = set(entry.get("primary_keys", []))
    # Spider primary_keys may contain nested lists for composite keys
    flat_pks: set[int] = set()
    for pk in primary_keys:
        if isinstance(pk, list):
            flat_pks.update(pk)
        else:
            flat_pks.add(pk)

    tables = [Table(name=name) for name in table_names]
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:  # global "*" pseudo-column in Spider's format
            continue
        tables[table_idx].columns.append(
            Column(
                name=col_name,
                type=column_types[col_idx],
                is_primary_key=col_idx in flat_pks,
            )
        )

    foreign_keys = []
    for from_idx, to_idx in entry.get("foreign_keys", []):
        from_t, from_c = column_names[from_idx]
        to_t, to_c = column_names[to_idx]
        foreign_keys.append(
            ForeignKey(
                table=table_names[from_t],
                column=from_c,
                ref_table=table_names[to_t],
                ref_column=to_c,
            )
        )
    return DBSchema(db_id=entry["db_id"], tables=tables, foreign_keys=foreign_keys)
