from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from sqlgen.data.schema import DBSchema


@dataclass
class ExtractionResult:
    # lowercased table name -> set of lowercased column names ("*" = all columns)
    used_columns: dict[str, set[str]] = field(default_factory=dict)
    method: str = "parser"  # "parser" | "fallback"
    error: str | None = None

    @property
    def tables(self) -> list[str]:
        return sorted(self.used_columns)

    def add(self, table: str, column: str | None = None) -> None:
        cols = self.used_columns.setdefault(table.lower(), set())
        if column is not None:
            cols.add(column.lower())


class ExtractionError(Exception):
    pass


def extract_used_schema(
    sql: str,
    schema: DBSchema,
    *,
    dialect: str = "sqlite",
    enable_fallback: bool = True,
) -> ExtractionResult:
    try:
        return _extract_with_parser(sql, schema, dialect=dialect)
    except Exception as e:  # noqa: BLE001 - any parse failure routes to fallback
        if not enable_fallback:
            raise ExtractionError(str(e)) from e
        result = _extract_with_token_fallback(sql, schema)
        result.error = f"{type(e).__name__}: {e}"
        if not result.used_columns:
            raise ExtractionError(f"fallback found no tables: {result.error}") from e
        return result


def _extract_with_parser(sql: str, schema: DBSchema, *, dialect: str) -> ExtractionResult:
    mapping = schema.to_sqlglot_mapping()
    parsed = sqlglot.parse_one(sql, read=dialect)
    qualified = qualify(
        parsed,
        schema=mapping,
        dialect=dialect,
        validate_qualify_columns=False,  # leave unresolvable names (string literals) alone
    )

    schema_cols = {
        t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables
    }

    result = ExtractionResult(method="parser")
    for scope in traverse_scope(qualified):
        physical: dict[str, str] = {}  # alias (lower) -> real table name (lower)
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table) and source.name.lower() in schema_cols:
                physical[alias.lower()] = source.name.lower()
                result.add(source.name)

        for column in scope.columns:
            col_name = column.name.lower()
            alias = column.table.lower()
            if alias:
                table = physical.get(alias)
                if table is None:
                    continue  # column of a subquery/CTE; counted in its own scope
                if isinstance(column.this, exp.Star):
                    result.add(table, "*")
                elif col_name in schema_cols[table]:
                    result.add(table, col_name)
            else:
                # Unqualified even after qualify(): accept only if exactly one table in
                # scope has the column — otherwise it's likely a double-quoted literal
                # that parsed as an identifier.
                candidates = {
                    t for t in physical.values() if col_name in schema_cols[t]
                }
                if len(candidates) == 1:
                    result.add(candidates.pop(), col_name)

    if not result.used_columns:
        raise ExtractionError("parser resolved no tables")
    return result


def _extract_with_token_fallback(sql: str, schema: DBSchema) -> ExtractionResult:
    # columns matched only within matched tables to limit false positives
    tokens = {
        tok.strip("`\"'(),;").lower()
        for tok in sql.replace(".", " ").replace("=", " ").split()
    }
    result = ExtractionResult(method="fallback")
    matched_tables = [t for t in schema.tables if t.name.lower() in tokens]
    for table in matched_tables:
        result.add(table.name)
        for col in table.columns:
            if col.name.lower() in tokens:
                result.add(table.name, col.name)
    return result
