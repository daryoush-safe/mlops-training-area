# Bump PROMPT_VERSION on any semantic change; prompt_hash() catches accidental edits within a version.
from __future__ import annotations

import hashlib
import re

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a schema pruner for a text-to-SQL system. Given a database schema "
    "and a user question, select the minimal set of tables and columns needed to "
    "answer the question, including primary keys of the selected tables and the "
    "foreign keys that link them.\n"
    "Answer with one line per table, exactly in the format:\n"
    "table_name(column_1, column_2)\n"
    "Output nothing else."
)

USER_TEMPLATE = """Database schema:
{schema}

Question: {question}"""


def prompt_hash() -> str:
    payload = "\n".join([PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_messages(
    question: str, schema_text: str, target: str | None = None
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema_text, question=question)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def format_target(tables: list[str], columns: dict[str, list[str]]) -> str:
    return "\n".join(f"{t}({', '.join(columns.get(t, []))})" for t in tables)


class OutputParseError(ValueError):
    pass


_LINE_RE = re.compile(r"^(\w+)\s*\(([^()]*)\)$")


def parse_output(text: str) -> tuple[list[str], dict[str, list[str]]]:
    # Tolerates code fences and trailing punctuation; raises OutputParseError so callers can count failures.
    tables: list[str] = []
    columns: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        if line.strip().startswith("```"):
            continue
        line = line.strip().strip("`").rstrip(",;.").strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match is None:
            raise OutputParseError(f"unparseable line: {line!r}")
        table = match.group(1)
        cols = [c.strip() for c in match.group(2).split(",") if c.strip()]
        if table in columns:  # duplicate table line: merge, keep first position
            columns[table].extend(c for c in cols if c not in columns[table])
        else:
            tables.append(table)
            columns[table] = cols
    if not tables:
        raise OutputParseError("no table selections found in output")
    return tables, columns
