# Bump PROMPT_VERSION on any semantic change; prompt_hash() catches accidental edits within a version.
from __future__ import annotations

import hashlib
import re

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a text-to-SQL generator for a SQLite database. Given a database schema "
    "and a user question, write a single SQL query that answers the question.\n"
    "Rules:\n"
    "- Use only the tables and columns that appear in the provided schema.\n"
    "- Target the SQLite dialect.\n"
    "- Return exactly one SQL statement.\n"
    "- Output only the SQL query: no explanation, no comments, no markdown fences."
)

USER_TEMPLATE = """Database schema:
{schema}

Question: {question}"""

RETRY_TEMPLATE = """Database schema:
{schema}

Question: {question}

Your previous SQL query failed and must be corrected.
Previous SQL:
{prev_sql}

Database error:
{error}"""


def prompt_hash() -> str:
    payload = "\n".join([PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, RETRY_TEMPLATE])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_messages(
    question: str,
    schema_text: str,
    target: str | None = None,
    *,
    prev_sql: str | None = None,
    error: str | None = None,
    history: list[dict] | None = None,
) -> list[dict[str, str]]:
    if prev_sql is not None and error is not None:
        user = RETRY_TEMPLATE.format(
            schema=schema_text, question=question, prev_sql=prev_sql, error=error
        )
    else:
        user = USER_TEMPLATE.format(schema=schema_text, question=question)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("sql"):
            messages.append({"role": "assistant", "content": turn["sql"]})
    messages.append({"role": "user", "content": user})
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


class OutputParseError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START_RE = re.compile(r"\b(WITH|SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = _SQL_START_RE.search(text)
    if start is None:
        raise OutputParseError(f"no SQL statement found in output: {text[:80]!r}")
    text = text[start.start() :]
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()
