# Bump PROMPT_VERSION on any semantic change; prompt_hash() catches accidental edits within a version.
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

PROMPT_VERSION = "v1"

ALLOWED_CHART_TYPES = ("bar", "line", "pie", "scatter", "area", "histogram")

SYSTEM_PROMPT = (
    "You turn a SQL query result into a short answer and an optional chart specification "
    "for a user's question. You are given the user's question and a preview of the result "
    "rows (column names plus a few rows).\n"
    "Respond with exactly ONE JSON object and nothing else (no prose, no markdown fences), "
    "with these keys, in this order:\n"
    '  "chart": either null, or an object describing how to plot the result:\n'
    '      {"type": <chart type>, "x": <column>, "y": [<column>, ...], "series": <column or null>}\n'
    f"      type must be one of: {', '.join(ALLOWED_CHART_TYPES)}.\n"
    "      x, y and series must be column names taken verbatim from the preview header.\n"
    '  "answer": a 1-3 sentence plain-English answer to the question, grounded in the rows.\n'
    "Decide the chart first, then write the answer.\n"
    "Use null for chart when the result is not worth plotting: a single value, a single row, "
    "free text, or a bare list of identifiers/names. Prefer a chart when the result compares a "
    "numeric measure across categories or over time."
)

USER_TEMPLATE = """Question: {question}

Result preview ({n_rows} row(s)):
{preview}"""


def prompt_hash() -> str:
    payload = "\n".join([PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_rows(columns: Sequence[str], rows: Sequence[Any]) -> list[list[Any]]:
    """Accept rows as sequences or dicts; return list-of-lists aligned to ``columns``."""
    out: list[list[Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append([row.get(c) for c in columns])
        else:
            out.append(list(row))
    return out


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= 40 else text[:37] + "..."


def format_result_preview(
    columns: Sequence[str],
    rows: Sequence[Any],
    *,
    max_rows: int = 20,
    max_cols: int = 12,
) -> str:
    """Render a compact markdown-ish preview of the result, clipped to keep the prompt small."""
    columns = list(columns)
    shown_cols = columns[:max_cols]
    norm = _normalize_rows(columns, rows)
    header = " | ".join(shown_cols)
    sep = " | ".join("---" for _ in shown_cols)
    body_lines = [" | ".join(_cell(r[i]) for i in range(len(shown_cols))) for r in norm[:max_rows]]
    lines = [header, sep, *body_lines]
    if len(columns) > max_cols:
        lines.append(f"... ({len(columns) - max_cols} more column(s) omitted)")
    if len(norm) > max_rows:
        lines.append(f"... ({len(norm) - max_rows} more row(s) omitted)")
    return "\n".join(lines)


def build_messages(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Any],
    *,
    max_rows: int = 20,
    max_cols: int = 12,
) -> list[dict[str, str]]:
    preview = format_result_preview(columns, rows, max_rows=max_rows, max_cols=max_cols)
    user = USER_TEMPLATE.format(question=question, n_rows=len(rows), preview=preview)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class ChartSpec(BaseModel):
    type: str
    x: str
    y: list[str]
    series: str | None = None


class Presentation(BaseModel):
    chart: ChartSpec | None
    answer: str


class OutputParseError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    match = _OBJECT_RE.search(text)
    if match is None:
        raise OutputParseError(f"no JSON object found in output: {text[:80]!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise OutputParseError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise OutputParseError("top-level JSON value is not an object")
    return obj


def parse_output(
    text: str,
    columns: Sequence[str],
    *,
    allowed_types: Sequence[str] = ALLOWED_CHART_TYPES,
) -> Presentation:
    """Parse model output into a validated Presentation.

    Tolerates ```json fences and leading/trailing prose. Raises OutputParseError when the JSON
    is malformed, the answer is missing, the chart type is unknown, or a referenced column is
    not in ``columns`` so callers can count failures.
    """
    obj = _load_json_object(text)
    try:
        result = Presentation.model_validate(obj)
    except ValidationError as e:
        raise OutputParseError(f"output does not match schema: {e}") from e

    if not result.answer.strip():
        raise OutputParseError("answer is empty")

    chart = result.chart
    if chart is not None:
        known = set(columns)
        if chart.type not in allowed_types:
            raise OutputParseError(
                f"unknown chart type {chart.type!r}; allowed: {', '.join(allowed_types)}"
            )
        if not chart.y:
            raise OutputParseError("chart.y must list at least one column")
        referenced = [chart.x, *chart.y, *([chart.series] if chart.series else [])]
        unknown = [c for c in referenced if c not in known]
        if unknown:
            raise OutputParseError(f"chart references unknown column(s): {unknown}")

    return result
