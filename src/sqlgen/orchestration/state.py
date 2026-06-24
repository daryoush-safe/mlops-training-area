from __future__ import annotations

from typing import TypedDict


class InferenceState(TypedDict, total=False):
    """State carried across LangGraph nodes for one text-to-SQL request."""

    question: str
    db_id: str
    schema: str | None  # full schema, serialized
    pruned_schema: str | None  # pruned schema, serialized
    sql: str | None
    error: str | None
    attempts: int
    max_retries: int
    status: str  # "running" | "retry" | "success" | "failed"
