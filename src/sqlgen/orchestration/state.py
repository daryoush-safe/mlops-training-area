from __future__ import annotations

import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Turn(TypedDict, total=False):
    question: str
    sql: str | None
    answer: str | None


def append_history(left: list[Turn] | None, right: list[Turn] | None) -> list[Turn]:
    return (left or []) + (right or [])


class InferenceState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    history: Annotated[list[Turn], append_history]
    question: str
    user_token: str
    db_id: uuid.UUID
    schema: str | None
    pruned_schema: str | None
    sql: str | None
    rows: list[list] | None
    columns: list[str] | None
    answer: str | None
    chart: dict | None
    error: str | None
    error_history: list[str]
    attempts: int
    status: str  # running | retry | success | failed
