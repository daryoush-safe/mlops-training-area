from __future__ import annotations

import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InferenceState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
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
