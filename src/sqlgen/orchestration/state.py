from __future__ import annotations

import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Turn(TypedDict, total=False):
    """One completed exchange in a conversation, persisted across turns.

    Carried on the ``history`` channel and fed back into later generations so
    follow-up questions ("only the French ones?") can resolve against the prior
    question and the SQL that answered it.
    """

    question: str
    sql: str | None
    answer: str | None


def append_history(left: list[Turn] | None, right: list[Turn] | None) -> list[Turn]:
    """Reducer for the ``history`` channel: append new turns to the running log."""
    return (left or []) + (right or [])


class InferenceState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    # Completed turns for this thread, replayed from the checkpointer so the graph
    # has conversational memory. Never seeded by the initial input (that would
    # duplicate it) -- only terminal nodes append to it.
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
