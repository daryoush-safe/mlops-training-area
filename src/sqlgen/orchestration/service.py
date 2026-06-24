from __future__ import annotations

import logging
from typing import Any

from sqlgen.config import Params, load_params
from sqlgen.orchestration.graph import build_graph, recursion_limit_for
from sqlgen.orchestration.nodes import InferencePipeline
from sqlgen.orchestration.state import InferenceState

log = logging.getLogger(__name__)


class InferenceService:
    """Long-lived wrapper that builds the graph once and reuses it for requests."""

    def __init__(self, params: Params | None = None, *, params_path: str = "params.yaml") -> None:
        self.params = params or load_params(params_path)
        self.pipeline = InferencePipeline(self.params)
        self.app = build_graph(self.pipeline)
        self.recursion_limit = recursion_limit_for(self.pipeline)

    def run(self, question: str, db_id: str) -> InferenceState:
        return self.app.invoke(
            {"question": question, "db_id": db_id},
            config={"recursion_limit": self.recursion_limit},
        )

    def run_many(self, requests: list[dict[str, Any]]) -> list[InferenceState]:
        return [self.run(req["question"], req["db_id"]) for req in requests]
