from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.messages import HumanMessage

from sqlgen.config import Params, load_params
from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.graph import build_graph, recursion_limit_for
from sqlgen.orchestration.state import InferenceState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

log = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        params: Params | None = None,
        *,
        params_path: str = "params.yaml",
        checkpointer: "BaseCheckpointSaver | None" = None,
    ) -> None:
        self.params = params or load_params(params_path)
        self.deps = Deps.build(self.params)
        self.app = build_graph(self.deps, checkpointer=checkpointer)
        self.recursion_limit = recursion_limit_for(self.params.inference.max_retries)

    def _initial_state(
        self, question: str, schema: str, db_id: uuid.UUID, user_token: str
    ) -> InferenceState:
        return {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "schema": schema,
            "db_id": db_id,
            "user_token": user_token,
            "attempts": 0,
            "error": None,
            "error_history": [],
            "status": "running",
        }

    async def stream_run(
        self,
        question: str,
        schema: str,
        db_id: uuid.UUID,
        user_token: str,
        *,
        thread_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        inputs = self._initial_state(question, schema, db_id, user_token)
        resolved_thread_id = thread_id or str(uuid.uuid4())
        config = {
            "recursion_limit": self.recursion_limit,
            "configurable": {"thread_id": resolved_thread_id},
        }

        try:
            async for chunk in self.app.astream(inputs, config=config, stream_mode="updates"):
                yield f"data: {json.dumps(chunk, default=str)}\n\n"
        except Exception as e:
            # Failures surface mid-stream, after the response headers (and earlier
            # events) have already been sent, so the HTTP layer can no longer turn
            # them into a 5xx. Emit a terminal error event the client can render
            # instead of leaving it to guess at a truncated stream.
            log.exception("inference stream failed for thread %s", resolved_thread_id)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
