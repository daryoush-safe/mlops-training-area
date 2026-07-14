from __future__ import annotations

from langchain_core.messages import AIMessage

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState


def make_failure_node(deps: Deps):
    def handle_failure(state: InferenceState) -> InferenceState:
        answer = "Sorry, I am potato."
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    return handle_failure
