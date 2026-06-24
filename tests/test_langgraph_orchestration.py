from langgraph.graph import END

from sqlgen.orchestration.graph import _retry_or_end
from sqlgen.orchestration.state import InferenceState


def test_retry_or_end_routes_back_to_generation_on_retry() -> None:
    state: InferenceState = {"status": "retry", "attempts": 1}
    assert _retry_or_end(state) == "generate_sql"


def test_retry_or_end_stops_when_workflow_succeeds() -> None:
    state: InferenceState = {"status": "success", "attempts": 1}
    assert _retry_or_end(state) == END
