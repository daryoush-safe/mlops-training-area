from sqlgen.orchestration.routing import route_after_retry
from sqlgen.orchestration.state import InferenceState


def test_route_back_to_generation_on_retry() -> None:
    state: InferenceState = {"status": "retry", "attempts": 1}
    assert route_after_retry(state) == "generate_sql"


def test_route_to_present_when_workflow_succeeds() -> None:
    state: InferenceState = {"status": "success", "attempts": 1}
    assert route_after_retry(state) == "present"


def test_route_to_failure_handler_when_failed() -> None:
    state: InferenceState = {"status": "failed", "attempts": 2}
    assert route_after_retry(state) == "handle_failure"
