from sqlgen.orchestration.state import InferenceState


def route_after_retry(state: InferenceState) -> str:
    status = state.get("status")
    if status == "retry":
        return "generate_sql"
    if status == "success":
        return "present"
    return "handle_failure"
