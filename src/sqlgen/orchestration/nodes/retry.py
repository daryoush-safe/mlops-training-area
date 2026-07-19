from __future__ import annotations

import logging

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState

log = logging.getLogger(__name__)


def make_retry_node(deps: Deps):
    max_retries = deps.params.inference.max_retries

    def retry_or_finish(state: InferenceState) -> InferenceState:
        error = state.get("error")
        if error is None:
            return {"status": "success"}

        if state.get("status") == "failed":
            log.info("fatal database error, not retrying: %s", error)
            return {"status": "failed"}

        attempts = state.get("attempts", 0) + 1
        history = list(state.get("error_history") or [])
        history.append(f"attempt {attempts}: {state.get('sql')!r} -> {error}")

        if attempts >= max_retries:
            log.info("sql generation failed after %d attempts: %s", attempts, error)
            return {"attempts": attempts, "error_history": history, "status": "failed"}

        return {"attempts": attempts, "error_history": history, "status": "retry"}

    return retry_or_finish
