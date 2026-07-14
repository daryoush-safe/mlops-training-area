import asyncio
import json
import uuid

from sqlgen.orchestration.service import InferenceService


class _RaisingApp:
    """Stand-in graph whose stream blows up on first iteration."""

    async def astream(self, *args, **kwargs):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover  # makes this an async generator


class _OkApp:
    async def astream(self, *args, **kwargs):
        yield {"generate_sql": {"sql": "SELECT 1"}}
        yield {"present": {"answer": "one"}}


def _drain(agen) -> list[str]:
    async def run() -> list[str]:
        return [item async for item in agen]

    return asyncio.run(run())


def _service_with_app(app) -> InferenceService:
    service = InferenceService()
    service.app = app  # bypass the real graph/clients; we only exercise streaming
    return service


def test_stream_run_streams_updates_then_done():
    service = _service_with_app(_OkApp())
    events = _drain(service.stream_run("q", "schema", uuid.uuid4(), "tok"))
    assert any('"sql"' in e for e in events)
    assert events[-1] == "data: [DONE]\n\n"


def test_stream_run_emits_error_event_then_done_on_failure():
    service = _service_with_app(_RaisingApp())
    events = _drain(service.stream_run("q", "schema", uuid.uuid4(), "tok", thread_id="t1"))
    # The stream must still terminate cleanly...
    assert events[-1] == "data: [DONE]\n\n"
    # ...and carry a terminal error event instead of a truncated stream.
    error_events = [e for e in events if '"error"' in e]
    assert error_events, events
    payload = json.loads(error_events[0].removeprefix("data: ").strip())
    assert "kaboom" in payload["error"]
