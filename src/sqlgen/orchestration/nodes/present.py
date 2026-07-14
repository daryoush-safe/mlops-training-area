from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState
from sqlgen.prompts.presenter import OutputParseError, build_messages, parse_output

log = logging.getLogger(__name__)


def make_present_node(deps: Deps):
    def present(state: InferenceState) -> InferenceState:
        cfg = deps.params.presenter
        columns = state.get("columns") or []
        prompt_messages = build_messages(
            question=state["question"],
            columns=columns,
            rows=state.get("rows") or [],
            max_rows=cfg.max_preview_rows,
            max_cols=cfg.max_preview_cols,
        )
        text = deps.presenter.complete(messages=prompt_messages, max_tokens=cfg.max_new_tokens)
        try:
            result = parse_output(text, columns, allowed_types=cfg.chart_types)
        except OutputParseError as e:
            log.warning("presenter output unparseable: %s", e)
            return {"answer": text, "chart": None, "messages": [AIMessage(content=text)]}
        chart = result.chart.model_dump() if result.chart is not None else None
        return {
            "answer": result.answer,
            "chart": chart,
            "messages": [AIMessage(content=result.answer)],
        }

    return present
