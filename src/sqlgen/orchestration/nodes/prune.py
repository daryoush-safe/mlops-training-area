from __future__ import annotations

import logging

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState
from sqlgen.prompts.pruner import OutputParseError, build_messages, format_target, parse_output

log = logging.getLogger(__name__)


def make_prune_node(deps: Deps):
    def prune_schema(state: InferenceState) -> InferenceState:
        messages = build_messages(question=state["question"], schema_text=state["schema"])
        text = deps.pruner.complete(
            messages=messages, max_tokens=deps.params.inference.prune_max_new_tokens
        )
        try:
            tables, columns = parse_output(text)
            pruned_schema = format_target(tables, columns)
        except OutputParseError as e:
            log.warning("pruner output unparseable, falling back to full schema: %s", e)
            pruned_schema = state["schema"]
        return {"pruned_schema": pruned_schema}

    return prune_schema
