from __future__ import annotations

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState
from sqlgen.prompts.sqlgen import OutputParseError, build_messages, extract_sql


def make_generate_node(deps: Deps):
    def generate_sql(state: InferenceState) -> InferenceState:
        if state.get("status") == "retry":
            messages = build_messages(
                question=state["question"],
                schema_text=state["schema"],
                prev_sql=state.get("sql"),
                error=state.get("error"),
            )
        else:
            messages = build_messages(
                question=state["question"], schema_text=state["pruned_schema"]
            )
        text = deps.sql_generator.complete(
            messages=messages, max_tokens=deps.params.inference.prune_max_new_tokens
        )
        try:
            sql = extract_sql(text)
        except OutputParseError as e:
            return {"sql": None, "error": f"sql generation failed to parse: {e}"}
        return {"sql": sql, "error": None}

    return generate_sql
