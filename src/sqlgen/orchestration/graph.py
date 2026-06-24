from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from sqlgen.orchestration.state import InferenceState

if TYPE_CHECKING:
    from sqlgen.orchestration.nodes import InferencePipeline


def _failed_or(next_node: str):
    def route(state: InferenceState) -> str:
        return END if state.get("status") == "failed" else next_node

    return route


def _retry_or_end(state: InferenceState) -> str:
    return "generate_sql" if state.get("status") == "retry" else END


def build_graph(pipeline: "InferencePipeline"):
    """Wires the text-to-SQL inference graph around an already-loaded pipeline.

    parse_request -> load_schema -> prune_schema -> generate_sql -> validate_sql -> retry_or_finish
    retry_or_finish loops back to generate_sql while attempts < max_retries, else ends.

    """
    graph = StateGraph(InferenceState)
    graph.add_node("parse_request", pipeline.parse_request)
    graph.add_node("load_schema", pipeline.load_schema)
    graph.add_node("prune_schema", pipeline.prune_schema)
    graph.add_node("generate_sql", pipeline.generate_sql)
    graph.add_node("validate_sql", pipeline.validate_sql)
    graph.add_node("retry_or_finish", pipeline.retry_or_finish)

    graph.set_entry_point("parse_request")
    graph.add_conditional_edges("parse_request", _failed_or("load_schema"))
    graph.add_conditional_edges("load_schema", _failed_or("prune_schema"))
    graph.add_conditional_edges("prune_schema", _failed_or("generate_sql"))
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_edge("validate_sql", "retry_or_finish")
    graph.add_conditional_edges("retry_or_finish", _retry_or_end)

    return graph.compile()


def recursion_limit_for(pipeline: "InferencePipeline") -> int:
    """Bounds LangGraph's step count so a misconfigured max_retries can't silently
    hit the framework's default recursion limit instead of failing cleanly."""
    return pipeline.max_retries * 4 + 10
