from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.nodes.execute import make_execute_node
from sqlgen.orchestration.nodes.failure import make_failure_node
from sqlgen.orchestration.nodes.generate import make_generate_node
from sqlgen.orchestration.nodes.present import make_present_node
from sqlgen.orchestration.nodes.prune import make_prune_node
from sqlgen.orchestration.nodes.retry import make_retry_node
from sqlgen.orchestration.routing import route_after_retry
from sqlgen.orchestration.state import InferenceState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_graph(deps: Deps, checkpointer: "BaseCheckpointSaver | None" = None):
    graph = StateGraph(InferenceState)
    graph.add_node("prune_schema", make_prune_node(deps=deps))
    graph.add_node("generate_sql", make_generate_node(deps=deps))
    graph.add_node("execute_sql", make_execute_node(deps=deps))
    graph.add_node("retry_or_finish", make_retry_node(deps=deps))
    graph.add_node("present", make_present_node(deps=deps))
    graph.add_node("handle_failure", make_failure_node(deps=deps))

    graph.set_entry_point("prune_schema")
    graph.add_edge("prune_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "retry_or_finish")
    graph.add_conditional_edges(
        "retry_or_finish",
        route_after_retry,
        {
            "generate_sql": "generate_sql",
            "present": "present",
            "handle_failure": "handle_failure",
        },
    )
    graph.add_edge("present", END)
    graph.add_edge("handle_failure", END)

    return graph.compile(checkpointer=checkpointer)


def recursion_limit_for(max_retries: int) -> int:
    return max_retries * 4 + 10
