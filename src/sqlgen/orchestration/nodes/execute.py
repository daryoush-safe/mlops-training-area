from __future__ import annotations

from sqlgen.clients.database import DBServiceError, SQLExecutionError
from sqlgen.orchestration.deps import Deps
from sqlgen.orchestration.state import InferenceState


def make_execute_node(deps: Deps):
    async def execute_sql(state: InferenceState) -> InferenceState:
        sql = state.get("sql")
        if not sql:
            return {"error": state.get("error") or "no SQL was generated"}
        try:
            result = await deps.db_client.execute(
                state["db_id"], sql=sql, user_token=state["user_token"]
            )
        except SQLExecutionError as e:
            return {"rows": None, "columns": None, "error": str(e)}
        except DBServiceError as e:
            return {"rows": None, "columns": None, "error": str(e), "status": "failed"}
        return {"rows": result.rows, "columns": result.columns, "error": None}

    return execute_sql
