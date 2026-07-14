from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from sqlgen.config import load_params
from sqlgen.orchestration.service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    params = load_params()
    async with AsyncConnectionPool(
        conninfo=params.checkpoint.dsn(),
        max_size=params.checkpoint.pool_max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.inference_service = InferenceService(params=params, checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class QueryRequest(BaseModel):
    question: str
    db_schema: str = Field(alias="schema")
    db_id: uuid.UUID
    thread_id: str | None = None


@app.post("/stream-query")
async def stream_query(
    request: QueryRequest, http_request: Request, authorization: str = Header(...)
):
    user_token = authorization.removeprefix("Bearer ").strip()
    service: InferenceService = http_request.app.state.inference_service
    try:
        return StreamingResponse(
            service.stream_run(
                request.question,
                request.db_schema,
                request.db_id,
                user_token,
                thread_id=request.thread_id,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
