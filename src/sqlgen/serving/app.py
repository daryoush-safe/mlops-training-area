from __future__ import annotations

import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sqlgen.orchestration.service import InferenceService

app = FastAPI()
inference_service = InferenceService()


class QueryRequest(BaseModel):
    question: str
    db_schema: str = Field(alias="schema")
    db_id: uuid.UUID
    thread_id: str | None = None


@app.post("/stream-query")
async def stream_query(request: QueryRequest, authorization: str = Header(...)):
    user_token = authorization.removeprefix("Bearer ").strip()
    try:
        return StreamingResponse(
            inference_service.stream_run(
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
