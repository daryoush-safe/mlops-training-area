from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx


@dataclass
class ExecResult:
    columns: list[str]
    rows: list[list]


class DBError(Exception):
    """Base class for database-backend failures."""


class SQLExecutionError(DBError): ...


class DBServiceError(DBError): ...


class DatabaseClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._url = f"{base_url.rstrip('/')}/api/v1/query"
        self._timeout = timeout

    async def execute(self, db_id: uuid.UUID, sql: str, user_token: str) -> ExecResult:
        headers = {"Authorization": f"Bearer {user_token}"}
        payload = {"connection_id": str(db_id), "sql": sql}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise DBServiceError(f"database service unreachable: {e}") from e

        if response.status_code == 200:
            data = response.json()
            return ExecResult(columns=data["columns"], rows=data["rows"])

        # DBService reports every domain error as JSON `{"detail": "..."}`.
        detail = self._detail(response)
        if response.status_code == 422:
            raise SQLExecutionError(detail)
        raise DBServiceError(f"query failed ({response.status_code}): {detail}")

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text or response.reason_phrase
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)
