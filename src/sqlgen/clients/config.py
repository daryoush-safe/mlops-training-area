from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMEndpoint:
    base_url: str
    model: str


@dataclass(frozen=True)
class ClientsConfig:
    pruner: VLLMEndpoint
    sqlgen: VLLMEndpoint
    presenter: VLLMEndpoint
    db_base_url: str
    request_timeout: float = 120.0
    db_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ClientsConfig":
        pruner_url = os.environ.get("PRUNER_BASE_URL", "http://localhost:8001/v1")
        sqlgen_url = os.environ.get("SQLGEN_BASE_URL", "http://localhost:8000/v1")
        db_url = os.environ.get("DB_SERVICE_URL", "http://localhost:8003")
        return cls(
            pruner=VLLMEndpoint(pruner_url, os.environ.get("PRUNER_MODEL", "schema-pruner")),
            sqlgen=VLLMEndpoint(sqlgen_url, os.environ.get("SQLGEN_MODEL", "sqlgen")),
            presenter=VLLMEndpoint(sqlgen_url, os.environ.get("PRESENTER_MODEL", "presenter")),
            db_base_url=db_url,
        )
