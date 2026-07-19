from __future__ import annotations

import os

import httpx

CHAMPION = "champion"
_MLFLOW_ARTIFACTS_SCHEME = "mlflow-artifacts:/"
_ARTIFACT_BUCKET = os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow")


def _mlflow_base_url() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/")


def resolve_champion(name: str, alias: str = CHAMPION) -> dict:
    url = f"{_mlflow_base_url()}/api/2.0/mlflow/registered-models/alias"
    resp = httpx.get(url, params={"name": name, "alias": alias}, timeout=30)
    resp.raise_for_status()
    return resp.json()["model_version"]


def artifact_s3_prefix(source: str) -> str:
    if source.startswith(_MLFLOW_ARTIFACTS_SCHEME):
        return f"{_ARTIFACT_BUCKET}/{source[len(_MLFLOW_ARTIFACTS_SCHEME) :].lstrip('/')}"
    if source.startswith("s3://"):
        return source[len("s3://") :]
    raise ValueError(
        f"cannot resolve MinIO location for model source {source!r}; "
        "expected an mlflow-artifacts:/ or s3:// URI"
    )
