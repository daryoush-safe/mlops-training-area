from __future__ import annotations

import os

import s3fs


def _endpoint() -> str:
    return (
        os.environ.get("AWS_S3_ENDPOINT_URL")
        or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        or os.environ.get("DVC_REMOTE_ENDPOINT")
        or "http://localhost:9000"
    )


def filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": _endpoint()},
    )
