from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import s3fs

from sqlgen.config import ModelConfig

log = logging.getLogger(__name__)

_MARKER = "config.json"


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


def local_base_dir(cfg: ModelConfig) -> Path:
    root = os.environ.get("SQLGEN_MODEL_CACHE")
    if not root:
        hf_home = os.environ.get("HF_HOME")
        root = (
            f"{hf_home}/sqlgen-base" if hf_home else str(Path.home() / ".cache" / "sqlgen" / "base")
        )
    return Path(root) / cfg.base / cfg.revision


def is_mirrored(cfg: ModelConfig, fs: s3fs.S3FileSystem | None = None) -> bool:
    fs = fs or filesystem()
    return fs.exists(f"{cfg.mirror_bucket}/{cfg.mirror_key}/{_MARKER}")


def upload_base_model(cfg: ModelConfig, src: Path) -> str:
    fs = filesystem()
    if not fs.exists(cfg.mirror_bucket):
        fs.mkdir(cfg.mirror_bucket)
    target = f"{cfg.mirror_bucket}/{cfg.mirror_key}"
    fs.put(f"{str(src).rstrip('/')}/", f"{target}/", recursive=True)
    return cfg.mirror_uri


def ensure_base_model(cfg: ModelConfig) -> str:
    local = local_base_dir(cfg)
    if (local / _MARKER).exists():
        return str(local)

    fs = filesystem()
    src = f"{cfg.mirror_bucket}/{cfg.mirror_key}"
    if not fs.exists(f"{src}/{_MARKER}"):
        raise FileNotFoundError(
            f"Base model not found at {cfg.mirror_uri}. Seed it once with `make mirror-base`."
        )

    # Download into a sibling temp dir, then atomically swap in, so a crashed
    # download never leaves a half-populated path that the marker check trusts.
    tmp = local.with_name(local.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    log.info("downloading base model from %s -> %s", cfg.mirror_uri, local)
    fs.get(f"{src}/", f"{str(tmp)}/", recursive=True)
    tmp.replace(local)
    return str(local)
