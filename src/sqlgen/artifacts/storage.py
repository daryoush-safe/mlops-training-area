from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import httpx
import s3fs

from sqlgen.artifacts.s3 import filesystem
from sqlgen.config import ModelConfig
from sqlgen.registry import resolve

log = logging.getLogger(__name__)

_MARKER = "config.json"
_ADAPTER_MARKER = "adapter_config.json"
_STAMP = ".champion.json"


def _download_tree(fs: s3fs.S3FileSystem, src_prefix: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    fs.get(f"{src_prefix.rstrip('/')}/", f"{str(tmp)}/", recursive=True)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)


def ensure_dir_from_s3(
    src_prefix: str,
    dest: Path | str,
    *,
    marker: str = _MARKER,
    force: bool = False,
    fs: s3fs.S3FileSystem | None = None,
) -> str:
    dest = Path(dest)
    if not force and (dest / marker).exists():
        return str(dest)

    fs = fs or filesystem()
    if not fs.exists(f"{src_prefix.rstrip('/')}/{marker}"):
        raise FileNotFoundError(f"No object tree at s3://{src_prefix} (missing {marker}).")

    log.info("downloading s3://%s -> %s", src_prefix, dest)
    _download_tree(fs, src_prefix, dest)
    return str(dest)


def local_base_dir(cfg: ModelConfig) -> Path:
    root = os.environ.get("SQLGEN_MODEL_CACHE")
    if not root:
        hf_home = os.environ.get("HF_HOME")
        root = (
            f"{hf_home}/sqlgen-base" if hf_home else str(Path.home() / ".cache" / "sqlgen" / "base")
        )
    return Path(root) / cfg.base / cfg.revision


def ensure_base_model(cfg: ModelConfig) -> str:
    local = local_base_dir(cfg)
    if (local / _MARKER).exists():
        return str(local)
    try:
        return ensure_dir_from_s3(f"{cfg.mirror_bucket}/{cfg.mirror_key}", local)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Base model not found at {cfg.mirror_uri}. Seed it once with `make mirror-base`."
        ) from exc


def _read_stamp(dest: Path) -> dict:
    stamp = dest / _STAMP
    if not stamp.exists():
        return {}
    try:
        return json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_stamp(dest: Path, payload: dict) -> None:
    (dest / _STAMP).write_text(json.dumps(payload, indent=2))


def ensure_champion_adapter(
    name: str,
    dest: Path | str,
    *,
    alias: str = resolve.CHAMPION,
    fs: s3fs.S3FileSystem | None = None,
) -> dict:
    dest = Path(dest)
    present = (dest / _ADAPTER_MARKER).exists()

    try:
        champion = resolve.resolve_champion(name, alias)
    except (httpx.HTTPError, KeyError) as exc:
        if present:
            log.warning(
                "cannot reach MLflow champion for %s (%s); keeping local adapter", name, exc
            )
            return _read_stamp(dest)
        raise RuntimeError(
            f"no local adapter for {name!r} and MLflow champion lookup failed: {exc}"
        ) from exc

    version = str(champion["version"])
    if present and _read_stamp(dest).get("version") == version:
        return _read_stamp(dest)

    src = resolve.artifact_s3_prefix(champion["source"])
    log.info("loading champion adapter v%s for %s from s3://%s", version, name, src)
    ensure_dir_from_s3(src, dest, marker=_ADAPTER_MARKER, force=True, fs=fs)

    stamp = {
        "registered_model": name,
        "alias": alias,
        "version": version,
        "run_id": champion.get("run_id"),
        "source": champion["source"],
    }
    _write_stamp(dest, stamp)
    return stamp
