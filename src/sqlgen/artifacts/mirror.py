from __future__ import annotations

import argparse
import logging
from pathlib import Path

import s3fs
from huggingface_hub import snapshot_download

from sqlgen.artifacts import s3
from sqlgen.config import ModelConfig, load_params

log = logging.getLogger(__name__)

_MARKER = "config.json"


def is_mirrored(cfg: ModelConfig, fs: s3fs.S3FileSystem | None = None) -> bool:
    fs = fs or s3.filesystem()
    return fs.exists(f"{cfg.mirror_bucket}/{cfg.mirror_key}/{_MARKER}")


def upload_base_model(cfg: ModelConfig, src: Path, fs: s3fs.S3FileSystem | None = None) -> str:
    fs = fs or s3.filesystem()
    if not fs.exists(cfg.mirror_bucket):
        fs.mkdir(cfg.mirror_bucket)
    target = f"{cfg.mirror_bucket}/{cfg.mirror_key}"
    fs.put(f"{str(src).rstrip('/')}/", f"{target}/", recursive=True)
    return cfg.mirror_uri


def mirror_one(name: str, cfg: ModelConfig, *, force: bool = False) -> None:
    fs = s3.filesystem()
    if not force and is_mirrored(cfg, fs):
        log.info("[%s] already mirrored at %s (use --force to overwrite)", name, cfg.mirror_uri)
        return

    log.info("[%s] downloading %s@%s from HuggingFace...", name, cfg.base, cfg.revision)
    local = snapshot_download(cfg.base, revision=cfg.revision)

    log.info("[%s] uploading snapshot to %s ...", name, cfg.mirror_uri)
    upload_base_model(cfg, Path(local), fs)

    if not is_mirrored(cfg, fs):
        raise RuntimeError(
            f"[{name}] upload to {cfg.mirror_uri} did not land ({_MARKER} not found on MinIO)"
        )
    log.info("[%s] mirrored base model to %s", name, cfg.mirror_uri)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--model",
        default="all",
        help="registry key to mirror (e.g. pruner, sqlgen) or 'all' (default)",
    )
    parser.add_argument("--force", action="store_true", help="re-upload even if already mirrored")
    args = parser.parse_args()

    registry = dict(load_params(args.params).models)
    if args.model == "all":
        targets = registry
    elif args.model in registry:
        targets = {args.model: registry[args.model]}
    else:
        parser.error(f"unknown model {args.model!r}; known: {sorted(registry)} or 'all'")

    for name, cfg in targets.items():
        mirror_one(name, cfg, force=args.force)


if __name__ == "__main__":
    main()
