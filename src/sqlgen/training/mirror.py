from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from sqlgen import storage
from sqlgen.config import ModelConfig, load_params

log = logging.getLogger(__name__)


def mirror_one(name: str, cfg: ModelConfig, *, force: bool = False) -> None:
    if not force and storage.is_mirrored(cfg):
        log.info("[%s] already mirrored at %s (use --force to overwrite)", name, cfg.mirror_uri)
        return

    log.info("[%s] downloading %s@%s from HuggingFace...", name, cfg.base, cfg.revision)
    local = snapshot_download(cfg.base, revision=cfg.revision)

    log.info("[%s] uploading snapshot to %s ...", name, cfg.mirror_uri)
    uri = storage.upload_base_model(cfg, Path(local))
    log.info("[%s] mirrored base model to %s", name, uri)


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

    registry = dict(load_params(args.params).models)  # {role: ModelConfig}
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
