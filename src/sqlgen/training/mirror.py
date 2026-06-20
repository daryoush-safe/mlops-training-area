from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from sqlgen import storage
from sqlgen.config import load_params

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--force", action="store_true", help="re-upload even if already mirrored")
    args = parser.parse_args()

    cfg = load_params(args.params).model

    if not args.force and storage.is_mirrored(cfg):
        log.info("already mirrored at %s (use --force to overwrite)", cfg.mirror_uri)
        return

    log.info("downloading %s@%s from HuggingFace...", cfg.base, cfg.revision)
    local = snapshot_download(cfg.base, revision=cfg.revision)

    log.info("uploading snapshot to %s ...", cfg.mirror_uri)
    uri = storage.upload_base_model(cfg, Path(local))
    log.info("mirrored base model to %s", uri)


if __name__ == "__main__":
    main()
