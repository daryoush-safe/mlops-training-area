from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlgen.artifacts import storage
from sqlgen.config import ModelConfig, load_params
from sqlgen.registry import resolve

log = logging.getLogger(__name__)

DEFAULT_SERVING_DIR = Path("models/serving")
_BASE_MARKER = "config.json"


def ensure_base(base_dir: Path, cfg: ModelConfig) -> None:
    if (base_dir / _BASE_MARKER).exists():
        return
    src = f"{cfg.mirror_bucket}/{cfg.mirror_key}"
    storage.ensure_dir_from_s3(src, base_dir, marker=_BASE_MARKER)


def ensure_adapter(adapter_dir: Path, registered_model: str) -> dict:
    return storage.ensure_champion_adapter(
        registered_model, adapter_dir, alias=resolve.CHAMPION
    )


def prepare(serving_dir: Path = DEFAULT_SERVING_DIR) -> dict:
    params = load_params()
    servers: dict[str, dict] = {}

    for role, cfg in dict(params.models).items():
        role_dir = serving_dir / role
        base_dir = role_dir / "base"
        ensure_base(base_dir, cfg)
        server = {"base": str(base_dir)}

        if cfg.adapter:
            adapter_dir = role_dir / "adapter"
            stamp = ensure_adapter(adapter_dir, cfg.adapter)
            server["adapter"] = str(adapter_dir)
            server["champion"] = {
                k: stamp[k]
                for k in ("registered_model", "version", "run_id", "source")
                if k in stamp
            }
        servers[role] = server

    manifest = {"serving_dir": str(serving_dir), "servers": servers}
    serving_dir.mkdir(parents=True, exist_ok=True)
    (serving_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("serving models ready under %s", serving_dir)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serving-dir",
        type=Path,
        default=DEFAULT_SERVING_DIR,
        help=f"local dir to materialize serving models into (default: {DEFAULT_SERVING_DIR})",
    )
    args = parser.parse_args()
    prepare(args.serving_dir)


if __name__ == "__main__":
    main()
