from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlgen.artifacts import storage
from sqlgen.config import load_params
from sqlgen.registry import resolve

log = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("models/serving/manifest.json")
_BASE_MARKER = "config.json"


def ensure_base(role: str, server: dict, registry: dict) -> None:
    cfg = registry.get(role)
    if cfg is None:
        raise KeyError(f"manifest server {role!r} has no matching entry in params.models")
    dest = Path(server["base"])
    src = f"{cfg.mirror_bucket}/{cfg.mirror_key}"
    if (dest / _BASE_MARKER).exists():
        return
    storage.ensure_dir_from_s3(src, dest, marker=_BASE_MARKER)


def ensure_adapter(role: str, server: dict, manifest: dict) -> None:
    adapter = manifest.get("adapter", {})
    name = adapter.get("registered_model")
    if not name:
        raise KeyError("manifest.adapter.registered_model is required to resolve the champion")
    dest = Path(server["adapter"])
    stamp = storage.ensure_champion_adapter(
        name, dest, alias=adapter.get("alias", resolve.CHAMPION)
    )
    adapter.update({k: stamp[k] for k in ("version", "run_id", "source") if k in stamp})


def prepare(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    registry = dict(load_params().models)  # {role: ModelConfig}

    for role, server in manifest["servers"].items():
        ensure_base(role, server, registry)
        if "adapter" in server:
            ensure_adapter(role, server, manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("serving models ready under %s", manifest.get("serving_dir", manifest_path.parent))
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to the serving manifest (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()
    prepare(args.manifest)


if __name__ == "__main__":
    main()
