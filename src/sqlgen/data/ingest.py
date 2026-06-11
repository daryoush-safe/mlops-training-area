from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

from sqlgen.config import load_params

log = logging.getLogger(__name__)

# only JSON annotation files; sqlite databases are not needed (schemas come from tables.json)
WANTED_FILES = [
    "tables.json",
    "train_spider.json",
    "train_others.json",
    "dev.json",
]


def ingest(raw_zip: Path, raw_dir: Path) -> None:
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    with zipfile.ZipFile(raw_zip) as zf:
        names = {Path(n).name: n for n in zf.namelist() if not n.startswith("__MACOSX")}
        for fname in WANTED_FILES:
            if fname not in names:
                raise FileNotFoundError(f"{fname} not found in {raw_zip}")
            with zf.open(names[fname]) as src, open(raw_dir / fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
            log.info("extracted %s", fname)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    ingest(params.paths.raw_zip, params.paths.raw_dir)


if __name__ == "__main__":
    main()
