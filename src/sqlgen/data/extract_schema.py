from __future__ import annotations

import argparse
import logging

from sqlgen.config import load_params
from sqlgen.data.io import read_json, write_json
from sqlgen.data.schema import schema_from_spider_entry

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    entries = read_json(params.paths.raw_dir / "tables.json")

    schemas = {}
    for entry in entries:
        schema = schema_from_spider_entry(entry)
        if schema.db_id in schemas:
            raise ValueError(f"duplicate db_id in tables.json: {schema.db_id}")
        schemas[schema.db_id] = schema.model_dump()

    out = params.paths.interim_dir / "schemas.json"
    write_json(out, schemas)
    log.info("wrote %d database schemas to %s", len(schemas), out)


if __name__ == "__main__":
    main()
