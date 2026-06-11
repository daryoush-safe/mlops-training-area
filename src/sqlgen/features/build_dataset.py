from __future__ import annotations

import argparse
import logging

from sqlgen.config import load_params
from sqlgen.data.io import read_json, read_jsonl, write_jsonl
from sqlgen.data.schema import DBSchema, serialize_schema

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    ser = params.serialization
    schemas = {
        db_id: DBSchema.model_validate(raw)
        for db_id, raw in read_json(params.paths.interim_dir / "schemas.json").items()
    }
    full_schema_text = {
        db_id: serialize_schema(s, style=ser.style, include_types=ser.include_types)
        for db_id, s in schemas.items()
    }

    for split in params.splits.model_dump():
        rows = []
        for row in read_jsonl(params.paths.interim_dir / "labeled" / f"{split}.jsonl"):
            pruned = DBSchema.model_validate(row["pruned_schema"])
            rows.append(
                {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "query": row["query"],
                    "full_schema": schemas[row["db_id"]].model_dump(),
                    "full_schema_text": full_schema_text[row["db_id"]],
                    "pruned_schema": row["pruned_schema"],
                    "pruned_schema_text": serialize_schema(
                        pruned, style=ser.style, include_types=ser.include_types
                    ),
                    "tables": row["tables"],
                    "columns": row["columns"],
                    "extraction_method": row["extraction_method"],
                    "source": row["source"],
                }
            )
        out = params.paths.processed_dir / f"{split}.jsonl"
        n = write_jsonl(out, rows)
        log.info("split=%s: wrote %d model-ready examples to %s", split, n, out)


if __name__ == "__main__":
    main()
