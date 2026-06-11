from __future__ import annotations

import argparse
import logging
from collections import Counter

from sqlgen.config import load_params
from sqlgen.data.io import read_json, read_jsonl, write_json, write_jsonl
from sqlgen.data.prune import ExtractionError, extract_used_schema
from sqlgen.data.schema import DBSchema

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    prune_cfg = params.prune

    schemas = {
        db_id: DBSchema.model_validate(raw)
        for db_id, raw in read_json(params.paths.interim_dir / "schemas.json").items()
    }

    report: dict[str, dict] = {}
    for split in params.splits.model_dump():
        pairs_path = params.paths.interim_dir / "pairs" / f"{split}.jsonl"
        stats: Counter[str] = Counter()
        failures: list[dict] = []
        rows = []

        for pair in read_jsonl(pairs_path):
            stats["total"] += 1
            schema = schemas.get(pair["db_id"])
            if schema is None:
                stats["unknown_db"] += 1
                failures.append({**pair, "reason": "unknown db_id"})
                continue
            try:
                extraction = extract_used_schema(
                    pair["query"],
                    schema,
                    dialect=prune_cfg.dialect,
                    enable_fallback=prune_cfg.enable_fallback,
                )
            except ExtractionError as e:
                stats["extraction_failed"] += 1
                failures.append({**pair, "reason": str(e)})
                continue

            pruned = schema.prune(
                extraction.used_columns,
                include_primary_keys=prune_cfg.include_primary_keys,
                include_foreign_keys=prune_cfg.include_foreign_keys,
            )
            stats[extraction.method] += 1
            rows.append(
                {
                    **pair,
                    "tables": [t.name for t in pruned.tables],
                    "columns": {t.name: t.column_names() for t in pruned.tables},
                    "pruned_schema": pruned.model_dump(),
                    "extraction_method": extraction.method,
                }
            )

        out = params.paths.interim_dir / "labeled" / f"{split}.jsonl"
        n = write_jsonl(out, rows)
        report[split] = {
            "total": stats["total"],
            "labeled": n,
            "parser": stats["parser"],
            "fallback": stats["fallback"],
            "extraction_failed": stats["extraction_failed"],
            "unknown_db": stats["unknown_db"],
            "failures": failures[:50],  # sample for debugging
        }
        log.info(
            "split=%s: %d/%d labeled (parser=%d fallback=%d failed=%d)",
            split, n, stats["total"], stats["parser"], stats["fallback"],
            stats["extraction_failed"],
        )

    write_json(params.paths.reports_dir / "prune_report.json", report)


if __name__ == "__main__":
    main()
