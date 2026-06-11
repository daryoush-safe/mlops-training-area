from __future__ import annotations

import argparse
import logging
import sys

from sqlgen.config import load_params
from sqlgen.data.io import read_json, read_jsonl, write_json
from sqlgen.data.schema import DBSchema

log = logging.getLogger(__name__)


def validate_example(row: dict, schemas: dict[str, DBSchema]) -> str | None:
    schema = schemas.get(row["db_id"])
    if schema is None:
        return f"unknown db_id {row['db_id']}"
    if not row.get("tables"):
        return "no tables in label"
    table_map = schema.table_map()
    for tname in row["tables"]:
        table = table_map.get(tname.lower())
        if table is None:
            return f"table {tname} not in schema"
        valid_cols = {c.lower() for c in table.column_names()}
        for col in row["columns"].get(tname, []):
            if col.lower() not in valid_cols:
                return f"column {tname}.{col} not in schema"
    if not row["question"].strip() or not row["query"].strip():
        return "empty question or query"
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    cfg = params.validation
    schemas = {
        db_id: DBSchema.model_validate(raw)
        for db_id, raw in read_json(params.paths.interim_dir / "schemas.json").items()
    }
    prune_report = read_json(params.paths.reports_dir / "prune_report.json")

    metrics: dict[str, dict] = {}
    violations: list[str] = []

    for split, stats in prune_report.items():
        labeled_path = params.paths.interim_dir / "labeled" / f"{split}.jsonl"
        bad: list[str] = []
        n = 0
        for row in read_jsonl(labeled_path):
            n += 1
            error = validate_example(row, schemas)
            if error:
                bad.append(f"{row['id']}: {error}")

        total = max(stats["total"], 1)
        failure_rate = (stats["extraction_failed"] + stats["unknown_db"]) / total
        fallback_rate = stats["fallback"] / total
        metrics[split] = {
            "examples": n,
            "extraction_failure_rate": round(failure_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "invalid_examples": len(bad),
        }

        if n < cfg.min_examples_per_split:
            violations.append(f"{split}: only {n} examples (min {cfg.min_examples_per_split})")
        if failure_rate > cfg.max_parse_failure_rate:
            violations.append(
                f"{split}: extraction failure rate {failure_rate:.4f} "
                f"> {cfg.max_parse_failure_rate}"
            )
        if fallback_rate > cfg.max_fallback_rate:
            violations.append(
                f"{split}: fallback rate {fallback_rate:.4f} > {cfg.max_fallback_rate}"
            )
        if bad:
            violations.append(f"{split}: {len(bad)} structurally invalid examples")
            for line in bad[:10]:
                log.error("invalid example: %s", line)

    write_json(
        params.paths.reports_dir / "validation.json",
        {"splits": metrics, "violations": violations, "passed": not violations},
    )

    if violations:
        for v in violations:
            log.error("VALIDATION FAILED: %s", v)
        sys.exit(1)
    log.info("validation passed: %s", metrics)


if __name__ == "__main__":
    main()
