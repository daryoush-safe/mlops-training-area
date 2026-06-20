from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from sqlgen import tracking
from sqlgen.config import EvalGatesConfig, load_params
from sqlgen.data.io import read_json, write_json
from sqlgen.prompts import pruner as pruner_prompt
from sqlgen.registry import promotion

log = logging.getLogger(__name__)


def check_gates(metrics: dict, gates: EvalGatesConfig) -> list[str]:
    failures = []
    if metrics["table_recall"] < gates.min_table_recall:
        failures.append(f"table_recall {metrics['table_recall']:.4f} < {gates.min_table_recall}")
    if metrics["column_recall"] < gates.min_column_recall:
        failures.append(
            f"column_recall {metrics['column_recall']:.4f} < {gates.min_column_recall}"
        )
    if metrics["unparseable_rate"] > gates.max_unparseable_rate:
        failures.append(
            f"unparseable_rate {metrics['unparseable_rate']:.4f} > {gates.max_unparseable_rate}"
        )
    return failures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    reports_dir = params.paths.reports_dir
    train_report = read_json(reports_dir / "train_report.json")
    eval_report = read_json(reports_dir / "eval_schema_pruner.json")

    failures = check_gates(eval_report["metrics"], params.eval.gates)
    if failures:
        for failure in failures:
            log.error("quality gate failed: %s", failure)
        sys.exit(1)

    tracking.setup_mlflow(params.mlflow.experiment)
    name = params.mlflow.registered_model
    version = mlflow.register_model(train_report["model_uri"], name)
    client = MlflowClient()
    client.set_model_version_tag(name, version.version, "prompt_version", pruner_prompt.PROMPT_VERSION)
    promotion.set_candidate(client, name, version.version)
    log.info("registered %s version %s (alias: %s)", name, version.version, promotion.CANDIDATE)

    write_json(
        reports_dir / "registration.json",
        {
            "model_name": name,
            "version": version.version,
            "alias": promotion.CANDIDATE,
            "run_id": train_report["run_id"],
            "model_uri": train_report["model_uri"],
            "metrics": eval_report["metrics"],
        },
    )


if __name__ == "__main__":
    main()
