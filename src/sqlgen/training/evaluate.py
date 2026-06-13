from __future__ import annotations

import argparse
import json
import logging

import mlflow
import mlflow.artifacts
import torch
from peft import PeftModel

from sqlgen import tracking
from sqlgen.config import load_params
from sqlgen.data.io import read_json, write_json
from sqlgen.evaluation.pruner import aggregate, score_example
from sqlgen.prompts.pruner import OutputParseError, build_messages, format_target, parse_output
from sqlgen.training.dataset import load_rows
from sqlgen.training.model import load_base_model, load_tokenizer

log = logging.getLogger(__name__)

N_LOGGED_SAMPLES = 50


def generate_batch(model, tokenizer, rows: list[dict], max_new_tokens: int) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(row["question"], row["full_schema_text"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completions = out[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    cfg = params.eval
    train_report = read_json(params.paths.reports_dir / "train_report.json")

    tracking.setup_mlflow(params.mlflow.experiment)
    adapter_path = mlflow.artifacts.download_artifacts(train_report["model_uri"])

    tokenizer = load_tokenizer(params.model)
    tokenizer.padding_side = "left"  # decoder-only batched generation
    model = load_base_model(params.model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    rows = load_rows(params.paths.processed_dir / "val.jsonl", cfg.num_samples)
    scores, samples = [], []
    n_unparseable = 0
    for i in range(0, len(rows), cfg.batch_size):
        batch = rows[i : i + cfg.batch_size]
        outputs = generate_batch(model, tokenizer, batch, cfg.max_new_tokens)
        for row, text in zip(batch, outputs):
            parse_error = None
            try:
                pred_tables, pred_columns = parse_output(text)
            except OutputParseError as e:
                n_unparseable += 1
                parse_error = str(e)
            else:
                scores.append(
                    score_example(row["tables"], row["columns"], pred_tables, pred_columns)
                )
            if len(samples) < N_LOGGED_SAMPLES:
                samples.append(
                    {
                        "id": row["id"],
                        "db_id": row["db_id"],
                        "question": row["question"],
                        "gold": format_target(row["tables"], row["columns"]),
                        "generated": text,
                        "parse_error": parse_error,
                    }
                )
        log.info("evaluated %d/%d", min(i + cfg.batch_size, len(rows)), len(rows))

    metrics = aggregate(scores, n_unparseable)
    log.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    with mlflow.start_run(run_id=train_report["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            "eval_samples.jsonl",
        )

    write_json(
        params.paths.reports_dir / "eval_schema_pruner.json",
        {"run_id": train_report["run_id"], "metrics": metrics},
    )


if __name__ == "__main__":
    main()
