from __future__ import annotations

import argparse
import logging
import time

import mlflow
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from sqlgen import tracking
from sqlgen.config import load_params
from sqlgen.data.io import write_json
from sqlgen.training.dataset import load_rows, to_chat_dataset
from sqlgen.training.model import build_lora, load_base_model, load_tokenizer

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    cfg = params.train
    set_seed(cfg.seed)

    tracking.setup_mlflow(params.mlflow.experiment)
    run_name = f"train-{params.model.base.split('/')[-1].lower()}"
    with mlflow.start_run(run_name=run_name) as run:
        tracking.log_lineage(params)
        tracking.log_params_sections(params, ["model", "train"])

        tokenizer = load_tokenizer(params.model)
        train_rows = load_rows(params.paths.processed_dir / "train.jsonl", cfg.limit)
        val_rows = load_rows(params.paths.processed_dir / "val.jsonl", cfg.limit)
        train_ds, train_dropped = to_chat_dataset(train_rows, tokenizer, cfg.max_seq_len)
        val_ds, val_dropped = to_chat_dataset(val_rows, tokenizer, cfg.max_seq_len)
        mlflow.log_metrics(
            {"train_examples_dropped": train_dropped, "val_examples_dropped": val_dropped}
        )
        log.info(
            "train=%d val=%d (dropped %d/%d)",
            len(train_ds),
            len(val_ds),
            train_dropped,
            val_dropped,
        )

        model = load_base_model(params.model, bf16=cfg.bf16)

        sft_config = SFTConfig(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            bf16=cfg.bf16,
            max_length=cfg.max_seq_len,
            logging_steps=cfg.logging_steps,
            eval_strategy="epoch",
            save_strategy="no",  # only the final adapter is kept (saved below)
            seed=cfg.seed,
            report_to=["mlflow"],  # HF callback streams losses into the active run
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=build_lora(cfg.lora),
            processing_class=tokenizer,
        )

        start = time.time()
        result = trainer.train()
        eval_metrics = trainer.evaluate()

        trainer.save_model(str(cfg.output_dir))  # adapter + peft config
        tokenizer.save_pretrained(str(cfg.output_dir))
        # register_pruner points at this artifact path.
        mlflow.log_artifacts(str(cfg.output_dir), artifact_path="model")

        report = {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "train_runtime_s": round(time.time() - start, 1),
            "train_examples": len(train_ds),
            "train_examples_dropped": train_dropped,
        }
        write_json(params.paths.reports_dir / "train_report.json", report)
        log.info(
            "run %s done: train_loss=%.4f eval_loss=%s",
            run.info.run_id,
            result.training_loss,
            eval_metrics.get("eval_loss"),
        )


if __name__ == "__main__":
    main()
