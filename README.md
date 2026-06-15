# sqlgen-mlops

Text-to-SQL MLOps project built around two LLMs:

1. **schema pruner** — input: user question + full DB schema → output: the pruned
   schema (only the tables/columns relevant to the question)
2. **sqlgen** — input: user question + pruned schema → output: SQL

This repository implements the **offline data pipeline** for the schema pruner
(Spider → model-ready fine-tuning examples, by parsing each gold SQL query and
extracting exactly which tables/columns it uses) and its **training pipeline**
(LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct, tracked in MLflow, gated
registration into the MLflow model registry).

## Pipeline (DVC DAG)

```
spider_data.zip (dvc-tracked)
  └─ ingest            unpack annotation files → data/raw/spider
       ├─ extract_schema   tables.json → canonical per-DB schemas (data/interim/schemas.json)
       └─ extract_pairs    question/SQL files → deduped pairs per split (data/interim/pairs)
            └─ build_labels    sqlglot: gold SQL → used tables/columns → pruned schema
                                (data/interim/labeled + reports/prune_report.json)
                 └─ validate       quality gate: parse-failure / fallback rates, structural checks
                                    (reports/validation.json, fails pipeline on violation)
                      └─ build_dataset  model-ready JSONL → data/processed/schema_pruner/{train,val}.jsonl
                           └─ train         LoRA SFT (Qwen2.5-Coder-1.5B-Instruct) → models/schema_pruner
                                             one MLflow run: lineage + hyperparams + loss curves + adapter
                                └─ evaluate     generate on val → table/column P/R/F1, exact match
                                                 (reports/eval_schema_pruner.json, metrics → same run)
                                     └─ register     quality gates → MLflow model registry + `candidate` alias
                                                      (fails the pipeline if gates are violated)
```

Each processed row contains: `question`, `query`, structured + serialized
`full_schema` and `pruned_schema`, plus the used `tables`/`columns` and
provenance.

**Stack**: DVC (data versioning + pipeline cache, remote on MinIO/S3),
Prefect 3 (orchestration, retries, observability), sqlglot (SQL parsing +
column resolution), pydantic (config + data models), MLflow (experiment
tracking + model registry, artifacts on MinIO), transformers/peft/trl (LoRA
SFT), Docker Compose (infra).

## Quickstart

```bash
cp .env.example .env
make setup          # uv sync
make infra-up       # MinIO (:9000/:9001) + MLflow (:5000) + Prefect server (:4200)
make pipeline       # run the data flow dockerized (or: make pipeline-local)
make dvc-push       # push data/cache to MinIO (the flow already does this)

make setup-train    # + torch/transformers/peft/trl/mlflow (uv sync --extra train)
make mirror-base    # one-time: mirror the base model from HuggingFace into MinIO

make training       # train → evaluate → register dockerized (needs NVIDIA Container Toolkit)
make training-local # or on the host (GPU recommended)
```

`mirror-base` seeds `s3://models/base/<model>/<revision>` from HuggingFace once;
training and evaluation then load the base model from MinIO, so jobs don't depend
on huggingface.co at runtime. Re-run it only when you change `model.base`/`revision`.

For a quick end-to-end smoke test of the training plumbing, set
`train.limit: 64` and `eval.num_samples: 32` in `params.yaml` first.

Without docker/Prefect, the pure DVC pipeline also works: `make repro`.

Run tests with `make test`.

## Configuration

Everything lives in `params.yaml` (tracked by DVC, so changing a param
invalidates exactly the affected stages): split composition, SQL dialect,
whether pruned schemas keep primary/foreign keys, schema serialization style,
the validation thresholds, the base model + LoRA/SFT hyperparameters, and the
evaluation quality gates. The MLflow tracking URI is environment, not config
(`MLFLOW_TRACKING_URI` in `.env`, defaults to `http://localhost:5000`).

## Training, tracking & registry

`train` runs LoRA SFT with `trl`'s `SFTTrainer` on chat-formatted examples
(prompt template in `src/sqlgen/prompts/pruner.py` — the template, the target
format and its parser live in one file so they cannot drift apart). Every run
logs full lineage to MLflow: git commit + dirty flag, the md5s of the training
data from `dvc.lock`, prompt version/hash + rendered template, base model,
all hyperparameters, loss curves, dropped-example counts, and the adapter
itself as a run artifact.

`evaluate` is pure measurement (table/column precision/recall/F1, exact match,
unparseable rate — logged to the same run); `register` is pure policy: it
applies the `eval.gates` thresholds and only then registers the adapter as a
new version of the registered model with the `candidate` alias. Promotion
(`candidate` → `champion`, old champion → `prev-champion`) lives in
`src/sqlgen/registry/promotion.py`.

## Layout

```
├── dvc.yaml / params.yaml      # pipeline DAG + parameters
├── infra/                      # docker-compose (MinIO, MLflow, Prefect), Dockerfile.pipeline
├── data/                       # raw / interim / processed — DVC-managed, gitignored
├── models/schema_pruner/       # trained LoRA adapter — DVC-managed, gitignored
├── src/sqlgen/
│   ├── config.py               # typed params.yaml access
│   ├── tracking.py             # MLflow run setup + lineage logging (git/dvc/prompt)
│   ├── data/                   # ingest, extract_schema, extract_pairs, prune, build_labels, validate
│   ├── features/build_dataset.py
│   ├── prompts/pruner.py       # versioned prompt template + target format + parser
│   ├── training/               # dataset, model/LoRA, train + evaluate entrypoints
│   ├── evaluation/pruner.py    # pure metric functions (P/R/F1, exact match)
│   └── registry/               # promotion (alias lifecycle) + register_pruner (gates)
├── flows/                      # Prefect flows: offline_data, training (wrap DVC stages + push)
├── reports/                    # prune/validation/train/eval/registration reports (git-tracked)
└── tests/
```

## Online part (planned, not implemented yet)

Curated sqlgen outputs (user question + accepted generated SQL) will be labeled
with the same `extract_used_schema()` core and land in `data/feedback/` as an
online feature source, joined into training by `build_dataset`.
