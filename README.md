# sqlgen-mlops

Text-to-SQL MLOps project built around two LLMs:

1. **schema pruner** — input: user question + full DB schema → output: the pruned
   schema (only the tables/columns relevant to the question)
2. **sqlgen** — input: user question + pruned schema → output: SQL

This repository currently implements the **offline data pipeline for the schema
pruner**: it turns the Spider dataset into model-ready fine-tuning examples by
parsing each gold SQL query and extracting exactly which tables/columns it uses.

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
```

Each processed row contains: `question`, `query`, structured + serialized
`full_schema` and `pruned_schema`, plus the used `tables`/`columns` and
provenance. Prompt templating is deliberately left to the (future) training
stage so it can change without re-labeling.

**Stack**: DVC (data versioning + pipeline cache, remote on MinIO/S3),
Prefect 3 (orchestration, retries, observability), sqlglot (SQL parsing +
column resolution), pydantic (config + data models), Docker Compose (infra).

## Quickstart

```bash
cp .env.example .env
make setup        # uv sync
make infra-up     # MinIO (:9000/:9001) + Prefect server (:4200)
make pipeline     # run the flow dockerized (or: make pipeline-local)
make dvc-push     # push data/cache to MinIO (the flow already does this)
```

Without docker/Prefect, the pure DVC pipeline also works: `make repro`.

Run tests with `make test`.

> **Note (host runs):** if your shell exports a SOCKS proxy
> (`all_proxy=socks://...`), Prefect's local client and DVC's S3 calls to
> localhost will fail. Run host commands with the proxy unset, e.g.
> `env -u all_proxy -u http_proxy -u https_proxy make pipeline-local`.
> Containers are unaffected.

## Configuration

Everything lives in `params.yaml` (tracked by DVC, so changing a param
invalidates exactly the affected stages): split composition, SQL dialect,
whether pruned schemas keep primary/foreign keys, schema serialization style,
and the validation thresholds.

## Layout

```
├── dvc.yaml / params.yaml      # pipeline DAG + parameters
├── infra/                      # docker-compose (MinIO, Prefect), Dockerfile.pipeline
├── data/                       # raw / interim / processed — DVC-managed, gitignored
├── src/sqlgen/
│   ├── config.py               # typed params.yaml access
│   ├── data/                   # ingest, extract_schema, extract_pairs, prune, build_labels, validate
│   └── features/build_dataset.py
├── flows/offline_data.py       # Prefect flow wrapping the DVC stages + push
├── reports/                    # prune report + validation metrics (git-tracked)
└── tests/
```

## Online part (planned, not implemented yet)

Curated sqlgen outputs (user question + accepted generated SQL) will be labeled
with the same `extract_used_schema()` core and land in `data/feedback/` as an
online feature source, joined into training by `build_dataset`.
