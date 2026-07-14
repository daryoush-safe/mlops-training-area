.PHONY: setup setup-train infra-up infra-down mirror-base pipeline pipeline-local training training-local inference inference-local test lint dvc-push dvc-pull

COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

setup: ## install local dev environment
	uv sync

setup-train: ## install local dev environment + training stack (torch & friends)
	uv sync --extra train

infra-up: ## start MinIO + MLflow + Prefect server
	$(COMPOSE) up -d minio minio-init mlflow prefect

infra-down:
	$(COMPOSE) down

# Seed every registry base model into MinIO from HuggingFace (needs setup-train + infra-up).
# Override MIRROR_MODEL=pruner|sqlgen to seed just one (default: all).
MIRROR_MODEL ?= all
mirror-base:
	set -a; [ -f .env ] && . ./.env; set +a; \
	AWS_S3_ENDPOINT_URL=http://localhost:9000 uv run python -m sqlgen.training.mirror --model $(MIRROR_MODEL)

pipeline: ## run the offline data pipeline in docker (builds image)
	$(COMPOSE) up --build pipeline

training: ## run train -> evaluate -> register in docker (needs NVIDIA Container Toolkit)
	$(COMPOSE) up --build training

inference: ## run the LangGraph text-to-SQL inference flow in docker
# 	$(COMPOSE) run --rm inference --question "$(QUESTION)" --db-id "$(DB_ID)"
	$(COMPOSE) run --rm inference python flows/inference.py --question "$(QUESTION)" --db-id "$(DB_ID)"

# The dockerized pipeline writes .dvc/config.local pointing at minio:9000
# (in-network endpoint); host targets reset it to localhost before running.
dvc-localhost:
	uv run dvc remote modify --local minio endpointurl http://localhost:9000

pipeline-local: dvc-localhost ## run the Prefect flow on the host (needs infra-up)
	uv run python flows/offline_data.py

# Training runs on the host (the pipeline container has no GPU); needs setup-train + infra-up.
training-local: dvc-localhost ## run train -> evaluate -> register via Prefect on the host
	uv run python flows/training.py

inference-local: dvc-localhost ## run the LangGraph inference flow locally
	uv run python flows/inference.py --question "$(QUESTION)" --db-id "$(DB_ID)"

repro: ## run the raw DVC pipeline without Prefect
	uv run dvc repro

dvc-push: dvc-localhost
	uv run dvc push

dvc-pull: dvc-localhost
	uv run dvc pull

test:
	uv run pytest -q

lint:
	uv run ruff check src tests flows
