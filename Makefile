.PHONY: setup setup-train infra-up infra-down mirror-base pipeline pipeline-local training training-local test lint dvc-push dvc-pull

COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

setup: ## install local dev environment
	uv sync

setup-train: ## install local dev environment + training stack (torch & friends)
	uv sync --extra train

infra-up: ## start MinIO + MLflow + Prefect server
	$(COMPOSE) up -d minio minio-init mlflow prefect

infra-down:
	$(COMPOSE) down

# Seed the base model into MinIO from HuggingFace (needs setup-train + infra-up).
# Run once per base/revision; training then pulls the model from MinIO, not HF.
mirror-base:
	set -a; [ -f .env ] && . ./.env; set +a; \
	AWS_S3_ENDPOINT_URL=http://localhost:9000 uv run python -m sqlgen.training.mirror

pipeline: ## run the offline data pipeline in docker (builds image)
	$(COMPOSE) up --build pipeline

training: ## run train -> evaluate -> register in docker (needs NVIDIA Container Toolkit)
	$(COMPOSE) up --build training

# The dockerized pipeline writes .dvc/config.local pointing at minio:9000
# (in-network endpoint); host targets reset it to localhost before running.
dvc-localhost:
	uv run dvc remote modify --local minio endpointurl http://localhost:9000

pipeline-local: dvc-localhost ## run the Prefect flow on the host (needs infra-up)
	uv run python flows/offline_data.py

# Training runs on the host (the pipeline container has no GPU); needs setup-train + infra-up.
training-local: dvc-localhost ## run train -> evaluate -> register via Prefect on the host
	uv run python flows/training.py

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
