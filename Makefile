.PHONY: setup infra-up infra-down pipeline pipeline-local test lint dvc-push dvc-pull

COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

setup: ## install local dev environment
	uv sync

infra-up: ## start MinIO + Prefect server
	$(COMPOSE) up -d minio minio-init prefect

infra-down:
	$(COMPOSE) down

pipeline: ## run the offline data pipeline in docker (builds image)
	$(COMPOSE) up --build pipeline

# The dockerized pipeline writes .dvc/config.local pointing at minio:9000
# (in-network endpoint); host targets reset it to localhost before running.
dvc-localhost:
	uv run dvc remote modify --local minio endpointurl http://localhost:9000

pipeline-local: dvc-localhost ## run the Prefect flow on the host (needs infra-up)
	uv run python flows/offline_data.py

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
