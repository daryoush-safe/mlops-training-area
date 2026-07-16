from __future__ import annotations

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CANDIDATE = "candidate"
CHAMPION = "champion"
PREV_CHAMPION = "prev-champion"


def get_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except MlflowException:
        return None


def get_champion_version(client: MlflowClient, name: str) -> str | None:
    return get_alias_version(client, name, CHAMPION)


def set_candidate(client: MlflowClient, name: str, version: str) -> None:
    client.set_registered_model_alias(name, CANDIDATE, version)


def promote_candidate(client: MlflowClient, name: str) -> str:
    candidate = get_alias_version(client, name, CANDIDATE)
    if candidate is None:
        raise ValueError(f"Model {name!r} has no '{CANDIDATE}' alias to promote.")
    champion = get_champion_version(client, name)
    if champion is not None and champion != candidate:
        client.set_registered_model_alias(name, PREV_CHAMPION, champion)
    client.set_registered_model_alias(name, CHAMPION, candidate)
    client.delete_registered_model_alias(name, CANDIDATE)
    return candidate
