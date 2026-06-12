from __future__ import annotations

CANDIDATE = "candidate"
CHAMPION = "champion"
PREV_CHAMPION = "prev-champion"


def get_alias_version(client, name: str, alias: str) -> str | None:
    # RestException when alias unset; duck-typed client forces broad except.
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:
        return None


def set_candidate(client, name: str, version: str) -> None:
    client.set_registered_model_alias(name, CANDIDATE, version)


def promote_candidate(client, name: str) -> str:
    candidate = get_alias_version(client, name, CANDIDATE)
    if candidate is None:
        raise ValueError(f"model {name!r} has no '{CANDIDATE}' alias to promote")
    champion = get_alias_version(client, name, CHAMPION)
    if champion is not None and champion != candidate:
        client.set_registered_model_alias(name, PREV_CHAMPION, champion)
    client.set_registered_model_alias(name, CHAMPION, candidate)
    client.delete_registered_model_alias(name, CANDIDATE)
    return candidate
