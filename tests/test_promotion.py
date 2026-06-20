from types import SimpleNamespace

import pytest

from sqlgen.registry import promotion


class FakeClient:
    """Just enough of MlflowClient's alias API for promotion logic."""

    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases = dict(aliases or {})

    def get_model_version_by_alias(self, name: str, alias: str):
        if alias not in self.aliases:
            raise KeyError(alias)
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.aliases.pop(alias, None)


def test_set_candidate():
    client = FakeClient()
    promotion.set_candidate(client, "m", "3")
    assert client.aliases == {"candidate": "3"}


def test_promote_first_candidate_becomes_champion():
    client = FakeClient({"candidate": "1"})
    assert promotion.promote_candidate(client, "m") == "1"
    assert client.aliases == {"champion": "1"}


def test_promote_demotes_old_champion():
    client = FakeClient({"candidate": "2", "champion": "1"})
    promotion.promote_candidate(client, "m")
    assert client.aliases == {"champion": "2", "prev-champion": "1"}


def test_promote_same_version_does_not_self_demote():
    client = FakeClient({"candidate": "1", "champion": "1"})
    promotion.promote_candidate(client, "m")
    assert client.aliases == {"champion": "1"}


def test_promote_without_candidate_raises():
    with pytest.raises(ValueError):
        promotion.promote_candidate(FakeClient(), "m")
