import pytest

from sqlgen.config import Params, load_params


def test_registry_contains_pruner_and_sqlgen_by_default():
    params = Params()
    assert set(dict(params.models)) >= {"pruner", "sqlgen"}


def test_model_for_matches_typed_attribute_access():
    params = Params()
    assert params.model_for("pruner") is params.models.pruner
    assert params.model_for("sqlgen") is params.models.sqlgen


def test_sqlgen_uses_3b_base():
    assert "3B" in Params().model_for("sqlgen").base


def test_each_model_has_distinct_mirror_key():
    params = Params()
    keys = {name: cfg.mirror_key for name, cfg in dict(params.models).items()}
    assert len(set(keys.values())) == len(keys)


def test_model_for_unknown_raises():
    with pytest.raises(KeyError):
        Params().model_for("nope")


def test_params_yaml_registers_both_models():
    params = load_params("params.yaml")
    assert "1.5B" in params.model_for("pruner").base
    assert "3B" in params.model_for("sqlgen").base
