import pytest

from sqlgen.evaluation.pruner import aggregate, qualified_columns, score_example

GOLD_TABLES = ["singer", "concert"]
GOLD_COLUMNS = {"singer": ["Singer_ID", "Name"], "concert": ["Concert_ID"]}


def test_perfect_prediction():
    score = score_example(GOLD_TABLES, GOLD_COLUMNS, GOLD_TABLES, GOLD_COLUMNS)
    assert score.table_f1 == 1.0
    assert score.column_f1 == 1.0
    assert score.exact_match


def test_comparison_is_case_insensitive():
    pred_columns = {"SINGER": ["singer_id", "name"], "Concert": ["concert_id"]}
    score = score_example(GOLD_TABLES, GOLD_COLUMNS, ["Singer", "CONCERT"], pred_columns)
    assert score.exact_match


def test_missing_table_hits_recall():
    score = score_example(GOLD_TABLES, GOLD_COLUMNS, ["singer"], {"singer": ["Singer_ID", "Name"]})
    assert score.table_recall == 0.5
    assert score.table_precision == 1.0
    assert score.column_recall == pytest.approx(2 / 3)
    assert not score.exact_match


def test_extra_column_hits_precision():
    pred_columns = {"singer": ["Singer_ID", "Name", "Age"], "concert": ["Concert_ID"]}
    score = score_example(GOLD_TABLES, GOLD_COLUMNS, GOLD_TABLES, pred_columns)
    assert score.column_precision == 0.75
    assert score.column_recall == 1.0


def test_qualified_columns():
    assert qualified_columns({"singer": ["Name"], "concert": ["Theme"]}) == {
        "singer.name",
        "concert.theme",
    }


def test_aggregate_counts_unparseable_as_misses():
    perfect = score_example(GOLD_TABLES, GOLD_COLUMNS, GOLD_TABLES, GOLD_COLUMNS)
    metrics = aggregate([perfect], n_unparseable=1)
    assert metrics["table_f1"] == 0.5
    assert metrics["exact_match"] == 0.5
    assert metrics["unparseable_rate"] == 0.5
    assert metrics["n_examples"] == 2.0


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([], n_unparseable=0)
