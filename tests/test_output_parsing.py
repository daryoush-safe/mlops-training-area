import pytest

from sqlgen.prompts.pruner import OutputParseError, format_target, parse_output


def test_round_trip():
    tables = ["singer", "singer_in_concert"]
    columns = {"singer": ["Singer_ID", "Name"], "singer_in_concert": ["Singer_ID"]}
    assert parse_output(format_target(tables, columns)) == (tables, columns)


def test_parse_single_table():
    assert parse_output("head(head_ID, age)") == (["head"], {"head": ["head_ID", "age"]})


def test_parse_empty_columns():
    assert parse_output("head()") == (["head"], {"head": []})


def test_parse_tolerates_code_fences_and_whitespace():
    text = "```\n  head(head_ID, age) \n\n```"
    assert parse_output(text) == (["head"], {"head": ["head_ID", "age"]})


def test_parse_tolerates_trailing_punctuation():
    assert parse_output("head(head_ID, age),") == (["head"], {"head": ["head_ID", "age"]})


def test_parse_merges_duplicate_table_lines():
    tables, columns = parse_output("head(head_ID)\nhead(age, head_ID)")
    assert tables == ["head"]
    assert columns == {"head": ["head_ID", "age"]}


def test_parse_rejects_prose():
    with pytest.raises(OutputParseError):
        parse_output("The relevant tables are head and department.")


def test_parse_rejects_empty_output():
    with pytest.raises(OutputParseError):
        parse_output("   \n ")
