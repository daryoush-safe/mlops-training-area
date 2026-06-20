import pytest

from sqlgen.prompts.sqlgen import (
    PROMPT_VERSION,
    OutputParseError,
    build_messages,
    extract_sql,
    prompt_hash,
)


def test_build_messages_inference():
    messages = build_messages("How many singers?", "table singer (Singer_ID number)")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "How many singers?" in messages[1]["content"]
    assert "table singer (Singer_ID number)" in messages[1]["content"]


def test_build_messages_training_includes_target():
    messages = build_messages("q", "schema", "SELECT count(*) FROM singer")
    assert messages[-1] == {"role": "assistant", "content": "SELECT count(*) FROM singer"}


def test_build_messages_retry_includes_error_and_prev_sql():
    messages = build_messages(
        "q", "full schema", prev_sql="SELECT * FROM nope", error="no such table: nope"
    )
    content = messages[1]["content"]
    assert "SELECT * FROM nope" in content
    assert "no such table: nope" in content
    assert "full schema" in content


def test_extract_sql_plain():
    assert extract_sql("SELECT name FROM singer") == "SELECT name FROM singer"


def test_extract_sql_strips_fence_and_prose():
    out = extract_sql("Here is the query:\n```sql\nSELECT name FROM singer;\n```")
    assert out == "SELECT name FROM singer"


def test_extract_sql_keeps_only_first_statement():
    assert extract_sql("SELECT 1; DROP TABLE x") == "SELECT 1"


def test_extract_sql_raises_without_sql():
    with pytest.raises(OutputParseError):
        extract_sql("I cannot answer that question.")


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 16
    int(prompt_hash(), 16)  # hex


def test_prompt_version_set():
    assert PROMPT_VERSION
