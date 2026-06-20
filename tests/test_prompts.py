from sqlgen.prompts.pruner import (
    PROMPT_VERSION,
    build_messages,
    format_target,
    prompt_hash,
)


def test_build_messages_inference():
    messages = build_messages("How many heads?", "table head (head_ID number)")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "How many heads?" in messages[1]["content"]
    assert "table head (head_ID number)" in messages[1]["content"]


def test_build_messages_training_includes_target():
    target = format_target(["head"], {"head": ["head_ID", "age"]})
    messages = build_messages("q", "schema", target)
    assert messages[-1] == {"role": "assistant", "content": "head(head_ID, age)"}


def test_format_target_multiple_tables_preserves_order():
    target = format_target(
        ["singer", "singer_in_concert"],
        {"singer_in_concert": ["Singer_ID"], "singer": ["Singer_ID", "Name"]},
    )
    assert target == "singer(Singer_ID, Name)\nsinger_in_concert(Singer_ID)"


def test_format_target_table_without_columns():
    assert format_target(["head"], {}) == "head()"


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 16
    int(prompt_hash(), 16)  # hex


def test_prompt_version_set():
    assert PROMPT_VERSION
