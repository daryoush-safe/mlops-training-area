import json

import pytest

from sqlgen.prompts.presenter import (
    PROMPT_VERSION,
    ChartSpec,
    OutputParseError,
    Presentation,
    build_messages,
    format_result_preview,
    parse_output,
    prompt_hash,
)

COLUMNS = ["month", "revenue"]
ROWS = [["Jan", 100], ["Feb", 140], ["Mar", 220]]


def _payload(chart, answer="Revenue rose over the quarter.") -> str:
    return json.dumps({"chart": chart, "answer": answer})


def test_build_messages_includes_question_and_preview():
    messages = build_messages("How did revenue trend?", COLUMNS, ROWS)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "How did revenue trend?" in user
    assert "month | revenue" in user
    assert "3 row(s)" in user


def test_preview_clips_rows_and_cols():
    cols = [f"c{i}" for i in range(15)]
    rows = [[i] * 15 for i in range(50)]
    preview = format_result_preview(cols, rows, max_rows=5, max_cols=3)
    assert "more column(s) omitted" in preview
    assert "more row(s) omitted" in preview
    assert "c14" not in preview


def test_preview_accepts_dict_rows():
    preview = format_result_preview(COLUMNS, [{"month": "Jan", "revenue": 100}])
    assert "Jan" in preview and "100" in preview


def test_parse_output_valid_chart():
    text = _payload({"type": "bar", "x": "month", "y": ["revenue"], "series": None})
    result = parse_output(text, COLUMNS)
    assert isinstance(result, Presentation)
    assert result.chart == ChartSpec(type="bar", x="month", y=["revenue"])
    assert result.answer


def test_parse_output_null_chart():
    result = parse_output(_payload(None), COLUMNS)
    assert result.chart is None


def test_parse_output_tolerates_json_fence_and_prose():
    text = "Here you go:\n```json\n" + _payload(None) + "\n```"
    assert parse_output(text, COLUMNS).chart is None


def test_parse_output_rejects_unknown_chart_type():
    text = _payload({"type": "donut", "x": "month", "y": ["revenue"], "series": None})
    with pytest.raises(OutputParseError):
        parse_output(text, COLUMNS)


def test_parse_output_rejects_unknown_column():
    text = _payload({"type": "bar", "x": "month", "y": ["profit"], "series": None})
    with pytest.raises(OutputParseError):
        parse_output(text, COLUMNS)


def test_parse_output_rejects_empty_answer():
    text = _payload(None, answer="   ")
    with pytest.raises(OutputParseError):
        parse_output(text, COLUMNS)


def test_parse_output_rejects_non_json():
    with pytest.raises(OutputParseError):
        parse_output("I cannot answer that.", COLUMNS)


def test_prompt_hash_is_stable_hex16():
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 16
    int(prompt_hash(), 16)


def test_prompt_version_set():
    assert PROMPT_VERSION
