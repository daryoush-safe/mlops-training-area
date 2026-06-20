from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset

from sqlgen.data.io import read_jsonl
from sqlgen.prompts.pruner import build_messages, format_target

log = logging.getLogger(__name__)


def load_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for row in read_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def to_chat_dataset(rows: list[dict], tokenizer, max_seq_len: int) -> tuple[Dataset, int]:
    # Drop (not truncate) over-length examples — truncated targets teach the model to emit incomplete selections.
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        target = format_target(row["tables"], row["columns"])
        messages = build_messages(row["question"], row["full_schema_text"], target)
        n_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True))
        if n_tokens > max_seq_len:
            dropped += 1
            continue
        kept.append({"messages": messages})
    if dropped:
        log.warning("dropped %d/%d examples over %d tokens", dropped, len(rows), max_seq_len)
    return Dataset.from_list(kept), dropped
