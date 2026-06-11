from __future__ import annotations

import argparse
import hashlib
import logging

from sqlgen.config import load_params
from sqlgen.data.io import read_json, write_jsonl

log = logging.getLogger(__name__)


def example_id(db_id: str, question: str, query: str) -> str:
    digest = hashlib.sha256(f"{db_id}\x00{question}\x00{query}".encode()).hexdigest()
    return digest[:16]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    splits = params.splits.model_dump()

    for split, files in splits.items():
        seen: set[str] = set()
        rows = []
        for fname in files:
            for entry in read_json(params.paths.raw_dir / fname):
                question = entry["question"].strip()
                query = entry["query"].strip()
                db_id = entry["db_id"]
                if not question or not query:
                    continue
                eid = example_id(db_id, question, query)
                if eid in seen:
                    continue
                seen.add(eid)
                rows.append(
                    {
                        "id": eid,
                        "db_id": db_id,
                        "question": question,
                        "query": query,
                        "source": fname,
                    }
                )
        out = params.paths.interim_dir / "pairs" / f"{split}.jsonl"
        n = write_jsonl(out, rows)
        log.info("split=%s: wrote %d pairs to %s", split, n, out)


if __name__ == "__main__":
    main()
