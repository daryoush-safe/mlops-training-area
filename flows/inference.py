from __future__ import annotations

import argparse
import json
import logging

from sqlgen.orchestration.service import InferenceService
from sqlgen.orchestration.state import InferenceState

log = logging.getLogger(__name__)


def run_inference(question: str, db_id: str, *, params_path: str = "params.yaml") -> InferenceState:
    service = InferenceService(params_path=params_path)
    return service.run(question, db_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the LangGraph text-to-SQL inference flow")
    parser.add_argument("--question", required=True)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    result = run_inference(args.question, args.db_id, params_path=args.params)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
