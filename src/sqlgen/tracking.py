from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

from sqlgen.config import Params
from sqlgen.prompts import pruner as pruner_prompt


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True)
    except FileNotFoundError:  # no git binary (e.g. minimal container)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def dataset_hashes(stage: str = "build_dataset", lock_path: str | Path = "dvc.lock") -> dict:
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    outs = lock.get("stages", {}).get(stage, {}).get("outs", [])
    return {out["path"]: out.get("md5", "") for out in outs}


def log_lineage(params: Params) -> None:
    tags = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))),
        "prompt_version": pruner_prompt.PROMPT_VERSION,
        "prompt_hash": pruner_prompt.prompt_hash(),
        "base_model": params.model.base,
        "base_model_revision": params.model.revision,
    }
    for path, md5 in dataset_hashes().items():
        tags[f"data_md5.{Path(path).name}"] = md5
    mlflow.set_tags(tags)
    mlflow.log_text(
        f"# prompt {pruner_prompt.PROMPT_VERSION} ({pruner_prompt.prompt_hash()})\n\n"
        f"## system\n{pruner_prompt.SYSTEM_PROMPT}\n\n## user\n{pruner_prompt.USER_TEMPLATE}\n",
        "prompt_template.md",
    )


def _flatten(data: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def log_params_sections(params: Params, sections: list[str]) -> None:
    dump = params.model_dump(mode="json")
    flat = {}
    for section in sections:
        flat.update(_flatten({section: dump[section]}))
    mlflow.log_params(flat)
