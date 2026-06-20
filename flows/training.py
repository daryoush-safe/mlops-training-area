from __future__ import annotations

import subprocess

from prefect import flow, get_run_logger, task

# dvc skips stages that are already up to date
DVC_STAGES = [
    "train",
    "evaluate",
    "register",
]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{output}")
    return output


@task(retries=2, retry_delay_seconds=10)
def dvc_pull() -> None:
    logger = get_run_logger()
    try:
        logger.info(_run(["dvc", "pull", "--allow-missing"]))
    except RuntimeError as e:
        # first run against an empty remote: nothing to pull is fine
        logger.warning("dvc pull skipped: %s", e)


@task
def dvc_repro_stage(stage: str) -> str:
    logger = get_run_logger()
    output = _run(["dvc", "repro", "--single-item", stage])
    logger.info(output)
    return output


@task(retries=2, retry_delay_seconds=10)
def dvc_push() -> None:
    logger = get_run_logger()
    logger.info(_run(["dvc", "push"]))


@flow(name="schema-pruner-training", log_prints=True)
def training_flow(push: bool = True) -> None:
    dvc_pull()
    for stage in DVC_STAGES:
        dvc_repro_stage.with_options(name=f"dvc-{stage}")(stage)
    if push:
        dvc_push()


if __name__ == "__main__":
    training_flow()
