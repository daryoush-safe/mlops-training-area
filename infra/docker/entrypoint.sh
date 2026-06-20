#!/bin/sh
set -eu

# Re-sync in case the bind mount brought a newer lockfile than the image layer.
# UV_SYNC_ARGS lets the training image pull in its extra dependencies.
uv sync --frozen --no-dev ${UV_SYNC_ARGS:-} >/dev/null

# Point the DVC remote at the in-network MinIO endpoint without touching the
# committed .dvc/config (which defaults to localhost for host runs).
if [ -n "${DVC_REMOTE_ENDPOINT:-}" ]; then
    dvc remote modify --local minio endpointurl "$DVC_REMOTE_ENDPOINT"
fi

# The repo is bind-mounted with host ownership; without this, git (used for
# MLflow lineage tags) refuses to read it when running as root.
if command -v git >/dev/null 2>&1; then
    git config --global --add safe.directory /app
fi

exec "$@"
