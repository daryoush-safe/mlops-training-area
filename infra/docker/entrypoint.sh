#!/bin/sh
set -eu

# Re-sync in case the bind mount brought a newer lockfile than the image layer.
uv sync --frozen --no-dev >/dev/null

# Point the DVC remote at the in-network MinIO endpoint without touching the
# committed .dvc/config (which defaults to localhost for host runs).
if [ -n "${DVC_REMOTE_ENDPOINT:-}" ]; then
    dvc remote modify --local minio endpointurl "$DVC_REMOTE_ENDPOINT"
fi

exec "$@"
