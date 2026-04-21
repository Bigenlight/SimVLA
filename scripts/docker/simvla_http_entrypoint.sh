#!/usr/bin/env bash
# Entrypoint: launch the SimVLA FastAPI server from the bind-mounted /app tree.

set -euo pipefail

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

exec python scripts/serve_simvla_http.py \
    --host 0.0.0.0 \
    --port "${SIMVLA_HTTP_PORT:-8700}" \
    --checkpoint /checkpoint \
    --norm-stats /app/norm_stats/libero_norm.json \
    ${SIMVLA_HTTP_ARGS:-}
