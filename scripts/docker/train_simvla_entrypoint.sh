#!/usr/bin/env bash
# Entrypoint for bigenlight/simvla-train: bind-mounts are only present at
# runtime, so we do the LIBERO-Plus editable install here (once per container
# start, idempotent — pip skips if already installed).
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

# --- LIBERO-Plus editable install into the `libero` env --------------------- #
LIBERO_PLUS_SRC=/libero_plus
if [[ -d "$LIBERO_PLUS_SRC" && -f "$LIBERO_PLUS_SRC/setup.py" ]]; then
    if ! conda run -n libero python -c "import libero" &>/dev/null; then
        echo "[entrypoint] installing LIBERO-Plus (editable) into libero env..."
        conda run -n libero pip install -e "$LIBERO_PLUS_SRC"
    else
        echo "[entrypoint] libero env: LIBERO-Plus already installed"
    fi
else
    echo "[entrypoint] WARNING: $LIBERO_PLUS_SRC/setup.py not found — skipping editable install."
fi

# --- pre-create libero config.yaml (skip interactive first-import prompt) --- #
mkdir -p /root/.libero
if [[ ! -f /root/.libero/config.yaml ]]; then
    cat > /root/.libero/config.yaml <<YAML
benchmark_root: /libero_plus/libero/libero
bddl_files: /libero_plus/libero/libero/bddl_files
init_states: /libero_plus/libero/libero/init_files
datasets: /libero_plus/libero/datasets
assets: /libero_plus/libero/libero/assets
YAML
fi

# --- hand off to the user's command (default: bash shell) ------------------- #
if [[ $# -eq 0 ]]; then
    exec bash -l
else
    exec "$@"
fi
