# Dockerfile for serving SimVLA over the unified VLA HTTP protocol.
#
# Deps-only image: the SimVLA source tree is bind-mounted at /app at runtime
# (see simvla_http_compose.yml), so `git pull/push` from the host works
# without rebuilding. The SimVLA-LIBERO checkpoint (~3 GB) is bind-mounted
# read-only at /checkpoint so it is downloaded only once on the host.
#
# Build (from the SimVLA repo root):
#   docker build -t bigenlight/simvla-http:latest \
#     -f scripts/docker/serve_simvla_http.Dockerfile .
#
# Run (HF cache + SimVLA-LIBERO checkpoint mounted; the user should first
# clone the checkpoint into the repo root, one-time ~3 GB:
#   git lfs clone https://huggingface.co/YuankaiLuo/SimVLA-LIBERO SimVLA-LIBERO
# ):
#   docker run --rm -it \
#     --network host \
#     --gpus all \
#     -v "$(pwd):/app" \
#     -v "$(pwd)/SimVLA-LIBERO:/checkpoint:ro" \
#     -v "$HOME/.cache/huggingface:/hf_cache" \
#     -e HF_HOME=/hf_cache \
#     bigenlight/simvla-http:latest
#
# Protocol smoke-test without a real checkpoint (server exposes the HTTP API
# backed by zero-action dummy inference — validates the container + protocol
# wiring without downloading SmolVLM weights):
#   docker run --rm -it --network host -v "$(pwd):/app" \
#     -e SIMVLA_HTTP_ARGS="--dummy" bigenlight/simvla-http:latest

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential ninja-build curl ca-certificates \
        python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

# --- PyTorch + SimVLA deps, pinned via PIP_CONSTRAINT ------------------------ #
# Subsequent pip installs must NOT upgrade torch — transformers / accelerate
# happily pull the latest nightly if left unconstrained, which on this box ends
# up as torch 2.11+cu130 (driver too old → cuda_ok=False). Pinning via a global
# constraints file is the only way to make pip keep our torch 2.5.1+cu124.
RUN printf "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n" > /etc/pip.constraints
ENV PIP_CONSTRAINT=/etc/pip.constraints

RUN python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# SimVLA inference deps (pinned per README section 2.1 `simvla` env).
# Intentionally skips tensorflow — not used by the inference path.
RUN python -m pip install \
    transformers==4.57.3 \
    accelerate==1.2.1 \
    peft==0.17.1 \
    safetensors==0.4.5 \
    tokenizers==0.22.1 \
    huggingface-hub==0.36.0

# Additional modeling / utility deps. `num2words` is in SimVLA's requirements.txt;
# the rest are transitive imports the modeling code reaches for (einops, timm,
# scipy, mmengine, pyarrow, h5py, mediapy, av). msgpack_numpy and websockets
# are pulled in for import-compatibility with openpi-flavored modeling modules
# even though the HTTP path itself doesn't touch them.
RUN python -m pip install \
    num2words \
    scipy \
    einops \
    timm \
    mmengine \
    pyarrow \
    h5py \
    mediapy \
    av \
    msgpack_numpy \
    websockets

# HTTP serving deps
RUN python -m pip install \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    pillow \
    requests \
    json-numpy

# --- flash-attn (non-fatal) -------------------------------------------------- #
# Heavy build (~10 min). Failure leaves the image usable with --attn-impl sdpa.
# NOTE: PIP_CONSTRAINT must be unset for this install because flash-attn's
# build backend re-resolves torch in a fresh env, and our cu124 wheel isn't
# visible on the default index — so the constraint would block it. We still
# tolerate the failure and fall back to sdpa at runtime.
RUN python -m pip install packaging && \
    (PIP_CONSTRAINT= python -m pip install flash-attn==2.5.6 --no-build-isolation \
        || echo "[warn] flash-attn install failed; server will fall back to --attn-impl sdpa")

# Runtime shared libs required by opencv-python (transitively imported by
# models/modeling_smolvlm_vla.py). `libxcb1 libgl1 libglib2.0-0` are the
# minimal set; headless servers don't need the full X stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV SIMVLA_HTTP_PORT=8700 \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache
EXPOSE 8700

WORKDIR /app

# Default command: run the entrypoint script from the bind-mounted source tree.
# Nothing from the repo is baked into the image — all code lives in /app so the
# user can `git pull/push` from the host at will.
CMD ["/bin/bash", "/app/scripts/docker/simvla_http_entrypoint.sh"]
