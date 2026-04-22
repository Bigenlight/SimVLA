# Training container for SimVLA — mounts SimVLA source, LIBERO-Plus source+data,
# and the shipped SimVLA-LIBERO checkpoint. Two conda envs inside:
#
#   simvla  (Python 3.10)  — training deps: torch 2.5.1+cu124, transformers 4.57.3,
#                             peft 0.17.1, flash-attn 2.5.6, wandb, tensorboard.
#                             Used for: scripts/serve_simvla_http.py and
#                             train_smolvlm.py. MATCHES the inference image.
#   libero  (Python 3.8.13) — LIBERO-Plus deps: robosuite 1.4.0, bddl 1.0.1,
#                             transformers 4.21.1, hydra-core 1.2.0. (transformers
#                             pin is incompatible with simvla env — hence the split.)
#                             Used for: pip install -e /libero_plus (drop-in LIBERO)
#                             and our eval driver scripts/eval_taskAB_libero_plus.py.
#
# Bind-mount layout at runtime:
#   /app              ← SimVLA repo (source + scripts + checkpoints under /app/runs)
#   /libero_plus      ← LIBERO-plus repo (source + data/libero_plus_lerobot +
#                                            data/libero_plus_hdf5)
#   /hf_cache         ← ~/.cache/huggingface (SmolVLM base cached)
#
# Build:
#   docker build -t bigenlight/simvla-train:latest \
#       -f scripts/docker/train_simvla.Dockerfile .
#
# Run (typical training):
#   docker run --rm -it --network host --gpus all \
#     -v "$(pwd):/app" \
#     -v "$(pwd)/../LIBERO-plus:/libero_plus" \
#     -v "$HOME/.cache/huggingface:/hf_cache" \
#     -e HF_HOME=/hf_cache \
#     bigenlight/simvla-train:latest \
#     bash -lc "conda activate simvla && bash /app/scripts/train_simvla_libero_plus.sh"

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --- System deps: includes runtime libs for cv2 (simvla side) and robosuite (libero side). #
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential ninja-build curl wget ca-certificates \
        libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libexpat1 libfontconfig1-dev libmagickwand-dev \
        libosmesa6-dev libegl1 libegl-mesa0 libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda for dual env isolation. -------------------------------------- #
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
# miniforge avoids the Anaconda "defaults" channel TOS. All subsequent conda
# operations use conda-forge.
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p ${CONDA_DIR} && rm /tmp/mc.sh && \
    conda config --set always_yes yes --set changeps1 no

# --- simvla env (Python 3.10 — matches the inference image). ---------------- #
# Pinned torch/transformers combo same as bigenlight/simvla-http to keep two
# containers bit-for-bit compatible; PIP_CONSTRAINT guards against silent
# transitive torch upgrades from downstream installs.
RUN conda create -n simvla python=3.10 -q && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "simvla", "/bin/bash", "-c"]

RUN printf "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n" > /etc/pip.constraints.simvla
ENV PIP_CONSTRAINT=/etc/pip.constraints.simvla

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
        transformers==4.57.3 \
        accelerate==1.2.1 \
        peft==0.17.1 \
        safetensors==0.4.5 \
        tokenizers==0.22.1 \
        huggingface-hub==0.36.0

RUN pip install \
        num2words scipy einops timm mmengine pyarrow h5py mediapy av \
        msgpack_numpy websockets json-numpy \
        fastapi "uvicorn[standard]>=0.32" pillow requests \
        wandb tensorboard

# flash-attn build must unset PIP_CONSTRAINT (build backend re-resolves torch).
# Non-fatal: on failure the trainer falls back to SDPA.
RUN pip install packaging && \
    (PIP_CONSTRAINT= pip install flash-attn==2.5.6 --no-build-isolation \
        || echo "[warn] flash-attn build failed; SDPA fallback at train time")

# --- libero env (Python 3.8.13 — LIBERO-Plus's transformers 4.21.1 pin). ---- #
# Kept isolated so it cannot drag older numpy / transformers into the simvla env.
# IMPORTANT: unset PIP_CONSTRAINT here — the simvla-side torch==2.5.1 pin must
# not apply to this env (we want torch compatible with robomimic 0.2.0).
ENV PIP_CONSTRAINT=
SHELL ["/bin/bash", "-c"]
RUN conda create -n libero python=3.8.13 -q && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "libero", "/bin/bash", "-c"]
# libero env is for simulation (MuJoCo/robosuite) + HTTP client only — we do NOT
# train here, so CPU-only torch is sufficient. We also DON'T pin torch — pip's
# resolver fails on torch==1.11.0+cu113 against modern wheels; letting it pick
# a compatible version resolves the conflict while still satisfying robosuite
# / robomimic import-time needs. The second pip pass installs the transitive
# imports that robosuite 1.4.0 / libero require at runtime but don't declare:
# termcolor, matplotlib, easydict, future, cloudpickle, thop, imageio-ffmpeg.
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        numpy==1.22.4 \
        hydra-core==1.2.0 \
        transformers==4.21.1 \
        "robosuite==1.4.0" \
        "bddl==1.0.1" \
        "gym==0.25.2" \
        wand scikit-image \
        json_numpy imageio imageio-ffmpeg requests tqdm pillow opencv-python-headless \
        h5py pyarrow \
        termcolor matplotlib easydict future cloudpickle thop && \
    pip install torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu || \
        pip install torchvision && \
    pip install --no-deps "robomimic==0.2.0" || \
        pip install robomimic && \
    python -m robosuite.scripts.setup_macros || true

# --- Back to base shell and final ENV -------------------------------------- #
SHELL ["/bin/bash", "-c"]

# Clean PIP_CONSTRAINT so docker exec / interactive shells don't inherit it.
ENV PIP_CONSTRAINT=
ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache

# Source conda on every interactive shell so `conda activate <env>` works.
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc

WORKDIR /app

# At runtime the entrypoint `pip install -e`s LIBERO-Plus into the libero env
# (source is bind-mounted so we can't bake this into the image).
COPY scripts/docker/train_simvla_entrypoint.sh /usr/local/bin/train_simvla_entrypoint.sh
RUN chmod +x /usr/local/bin/train_simvla_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/train_simvla_entrypoint.sh"]

# Default: interactive bash. Override with a concrete training/eval command.
CMD ["bash", "-l"]
