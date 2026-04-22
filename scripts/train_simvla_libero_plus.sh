#!/usr/bin/env bash
# Fine-tune SimVLA-LIBERO on LIBERO-Plus Task A + Task B.
#
# Hyperparameters chosen from SimVLA paper Table 7 (p.13) + Table 6 ablation (p.8):
#   - LR 5e-5 (below pretrain 2e-4; paper's 5e-5 cell trained cleanly at 90.6%).
#   - VLM LR multiplier 0.1 (paper flagged 1.0 → collapses to 44.2%).
#   - freeze_steps 0 + warmup_steps 0 (we RESUME an already-trained ckpt; re-freezing
#     the VLM for 1000 steps wastes compute).
#   - num_actions 10 (paper: LIBERO optimum).
#   - Large arch flags (hidden 1024 / depth 24 / heads 16) — shipped SimVLA-LIBERO
#     ckpt is the Large variant (config.json:1).
#   - batch_size 4 per GPU × 4 GPUs × grad_accum 2 = global 32. Paper used 256 for
#     pretrain; for a 3k-step fine-tune on ~700 demos this is sufficient.
#   - Data shuffling stays on (paper: off → 9.9% collapse).
#
# Run inside bigenlight/simvla-train:latest:
#   conda activate simvla
#   bash /app/scripts/train_simvla_libero_plus.sh
set -euo pipefail

cd /app
source /opt/conda/etc/profile.d/conda.sh
conda activate simvla
export PYTHONPATH="/app:${PYTHONPATH:-}"

RESUME_CKPT="${RESUME_CKPT:-/app/SimVLA-LIBERO}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/runs/simvla_libero_plus_ft}"
META_JSON="${META_JSON:-/app/datasets/metas/libero_plus_taskAB.json}"
NORM_STATS="${NORM_STATS:-/app/norm_stats/libero_plus_norm.json}"

# If task-specific norm stats weren't computed yet, fall back to the pretrain ones.
if [[ ! -f "$NORM_STATS" ]]; then
    echo "[train] norm_stats at $NORM_STATS missing — falling back to libero_norm.json"
    NORM_STATS=/app/norm_stats/libero_norm.json
fi

NGPU="${NGPU:-4}"
BS_PER_GPU="${BS_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ITERS="${ITERS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LR="${LR:-5e-5}"
LR_COEF="${LR_COEF:-0.1}"

echo "=============================================="
echo "SimVLA LIBERO-Plus Fine-Tune"
echo "   resume:    $RESUME_CKPT"
echo "   output:    $OUTPUT_DIR"
echo "   meta:      $META_JSON"
echo "   norm:      $NORM_STATS"
echo "   ngpu:      $NGPU"
echo "   bs/gpu:    $BS_PER_GPU (× grad_accum $GRAD_ACCUM → eff $(( BS_PER_GPU * NGPU * GRAD_ACCUM )))"
echo "   iters:     $ITERS  (save every $SAVE_EVERY)"
echo "   lr:        $LR (vlm mult $LR_COEF)"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# NOTE: we DO pass --models (loads weights from SimVLA-LIBERO) but do NOT pass
# --resume. --resume additionally loads state.json's `global_step` — the shipped
# ckpt was saved at step 150000, which would cause our short fine-tune (iters
# 3000) to exit immediately. Without --resume we load weights only and count
# from step 0, which is the correct fine-tune semantics.
accelerate launch \
    --num_processes "$NGPU" \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_smolvlm.py \
        --models "$RESUME_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --train_metas_path "$META_JSON" \
        --smolvlm_model_path HuggingFaceTB/SmolVLM-500M-Instruct \
        --action_mode libero_joint \
        --norm_stats_path "$NORM_STATS" \
        --hidden_size 1024 --depth 24 --num_heads 16 \
        --image_size 384 --num_actions 10 \
        --batch_size "$BS_PER_GPU" --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LR" --learning_coef "$LR_COEF" \
        --iters "$ITERS" --warmup_steps 0 --freeze_steps 0 \
        --save_interval "$SAVE_EVERY" --log_interval 10 \
        --max_grad_norm 1.0 --num_workers 4
