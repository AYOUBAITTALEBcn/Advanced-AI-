#!/usr/bin/env bash
# Full pipeline: download -> train 3 main models -> eval -> ablation grid -> figures.
# Run from the repo root:  bash scripts/run_all.sh [--smoke] [--gpu_cache] [--resume]
# --smoke: tiny models, ~500 iters each, for an end-to-end sanity pass.
# --gpu_cache: precompute training patches straight onto the GPU instead of
#   using the CPU DataLoader -- ~20x faster on this hardware, since the
#   default num_workers=0 (kept low deliberately for RAM headroom) makes the
#   DataLoader path decode a full-res image per sample. See README.
# --resume: pick up each model from checkpoints/<run>/last.pth if present,
#   instead of restarting at iter 0 -- safe to pass every time, a no-op when
#   there's nothing to resume. Re-running this script after an interrupted
#   (crashed/killed) full run should always include --resume, or already
#   -completed models will retrain from scratch too (the training loop has
#   no "already done" check beyond the checkpoint's saved iter count).
set -euo pipefail

SMOKE=""
GPU_CACHE=""
RESUME=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="--smoke" ;;
    --gpu_cache) GPU_CACHE="--gpu_cache" ;;
    --resume) RESUME="--resume" ;;
  esac
done

PY="${PYTHON:-python}"

echo "=== 1/5 download datasets ==="
$PY -m data.download

echo "=== 2/5 train main models ==="
for cfg in cnn swinir vit; do
  EXTRA="$GPU_CACHE $RESUME"
  # SwinIR is the deepest model (24 transformer blocks): needs gradient
  # checkpointing to fit a 6GB GPU at its configured batch size.
  if [[ "$cfg" == "swinir" ]]; then EXTRA="$EXTRA --grad_checkpoint"; fi
  $PY -m engine.train --config "configs/${cfg}.yaml" $SMOKE $EXTRA
done

echo "=== 3/5 evaluate ==="
BUDGET="lite"; SUFFIX=""
if [[ -n "$SMOKE" ]]; then BUDGET="tiny"; SUFFIX="_smoke"; fi
$PY -m engine.eval --config configs/cnn.yaml --bilinear
for m in cnn swinir vit; do
  $PY -m engine.eval --run_name "${m}_${BUDGET}${SUFFIX}"
done

echo "=== 4/5 PE x window ablation ==="
$PY scripts/run_ablation.py $SMOKE $GPU_CACHE $RESUME

echo "=== 5/5 tables + figures ==="
$PY scripts/make_figures.py

echo "done: see results/summary.md and figures/"
