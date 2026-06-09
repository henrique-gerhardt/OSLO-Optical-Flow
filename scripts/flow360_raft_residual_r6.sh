#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${RESOLUTION:-6}"
DIRECTION="${DIRECTION:-forward}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/raft_residual_r${RESOLUTION}_${DIRECTION}}"
RAFT_CACHE_DIR="${RAFT_CACHE_DIR:-/outputs/raft_cache}"

args=(
  python run_flow360_raft_residual.py
  --data-root "${FLOW360_ROOT:-/data/flow360}"
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}"
  --raft-cache-dir "${RAFT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --train-split "${TRAIN_SPLIT:-train}"
  --val-split "${VAL_SPLIT:-test}"
  --direction "${DIRECTION}"
  --resolution "${RESOLUTION}"
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE:-1}"
  --steps "${STEPS:-3000}"
  --lr "${LR:-1e-4}"
  --hidden-channels "${HIDDEN_CHANNELS:-48}"
  --residual-max-rad "${RESIDUAL_MAX_RAD:-0.05}"
  --residual-reg-weight "${RESIDUAL_REG_WEIGHT:-0.01}"
  --pole-residual-reg-weight "${POLE_RESIDUAL_REG_WEIGHT:-0.0}"
  --active-thresholds-deg "${ACTIVE_THRESHOLDS_DEG:-0.25,0.5,1.0}"
  --target-quantile-max-samples "${TARGET_QUANTILE_MAX_SAMPLES:-2000000}"
  --seed "${SEED:-7}"
  --num-workers "${NUM_WORKERS:-4}"
  --log-every "${LOG_EVERY:-50}"
)

if [[ "${AMP:-1}" == "1" ]]; then
  args+=(--amp)
fi

if [[ -n "${MAX_TRAIN_PAIRS:-}" ]]; then
  args+=(--max-train-pairs "${MAX_TRAIN_PAIRS}")
fi

if [[ -n "${MAX_VAL_PAIRS:-}" ]]; then
  args+=(--max-val-pairs "${MAX_VAL_PAIRS}")
fi

"${args[@]}"
