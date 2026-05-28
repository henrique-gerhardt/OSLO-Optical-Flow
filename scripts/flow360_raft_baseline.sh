#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${RESOLUTION:-6}"
DIRECTION="${DIRECTION:-forward}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/raft_r${RESOLUTION}_${DIRECTION}}"
export TORCH_HOME="${TORCH_HOME:-/models/torch}"

args=(
  python run_erp_raft_baseline.py
  --data-root "${FLOW360_ROOT:-/data/flow360}"
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT:-test}"
  --direction "${DIRECTION}"
  --resolution "${RESOLUTION}"
  --model "${RAFT_MODEL:-raft_large}"
  --weights "${RAFT_WEIGHTS:-default}"
  --flow-transform "${RAFT_FLOW_TRANSFORM:-identity}"
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE:-1}"
  --flow-scale "${FLOW_SCALE:-1.0}"
  --active-thresholds-deg "${ACTIVE_THRESHOLDS_DEG:-0.25,0.5,1.0}"
  --target-quantile-max-samples "${TARGET_QUANTILE_MAX_SAMPLES:-2000000}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  args+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ "${SAVE_PREDICTIONS:-0}" == "1" ]]; then
  args+=(--save-predictions)
fi

"${args[@]}"
