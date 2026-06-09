#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${RESOLUTION:-6}"
DIRECTION="${DIRECTION:-forward}"
SPLIT="${SPLIT:-test}"
CACHE_DIR="${RAFT_CACHE_DIR:-/outputs/raft_cache}"
export TORCH_HOME="${TORCH_HOME:-/models/torch}"

args=(
  python run_flow360_cache_raft.py
  --data-root "${FLOW360_ROOT:-/data/flow360}"
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}"
  --cache-dir "${CACHE_DIR}"
  --split "${SPLIT}"
  --direction "${DIRECTION}"
  --resolution "${RESOLUTION}"
  --model "${RAFT_MODEL:-raft_large}"
  --weights "${RAFT_WEIGHTS:-default}"
  --flow-transform "${RAFT_FLOW_TRANSFORM:-negated}"
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE:-1}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  args+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

"${args[@]}"
