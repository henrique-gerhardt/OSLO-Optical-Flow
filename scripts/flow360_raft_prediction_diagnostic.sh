#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/outputs/raft_r6_forward_debug}"
DIRECTION="${DIRECTION:-forward}"

args=(
  python analyze_raft_predictions.py
  --data-root "${FLOW360_ROOT:-/data/flow360}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT:-test}"
  --direction "${DIRECTION}"
  --flow-scale "${FLOW_SCALE:-1.0}"
  --max-samples "${MAX_SAMPLES:-2000000}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  args+=(--max-pairs "${MAX_PAIRS}")
fi

"${args[@]}"
