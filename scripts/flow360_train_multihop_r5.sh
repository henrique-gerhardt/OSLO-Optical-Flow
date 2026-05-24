#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${RESOLUTION:-5}"
COST_NUM_HOPS="${COST_NUM_HOPS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/r${RESOLUTION}_costh${COST_NUM_HOPS}}"

python run_flow360_mvp.py \
  --data-root "${FLOW360_ROOT:-/data/flow360}" \
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --amp \
  --resolution "${RESOLUTION}" \
  --cost-num-hops "${COST_NUM_HOPS}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --steps "${STEPS:-3000}" \
  --lr "${LR:-1e-4}" \
  --hidden-channels "${HIDDEN_CHANNELS:-48}" \
  --feature-channels "${FEATURE_CHANNELS:-32}" \
  --max-flow-rad "${MAX_FLOW_RAD:-1.2}" \
  --loss-motion-weight "${LOSS_MOTION_WEIGHT:-0.0}" \
  --loss-motion-ref-deg "${LOSS_MOTION_REF_DEG:-1.0}" \
  --active-thresholds-deg "${ACTIVE_THRESHOLDS_DEG:-0.25,0.5,1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --log-every "${LOG_EVERY:-50}"
