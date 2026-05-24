#!/usr/bin/env bash
set -euo pipefail

python run_flow360_mvp.py \
  --data-root "${FLOW360_ROOT:-/data/flow360}" \
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}" \
  --output-dir "${OUTPUT_DIR:-/outputs}" \
  --device cuda \
  --amp \
  --resolution "${RESOLUTION:-5}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --steps "${STEPS:-3000}" \
  --lr "${LR:-1e-4}" \
  --hidden-channels "${HIDDEN_CHANNELS:-48}" \
  --feature-channels "${FEATURE_CHANNELS:-32}" \
  --max-flow-rad "${MAX_FLOW_RAD:-1.2}" \
  --loss-motion-weight "${LOSS_MOTION_WEIGHT:-4.0}" \
  --loss-motion-ref-deg "${LOSS_MOTION_REF_DEG:-1.0}" \
  --active-thresholds-deg "${ACTIVE_THRESHOLDS_DEG:-0.25,0.5,1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --log-every "${LOG_EVERY:-50}"
