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
  --steps "${STEPS:-2000}" \
  --lr "${LR:-1e-4}" \
  --hidden-channels "${HIDDEN_CHANNELS:-48}" \
  --feature-channels "${FEATURE_CHANNELS:-32}" \
  --max-flow-rad "${MAX_FLOW_RAD:-1.2}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --log-every "${LOG_EVERY:-50}"
