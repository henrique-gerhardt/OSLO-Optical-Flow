#!/usr/bin/env bash
set -euo pipefail

python run_flow360_mvp.py \
  --data-root "${FLOW360_ROOT:-/data/flow360}" \
  --grid-dir "${OSLO_GRID_DIR:-/data/oslo_data/neighbor_grids}" \
  --output-dir "${OUTPUT_DIR:-/outputs}" \
  --device cuda \
  --amp \
  --resolution "${RESOLUTION:-4}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --max-train-pairs "${MAX_TRAIN_PAIRS:-8}" \
  --max-val-pairs "${MAX_VAL_PAIRS:-8}" \
  --smoke-test
