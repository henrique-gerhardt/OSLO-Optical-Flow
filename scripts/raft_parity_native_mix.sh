#!/usr/bin/env bash
# RAFT-parity: native resolution (retina r8 / est r5 / sup r8, quarter-sphere patches)
# on the top-down recipe — replica360 + flowscape, 10 000 steps, scored on flowscape:val
# and replica360:val. Compared against raft_parity_large/base_s7 (r7/r4/r6, whole sphere).
# See docs/plans/RAFT_PARITY.md section 7.6.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_native_mix.sh
set -euo pipefail

export RUN_NAME="${RUN_NAME:-raft_parity_native_mix}"
export TRAIN_SOURCES="${TRAIN_SOURCES:-replica360:train,flowscape:train}"
export VAL_SOURCES="${VAL_SOURCES:-flowscape:val}"
export EXTRA_VAL_SOURCES="${EXTRA_VAL_SOURCES:-replica360:val}"
export ARMS="${ARMS:-base}"
export SEEDS="${SEEDS:-7}"
export STEPS="${STEPS:-10000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export RETINA_RESOLUTION=8 RESOLUTION=8 ESTIMATION_RESOLUTION=5
export BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM="${GRAD_ACCUM:-2}"
export EXTRA_FLAGS="${EXTRA_FLAGS:---patch-coarse-pixels 48 --patch-bank 24}"
export PREFLIGHT_ARMS="${PREFLIGHT_ARMS:-}"

exec bash "$(dirname "$0")/raft_parity_ab.sh"
