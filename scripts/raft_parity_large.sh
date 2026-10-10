#!/usr/bin/env bash
# RAFT-parity, top-down: the small model against the RAFT-large layout, both from
# scratch on replica360 + flowscape, scored on flowscape:val and replica360:val.
# See docs/plans/RAFT_PARITY.md section 5.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_large.sh
#
# If the preflight runs out of memory: BATCH_SIZE=1 GRAD_ACCUM=2 (applies to every arm).
set -euo pipefail

export RUN_NAME="${RUN_NAME:-raft_parity_large}"
export TRAIN_SOURCES="${TRAIN_SOURCES:-replica360:train,flowscape:train}"
export VAL_SOURCES="${VAL_SOURCES:-flowscape:val}"
export EXTRA_VAL_SOURCES="${EXTRA_VAL_SOURCES:-replica360:val}"
export ARMS="${ARMS:-base raftfull}"
export PREFLIGHT_ARMS="${PREFLIGHT_ARMS:-raftfull}"
export SEEDS="${SEEDS:-7}"
export STEPS="${STEPS:-10000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"

exec bash "$(dirname "$0")/raft_parity_ab.sh"
