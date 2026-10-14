#!/usr/bin/env bash
# Patch training at production scale, replica360, the round-1 recipe (2000 steps):
#
#   r7_patch   retina r7 / est r4 / sup r6, quarter-sphere patches, batch 8
#              (same nodes per step as the whole-sphere `base` of raft_parity_ab)
#   r8_native  retina r8 / est r5 / sup r8 — RAFT's resolution on a 1024x512 ERP —
#              quarter-sphere patches, 8 samples per optimizer step
#
# The whole-sphere reference is raft_parity_ab/base_s*. See docs/plans/RAFT_PARITY.md §7.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_patch.sh
#
# Samples per optimizer step must stay at 8 (§7.4): if r8_native runs out of memory,
# lower NATIVE_BATCH and raise NATIVE_ACCUM so their product is still 8.
set -euo pipefail

here="$(dirname "$0")"
export ARMS="${ARMS:-base}"
export SEEDS="${SEEDS:-7}"

RUN_NAME=raft_parity_patch/r7_patch \
BATCH_SIZE=8 GRAD_ACCUM=1 \
EXTRA_FLAGS="--patch-coarse-pixels 12" \
  bash "$here/raft_parity_ab.sh"

RUN_NAME=raft_parity_patch/r8_native \
RETINA_RESOLUTION=8 RESOLUTION=8 ESTIMATION_RESOLUTION=5 \
BATCH_SIZE="${NATIVE_BATCH:-4}" GRAD_ACCUM="${NATIVE_ACCUM:-2}" \
EXTRA_FLAGS="--patch-coarse-pixels 48 --patch-bank 24" \
PREFLIGHT_ARMS="$ARMS" \
  bash "$here/raft_parity_ab.sh"
