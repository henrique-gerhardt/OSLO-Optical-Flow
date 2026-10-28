#!/usr/bin/env bash
# Operator sanity checks (run_sanity_checks.py) on the production grids r7/r6/r4:
# default neighbor graph, the fixed graph (gridfix arm), and the encoder checks with the
# trained base checkpoint. Failures are findings, not job errors: always exits 0.
set -uo pipefail

OUT="${OUTPUT_DIR:-/outputs}/sanity_checks"
BASE_CKPT="${BASE_CKPT:-${OUTPUT_DIR:-/outputs}/raft_parity_large/base_s7/oslo_raft.pt}"
mkdir -p "$OUT"

run() {  # run <name> <flags...>
  local name="$1"
  shift
  echo "=== sanity $name ==="
  python run_sanity_checks.py "$@" 2>&1 | grep -v -E "RuntimeWarning|np.stack" | tee "$OUT/$name.txt"
}

run default
run gridfix --sort-cut-deg=-157.5 --fill-missing-neighbors
if [ -f "$BASE_CKPT" ]; then
  run trained_base --checkpoint "$BASE_CKPT"
fi

# whole pipeline on real shards: GT warp per source, OSLO and planar loaders (SO(3),
# roll/flip, resize), and the trained base model (batch, checkpointing, gradients,
# yaw equivariance, static input, vs zero, iterations)
echo "=== pipeline checks ==="
ckpt=()
[ -f "$BASE_CKPT" ] && ckpt=(--checkpoint "$BASE_CKPT")
python run_pipeline_checks.py --shards "${OSLO_SHARDS:-/data/shards}" \
  --sources "${PIPELINE_SOURCES:-flowscape:val,replica360:val,mpf:val,flow360:val}" \
  --pairs "${PIPELINE_PAIRS:-8}" ${ckpt[@]+"${ckpt[@]}"} \
  2>&1 | grep -v -E "RuntimeWarning|np.stack" | tee "$OUT/pipeline.txt"

echo
echo "=== sanity summary (failed checks) ==="
for f in "$OUT"/*.txt; do
  echo "--- $(basename "$f" .txt)"
  grep -E "^FAIL|^sanity:|^pipeline:" "$f"
done
exit 0
