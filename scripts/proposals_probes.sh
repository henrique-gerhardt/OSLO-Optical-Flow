#!/usr/bin/env bash
# Minimal experiments for the five proposals of reports/Formulações de fluxo para HEALPix.md
# (experiments/proposals/README.md). Eval-only except P5 (small isolated training).
# PROBES selects a subset, e.g. PROBES="p1 p4". Each probe skips if its JSON exists.
set -uo pipefail

OUT="${OUTPUT_DIR:-/outputs}/proposals_probes${TAG:+_$TAG}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
CKPTS="${CKPTS:-${OUTPUT_DIR:-/outputs}/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s7/oslo_raft.pt}"
PROBES="${PROBES:-p1 p2 p3 p4 p5}"
DEV="${DEVICE:-cuda}"
mkdir -p "$OUT"
cd experiments/proposals
export PYTHONPATH="$(cd ../.. && pwd)"
common=(--shards "$SHARDS" --sources flowscape:test)

run() {  # run <name> <script> <args...>
  local name="$1"; shift
  if [ -f "$OUT/$name/done" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name ==="
  python "$@" --output-dir "$OUT/$name" 2>&1 | grep -v -E "RuntimeWarning|np.stack" | tee "$OUT/$name.log" \
    && touch "$OUT/$name/done" || echo "FAILED $name"
}

for p in $PROBES; do
  case "$p" in
    p1) run p1_rigid_ceiling p1_rigid_ceiling.py "${common[@]}" --stride 5 --max-pairs 280 ;;
    p2) run p2_cascade_probe p2_cascade_probe.py "${common[@]}" --stride 14 --max-pairs 100 ;;
    p3) for ck in $CKPTS; do
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          run "p3_gate_$tag" p3_gate_probe.py "${common[@]}" --stride 13 --max-pairs 106 \
            --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache" --device "$DEV"
        done ;;
    p4) for m in raft_small raft_large; do
          run "p4_faces_${m}_r8" p4_faces_probe.py "${common[@]}" --stride 13 --max-pairs 106 \
            --model "$m" --face-resolution 8 --pad 64 --device "$DEV"
        done ;;
    p5) run p5_waft_toy p5_waft_toy.py "${common[@]}" --stride 13 --max-pairs 106 \
          --train-sources flowscape:train --train-pairs 1000 --steps "${P5_STEPS:-3000}" --device "$DEV" ;;
  esac
done
exit 0
