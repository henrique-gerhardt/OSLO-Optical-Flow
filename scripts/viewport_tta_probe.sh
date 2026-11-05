#!/usr/bin/env bash
# Plan A + probe B (docs/plans/VIEWPORT_TTA.md):
#   1. viewport EPE known-answer checks (run_viewport_smoke.py)
#   2. seam check: node targets corrupted by unwrapped GT du, per source
#   3. per-pair spherical refinement of an OSLO checkpoint on flowscape:test
# Eval only. The viewport benchmark rows run separately (benchmark_flowscape.sh, TAG=viewport).
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
CKPT="${CKPT:-$OUT/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt}"
SRC="${SOURCE:-flowscape:test}"
PAIRS="${MAX_PAIRS:-100}"
STRIDE="${PAIR_STRIDE:-13}"
DEST="$OUT/tta_probe${TAG:+_$TAG}"
mkdir -p "$DEST"

echo "=== viewport smoke ==="
python run_viewport_smoke.py 2>&1 | grep -v -E "RuntimeWarning|np.stack"

echo "=== seam target check ==="
python run_seam_target_check.py --shards "$SHARDS" \
  --sources "${SEAM_SOURCES:-flowscape:train,flowscape:test,chairs360:train,replica360:train}" \
  --max-pairs 50 2>&1 | grep -v -E "RuntimeWarning|np.stack" | tee "$DEST/seam_check.txt"

# VARIANTS: "name|flags" per line
VARIANTS="${VARIANTS:-anchor05_fb_c5|--coarsest-level 5 --anchor-weight 0.05 --occlusion fb
anchor02_fb_c4|--coarsest-level 4 --anchor-weight 0.02 --occlusion fb}"
while IFS='|' read -r name flags; do
  [ -n "$name" ] || continue
  [ -f "$DEST/$name/tta_probe.json" ] && { echo "skip $name (done)"; continue; }
  echo "=== tta probe $name on $SRC ($PAIRS pairs, stride $STRIDE) ==="
  # shellcheck disable=SC2086
  python run_tta_refine_probe.py --shards "$SHARDS" --sources "$SRC" \
    --init-checkpoint "$CKPT" --pyramid-cache "$OUT/pyramid_cache" \
    --max-pairs "$PAIRS" --pair-stride "$STRIDE" --device "${DEVICE:-cuda}" \
    --smooth-weights "${SMOOTH_WEIGHTS:-0.05,0.2,0.5}" \
    --output-dir "$DEST/$name" $flags < /dev/null 2>&1 \
    | grep -v -E "RuntimeWarning|np.stack" | tee "$DEST/$name.log"
done <<< "$VARIANTS"
