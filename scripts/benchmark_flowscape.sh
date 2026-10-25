#!/usr/bin/env bash
# Published-benchmark table on FlowScape test (1386 of the official 1400 pairs), in the
# protocol of PriOr-Flow's evaluate.py: ERP-pixel EPE over every pixel and SEPE
# (great-circle endpoint distance), All / Equator / Poles split at 45 degrees latitude.
# The released PriOr-RAFT and PanoFlow checkpoints are the reproduction check against
# the published table (EPE 2.33 / 3.38 px, SEPE 3.49 / 4.78). Eval only.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
SRC="${SOURCE:-flowscape:test}"
DEST="$OUT/benchmark_${SRC//:/_}${TAG:+_$TAG}"
PRIOR="${PRIOR_CKPT:-$OUT/prior_weights/FlowScape-final.pth}"
PANOFLOW="${PANOFLOW_CKPT:-$(find "$OUT" -maxdepth 3 -name 'PanoFlow(CSFlow)*.pth' 2>/dev/null | head -1)}"
# ROWS: "name|predictor|checkpoint (empty = none)|run_epe_eval.py flags"
ROWS="${ROWS:-zero|zero||
oracle_r6|oracle||--resolution 6
prior_raft|raft|$PRIOR|--prior-checkpoint $PRIOR
panoflow|raft|$PANOFLOW|--panoflow-checkpoint $PANOFLOW
frozen_raft_large|raft||--model raft_large --weights default
raft_small_so3|raft|$OUT/raft_parity_planar/raft_small_so3_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_so3_s7/raft_erp.pt
raft_large_so3_scale|raft|$OUT/raft_parity_planar_scale/raft_large_so3_s7/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_planar_scale/raft_large_so3_s7/raft_erp.pt
oslo_base|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt
oslo_noso3|oslo|$OUT/raft_parity_large/noso3_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/noso3_s7/oslo_raft.pt
oslo_large_scale|oslo|$OUT/raft_parity_scale/large_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_scale/large_s7/oslo_raft.pt
oslo_r8_cp4|oslo|$OUT/raft_parity_native_mix_cp4/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_native_mix_cp4/base_s7/oslo_raft.pt}"
mkdir -p "$DEST"

names=()
while IFS='|' read -r name predictor ckpt flags; do
  [ -n "$name" ] || continue
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "missing checkpoint '$ckpt', leaving $name out"
    continue
  fi
  names+=("$name")
  dir="$DEST/$name"
  [ -f "$dir/epe_metrics.json" ] && { echo "skip $name (done)"; continue; }
  echo "=== $name ($predictor) on $SRC ==="
  # shellcheck disable=SC2086
  python run_epe_eval.py --shards "$SHARDS" --sources "$SRC" --predictor "$predictor" \
    --device "${DEVICE:-cuda}" --pyramid-cache "$OUT/pyramid_cache" \
    --output-dir "$dir" $flags ${EVAL_FLAGS:-} < /dev/null 2>&1 | tee "$dir.log" \
    || echo "FAILED $name (see $dir.log); the other rows continue"
done <<< "$ROWS"

python - "$DEST" "${names[@]}" <<'PY'
import json, math, sys
from pathlib import Path

root, names = Path(sys.argv[1]), sys.argv[2:]
published = {"prior_raft": (2.33, 3.49), "panoflow": (3.38, 4.78)}
print("\nFlowScape test, PriOr-Flow protocol (all pixels; SEPE in 1e-3 rad; poles = |lat| > 45)")
print(f"{'row':<22}{'EPE all':>9}{'EPE eq':>9}{'EPE pol':>9}{'SEPE all':>10}{'SEPE eq':>9}"
      f"{'SEPE pol':>10}{'pairs':>7}   published EPE / SEPE")
for n in names:
    path = root / n / "epe_metrics.json"
    if not path.exists():
        print(f"{n:<22}  missing {path}")
        continue
    m = json.loads(path.read_text())["metrics"]
    row = f"{n:<22}"
    for r in ("all", "equator", "poles"):
        row += f"{m['official_' + r + '_epe']:>9.2f}"
    for r in ("all", "equator", "poles"):
        row += f"{m['official_' + r + '_sepe'] * 1e3:>{10 if r != 'equator' else 9}.2f}"
    row += f"{m['pairs']:>7}"
    if n in published:
        row += f"   {published[n][0]:.2f} / {published[n][1]:.2f}"
    print(row)
print("\nPublished (PriOr-Flow Table, FlowScape all weathers, EPE / SEPE): TanImg 18.3 / 25.3, "
      "SLOF 7.59 / 5.79, PanoFlow 3.38 / 4.78, PriOr-Flow 2.33 / 3.49")
PY
