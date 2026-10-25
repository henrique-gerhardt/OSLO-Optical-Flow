#!/usr/bin/env bash
# Checks the flow360 shards on this machine for the forward-flow sign defect
# (memory flow360-convention-defect). Frozen RAFT-large scores flow360:test per
# direction against the GT as stored (identity) and sign-flipped (negated). Correct
# shards: identity wins both directions by about 120 points.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}/flow360_convention_check"
SHARDS="${OSLO_SHARDS:-/data/shards}"
mkdir -p "$OUT"
for dir in forward backward; do
  for gt in identity negated; do
    d="$OUT/${dir}_${gt}"
    [ -f "$d/raft_metrics.json" ] && { echo "skip $d (done)"; continue; }
    echo "=== flow360:test $dir, GT $gt ==="
    python run_raft_shard_baseline.py --shards "$SHARDS" --sources flow360:test \
      --model raft_large --weights default --resolution 6 --geodesic-metric haversine \
      --directions "$dir" --gt-transform "$gt" --device "${DEVICE:-cuda}" --batch-size 1 \
      --max-pairs "${MAX_PAIRS:-300}" --output-dir "$d" < /dev/null
  done
done

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print("\n[flow360:test] frozen RAFT-large, global improvement over zero flow, %")
print(f"{'direction':<10}{'identity':>10}{'negated':>10}  verdict")
for d in ("forward", "backward"):
    v = {g: json.loads((root / f"{d}_{g}" / "raft_metrics.json").read_text())["metrics"]
         ["global_improvement_pct"] for g in ("identity", "negated")}
    ok = "stored convention correct" if v["identity"] > v["negated"] else "SIGN-INVERTED"
    print(f"{d:<10}{v['identity']:>10.1f}{v['negated']:>10.1f}  {ok}")
PY
