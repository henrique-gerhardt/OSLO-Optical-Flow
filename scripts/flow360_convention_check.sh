#!/usr/bin/env bash
# Checks a source's shards on this machine for a flow sign defect (memory
# flow360-convention-defect). Frozen RAFT-large scores SOURCE (default flow360:test) per
# direction against the GT as stored (identity) and sign-flipped (negated). Correct
# shards: identity wins every direction by a wide margin. Sources without a direction
# field in their records need DIRECTIONS=both.
set -euo pipefail

SOURCE="${SOURCE:-flow360:test}"
DIRECTIONS="${DIRECTIONS:-forward backward}"
OUT="${OUTPUT_DIR:-/outputs}/${SOURCE%%:*}_convention_check"
SHARDS="${OSLO_SHARDS:-/data/shards}"
mkdir -p "$OUT"
for dir in $DIRECTIONS; do
  for gt in identity negated; do
    d="$OUT/${dir}_${gt}"
    [ -f "$d/raft_metrics.json" ] && { echo "skip $d (done)"; continue; }
    echo "=== $SOURCE $dir, GT $gt ==="
    python run_raft_shard_baseline.py --shards "$SHARDS" --sources "$SOURCE" \
      --model raft_large --weights default --resolution 6 --geodesic-metric haversine \
      --directions "$dir" --gt-transform "$gt" --device "${DEVICE:-cuda}" --batch-size 1 \
      --max-pairs "${MAX_PAIRS:-300}" --output-dir "$d" < /dev/null
  done
done

python - "$OUT" "$SOURCE" $DIRECTIONS <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"\n[{sys.argv[2]}] frozen RAFT-large, global improvement over zero flow, %")
print(f"{'direction':<10}{'identity':>10}{'negated':>10}  verdict")
for d in sys.argv[3:]:
    v = {g: json.loads((root / f"{d}_{g}" / "raft_metrics.json").read_text())["metrics"]
         ["global_improvement_pct"] for g in ("identity", "negated")}
    ok = "stored convention correct" if v["identity"] > v["negated"] else "SIGN-INVERTED"
    print(f"{d:<10}{v['identity']:>10.1f}{v['negated']:>10.1f}  {ok}")
PY
