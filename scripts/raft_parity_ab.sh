#!/usr/bin/env bash
# RAFT-parity A/B on replica360 (run inside the oslo-raft container on the box).
#
# Four arms x seeds, from scratch, replica360:train -> replica360:val, at the A1
# matched-pair recipe (retina r7 / supervision r6 / estimation r4, 2000 steps,
# one-cycle, haversine). Only --flow-unit and --lookup-interp differ between arms.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_ab.sh
#
# Finished runs are skipped, so the script can be re-launched after an interruption.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}/raft_parity_ab"
SEEDS="${SEEDS:-7 11 19}"
ARMS="${ARMS:-base unit bary both}"
STEPS="${STEPS:-2000}"

arm_flags() {
  case "$1" in
    base) echo "" ;;
    unit) echo "--flow-unit node" ;;
    bary) echo "--lookup-interp barycentric" ;;
    both) echo "--flow-unit node --lookup-interp barycentric" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    dir="$OUT/${arm}_s${seed}"
    if [ -f "$dir/oslo_raft_metrics.json" ]; then
      echo "skip $dir (done)"
      continue
    fi
    mkdir -p "$dir"
    echo "=== arm=$arm seed=$seed ==="
    # shellcheck disable=SC2046
    python run_oslo_raft.py \
      --grid healpix --retina \
      --retina-resolution "${RETINA_RESOLUTION:-7}" \
      --resolution "${RESOLUTION:-6}" \
      --estimation-resolution "${ESTIMATION_RESOLUTION:-4}" \
      --device cuda --amp --onecycle \
      --geodesic-metric haversine \
      --shards "${OSLO_SHARDS:-/data/shards}" \
      --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache" \
      --train-sources replica360:train --val-sources replica360:val \
      --steps "$STEPS" --seed "$seed" \
      --batch-size "${BATCH_SIZE:-2}" --num-workers "${NUM_WORKERS:-4}" \
      --eval-every "${EVAL_EVERY:-500}" --log-every "${LOG_EVERY:-100}" \
      --output-dir "$dir" \
      $(arm_flags "$arm") 2>&1 | tee "$dir/train.log"
  done
done

python - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
keys = ["global_improvement_pct", "poles_improvement_pct", "equator_improvement_pct", "global_geo_deg"]
rows = {}
for path in sorted(out.glob("*_s*/oslo_raft_metrics.json")):
    arm, seed = path.parent.name.rsplit("_s", 1)
    rows.setdefault(arm, {})[seed] = json.loads(path.read_text())["metrics"]
print(f"{'arm':6s} {'seeds':10s} " + " ".join(f"{k:>28s}" for k in keys))
for arm, by_seed in rows.items():
    cells = []
    for k in keys:
        vals = [m[k] for m in by_seed.values()]
        spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"{statistics.mean(vals):>19.2f} ±{spread:>6.2f}")
    print(f"{arm:6s} {','.join(sorted(by_seed)):10s} " + " ".join(f"{c:>28s}" for c in cells))
PY
