#!/usr/bin/env bash
# ERP-pixel EPE table: zero flow, the grid floor, frozen RAFT-small and RAFT-large,
# and OSLO-RAFT-R checkpoints, on the same pairs (run inside the oslo-raft container).
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_epe.sh
#
# CHECKPOINTS is a space-separated list of name=path; the default scores the three
# round-1 `base` seeds. Finished rows are skipped.
set -euo pipefail

ROOT="${OUTPUT_DIR:-/outputs}"
OUT="$ROOT/${RUN_NAME:-raft_parity_epe}"
SOURCES="${SOURCES:-replica360:val}"
CHECKPOINTS="${CHECKPOINTS:-base_s7=$ROOT/raft_parity_ab/base_s7/oslo_raft.pt base_s11=$ROOT/raft_parity_ab/base_s11/oslo_raft.pt base_s19=$ROOT/raft_parity_ab/base_s19/oslo_raft.pt}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/torch_home}"

# row <source> <name> <predictor flags...>
row() {
  local src="$1" name="$2"
  shift 2
  local dir="$OUT/${src//:/_}/$name"
  if [ -f "$dir/epe_metrics.json" ]; then
    echo "skip $dir (done)"
    return
  fi
  echo "=== $src $name ==="
  python run_epe_eval.py \
    --shards "${OSLO_SHARDS:-/data/shards}" --sources "$src" \
    --resolution "${RESOLUTION:-6}" \
    --retina-resolution "${RETINA_RESOLUTION:-7}" \
    --estimation-resolution "${ESTIMATION_RESOLUTION:-4}" \
    --pyramid-cache "$ROOT/pyramid_cache" \
    --device cuda --output-dir "$dir" "$@"
}

for src in $SOURCES; do
  row "$src" zero --predictor zero
  row "$src" grid_floor --predictor oracle
  row "$src" raft_small --predictor raft --model raft_small
  row "$src" raft_large --predictor raft --model raft_large
  for entry in $CHECKPOINTS; do
    name="${entry%%=*}"
    path="${entry#*=}"
    if [ -f "$path" ]; then
      row "$src" "oslo_$name" --predictor oslo --init-checkpoint "$path"
    else
      echo "missing checkpoint, row skipped: $path"
    fi
  done
done

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
regions = ["global", "equator", "poles"]
for src_dir in sorted(p for p in out.iterdir() if p.is_dir()):
    print(f"\n[{src_dir.name}]  EPE in ERP px (improvement over zero flow, %)")
    print(f"{'row':18s} " + " ".join(f"{r:>22s}" for r in regions))
    for path in sorted(src_dir.glob("*/epe_metrics.json")):
        m = json.loads(path.read_text())["metrics"]
        cells = []
        for r in regions:
            epe, gain = m.get(f"{r}_epe_px"), m.get(f"{r}_epe_improvement_pct")
            cells.append("-" if epe is None else
                         f"{epe:9.3f}" + ("" if gain is None else f" ({gain:+6.1f})"))
        print(f"{path.parent.name:18s} " + " ".join(f"{c:>22s}" for c in cells))
PY
