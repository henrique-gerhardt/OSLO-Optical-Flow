#!/usr/bin/env bash
# RAFT-parity: what training without (or with less) SO(3) augmentation costs in rotation
# robustness. Each checkpoint is scored upright and with every val pair randomly rotated
# (--val-so3-prob 1; same seed and worker count, so every checkpoint sees the same
# rotations). Eval only. See docs/plans/RAFT_PARITY.md 8.
#
# RUNS: one checkpoint per line, "name|run dir under /outputs|architecture flags".
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
SOURCES="${SOURCES:-flowscape:val replica360:val}"
R7="--retina-resolution 7 --resolution 6 --estimation-resolution 4"
RUNS="${RUNS:-base|raft_parity_large/base_s7|$R7
noso3|raft_parity_large/noso3_s7|$R7}"
DEST="$OUT/raft_parity_rotval"
mkdir -p "$DEST"

names=()
while IFS='|' read -r name dir flags; do
  [ -n "$name" ] || continue
  ckpt="$OUT/$dir/oslo_raft.pt"
  if [ ! -f "$ckpt" ]; then
    echo "missing checkpoint $ckpt, leaving $name out"
    continue
  fi
  names+=("$name")
  for src in $SOURCES; do
    for view in upright rotated; do
      metrics="$DEST/${name}_${src//:/_}_${view}.json"
      [ -f "$metrics" ] && { echo "skip $metrics (done)"; continue; }
      prob=0; [ "$view" = rotated ] && prob=1
      echo "=== $name on $src, $view ==="
      # shellcheck disable=SC2086
      python run_oslo_raft.py \
        --grid healpix --retina --device cuda --amp \
        --geodesic-metric haversine \
        --shards "${OSLO_SHARDS:-/data/shards}" \
        --pyramid-cache "$OUT/pyramid_cache" \
        --train-sources "$src" --val-sources "$src" \
        --steps 1 --seed 7 --eval-only --init-checkpoint "$ckpt" \
        --num-workers 2 --val-so3-prob "$prob" \
        --output-dir "$DEST/$name" --metrics-out "$metrics" $flags < /dev/null
    done
  done
done <<< "$RUNS"

python - "$DEST" "$SOURCES" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path

root, sources, names = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
cols = ("global", "poles", "equator")
for src in sources:
    tag = src.replace(":", "_")
    print(f"\n[{src}]  improvement over zero flow, %, upright -> rotated")
    print(f"{'run':<14}" + "".join(f"{c:>24}" for c in cols))
    for n in names:
        up, rot = (json.loads((root / f"{n}_{tag}_{v}.json").read_text())["metrics"]
                   for v in ("upright", "rotated"))
        row = f"{n:<14}"
        for c in cols:
            k = f"{c}_improvement_pct"
            row += f"{up[k]:>11.1f} -> {rot[k]:>8.1f}"
        print(row)
PY
