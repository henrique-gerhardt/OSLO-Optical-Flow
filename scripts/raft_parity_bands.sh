#!/usr/bin/env bash
# RAFT-parity: error by displacement band, crossed with poles/equator, for a list of
# checkpoints. Eval only, no training. See docs/plans/RAFT_PARITY.md 7.7.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_bands.sh
#
# RUNS: one checkpoint per line, "name|run dir under /outputs|architecture flags".
# The first line is the reference; results land in /outputs/raft_parity_bands.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
BANDS="${BANDS:-0,0.25,0.5,1,2,4,8,16,32,inf}"
SOURCES="${SOURCES:-flowscape:val replica360:val}"
R8="--retina-resolution 8 --resolution 8 --estimation-resolution 5"
RUNS="${RUNS:-r7_whole|raft_parity_large/base_s7|--retina-resolution 7 --resolution 6 --estimation-resolution 4
r8_native|raft_parity_native_mix/base_s7|$R8}"

eval_ckpt() {
  local name="$1" ckpt="$2" src="$3"
  shift 3
  local metrics="$OUT/raft_parity_bands/${name}_${src//:/_}.json"
  mkdir -p "$OUT/raft_parity_bands"
  if [ -f "$metrics" ]; then
    echo "skip $metrics (done)"
    return
  fi
  echo "=== $name on $src ==="
  python run_oslo_raft.py \
    --grid healpix --retina --device cuda --amp \
    --geodesic-metric haversine \
    --shards "${OSLO_SHARDS:-/data/shards}" \
    --pyramid-cache "$OUT/pyramid_cache" \
    --train-sources "$src" --val-sources "$src" \
    --steps 1 --seed 7 --eval-only --init-checkpoint "$ckpt" \
    --num-workers "${NUM_WORKERS:-4}" \
    --motion-bands-deg "$BANDS" --motion-band-regions poles,equator \
    --output-dir "$OUT/raft_parity_bands/$name" --metrics-out "$metrics" "$@" < /dev/null
}

# A run whose checkpoint is missing (its training failed) is left out of the tables.
names=()
while IFS='|' read -r name dir _; do
  [ -n "$name" ] || continue
  if [ -f "$OUT/$dir/oslo_raft.pt" ]; then
    names+=("$name")
  else
    echo "missing checkpoint $OUT/$dir/oslo_raft.pt, leaving $name out"
  fi
done <<< "$RUNS"
for src in $SOURCES; do
  while IFS='|' read -r name dir flags; do
    [ -f "$OUT/$dir/oslo_raft.pt" ] || continue
    # shellcheck disable=SC2086
    eval_ckpt "$name" "$OUT/$dir/oslo_raft.pt" "$src" $flags
  done <<< "$RUNS"
done

python - "$OUT/raft_parity_bands" "$SOURCES" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path

root, sources, names = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
for src in sources:
    tag = src.replace(":", "_")
    runs = {n: json.loads((root / f"{n}_{tag}.json").read_text())["metrics"] for n in names}
    ref = runs[names[0]]
    bands = [k[: -len("_frac")] for k in ref if k.startswith("band_") and k.endswith("_frac")]
    for region, label in (("", "all"), ("poles_", "poles"), ("equator_", "equator")):
        print(f"\n[{src} / {label}]  improvement over zero flow, % (fraction of nodes in band)")
        print(f"{'band (deg)':<14}" + "".join(f"{n:>22}" for n in names))
        for band in bands:
            row = f"{band[5:].replace('_', ' '):<14}"
            for n in names:
                m, key = runs[n], f"{region}{band}"
                if f"{key}_improvement_pct" in m:
                    row += f"{m[key + '_improvement_pct']:>14.1f} ({m[key + '_frac']:.3f})"
                else:
                    row += f"{'-':>22}"
            print(row)
PY
