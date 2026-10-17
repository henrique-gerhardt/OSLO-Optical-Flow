#!/usr/bin/env bash
# RAFT-parity: error by displacement band, crossed with poles/equator, for the r7 whole-sphere
# checkpoint (raft_parity_large/base_s7) and the r8 native checkpoint
# (raft_parity_native_mix/base_s7). Eval only, no training. See docs/plans/RAFT_PARITY.md 7.7.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_bands.sh
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
BANDS="${BANDS:-0,0.25,0.5,1,2,4,8,16,32,inf}"
SOURCES="${SOURCES:-flowscape:val replica360:val}"

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
    --motion-bands-deg "$BANDS" --motion-band-regions poles,equator \
    --output-dir "$OUT/raft_parity_bands/$name" --metrics-out "$metrics" "$@"
}

for src in $SOURCES; do
  eval_ckpt r7_whole "$OUT/raft_parity_large/base_s7/oslo_raft.pt" "$src" \
    --retina-resolution 7 --resolution 6 --estimation-resolution 4
  eval_ckpt r8_native "$OUT/raft_parity_native_mix/base_s7/oslo_raft.pt" "$src" \
    --retina-resolution 8 --resolution 8 --estimation-resolution 5
done

python - "$OUT/raft_parity_bands" $SOURCES <<'PY'
import json, sys
from pathlib import Path

root, sources = Path(sys.argv[1]), sys.argv[2:]
for src in sources:
    tag = src.replace(":", "_")
    runs = {n: json.loads((root / f"{n}_{tag}.json").read_text())["metrics"]
            for n in ("r7_whole", "r8_native")}
    m7 = runs["r7_whole"]
    bands = [k[: -len("_frac")] for k in m7 if k.startswith("band_") and k.endswith("_frac")]
    print(f"\n[{src}]  improvement over zero flow, % (fraction of nodes in band)")
    print(f"{'band (deg)':<14}" + "".join(f"{r + ' ' + n:>22}" for r in ("all", "poles", "equator")
                                         for n in ("r7", "r8")))
    for band in bands:
        row = f"{band[5:].replace('_', ' '):<14}"
        for region in ("", "poles_", "equator_"):
            for name in ("r7_whole", "r8_native"):
                m = runs[name]
                key = f"{region}{band}"
                if f"{key}_improvement_pct" in m:
                    row += f"{m[key + '_improvement_pct']:>14.1f} ({m[key + '_frac']:.3f})"
                else:
                    row += f"{'-':>22}"
        print(row)
PY
