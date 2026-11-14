#!/usr/bin/env bash
# Orientation-robustness suite (docs/plans/ORIENTATION_ROBUSTNESS.md, WP1 and WP2). Eval only.
# Every row is scored on the same pairs under each view; the rotation stream is seeded, so
# every row sees the same rotations.
#
# Views (VIEWS):
#   upright  no rotation
#   haar     Haar-uniform SO(3) rotation
#   rollNN   exact circular shift by 0.NN of the width (a yaw with no resampling: lossless control)
#   canon    the haar pairs rotated back with the known rotation (oracle canonization ceiling)
#   canonfull / canonup  the haar pairs rotated back with the rotation ESTIMATED by the canonizer (tilt + heading / tilt only)
#   canonfull2  canonfull with two estimation passes; canonid  the canonizer applied to UPRIGHT pairs (must cost nothing)
#   yaw      rotation about the vertical axis, angle uniform in [0, 180] (keeps gravity: control)
#   tiltN    rotation of exactly N degrees about a random horizontal axis (dose-response)
# ONLY: space-separated row names to keep. SOURCES: dataset:split list. ROWS: "name|run_raft_shard_baseline.py flags"; a row whose
# checkpoint file is missing is left out. A failed cell is reported and the suite moves on.
set -uo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
DEST="$OUT/orientation_suite${TAG:+_$TAG}"
SOURCES="${SOURCES:-flowscape:test}"
VIEWS="${VIEWS:-upright haar yaw tilt15 tilt30 tilt45 tilt90}"
MAX_PAIRS="${MAX_PAIRS:-1500}"
PRIOR="${PRIOR_CKPT:-$OUT/prior_weights/FlowScape-final.pth}"
PANOFLOW="${PANOFLOW_CKPT:-$(find "$OUT" -maxdepth 3 -name 'PanoFlow(CSFlow)*.pth' 2>/dev/null | head -1)}"
P11="$OUT/proposals_probes"
PL="$OUT/raft_parity_planar"
ROWS="${ROWS:-prior_raft|--prior-checkpoint $PRIOR
panoflow|--panoflow-checkpoint $PANOFLOW
frozen_raft_large|--model raft_large --weights default
prior_raft_so3ft|--prior-checkpoint $P11/p11_so3ft_so3/prior_raft_so3.pth
prior_raft_ctrlft|--prior-checkpoint $P11/p11_so3ft_ctrl/prior_raft_ctrl.pth
raft_small_noaug|--model raft_small --tv-checkpoint $PL/raft_small_noaug_s7/raft_erp.pt
raft_small_so3|--model raft_small --tv-checkpoint $PL/raft_small_so3_s7/raft_erp.pt
raft_large_so3_scale|--model raft_large --tv-checkpoint ${PL}_scale/raft_large_so3_s7/raft_erp.pt}"
mkdir -p "$DEST"

view_flags() {
  case "$1" in
    upright) echo "" ;;
    canonfull|canonup) echo "--val-so3-prob 1 --val-so3-uniform --val-so3-canonizer ${CANONIZER:-$OUT/proposals_probes/p12_canonizer/canonizer.pth} --val-so3-canon-mode ${1#canon}" ;;
    canonfull2) echo "--val-so3-prob 1 --val-so3-uniform --val-so3-canonizer ${CANONIZER:-$OUT/proposals_probes/p12_canonizer/canonizer.pth} --val-so3-canon-iters 2" ;;
    canonid) echo "--val-so3-prob 1 --val-so3-axis-mode horizontal --val-so3-fixed-angle-deg 0 --val-so3-canonizer ${CANONIZER:-$OUT/proposals_probes/p12_canonizer/canonizer.pth}" ;;
    canon) echo "--val-so3-prob 1 --val-so3-uniform --val-so3-derotate" ;;   # haar, then rotated back with the known rotation
    haar) echo "--val-so3-prob 1 --val-so3-uniform" ;;
    roll*) echo "--val-roll-frac 0.${1#roll}" ;;   # roll25 / roll50: exact circular shift, no resampling
    yaw) echo "--val-so3-prob 1 --val-so3-axis-mode vertical" ;;
    tilt*) echo "--val-so3-prob 1 --val-so3-axis-mode horizontal --val-so3-fixed-angle-deg ${1#tilt}" ;;
    *) echo "unknown view $1" >&2; return 1 ;;
  esac
}

names=()
while IFS='|' read -r name flags; do
  [ -n "$name" ] || continue
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  ckpt="$(sed -n 's/.*-checkpoint \(.*\)$/\1/p' <<< "$flags")"
  if [[ "$flags" == *-checkpoint* ]] && [ ! -f "$ckpt" ]; then
    echo "missing checkpoint '$ckpt', leaving $name out"
    continue
  fi
  names+=("$name")
  for src in $SOURCES; do
    for view in $VIEWS; do
      dir="$DEST/${name}_${src//:/_}_${view}"
      [ -f "$dir/raft_metrics.json" ] && { echo "skip $dir (done)"; continue; }
      echo "=== $name on $src, $view ==="
      if [[ "$flags" == *-checkpoint* ]]; then
        ck=(${flags%% /*} "$ckpt")   # the checkpoint path may contain spaces or parentheses
      else
        # shellcheck disable=SC2206
        ck=($flags)
      fi
      # shellcheck disable=SC2046
      python run_raft_shard_baseline.py --shards "$SHARDS" --sources "$src" --resolution 6 \
        --geodesic-metric haversine --device "${DEVICE:-cuda}" --batch-size 1 \
        --max-pairs "$MAX_PAIRS" $(view_flags "$view") "${ck[@]}" \
        --output-dir "$dir" < /dev/null 2>&1 | grep -v -E "RuntimeWarning|pairs:" | tail -4 \
        || echo "FAILED $name $src $view"
      [ -f "$dir/raft_metrics.json" ] || echo "FAILED $name $src $view (no metrics)"
    done
  done
done <<< "$ROWS"

python - "$DEST" "$SOURCES" "$VIEWS" "${names[@]}" <<'PY'
import json, os, sys
from pathlib import Path

root, sources, views, names = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split(), sys.argv[4:]


def load(name, tag, view):
    f = root / f"{name}_{tag}_{view}" / "raft_metrics.json"
    return json.loads(f.read_text())["metrics"] if f.is_file() else None


for src in sources:
    tag = src.replace(":", "_")
    keys = (("global_geo_deg", "sphere-mean geodesic error, deg"),
            ("poles_geo_deg", "poles, deg"), ("equator_geo_deg", "equator, deg"))
    for key, title in keys[:1] if os.environ.get("TABLE") == "global" else keys:
        print(f"\n[{src}] {title} (change vs upright, %)")
        print(f"{'row':<22}" + "".join(f"{v:>18}" for v in views))
        zero = ["-"] * len(views)
        for n in names:
            cells, base = [], None
            for i, v in enumerate(views):
                m = load(n, tag, v)
                if m is None or key not in m:
                    cells.append(f"{'-':>18}")
                    continue
                if v == "upright":
                    base = m[key]
                zk = key.replace("_geo_deg", "_zero_geo_deg")
                if zk in m:
                    zero[i] = f"{m[zk]:.4f}"
                rel = f" ({100 * (m[key] / base - 1):+.0f})" if base and v != "upright" else ""
                cells.append(f"{m[key]:.4f}{rel}".rjust(18))
            print(f"{n:<22}" + "".join(cells))
        print(f"{'zero flow':<22}" + "".join(f"{z:>18}" for z in zero))
PY
exit 0
