#!/usr/bin/env bash
# RAFT-parity, matched backbone: planar TorchVision RAFT trained from scratch on the ERP
# frames of the same shards, same steps and pairs per step as the OSLO r7 runs, scored
# with the same spherical metrics on the same supervision grid (r6). Also scores the
# frozen pretrained RAFTs (zero-shot) and puts the OSLO r7 checkpoints in the table.
# See docs/plans/RAFT_PARITY.md section 9.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
DEST="$OUT/${RUN_NAME:-raft_parity_planar}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
# ARMS: one planar arm per line, "name|model|run_erp_raft_train.py flags". raft_small =
# RAFT's recipe on ERP; raft_small_noaug = no augmentation at all (no color jitter, flip
# or roll; 512x1024 crop = a whole flowscape frame), like the OSLO base, which separates
# the architecture from the augmentation.
ARMS="${ARMS:-raft_small|raft_small|
raft_small_noaug|raft_small|--photo-prob 0 --hflip-prob 0 --no-roll --crop 512x1024}"
FROZEN="${FROZEN:-raft_small raft_large}"
SEED="${SEED:-7}"
STEPS="${STEPS:-10000}"
TRAIN_SOURCES="${TRAIN_SOURCES:-replica360:train,flowscape:train}"
VAL_SOURCES="${VAL_SOURCES:-flowscape:val replica360:val}"
# EVAL_TAG: suffix for the eval directories, so a second view of the same checkpoints
# (EVAL_FLAGS=--val-so3-prob 1, the rotated view) does not skip as already done.
EVAL_TAG="${EVAL_TAG:-}"
# OSLO rows for the table: "name|run dir under /outputs" (r7 runs, scored on r6), or
# "name|metrics json under /outputs" with {tag} for the val source (flowscape_val ...).
OSLO_RUNS="${OSLO_RUNS:-oslo_base|raft_parity_large/base_s7
oslo_noso3|raft_parity_large/noso3_s7}"
mkdir -p "$DEST"

score() {  # score <name> <val source> [run_raft_shard_baseline flags...]
  local name="$1" src="$2"
  shift 2
  local dir="$DEST/eval/${name}_${src//:/_}${EVAL_TAG:+_$EVAL_TAG}"
  [ -f "$dir/raft_metrics.json" ] && { echo "skip $dir (done)"; return; }
  echo "=== score $name on $src ==="
  python run_raft_shard_baseline.py --shards "$SHARDS" --sources "$src" --resolution 6 \
    --geodesic-metric haversine --device "${DEVICE:-cuda}" --batch-size 1 ${EVAL_FLAGS:-} \
    --output-dir "$dir" "$@" < /dev/null
}

names=()
while IFS='|' read -r name model flags; do
  [ -n "$name" ] || continue
  names+=("${name}_s${SEED}")
  dir="$DEST/${name}_s${SEED}"
  if [ ! -f "$dir/raft_erp.pt" ]; then
    echo "=== train $name ($model $flags) seed $SEED ==="
    mkdir -p "$dir"
    # shellcheck disable=SC2086
    python run_erp_raft_train.py --shards "$SHARDS" --train-sources "$TRAIN_SOURCES" \
      --model "$model" --steps "$STEPS" --seed "$SEED" \
      --batch-size "${BATCH_SIZE:-2}" --grad-accum "${GRAD_ACCUM:-1}" \
      --num-workers "${NUM_WORKERS:-4}" --device "${DEVICE:-cuda}" --amp $flags ${TRAIN_FLAGS:-} \
      --output-dir "$dir" < /dev/null \
      2>&1 | tee "$dir.train.log"
  fi
  for src in $VAL_SOURCES; do
    score "${name}_s${SEED}" "$src" --model "$model" --tv-checkpoint "$dir/raft_erp.pt"
  done
done <<< "$ARMS"
for model in $FROZEN; do
  names+=("frozen_${model}")
  for src in $VAL_SOURCES; do
    score "frozen_${model}" "$src" --model "$model" --weights default
  done
done

python - "$OUT" "$DEST" "$VAL_SOURCES" "$OSLO_RUNS" "$EVAL_TAG" "${names[@]}" <<'PY'
import json, sys
from pathlib import Path

out, dest, sources, oslo_runs, eval_tag = sys.argv[1:6]
suffix = f"_{eval_tag}" if eval_tag else ""
out, dest = Path(out), Path(dest)
rows = []
for line in oslo_runs.splitlines():
    if line.strip():
        name, run = line.split("|")
        rows.append((name, "oslo", out / run))
rows += [(n, "raft", None) for n in sys.argv[6:]]

keys = ("global", "active_0_5", "poles", "equator")
for src in sources.split():
    tag = src.replace(":", "_")
    print(f"\n[{src}{' ' + eval_tag if eval_tag else ''}]  improvement over zero flow, % "
          "(r6 nodes, haversine; equal zero deg = same pairs and grid)")
    print(f"{'row':<24}" + "".join(f"{k:>12}" for k in keys) + f"{'global deg':>12}{'zero deg':>10}")
    for name, kind, run in rows:
        if kind == "oslo" and "{tag}" in str(run):
            path = Path(str(run).format(tag=tag))
        elif kind == "oslo":
            path = run / ("oslo_raft_metrics.json" if src == "flowscape:val"
                          else f"metrics_{tag}.json")
        else:
            path = dest / "eval" / f"{name}_{tag}{suffix}" / "raft_metrics.json"
        if not path.exists():
            print(f"{name:<24}  missing {path}")
            continue
        m = json.loads(path.read_text())["metrics"]
        print(f"{name:<24}" + "".join(f"{m[k + '_improvement_pct']:>12.1f}" for k in keys)
              + f"{m['global_geo_deg']:>12.3f}{m['global_zero_geo_deg']:>10.4f}")
PY
