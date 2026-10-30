#!/usr/bin/env bash
# RAFT-parity A/B on replica360 (run inside the oslo-raft container on the box).
#
# Arms x seeds, from scratch, replica360:train -> replica360:val, at the A1
# matched-pair recipe (retina r7 / supervision r6 / estimation r4, 2000 steps,
# one-cycle, haversine). Each arm changes one RAFT difference against `base`;
# `both` and `raftfull` are the combinations. See docs/plans/RAFT_PARITY.md.
#
#   SHARDS_HOST=/path/to/sfprep/shards \
#   docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
#       bash scripts/raft_parity_ab.sh
#
# Finished runs are skipped, so the script can be re-launched after an interruption.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}/${RUN_NAME:-raft_parity_ab}"
TRAIN_SOURCES="${TRAIN_SOURCES:-replica360:train}"
VAL_SOURCES="${VAL_SOURCES:-replica360:val}"
# Space-separated extra val sources, each scored by an eval-only pass after training.
EXTRA_VAL_SOURCES="${EXTRA_VAL_SOURCES:-}"
# EXTRA_FLAGS: runner flags appended to every run of this invocation.
# Arms that get a one-step run before anything else, so an out-of-memory arm fails in
# the first minute instead of after the arms ahead of it have trained.
PREFLIGHT_ARMS="${PREFLIGHT_ARMS:-}"
SEEDS="${SEEDS:-7 11 19}"
ARMS="${ARMS:-base unit bary both raw rings4 noaux enc2 raftblock iters12 sched raftfull}"
STEPS="${STEPS:-2000}"

arm_flags() {
  case "$1" in
    base) echo "" ;;
    unit) echo "--flow-unit node" ;;
    bary) echo "--lookup-interp barycentric" ;;
    both) echo "--flow-unit node --lookup-interp barycentric" ;;
    raw) echo "--corr-norm raw" ;;
    rings1) echo "--lookup-rings 1" ;;
    rings4) echo "--lookup-rings 4" ;;
    rings4dense) echo "--lookup-rings 4 --lookup-ring-points 16" ;;
    noaux) echo "--aux-match-weight 0" ;;
    enc2) echo "--encoder-blocks-per-level 2" ;;
    estblocks) echo "--encoder-est-blocks 2" ;;
    raftblock) echo "--update-block raft" ;;
    iters12) echo "--iters 12" ;;
    noso3) echo "--so3-prob 0" ;;
    # same SDPAConv neighbor orientation in both hemispheres (atan2 order differs by 45 deg)
    slotfix) echo "--neighbor-sort-cut-deg -157.5" ;;
    # slotfix + an 8th neighbor for the 24 retina pixels that have 7 (no fixed pattern)
    gridfix) echo "--neighbor-sort-cut-deg -157.5 --fill-missing-neighbors" ;;
    # about RAFT-small's 0.99M parameters (base is 1.56M)
    small) echo "--hidden-channels 72 --context-dim 48" ;;
    small_noso3) echo "--hidden-channels 72 --context-dim 48 --so3-prob 0" ;;
    # upright, trained on the benchmark's ERP-pixel EPE instead of the geodesic
    noso3_erppx) echo "--so3-prob 0 --loss-space erp_px" ;;
    # matched size (about RAFT-small) with the ERP-pixel loss
    small_noso3_erppx) echo "--hidden-channels 72 --context-dim 48 --so3-prob 0 --loss-space erp_px" ;;
    # ERP-pixel loss without the density cap of 20 (r6 polar nodes reach about 78)
    noso3_erppx_nocap) echo "--so3-prob 0 --loss-space erp_px --loss-erp-max-density 1000" ;;
    # 4 fixed batches repeated: can the model fit 8 pairs? (train_global must go to ~0)
    overfit) echo "--overfit-batches 4" ;;
    yaw) echo "--so3-axis vertical" ;;
    so3small) echo "--so3-max-angle-deg 30" ;;
    sched) echo "--onecycle-pct-start 0.05 --onecycle-anneal linear --weight-decay 1e-4" ;;
    large) echo "--lookup-rings 4 --encoder-blocks-per-level 2 --update-block raft" \
                "--hidden-channels 128 --context-dim 128" ;;
    raftfull) echo "--flow-unit node --lookup-interp barycentric --corr-norm raw" \
                   "--lookup-rings 4 --encoder-blocks-per-level 2 --update-block raft" \
                   "--hidden-channels 128 --context-dim 128 --iters 12" \
                   "--onecycle-pct-start 0.05 --onecycle-anneal linear --weight-decay 1e-4" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

# run <arm> <seed> <output-dir> [extra runner flags...]
run() {
  local arm="$1" seed="$2" dir="$3"
  shift 3
  mkdir -p "$dir"
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
    --train-sources "$TRAIN_SOURCES" --val-sources "$VAL_SOURCES" \
    --steps "$STEPS" --seed "$seed" \
    --batch-size "${BATCH_SIZE:-2}" --grad-accum "${GRAD_ACCUM:-1}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --eval-every "${EVAL_EVERY:-500}" --log-every "${LOG_EVERY:-100}" \
    --output-dir "$dir" \
    ${EXTRA_FLAGS:-} $(arm_flags "$arm") "$@"
}

mkdir -p "$OUT"

for arm in $PREFLIGHT_ARMS; do
  echo "=== preflight arm=$arm ==="
  run "$arm" 0 "$OUT/_preflight/$arm" --smoke-test --max-val-pairs 2
done

for seed in $SEEDS; do
  for arm in $ARMS; do
    dir="$OUT/${arm}_s${seed}"
    if [ -f "$dir/oslo_raft_metrics.json" ]; then
      echo "skip $dir (done)"
    else
      echo "=== arm=$arm seed=$seed ==="
      run "$arm" "$seed" "$dir" 2>&1 | tee "$dir.train.log"
    fi
    for src in $EXTRA_VAL_SOURCES; do
      extra="$dir/metrics_${src//:/_}.json"
      if [ ! -f "$extra" ]; then
        echo "=== arm=$arm seed=$seed eval on $src ==="
        VAL_SOURCES="$src" run "$arm" "$seed" "$dir" --eval-only \
          --init-checkpoint "$dir/oslo_raft.pt" --metrics-out "$extra"
      fi
    done
  done
done

python - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
keys = ["global_improvement_pct", "active_0_5_improvement_pct", "poles_improvement_pct",
        "equator_improvement_pct", "global_geo_deg"]
tables = {}
for path in sorted(out.glob("*_s*/*.json")):
    if path.name == "oslo_raft_metrics.json":
        table = "training val"
    elif path.name.startswith("metrics_"):
        table = path.stem[len("metrics_"):]
    else:
        continue
    arm, seed = path.parent.name.rsplit("_s", 1)
    payload = json.loads(path.read_text())
    tables.setdefault(table, {}).setdefault(arm, {})[seed] = (payload["metrics"], payload["params"])
for table, rows in tables.items():
    print(f"\n[{table}]")
    print(f"{'arm':10s} {'params':>10s} {'seeds':10s} " + " ".join(f"{k:>28s}" for k in keys))
    for arm, by_seed in rows.items():
        cells = []
        for k in keys:
            vals = [m[k] for m, _ in by_seed.values()]
            spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):>19.2f} ±{spread:>6.2f}")
        params = next(iter(by_seed.values()))[1]
        print(f"{arm:10s} {params:>10,} {','.join(sorted(by_seed)):10s} "
              + " ".join(f"{c:>28s}" for c in cells))
PY
