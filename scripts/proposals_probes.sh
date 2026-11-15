#!/usr/bin/env bash
# Minimal experiments for the five proposals of reports/Formulações de fluxo para HEALPix.md
# (experiments/proposals/README.md). Eval-only except P5 (small isolated training).
# PROBES selects a subset, e.g. PROBES="p1 p4". Each probe skips if its JSON exists.
set -uo pipefail

OUT="${OUTPUT_DIR:-/outputs}/proposals_probes${TAG:+_$TAG}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
CKPTS="${CKPTS:-${OUTPUT_DIR:-/outputs}/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s7/oslo_raft.pt}"
P6_CKPTS="${P6_CKPTS:-${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s7/oslo_raft.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s11/oslo_raft.pt}"
P7_CKPTS="${P7_CKPTS:-${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s7/oslo_raft.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k/noso3_erppx_s11/oslo_raft.pt ${OUTPUT_DIR:-/outputs}/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt}"
P7R_CKPTS="${P7R_CKPTS:-${OUTPUT_DIR:-/outputs}/scale_fs30k_planar/raft_small_fs_s7/raft_erp.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k_planar/raft_small_fs_s11/raft_erp.pt ${OUTPUT_DIR:-/outputs}/raft_pretrain/raft_small_c360_ft_s7/raft_erp.pt}"
P6R_CKPTS="${P6R_CKPTS:-${OUTPUT_DIR:-/outputs}/scale_fs30k_planar/raft_small_fs_s7/raft_erp.pt ${OUTPUT_DIR:-/outputs}/scale_fs30k_planar/raft_small_fs_s11/raft_erp.pt}"
PROBES="${PROBES:-p1 p2 p3 p4 p5}"
DEV="${DEVICE:-cuda}"
mkdir -p "$OUT"
cd experiments/proposals
export PYTHONPATH="$(cd ../.. && pwd)"
common=(--shards "$SHARDS" --sources flowscape:test)

run() {  # run <name> <script> <args...>
  local name="$1"; shift
  if [ -f "$OUT/$name/done" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name ==="
  python "$@" --output-dir "$OUT/$name" 2>&1 | grep -v -E "RuntimeWarning|np.stack" | tee "$OUT/$name.log" \
    && touch "$OUT/$name/done" || echo "FAILED $name"
}

for p in $PROBES; do
  case "$p" in
    p1) run p1_rigid_ceiling p1_rigid_ceiling.py "${common[@]}" --stride 5 --max-pairs 280 ;;
    p2) run p2_cascade_probe p2_cascade_probe.py "${common[@]}" --stride 14 --max-pairs 100 ;;
    p3) for ck in $CKPTS; do
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          run "p3_gate_$tag" p3_gate_probe.py "${common[@]}" --stride 13 --max-pairs 106 \
            --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache" --device "$DEV"
        done ;;
    p4) for m in raft_small raft_large; do
          run "p4_faces_${m}_r8" p4_faces_probe.py "${common[@]}" --stride 13 --max-pairs 106 \
            --model "$m" --face-resolution 8 --pad 64 --device "$DEV"
        done ;;
    p5) run p5_waft_toy p5_waft_toy.py "${common[@]}" --stride 13 --max-pairs 106 \
          --train-sources flowscape:train --train-pairs 1000 --steps "${P5_STEPS:-3000}" --device "$DEV" ;;
    p6) for ck in $P6_CKPTS; do
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$ck")")"
          run "p6_band_$tag" p6_polar_band_head.py "${common[@]}" --stride 1 --max-pairs 100000 \
            --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache" --device "$DEV" \
            --train-sources flowscape:train --train-pairs "${P6_PAIRS:-1000}" --train-stride 2 --steps "${P6_STEPS:-6000}"
        done ;;
    p6r) for ck in $P6R_CKPTS; do  # RAFT-small control of p6: same band head, same recipe
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          run "p6r_band_$tag" p6_polar_band_head.py "${common[@]}" --stride 1 --max-pairs 100000 \
            --predictor raft --raft-model raft_small --tv-checkpoint "$ck" --device "$DEV" \
            --train-sources flowscape:train --train-pairs "${P6_PAIRS:-1000}" --train-stride 2 --steps "${P6_STEPS:-6000}"
        done ;;
    p7) for ck in $P7_CKPTS; do
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          run "p7_gate_$tag" p7_learned_gate.py "${common[@]}" --stride 1 --max-pairs 100000 \
            --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache" --device "$DEV" \
            --train-sources flowscape:train --train-pairs 1500 --train-stride 2 --steps "${P7_STEPS:-4000}"
        done ;;
    p7r) for ck in $P7R_CKPTS; do  # RAFT-small control of p7: same gate, same recipe
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          run "p7r_gate_$tag" p7_learned_gate.py "${common[@]}" --stride 1 --max-pairs 100000 \
            --predictor raft --raft-model raft_small --tv-checkpoint "$ck" --device "$DEV" \
            --train-sources flowscape:train --train-pairs 1500 --train-stride 2 --steps "${P7_STEPS:-4000}"
        done ;;
    p8) for ck in $P7_CKPTS $P7R_CKPTS; do  # tuned magnitude threshold vs the p7 learned gate, no training
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          case "$ck" in
            *raft_erp.pt) pred=(--predictor raft --raft-model raft_small --tv-checkpoint "$ck"); p7="$OUT/p7r_gate_$tag" ;;
            *) pred=(--init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache"); p7="$OUT/p7_gate_$tag" ;;
          esac
          run "p8_mag_$tag" p8_mag_sweep.py "${common[@]}" --stride 1 --max-pairs 100000 "${pred[@]}" --device "$DEV" \
            --train-sources flowscape:train --train-pairs 300 --train-stride 5 --p7-json "$p7/p7_learned_gate.json"
        done ;;
    p9) for ck in $P6_CKPTS $P6R_CKPTS; do  # stack the trained p6 band head + p7 gate, no training
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          case "$ck" in
            *raft_erp.pt) pred=(--predictor raft --raft-model raft_small --tv-checkpoint "$ck")
                          band="$OUT/p6r_band_$tag"; gate="$OUT/p7r_gate_$tag" ;;
            *) pred=(--init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache")
               band="$OUT/p6_band_$(basename "$(dirname "$ck")")"; gate="$OUT/p7_gate_$tag" ;;
          esac
          [ -f "$band/band_head.pt" ] && [ -f "$gate/gate_net.pt" ] || { echo "missing heads for $tag"; continue; }
          run "p9_stack_$tag" p9_band_gate_stack.py "${common[@]}" --stride 1 --max-pairs 100000 "${pred[@]}" \
            --device "$DEV" --band-head "$band/band_head.pt" --gate-net "$gate/gate_net.pt"
        done ;;
    p10) for ck in $P6_CKPTS $P6R_CKPTS; do  # frame prediction on HEALPix nodes from each model's flow, no training
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          case "$ck" in
            *raft_erp.pt) pred=(--predictor raft --raft-model raft_small --tv-checkpoint "$ck") ;;
            *) pred=(--predictor oslo --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache") ;;
          esac
          run "p10_pred_$tag" p10_prediction_probe.py "${common[@]}" --stride 1 --max-pairs 100000 "${pred[@]}" \
            --device "$DEV" --grid-resolution "${P10_GRID:-8}"
        done ;;
    p11) prior="${PRIOR_CKPT:-${OUTPUT_DIR:-/outputs}/prior_weights/FlowScape-final.pth}"  # SO(3) fine-tune of released PriOr-RAFT
        [ -f "$prior" ] || { echo "missing $prior"; continue; }
        for arm in ${P11_ARMS:-so3 ctrl}; do
          run "p11_so3ft_$arm" p11_so3_finetune_probe.py --shards "$SHARDS" --checkpoint "$prior" --arm "$arm" \
            --steps "${P11_STEPS:-5000}" --eval-at "${P11_EVAL_AT:-1000,2500}" --eval-pairs 100 --eval-stride 13 \
            --lr "${P11_LR:-2e-5}" --so3-prob "${P11_SO3_PROB:-0.5}" --so3-max-angle-deg "${P11_MAX_ANGLE:-180}" --device "$DEV" \
              --accum "${P11_ACCUM:-1}" --lr-schedule "${P11_SCHEDULE:-constant}" --ema "${P11_EMA:-0}" --log-every "${P11_LOG_EVERY:-10}"
        done ;;
    p12) run p12_canonizer p12_canonizer.py --shards "$SHARDS" --steps "${P12_STEPS:-8000}" --device "$DEV" \
          --train-frames "${P12_FRAMES:-2500}" --train-stride "${P12_STRIDE:-2}" \
          --train-sources "${P12_TRAIN:-flowscape:train}" --sources "${P12_TEST:-flowscape:test}" ;;
    p13) for ck in ${P13_CKPTS:-$P6_CKPTS $P6R_CKPTS}; do  # NESTED residual levels on a frozen model's field (hierarchy as representation)
          [ -f "$ck" ] || { echo "missing $ck"; continue; }
          tag="$(basename "$(dirname "$(dirname "$ck")")")_$(basename "$(dirname "$ck")")"
          case "$ck" in
            *raft_erp.pt) pred=(--predictor raft --raft-model raft_small --tv-checkpoint "$ck"); sets="${P13_RAFT_LEVELS:-6_7}" ;;
            *) pred=(--predictor oslo --init-checkpoint "$ck" --pyramid-cache "${OUTPUT_DIR:-/outputs}/pyramid_cache"); sets="${P13_OSLO_LEVELS:-6_7 6_6}" ;;
          esac
          for lv in $sets; do
            run "p13_h${lv}_$tag" p13_nested_residual.py "${common[@]}" --stride "${P13_EVAL_STRIDE:-2}" --max-pairs 100000 \
              "${pred[@]}" --levels ${lv//_/ } --steps "${P13_STEPS:-6000}" --train-pairs "${P13_TRAIN_PAIRS:-600}" --device "$DEV"
          done
        done
        python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
print("\n[p13 summary] sphere-mean geodesic error, degrees, r6 nodes")
print(f"{'run':<52}{'zero':>8}{'base':>8}{'final':>8}{'gain %':>8}{'static':>8}{'moving':>8}{'px base':>9}{'px final':>9}")
for path in sorted(Path(sys.argv[1]).glob("p13_h*/p13_nested_residual.json")):
    d = json.loads(path.read_text()); t = d["table"]; f = t[d["final_row"]]; e = d.get("erp", {})
    px = lambda row: e.get(row, {}).get("official_inraster_epe", float("nan"))
    print(f"{path.parent.name:<52}{t['zero']['all']:>8.4f}{t['base']['all']:>8.4f}{f['all']:>8.4f}{d['gain_pct']:>8.1f}"
          f"{f['static']:>8.4f}{f['moving']:>8.4f}{px('native_erp' if 'native_erp' in e else 'base_nodes'):>9.3f}{px('final'):>9.3f}")
PY
        ;;
    p14) for seed in ${P14_SEEDS:-7 11}; do  # pretrained RAFT-small fine-tuned on HEALPix faces vs on the ERP raster, same loss and pixel budget
          for repr in ${P14_REPRS:-faces erp}; do
            run "p14_${repr}_s${seed}" p14_faces_finetune.py "${common[@]}" --stride "${P14_EVAL_STRIDE:-3}" --max-pairs 100000 \
              --repr "$repr" --seed "$seed" --steps "${P14_STEPS:-8000}" --eval-at "${P14_EVAL_AT:-2000,4000}" --device "$DEV"
          done
        done
        python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
print("\n[p14 summary] sphere-mean geodesic error, degrees, r6 nodes (pretrained -> fine-tuned)")
print(f"{'run':<18}{'upright 0':>11}{'upright':>10}{'haar 0':>10}{'haar':>10}{'haar/upright':>14}")
for path in sorted(Path(sys.argv[1]).glob("p14_*/p14_faces_finetune.json")):
    h = json.loads(path.read_text())["history"]; a, b = h[0], h[-1]
    g = lambda row, v: row.get(v, {}).get("deg", float("nan"))
    print(f"{path.parent.name:<18}{g(a, 'upright'):>11.4f}{g(b, 'upright'):>10.4f}{g(a, 'haar'):>10.4f}{g(b, 'haar'):>10.4f}"
          f"{g(b, 'haar') / g(b, 'upright'):>14.2f}")
PY
        ;;
  esac
done
exit 0
