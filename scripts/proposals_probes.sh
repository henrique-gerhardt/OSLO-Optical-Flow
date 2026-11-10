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
  esac
done
exit 0
