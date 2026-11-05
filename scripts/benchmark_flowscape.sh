#!/usr/bin/env bash
# Published-benchmark table on FlowScape test (1386 of the official 1400 pairs), in the
# protocol of PriOr-Flow's evaluate.py: ERP-pixel EPE over every pixel and SEPE
# (great-circle endpoint distance), All / Equator / Poles split at 45 degrees latitude.
# The released PriOr-RAFT and PanoFlow checkpoints are the reproduction check against
# the published table (EPE 2.33 / 3.38 px, SEPE 3.49 / 4.78). Eval only.
set -euo pipefail

OUT="${OUTPUT_DIR:-/outputs}"
SHARDS="${OSLO_SHARDS:-/data/shards}"
SRC="${SOURCE:-flowscape:test}"
DEST="$OUT/benchmark_${SRC//:/_}${TAG:+_$TAG}"
PRIOR="${PRIOR_CKPT:-$OUT/prior_weights/FlowScape-final.pth}"
PANOFLOW="${PANOFLOW_CKPT:-$(find "$OUT" -maxdepth 3 -name 'PanoFlow(CSFlow)*.pth' 2>/dev/null | head -1)}"
# ROWS: "name|predictor|checkpoint (empty = none)|run_epe_eval.py flags"
ROWS="${ROWS:-zero|zero||
oracle_r6|oracle||--resolution 6
oracle_r7|oracle||--resolution 7
oracle_r8|oracle||--resolution 8
prior_raft|raft|$PRIOR|--prior-checkpoint $PRIOR
panoflow|raft|$PANOFLOW|--panoflow-checkpoint $PANOFLOW
frozen_raft_large|raft||--model raft_large --weights default
raft_small_so3|raft|$OUT/raft_parity_planar/raft_small_so3_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_so3_s7/raft_erp.pt
raft_large_so3_scale|raft|$OUT/raft_parity_planar_scale/raft_large_so3_s7/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_planar_scale/raft_large_so3_s7/raft_erp.pt
oslo_base|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt
oslo_base_it4|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt --eval-iters 4
oslo_base_it8|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt --eval-iters 8
oslo_base_it24|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt --eval-iters 24
oslo_base_it32|oslo|$OUT/raft_parity_large/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/base_s7/oslo_raft.pt --eval-iters 32
oslo_noso3|oslo|$OUT/raft_parity_large/noso3_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/noso3_s7/oslo_raft.pt
oslo_large_scale|oslo|$OUT/raft_parity_scale/large_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_scale/large_s7/oslo_raft.pt
oslo_r8_cp4|oslo|$OUT/raft_parity_native_mix_cp4/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_native_mix_cp4/base_s7/oslo_raft.pt
oslo_reach29|oslo|$OUT/raft_parity_large/rings1_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/rings1_s7/oslo_raft.pt
oslo_reach117|oslo|$OUT/raft_parity_large/rings4_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/rings4_s7/oslo_raft.pt
oslo_reach117_dense|oslo|$OUT/raft_parity_large/rings4dense_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/rings4dense_s7/oslo_raft.pt
oslo_sup_r7|oslo|$OUT/raft_parity_sup7/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_sup7/base_s7/oslo_raft.pt
oslo_base_30k|oslo|$OUT/raft_parity_budget/base_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_budget/base_s7/oslo_raft.pt
oslo_gridfix|oslo|$OUT/raft_parity_large/gridfix_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/gridfix_s7/oslo_raft.pt
oslo_small|oslo|$OUT/raft_parity_large/small_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/small_s7/oslo_raft.pt
raft_small_so3_320|raft|$OUT/raft_parity_planar/raft_small_so3_320_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_so3_320_s7/raft_erp.pt --infer-size 320x640
oslo_gridfix_s11|oslo|$OUT/raft_parity_large/gridfix_s11/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/gridfix_s11/oslo_raft.pt
oslo_small_s11|oslo|$OUT/raft_parity_large/small_s11/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/small_s11/oslo_raft.pt
raft_small_so3_320_s11|raft|$OUT/raft_parity_planar/raft_small_so3_320_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_so3_320_s11/raft_erp.pt --infer-size 320x640
oslo_small_noso3|oslo|$OUT/raft_parity_large/small_noso3_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/small_noso3_s7/oslo_raft.pt
raft_small_noaug|raft|$OUT/raft_parity_planar/raft_small_noaug_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_s7/raft_erp.pt
raft_small_noaug_320|raft|$OUT/raft_parity_planar/raft_small_noaug_320_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_320_s7/raft_erp.pt --infer-size 320x640
raft_large_ft|raft|$OUT/raft_parity_recipe/raft_large_ft_s7/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_recipe/raft_large_ft_s7/raft_erp.pt
raft_large_fs|raft|$OUT/raft_parity_recipe/raft_large_fs_s7/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_recipe/raft_large_fs_s7/raft_erp.pt
raft_small_ft|raft|$OUT/raft_parity_recipe/raft_small_ft_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_recipe/raft_small_ft_s7/raft_erp.pt
raft_small_fs|raft|$OUT/raft_parity_recipe/raft_small_fs_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_recipe/raft_small_fs_s7/raft_erp.pt
oslo_fs_noso3|oslo|$OUT/raft_parity_fsonly/noso3_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_fsonly/noso3_s7/oslo_raft.pt
oslo_noso3_erppx|oslo|$OUT/raft_parity_large/noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/noso3_erppx_s7/oslo_raft.pt
raft_small_noaug_geo|raft|$OUT/raft_parity_planar/raft_small_noaug_geo_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_geo_s7/raft_erp.pt
raft_small_noaug_bn7|raft|$OUT/raft_parity_planar/raft_small_noaug_bn7_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_bn7_s7/raft_erp.pt --retina-bottleneck 7
raft_large_ft_bn7|raft|$OUT/raft_parity_recipe/raft_large_ft_bn7_s7/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_recipe/raft_large_ft_bn7_s7/raft_erp.pt --retina-bottleneck 7
oslo_small_noso3_erppx|oslo|$OUT/raft_parity_large/small_noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/small_noso3_erppx_s7/oslo_raft.pt
oslo_noso3_erppx_nocap|oslo|$OUT/raft_parity_large/noso3_erppx_nocap_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/noso3_erppx_nocap_s7/oslo_raft.pt
oslo_sup7_noso3_erppx|oslo|$OUT/raft_parity_sup7/noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_sup7/noso3_erppx_s7/oslo_raft.pt
oslo_noso3_erppx_s11|oslo|$OUT/raft_parity_large/noso3_erppx_s11/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/noso3_erppx_s11/oslo_raft.pt
raft_small_noaug_bn7_s11|raft|$OUT/raft_parity_planar/raft_small_noaug_bn7_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_bn7_s11/raft_erp.pt --retina-bottleneck 7
raft_large_ft_bn7_s11|raft|$OUT/raft_parity_recipe/raft_large_ft_bn7_s11/raft_erp.pt|--model raft_large --tv-checkpoint $OUT/raft_parity_recipe/raft_large_ft_bn7_s11/raft_erp.pt --retina-bottleneck 7
oslo_pretrain|oslo|$OUT/oslo_pretrain/base_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain/base_s7/oslo_raft.pt
oslo_pretrain_ft|oslo|$OUT/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain_ft/noso3_erppx_s7/oslo_raft.pt
oslo_fs_erppx|oslo|$OUT/raft_parity_fsonly/noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_fsonly/noso3_erppx_s7/oslo_raft.pt
raft_small_c360|raft|$OUT/raft_pretrain/raft_small_c360_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_pretrain/raft_small_c360_s7/raft_erp.pt
raft_small_c360_ft|raft|$OUT/raft_pretrain/raft_small_c360_ft_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_pretrain/raft_small_c360_ft_s7/raft_erp.pt
oslo_est5_erppx|oslo|$OUT/raft_parity_large/est5_noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/est5_noso3_erppx_s7/oslo_raft.pt
oslo_pt_est5_10k|oslo|$OUT/oslo_pretrain_est5_10k/est5_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain_est5_10k/est5_s7/oslo_raft.pt
oslo_pt_est5_10k_ft|oslo|$OUT/oslo_pretrain_est5_10k_ft/est5_noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain_est5_10k_ft/est5_noso3_erppx_s7/oslo_raft.pt
oslo_small_noso3_erppx_s11|oslo|$OUT/raft_parity_large/small_noso3_erppx_s11/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_large/small_noso3_erppx_s11/oslo_raft.pt
raft_small_noaug_320_s11|raft|$OUT/raft_parity_planar/raft_small_noaug_320_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_planar/raft_small_noaug_320_s11/raft_erp.pt --infer-size 320x640
oslo_pretrain_ft_s11|oslo|$OUT/oslo_pretrain_ft/noso3_erppx_s11/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain_ft/noso3_erppx_s11/oslo_raft.pt
oslo_fs_erppx_s11|oslo|$OUT/raft_parity_fsonly/noso3_erppx_s11/oslo_raft.pt|--init-checkpoint $OUT/raft_parity_fsonly/noso3_erppx_s11/oslo_raft.pt
raft_small_c360_ft_s11|raft|$OUT/raft_pretrain/raft_small_c360_ft_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_pretrain/raft_small_c360_ft_s11/raft_erp.pt
raft_small_fs_s11|raft|$OUT/raft_parity_recipe/raft_small_fs_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_recipe/raft_small_fs_s11/raft_erp.pt
raft_small_fs_s11b|raft|$OUT/raft_parity_recipe_r2/raft_small_fs_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/raft_parity_recipe_r2/raft_small_fs_s11/raft_erp.pt
oslo_fs_erppx_30k|oslo|$OUT/scale_fs30k/noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/scale_fs30k/noso3_erppx_s7/oslo_raft.pt
raft_small_fs_30k|raft|$OUT/scale_fs30k_planar/raft_small_fs_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/scale_fs30k_planar/raft_small_fs_s7/raft_erp.pt
raft_small_fs_30k_s11|raft|$OUT/scale_fs30k_planar/raft_small_fs_s11/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/scale_fs30k_planar/raft_small_fs_s11/raft_erp.pt
raft_small_fs_60k|raft|$OUT/scale_fs60k_planar/raft_small_fs_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/scale_fs60k_planar/raft_small_fs_s7/raft_erp.pt
oslo_fs_erppx_30k_s11|oslo|$OUT/scale_fs30k/noso3_erppx_s11/oslo_raft.pt|--init-checkpoint $OUT/scale_fs30k/noso3_erppx_s11/oslo_raft.pt
raft_small_fs_60k_r2|raft|$OUT/scale_fs60k_planar_r2/raft_small_fs_s7/raft_erp.pt|--model raft_small --tv-checkpoint $OUT/scale_fs60k_planar_r2/raft_small_fs_s7/raft_erp.pt
oslo_pretrain_comb|oslo|$OUT/oslo_pretrain_comb/est5_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_pretrain_comb/est5_s7/oslo_raft.pt
oslo_comb_ft|oslo|$OUT/oslo_comb_ft/est5_noso3_erppx_s7/oslo_raft.pt|--init-checkpoint $OUT/oslo_comb_ft/est5_noso3_erppx_s7/oslo_raft.pt}"
mkdir -p "$DEST"

names=()
while IFS='|' read -r name predictor ckpt flags; do
  [ -n "$name" ] || continue
  # ONLY: space-separated row names to run (default: every row)
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "missing checkpoint '$ckpt', leaving $name out"
    continue
  fi
  names+=("$name")
  dir="$DEST/$name"
  [ -f "$dir/epe_metrics.json" ] && { echo "skip $name (done)"; continue; }
  echo "=== $name ($predictor) on $SRC ==="
  # shellcheck disable=SC2086
  python run_epe_eval.py --shards "$SHARDS" --sources "$SRC" --predictor "$predictor" \
    --device "${DEVICE:-cuda}" --pyramid-cache "$OUT/pyramid_cache" \
    --output-dir "$dir" $flags ${EVAL_FLAGS:-} < /dev/null 2>&1 | tee "$dir.log" \
    || echo "FAILED $name (see $dir.log); the other rows continue"
done <<< "$ROWS"

python - "$DEST" "${names[@]}" <<'PY'
import json, math, sys
from pathlib import Path

root, names = Path(sys.argv[1]), sys.argv[2:]
published = {"prior_raft": (2.33, 3.49), "panoflow": (3.38, 4.78)}
print("\nFlowScape test, PriOr-Flow protocol (all pixels; SEPE in 1e-3 rad; poles = |lat| > 45)")
print(f"{'row':<22}{'EPE all':>9}{'EPE eq':>9}{'EPE pol':>9}{'SEPE all':>10}{'SEPE eq':>9}"
      f"{'SEPE pol':>10}{'pairs':>7}   published EPE / SEPE")
for n in names:
    path = root / n / "epe_metrics.json"
    if not path.exists():
        print(f"{n:<22}  missing {path}")
        continue
    m = json.loads(path.read_text())["metrics"]
    row = f"{n:<22}"
    for r in ("all", "equator", "poles"):
        row += f"{m['official_' + r + '_epe']:>9.2f}"
    for r in ("all", "equator", "poles"):
        row += f"{m['official_' + r + '_sepe'] * 1e3:>{10 if r != 'equator' else 9}.2f}"
    row += f"{m['pairs']:>7}"
    if n in published:
        row += f"   {published[n][0]:.2f} / {published[n][1]:.2f}"
    print(row)
bins = ("0-5", "5-15", "15-30", "30-60", "60-90", "90-180")
print("\nEPE (px) by GT displacement (degrees); OSLO reach 59 deg at 2 rings (29 / 117 at 1 / 4), "
      "RAFT 90 deg")
print(f"{'row':<22}" + "".join(f"{b:>9}" for b in bins))
frac_row = None
for n in names:
    path = root / n / "epe_metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())["metrics"]
    if "official_disp0-5_epe" not in m:
        print(f"{n:<22}  (no displacement bins: evaluated before they existed)")
        continue
    print(f"{n:<22}" + "".join(
        f"{m.get('official_disp' + b + '_epe', float('nan')):>9.2f}" for b in bins))
    frac_row = frac_row or m
if frac_row:
    print(f"{'% of pixels':<22}" + "".join(
        f"{100 * frac_row.get('official_disp' + b + '_fraction', 0.0):>9.1f}" for b in bins))
over_rows = []
for n in names:
    path = root / n / "epe_metrics.json"
    if path.exists():
        m = json.loads(path.read_text())["metrics"]
        if "official_overshoot_epe" in m:
            over_rows.append((n, m))
if over_rows:
    frac = over_rows[0][1].get("official_overshoot_fraction", float("nan"))
    print(f"\nNear-nadir overshoot pixels (GT endpoint leaves the raster; {100 * frac:.2f}% of pixels): "
          "EPE there vs everywhere else")
    print(f"{'row':<26}{'overshoot':>11}{'inraster':>10}{'share of EPE':>14}")
    for n, m in over_rows:
        share = frac * m["official_overshoot_epe"] / max(m["official_all_epe"], 1e-9)
        print(f"{n:<26}{m['official_overshoot_epe']:>11.2f}{m['official_inraster_epe']:>10.2f}{100 * share:>13.1f}%")
split_rows = [(n, m) for n, m in over_rows if "official_disp0-5_overshoot_epe" in m]
if split_rows:
    print("\nEPE (px) by GT displacement, split into overshoot / in-raster pixels; "
          "contrib = bin EPE x pixel fraction (px of EPE all)")
    print(f"{'row':<26}{'bin':>8}{'over EPE':>10}{'in EPE':>9}{'over %px':>10}"
          f"{'over contrib':>14}{'in contrib':>12}")
    for n, m in split_rows:
        for b in bins[:3]:
            fo = m.get(f"official_disp{b}_overshoot_fraction", 0.0)
            fi = m.get(f"official_disp{b}_inraster_fraction", 0.0)
            eo = m.get(f"official_disp{b}_overshoot_epe", float("nan"))
            ei = m.get(f"official_disp{b}_inraster_epe", float("nan"))
            print(f"{n:<26}{b:>8}{eo:>10.2f}{ei:>9.2f}{100 * fo:>10.2f}"
                  f"{(eo * fo if fo else 0.0):>14.3f}{(ei * fi if fi else 0.0):>12.3f}")
layouts = ("cube", "paper")
vp_rows = []
for n in names:
    path = root / n / "epe_metrics.json"
    if path.exists():
        m = json.loads(path.read_text())["metrics"]
        if "viewport_cube_all_epe_px" in m:
            vp_rows.append((n, m))
if vp_rows:
    print("\nViewport EPE (px in rectilinear viewports, arXiv 2610.32561 protocol); "
          "cube = six 90-deg faces, paper = 4 equator 150x60 + 2 poles")
    print(f"{'row':<26}" + "".join(f"{L + ' ' + g:>14}" for L in layouts
                                   for g in ("all", "equator", "poles")))
    for n, m in vp_rows:
        print(f"{n:<26}" + "".join(f"{m[f'viewport_{L}_{g}_epe_px']:>14.3f}" for L in layouts
                                   for g in ("all", "equator", "poles")))
print("\nPublished (PriOr-Flow Table, FlowScape all weathers, EPE / SEPE): TanImg 18.3 / 25.3, "
      "SLOF 7.59 / 5.79, PanoFlow 3.38 / 4.78, PriOr-Flow 2.33 / 3.49")
PY
