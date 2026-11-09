"""Control for p7: how much of the learned gate does a tuned magnitude threshold recover?

The learned gate (p7) ranks static pixels about as well as the predicted motion itself (AUROC
0.876 vs 0.865-0.872 for -|pred|). The p3/p7 proxy was only swept up to 0.5 deg, so a larger
threshold could recover most of the gain without a network. This script needs no training.
It picks the threshold on train pairs and reports it on flowscape:test. It does the same for
a global scalar shrink of the prediction, which tests pure magnitude calibration.

Rows on flowscape:test:
  base          the frozen predictor (= its benchmark row)
  mag_train     pred zeroed where the predicted motion < tau*, with tau* picked on train
  shrink_train  pred * s*, with s* picked on train
  mag_<tau>     the whole threshold sweep on test (no viewport metric; this is the optimistic upper bound)

With --p7-json, it prints the fraction of the learned gate's gain that each row recovers,
f = (base - row) / (base - gate_soft).

Pre-registered reading, on mag_train:
  f >= 0.75   the learned gate is mostly a threshold on |pred|; report the threshold, not a network.
  f <= 0.5    the network does work a threshold cannot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
from p7_learned_gate import build_raft_runner, motion_deg
from p6_polar_band_head import build_oslo_runner

TAUS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
SCALES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


def candidates(pred: torch.Tensor):
    deg = motion_deg(pred)
    for tau in TAUS:
        yield f"mag_{tau:g}", pred.masked_fill((deg < tau)[..., None], 0.0)
    for s in SCALES:
        yield f"shrink_{s:g}", pred * s


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--predictor", default="oslo", choices=["oslo", "raft"])
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--tv-checkpoint", default="")
    p.add_argument("--raft-model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--train-sources", default="flowscape:train")
    p.add_argument("--train-split", default="train", help="raw-FlowScape split (local runs)")
    p.add_argument("--train-pairs", type=int, default=300)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--p7-json", default="", help="p7_learned_gate.json of the same checkpoint (optional)")
    args = p.parse_args()
    if args.predictor == "raft":
        run = build_raft_runner(args)
        ckpt = args.tv_checkpoint
    else:
        run = build_oslo_runner(args)
        ckpt = args.init_checkpoint

    # 1) pick tau* and s* on train pairs (official all-pixel EPE, pooled over pixels)
    targs = argparse.Namespace(**{**vars(args), "split": args.train_split, "sources": args.train_sources,
                                  "stride": args.train_stride, "max_pairs": args.train_pairs})
    sums, npx, start = {}, 0, time.time()
    for i, rec in enumerate(C.iter_pairs(targs)):
        pred = run(rec)
        gt = torch.from_numpy(np.ascontiguousarray(rec["flow"])).float()
        for name, cand in candidates(pred):
            sums[name] = sums.get(name, 0.0) + C.ee.official_maps(cand, gt)[0].sum().item()
        npx += gt.shape[0] * gt.shape[1]
        if (i + 1) % 100 == 0:
            print(f"train pair {i + 1} ({time.time() - start:.0f}s)", flush=True)
    train_epe = {k: v / npx for k, v in sums.items()}
    tau_best = min(TAUS, key=lambda t: train_epe[f"mag_{t:g}"])
    s_best = min(SCALES, key=lambda s: train_epe[f"shrink_{s:g}"])
    print("train EPE: " + "  ".join(f"{k} {v:.3f}" for k, v in train_epe.items()), flush=True)
    print(f"picked on train: tau* = {tau_best:g} deg, s* = {s_best:g}", flush=True)

    # 2) test: picked rows with all metrics, the full sweep without viewports
    main_sc, sweep_sc, start = C.Scorer(), C.Scorer(layouts=()), time.time()
    for i, rec in enumerate(C.iter_pairs(args)):
        pred = run(rec)
        flow = rec["flow"]
        deg = motion_deg(pred)
        main_sc.add("base", pred, flow)
        main_sc.add("mag_train", pred.masked_fill((deg < tau_best)[..., None], 0.0), flow)
        main_sc.add("shrink_train", pred * s_best, flow)
        for tau in TAUS:
            sweep_sc.add(f"mag_{tau:g}", pred.masked_fill((deg < tau)[..., None], 0.0), flow)
        if (i + 1) % 100 == 0:
            print(f"test pair {i + 1}: base {main_sc.per_pair['base'][-1]:.3f} "
                  f"mag_train {main_sc.per_pair['mag_train'][-1]:.3f} ({time.time() - start:.0f}s)", flush=True)
    summary = main_sc.summary(base_row="base")
    sweep = sweep_sc.summary()
    print(f"\nP8 magnitude-threshold control on {args.predictor} {ckpt} "
          f"(tau* = {tau_best:g} deg, s* = {s_best:g}, picked on {args.train_pairs} train pairs). " + C.REFERENCE)
    main_sc.print_table(summary)
    print("\ntest sweep (optimistic, tau picked on test):")
    sweep_sc.print_table(sweep)

    base = summary["base"]["official_all_epe"]
    fractions = {}
    if args.p7_json and Path(args.p7_json).exists():
        p7 = json.loads(Path(args.p7_json).read_text())["rows"]
        soft = p7["gate_soft"]["official_all_epe"]
        p7_base = p7["base"]["official_all_epe"]
        if abs(p7_base - base) > 0.01:
            print(f"WARNING: p7 base {p7_base:.3f} != this base {base:.3f} (different test set?)")
        gain = base - soft
        rows = {**{k: v["official_all_epe"] for k, v in summary.items()},
                **{k: v["official_all_epe"] for k, v in sweep.items()}}
        print(f"\nfraction of the learned gate's gain recovered (p7 gate_soft {soft:.3f}, gain {gain:.3f} px):")
        for k, v in rows.items():
            fractions[k] = (base - v) / gain if gain > 0 else float("nan")
            print(f"  {k:<14} {v:.3f} px   f = {fractions[k]:+.2f}")
    C.save(args.output_dir, {"args": vars(args), "tau_train": tau_best, "scale_train": s_best,
                             "train_epe": train_epe, "rows": summary, "sweep": sweep,
                             "fraction_of_p7_gain": fractions}, "p8_mag_sweep.json")


if __name__ == "__main__":
    main()
