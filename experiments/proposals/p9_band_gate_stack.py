"""Stack the two trained heads: P6 polar-band head + P7 learned static gate, on one frozen predictor.

P6 fixes the near-nadir overshoot pixels, which are OSLO's main deficit. P7 shrinks wrong
predictions on static pixels, which are RAFT-small's main deficit. Each head was trained alone on
the frozen predictor's readout. This script trains nothing: it loads band_head.pt (p6 / p6r run)
and gate_net.pt (p7 / p7r run) for the same checkpoint and scores the combinations on
flowscape:test. Both heads see the base readout, exactly as in training.

Rows:
  base           the frozen predictor (= its benchmark row)
  band           P6 head on the K polar rows
  gate           P7 gate_soft, pred * (1 - g)
  stack          band rows from the P6 head, gate_soft on all other rows
  stack_all      P6 head, then gate_soft on every pixel (band rows included)

Pre-registered reading, on stack, for the fs30k pairs (OSLO noso3_erppx vs RAFT-small fs, seeds 7, 11):
  OSLO ahead    stacked OSLO EPE < stacked RAFT-small EPE on both seeds.
  RAFT ahead    stacked RAFT-small EPE < stacked OSLO EPE on both seeds.
  otherwise     tie within seed spread.
stack_all is reported but not used for the reading.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common as C
from p6_polar_band_head import BandHead, bands, build_predictor, predict, unband
from p7_learned_gate import GateNet, gate_inputs, logit_at, motion_deg


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--predictor", default="oslo", choices=["oslo", "raft"])
    p.add_argument("--init-checkpoint", default="", help="OSLO checkpoint (--predictor oslo)")
    p.add_argument("--tv-checkpoint", default="", help="trained RAFT raft_erp.pt (--predictor raft)")
    p.add_argument("--raft-model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--band-head", required=True, help="band_head.pt of the p6/p6r run on this checkpoint")
    p.add_argument("--gate-net", required=True, help="gate_net.pt of the p7/p7r run on this checkpoint")
    p.add_argument("--static-deg", type=float, default=0.25)
    args = p.parse_args()
    device = torch.device(args.device)
    run = build_predictor(args)
    ckpt_name = args.tv_checkpoint if args.predictor == "raft" else args.init_checkpoint

    bstate = torch.load(args.band_head, map_location=device)
    head = BandHead(c=bstate["args"]["width"]).to(device)
    head.load_state_dict(bstate["state_dict"])
    head.eval()
    k = bstate["k"]
    gstate = torch.load(args.gate_net, map_location=device)
    net = GateNet(c=gstate["args"]["width"]).to(device)
    net.load_state_dict(gstate["state_dict"])
    net.eval()

    scorer = C.Scorer()
    start = time.time()
    for i, rec in enumerate(C.iter_pairs(args)):
        pred = run(rec)
        if not torch.is_tensor(pred):
            pred = torch.from_numpy(np.asarray(pred))
        pred = pred.float()
        flow = rec["flow"]
        h, w = pred.shape[:2]
        band = torch.zeros((h, w), dtype=torch.bool)
        band[:k] = True
        band[h - k:] = True
        static = motion_deg(torch.from_numpy(np.ascontiguousarray(flow)).float()) < args.static_deg
        extra = {"band": band, "static": static, "moving": ~static}

        with torch.no_grad():
            g = torch.sigmoid(logit_at(net, gate_inputs(rec, pred)[None].float().to(device), h, w)[0, 0].cpu())
        banded = torch.from_numpy(unband(pred.numpy(), predict(head, bands(rec, pred.numpy(), k), device), k))
        gated = pred * (1 - g)[..., None]
        stack = torch.where(band[..., None], banded, gated)

        scorer.add("base", pred, flow, extra)
        scorer.add("band", banded, flow, extra)
        scorer.add("gate", gated, flow, extra)
        scorer.add("stack", stack, flow, extra)
        scorer.add("stack_all", banded * (1 - g)[..., None], flow, extra)
        if (i + 1) % 50 == 0:
            pp = scorer.per_pair
            print(f"test pair {i + 1}: base {pp['base'][-1]:.3f} band {pp['band'][-1]:.3f} "
                  f"gate {pp['gate'][-1]:.3f} stack {pp['stack'][-1]:.3f} ({time.time() - start:.0f}s)", flush=True)

    summary = scorer.summary(base_row="base")
    base = summary["base"]["official_all_epe"]
    print(f"\nP9 band head + gate stack (K={k} rows) on {args.predictor} {ckpt_name}. " + C.REFERENCE)
    scorer.print_table(summary, extra_cols=[("band", "official_band_epe"), ("static", "official_static_epe"),
                                            ("moving", "official_moving_epe")])
    for row, m in summary.items():
        print(f"  {row:<10} EPE change vs base {100 * (m['official_all_epe'] - base) / base:+.1f}%")
    C.save(args.output_dir, {"args": vars(args), "k": k, "rows": summary}, "p9_band_gate_stack.json")


if __name__ == "__main__":
    main()
