"""Proposal 3, minimal: is a static gate worth training? (training-free ceiling + cheap proxy)

Runs an OSLO checkpoint, reads its flow out to ERP, and scores:
  base             the network as is (= the benchmark row)
  oracle_gate      prediction zeroed where the GT moves less than --static-deg: the ceiling
                   of ANY static/motion gate on this checkpoint
  mag_gate_<tau>   prediction zeroed where the PREDICTED motion is below tau degrees: a gate
                   realizable without training
plus the AUROC of -|pred| as a static detector and the static-pixel fraction.

Kill (pre-registered in the report): oracle_gate improves EPE by < 5 % -> a gate cannot pay.
Go: oracle_gate >= 10 % and the gap to the best mag_gate is large (a learned gate has room).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

import common as C
from spherical_flow import tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, nodes_to_erp_flow
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes


def auroc(score, label):
    """P(score_pos > score_neg) via ranks; label True = positive."""
    s = score.double()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    npos = label.sum().item()
    nneg = label.numel() - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--static-deg", type=float, default=0.25)
    p.add_argument("--taus", default="0.05,0.1,0.2,0.3,0.5")
    args = p.parse_args()
    argv = ["x", "--predictor", "oslo", "--init-checkpoint", args.init_checkpoint,
            "--pyramid-cache", args.pyramid_cache, "--eval-iters", str(args.eval_iters),
            "--device", args.device]
    sys.argv = argv
    eargs = C.ee.parse_args()
    C.ee.adopt_checkpoint_grids(eargs)
    device = torch.device(args.device)
    model, pyramid = C.ee.build_oslo(eargs, device)
    from spherical_flow import healpix_unit_vectors

    sup = healpix_unit_vectors(eargs.resolution)
    sbe, sbn = tangent_basis(sup)
    ret = pyramid.retina_level.points.cpu()
    rbe, rbn = tangent_basis(ret)
    taus = [float(x) for x in args.taus.split(",")]
    scorer = C.Scorer()
    aucs, static_frac = [], []
    cache = {}
    start = time.time()
    for k, rec in enumerate(C.iter_pairs(args)):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        if (h, w) not in cache:
            cache[(h, w)] = (bilinear_node_weights(eargs.resolution, h, w), erp_pixel_directions(h, w), C.pixel_dirs64(h, w))
        wts, dirs, dirs64 = cache[(h, w)]
        gt = torch.from_numpy(np.ascontiguousarray(flow)).float()
        nodes = sample_pair_to_nodes(_to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"]), gt,
                                     torch.from_numpy(np.ascontiguousarray(rec["valid"])), ret, rbe, rbn,
                                     target_points=sup)
        with torch.no_grad():
            node = model(nodes["frame1"][None].to(device), nodes["frame2"][None].to(device), pyramid,
                         iters=eargs.eval_iters)[-1][0].float().cpu()
        pred = nodes_to_erp_flow(node, sup, sbe, sbn, h, w, wts, pixel_dirs=dirs)
        gt_deg = torch.rad2deg(C.ee.official_maps(torch.zeros_like(gt), gt)[1])  # GT angular motion
        pr_deg = torch.rad2deg(C.ee.official_maps(torch.zeros_like(pred), pred)[1])
        static = gt_deg < args.static_deg
        static_frac.append(static.float().mean().item())
        aucs.append(auroc(-pr_deg.reshape(-1), static.reshape(-1)))
        extra = {"static": static, "moving": ~static}
        scorer.add("base", pred, flow, extra)
        scorer.add("oracle_gate", pred.masked_fill(static[..., None], 0.0), flow, extra)
        for tau in taus:
            scorer.add(f"mag_gate_{tau:g}", pred.masked_fill((pr_deg < tau)[..., None], 0.0), flow, extra)
        print(f"pair {k + 1} {rec['name']}: base {scorer.per_pair['base'][-1]:.3f} "
              f"oracle {scorer.per_pair['oracle_gate'][-1]:.3f} static {static_frac[-1]:.2f} "
              f"auroc {aucs[-1]:.3f} ({time.time() - start:.0f}s)", flush=True)
    summary = scorer.summary(base_row="base")
    base = summary["base"]["official_all_epe"]
    print(f"\nP3 static-gate probe (static = GT < {args.static_deg} deg; static fraction "
          f"{np.mean(static_frac):.3f}; AUROC of -|pred| {np.nanmean(aucs):.3f}). " + C.REFERENCE)
    scorer.print_table(summary, extra_cols=[("static", "official_static_epe"), ("moving", "official_moving_epe")])
    for row, m in summary.items():
        print(f"  {row:<22} EPE change vs base {100 * (m['official_all_epe'] - base) / base:+.1f}%")
    C.save(args.output_dir, {"args": vars(args), "rows": summary, "auroc_mean": float(np.nanmean(aucs)),
                             "static_fraction": float(np.mean(static_frac))}, "p3_gate_probe.json")


if __name__ == "__main__":
    main()
