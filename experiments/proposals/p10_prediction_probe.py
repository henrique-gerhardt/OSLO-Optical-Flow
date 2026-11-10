"""Prediction probe: is a model's flow a good motion field for predicting a frame on HEALPix nodes?

Direction under test (reports/Horizontes HEALPix para fluxo óptico.md): use the flow as the motion
of an on-sphere video codec. There the cost is counted per equal-area sample and the consumer
lives on HEALPix nodes, which are the conditions under which the OSLO line works in image coding.
This script trains nothing. It measures only the first thing such a codec needs: how much of the
frame the motion field predicts.

Both frames are sampled on a HEALPix grid (default r8, 786k nodes; bilinear ERP lookup with
longitude wrap, the same for every row). The flow is anchored at frame 1, so the prediction is a
backward warp: frame1_hat(x_i) = frame2(endpoint_i), with endpoint_i the flow endpoint of node i.
This is the prediction of frame 1 with frame 2 as the reference. The residual is
frame1(x_i) - frame1_hat(x_i), in 8-bit units, mean over RGB. HEALPix nodes have equal area, so a
plain mean over nodes is the sphere-weighted mean.

Rows on flowscape:test:
  zero       endpoint = the node itself (the codec's "skip" mode; the floor every row must beat)
  gt         GT endpoints, resampled to the nodes. A reference, not a strict ceiling: a row can go
             below it where the GT is not the best photometric match (occlusions, sky, near-pole GT)
  pred       the frozen predictor. OSLO: its node flow read at the grid nodes with the HEALPix
             stencil of the benchmark readout, no ERP detour. RAFT: its ERP flow resampled to the
             nodes (ambient displacement, bilinear).
  pred_erp   OSLO only: its ERP benchmark readout resampled exactly like the RAFT row. Shows what
             the ERP detour costs; not used for the reading.
  gt_rK, pred_rK   the same field reduced to a HEALPix rK motion field (--rate-levels, default 6 5 4)
             and read back at the grid nodes: chord displacements averaged over the NESTED children
             of each rK node, then interpolated with the HEALPix stencil of the benchmark readout.
             The number of motion vectors (12 * 4^K: 49152 / 12288 / 3072) is the rate proxy. OSLO
             writes an r6 field natively, so its pred row is already an r6-rate row and its pred_r6
             row measures what the reduction operator itself costs.

Columns: residual MSE and PSNR over all nodes, MSE by region (poles = |lat| > 45 deg, as in the
official protocol), by GT motion (static < --static-deg), in-raster / overshoot (nearest ERP
pixel), and "closed" = (zero - row) / (zero - gt), the share of the zero-to-GT gap a row closes
(above 100 % = below the GT residual).

Sanity: gt must be below zero, otherwise the probe (or the flow convention) is broken.

Pre-registered reading, on the MSE over all nodes of the pred row, for the fs30k pairs
(OSLO noso3_erppx vs RAFT-small fs, seeds 7 and 11):
  alive   OSLO MSE <= RAFT-small MSE on both seeds. Next step: price the motion field (rate).
  dead    RAFT-small MSE lower by >= 5 % on both seeds. OSLO's flow has nothing to offer a codec.
  tie     anything else. The direction then rests only on the rate of the motion field.

Pre-registered reading at matched rate (r6 motion field on both sides): OSLO pred against
RAFT-small pred_r6, same thresholds. RAFT-small ahead at full resolution but not at r6 means its
advantage is field density, which a codec pays for. RAFT-small ahead at r6 too closes the direction.
If OSLO's pred_r6 is more than 5 % above its pred, the reduction operator is lossy on its own and
the matched-rate comparison is biased against RAFT-small by about that much.

Limits: one reference frame, no residual coding, JPEG frames. The rate proxy is a vector count, not
an entropy: a smooth field costs fewer bits than a noisy one at the same count. The reduction is a
plain average, not a rate-distortion-optimal choice of vectors. A lower residual is necessary for
the direction, not sufficient.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

import common as C
from p6_polar_band_head import build_raft_runner
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import (
    bilinear_node_weights,
    erp_pixel_directions,
    node_weights_at_dirs,
    nodes_to_endpoints,
    nodes_to_erp_flow,
)
from spherical_flow.geometry import points_to_equirectangular_pixels
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes


def sample_erp(img: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of an ERP field [H, W, C] at unit directions [M, 3] -> [M, C] (lon wraps, lat clamps)."""
    h, w = img.shape[:2]
    u, v = points_to_equirectangular_pixels(dirs.to(img.dtype), h, w)
    v = v.clamp(0.0, h - 1.0)
    u0, v0 = torch.floor(u), torch.floor(v)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    u0 = u0.long() % w
    u1 = (u0 + 1) % w
    v0 = v0.long().clamp(0, h - 1)
    v1 = (v0 + 1).clamp(max=h - 1)
    top = img[v0, u0] * (1 - fu) + img[v0, u1] * fu
    bot = img[v1, u0] * (1 - fu) + img[v1, u1] * fu
    return top * (1 - fv) + bot * fv


def erp_flow_to_node_endpoints(flow, grid: torch.Tensor) -> torch.Tensor:
    """ERP pixel flow [H, W, 2] -> endpoints [M, 3] (fp64) at the grid nodes.

    The ambient displacement (endpoint - pixel direction) is interpolated, not (du, dv): near a pole
    a small motion is a large du, and du does not interpolate.
    """
    flow = np.asarray(flow, dtype=np.float32)
    h, w = flow.shape[:2]
    disp = (C.gt_endpoints(flow) - C.pixel_dirs64(h, w)).reshape(h, w, 3)
    end = grid.double() + sample_erp(disp, grid.double())
    return end / end.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def reduce_to_level(end: torch.Tensor, grid: torch.Tensor, level_grid: torch.Tensor, weights) -> torch.Tensor:
    """Endpoints [M, 3] on the NESTED grid -> endpoints of the same field carried by a coarser HEALPix level.

    Chord displacements (endpoint - node) are averaged over the NESTED children of each coarse node,
    made tangent there, and interpolated back to the grid nodes with ``weights`` (the stencil of
    ``node_weights_at_dirs`` for the coarse level at the grid nodes).
    """
    n_coarse = level_grid.shape[0]
    disp = (end - grid).reshape(n_coarse, -1, 3).mean(dim=1)
    disp = disp - (disp * level_grid).sum(-1, keepdim=True) * level_grid
    idx, w = weights
    out = grid + (disp[idx] * w.double().unsqueeze(-1)).sum(dim=0)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nearest_pixel(grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Row-major index of the ERP pixel nearest to each node."""
    u, v = points_to_equirectangular_pixels(grid.float(), h, w)
    return torch.round(v).long().clamp(0, h - 1) * w + torch.round(u).long() % w


def build_oslo_node_runner(args, grid: torch.Tensor):
    """Frozen OSLO -> function(record) -> {"pred": endpoints at the grid nodes, "pred_erp": ERP flow}.

    Same model path as p6.build_oslo_runner; the node flow is also read at the grid nodes.
    """
    sys.argv = ["x", "--predictor", "oslo", "--init-checkpoint", args.init_checkpoint,
                "--pyramid-cache", args.pyramid_cache, "--eval-iters", str(args.eval_iters),
                "--device", args.device]
    eargs = C.ee.parse_args()
    C.ee.adopt_checkpoint_grids(eargs)
    device = torch.device(args.device)
    model, pyramid = C.ee.build_oslo(eargs, device)
    model.eval()
    sup = healpix_unit_vectors(eargs.resolution)
    sbe, sbn = tangent_basis(sup)
    ret = pyramid.retina_level.points.cpu()
    rbe, rbn = tangent_basis(ret)
    grid32 = grid.float()
    grid_wts = node_weights_at_dirs(eargs.resolution, grid.double().numpy())
    cache = {}
    print(f"oslo output grid r{eargs.resolution} ({sup.shape[0]} nodes) -> prediction grid {grid.shape[0]} nodes",
          flush=True)

    def run(rec):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        if (h, w) not in cache:
            cache[(h, w)] = (bilinear_node_weights(eargs.resolution, h, w), erp_pixel_directions(h, w))
        wts, dirs = cache[(h, w)]
        gt = torch.from_numpy(np.ascontiguousarray(flow)).float()
        nodes = sample_pair_to_nodes(_to_chw_free_float(rec["frame1"]), _to_chw_free_float(rec["frame2"]), gt,
                                     torch.from_numpy(np.ascontiguousarray(rec["valid"])), ret, rbe, rbn,
                                     target_points=sup)
        with torch.no_grad():
            node = model(nodes["frame1"][None].to(device), nodes["frame2"][None].to(device), pyramid,
                         iters=eargs.eval_iters)[-1][0].float().cpu()
        erp = nodes_to_erp_flow(node, sup, sbe, sbn, h, w, wts, pixel_dirs=dirs)
        return {"pred": nodes_to_endpoints(node, sbe, sbn, grid_wts, grid32).double(),
                "pred_erp": erp_flow_to_node_endpoints(erp.numpy(), grid)}

    return run


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--predictor", default="oslo", choices=["oslo", "raft", "none"],
                   help="none = zero and gt rows only (sanity run)")
    p.add_argument("--init-checkpoint", default="", help="OSLO checkpoint (--predictor oslo)")
    p.add_argument("--tv-checkpoint", default="", help="trained RAFT raft_erp.pt (--predictor raft)")
    p.add_argument("--raft-model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--grid-resolution", type=int, default=8, help="HEALPix resolution of the prediction grid")
    p.add_argument("--static-deg", type=float, default=0.25)
    p.add_argument("--rate-levels", type=int, nargs="*", default=[6, 5, 4],
                   help="HEALPix levels of the reduced motion fields (gt_rK, pred_rK rows); empty = none")
    args = p.parse_args()

    grid = healpix_unit_vectors(args.grid_resolution).double()
    lat = torch.asin(grid[:, 2].clamp(-1.0, 1.0))
    poles = lat.abs() > math.radians(45.0)
    levels = {}
    for k in args.rate_levels:
        if not 0 <= k < args.grid_resolution:
            raise SystemExit(f"--rate-levels {k} must be below --grid-resolution {args.grid_resolution}")
        levels[k] = (healpix_unit_vectors(k).double(), node_weights_at_dirs(k, grid.numpy()))
    if args.predictor == "oslo":
        if not args.init_checkpoint:
            raise SystemExit("--predictor oslo needs --init-checkpoint")
        run = build_oslo_node_runner(args, grid)
    elif args.predictor == "raft":
        if not args.tv_checkpoint:
            raise SystemExit("--predictor raft needs --tv-checkpoint")
        raft = build_raft_runner(args)
        run = lambda rec: {"pred": erp_flow_to_node_endpoints(raft(rec).numpy(), grid)}  # noqa: E731
    else:
        run = lambda rec: {}  # noqa: E731
    ckpt_name = {"oslo": args.init_checkpoint, "raft": args.tv_checkpoint, "none": "-"}[args.predictor]

    sums = {}      # row -> region -> [sum sq, sum abs, n]
    per_pair = {}  # row -> list of MSE over all nodes
    start = time.time()
    for i, rec in enumerate(C.iter_pairs(args)):
        f1 = torch.from_numpy(np.ascontiguousarray(rec["frame1"])).float()
        f2 = torch.from_numpy(np.ascontiguousarray(rec["frame2"])).float()
        h, w = f1.shape[:2]
        flow = np.asarray(rec["flow"], dtype=np.float32)
        gt_t = torch.from_numpy(np.ascontiguousarray(flow))
        target = sample_erp(f1, grid)
        ends = {"zero": grid, "gt": erp_flow_to_node_endpoints(flow, grid)}
        ends.update(run(rec))
        for row in ("gt", "pred"):
            for k, (level_grid, level_wts) in levels.items():
                if row in ends:
                    ends[f"{row}_r{k}"] = reduce_to_level(ends[row], grid, level_grid, level_wts)

        cosang = (grid * ends["gt"]).sum(-1)
        sinang = torch.linalg.cross(grid, ends["gt"]).norm(dim=-1)
        static = torch.rad2deg(torch.atan2(sinang, cosang)) < args.static_deg
        over = C.ee.overshoot_masks(gt_t)["overshoot"].reshape(-1)[nearest_pixel(grid, h, w)]
        regions = {"all": torch.ones_like(poles), "poles": poles, "equator": ~poles, "static": static,
                   "moving": ~static, "overshoot": over, "inraster": ~over}
        for row, end in ends.items():
            res = target - sample_erp(f2, end)
            sq, ab = res.pow(2).mean(-1), res.abs().mean(-1)
            acc = sums.setdefault(row, {})
            for region, m in regions.items():
                s = acc.setdefault(region, [0.0, 0.0, 0.0])
                s[0] += sq[m].sum().item()
                s[1] += ab[m].sum().item()
                s[2] += m.sum().item()
            per_pair.setdefault(row, []).append(sq.mean().item())
        if (i + 1) % 50 == 0:
            print(f"test pair {i + 1}: " + " ".join(f"{r} {v[-1]:.2f}" for r, v in per_pair.items())
                  + f" ({time.time() - start:.0f}s)", flush=True)

    if not per_pair:
        raise SystemExit("no pairs")
    rows = {}
    for row, acc in sums.items():
        m = {}
        for region, (sq, ab, n) in acc.items():
            if n > 0:
                m[f"{region}_mse"] = sq / n
                m[f"{region}_mae"] = ab / n
                m[f"{region}_psnr"] = 10.0 * math.log10(255.0 ** 2 / max(sq / n, 1e-12))
        m["node_fraction"] = {region: n / acc["all"][2] for region, (_, _, n) in acc.items()}
        d = np.array(per_pair[row]) - np.array(per_pair["zero"])
        m["paired_delta_vs_zero"] = float(d.mean())
        m["paired_delta_se"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        m["paired_better_than_zero_fraction"] = float((d < 0).mean())
        rows[row] = m
    gap = rows["zero"]["all_mse"] - rows["gt"]["all_mse"]
    for m in rows.values():
        m["gap_closed"] = (rows["zero"]["all_mse"] - m["all_mse"]) / gap if gap > 0 else float("nan")

    print(f"\nP10 prediction probe on {args.predictor} {ckpt_name}: r{args.grid_resolution} grid, "
          f"{len(per_pair['zero'])} pairs, residual MSE in 8-bit units")
    cols = [("MSE all", "all_mse"), ("PSNR", "all_psnr"), ("MAE", "all_mae"), ("equator", "equator_mse"),
            ("poles", "poles_mse"), ("static", "static_mse"), ("moving", "moving_mse"),
            ("inraster", "inraster_mse"), ("overshoot", "overshoot_mse")]
    print(f"{'row':<10}" + "".join(f"{c:>10}" for c, _ in cols) + f"{'closed':>10}{'pairs<0':>10}{'vectors':>10}")
    vectors = {f"{row}_r{k}": 12 * 4 ** k for row in ("gt", "pred") for k in levels}
    for row, m in rows.items():
        m["motion_vectors"] = vectors.get(row)
    for row, m in rows.items():
        print(f"{row:<10}" + "".join(f"{m.get(k, float('nan')):>10.2f}" for _, k in cols)
              + f"{100 * m['gap_closed']:>9.1f}%{100 * m['paired_better_than_zero_fraction']:>9.1f}%"
              + f"{(m['motion_vectors'] or '-'):>10}")
    frac = rows["zero"]["node_fraction"]
    print("node fractions: " + " ".join(f"{k} {v:.3f}" for k, v in frac.items() if k != "all"))
    if gap <= 0:
        print("SANITY FAILED: the gt row does not beat zero; do not read the pred rows.")
    C.save(args.output_dir, {"args": vars(args), "rows": rows, "per_pair_mse": per_pair},
           "p10_prediction_probe.json")


if __name__ == "__main__":
    main()
