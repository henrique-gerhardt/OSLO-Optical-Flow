"""Probe B: per-pair, training-free refinement of an OSLO prediction on the sphere.

The network's flow (supervision grid, r6) is moved to a finer HEALPix grid (default
r7) and optimized per pair against an unsupervised objective:

  data    census distance between frame 1 at node x and frame 2 at the endpoint
          e(x) = expmap(x, t(x)), over the 8 HEALPix neighbors (robust to
          brightness change); frame 2 is read from the full-resolution ERP raster
  smooth  first-order smoothness of the ambient tangent field across HEALPix edges,
          in units of node spacing, down-weighted across frame-1 edges
  anchor  optional pull towards the network's field
  occl    optional forward-backward mask (second network pass on frame2 -> frame1)

Each stage blurs both frames (Gaussian on ERP, sigma in ERP px) to widen the
convergence basin. Scored on the same pairs, in the PriOr-Flow protocol (ERP-px EPE,
SEPE, displacement bins) and the viewport EPE:
  base     the network output, read out from its supervision grid (= benchmark row)
  init     the same field resampled to the refinement grid, not optimized
  <cfg>    the refined field, one row per --smooth-weights value

--synthetic-rotation-deg D replaces frame 2 with frame 1 rotated by D degrees about a
fixed tilted axis (exact GT), and --init zero starts from zero flow: the self-test
that the objective and the optimizer recover a known field.

Example (box):
    python run_tta_refine_probe.py --shards /data/shards --sources flowscape:test \\
      --max-pairs 100 --init-checkpoint /outputs/.../oslo_raft.pt \\
      --output-dir /outputs/tta_probe --device cuda
Unknown flags are passed to run_epe_eval's parser (grids, pyramid cache, iters).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import run_epe_eval as ee
from run_epe_smoke import rotation_matrix
from run_raft_shard_baseline import get_device, iter_source_records, parse_sources
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import (
    bilinear_node_weights,
    erp_pixel_directions,
    node_weights_at_dirs,
    nodes_to_endpoints,
    nodes_to_erp_flow,
    wrap_px,
)
from spherical_flow.geometry import expmap, logmap, points_to_equirectangular_pixels
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes
from spherical_flow.viewport_metrics import (
    accumulate_viewport,
    build_viewports,
    erp_flow_endpoints,
    finalize_viewport,
    viewport_epe,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--refine-resolution", type=int, default=7)
    p.add_argument("--stages", default="2:60,1:60,0:60",
                   help="Comma list of blur_sigma_px:iters, coarse to fine.")
    p.add_argument("--lr", type=float, default=0.2,
                   help="Adam step in units of refinement-node spacing.")
    p.add_argument("--smooth-weights", default="0.1,0.5,2.0",
                   help="Comma list; one refined row per value.")
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--edge-alpha", type=float, default=10.0,
                   help="Smoothness edge weight exp(-alpha * |dI1|), I in [0, 1].")
    p.add_argument("--occlusion", default="none", choices=["none", "fb"])
    p.add_argument("--init", default="oslo", choices=["oslo", "zero"])
    p.add_argument("--synthetic-rotation-deg", type=float, default=0.0)
    p.add_argument("--coarsest-level", type=int, default=3,
                   help="Coarsest quad-tree level of the residual parametrization.")
    p.add_argument("--pair-stride", type=int, default=1,
                   help="Score every k-th pair (spreads --max-pairs over the split).")
    p.add_argument("--viewport-layouts-probe", default="cube,paper")
    my, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    base = ee.parse_args()
    return my, base


def sample_erp(img: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear, seam-wrapped read of an ERP image [C, H, W] at unit dirs [N, 3] -> [N, C]."""
    c, h, w = img.shape
    x, y, z = dirs.unbind(-1)
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x * x + y * y).clamp_min(1e-9))
    u = (lon + math.pi) / (2 * math.pi) * w - 0.5 + 1.0  # +1: one wrapped column on the left
    v = (math.pi / 2 - lat) / math.pi * h - 0.5
    padded = torch.cat([img[..., -1:], img, img[..., :1]], dim=-1)
    grid = torch.stack([(u + 0.5) / (w + 2) * 2 - 1, (v + 0.5) / h * 2 - 1], dim=-1)
    out = F.grid_sample(padded[None], grid.view(1, 1, -1, 2).to(img.dtype), mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out[0, :, 0].t()


def blur_erp(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=img.dtype, device=img.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = img.shape[0]
    x = F.pad(img[None], (r, r, 0, 0), mode="circular")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0]


def census(i_center: torch.Tensor, i_nbr: torch.Tensor) -> torch.Tensor:
    d = i_nbr - i_center[:, None]
    return d / torch.sqrt(0.81 + d * d)


def neighbors(resolution: int, device) -> tuple:
    from astropy_healpix import neighbours
    nside = 1 << resolution
    n = 12 * nside * nside
    nbr = torch.from_numpy(np.ascontiguousarray(neighbours(np.arange(n), nside, order="nested").T)).long()
    valid = nbr >= 0
    nbr = torch.where(valid, nbr, torch.arange(n).unsqueeze(1).expand_as(nbr))
    return nbr.to(device), valid.to(device)


def ambient(points, comps, be, bn):
    return comps[:, :1] * be + comps[:, 1:] * bn


def refine(t0, pts, nbr, nvalid, g1_stages, g2_stages, stages, spacing, ws, wa, alpha, mask):
    """Residual = sum over quad-tree levels of per-node ambient offsets (nearest parent).

    Coarse levels pool the data gradient of all their descendants, so the residual
    moves coherently where per-node gradients are noise (textureless regions).
    """
    n = pts.shape[0]
    res = int(round(math.log(n / 12, 4)))
    lo = min(COARSEST, res)
    child = torch.arange(n, device=pts.device)
    deltas = [torch.zeros(12 * 4 ** l, 3, device=pts.device, requires_grad=True)
              for l in range(lo, res + 1)]
    parents = [child // (4 ** (res - l)) for l in range(lo, res + 1)]
    opt = torch.optim.Adam(deltas, lr=0.0)

    def field():
        t = t0 + sum(d[pi] for d, pi in zip(deltas, parents))
        return t - (t * pts).sum(-1, keepdim=True) * pts

    for (sigma, iters), g1, g2 in zip(stages, g1_stages, g2_stages):
        i1 = sample_erp(g1, pts)[:, 0]
        c1 = census(i1, i1[nbr])
        edge = torch.exp(-alpha * (i1[nbr] - i1[:, None]).abs()) * nvalid
        for group in opt.param_groups:
            group["lr"] = LR * spacing
        for _ in range(iters):
            t = field()
            e = expmap(pts, t[None])[0]
            i2 = sample_erp(g2, e)[:, 0]
            c2 = census(i2, i2[nbr])
            dc = (c1 - c2) ** 2
            data = ((dc / (0.1 + dc)) * nvalid).sum(-1) / nvalid.sum(-1).clamp_min(1)
            data = (data * mask).sum() / mask.sum().clamp_min(1)
            diff = t[nbr] - t[:, None]
            diff = diff - (diff * pts[:, None]).sum(-1, keepdim=True) * pts[:, None]
            sm = torch.sqrt((diff / spacing).pow(2).sum(-1) + 1e-6)
            smooth = (sm * edge).sum() / edge.sum().clamp_min(1)
            loss = data + ws * smooth
            if wa > 0:
                loss = loss + wa * torch.sqrt(((t - t0) / spacing).pow(2).sum(-1) + 1e-6).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return field()


LR = 0.2
COARSEST = 3


def main() -> None:
    global LR, COARSEST
    my, args = parse_args()
    LR, COARSEST = my.lr, my.coarsest_level
    ee.adopt_checkpoint_grids(args) if (my.init == "oslo" and args.init_checkpoint) else None
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [(float(a), int(b)) for a, b in (s.split(":") for s in my.stages.split(","))]
    smooth_ws = [float(x) for x in my.smooth_weights.split(",") if x]
    layouts = [x for x in my.viewport_layouts_probe.split(",") if x]

    model = pyramid = None
    sup_res = args.resolution
    if my.init == "oslo":
        model, pyramid = ee.build_oslo(args, device)
        retina_points = pyramid.retina_level.points.cpu()
        retina_be, retina_bn = tangent_basis(retina_points)
    sup_pts = healpix_unit_vectors(sup_res)
    sup_be, sup_bn = tangent_basis(sup_pts)
    R = my.refine_resolution
    pts = healpix_unit_vectors(R).to(device)
    be, bn = tangent_basis(pts)
    nbr, nvalid = neighbors(R, device)
    nvalid = nvalid.float()
    spacing = math.sqrt(4 * math.pi / pts.shape[0])
    w_sup_at_R = tuple(x.to(device) for x in node_weights_at_dirs(sup_res, pts.cpu().numpy()))
    print(f"refine r{R} ({pts.shape[0]} nodes, spacing {math.degrees(spacing):.3f} deg) "
          f"stages={stages} lr={LR} smooth={smooth_ws} anchor={my.anchor_weight} "
          f"occlusion={my.occlusion} init={my.init}", flush=True)

    rows = ["base", "init"] + [f"smooth{w:g}" for w in smooth_ws]
    official: Dict[str, Dict[str, list]] = {r: {} for r in rows}
    vp_tot = {r: {L: ({}, {}) for L in layouts} for r in rows}
    per_pair: Dict[str, List[float]] = {r: [] for r in rows}
    cache = {}
    start = time.time()
    seen = 0
    for k, record in enumerate(iter_source_records(Path(args.shards), parse_sources(args.sources))):
        if k % my.pair_stride:
            continue
        img1 = torch.from_numpy(np.ascontiguousarray(record["frame1"])).float().permute(2, 0, 1) / 255
        img2 = torch.from_numpy(np.ascontiguousarray(record["frame2"])).float().permute(2, 0, 1) / 255
        gt = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
        valid = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        H, W = gt.shape[:2]
        if my.synthetic_rotation_deg:
            rot = rotation_matrix(torch.tensor([-0.3, 1.0, 0.8]), my.synthetic_rotation_deg)
            dirs = erp_pixel_directions(H, W)
            # frame2(R d) = frame1(d): frame2 at pixel dir q reads frame1 at R^T q
            img2 = sample_erp(img1, dirs @ rot).t().reshape(3, H, W)
            u2, v2 = points_to_equirectangular_pixels(dirs @ rot.T, H, W)
            u1 = torch.arange(W, dtype=u2.dtype).repeat(H)
            v1 = torch.arange(H, dtype=v2.dtype).repeat_interleave(W)
            gt = torch.stack([wrap_px(u2 - u1, W), v2 - v1], -1).reshape(H, W, 2)
            valid = torch.ones(H, W, dtype=torch.bool)
        key = (H, W)
        if key not in cache:
            cache[key] = {
                "w_sup": bilinear_node_weights(sup_res, H, W),
                "w_R": bilinear_node_weights(R, H, W),
                "dirs": erp_pixel_directions(H, W),
                "official": ee.official_region_masks(H, W),
                "vp": {L: build_viewports(L, H, W) for L in layouts},
            }
            for L in layouts:
                vp = cache[key]["vp"][L]
                vp["w_sup"] = node_weights_at_dirs(sup_res, vp["dirs"].numpy())
                vp["w_R"] = node_weights_at_dirs(R, vp["dirs"].numpy())
        c = cache[key]

        if my.init == "oslo":
            f1 = _to_chw_free_float(record["frame1"])
            f2 = img2.permute(1, 2, 0).contiguous() if my.synthetic_rotation_deg else _to_chw_free_float(record["frame2"])
            nodes = sample_pair_to_nodes(f1, f2, gt, valid, retina_points, retina_be, retina_bn,
                                         target_points=sup_pts)
            with torch.no_grad():
                a, b = nodes["frame1"][None].to(device), nodes["frame2"][None].to(device)
                node_pred = model(a, b, pyramid, iters=args.eval_iters)[-1][0].float()
                back = model(b, a, pyramid, iters=args.eval_iters)[-1][0].float() \
                    if my.occlusion == "fb" else None
        else:
            node_pred = torch.zeros(sup_pts.shape[0], 2, device=device)
            back = None

        e0 = nodes_to_endpoints(node_pred, sup_be.to(device), sup_bn.to(device), w_sup_at_R, pts)
        t0 = ambient(pts, logmap(pts, e0, be, bn)[0], be, bn)
        mask = torch.ones(pts.shape[0], device=device)
        if back is not None:
            w_e0 = tuple(x.to(device) for x in node_weights_at_dirs(sup_res, e0.cpu().numpy()))
            eb = nodes_to_endpoints(back, sup_be.to(device), sup_bn.to(device), w_e0, e0)
            fb = torch.acos((eb * pts).sum(-1).clamp(-1, 1))
            mag = t0.norm(dim=-1)
            mask = (fb < 0.05 * 2 * mag + 0.5 * spacing).float()

        g1s = [blur_erp(img1.to(device).mean(0, keepdim=True), s) for s, _ in stages]
        g2s = [blur_erp(img2.to(device).mean(0, keepdim=True), s) for s, _ in stages]

        fields = {"init": t0}
        for w in smooth_ws:
            fields[f"smooth{w:g}"] = refine(t0, pts, nbr, nvalid, g1s, g2s, stages, spacing, w,
                                            my.anchor_weight, my.edge_alpha, mask)

        # base: supervision-grid readout (identical to the benchmark row)
        preds = {"base": (nodes_to_erp_flow(node_pred.cpu(), sup_pts, sup_be, sup_bn, H, W, c["w_sup"],
                                            pixel_dirs=c["dirs"]), ("sup", node_pred.cpu()))}
        for name, t in fields.items():
            comps = logmap(pts, expmap(pts, t[None])[0], be, bn)[0].cpu()
            preds[name] = (nodes_to_erp_flow(comps, pts.cpu(), be.cpu(), bn.cpu(), H, W, c["w_R"],
                                             pixel_dirs=c["dirs"]), ("R", comps))
        regions = dict(c["official"])
        regions.update(ee.displacement_bin_masks(gt))
        for name, (erp, (grid, comps)) in preds.items():
            epe, sepe = ee.official_maps(erp, gt)
            per_pair[name].append(epe.mean().item())
            for region, m in regions.items():
                for kind, vals in (("epe", epe), ("sepe", sepe)):
                    k = f"{region}_{kind}"
                    s, n = official[name].get(k, (0.0, 0.0))
                    official[name][k] = (s + vals[m].sum().item(), n + m.sum().item())
            for L in layouts:
                vp = c["vp"][L]
                if grid == "sup":
                    end = nodes_to_endpoints(comps, sup_be, sup_bn, vp["w_sup"], vp["dirs"].float())
                else:
                    end = nodes_to_endpoints(comps, be.cpu(), bn.cpu(), vp["w_R"], vp["dirs"].float())
                accumulate_viewport(viewport_epe(end, erp_flow_endpoints(gt, vp), vp), vp, *vp_tot[name][L])
        seen += 1
        print(f"pair {seen}: " + "  ".join(f"{r} {per_pair[r][-1]:.3f}" for r in rows)
              + f"  ({time.time() - start:.0f}s)", flush=True)
        if args.max_pairs is not None and seen >= args.max_pairs:
            break

    summary = {}
    for r in rows:
        m = {f"official_{k}": s / n for k, (s, n) in official[r].items() if n > 0}
        for L in layouts:
            m.update(finalize_viewport(*vp_tot[r][L], L))
        diffs = np.array(per_pair[r]) - np.array(per_pair["base"])
        m["paired_epe_delta_mean"] = float(diffs.mean())
        m["paired_epe_delta_se"] = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        m["paired_improved_fraction"] = float((diffs < 0).mean())
        summary[r] = m
    print("\nrow            EPE all  EPE eq  EPE pol  SEPE all  0-5deg  5-15deg  vp cube  vp paper  dEPE vs base (se)  improved")
    for r in rows:
        m = summary[r]
        print(f"{r:<14}{m['official_all_epe']:>8.3f}{m['official_equator_epe']:>8.3f}{m['official_poles_epe']:>9.3f}"
              f"{m['official_all_sepe'] * 1e3:>10.2f}{m.get('official_disp0-5_epe', float('nan')):>8.3f}"
              f"{m.get('official_disp5-15_epe', float('nan')):>9.3f}"
              f"{m.get('viewport_cube_all_epe_px', float('nan')):>9.3f}{m.get('viewport_paper_all_epe_px', float('nan')):>10.3f}"
              f"{m['paired_epe_delta_mean']:>+10.3f} ({m['paired_epe_delta_se']:.3f}){m['paired_improved_fraction']:>9.2f}")
    result = {"probe_args": vars(my), "eval_args": vars(args), "pairs": seen,
              "elapsed_s": time.time() - start, "rows": summary}
    (out_dir / "tta_probe.json").write_text(json.dumps(result, indent=2))
    print(f"saved={out_dir / 'tta_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
