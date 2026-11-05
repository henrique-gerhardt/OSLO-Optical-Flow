"""Proposal 2, minimal: does coarse-to-fine estimation on the NESTED quad-tree beat a single grid?

Training-free and network-free, so it isolates the estimation STRUCTURE from learning. Every
arm uses the same matcher:
  descriptor  5x5 tangent stencil of blurred gray values around a direction, zero-mean and
              unit-norm; frame 2's stencil uses frame 1's basis parallel-transported along
              the geodesic to the candidate (no east/north mismatch near the poles)
  search      (2r+1)^2 candidates on a tangent grid of one node spacing around the current
              endpoint, soft-argmin (sub-node) over 1 - NCC; no update where the cost is flat
              (mean - min < --min-contrast, textureless regions)
  regularize  3x3 HEALPix-neighbour median of the ambient field after each level
  upsample    nearest parent (NESTED index // 4), re-projected to the child tangent plane

Arms: single_r4, single_r6 (r=3), single_r6_wide (r=6), cascade_r4_r6 (r4 -> r5 -> r6),
cascade_r3_r6. Scored with the benchmark metrics; zero is the floor reference.

Reading (the MS-RAFT ablation, 1.40 single 1/8 vs 1.58 single 1/4 vs 1.13 cascade): the
cascade should beat both single grids, especially in 0-5 deg and at the poles. If it does not
beat single_r4 with a classical matcher, the structure alone is not the lever.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

import common as C
from run_tta_refine_probe import blur_erp, sample_erp
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, nodes_to_erp_flow
from spherical_flow.geometry import expmap, logmap


def neighbours(res):
    from astropy_healpix import neighbours as nb

    n_side = 1 << res
    n = 12 * n_side * n_side
    idx = torch.from_numpy(np.ascontiguousarray(nb(np.arange(n), n_side, order="nested").T)).long()
    self_idx = torch.arange(n).unsqueeze(1).expand_as(idx)
    return torch.cat([torch.where(idx < 0, self_idx, idx), torch.arange(n).unsqueeze(1)], 1)


def transport(vec, a, b):
    """Parallel-transport tangent vectors vec at a to b along the geodesic (rows)."""
    axis = torch.linalg.cross(a, b)
    s = axis.norm(dim=-1, keepdim=True)
    c = (a * b).sum(-1, keepdim=True)
    k = axis / s.clamp_min(1e-12)
    rot = vec * c + torch.linalg.cross(k, vec) * s + k * (k * vec).sum(-1, keepdim=True) * (1 - c)
    return torch.where(s < 1e-12, vec, rot)


STENCIL = 2  # descriptor half-width in node spacings (5x5)
MIN_CONTRAST = 0.1


def descriptor(img, dirs, e1, n1, step):
    vals = []
    for i in range(-STENCIL, STENCIL + 1):
        for j in range(-STENCIL, STENCIL + 1):
            vals.append(sample_erp(img, expmap(dirs, ((i * e1 + j * n1) * step)[None])[0])[:, 0])
    d = torch.stack(vals, -1)
    d = d - d.mean(-1, keepdim=True)
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-4)


class Level:
    def __init__(self, res):
        self.res = res
        self.pts = healpix_unit_vectors(res)
        self.be, self.bn = tangent_basis(self.pts)
        self.spacing = math.sqrt(4 * math.pi / self.pts.shape[0])
        self.nbr = neighbours(res)


def match_level(L, t, g1, g2, radius, temp=0.05):
    """One search pass at level L from field t (ambient tangent at L.pts)."""
    s = L.spacing
    d1 = descriptor(g1, L.pts, L.be, L.bn, s)
    e0 = expmap(L.pts, t[None])[0]
    E, N = transport(L.be, L.pts, e0), transport(L.bn, L.pts, e0)
    offs, costs = [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            cand = expmap(e0, ((i * E + j * N) * s)[None])[0]
            Ec, Nc = transport(E, e0, cand), transport(N, e0, cand)
            costs.append(1 - (descriptor(g2, cand, Ec, Nc, s) * d1).sum(-1))
            offs.append((i, j))
    cost = torch.stack(costs, -1)
    wts = torch.softmax(-(cost - cost.min(-1, keepdim=True).values) / temp, -1)
    o = torch.tensor(offs, dtype=cost.dtype)
    di, dj = (wts @ o).unbind(-1)
    # flat cost (textureless): keep the current endpoint
    contrast = cost.mean(-1) - cost.min(-1).values
    keep = contrast < MIN_CONTRAST
    di, dj = di.masked_fill(keep, 0.0), dj.masked_fill(keep, 0.0)
    e = expmap(e0, ((di[:, None] * E + dj[:, None] * N) * s)[None])[0]
    return ambient_log(L, e)


def ambient_log(L, e):
    comps = logmap(L.pts, e, L.be, L.bn)[0]
    return comps[:, :1] * L.be + comps[:, 1:] * L.bn


def median_filter(L, t):
    m = t[L.nbr].median(dim=1).values
    return m - (m * L.pts).sum(-1, keepdim=True) * L.pts


def upsample(Lc, Lf, t):
    tf = t[torch.arange(Lf.pts.shape[0]) // 4]
    return tf - (tf * Lf.pts).sum(-1, keepdim=True) * Lf.pts


def run_arm(levels, radii, g, img_by_level):
    t = torch.zeros_like(levels[0].pts)
    for k, (L, r) in enumerate(zip(levels, radii)):
        if k:
            t = upsample(levels[k - 1], L, t)
        g1, g2 = img_by_level[L.res]
        t = median_filter(L, match_level(L, t, g1, g2, r))
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--synthetic-rotation-deg", type=float, default=0.0,
                   help="Self-test: frame 2 = frame 1 rotated, exact GT.")
    p.add_argument("--min-contrast", type=float, default=0.1)
    args = p.parse_args()
    global MIN_CONTRAST
    MIN_CONTRAST = args.min_contrast
    levels = {r: Level(r) for r in (3, 4, 5, 6)}
    arms = {
        "single_r4": ([4], [3]),
        "single_r6": ([6], [3]),
        "single_r6_wide": ([6], [6]),
        "cascade_r4_r6": ([4, 5, 6], [3, 3, 3]),
        "cascade_r3_r6": ([3, 4, 5, 6], [3, 3, 3, 3]),
    }
    scorer = C.Scorer()
    weights = {}
    start = time.time()
    for k, rec in enumerate(C.iter_pairs(args)):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        i1 = torch.from_numpy(rec["frame1"]).float().permute(2, 0, 1).mean(0, keepdim=True) / 255
        i2 = torch.from_numpy(rec["frame2"]).float().permute(2, 0, 1).mean(0, keepdim=True) / 255
        if args.synthetic_rotation_deg:
            from run_epe_smoke import rotation_matrix
            rot = rotation_matrix(torch.tensor([-0.3, 1.0, 0.8]), args.synthetic_rotation_deg)
            dirs = erp_pixel_directions(h, w)
            i2 = sample_erp(i1, dirs @ rot).t().reshape(1, h, w)
            flow = C.endpoints_to_erp_flow((dirs.double() @ rot.double().T), h, w).numpy()
        imgs = {}
        for r, L in levels.items():
            sigma = 0.5 * L.spacing / (2 * math.pi / w)
            imgs[r] = (blur_erp(i1, sigma), blur_erp(i2, sigma))
        scorer.add("zero", torch.zeros(h, w, 2), flow)
        for name, (res_list, radii) in arms.items():
            t = run_arm([levels[r] for r in res_list], radii, None, imgs)
            L = levels[res_list[-1]]
            key = (L.res, h, w)
            if key not in weights:
                weights[key] = bilinear_node_weights(L.res, h, w)
            comps = logmap(L.pts, expmap(L.pts, t[None])[0], L.be, L.bn)[0]
            erp = nodes_to_erp_flow(comps, L.pts, L.be, L.bn, h, w, weights[key], pixel_dirs=erp_pixel_directions(h, w))
            scorer.add(name, erp, flow)
        print(f"pair {k + 1} {rec['name']}: " + "  ".join(
            f"{n} {v[-1]:.2f}" for n, v in scorer.per_pair.items()) + f"  ({time.time() - start:.0f}s)", flush=True)
    summary = scorer.summary(base_row="single_r4")
    print("\nP2 classical cascade probe (no learning). " + C.REFERENCE)
    scorer.print_table(summary)
    C.save(args.output_dir, {"args": vars(args), "rows": summary}, "p2_cascade_probe.json")


if __name__ == "__main__":
    main()
