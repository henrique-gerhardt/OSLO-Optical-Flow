"""Proposal 1, E0: how much of FlowScape's GT flow is a rigid camera motion on the sphere?

Training-free oracle. For a static point seen in direction p with inverse depth rho, a camera
that rotates by R and translates by t (unit) sees it at q = normalize(R (p - rho t)): no pole,
no seam. Per pair this script fits (R, t) to the GT endpoints (robust essential matrix on
bearings, E = R [t]x, plus a rotation-only fallback) and rho per pixel in closed form, then
scores the rigid reconstructions with the benchmark metrics:

  rot_only     q = R p (best pure rotation)
  rigid_pixel  rho per ERP pixel: the ceiling of an ego-motion + depth head
  rigid_r6     rho per HEALPix r6 cell (median): the OSLO output grid
  rigid_r4     rho per r4 cell: the OSLO estimation grid

Reports, besides EPE / SEPE / viewport, the fraction of pixels reconstructed below 0.5 and
1 ERP px per region, displacement bin, semantic class (static vs vehicle-coloured) and the
near-nadir "overshoot" pixels whose GT endpoint leaves the raster (~3 % of pixels; no
sphere-consistent reading of them fits a rigid motion, so the fit excludes them).

Go (pre-registered in the report): rigid_r6 explains >= 85-90 % of pixels below 1 px, with
the residual at the poles and in 0-5 deg well under OSLO's error (reference in common.py).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

import common as C
from spherical_flow.geometry import healpix_unit_vectors  # noqa: F401  (import check)


def ang(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.linalg.cross(a, b).norm(dim=-1), (a * b).sum(-1))


def kabsch(p, q, w):
    h = (w[:, None] * p).T @ q  # sum w p q^T
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.T @ u.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], dtype=p.dtype))
    return vt.T @ D @ u.T  # R with q ~ R p


def fit_rotation(p, q, iters=5):
    w = torch.ones(p.shape[0], dtype=p.dtype)
    for _ in range(iters):
        R = kabsch(p, q, w)
        r = ang(p @ R.T, q)
        delta = max(float(r.median()), 1e-5)
        w = torch.where(r < delta, torch.ones_like(r), delta / r.clamp_min(1e-12))
    return R


def essential_ls(p, q, w=None):
    a = (q[:, :, None] * p[:, None, :]).reshape(-1, 9)
    if w is not None:
        a = a * w[:, None]
    _, _, vt = torch.linalg.svd(a, full_matrices=False)
    E = vt[-1].reshape(3, 3)
    u, s, vt = torch.linalg.svd(E)
    return u @ torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=p.dtype)) @ vt


def epi_dist(E, p, q):
    n = p @ E.T  # E p
    return torch.asin(((q * n).sum(-1).abs() / n.norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0))


def fit_essential(p, q, gen, iters=300, thresh=math.radians(0.05)):
    best, best_n = None, -1
    n = p.shape[0]
    for _ in range(iters):
        idx = torch.randint(0, n, (8,), generator=gen)
        E = essential_ls(p[idx], q[idx])
        k = int((epi_dist(E, p, q) < thresh).sum())
        if k > best_n:
            best, best_n = E, k
    inl = epi_dist(best, p, q) < thresh
    E = essential_ls(p[inl], q[inl]) if inl.sum() >= 8 else best
    for _ in range(3):  # IRLS refinement
        r = epi_dist(E, p, q)
        delta = max(float(r.median()), 1e-6)
        w = torch.where(r < delta, torch.ones_like(r), delta / r.clamp_min(1e-12)).sqrt()
        E = essential_ls(p, q, w)
    return E


def decompose(E):
    """E = R [t]x = [R t]x R -> candidate (R, t) pairs (t unit)."""
    u, _, vt = torch.linalg.svd(E)
    if torch.det(u) < 0:
        u = -u
    if torch.det(vt) < 0:
        vt = -vt
    W = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=E.dtype)
    out = []
    for Rc in (u @ W @ vt, u @ W.T @ vt):
        for tp in (u[:, 2], -u[:, 2]):  # t' = R t
            out.append((Rc, Rc.T @ tp))
    return out


def solve_rho(p, q, R, t, rho_max=1e3):
    """Per-point inverse depth of q ~ R (p - rho t), rho >= 0, in closed form."""
    a = q @ R  # R^T q (rows)
    n = torch.linalg.cross(p, t.expand_as(p))
    nn = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = a - (a * nn).sum(-1, keepdim=True) * nn
    ta = torch.linalg.cross(t.expand_as(a), a)
    pa = torch.linalg.cross(p, a)
    rho = (pa * ta).sum(-1) / (ta * ta).sum(-1).clamp_min(1e-18)
    return rho.clamp(0.0, rho_max)


def rigid_predict(p, R, t, rho):
    e = (p - rho[:, None] * t[None]) @ R.T
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def fit_rigid(p, q, gen):
    E = fit_essential(p, q, gen)
    best = None
    for R, t in decompose(E):
        rho = solve_rho(p, q, R, t)
        err = ang(rigid_predict(p, R, t, rho), q).mean()
        if best is None or err < best[0]:
            best = (err, R, t)
    return best[1], best[2]


def cell_median(values, cell, ncell):
    """Median of per-pixel values within each HEALPix cell -> [ncell] (0 where empty)."""
    v, c = values.numpy(), cell.numpy()
    order = np.lexsort((v, c))
    counts = np.bincount(c, minlength=ncell)
    starts = np.cumsum(counts) - counts
    out = np.zeros(ncell, dtype=v.dtype)
    has = counts > 0
    out[has] = v[order][starts[has] + counts[has] // 2]
    return torch.from_numpy(out)


def pixel_cells(dirs, resolution):
    from astropy_healpix import HEALPix
    from astropy import units as u

    hp = HEALPix(nside=1 << resolution, order="nested")
    lon = torch.atan2(dirs[:, 1], dirs[:, 0]).numpy()
    lat = torch.asin(dirs[:, 2].clamp(-1, 1)).numpy()
    return torch.from_numpy(hp.lonlat_to_healpix(lon * u.rad, lat * u.rad).astype(np.int64)), 12 * (1 << resolution) ** 2


def vehicle_mask(sem):
    """Pixels coloured like CARLA vehicles / pedestrians (RGB or BGR palette, jpg tolerance)."""
    if sem is None:
        return None
    s = torch.from_numpy(sem.astype(np.float32)).reshape(-1, 3)
    m = torch.zeros(s.shape[0], dtype=torch.bool)
    for col in ((0, 0, 142), (142, 0, 0), (220, 20, 60), (60, 20, 220), (0, 0, 70), (70, 0, 0)):
        m |= (s - torch.tensor(col, dtype=torch.float32)).norm(dim=-1) < 40
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--fit-samples", type=int, default=20000)
    args = p.parse_args()
    gen = torch.Generator().manual_seed(0)
    scorer = C.Scorer()
    frac = {}
    fits = []
    start = time.time()
    cells = {}
    for k, rec in enumerate(C.iter_pairs(args)):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        if (h, w) not in cells:
            dirs = C.pixel_dirs64(h, w)
            cells[(h, w)] = {"dirs": dirs, 6: pixel_cells(dirs, 6), 4: pixel_cells(dirs, 4),
                             "coslat": torch.cos(torch.asin(dirs[:, 2].clamp(-1, 1)))}
        cc = cells[(h, w)]
        pdir, q = cc["dirs"], C.gt_endpoints(flow)
        over = C.ee.overshoot_masks(torch.from_numpy(np.ascontiguousarray(flow)).float())["overshoot"].reshape(-1)
        idx = torch.multinomial(cc["coslat"] * (~over), args.fit_samples, replacement=False, generator=gen)
        R_rot = fit_rotation(pdir[idx], q[idx])
        R, t = fit_rigid(pdir[idx], q[idx], gen)
        rho = solve_rho(pdir, q, R, t)
        preds = {
            "rot_only": pdir @ R_rot.T,
            "rigid_pixel": rigid_predict(pdir, R, t, rho),
        }
        for res in (6, 4):
            cell, n = cc[res]
            rc = cell_median(rho, cell, n)
            preds[f"rigid_r{res}"] = rigid_predict(pdir, R, t, rc[cell])
        veh = vehicle_mask(rec.get("semantic"))
        extra = {}
        if veh is not None:
            extra = {"vehicle": veh.reshape(h, w), "static": ~veh.reshape(h, w)}
        gt = torch.from_numpy(np.ascontiguousarray(flow)).float()
        masks = dict(C.ee.official_region_masks(h, w))
        masks.update(C.ee.displacement_bin_masks(gt))
        masks.update(extra)
        masks.update(C.ee.overshoot_masks(gt))
        for name, e in preds.items():
            erp = C.endpoints_to_erp_flow(e, h, w)
            scorer.add(name, erp, flow, extra)
            epe, _ = C.ee.official_maps(erp, gt)
            for region, m in masks.items():
                for thr in (0.5, 1.0):
                    s = frac.setdefault(name, {}).setdefault(f"{region}_lt{thr}", [0.0, 0.0])
                    s[0] += (epe[m] < thr).sum().item()
                    s[1] += m.sum().item()
        fits.append({"name": rec["name"], "rotation_deg": math.degrees(math.acos(max(-1, min(1, (float(torch.trace(R_rot)) - 1) / 2)))),
                     "t": t.tolist(), "rho_median": float(rho.median()),
                     "vehicle_fraction": float(veh.float().mean()) if veh is not None else None})
        print(f"pair {k + 1} {rec['name']}: " + "  ".join(
            f"{n} {scorer.per_pair[n][-1]:.3f}" for n in preds) + f"  ({time.time() - start:.0f}s)", flush=True)

    summary = scorer.summary()
    for name, d in frac.items():
        summary[name]["fraction_below"] = {k: s / n for k, (s, n) in d.items() if n > 0}
    print("\nP1 rigid ceiling (oracle fit to GT). " + C.REFERENCE)
    scorer.print_table(summary)
    print("\nfraction of pixels with EPE < 1 px:")
    regions = ["all", "equator", "poles", "disp0-5", "disp5-15", "inraster", "overshoot", "static", "vehicle"]
    print(f"{'row':<26}" + "".join(f"{r:>10}" for r in regions))
    for name in summary:
        fb = summary[name]["fraction_below"]
        print(f"{name:<26}" + "".join(f"{fb.get(r + '_lt1.0', float('nan')):>10.3f}" for r in regions))
    C.save(args.output_dir, {"args": vars(args), "rows": summary, "fits": fits}, "p1_rigid_ceiling.json")


if __name__ == "__main__":
    main()
