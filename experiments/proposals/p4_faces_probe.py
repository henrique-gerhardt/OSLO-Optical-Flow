"""Proposal 4, minimal: can a pretrained PLANAR flow network run natively on HEALPix faces?

Training-free. The 12 HEALPix base faces are nside x nside square grids (NESTED = Z-order
inside each face), the representation DLWP-HPX uses to run ordinary Conv2d on the sphere.
This probe renders both frames onto each face with a P-pixel margin, runs a frozen
TorchVision RAFT on the 12 face pairs, maps the face-pixel flow back to sphere endpoints and
reads it out to ERP like any node model. Rows:

  raft_erp        the same frozen RAFT on the ERP frames (native ERP flow)
  raft_erp_nodes  raft_erp pushed through the r(log2 nside) nodes and read out again
                  (the node route's own cost, so faces vs ERP is compared fairly)
  raft_faces      RAFT on the padded HEALPix faces

Face chart: healpix_base::xyf2loc (Gorski et al. 2005) for face coordinates in [0, 1];
the margin continues the chart by a geodesic step along its Jacobian at the face edge
(smooth, no gaps at the polar vertices; approximate DLWP neighbour padding). The
interior chart is validated against astropy-healpix at start.

Reading: raft_faces within ~10 % of raft_erp_nodes (globally and at the poles) means the
face representation does not handicap a planar network, so a face-based OSLO could load
FlyingThings weights (proposal 4, E2). Much worse means the 45-deg-rotated, non-conformal
faces (or the margin) break the planar prior.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

import common as C
from run_tta_refine_probe import sample_erp
from spherical_flow import healpix_unit_vectors, tangent_basis
from spherical_flow.erp_readout import bilinear_node_weights, erp_pixel_directions, nodes_to_erp_flow
from spherical_flow.geometry import expmap, logmap
from spherical_flow.raft_adapter import erp_flow_to_tangent, load_raft_model, predict_raft_flow

JRLL = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.float64)
JPLL = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.float64)


def chart(face: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """HEALPix face coordinates (x, y in [0, 1]) -> unit vectors (xyf2loc)."""
    jr = JRLL[face] - x - y
    nr = torch.where(jr < 1, jr, torch.where(jr > 3, 4 - jr, torch.ones_like(jr)))
    z = torch.where(jr < 1, 1 - nr * nr / 3, torch.where(jr > 3, nr * nr / 3 - 1, (2 - jr) * 2 / 3))
    tmp = JPLL[face] * nr + x - y
    tmp = torch.where(tmp < 0, tmp + 8, torch.where(tmp >= 8, tmp - 8, tmp))
    phi = torch.where(nr < 1e-15, torch.zeros_like(tmp), (math.pi / 4) * tmp / nr.clamp_min(1e-15))
    s = torch.sqrt((1 - z * z).clamp_min(0))
    return torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], -1)


def chart_ext(face, x, y, eps=1e-6, h=1e-4):
    """Chart continued outside [0, 1]^2 by a geodesic step along the edge Jacobian."""
    xc, yc = x.clamp(eps, 1 - eps), y.clamp(eps, 1 - eps)
    d0 = chart(face, xc, yc)
    out = (x != xc) | (y != yc)
    if not out.any():
        return d0
    sx = torch.where(xc > 0.5, -h, h)
    sy = torch.where(yc > 0.5, -h, h)
    jx = (chart(face, xc + sx, yc) - d0) / sx[..., None]
    jy = (chart(face, xc, yc + sy) - d0) / sy[..., None]
    v = jx * (x - xc)[..., None] + jy * (y - yc)[..., None]
    v = v - (v * d0).sum(-1, keepdim=True) * d0
    shape = d0.shape
    ext = expmap(d0.reshape(-1, 3), v.reshape(1, -1, 3))[0].reshape(shape)
    return torch.where(out[..., None], ext, d0)


def spread_bits(v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(v)
    for b in range(16):
        out |= ((v >> b) & 1) << (2 * b)
    return out


def face_grid(nside: int, pad: int):
    size = nside + 2 * pad
    r, c = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x = (c - pad + 0.5).double() / nside
    y = (r - pad + 0.5).double() / nside
    faces = torch.arange(12).view(12, 1, 1).expand(12, size, size)
    dirs = chart_ext(faces, x.expand(12, size, size), y.expand(12, size, size))
    ix, iy = c[pad:pad + nside, pad:pad + nside], r[pad:pad + nside, pad:pad + nside]
    nest = (torch.arange(12).view(12, 1, 1) * nside * nside + spread_bits(ix - pad) + 2 * spread_bits(iy - pad))
    return dirs, nest  # [12, S, S, 3], [12, nside, nside]


def validate_chart(res: int = 4) -> float:
    """Max angle (deg) between the chart at pixel centres and astropy-healpix (fp64)."""
    from astropy_healpix import HEALPix

    nside = 1 << res
    dirs, nest = face_grid(nside, 0)
    lon, lat = HEALPix(nside=nside, order="nested").healpix_to_lonlat(nest.reshape(-1).numpy())
    lon = np.asarray(lon.to_value("rad"), dtype=np.float64)
    lat = np.asarray(lat.to_value("rad"), dtype=np.float64)
    ref = torch.from_numpy(np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1))
    return math.degrees(torch.acos((dirs.reshape(-1, 3) * ref).sum(-1).clamp(-1, 1)).max().item())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(p)
    p.add_argument("--face-resolution", type=int, default=7, help="HEALPix order of the faces (nside = 2^order)")
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--model", default="raft_small", choices=["raft_small", "raft_large"])
    p.add_argument("--synthetic-rotation-deg", type=float, default=0.0,
                   help="Self-test: frame 2 = frame 1 rotated, exact GT.")
    args = p.parse_args()
    err = validate_chart(4)
    print(f"face chart vs astropy-healpix (r4): max {err:.2e} deg", flush=True)
    if err > 1e-4:
        raise SystemExit("face chart does not match HEALPix NESTED pixel centres")
    device = torch.device(args.device)
    raft, transforms, wname, _ = load_raft_model(args.model, "default", device)
    print(f"{args.model} weights={wname}", flush=True)
    R = args.face_resolution
    nside, pad = 1 << R, args.pad
    if (nside + 2 * pad) % 8:
        raise SystemExit("nside + 2 * pad must be divisible by 8")
    dirs, nest = face_grid(nside, pad)
    S = nside + 2 * pad
    pts = healpix_unit_vectors(R)
    be, bn = tangent_basis(pts)
    scorer = C.Scorer()
    cache = {}
    start = time.time()
    for k, rec in enumerate(C.iter_pairs(args)):
        flow = rec["flow"]
        h, w = flow.shape[:2]
        if (h, w) not in cache:
            u, v = C.points_to_equirectangular_pixels(pts, h, w)
            cache[(h, w)] = (bilinear_node_weights(R, h, w), erp_pixel_directions(h, w), u, v)
        wts, edirs, u, v = cache[(h, w)]
        f1 = torch.from_numpy(rec["frame1"]).permute(2, 0, 1).contiguous()
        f2 = torch.from_numpy(rec["frame2"]).permute(2, 0, 1).contiguous()
        if args.synthetic_rotation_deg:
            from run_epe_smoke import rotation_matrix
            rot = rotation_matrix(torch.tensor([-0.3, 1.0, 0.8]), args.synthetic_rotation_deg)
            f2 = (sample_erp(f1.float() / 255, edirs @ rot).t().reshape(3, h, w) * 255).round().clamp(0, 255).to(torch.uint8)
            flow = C.endpoints_to_erp_flow(edirs.double() @ rot.double().T, h, w).numpy()
        with torch.no_grad():
            erp = predict_raft_flow(raft, transforms, f1[None], f2[None], device, "identity")[0].permute(1, 2, 0).float().cpu()
        scorer.add("raft_erp", erp, flow)
        node = erp_flow_to_tangent(erp, pts, be, bn, u, v, h, w)
        scorer.add("raft_erp_nodes", nodes_to_erp_flow(node, pts, be, bn, h, w, wts, pixel_dirs=edirs), flow)

        def render(img):
            vals = sample_erp(img.float() / 255, dirs.reshape(-1, 3).float())  # [N, 3]
            return (vals.t().reshape(3, 12, S, S).permute(1, 0, 2, 3) * 255).round().clamp(0, 255).to(torch.uint8)

        a, b = render(f1), render(f2)
        with torch.no_grad():
            ff = predict_raft_flow(raft, transforms, a, b, device, "identity").float().cpu()  # [12, 2, S, S]
        rr, cc = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        inner = (slice(None), slice(pad, pad + nside), slice(pad, pad + nside))
        cols = (cc.double()[None] + ff[:, 0].double())[inner]
        rows = (rr.double()[None] + ff[:, 1].double())[inner]
        faces = torch.arange(12).view(12, 1, 1).expand_as(cols)
        end = chart_ext(faces, (cols - pad + 0.5) / nside, (rows - pad + 0.5) / nside)
        e_nodes = torch.zeros(pts.shape[0], 3, dtype=torch.float64)
        e_nodes[nest.reshape(-1)] = end.reshape(-1, 3)
        comps = logmap(pts, e_nodes.float(), be, bn)[0]
        scorer.add("raft_faces", nodes_to_erp_flow(comps, pts, be, bn, h, w, wts, pixel_dirs=edirs), flow)
        print(f"pair {k + 1} {rec['name']}: " + "  ".join(f"{n} {v[-1]:.2f}" for n, v in scorer.per_pair.items())
              + f"  ({time.time() - start:.0f}s)", flush=True)
    summary = scorer.summary(base_row="raft_erp_nodes")
    print(f"\nP4 frozen {args.model} on HEALPix faces (r{R}, pad {pad}) vs ERP. " + C.REFERENCE)
    scorer.print_table(summary)
    C.save(args.output_dir, {"args": vars(args), "chart_check_deg": err, "rows": summary}, "p4_faces_probe.json")


if __name__ == "__main__":
    main()
