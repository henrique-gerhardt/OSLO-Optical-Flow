"""Sanity checks of the OSLO-RAFT-R operators, with known answers. CPU, no data, no weights.

Each check feeds an input whose correct output is known (a constant image, a single
dot, a rigid rotation) through one operator and compares. Prints one line per check
with PASS/FAIL and the measured numbers, then a summary; exit code 1 if any check fails.

    python run_sanity_checks.py                       # production grids r7 / r6 / r4
    python run_sanity_checks.py --retina 6 --sup 5 --est 3   # quick
    python run_sanity_checks.py --sort-cut-deg -157.5        # with the neighbor-order fix

Checks:
  graph       every node has 8 neighbors at about one node spacing, and the graph is
              symmetric (i sees j iff j sees i)
  slots       SDPAConv slot k points the same way at every node (it is the planar
              conv's "kernel tap k"); compared between the north and south hemispheres
  conv_const  a constant image through a random SDPAConv stays constant
  conv_shift  a one-tap kernel moves a dot by one node in the same direction in both
              hemispheres (the planar "shift" kernel)
  conv_blur   a box kernel applied 4 times to a dot gives a centered, round blob
              (the Gaussian of the planar test), at the equator and near the pole
  pool        4-to-1 pooling keeps a constant, and children sit inside their parent
  enc_white   a white image through the random-init feature encoder (retina -> est)
              gives the same feature at every node
  enc_dot     a black dot on a white image changes the encoder output only around the
              dot, centered on it, the same way in both hemispheres
  erp_sample  ERP -> node sampling of a constant and of a linear function is exact
  gt_flow     a rigid-rotation ERP flow, sampled to tangent flow at the nodes the way
              the training targets are, gives the rotation's endpoints
  readout     the exact tangent flow of a rotation, read out to ERP pixels the way the
              EPE benchmark does, gives the rotation's ERP flow
  upsample    convex upsampling (est -> sup) of a rotation field with uniform weights
              reproduces the field; zero stays zero
  lookup      the correlation lookup peaks at the true displacement: with zero flow at
              the stencil point of the true motion, with the true flow at the center
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from spherical_flow.erp_readout import bilinear_node_weights, nodes_to_erp_flow
from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.geometry import (endpoint_from_tangent_flow, equirectangular_pixels_to_unit_vectors,
                                     healpix_unit_vectors, logmap, points_to_equirectangular_pixels,
                                     tangent_basis)
from spherical_flow.healpix_pyramid import (SpherePyramid, build_healpix_pyramid,
                                            convex_upsample, pool_features)
from spherical_flow.oslo_raft_retina import (OSLORAFTRetina, build_feature_pyramid,
                                             interp_pyramid_lookup)
from spherical_flow.models import SDPAConv

RESULTS: List[Tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<26} {detail}", flush=True)


def spacing_rad(n: int) -> float:
    return math.sqrt(4.0 * math.pi / n)


def lat_deg(points: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.asin(points[:, 2].clamp(-1, 1)))


def neighbor_angles(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """[N, K] tangent angle (deg, east = 0, north = 90) of each neighbor slot."""
    east, north = tangent_basis(points)
    nb = points[index]
    t = nb - (nb * points[:, None]).sum(-1, keepdim=True) * points[:, None]
    return torch.rad2deg(torch.atan2((t * north[:, None]).sum(-1), (t * east[:, None]).sum(-1)))


def circ_mean_deg(a: torch.Tensor) -> float:
    r = torch.deg2rad(a)
    return math.degrees(math.atan2(r.sin().mean().item(), r.cos().mean().item()))


def circ_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def tangent_offset_deg(center: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """(distance deg, direction deg east=0) of ``target`` seen from ``center``."""
    e, n = tangent_basis(center[None])
    f = logmap(center[None], target[None], e, n)[0, 0]
    return math.degrees(f.norm().item()), math.degrees(math.atan2(f[1].item(), f[0].item()))


def node_near(points: torch.Tensor, lat: float, lon: float) -> int:
    d = torch.tensor([math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                      math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
                      math.sin(math.radians(lat))])
    return int((points @ d).argmax())


def rotation_z(angle_rad: float) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_axis(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    k = axis / axis.norm()
    kx = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return torch.eye(3) + math.sin(angle_rad) * kx + (1 - math.cos(angle_rad)) * kx @ kx


# --------------------------------------------------------------------------- #
# graph and convolution
# --------------------------------------------------------------------------- #
def check_graph(pyr: SpherePyramid) -> None:
    for r, level in sorted(pyr.levels.items()):
        if r < pyr.estimation_resolution:
            continue
        idx, valid, pts = level.conv_index, level.conv_valid, level.points
        n = pts.size(0)
        nvalid = valid.sum(1)
        dist = torch.rad2deg(torch.acos((pts[idx] * pts[:, None]).sum(-1).clamp(-1, 1)))
        ratio = dist[valid] / math.degrees(spacing_rad(n))
        edges = set()
        rows, cols = torch.nonzero(valid, as_tuple=True)
        src, dst = rows.tolist(), idx[rows, cols].tolist()
        edges = set(zip(src, dst))
        sym = sum((b, a) in edges for a, b in edges) / max(1, len(edges))
        # diagonal neighbors sit ~1.4 spacings away; the r<=6 kNN graph is slightly asymmetric
        ok = sym > 0.98 and ratio.max() < 2.2 and (nvalid < 7).sum() == 0
        report(f"graph r{r}", ok,
               f"N={n} nodes with 7 neighbors={int((nvalid == 7).sum())} "
               f"neighbor dist/spacing {ratio.min():.2f}..{ratio.max():.2f} symmetric={sym:.3f}")


def check_slots(pyr: SpherePyramid) -> None:
    """Slot k must point the same way everywhere, as tap k of a planar kernel does."""
    for r, level in sorted(pyr.levels.items()):
        if r < pyr.estimation_resolution:
            continue
        ang = neighbor_angles(level.points, level.conv_index)
        lat = lat_deg(level.points)
        full = level.conv_valid.all(1)
        north, south = full & (lat > 5) & (lat < 60), full & (lat < -5) & (lat > -60)
        diffs, cells = [], []
        for k in range(ang.size(1)):
            mn, ms = circ_mean_deg(ang[north, k]), circ_mean_deg(ang[south, k])
            diffs.append(circ_diff_deg(mn, ms))
            cells.append(f"{mn:+.0f}/{ms:+.0f}")
        worst = max(diffs)
        report(f"slots r{r}", worst < 10.0,
               f"slot direction north/south (deg): {' '.join(cells)}; worst mismatch {worst:.0f}")


def check_conv_const(pyr: SpherePyramid) -> None:
    torch.manual_seed(0)
    for r in (pyr.estimation_resolution, pyr.retina_resolution):
        level = pyr.levels[r]
        conv = SDPAConv(3, 16, kernel_size=9, node_dim=1)
        x = torch.ones(1, level.num_nodes, 3) * torch.tensor([0.2, 0.5, 0.9])
        with torch.no_grad():
            y = conv(x, level.conv_index, level.conv_weight, level.conv_valid)[0]
        full = level.conv_valid.all(1)
        spread = (y[full] - y[full].mean(0)).abs().max() / y[full].abs().mean()
        report(f"conv_const r{r}", spread.item() < 1e-5,
               f"max deviation / mean over nodes with 8 neighbors = {spread.item():.1e}"
               f" ({int((~full).sum())} nodes with 7 neighbors excluded)")


def one_tap_conv(k: int) -> SDPAConv:
    conv = SDPAConv(1, 1, kernel_size=9, node_dim=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[k, 0, 0] = 1.0
    return conv


def check_conv_shift(pyr: SpherePyramid) -> None:
    """out[j] = x[nbr[j, k]]: a dot at i reappears at the node j that sees i in slot k,
    i.e. one node opposite to slot k's direction. Must be the same in both hemispheres."""
    level = pyr.estimation_level
    pts = level.points
    sites = {"north 30": node_near(pts, 30, 40), "south 30": node_near(pts, -30, 40)}
    cells, dirs = [], {k: [] for k in (1, 3, 5, 7)}
    for k in (1, 3, 5, 7):
        conv = one_tap_conv(k + 1)  # weight[0] is the center, slot k is weight[k + 1]
        for name, i in sites.items():
            x = torch.zeros(1, level.num_nodes, 1)
            x[0, i, 0] = 1.0
            with torch.no_grad():
                y = conv(x, level.conv_index, level.conv_weight, level.conv_valid)[0, :, 0]
            j = int(y.argmax())
            _, direction = tangent_offset_deg(pts[i], pts[j])
            dirs[k].append(direction)
            cells.append(f"slot{k} {name}: {direction:+.0f}")
    worst = max(circ_diff_deg(*v) for v in dirs.values())
    report("conv_shift", worst < 20.0,
           f"direction the dot moves (deg): {'; '.join(cells)}; worst north/south mismatch {worst:.0f}")


def check_conv_blur(pyr: SpherePyramid) -> None:
    level = pyr.estimation_level
    pts = level.points
    box = SDPAConv(1, 1, kernel_size=9, node_dim=1, bias=False)
    with torch.no_grad():
        box.weight.fill_(1.0 / 9.0)
    sp = math.degrees(spacing_rad(level.num_nodes))
    for name, lat in (("equator", 2.0), ("north 45", 45.0), ("south 45", -45.0), ("pole 80", 80.0)):
        i = node_near(pts, lat, 40)
        x = torch.zeros(1, level.num_nodes, 1)
        x[0, i, 0] = 1.0
        with torch.no_grad():
            for _ in range(4):
                x = box(x, level.conv_index, level.conv_weight, level.conv_valid)
        w = x[0, :, 0]
        e, n = tangent_basis(pts[i][None])
        f = logmap(pts[i][None], pts[None], e, n)[0]          # tangent coords of every node
        f = torch.rad2deg(f) / sp                                # in node spacings
        mass = w.sum()
        c = (w[:, None] * f).sum(0) / mass
        d = f - c
        cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / mass
        ev = torch.linalg.eigvalsh(cov)
        aniso = math.sqrt(ev[1] / ev[0])
        # the kNN graph is slightly asymmetric (most near the poles), so mass drifts a bit
        ok = c.norm() < 0.25 and aniso < 1.5 and abs(mass.item() - 1.0) < 0.1
        report(f"conv_blur {name}", ok,
               f"mass {mass.item():.3f} centroid offset {c.norm().item():.2f} spacings "
               f"sigma {math.sqrt(ev.mean()):.2f} spacings anisotropy {aniso:.2f}")


def check_pool(pyr: SpherePyramid) -> None:
    for r, children in sorted(pyr.pool_index.items()):
        fine, coarse = pyr.levels[r + 1], pyr.levels[r]
        x = torch.full((1, fine.num_nodes, 2), 0.7)
        y = pool_features(x, children)
        centroid = F.normalize(fine.points[children].mean(1), dim=-1)
        off = torch.rad2deg(torch.acos((centroid * coarse.points).sum(-1).clamp(-1, 1)))
        ratio = off.max().item() / math.degrees(spacing_rad(coarse.num_nodes))
        ok = (y - 0.7).abs().max() < 1e-6 and ratio < 0.25
        report(f"pool r{r + 1}->r{r}", ok,
               f"constant kept (max err {(y - 0.7).abs().max():.1e}); children centroid "
               f"to parent max {ratio:.3f} spacings")


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
def make_model(pyr: SpherePyramid, checkpoint: str = "") -> OSLORAFTRetina:
    """Random-init base model, or the encoder of a trained checkpoint (base layout)."""
    torch.manual_seed(0)
    trained = {}
    payload = None
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trained = payload.get("args", {})
    model = OSLORAFTRetina(
        pyr, use_checkpoint_encoder=False,
        hidden_channels=trained.get("hidden_channels", 96),
        context_dim=trained.get("context_dim", 64),
        lookup_rings=trained.get("lookup_rings", 2),
        lookup_ring_points=trained.get("lookup_ring_points", 8),
        update_block=trained.get("update_block") or "small",
        encoder_blocks_per_level=trained.get("encoder_blocks_per_level") or 1,
        encoder_est_blocks=trained.get("encoder_est_blocks") or 0,
    )
    if payload is not None:
        model.load_state_dict(payload.get("model", payload), strict=False)
        print(f"encoder checks use the trained weights of {checkpoint}", flush=True)
    model.eval()
    return model


def check_encoder(pyr: SpherePyramid, model: OSLORAFTRetina) -> None:
    ret, est = pyr.retina_level, pyr.estimation_level
    white = torch.ones(1, ret.num_nodes, 3)
    with torch.no_grad():
        f_white = model.fnet(white, pyr)[0]                     # [N_est, C]
    # nodes whose receptive field holds a node with only 7 neighbors are excluded
    bad = (~ret.conv_valid.all(1)).float()[None, :, None]
    for r in range(pyr.retina_resolution, pyr.estimation_resolution, -1):
        bad = pool_features(bad, pyr.pool_index[r - 1])
    clean = bad[0, :, 0] == 0
    # the 7-neighbor nodes' effect spreads ~2 hops per level on the way down (about 4 est
    # spacings from r7) plus the est-level convs
    for _ in range(8):
        clean = clean & clean[est.conv_index].all(1)
    ref = f_white[clean].mean(0)
    scale = f_white[clean].abs().mean()
    dev = (f_white[clean] - ref).abs().max() / scale
    report("enc_white", dev.item() < 1e-4,
           f"white image -> est features: max deviation / mean {dev.item():.1e} over the "
           f"{int(clean.sum())}/{est.num_nodes} nodes away from the 7-neighbor nodes")
    if not clean.all():
        node_dev = (f_white - ref).abs().max(-1).values / scale
        frac = (node_dev > 0.01).float().mean().item()
        report("enc_white_corners", frac < 0.05,
               f"{int((~ret.conv_valid.all(1)).sum())} retina nodes with 7 neighbors leave a "
               f"fixed pattern: {frac:.0%} of est nodes deviate > 1% (max {node_dev.max():.0%}) "
               f"on a white image")
    deep = clean.clone()
    for _ in range(4):
        deep = deep & deep[est.conv_index].all(1)
    if not deep.any():
        deep = clean

    sp = math.degrees(spacing_rad(est.num_nodes))
    out = {}
    est_lat = lat_deg(est.points)
    for name, lat in (("north 30", 30.0), ("south 30", -30.0), ("equator", 2.0), ("pole 75", 75.0)):
        # a site well inside the clean region, so the 7-neighbor artifacts stay out
        score = (est_lat - lat).abs() + 1e3 * (~deep).float()
        i = int((ret.points @ est.points[int(score.argmin())]).argmax())
        img = white.clone()
        ring = (ret.points @ ret.points[i]) > math.cos(spacing_rad(ret.num_nodes) * 1.5)
        img[0, ring] = 0.0                                       # a black dot a few px wide
        with torch.no_grad():
            delta = model.fnet(img, pyr)[0] - f_white
        # GroupNorm statistics are global (as RAFT's instance norm): the dot shifts every
        # far node by the same vector. Remove that common shift, keep the local response.
        diff = (delta - delta.median(dim=0).values).norm(dim=-1)
        e, n = tangent_basis(ret.points[i][None])
        f = torch.rad2deg(logmap(ret.points[i][None], est.points[None], e, n)[0]) / sp
        diff = diff * clean          # the 7-neighbor artifacts react to the global shift too
        w = diff / diff.sum()
        c = (w[:, None] * f).sum(0)
        # support = the radius holding 90% of the local change
        dist = f.norm(dim=-1)
        order = dist.argsort()
        radius = dist[order][(w[order].cumsum(0) >= 0.9).nonzero()[0, 0]].item()
        out[name] = c
        ok = c.norm() < 0.5 and radius < 6.0
        report(f"enc_dot {name}", ok,
               f"response centroid offset {c.norm().item():.2f} est spacings, "
               f"90% of the change within {radius:.1f} spacings")


# --------------------------------------------------------------------------- #
# data path: ERP sampling, GT flow, readout
# --------------------------------------------------------------------------- #
def rotation_erp_flow(rot: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Exact ERP pixel flow [H, W, 2] of content moving p -> R p."""
    v, u = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                          torch.arange(width, dtype=torch.float64), indexing="ij")
    d = equirectangular_pixels_to_unit_vectors(u.reshape(-1), v.reshape(-1), height, width)
    u2, v2 = points_to_equirectangular_pixels(d @ rot.double().T, height, width)
    du = torch.remainder(u2 - u.reshape(-1) + width / 2, width) - width / 2
    return torch.stack([du, v2 - v.reshape(-1)], -1).reshape(height, width, 2).float()


def region_split(lat: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"equator": lat.abs() < 45, "poles": (lat.abs() >= 45) & (lat.abs() < 80),
            "caps>80": lat.abs() >= 80}


def check_data_path(sup_res: int, height: int, width: int) -> None:
    pts = healpix_unit_vectors(sup_res).float()
    e, n = tangent_basis(pts)
    u, v = points_to_equirectangular_pixels(pts, height, width)

    const = torch.full((height, width, 3), 0.37)
    lin_coef = torch.tensor([0.3, -0.5, 0.8])
    vv, uu = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                            torch.arange(width, dtype=torch.float32), indexing="ij")
    dirs = equirectangular_pixels_to_unit_vectors(uu.reshape(-1), vv.reshape(-1), height, width)
    lin = (dirs @ lin_coef).reshape(height, width, 1)
    s_const = bilinear_sample_erp(const, u, v)
    s_lin = bilinear_sample_erp(lin, u, v)[:, 0]
    lin_err = (s_lin - pts @ lin_coef).abs()
    report("erp_sample", (s_const - 0.37).abs().max() < 1e-6 and lin_err.max() < 0.01,
           f"constant max err {(s_const - 0.37).abs().max():.1e}; linear field max err "
           f"{lin_err.max():.1e} (range 1.0)")

    from spherical_flow.shard_dataset import sample_pair_to_nodes
    torch.manual_seed(0)
    lat = lat_deg(pts)
    px_deg = 360.0 / width
    for name, axis, angle in (("yaw 5deg", torch.tensor([0.0, 0.0, 1.0]), 5.0),
                              ("tilt 5deg", torch.tensor([1.0, 0.3, 0.2]), 5.0),
                              ("tilt 1deg", torch.tensor([0.2, 1.0, 0.1]), 1.0)):
        rot = rotation_axis(axis, math.radians(angle))
        flow_erp = rotation_erp_flow(rot, height, width)
        valid = torch.ones(height, width, dtype=torch.bool)
        tgt = sample_pair_to_nodes(const, const, flow_erp, valid, pts, e, n)
        end = tgt["endpoint"].reshape(-1, 3)
        err = torch.rad2deg(torch.acos((end * (pts @ rot.T)).sum(-1).clamp(-1, 1)))
        ok_mask = tgt["valid"].reshape(-1)
        parts = []
        ok = True
        for reg, m in region_split(lat).items():
            m = m & ok_mask
            if m.any():
                med, mx = err[m].median().item(), err[m].max().item()
                parts.append(f"{reg} median {med / px_deg:.3f} px max {mx / px_deg:.2f} px")
                if reg != "caps>80" and med / px_deg > 0.05:
                    ok = False
        report(f"gt_flow {name}", ok,
               f"target endpoint vs exact (in ERP px of {px_deg:.3f} deg): {'; '.join(parts)}; "
               f"valid {ok_mask.float().mean():.3f}")

    weights = bilinear_node_weights(sup_res, height, width)
    plat = lat_deg(dirs)
    for name, axis, angle in (("tilt 5deg", torch.tensor([1.0, 0.3, 0.2]), 5.0),):
        rot = rotation_axis(axis, math.radians(angle))
        exact_node = logmap(pts, (pts @ rot.T)[None], e, n)[0]
        erp = nodes_to_erp_flow(exact_node, pts, e, n, height, width, weights)
        gt = rotation_erp_flow(rot, height, width)
        epe = (erp - gt).norm(dim=-1).reshape(-1)
        parts = [f"{reg} mean {epe[m].mean():.3f} px" for reg, m in region_split(plat).items()]
        report(f"readout {name}", epe[plat.abs() < 80].mean() < 0.05,
               f"exact node flow -> ERP vs exact ERP flow EPE: {'; '.join(parts)}")


# --------------------------------------------------------------------------- #
# upsampler and correlation lookup
# --------------------------------------------------------------------------- #
def check_upsample(pyr: SpherePyramid) -> None:
    est, fine = pyr.estimation_level, pyr.fine_level
    rot = rotation_axis(torch.tensor([1.0, 0.3, 0.2]), math.radians(3.0))
    f_est = logmap(est.points, (est.points @ rot.T)[None], est.basis_east, est.basis_north)
    f_fine = logmap(fine.points, (fine.points @ rot.T)[None], fine.basis_east, fine.basis_north)
    d, k = pyr.descendant_index.size(1), pyr.upsample_neighbors.size(1)
    uni = torch.full((1, est.num_nodes, d, k), 1.0 / k)
    pwc = torch.zeros(1, est.num_nodes, d, k)
    pwc[..., 0] = 1.0
    mag = f_fine.norm(dim=-1).mean()
    rel = {}
    for name, w in (("uniform", uni), ("pwc", pwc)):
        up = convex_upsample(f_est, w, pyr)
        rel[name] = ((up - f_fine).norm(dim=-1).mean() / mag).item()
    zero = convex_upsample(torch.zeros_like(f_est), uni, pyr).abs().max().item()
    report("upsample", rel["uniform"] < 0.05 and zero == 0.0,
           f"rotation field error / magnitude: uniform {rel['uniform']:.3f}, "
           f"piecewise-constant {rel['pwc']:.3f}; zero field -> max {zero:.1e}")


def smooth_features(points: torch.Tensor, freqs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.cos(points @ freqs + phase), dim=-1)


def check_lookup(pyr: SpherePyramid) -> None:
    est = pyr.estimation_level
    pts = est.points
    sp = spacing_rad(est.num_nodes)
    torch.manual_seed(0)
    c = 64
    freqs = torch.randn(3, c) * (1.0 / (1.5 * sp))
    phase = torch.rand(c) * 2 * math.pi
    lat = lat_deg(pts)
    eq = lat.abs() < 20
    from spherical_flow.oslo_raft_retina import build_lookup_offsets
    offsets = build_lookup_offsets(pyr, 2, 8)          # [L, 17, 2]; level 0 = est grid
    m = offsets.size(1)
    for name, rot, expect_dir in (("east", rotation_z(sp), 0.0), ("west", rotation_z(-sp), 180.0)):
        f1 = smooth_features(pts, freqs, phase)[None]
        f2 = smooth_features(pts @ rot, freqs, phase)[None]    # content at p moved to R p
        f2_levels = build_feature_pyramid(f2, pyr)
        true_flow = logmap(pts, (pts @ rot.T)[None], est.basis_east, est.basis_north)
        with torch.no_grad():
            c0 = interp_pyramid_lookup(f1, f2_levels, torch.zeros_like(true_flow), pyr, offsets)
            c1 = interp_pyramid_lookup(f1, f2_levels, true_flow, pyr, offsets)
        arg0 = c0[0, :, :m].argmax(-1)
        arg1 = c1[0, :, :m].argmax(-1)
        stencil_dir = torch.rad2deg(torch.atan2(offsets[0, :, 1], offsets[0, :, 0]))
        stencil_r = offsets[0].norm(dim=-1) / sp
        want = int(((stencil_r - 1.0).abs() < 0.01).nonzero()[
            torch.tensor([circ_diff_deg(a.item(), expect_dir) for a in
                          stencil_dir[(stencil_r - 1.0).abs() < 0.01]]).argmin()])
        hit0 = (arg0[eq] == want).float().mean().item()
        hit1 = (arg1[eq] == 0).float().mean().item()
        report(f"lookup {name}", hit0 > 0.9 and hit1 > 0.9,
               f"equator, motion 1 node {name}: zero-flow argmax at the {name} stencil point "
               f"{hit0:.2f}; true-flow argmax at the center {hit1:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--retina", type=int, default=7)
    p.add_argument("--sup", type=int, default=6)
    p.add_argument("--est", type=int, default=4)
    p.add_argument("--corr-pool-levels", type=int, default=3)
    p.add_argument("--sort-cut-deg", type=float, default=None,
                   help="Neighbor-order start angle (run_oslo_raft.py --neighbor-sort-cut-deg).")
    p.add_argument("--fill-missing-neighbors", action="store_true",
                   help="run_oslo_raft.py --fill-missing-neighbors")
    p.add_argument("--erp", default="512x1024", help="ERP size for the data-path checks.")
    p.add_argument("--skip-encoder", action="store_true")
    p.add_argument("--checkpoint", default="",
                   help="OSLO checkpoint for the encoder checks (default: random init). Its "
                        "grids and neighbor order must match the flags above.")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.time()
    pyr = build_healpix_pyramid(fine_resolution=args.sup, estimation_resolution=args.est,
                                corr_pool_levels=args.corr_pool_levels,
                                retina_resolution=args.retina, sort_cut_deg=args.sort_cut_deg,
                                fill_missing_neighbors=args.fill_missing_neighbors)
    print(f"pyramid retina r{args.retina} sup r{args.sup} est r{args.est} "
          f"sort_cut_deg={args.sort_cut_deg} fill_missing={args.fill_missing_neighbors} built in {time.time() - t0:.0f}s", flush=True)
    height, width = (int(x) for x in args.erp.lower().split("x"))

    check_graph(pyr)
    check_slots(pyr)
    check_conv_const(pyr)
    check_conv_shift(pyr)
    check_conv_blur(pyr)
    check_pool(pyr)
    if not args.skip_encoder:
        check_encoder(pyr, make_model(pyr, args.checkpoint))
    check_data_path(args.sup, height, width)
    check_upsample(pyr)
    check_lookup(pyr)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\nsanity: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
