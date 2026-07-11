"""HEALPix-node flow -> dense ERP pixel flow, and ERP-pixel EPE metrics (plan P2A).

The readout is the exact inverse of ``raft_adapter.erp_flow_to_tangent``: node
tangent flow -> ambient 3D vectors -> HEALPix bilinear interpolation at every pixel
direction -> tangent re-projection -> expmap -> endpoint pixel -> (du, dv) in pixels.

Interpolation happens on the **ambient 3D vectors**, not per-node tangent
components: tangent bases at neighboring nodes differ (fastest near the poles), so
averaging components mixes frames; ambient vectors average cleanly and the radial
residual introduced by averaging is O(node_spacing^2), removed by the projection
step. For a rigid rotation the ambient field is linear in position, so the readout
reproduces rotation fields to interpolation-roundoff — ``run_epe_smoke.py`` asserts
this, seam and poles included.

EPE convention (comparability with published FLOW360/MPF tables first):
  - plain unweighted pixel mean of sqrt(du_err^2 + dv_err^2), per region;
  - the du error is seam-wrapped to (-W/2, W/2]: pred and GT endpoints may land on
    opposite sides of the +-180 deg seam;
  - a cos(lat)-weighted mean is reported alongside (solid-angle-fair), never in
    place of the plain one.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .geometry import (
    equirectangular_pixels_to_unit_vectors,
    expmap,
    points_to_equirectangular_pixels,
    tangent_components_to_3d,
)

try:  # same dependency ladder as geometry.healpix_unit_vectors
    import healpy as _hp
except ImportError:  # pragma: no cover - exercised in the astropy-only container
    _hp = None


def erp_pixel_directions(
    height: int, width: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Unit directions of every ERP pixel center, shape [H*W, 3] (row-major)."""
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    return equirectangular_pixels_to_unit_vectors(
        u.reshape(-1), v.reshape(-1), height, width
    )


def bilinear_node_weights(
    resolution: int,
    height: int,
    width: int,
    nest: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HEALPix bilinear interpolation stencil at every ERP pixel center.

    Returns ``(indices [4, H*W] long, weights [4, H*W] float32)`` into the node
    layout of ``healpix_unit_vectors(resolution, nest=nest)``. Precompute once per
    (resolution, H, W); the runner caches it.
    """
    nside = 1 << resolution
    dirs = erp_pixel_directions(height, width).double().numpy()
    lon = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), 2.0 * np.pi)
    lat = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    if _hp is not None:
        idx, w = _hp.get_interp_weights(nside, np.pi / 2.0 - lat, lon, nest=nest)
    else:
        from astropy_healpix import HEALPix
        from astropy import units as au

        hpx = HEALPix(nside=nside, order="nested" if nest else "ring")
        with np.errstate(divide="ignore", invalid="ignore"):
            idx, w = hpx.bilinear_interpolation_weights(lon * au.rad, lat * au.rad)
    idx = np.asarray(idx, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    # Polar-cap fallback. Two failure modes of the ring-based bilinear stencil near
    # the poles: (i) pixel centers poleward of the outermost ring center (89.8 deg
    # vs 89.27 deg at nside 64) get non-finite weights; (ii) polar ring i holds only
    # 4i nodes, so the stencil spans up to 90 deg of longitude and attenuates the
    # cos/sin(lon) harmonic every ambient vector field carries there by cos(gap/2)
    # (measured: a 2 deg yaw reads 0.707x at ring 1, 0.924x at ring 2 — px errors of
    # 1.7 / 0.4 at 512x1024). Both are fixed by min-norm affine-reproducing weights
    # over the K nearest nodes: sum(w) = 1 and sum(w * p_i) = p_pixel, so any field
    # linear in position (every rigid rotation) is reproduced *exactly*, longitude
    # gaps notwithstanding. Applied poleward of ring CAP_RING, where the bilinear
    # gap is >= 5 deg; run_epe_smoke.py asserts the result at the poles.
    K = 6
    CAP_RING = 10
    npix = idx.shape[1]
    bad = ~np.isfinite(w).all(axis=0) | (w < 0).any(axis=0)
    bad |= (idx < 0).any(axis=0) | (idx >= 12 * nside * nside).any(axis=0)
    i_cap = min(CAP_RING, nside)
    z_cap = 1.0 - (i_cap * i_cap) / (3.0 * nside * nside)
    bad |= np.abs(dirs[:, 2]) > z_cap
    idx_k = np.zeros((K, npix), dtype=np.int64)
    w_k = np.zeros((K, npix), dtype=np.float64)
    good = ~bad
    idx_k[:4, good] = np.where(np.isfinite(w[:, good]), idx[:, good], 0)
    w_k[:4, good] = np.nan_to_num(w[:, good])
    if bad.any():
        from .geometry import healpix_unit_vectors

        nodes = healpix_unit_vectors(resolution, nest=nest).double().numpy()  # [N, 3]
        # Candidate nodes: the cap rings themselves plus a 4-ring margin (~400 nodes
        # per pole) — a full [n_bad, N] kNN would materialize gigabytes for nothing.
        z_margin = 1.0 - ((i_cap + 4) ** 2) / (3.0 * nside * nside)
        for hemi in (1.0, -1.0):
            sel = bad & (dirs[:, 2] * hemi > 0)
            if not sel.any():
                continue
            cand = np.flatnonzero(nodes[:, 2] * hemi > z_margin)
            d = dirs[sel]  # [B, 3]
            nn_local = np.argpartition(-(d @ nodes[cand].T), K, axis=1)[:, :K]
            nn = cand[nn_local]  # [B, K] global node ids
            p = nodes[nn]  # [B, K, 3]
            a = np.concatenate([np.ones((d.shape[0], 1, K)), p.transpose(0, 2, 1)], axis=1)
            b = np.concatenate([np.ones((d.shape[0], 1)), d], axis=1)  # [B, 4]
            gram = a @ a.transpose(0, 2, 1) + 1e-12 * np.eye(4)[None]
            lam = np.linalg.solve(gram, b[..., None])  # [B, 4, 1]
            idx_k[:, sel] = nn.T
            w_k[:, sel] = (a.transpose(0, 2, 1) @ lam)[..., 0].T
    w_k = w_k / w_k.sum(axis=0, keepdims=True)
    indices = torch.as_tensor(np.ascontiguousarray(idx_k), dtype=torch.long, device=device)
    weights = torch.as_tensor(np.ascontiguousarray(w_k), dtype=torch.float32, device=device)
    return indices, weights


def nodes_to_erp_flow(
    node_flow: torch.Tensor,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    height: int,
    width: int,
    weights: Tuple[torch.Tensor, torch.Tensor],
    pixel_dirs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Read node tangent flow [N, 2] out to dense ERP pixel flow [H, W, 2] (px)."""
    if node_flow.ndim == 2:
        node_flow = node_flow.unsqueeze(0)
    idx, w = weights
    ambient = tangent_components_to_3d(node_flow, basis_east, basis_north)[0]  # [N, 3]
    amb_px = (ambient[idx] * w.unsqueeze(-1)).sum(dim=0)  # [H*W, 3]

    if pixel_dirs is None:
        pixel_dirs = erp_pixel_directions(height, width, device=node_flow.device)
    radial = (amb_px * pixel_dirs).sum(dim=-1, keepdim=True)
    tangent = amb_px - radial * pixel_dirs

    endpoints = expmap(pixel_dirs, tangent.unsqueeze(0))[0]
    u2, v2 = points_to_equirectangular_pixels(endpoints, height, width)

    u1 = torch.arange(width, dtype=u2.dtype, device=u2.device).repeat(height)
    v1 = torch.arange(height, dtype=v2.dtype, device=v2.device).repeat_interleave(width)
    du = wrap_px(u2 - u1, width)
    dv = v2 - v1
    return torch.stack([du, dv], dim=-1).reshape(height, width, 2)


def wrap_px(delta: torch.Tensor, width: int) -> torch.Tensor:
    """Wrap a horizontal pixel displacement/error to [-W/2, W/2)."""
    half = float(width) / 2.0
    return torch.remainder(delta + half, float(width)) - half


def build_pixel_region_masks(
    height: int,
    width: int,
    seam_width_deg: float = 15.0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Pixel-space analog of ``metrics.build_region_masks`` (same fp64+eps rule)."""
    eps = 1e-12
    v = torch.arange(height, dtype=torch.float64, device=device)
    u = torch.arange(width, dtype=torch.float64, device=device)
    lat = math.pi / 2.0 - ((v + 0.5) / float(height)) * math.pi  # [H]
    lon = ((u + 0.5) / float(width)) * (2.0 * math.pi) - math.pi  # [W]
    lat_abs = lat.abs().unsqueeze(1).expand(height, width)
    lon_abs = lon.abs().unsqueeze(0).expand(height, width)
    return {
        "global": torch.ones(height, width, dtype=torch.bool, device=device),
        "poles": lat_abs >= np.deg2rad(60.0) - eps,
        "equator": lat_abs <= np.deg2rad(30.0) + eps,
        "seam": (math.pi - lon_abs) <= np.deg2rad(seam_width_deg) + eps,
    }


def pixel_coslat(height: int, width: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """cos(latitude) per pixel, [H, W] — the solid-angle weight of an ERP row."""
    v = torch.arange(height, dtype=torch.float64, device=device)
    lat = math.pi / 2.0 - ((v + 0.5) / float(height)) * math.pi
    return lat.cos().unsqueeze(1).expand(height, width).to(torch.float32)


def compute_epe_maps(
    pred_px: torch.Tensor,
    gt_px: torch.Tensor,
    valid: torch.Tensor,
    width: int,
) -> Dict[str, torch.Tensor]:
    """Per-pixel EPE of the prediction and of the zero-flow baseline (|GT|)."""
    du_err = wrap_px(pred_px[..., 0] - gt_px[..., 0], width)
    dv_err = pred_px[..., 1] - gt_px[..., 1]
    epe = torch.sqrt(du_err.square() + dv_err.square())
    zero_epe = torch.sqrt(
        wrap_px(gt_px[..., 0], width).square() + gt_px[..., 1].square()
    )
    return {"epe": epe, "zero_epe": zero_epe, "valid": valid.bool()}


def accumulate_epe(
    maps: Dict[str, torch.Tensor],
    region_masks: Dict[str, torch.Tensor],
    coslat: torch.Tensor,
    totals: Dict[str, float],
    counts: Dict[str, float],
    sample_chunks: List[torch.Tensor],
    sample_stride: int = 61,
) -> None:
    """Exact fp64 sums/counts per region (plain and cos-lat weighted) + quantile samples."""
    valid = maps["valid"]
    for key in ("epe", "zero_epe"):
        vals = maps[key].to(torch.float64)
        for region, mask in region_masks.items():
            sel = valid & mask
            n = sel.sum().item()
            if n == 0:
                continue
            v = vals[sel]
            w = coslat[sel].to(torch.float64)
            totals[f"{region}_{key}"] = totals.get(f"{region}_{key}", 0.0) + v.sum().item()
            counts[f"{region}_{key}"] = counts.get(f"{region}_{key}", 0.0) + n
            totals[f"{region}_{key}_coslat"] = (
                totals.get(f"{region}_{key}_coslat", 0.0) + (v * w).sum().item()
            )
            counts[f"{region}_{key}_coslat"] = (
                counts.get(f"{region}_{key}_coslat", 0.0) + w.sum().item()
            )
    sample_chunks.append(maps["epe"][valid].flatten()[::sample_stride].float().cpu())


def finalize_epe(
    totals: Dict[str, float],
    counts: Dict[str, float],
    sample_chunks: List[torch.Tensor],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, total in totals.items():
        if counts.get(key, 0.0) > 0:
            metrics[f"{key}_px"] = total / counts[key]
    for region in ("global", "poles", "equator", "seam"):
        pred, zero = metrics.get(f"{region}_epe_px"), metrics.get(f"{region}_zero_epe_px")
        if pred is not None and zero is not None and zero > 0:
            metrics[f"{region}_epe_improvement_pct"] = (zero - pred) / zero * 100.0
    if sample_chunks:
        samples = torch.cat(sample_chunks)
        for q, name in ((0.5, "p50"), (0.9, "p90"), (0.95, "p95")):
            metrics[f"epe_px_{name}"] = torch.quantile(samples, q).item()
        metrics["epe_quantile_samples"] = float(samples.numel())
    return metrics
