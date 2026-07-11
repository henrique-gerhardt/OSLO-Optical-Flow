"""CPU smoke for the HEALPix->ERP readout (plan P2A gates G2 + G5). No shards, no GPU.

Three rotation tests at 512x1024 / r6 (the real raster and supervision grid), each
with closed-form per-pixel ERP ground truth:

  T1 pure yaw, 2 deg   — analytic GT is du = const (5.69 px), dv = 0 *everywhere*,
                         so it isolates the seam wrap (G2): pixels whose endpoint
                         crosses +-180 deg would read du ~ -1018 px unwrapped.
  T2 tilted axis, 0.5deg — GT computed per pixel as e = R d; checks the ambient-
                         interpolation choice at the poles (G5): tangent bases spin
                         fastest there, so component interpolation would blow up.
  T3 tilted axis, 0.1deg — sub-pixel regime (0.28 px at the equator): the readout
                         must track well below one pixel for the P2A grid-floor
                         numbers to mean anything.

A rotation's ambient flow field is linear in position, so the readout stencil
(HEALPix bilinear + affine-reproducing polar caps) reproduces it to roundoff;
failures here mean convention bugs (node ordering, wrap, basis), not resolution
limits. Measured at the first passing run (2026-07-10, container, r6/512x1024):
T1 mean 0.0010 / max 0.0165 / seam 0.0010 px; T2 poles-to-equator 1.49x at the
1e-4 px level; T3 mean 0.0007 px on 0.59 px motion. Thresholds sit far above.
"""

from __future__ import annotations

import math

import torch

from spherical_flow.erp_readout import (
    bilinear_node_weights,
    build_pixel_region_masks,
    erp_pixel_directions,
    nodes_to_erp_flow,
    wrap_px,
)
from spherical_flow.geometry import (
    healpix_unit_vectors,
    logmap,
    points_to_equirectangular_pixels,
    tangent_basis,
)

H, W, RES = 512, 1024, 6


def rotation_matrix(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    axis = axis / axis.norm()
    a = math.radians(angle_deg)
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return torch.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def analytic_erp_flow(rot: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    endpoints = dirs @ rot.T
    u2, v2 = points_to_equirectangular_pixels(endpoints, H, W)
    u1 = torch.arange(W, dtype=u2.dtype).repeat(H)
    v1 = torch.arange(H, dtype=v2.dtype).repeat_interleave(W)
    return torch.stack([wrap_px(u2 - u1, W), v2 - v1], dim=-1).reshape(H, W, 2)


def readout_erp_flow(rot, points, be, bn, weights, dirs) -> torch.Tensor:
    node_flow = logmap(points, points @ rot.T, be, bn)[0]
    return nodes_to_erp_flow(node_flow, points, be, bn, H, W, weights, pixel_dirs=dirs)


def epe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        wrap_px(pred[..., 0] - gt[..., 0], W).square()
        + (pred[..., 1] - gt[..., 1]).square()
    )


def main() -> None:
    torch.manual_seed(7)
    points = healpix_unit_vectors(RES)
    be, bn = tangent_basis(points)
    weights = bilinear_node_weights(RES, H, W)
    dirs = erp_pixel_directions(H, W)
    masks = build_pixel_region_masks(H, W)
    # Row 0 / row H-1 pixel endpoints can leave the raster vertically under tilted
    # rotations (analytic GT and readout both hit the asin edge); real evals mask
    # these via the shard valid channel, so the smoke drops the outermost rows.
    interior = torch.zeros(H, W, dtype=torch.bool)
    interior[1:-1, :] = True

    failures = []

    # T1 (G2): yaw crosses the seam; du must be constant across it.
    rot = rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), 2.0)
    gt = analytic_erp_flow(rot, dirs)
    du_expect = 2.0 / 360.0 * W
    assert torch.allclose(gt[..., 0], torch.full_like(gt[..., 0], du_expect), atol=1e-3)
    assert gt[..., 1].abs().max() < 1e-3
    err = epe(readout_erp_flow(rot, points, be, bn, weights, dirs), gt)
    seam_mean = err[masks["seam"] & interior].mean().item()
    t1_mean, t1_max = err[interior].mean().item(), err[interior].max().item()
    print(f"T1 yaw 2deg: mean {t1_mean:.4f} px, max {t1_max:.4f} px, seam mean {seam_mean:.4f} px")
    if not (t1_mean < 0.05 and t1_max < 0.60 and seam_mean < 0.05):
        failures.append("T1 (seam wrap)")

    # T2 (G5): tilted axis; poles must not blow up vs equator.
    rot = rotation_matrix(torch.tensor([1.0, 2.0, 0.5]), 0.5)
    err = epe(readout_erp_flow(rot, points, be, bn, weights, dirs), analytic_erp_flow(rot, dirs))
    eq = err[masks["equator"] & interior].mean().item()
    po = err[masks["poles"] & interior].mean().item()
    t2_mean = err[interior].mean().item()
    print(f"T2 tilt 0.5deg: mean {t2_mean:.4f} px, equator {eq:.4f} px, poles {po:.4f} px "
          f"(ratio {po / max(eq, 1e-9):.2f}x)")
    if not (t2_mean < 0.05 and po < 5.0 * max(eq, 1e-3)):
        failures.append("T2 (pole basis)")

    # T3: sub-pixel motion (0.28 px at the equator) must remain resolvable.
    rot = rotation_matrix(torch.tensor([-0.3, 1.0, 0.8]), 0.1)
    gt = analytic_erp_flow(rot, dirs)
    err = epe(readout_erp_flow(rot, points, be, bn, weights, dirs), gt)
    motion = gt[interior].norm(dim=-1).mean().item()
    t3_mean = err[interior].mean().item()
    print(f"T3 tilt 0.1deg: mean EPE {t3_mean:.4f} px vs mean motion {motion:.4f} px "
          f"(ratio {t3_mean / max(motion, 1e-9):.3f})")
    if not (t3_mean < 0.3 * motion):
        failures.append("T3 (sub-pixel)")

    if failures:
        raise SystemExit(f"EPE readout smoke FAILED: {failures}")
    print("EPE readout smoke PASSED (T1 seam wrap, T2 pole uniformity, T3 sub-pixel).")


if __name__ == "__main__":
    main()
