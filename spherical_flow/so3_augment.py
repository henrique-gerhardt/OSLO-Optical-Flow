"""SO(3) rotation augmentation for spherical optical flow (Phase 0, Week 1).

A HEALPix-native model sees only node samples, so we can augment a pair by rotating
the *whole world* by ``R`` and reading flow at the fixed nodes ``p``. Equivalently,
and far cheaper, rotate the sampling directions: sample the real ERP frames at
``q = p @ R`` and rotate the ground-truth endpoint back into the augmented world.
This is exact (no small-angle approximation) and is the data-side half of the
thesis's geometric claim — together with HEALPix it removes the pole/seam
pathologies an ERP grid bakes in.

For a node ``p`` under rotation ``R`` (a proper rotation matrix):

1. source direction ``q = p @ R``  (= ``R.T @ p`` per row);
2. sample frame1/frame2/flow/valid at ``q`` in the real ERP frames;
3. world endpoint ``e(q)`` from the sampled ERP displacement;
4. augmented endpoint ``e' = e(q) @ R.T``  (= ``R @ e(q)``);
5. target = ``logmap(p, e')``, i.e. tangent flow at the unrotated node.

Steps 2-5 are delegated to :func:`spherical_flow.shard_dataset.sample_pair_to_nodes`
via its ``query_points`` / ``endpoint_rotation`` seam, so augmented and unaugmented
samples share one validated sampling path.

The key algebraic property the diagnostic checks is equivariance: because rotation
is an isometry and ``p = R @ q``, ``logmap(R q, R e(q)) = R · logmap(q, e(q))`` — the
augmented target equals the original target transported into the rotated frame.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .geometry import tangent_basis
from .shard_dataset import sample_pair_to_nodes


def rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Proper rotation matrix from an axis and angle (Rodrigues' formula).

    Args:
        axis: ``[3]`` rotation axis (need not be normalized).
        angle: scalar tensor, radians.

    Returns:
        ``[3, 3]`` rotation matrix ``R`` with ``R @ v`` rotating column vector ``v``.
    """
    axis = axis / axis.norm().clamp_min(1e-8)
    ax, ay, az = axis.unbind(-1)
    zero = torch.zeros_like(ax)
    k = torch.stack(
        [
            torch.stack([zero, -az, ay]),
            torch.stack([az, zero, -ax]),
            torch.stack([-ay, ax, zero]),
        ]
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    return eye + sin_a * k + (1.0 - cos_a) * (k @ k)


def yaw_matrix(angle: torch.Tensor | float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rotation about the +z (vertical) axis by ``angle`` radians (a pure yaw)."""
    angle_t = torch.as_tensor(angle, dtype=dtype)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    return rotation_matrix(axis, angle_t)


def sample_rotation(
    generator: torch.Generator,
    *,
    max_angle_deg: float = 180.0,
    uniform_so3: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a random rotation matrix.

    Args:
        generator: torch RNG for reproducibility.
        max_angle_deg: maximum rotation angle in degrees. ``180`` with
            ``uniform_so3=False`` gives the plan's "axis uniform, angle uniform in
            [0, 180]" schedule (a configurable curriculum knob).
        uniform_so3: if True, draw from the Haar measure on SO(3) instead (angle
            density ``∝ 1 - cos θ``), ignoring ``max_angle_deg`` except as a clamp.
    """
    axis = torch.randn(3, generator=generator, device=device, dtype=dtype)
    axis = axis / axis.norm().clamp_min(1e-8)
    max_angle = math.radians(max_angle_deg)
    if uniform_so3:
        # Inverse-CDF of the SO(3) angle density f(θ) ∝ (1 - cos θ) on [0, π].
        u = torch.rand((), generator=generator, device=device, dtype=dtype)
        angle = _so3_angle_icdf(u) * (max_angle / math.pi)
    else:
        angle = torch.rand((), generator=generator, device=device, dtype=dtype) * max_angle
    return rotation_matrix(axis, angle)


def _so3_angle_icdf(u: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Inverse CDF of f(θ) = (1 - cos θ)/π on [0, π], by bisection."""
    # CDF(θ) = (θ - sin θ)/π. Monotone; solve CDF(θ) = u on [0, π].
    lo = torch.zeros_like(u)
    hi = torch.full_like(u, math.pi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = (mid - torch.sin(mid)) / math.pi
        hi = torch.where(cdf > u, mid, hi)
        lo = torch.where(cdf > u, lo, mid)
    return 0.5 * (lo + hi)


def so3_augment_pair(
    frame1_erp: torch.Tensor,
    frame2_erp: torch.Tensor,
    flow_erp: torch.Tensor,
    valid_erp: torch.Tensor,
    points: torch.Tensor,
    rotation: torch.Tensor,
    basis_east: Optional[torch.Tensor] = None,
    basis_north: Optional[torch.Tensor] = None,
    target_points: Optional[torch.Tensor] = None,
    target_basis_east: Optional[torch.Tensor] = None,
    target_basis_north: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Apply SO(3) augmentation to one ERP pair, returning node samples.

    Args:
        frame1_erp, frame2_erp: ``[H, W, 3]`` ERP frames in ``[0, 1]``.
        flow_erp: ``[H, W, 2]`` canonical ERP displacement.
        valid_erp: ``[H, W]`` validity.
        points: ``[N, 3]`` frame node directions (unrotated).
        rotation: ``[3, 3]`` rotation matrix ``R``.
        basis_east, basis_north: tangent basis at ``points`` (computed if omitted).
        target_points: optional supervision-grid directions (OSLO-RAFT-R); the same
            ``R`` rotates their sampling directions, so frames and targets stay
            consistently paired. ``None`` keeps the single-grid behavior.
        target_basis_east, target_basis_north: tangent basis at ``target_points``.

    Returns:
        The same dict shape as :func:`sample_pair_to_nodes`: ``frame1``, ``frame2``
        at the frame grid; ``flow``/``endpoint``/``valid`` at the target grid.
    """
    if basis_east is None or basis_north is None:
        basis_east, basis_north = tangent_basis(points)
    query_points = points @ rotation
    target_query_points = None if target_points is None else target_points @ rotation
    return sample_pair_to_nodes(
        frame1_erp,
        frame2_erp,
        flow_erp,
        valid_erp,
        points,
        basis_east,
        basis_north,
        query_points=query_points,
        endpoint_rotation=rotation,
        target_points=target_points,
        target_basis_east=target_basis_east,
        target_basis_north=target_basis_north,
        target_query_points=target_query_points,
    )
