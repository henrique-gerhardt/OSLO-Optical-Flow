"""CPU smoke for the multi-resolution OSLO-RAFT model (OSLORAFTPyramid).

Validates the full §4.1-4.5 wiring on a **synthetic** nested pyramid — no healpy/astropy,
no GPU required — so wiring bugs are caught locally before any container round-trip. The
synthetic pyramid mirrors the production structure (3-level encoder r6->r5->r4, 16 fine
descendants per estimation node, multi-level correlation/lookup) at tiny node counts, by
generating each finer level as a 4x perturbation of its parent (nested contiguity by
construction). Runs on CUDA if available, so it doubles as the in-container model check.

    python run_oslo_raft_pyramid_smoke.py
"""

from __future__ import annotations

import torch

from spherical_flow.geometry import _normalize, endpoint_from_tangent_flow
from spherical_flow.healpix_pyramid import (
    SpherePyramid,
    _build_level,
    nested_children_index,
    nested_descendant_index,
)
from spherical_flow.oslo_raft import sequence_geodesic_loss
from spherical_flow.oslo_raft_pyramid import OSLORAFTPyramid


def synthetic_pyramid(
    estimation_resolution: int = 4,
    fine_resolution: int = 6,
    corr_pool_levels: int = 2,
    coarsest_nodes: int = 12,
    conv_neighbors: int = 8,
    lookup_neighbors: int = 24,
    perturb: float = 0.06,
) -> SpherePyramid:
    """A small structurally-valid nested pyramid (no healpy) for wiring tests.

    With the defaults: corr levels r4/r3/r2, encoder r6->r5->r4, fine grid r6 = 3072 nodes,
    estimation grid r4 = 192 — exercising the production-shaped wiring at smoke scale.
    """
    corr_resolutions = [
        estimation_resolution - k
        for k in range(corr_pool_levels + 1)
        if estimation_resolution - k >= 0
    ]
    needed = sorted(set(range(estimation_resolution, fine_resolution + 1)) | set(corr_resolutions))

    torch.manual_seed(0)
    points = {needed[0]: _normalize(torch.randn(coarsest_nodes, 3))}
    for r in range(needed[0] + 1, needed[-1] + 1):
        parent = points[r - 1]
        children = parent.repeat_interleave(4, dim=0) + perturb * torch.randn(parent.size(0) * 4, 3)
        points[r] = _normalize(children)  # child 4i..4i+3 sits near parent i

    levels = {
        r: _build_level(points[r], conv_neighbors, lookup_neighbors, knn_chunk=1024)
        for r in needed
    }
    pool_index = {
        r: nested_children_index(levels[r].num_nodes) for r in needed if (r + 1) in levels
    }
    descendant_index = nested_descendant_index(
        levels[estimation_resolution].num_nodes, fine_resolution - estimation_resolution
    )
    est = levels[estimation_resolution]
    upsample_neighbors = torch.cat(
        [torch.arange(est.num_nodes, dtype=torch.long).unsqueeze(1), est.conv_index], dim=1
    )
    return SpherePyramid(
        levels=levels,
        estimation_resolution=estimation_resolution,
        fine_resolution=fine_resolution,
        corr_resolutions=corr_resolutions,
        pool_index=pool_index,
        descendant_index=descendant_index,
        upsample_neighbors=upsample_neighbors,
    )


def main() -> None:
    print("[oslo-raft pyramid model smoke]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyr = synthetic_pyramid().to(device)
    model = OSLORAFTPyramid(pyr).to(device)

    b, iters = 2, 4
    n_fine = pyr.num_fine_nodes
    f1 = torch.rand(b, n_fine, 3, device=device)
    f2 = torch.rand(b, n_fine, 3, device=device)

    preds = model(f1, f2, pyr, iters=iters)
    assert len(preds) == iters, f"expected {iters} predictions, got {len(preds)}"
    assert preds[-1].shape == (b, n_fine, 2), f"bad pred shape {preds[-1].shape}"
    print(f"PASS forward: {len(preds)} preds at fine res, shape={tuple(preds[-1].shape)} "
          f"(N_est={pyr.num_estimation_nodes}, N_fine={n_fine}, device={device.type})")

    # cold-start: zero-init delta head => flow stays 0 => convex_upsample(0)=0 at r=fine
    assert torch.count_nonzero(preds[0]) == 0, "cold-start prediction must be exactly zero"
    print("PASS cold-start zero at the fine grid (RAFT contract end-to-end)")

    # backward + finite grads
    gt = endpoint_from_tangent_flow(
        pyr.fine_level.points, torch.zeros(b, n_fine, 2, device=device),
        pyr.fine_level.basis_east, pyr.fine_level.basis_north,
    )
    valid = torch.ones(b, n_fine, dtype=torch.bool, device=device)
    loss = sequence_geodesic_loss(preds, gt, pyr.fine_level, valid)
    loss.backward()
    grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert grads_finite, "non-finite gradient"
    # the delta + upsample heads must actually receive gradient (not dead)
    assert model.flow_conv2.weight.grad is not None and model.flow_conv2.weight.grad.abs().sum() > 0
    assert model.upsample_head.conv2.weight.grad is not None
    print(f"PASS backward: loss={loss.item():.4f}, all grads finite, heads receive gradient")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000 <= n_params <= 3_000_000, f"param count {n_params:,} outside 1-3M budget"
    print(f"PASS param budget: {n_params:,} (1-3M)")

    print("ALL OK")


if __name__ == "__main__":
    main()
