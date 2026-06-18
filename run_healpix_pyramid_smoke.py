"""CPU smoke for the nested-HEALPix pyramid foundation (spherical_flow.healpix_pyramid).

Runs anywhere — no GPU, no healpy/astropy — by exercising the dependency-free index and
transport math on synthetic / low-res inputs. A final real-geometry section builds an
actual small pyramid *if* astropy-healpix (or healpy) is installed; otherwise it is skipped
with a message (it runs inside the CUDA container via scripts/container_smoke.sh).

    python run_healpix_pyramid_smoke.py
"""

from __future__ import annotations

import math

import torch

from spherical_flow.geometry import (
    _normalize,
    directional_knn_graph,
    fibonacci_unit_vectors,
    logmap,
    rotate_points,
    tangent_basis,
)
from spherical_flow.healpix_pyramid import (
    SpherePyramid,
    build_healpix_pyramid,
    chunked_directional_knn_graph,
    chunked_nearest,
    convex_upsample,
    nested_children_index,
    nested_descendant_index,
    nested_parent_index,
    pool_features,
)
from spherical_flow.oslo_raft import SphereLevel


def _minimal_level(points: torch.Tensor) -> SphereLevel:
    """A SphereLevel carrying only what convex_upsample needs (points + tangent basis)."""
    east, north = tangent_basis(points)
    n = points.size(0)
    dummy = torch.zeros(n, 1, dtype=torch.long)
    return SphereLevel(
        points=points,
        basis_east=east,
        basis_north=north,
        conv_index=dummy,
        conv_weight=torch.ones(n, 1),
        conv_valid=torch.ones(n, 1, dtype=torch.bool),
        lookup_index=dummy,
        ang2pix=lambda e: torch.zeros(e.shape[:-1], dtype=torch.long, device=e.device),
    )


def check_index_arithmetic() -> None:
    n = 50
    children = nested_children_index(n)                       # [n, 4]
    assert children.shape == (n, 4)
    parents = nested_parent_index(4 * n)
    # parent of every child returns its coarse node
    expect = torch.arange(n).repeat_interleave(4)
    assert torch.equal(parents[children.reshape(-1)], expect)
    # descendants are a bijection onto arange(n * 4**L)
    for levels in (1, 2, 3):
        desc = nested_descendant_index(n, levels)
        assert desc.shape == (n, 4 ** levels)
        flat = desc.reshape(-1)
        assert torch.equal(flat.sort().values, torch.arange(n * 4 ** levels))
    # one level down == children
    assert torch.equal(nested_descendant_index(n, 1), children)
    print("PASS index arithmetic (parent/children inverse, descendant bijection)")


def check_pooling() -> None:
    n, b, c = 32, 2, 7
    x_fine = torch.randn(b, 4 * n, c)
    children = nested_children_index(n)
    pooled = pool_features(x_fine, children)
    assert pooled.shape == (b, n, c)
    # children are contiguous 4-blocks, so pooling == reshape-mean
    assert torch.allclose(pooled, x_fine.reshape(b, n, 4, c).mean(dim=2), atol=1e-6)
    # constant field stays constant
    const = torch.full((b, 4 * n, c), 3.14)
    assert torch.allclose(pool_features(const, children), torch.full((b, n, c), 3.14))
    print("PASS 4-to-1 pooling (mean over children, reduces N by 4x)")


def check_chunked_knn() -> None:
    points = fibonacci_unit_vectors(200)
    idx_full, _, _ = directional_knn_graph(points, num_neighbors=8)
    idx_chunk, w, v = chunked_directional_knn_graph(points, num_neighbors=8, chunk_size=37)
    # identical neighbor sets (chunking must not change the graph)
    assert torch.equal(idx_full.sort(dim=1).values, idx_chunk.sort(dim=1).values)
    assert w.shape == idx_chunk.shape and v.all()
    # chunked_nearest matches a full topk
    k = 12
    full_near = (points @ points.t()).topk(k, dim=1, sorted=True).indices
    assert torch.equal(full_near, chunked_nearest(points, k, chunk_size=37))
    print("PASS chunked kNN == unblocked (conv graph + nearest)")


def _synthetic_pyramid(n_est: int = 24, perturb: float = 0.05) -> SpherePyramid:
    torch.manual_seed(0)
    est_points = _normalize(torch.randn(n_est, 3))
    descendant_index = nested_descendant_index(n_est, 2)             # [n_est, 16]
    d = descendant_index.size(1)
    fine_points = _normalize(
        est_points.unsqueeze(1) + perturb * torch.randn(n_est, d, 3)
    ).reshape(n_est * d, 3)
    upsample_neighbors = chunked_nearest(est_points, k=5)            # self at col 0
    return SpherePyramid(
        levels={4: _minimal_level(est_points), 6: _minimal_level(fine_points)},
        estimation_resolution=4,
        fine_resolution=6,
        corr_resolutions=[4, 3, 2, 1],
        pool_index={},
        descendant_index=descendant_index,
        upsample_neighbors=upsample_neighbors,
    )


def check_convex_upsample() -> None:
    pyr = _synthetic_pyramid()
    n_est = pyr.num_estimation_nodes
    d = pyr.descendant_index.size(1)
    k = pyr.upsample_neighbors.size(1)
    b = 2

    # (a) cold start: zero coarse flow -> exactly zero fine flow, for arbitrary weights.
    weights = torch.softmax(torch.randn(b, n_est, d, k), dim=-1)
    flow_zero = torch.zeros(b, n_est, 2)
    out_zero = convex_upsample(flow_zero, weights, pyr)
    assert out_zero.shape == (b, pyr.num_fine_nodes, 2)
    assert out_zero.abs().max().item() < 1e-6, "cold-start flow must upsample to zero"
    print("PASS convex_upsample cold-start zero (preserves RAFT contract)")

    # (b) finite gradients w.r.t. coarse flow and the weight logits.
    flow = torch.randn(b, n_est, 2, requires_grad=True)
    logits = torch.randn(b, n_est, d, k, requires_grad=True)
    out = convex_upsample(flow, torch.softmax(logits, dim=-1), pyr)
    out.pow(2).sum().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("PASS convex_upsample finite gradients (flow + weights)")

    # (c) transport consistency: a rigid small rotation field, upsampled with the weight
    #     fully on the center node, reproduces the rotation field at the fine nodes.
    axis = _normalize(torch.tensor([0.3, -0.7, 0.5]))
    angle = torch.tensor(0.02)
    est = pyr.estimation_level
    fine = pyr.fine_level
    flow_est = logmap(
        est.points, rotate_points(est.points, axis, angle), est.basis_east, est.basis_north
    )  # [1, N_est, 2]
    onehot = torch.zeros(1, n_est, d, k)
    onehot[..., 0] = 1.0  # center neighbor == the node itself
    up = convex_upsample(flow_est, onehot, pyr)  # [1, N_fine, 2]
    expected = logmap(
        fine.points, rotate_points(fine.points, axis, angle), fine.basis_east, fine.basis_north
    )
    err = (up - expected).abs().max().item()
    assert err < 2e-2, f"transport consistency error too large: {err}"
    print(f"PASS convex_upsample transport consistency (rotation field, err={err:.2e} rad)")


def check_real_geometry() -> None:
    try:
        pyr = build_healpix_pyramid(
            fine_resolution=4, estimation_resolution=2, corr_pool_levels=2, knn_chunk=512
        )
    except ImportError:
        print("SKIP real-geometry pyramid (healpy/astropy-healpix not installed)")
        return

    for r, level in pyr.levels.items():
        assert level.num_nodes == 12 * (4 ** r), f"r={r} node count"
    # nested children sit near their parent
    for r, child_idx in pyr.pool_index.items():
        parent_dir = pyr.levels[r].points
        child_dir = pyr.levels[r + 1].points[child_idx]               # [N_r, 4, 3]
        dots = (child_dir * parent_dir.unsqueeze(1)).sum(dim=-1)
        spacing = math.sqrt(4.0 * math.pi / pyr.levels[r].num_nodes)
        assert dots.min().item() > math.cos(2.0 * spacing), f"r={r} children far from parent"
    # fine descendants sit near their estimation node
    est = pyr.estimation_level
    desc_dir = pyr.fine_level.points[pyr.descendant_index]            # [N_est, D, 3]
    dots = (desc_dir * est.points.unsqueeze(1)).sum(dim=-1)
    spacing = math.sqrt(4.0 * math.pi / est.num_nodes)
    assert dots.min().item() > math.cos(2.0 * spacing), "descendants far from est node"
    print(
        f"PASS real-geometry pyramid (levels={sorted(pyr.levels)}, "
        f"N_est={est.num_nodes}, N_fine={pyr.num_fine_nodes})"
    )


def main() -> None:
    print("[healpix pyramid smoke]")
    check_index_arithmetic()
    check_pooling()
    check_chunked_knn()
    check_convex_upsample()
    check_real_geometry()
    print("ALL OK")


if __name__ == "__main__":
    main()
