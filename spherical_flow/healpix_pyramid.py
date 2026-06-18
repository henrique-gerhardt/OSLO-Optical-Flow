"""Nested-HEALPix multi-resolution geometry for OSLO-RAFT (the §4.1/4.2/4.5 foundation).

The single-resolution model (`oslo_raft.OSLORAFT`) estimates and supervises flow on one
:class:`~spherical_flow.oslo_raft.SphereLevel`. The plan's multi-resolution design
estimates flow at a coarse grid (r=4) and upsamples it to a fine grid (r=6) for the
loss/metrics — which needs the nested-HEALPix hierarchy that `build_knn_level` lacks.

This module is **pure geometry** (no learnable parameters, no model forward). It provides:

  - the exact nested index arithmetic (parent ``i>>2``; children ``4i..4i+3``; descendant
    blocks; 4-to-1 feature pooling) — integer ops, dependency-free, CPU-unit-testable;
  - a memory-bounded kNN (`chunked_directional_knn_graph`) so the conv/lookup grids for the
    fine levels (r5/r6) can be built without materializing a full ``[N, N]`` similarity
    (9.6 GB at r=6);
  - :class:`SpherePyramid` + :func:`build_healpix_pyramid`, which assemble a per-resolution
    bundle of `SphereLevel`s plus the pooling / descendant / upsample-neighbor maps;
  - :func:`convex_upsample`, the RAFT-style convex upsampler adapted to the sphere via
    **parallel transport of the tangent flow** (cold-start-zero preserving — see below).

Healpy-free, mirroring the rest of the codebase: only ``healpix_unit_vectors`` (which falls
back to astropy-healpix) touches real geometry, so the full :func:`build_healpix_pyramid`
runs in the CUDA container, while every index/transport helper is testable on CPU.

The model wiring that consumes this (encoder pooling §4.1, correlation pyramid §4.2,
multi-level lookup §4.3, the convex-weight head + iterative forward §4.5) is the next
increment; it will reuse the building blocks already in `oslo_raft.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .geometry import _normalize, healpix_unit_vectors, parallel_transport, tangent_basis
from .oslo_raft import SphereLevel


# --------------------------------------------------------------------------- #
# Nested index arithmetic (exact integer ops — no geometry, no dependencies)
# --------------------------------------------------------------------------- #
def nested_children_index(n_coarse: int) -> torch.Tensor:
    """``[n_coarse, 4]`` child indices: nested pixel ``i`` has children ``4i..4i+3``."""
    base = torch.arange(n_coarse, dtype=torch.long).unsqueeze(1) * 4
    return base + torch.arange(4, dtype=torch.long).unsqueeze(0)


def nested_parent_index(n_fine: int) -> torch.Tensor:
    """``[n_fine]`` parent index: the parent of nested pixel ``i`` is ``i >> 2``."""
    return torch.arange(n_fine, dtype=torch.long) >> 2


def nested_descendant_index(n_coarse: int, levels_down: int) -> torch.Tensor:
    """``[n_coarse, 4**levels_down]`` map from a coarse node to its fine descendants.

    In nested ordering the descendants of pixel ``i`` ``levels_down`` levels down form the
    contiguous block ``[i * 4**L, (i+1) * 4**L)``, so this is a bijection onto
    ``arange(n_coarse * 4**levels_down)``.
    """
    if levels_down < 0:
        raise ValueError("levels_down must be non-negative")
    span = 4 ** levels_down
    base = torch.arange(n_coarse, dtype=torch.long).unsqueeze(1) * span
    return base + torch.arange(span, dtype=torch.long).unsqueeze(0)


def pool_features(x_fine: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
    """4-to-1 nested pooling: average each parent's 4 children. ``[B,N_f,C]->[B,N_c,C]``."""
    return x_fine[:, children_index].mean(dim=2)


# --------------------------------------------------------------------------- #
# Memory-bounded neighbor grids (so r5/r6 levels never materialize [N, N])
# --------------------------------------------------------------------------- #
def chunked_directional_knn_graph(
    points: torch.Tensor,
    num_neighbors: int = 8,
    chunk_size: int = 2048,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blocked twin of :func:`geometry.directional_knn_graph` (identical contract/result).

    Computes the similarity/topk in row-blocks of ``chunk_size`` so the full ``[N, N]``
    matrix is never allocated. The ``num_neighbors`` nearest nodes (excluding self) are
    sorted by their local tangent angle, exactly as the unblocked version — so for a fixed
    neighbor set the returned ``index`` is identical.
    """
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")
    n = points.size(0)
    if n <= num_neighbors:
        raise ValueError("num_neighbors must be smaller than the number of points")

    points = _normalize(points)
    east, north = tangent_basis(points)
    index_rows: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = points[start:stop]                                  # [b, 3]
        sim = block @ points.t()                                    # [b, N]
        rows = torch.arange(stop - start, device=points.device)
        sim[rows, start + rows] = -float("inf")                     # exclude self
        _, nbr = sim.topk(num_neighbors, dim=1, largest=True, sorted=False)

        neighbor_points = points[nbr]                               # [b, k, 3]
        base = block.unsqueeze(1)
        tangent = neighbor_points - (neighbor_points * base).sum(dim=-1, keepdim=True) * base
        tangent = _normalize(tangent)
        east_coord = (tangent * east[start:stop].unsqueeze(1)).sum(dim=-1)
        north_coord = (tangent * north[start:stop].unsqueeze(1)).sum(dim=-1)
        order = torch.atan2(north_coord, east_coord).argsort(dim=1)
        index_rows.append(nbr.gather(1, order))

    index = torch.cat(index_rows, dim=0).long()
    weight = torch.ones(index.shape, dtype=dtype, device=points.device)
    valid = torch.ones(index.shape, dtype=torch.bool, device=points.device)
    return index, weight, valid


def chunked_nearest(points: torch.Tensor, k: int, chunk_size: int = 2048) -> torch.Tensor:
    """``[N, k]`` indices of the ``k`` nearest nodes incl. self (col 0), blocked over rows.

    Matches the ``lookup_index`` that ``build_knn_level`` builds via a full ``[N, N]`` topk,
    but never allocates the full matrix.
    """
    n = points.size(0)
    points = _normalize(points)
    rows: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sim = points[start:stop] @ points.t()                       # [b, N]
        _, nbr = sim.topk(k, dim=1, largest=True, sorted=True)      # self (sim=1) at col 0
        rows.append(nbr)
    return torch.cat(rows, dim=0).long()


def _build_level(
    points: torch.Tensor,
    conv_neighbors: int,
    lookup_neighbors: int,
    knn_chunk: int,
) -> SphereLevel:
    """Memory-safe `SphereLevel` builder (the chunked generalization of build_knn_level)."""
    points = points.detach().float()
    n = points.size(0)
    east, north = tangent_basis(points)
    # A level holds at most n-1 conv neighbors and n lookup entries (incl. self); clamp so
    # very coarse correlation levels (small n) don't request more neighbors than exist.
    conv_k = min(conv_neighbors, n - 1)
    lookup_k = min(lookup_neighbors + 1, n)
    conv_index, conv_weight, conv_valid = chunked_directional_knn_graph(
        points, conv_k, knn_chunk
    )
    lookup_index = chunked_nearest(points, lookup_k, knn_chunk)
    points_for_lookup = points

    def ang2pix(endpoints: torch.Tensor) -> torch.Tensor:
        flat = endpoints.reshape(-1, 3)
        pts = points_for_lookup.to(flat.device)  # device-follow (see SphereLevel.to note)
        idx = (flat @ pts.t()).argmax(dim=-1)
        return idx.reshape(endpoints.shape[:-1])

    return SphereLevel(
        points=points,
        basis_east=east,
        basis_north=north,
        conv_index=conv_index,
        conv_weight=conv_weight,
        conv_valid=conv_valid,
        lookup_index=lookup_index,
        ang2pix=ang2pix,
    )


# --------------------------------------------------------------------------- #
# Pyramid bundle
# --------------------------------------------------------------------------- #
@dataclass
class SpherePyramid:
    """Nested-HEALPix multi-resolution geometry the multi-res model operates on.

    Attributes:
        levels: ``{resolution: SphereLevel}`` for every resolution used — the encoder
            range ``estimation..fine`` and the correlation range ``estimation..est-pool``.
        estimation_resolution: grid where flow is estimated (correlation/GRU live here).
        fine_resolution: grid where flow is supervised (the upsampler target).
        corr_resolutions: descending list for the §4.2 second-image correlation pyramid.
        pool_index: ``{r: [N_r, 4]}`` child indices into level ``r+1`` (encoder fine->coarse
            pooling via :func:`pool_features`).
        descendant_index: ``[N_est, 4**(fine-est)]`` est->fine map (upsampler scatter).
        upsample_neighbors: ``[N_est, K]`` the estimation 1-hop neighborhood incl. center
            (the nodes the convex weights mix over).
    """

    levels: Dict[int, SphereLevel]
    estimation_resolution: int
    fine_resolution: int
    corr_resolutions: List[int]
    pool_index: Dict[int, torch.Tensor]
    descendant_index: torch.Tensor
    upsample_neighbors: torch.Tensor

    @property
    def estimation_level(self) -> SphereLevel:
        return self.levels[self.estimation_resolution]

    @property
    def fine_level(self) -> SphereLevel:
        return self.levels[self.fine_resolution]

    @property
    def num_estimation_nodes(self) -> int:
        return self.estimation_level.num_nodes

    @property
    def num_fine_nodes(self) -> int:
        return self.fine_level.num_nodes

    def to(self, device: torch.device) -> "SpherePyramid":
        return SpherePyramid(
            levels={r: lvl.to(device) for r, lvl in self.levels.items()},
            estimation_resolution=self.estimation_resolution,
            fine_resolution=self.fine_resolution,
            corr_resolutions=list(self.corr_resolutions),
            pool_index={r: idx.to(device) for r, idx in self.pool_index.items()},
            descendant_index=self.descendant_index.to(device),
            upsample_neighbors=self.upsample_neighbors.to(device),
        )


def build_healpix_pyramid(
    fine_resolution: int = 6,
    estimation_resolution: int = 4,
    corr_pool_levels: int = 3,
    conv_neighbors: int = 8,
    lookup_neighbors: int = 24,
    knn_chunk: int = 2048,
) -> SpherePyramid:
    """Assemble the nested-HEALPix :class:`SpherePyramid` (needs healpy/astropy-healpix).

    Defaults follow plan §4.1-4.2: estimate at r=4, supervise at r=6, correlation pyramid
    r4->r1, SDPAConv kernel 9 (``conv_neighbors=8``), lookup M=25 (``lookup_neighbors=24``).
    """
    if fine_resolution < estimation_resolution:
        raise ValueError("fine_resolution must be >= estimation_resolution")

    corr_resolutions = [
        estimation_resolution - k
        for k in range(corr_pool_levels + 1)
        if estimation_resolution - k >= 0
    ]
    encoder_resolutions = list(range(estimation_resolution, fine_resolution + 1))
    needed = sorted(set(encoder_resolutions) | set(corr_resolutions))

    levels: Dict[int, SphereLevel] = {}
    for r in needed:
        points = healpix_unit_vectors(r)  # nested ordering
        levels[r] = _build_level(points, conv_neighbors, lookup_neighbors, knn_chunk)

    # Encoder pooling maps: for each coarse level r with r+1 present, children at r+1.
    pool_index: Dict[int, torch.Tensor] = {
        r: nested_children_index(levels[r].num_nodes)
        for r in needed
        if (r + 1) in levels
    }

    descendant_index = nested_descendant_index(
        levels[estimation_resolution].num_nodes, fine_resolution - estimation_resolution
    )

    est = levels[estimation_resolution]
    center = torch.arange(est.num_nodes, dtype=torch.long).unsqueeze(1)
    upsample_neighbors = torch.cat([center, est.conv_index], dim=1)  # [N_est, 1+conv_neighbors]

    return SpherePyramid(
        levels=levels,
        estimation_resolution=estimation_resolution,
        fine_resolution=fine_resolution,
        corr_resolutions=corr_resolutions,
        pool_index=pool_index,
        descendant_index=descendant_index,
        upsample_neighbors=upsample_neighbors,
    )


# --------------------------------------------------------------------------- #
# Convex upsampling (RAFT §4.5 adapted to the sphere by parallel transport)
# --------------------------------------------------------------------------- #
def convex_upsample(
    flow_est: torch.Tensor,
    weights: torch.Tensor,
    pyramid: SpherePyramid,
) -> torch.Tensor:
    """Upsample coarse tangent flow to the fine grid by convex parallel transport.

    For each fine descendant ``d`` of estimation node ``q`` (with 1-hop neighbors
    ``{q_j} = upsample_neighbors[q]``): transport each neighbor's tangent flow from its
    node to ``d``, take the ``weights``-weighted sum in ``d``'s tangent plane, and express
    it in ``d``'s basis. Because ``parallel_transport(0) = 0``, a zero coarse flow upsamples
    to exactly zero for any weights — preserving the RAFT cold-start contract (this is why
    we transport the tangent *flow* rather than averaging absolute endpoints).

    Args:
        flow_est: ``[B, N_est, 2]`` coarse tangent flow.
        weights: ``[B, N_est, D, K]`` convex weights (sum over ``K`` = 1) for the ``D``
            descendants over the ``K``-node estimation neighborhood.
        pyramid: the geometry bundle.

    Returns:
        ``[B, N_fine, 2]`` upsampled tangent flow.
    """
    est = pyramid.estimation_level
    fine = pyramid.fine_level
    nbr = pyramid.upsample_neighbors        # [N_est, K]
    desc = pyramid.descendant_index         # [N_est, D]

    b = flow_est.size(0)
    n_est, k = nbr.shape
    d = desc.size(1)

    # Neighbor tangent flows -> 3D tangent vectors at the neighbor nodes.
    nbr_flow = flow_est[:, nbr, :]                              # [B, N_est, K, 2]
    nbr_e = est.basis_east[nbr]                                 # [N_est, K, 3]
    nbr_n = est.basis_north[nbr]                                # [N_est, K, 3]
    t_nbr = nbr_flow[..., 0:1] * nbr_e + nbr_flow[..., 1:2] * nbr_n   # [B, N_est, K, 3]

    # Transport every neighbor's tangent to every descendant's fine node.
    a = est.points[nbr].reshape(1, n_est, 1, k, 3)             # neighbor nodes
    fine_dir = fine.points[desc]                                # [N_est, D, 3]
    b_pt = fine_dir.reshape(1, n_est, d, 1, 3)                  # descendant nodes
    t = t_nbr.reshape(b, n_est, 1, k, 3)                        # broadcast over D
    transported = parallel_transport(t, a, b_pt)               # [B, N_est, D, K, 3]

    # Convex combination in each descendant's tangent plane, then express in its basis.
    pooled = (weights.unsqueeze(-1) * transported).sum(dim=3)   # [B, N_est, D, 3]
    fine_dir_b = fine_dir.unsqueeze(0)                          # [1, N_est, D, 3]
    pooled = pooled - (pooled * fine_dir_b).sum(dim=-1, keepdim=True) * fine_dir_b
    fe = fine.basis_east[desc]                                  # [N_est, D, 3]
    fn = fine.basis_north[desc]
    flow_grouped = torch.stack(
        [(pooled * fe).sum(dim=-1), (pooled * fn).sum(dim=-1)], dim=-1
    )                                                          # [B, N_est, D, 2]

    # Scatter to the fine grid (descendant_index is a permutation of arange(N_fine)).
    flow_fine = flow_est.new_zeros(b, fine.num_nodes, 2)
    flow_fine.index_copy_(1, desc.reshape(-1), flow_grouped.reshape(b, n_est * d, 2))
    return flow_fine
