"""Equiangular (lat-lon) twin of the HEALPix pyramid — the causal control for A1.

Every published OSLO-RAFT row compares a *spherical* model against a *raster* model
(RAFT-large on ERP), which confounds two things: the architecture and the sampling.
"Equal-area sampling buys polar accuracy" is therefore a correlation across two
different networks, not a measured cause. This module removes the confound by running
the identical architecture on a grid that samples the sphere the way ERP does.

The grid: level ``l`` has ``n_lat = 2**(l+1)`` rows and ``n_lon = 3 * 2**(l+1)`` columns,
so the node count is ``12 * 4**l`` — **exactly HEALPix's count at every level**, which is
what makes the control fair. Cell centers sit at

    theta = pi * (row + 0.5) / n_lat,      phi = 2 pi * (col + 0.5) / n_lon

so cell solid angle goes as ``sin(theta)``: dense at the poles, sparse at the equator —
the ERP pathology, reproduced deliberately.

**Node ordering.** The pyramid's nesting helpers (``nested_children_index``,
``nested_descendant_index``) require that the four children of flat index ``i`` be
``4i..4i+3``. Splitting longitude into three square blocks and Morton-interleaving
``(row, col)`` inside each block gives exactly that: refining appends one bit pair to the
Morton code, which multiplies the index by 4 and adds 0..3. So every nesting helper,
``pool_features`` and ``convex_upsample`` are reused unchanged.

Following the house convention, the HEALPix path is not touched: this is a sibling
builder, so the validated geometry stays byte-identical.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .geometry import tangent_basis
from .healpix_pyramid import (
    SphereLevel,
    SpherePyramid,
    _build_level,
    _normalize,
    _sort_neighbors_by_tangent_angle,
    nested_children_index,
    nested_descendant_index,
)

# 8-neighborhood offsets in (row, col); longitude wraps, latitude does not.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def grid_shape(level: int) -> Tuple[int, int]:
    """``(n_lat, n_lon)`` for ``level``; the product is ``12 * 4**level``."""
    if level < 0:
        raise ValueError("level must be non-negative")
    n_lat = 1 << (level + 1)
    return n_lat, 3 * n_lat


def _morton(row: torch.Tensor, col: torch.Tensor, bits: int) -> torch.Tensor:
    """Interleave ``row`` and ``col`` bits, row taking the high bit of each pair.

    The pair order fixes the child order to ``(dr, dc) = (0,0), (0,1), (1,0), (1,1)``,
    i.e. child ``k`` of a cell is ``4i + 2*dr + dc``.
    """
    out = torch.zeros_like(row)
    for b in range(bits):
        out |= ((row >> b) & 1) << (2 * b + 1)
        out |= ((col >> b) & 1) << (2 * b)
    return out


def _ordering(level: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(row, col, grid_to_flat)`` with ``row``/``col`` indexed by flat node id."""
    n_lat, n_lon = grid_shape(level)
    rows = torch.arange(n_lat).unsqueeze(1).expand(n_lat, n_lon).reshape(-1)
    cols = torch.arange(n_lon).unsqueeze(0).expand(n_lat, n_lon).reshape(-1)

    side = n_lat                       # each longitude block is side x side
    block = cols // side               # 0, 1 or 2
    within = cols % side
    flat = block * side * side + _morton(rows, within, level + 1)

    order = torch.argsort(flat)        # grid position holding each flat id
    row_of, col_of = rows[order], cols[order]
    grid_to_flat = torch.empty(n_lat, n_lon, dtype=torch.long)
    grid_to_flat[row_of, col_of] = torch.arange(n_lat * n_lon, dtype=torch.long)
    return row_of, col_of, grid_to_flat


def equiangular_unit_vectors(level: int) -> torch.Tensor:
    """``[12 * 4**level, 3]`` unit vectors at equiangular cell centers, nested order."""
    n_lat, n_lon = grid_shape(level)
    row_of, col_of, _ = _ordering(level)
    theta = math.pi * (row_of.double() + 0.5) / n_lat
    phi = 2.0 * math.pi * (col_of.double() + 0.5) / n_lon
    sin_t = torch.sin(theta)
    return torch.stack(
        [sin_t * torch.cos(phi), sin_t * torch.sin(phi), torch.cos(theta)], dim=-1
    ).float()


def level_from_num_nodes(num_nodes: int) -> int:
    """Invert ``12 * 4**level``; raises if ``num_nodes`` is not a valid grid size."""
    level = int(round(math.log(num_nodes / 12.0, 4.0))) if num_nodes >= 12 else -1
    if level < 0 or 12 * 4 ** level != num_nodes:
        raise ValueError(f"{num_nodes} is not 12 * 4**l for any integer l")
    return level


def equiangular_solid_angles(level: int) -> torch.Tensor:
    """Per-node cell solid angle in nested order, normalized to mean 1.

    Metrics average per node. On an equal-area grid that is already the average per unit
    solid angle, but here node density goes as ``1 / sin(theta)``, so an unweighted mean
    over-counts the poles by ~2.5x inside the ``|lat| >= 60`` mask and under-counts the
    equator. These weights restore the per-area average, which is the only aggregate that
    means the same thing on both grids.

    Row ``k`` spans colatitude ``[k, k+1] * pi / n_lat``, so its cell subtends
    ``(2 pi / n_lon) * (cos theta_k - cos theta_{k+1})`` exactly — no small-angle
    approximation, and the untruncated sum is ``4 pi``.
    """
    n_lat, n_lon = grid_shape(level)
    row_of, _, _ = _ordering(level)
    edges = torch.arange(n_lat + 1, dtype=torch.float64) * (math.pi / n_lat)
    cos_edges = torch.cos(edges)
    row_area = (cos_edges[:-1] - cos_edges[1:]) * (2.0 * math.pi / n_lon)
    weights = row_area[row_of]
    return (weights / weights.mean()).float()


def equiangular_neighbor_graph(
    level: int, points: Optional[torch.Tensor] = None, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """O(N) SDPAConv graph from lat-lon topology — the twin of ``healpix_neighbor_graph``.

    Longitude wraps; latitude does not, so the top and bottom rows have three missing
    slots. Those are padded with the node's own index and marked invalid, exactly as the
    HEALPix builder does for its 7-neighbor pixels, and SDPAConv skips them.
    """
    n_lat, n_lon = grid_shape(level)
    n = n_lat * n_lon
    if points is None:
        points = equiangular_unit_vectors(level)
    points = _normalize(points.detach().float())
    if points.size(0) != n:
        raise ValueError(f"points count {points.size(0)} != {n}")

    row_of, col_of, grid_to_flat = _ordering(level)
    self_idx = torch.arange(n, dtype=torch.long)

    cols, valids = [], []
    for d_row, d_col in _OFFSETS:
        r = row_of + d_row
        inside = (r >= 0) & (r < n_lat)
        c = (col_of + d_col) % n_lon                       # longitude is periodic
        nbr = grid_to_flat[r.clamp(0, n_lat - 1), c]
        cols.append(torch.where(inside, nbr, self_idx))
        valids.append(inside)
    nbr = torch.stack(cols, dim=1)                         # [N, 8]
    valid_raw = torch.stack(valids, dim=1)

    east, north = tangent_basis(points)
    index = _sort_neighbors_by_tangent_angle(points, nbr, east, north)
    # Recover validity after the sort: an invalid slot is exactly the self pad, and a
    # lat-lon cell is never its own topological neighbor, so this is exact.
    valid = index != self_idx.unsqueeze(1)
    if int(valid.sum()) != int(valid_raw.sum()):
        raise RuntimeError("validity count changed across the tangent-angle sort")
    weight = torch.ones(index.shape, dtype=dtype)
    return index, weight, valid


def build_equiangular_pyramid(
    fine_resolution: int = 6,
    estimation_resolution: int = 4,
    corr_pool_levels: int = 3,
    conv_neighbors: int = 8,
    lookup_neighbors: int = 24,
    knn_chunk: int = 2048,
    retina_resolution: Optional[int] = None,
) -> SpherePyramid:
    """Assemble the equiangular :class:`SpherePyramid`, argument-for-argument like the
    HEALPix builder so a run differs only in ``--grid``."""
    if fine_resolution < estimation_resolution:
        raise ValueError("fine_resolution must be >= estimation_resolution")
    if retina_resolution is not None and retina_resolution < fine_resolution:
        raise ValueError("retina_resolution must be >= fine_resolution")

    top = retina_resolution if retina_resolution is not None else fine_resolution
    corr_resolutions = [
        estimation_resolution - k
        for k in range(corr_pool_levels + 1)
        if estimation_resolution - k >= 0
    ]
    needed = sorted(set(range(estimation_resolution, top + 1)) | set(corr_resolutions))

    levels: Dict[int, SphereLevel] = {}
    for r in needed:
        points = equiangular_unit_vectors(r)
        # The topological graph is O(N) and exact here, so unlike the HEALPix path there
        # is no node-count threshold: use it whenever the kernel is the standard 9.
        graph = equiangular_neighbor_graph(r, points) if conv_neighbors == 8 else None
        level_lookup = lookup_neighbors if r <= fine_resolution else 0
        levels[r] = _build_level(points, conv_neighbors, level_lookup, knn_chunk, graph)

    pool_index = {
        r: nested_children_index(levels[r].num_nodes) for r in needed if (r + 1) in levels
    }
    descendant_index = nested_descendant_index(
        levels[estimation_resolution].num_nodes, fine_resolution - estimation_resolution
    )
    est = levels[estimation_resolution]
    center = torch.arange(est.num_nodes, dtype=torch.long).unsqueeze(1)

    return SpherePyramid(
        levels=levels,
        estimation_resolution=estimation_resolution,
        fine_resolution=fine_resolution,
        corr_resolutions=corr_resolutions,
        pool_index=pool_index,
        descendant_index=descendant_index,
        upsample_neighbors=torch.cat([center, est.conv_index], dim=1),
        retina_resolution=retina_resolution,
    )
