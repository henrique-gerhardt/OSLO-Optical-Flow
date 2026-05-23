import math
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import healpy as hp
except ImportError:
    hp = None


def healpix_unit_vectors(
    resolution: int,
    nest: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return HEALPix pixel centers as unit 3D vectors with shape [N, 3]."""
    nside = 1 << resolution
    n_pix = 12 * nside * nside
    if hp is not None:
        x, y, z = hp.pix2vec(nside, np.arange(n_pix), nest=nest)
    else:
        try:
            from astropy_healpix import HEALPix
        except ImportError as exc:
            raise ImportError(
                "HEALPix vectors require healpy or astropy-healpix; "
                "use --grid fibonacci for the dependency-light MVP."
            ) from exc
        order = "nested" if nest else "ring"
        astropy_hp = HEALPix(nside=nside, order=order)
        x, y, z = astropy_hp.healpix_to_xyz(np.arange(n_pix))
    vectors = np.stack([x, y, z], axis=-1)
    return torch.as_tensor(vectors, dtype=dtype, device=device)


def _square_ring_indices(k: int, radius: int, start_sw: bool = True) -> list[int]:
    if k <= 1 or k % 2 == 0 or radius > k // 2:
        raise ValueError("invalid square-ring parameters")
    center = (k // 2) * (k + 1)
    if radius == 0:
        return [center]

    indices = []
    for i in range(center - radius * (k + 1), center - radius * (k - 1) + 1):
        indices.append(i)
    for i in range(indices[-1] + k, indices[-1] + k * (2 * radius + 1), k):
        indices.append(i)
    for i in range(indices[-1] - 1, indices[-1] - 1 - 2 * radius, -1):
        indices.append(i)
    for i in range(indices[-1] - k, indices[-1] - k * (2 * radius), -k):
        indices.append(i)
    shift = (int(start_sw) + 1) * radius
    return indices[-shift:] + indices[:-shift]


def _healpix_grid_order(num_hops: int, include_center: bool = False, start_sw: bool = False) -> list[int]:
    num_neighbors = 8 * num_hops * (num_hops + 1) // 2
    k_size = int(math.sqrt(num_neighbors + 1))
    if k_size * k_size - 1 != num_neighbors:
        raise ValueError("num_hops does not form an odd square neighborhood")
    order = []
    for hop in range(0 if include_center else 1, num_hops + 1):
        order.extend(_square_ring_indices(k_size, hop, start_sw=start_sw))
    return order


def healpix_grid_struct(
    grid_file: str,
    resolution: int,
    num_hops: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load an OSLO neighbor-grid npz into SDPAConv index/weight/mask tensors."""
    int32_max = np.iinfo(np.int32).max
    nside = 1 << resolution
    n_pix = 12 * nside * nside
    k_size = num_hops * 2 + 1
    grid_to_healpix = _healpix_grid_order(num_hops, include_center=False, start_sw=False)

    grid = np.load(grid_file)
    pad = grid["part_1"].shape[0] - nside
    if pad % 2 != 0:
        raise ValueError("grid padding is not symmetrical")
    idx_offset = pad // 2 - num_hops

    index = torch.zeros((n_pix, k_size, k_size), dtype=torch.int32)
    pixel_idx = 0
    for part in range(1, 5):
        grid_part = torch.as_tensor(grid[f"part_{part}"], dtype=torch.int32)
        for row in range(nside):
            for col in range(3 * nside):
                index[pixel_idx, :, :] = grid_part[
                    row + idx_offset : row + idx_offset + k_size,
                    col + idx_offset : col + idx_offset + k_size,
                ]
                pixel_idx += 1

    index = index.reshape(index.shape[0], -1)
    center_idx = num_hops * (k_size + 1)
    sort_idx = torch.argsort(index[:, center_idx : center_idx + 1], dim=0)
    index = torch.take_along_dim(index, sort_idx, dim=0)
    index = index[:, grid_to_healpix]

    valid = index != int32_max
    index = torch.where(valid, index, torch.zeros_like(index)).long()
    weight = valid.to(dtype=dtype)
    return index, weight, valid


def fibonacci_unit_vectors(
    num_nodes: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return approximately uniform unit vectors on the sphere without external deps."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")

    i = torch.arange(num_nodes, dtype=dtype, device=device)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / float(num_nodes)
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    theta = golden_angle * i
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def tangent_basis(points: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a stable east/north tangent basis for unit sphere points."""
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")

    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype, device=points.device)

    east = torch.cross(z_axis.expand_as(points), points, dim=-1)
    fallback = torch.cross(x_axis.expand_as(points), points, dim=-1)
    use_fallback = east.norm(dim=-1, keepdim=True) < eps
    east = torch.where(use_fallback, fallback, east)
    east = _normalize(east, eps)

    north = torch.cross(points, east, dim=-1)
    north = _normalize(north, eps)
    return east, north


def tangent_components_to_3d(
    flow: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
) -> torch.Tensor:
    """Convert tangent components [B, N, 2] to 3D tangent vectors [B, N, 3]."""
    if flow.ndim != 3 or flow.size(-1) != 2:
        raise ValueError("flow must have shape [B, N, 2]")
    east = basis_east.unsqueeze(0)
    north = basis_north.unsqueeze(0)
    return flow[..., 0:1] * east + flow[..., 1:2] * north


def expmap(points: torch.Tensor, tangent_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map 3D tangent vectors at `points` to endpoints on the unit sphere."""
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")
    if tangent_vectors.ndim != 3 or tangent_vectors.size(-1) != 3:
        raise ValueError("tangent_vectors must have shape [B, N, 3]")

    base = points.unsqueeze(0)
    theta = tangent_vectors.norm(dim=-1, keepdim=True)
    direction_scale = torch.sin(theta) / theta.clamp_min(eps)
    endpoint = torch.cos(theta) * base + direction_scale * tangent_vectors
    endpoint = torch.where(theta < eps, base + tangent_vectors, endpoint)
    return _normalize(endpoint, eps)


def endpoint_from_tangent_flow(
    points: torch.Tensor,
    flow: torch.Tensor,
    basis_east: Optional[torch.Tensor] = None,
    basis_north: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert predicted tangent flow components to spherical endpoints."""
    if basis_east is None or basis_north is None:
        basis_east, basis_north = tangent_basis(points)
    tangent_3d = tangent_components_to_3d(flow, basis_east, basis_north)
    return expmap(points, tangent_3d)


def logmap(
    points: torch.Tensor,
    endpoints: torch.Tensor,
    basis_east: Optional[torch.Tensor] = None,
    basis_north: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map spherical endpoints back to tangent components at `points`."""
    if endpoints.ndim == 2:
        endpoints = endpoints.unsqueeze(0)
    base = points.unsqueeze(0)
    dot = (base * endpoints).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    tangent_3d = endpoints - dot * base
    tangent_3d = tangent_3d * (theta / torch.sin(theta).clamp_min(eps))
    tangent_3d = torch.where(theta < eps, torch.zeros_like(tangent_3d), tangent_3d)

    if basis_east is None or basis_north is None:
        basis_east, basis_north = tangent_basis(points)
    east = basis_east.unsqueeze(0)
    north = basis_north.unsqueeze(0)
    return torch.cat(
        [
            (tangent_3d * east).sum(dim=-1, keepdim=True),
            (tangent_3d * north).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )


def geodesic_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Great-circle distance in radians between unit vectors."""
    dot = (a * b).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dot)


def points_to_equirectangular_pixels(
    points: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map unit sphere points to ERP pixel-center coordinates."""
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")

    x, y, z = points.unbind(dim=-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z.clamp(-1.0, 1.0))
    u = ((lon + math.pi) / (2.0 * math.pi)) * float(width) - 0.5
    v = ((math.pi / 2.0 - lat) / math.pi) * float(height) - 0.5
    return u, v


def equirectangular_pixels_to_unit_vectors(
    u: torch.Tensor,
    v: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map ERP pixel-center coordinates to unit sphere points."""
    lon = ((torch.remainder(u + 0.5, float(width))) / float(width)) * (2.0 * math.pi) - math.pi
    v = v.clamp(0.0, float(height - 1))
    lat = math.pi / 2.0 - ((v + 0.5) / float(height)) * math.pi
    cos_lat = torch.cos(lat)
    return torch.stack(
        [
            cos_lat * torch.cos(lon),
            cos_lat * torch.sin(lon),
            torch.sin(lat),
        ],
        dim=-1,
    )


def directional_knn_graph(
    points: torch.Tensor,
    num_neighbors: int = 8,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a local directional graph for arbitrary sphere samples.

    The first `num_neighbors` nearest points are sorted by their local tangent
    angle around each node. This is not a HEALPix replacement, but it exercises
    the same SDPAConv interface when healpy is unavailable.
    """
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3]")
    if points.size(0) <= num_neighbors:
        raise ValueError("num_neighbors must be smaller than the number of points")

    points = _normalize(points)
    similarity = points @ points.t()
    similarity.fill_diagonal_(-float("inf"))
    _, index = similarity.topk(k=num_neighbors, dim=1, largest=True, sorted=False)

    east, north = tangent_basis(points)
    neighbor_points = points.index_select(0, index.reshape(-1)).reshape(points.size(0), num_neighbors, 3)
    base = points.unsqueeze(1)
    tangent = neighbor_points - (neighbor_points * base).sum(dim=-1, keepdim=True) * base
    tangent = _normalize(tangent)
    east_coord = (tangent * east.unsqueeze(1)).sum(dim=-1)
    north_coord = (tangent * north.unsqueeze(1)).sum(dim=-1)
    angle = torch.atan2(north_coord, east_coord)
    order = angle.argsort(dim=1)
    index = index.gather(1, order)

    weight = torch.ones(index.shape, dtype=dtype, device=points.device)
    valid = torch.ones(index.shape, dtype=torch.bool, device=points.device)
    return index.long(), weight, valid


def rotate_points(points: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate unit vectors with Rodrigues' formula."""
    if points.size(-1) != 3:
        raise ValueError("points must end with dimension 3")
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    while axis.ndim < points.ndim:
        axis = axis.unsqueeze(-2)
    while angle.ndim < points.ndim:
        angle = angle.unsqueeze(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    return (
        points * cos_a
        + torch.cross(axis.expand_as(points), points, dim=-1) * sin_a
        + axis * (axis * points).sum(dim=-1, keepdim=True) * (1.0 - cos_a)
    )


def random_rotation(
    generator: torch.Generator,
    max_angle_deg: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = torch.randn(3, generator=generator, device=device, dtype=dtype)
    axis = axis / axis.norm().clamp_min(1e-8)
    max_angle = math.radians(max_angle_deg)
    angle = (2.0 * torch.rand((), generator=generator, device=device, dtype=dtype) - 1.0) * max_angle
    return axis, angle
