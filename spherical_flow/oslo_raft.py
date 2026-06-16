"""OSLO-RAFT: a HEALPix-native RAFT recast on the sphere (Phase 1).

This is the standalone iterative model from docs/OSLO_RAFT_PLAN.md Section 4. It keeps
RAFT's paradigm — siamese feature/context encoders, a correlation volume, and an
iterative GRU that refines flow through a learned lookup — but every spatial operator
is OSLO's SDPAConv on a HEALPix grid, and the lookup is the spherical exp-map +
neighbor-grid gather that is the model's central geometric claim (Section 4.3).

Design split (so it is testable on CPU without healpy):
  - The network is pure tensor ops over a precomputed :class:`SphereLevel` bundle
    (node directions, tangent basis, SDPAConv neighbor grids, a lookup neighborhood,
    and an ``ang2pix`` resolver). Nothing here imports healpy.
  - :func:`build_knn_level` assembles a level from node directions using the
    healpy-free ``directional_knn_graph`` + nearest-node resolver — enough for the
    overfit smoke test. The real nested-HEALPix multi-level builder (4-to-1 pooling,
    healpy ``ang2pix``, convex-upsample descendant maps) is a drop-in replacement
    produced in the CUDA container.

What is implemented here is single-resolution: all-pairs correlation and supervision
at the estimation grid (the plan allows "r=4 first if memory/debug demands"). The
second-image correlation pyramid (4.2) and the convex HEALPix upsampler (4.5) are the
explicit next increment; the forward signature already leaves room for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    directional_knn_graph,
    endpoint_from_tangent_flow,
    geodesic_distance,
    tangent_basis,
)
from .models import SDPAConv, SphereConvBlock


# --------------------------------------------------------------------------- #
# Geometry bundle
# --------------------------------------------------------------------------- #
@dataclass
class SphereLevel:
    """Precomputed per-resolution geometry the network operates on.

    Attributes:
        points: ``[N, 3]`` unit node directions.
        basis_east, basis_north: ``[N, 3]`` tangent basis at ``points``.
        conv_index, conv_weight, conv_valid: SDPAConv neighbor grids ``[N, K]``.
        lookup_index: ``[N, M]`` correlation-gather neighborhood (incl. center).
        ang2pix: maps endpoints ``[..., 3]`` to nearest node index ``[...]``.
    """

    points: torch.Tensor
    basis_east: torch.Tensor
    basis_north: torch.Tensor
    conv_index: torch.Tensor
    conv_weight: torch.Tensor
    conv_valid: torch.Tensor
    lookup_index: torch.Tensor
    ang2pix: Callable[[torch.Tensor], torch.Tensor]

    @property
    def num_nodes(self) -> int:
        return self.points.size(0)

    @property
    def conv_kernel_size(self) -> int:
        return self.conv_index.size(1) + 1

    def to(self, device: torch.device) -> "SphereLevel":
        return SphereLevel(
            points=self.points.to(device),
            basis_east=self.basis_east.to(device),
            basis_north=self.basis_north.to(device),
            conv_index=self.conv_index.to(device),
            conv_weight=self.conv_weight.to(device),
            conv_valid=self.conv_valid.to(device),
            lookup_index=self.lookup_index.to(device),
            ang2pix=self.ang2pix,
        )


def build_knn_level(
    points: torch.Tensor,
    conv_neighbors: int = 8,
    lookup_neighbors: int = 24,
) -> SphereLevel:
    """Assemble a :class:`SphereLevel` from node directions, healpy-free.

    Uses ``directional_knn_graph`` for the SDPAConv grid and a brute-force nearest
    resolver for ``ang2pix`` (exact O(N^2); fine for smoke-scale grids, swapped for
    healpy ``ang2pix`` in the container).
    """
    points = points.detach().float()
    east, north = tangent_basis(points)
    conv_index, conv_weight, conv_valid = directional_knn_graph(points, conv_neighbors)

    # Lookup neighborhood: the `lookup_neighbors` nearest nodes plus the center.
    sim = points @ points.t()
    _, nearest = sim.topk(k=lookup_neighbors + 1, dim=1, largest=True, sorted=True)
    lookup_index = nearest.long()  # column 0 is the node itself (self-similarity = 1)

    points_for_lookup = points

    def ang2pix(endpoints: torch.Tensor) -> torch.Tensor:
        flat = endpoints.reshape(-1, 3)
        idx = (flat @ points_for_lookup.t()).argmax(dim=-1)
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
# Encoders
# --------------------------------------------------------------------------- #
class ResidualSphereBlock(nn.Module):
    """Two SDPAConv layers with a (projected) skip connection and GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 8):
        super().__init__()
        self.conv1 = SDPAConv(in_channels, out_channels, kernel_size=kernel_size, node_dim=1)
        self.conv2 = SDPAConv(out_channels, out_channels, kernel_size=kernel_size, node_dim=1)
        self.norm1 = nn.GroupNorm(_groups(groups, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(_groups(groups, out_channels), out_channels)
        self.proj = (
            SDPAConv(in_channels, out_channels, kernel_size=1, node_dim=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, index, weight, valid):
        y = F.relu(_gn(self.norm1, self.conv1(x, index, weight, valid)))
        y = _gn(self.norm2, self.conv2(y, index, weight, valid))
        skip = x if self.proj is None else self.proj(x)
        return F.relu(y + skip)


class SphereEncoder(nn.Module):
    """Siamese encoder: stacked residual SDPAConv blocks at one resolution."""

    def __init__(self, in_channels: int, channels: tuple[int, ...], kernel_size: int):
        super().__init__()
        blocks = []
        c_prev = in_channels
        for c in channels:
            blocks.append(ResidualSphereBlock(c_prev, c, kernel_size))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = c_prev

    def forward(self, x, index, weight, valid):
        for block in self.blocks:
            x = block(x, index, weight, valid)
        return x


# --------------------------------------------------------------------------- #
# Correlation + spherical lookup
# --------------------------------------------------------------------------- #
class AllPairsCorrelation(nn.Module):
    """Cosine all-pairs correlation at the estimation grid: ``[B, N, N]``."""

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        scale = math.sqrt(f1.size(-1))
        return torch.matmul(f1, f2.transpose(1, 2)) / scale


def spherical_lookup(
    corr: torch.Tensor,
    flow: torch.Tensor,
    level: SphereLevel,
) -> torch.Tensor:
    """Exp-map + neighbor-grid correlation gather (the geometric core, §4.3).

    For each node ``i`` with current tangent flow ``f_i``: endpoint ``e_i =
    expmap(p_i, f_i)``, center ``c_i = ang2pix(e_i)``, then gather the correlation of
    ``c_i``'s lookup neighborhood from the all-pairs volume.

    Args:
        corr: ``[B, N, N]`` correlation (axis 2 indexes the second image).
        flow: ``[B, N, 2]`` current tangent flow.
        level: geometry bundle.

    Returns:
        ``[B, N, M]`` per-node correlation feature.
    """
    endpoint = endpoint_from_tangent_flow(level.points, flow, level.basis_east, level.basis_north)
    center = level.ang2pix(endpoint)                  # [B, N]
    nbr = level.lookup_index[center]                  # [B, N, M]
    return torch.gather(corr, 2, nbr)                 # [B, N, M]


# --------------------------------------------------------------------------- #
# Iterative update (SDPAConv ConvGRU)
# --------------------------------------------------------------------------- #
class GraphConvGRU(nn.Module):
    """ConvGRU where every conv is an SDPAConv on the estimation grid."""

    def __init__(self, hidden_channels: int, input_channels: int, kernel_size: int):
        super().__init__()
        c = hidden_channels + input_channels
        self.conv_z = SDPAConv(c, hidden_channels, kernel_size=kernel_size, node_dim=1)
        self.conv_r = SDPAConv(c, hidden_channels, kernel_size=kernel_size, node_dim=1)
        self.conv_q = SDPAConv(c, hidden_channels, kernel_size=kernel_size, node_dim=1)

    def forward(self, h, x, index, weight, valid):
        hx = torch.cat([h, x], dim=-1)
        z = torch.sigmoid(self.conv_z(hx, index, weight, valid))
        r = torch.sigmoid(self.conv_r(hx, index, weight, valid))
        rh_x = torch.cat([r * h, x], dim=-1)
        q = torch.tanh(self.conv_q(rh_x, index, weight, valid))
        return (1.0 - z) * h + z * q


class MotionEncoder(nn.Module):
    """Encode the lookup correlation feature and current flow into motion features."""

    def __init__(self, corr_channels: int, kernel_size: int, motion_channels: int = 64):
        super().__init__()
        self.corr_conv = SDPAConv(corr_channels, 96, kernel_size=kernel_size, node_dim=1)
        self.flow_conv = SDPAConv(2, 32, kernel_size=kernel_size, node_dim=1)
        self.out_conv = SDPAConv(96 + 32, motion_channels - 2, kernel_size=kernel_size, node_dim=1)
        self.out_channels = motion_channels

    def forward(self, corr_feat, flow, index, weight, valid):
        c = F.relu(self.corr_conv(corr_feat, index, weight, valid))
        f = F.relu(self.flow_conv(flow, index, weight, valid))
        m = F.relu(self.out_conv(torch.cat([c, f], dim=-1), index, weight, valid))
        return torch.cat([m, flow], dim=-1)  # keep raw flow alongside motion features


class OSLORAFT(nn.Module):
    """HEALPix-native iterative spherical optical-flow model."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: tuple[int, ...] = (32, 64, 96),
        context_channels: tuple[int, ...] = (32, 64, 96),
        hidden_channels: int = 96,
        context_dim: int = 64,
        kernel_size: int = 9,
        lookup_neighbors: int = 25,
        flow_scale: float = 0.5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.flow_scale = flow_scale
        self.include_xyz = True

        enc_in = in_channels + (3 if self.include_xyz else 0)
        self.fnet = SphereEncoder(enc_in, feature_channels, kernel_size)
        self.cnet = SphereEncoder(enc_in, context_channels, kernel_size)
        self.context_head = SDPAConv(
            self.cnet.out_channels, hidden_channels + context_dim, kernel_size=1, node_dim=1
        )
        self.correlation = AllPairsCorrelation()
        self.motion_encoder = MotionEncoder(lookup_neighbors, kernel_size)
        self.gru = GraphConvGRU(
            hidden_channels, self.motion_encoder.out_channels + context_dim, kernel_size
        )
        self.flow_conv1 = SDPAConv(hidden_channels, hidden_channels, kernel_size=kernel_size, node_dim=1)
        self.flow_conv2 = SDPAConv(hidden_channels, 2, kernel_size=kernel_size, node_dim=1)
        self._zero_init(self.flow_conv2)

    @staticmethod
    def _zero_init(conv: SDPAConv) -> None:
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def _prep_input(self, frame, level):
        if self.include_xyz:
            xyz = level.points.unsqueeze(0).expand(frame.size(0), -1, -1)
            return torch.cat([frame, xyz], dim=-1)
        return frame

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        level: SphereLevel,
        iters: int = 8,
        flow_init: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return the list of tangent-flow predictions, one per refinement iteration."""
        idx, wgt, val = level.conv_index, level.conv_weight, level.conv_valid

        f1 = self.fnet(self._prep_input(frame1, level), idx, wgt, val)
        f2 = self.fnet(self._prep_input(frame2, level), idx, wgt, val)
        corr = self.correlation(f1, f2)

        ctx = self.cnet(self._prep_input(frame1, level), idx, wgt, val)
        ctx = self.context_head(ctx, idx, wgt, val)
        h, context = torch.split(ctx, [self.hidden_channels, self.context_dim], dim=-1)
        h = torch.tanh(h)
        context = F.relu(context)

        b, n = frame1.size(0), frame1.size(1)
        flow = (
            flow_init
            if flow_init is not None
            else torch.zeros(b, n, 2, device=frame1.device, dtype=frame1.dtype)
        )

        predictions: List[torch.Tensor] = []
        for _ in range(iters):
            flow = flow.detach()
            corr_feat = spherical_lookup(corr, flow, level)
            motion = self.motion_encoder(corr_feat, flow, idx, wgt, val)
            gru_in = torch.cat([motion, context], dim=-1)
            h = self.gru(h, gru_in, idx, wgt, val)
            delta = self.flow_conv2(F.relu(self.flow_conv1(h, idx, wgt, val)), idx, wgt, val)
            flow = flow + self.flow_scale * delta
            predictions.append(flow)
        return predictions


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def sequence_geodesic_loss(
    predictions: List[torch.Tensor],
    gt_endpoint: torch.Tensor,
    level: SphereLevel,
    valid: torch.Tensor,
    gamma: float = 0.8,
) -> torch.Tensor:
    """Iteration-weighted geodesic endpoint loss (the spherical RAFT sequence loss).

    ``L = sum_t gamma^(T-t) * mean_valid( geodesic(expmap(p, f_t), gt_endpoint) )``.
    """
    n_pred = len(predictions)
    if valid.dim() == 1:
        valid = valid.unsqueeze(0).expand(gt_endpoint.shape[:-1])
    mask = valid.float()
    denom = mask.sum().clamp_min(1.0)

    total = predictions[0].new_zeros(())
    for t, flow in enumerate(predictions):
        endpoint = endpoint_from_tangent_flow(level.points, flow, level.basis_east, level.basis_north)
        geo = geodesic_distance(endpoint, gt_endpoint)  # [B, N], radians
        weight = gamma ** (n_pred - 1 - t)
        total = total + weight * (geo * mask).sum() / denom
    return total


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _groups(groups: int, channels: int) -> int:
    g = math.gcd(groups, channels)
    return max(1, g)


def _gn(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    # SDPAConv emits [B, N, C]; GroupNorm wants channels at dim 1.
    return norm(x.transpose(1, 2)).transpose(1, 2)
