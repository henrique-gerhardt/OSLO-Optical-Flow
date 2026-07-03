"""OSLO-RAFT multi-resolution model: estimate at r=4, supervise at r=6 (plan §4.1-4.5).

The single-resolution :class:`~spherical_flow.oslo_raft.OSLORAFT` is resolution-capped — at
r=4 the dataset's motion is sub-node, so it cannot represent the fine flow that dominates
the GT (see the r=4 shakeout). This model fixes that with RAFT's estimate-coarse /
supervise-fine recipe on nested HEALPix:

  - §4.1 encoder downsamples the fine input (r6) to the estimation grid (r4) by nested
    4-to-1 pooling between SDPAConv residual blocks;
  - §4.2 builds the all-pairs correlation at r4, then a correlation pyramid by pooling the
    *second-image* axis to r3/r2/r1;
  - §4.3 the spherical lookup gathers each corr level's neighborhood around the flow
    endpoint and concatenates across levels;
  - §4.4 the SDPAConv ConvGRU refines flow at r4 with a zero-init delta head;
  - §4.5 a convex-weight head feeds :func:`~spherical_flow.healpix_pyramid.convex_upsample`
    to lift the r4 flow to r6, where the loss/metrics live.

Everything spatial is OSLO's SDPAConv; all building-block modules are reused from
``oslo_raft.py`` and all geometry from ``healpix_pyramid.py`` (one-directional imports, no
cycle). The single-resolution model is left untouched.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import endpoint_from_tangent_flow
from .healpix_pyramid import SpherePyramid, convex_upsample, pool_features
from .models import SDPAConv
from .oslo_raft import (
    AllPairsCorrelation,
    GraphConvGRU,
    MotionEncoder,
    ResidualSphereBlock,
)


# --------------------------------------------------------------------------- #
# §4.1 encoder downsampling pyramid
# --------------------------------------------------------------------------- #
class PyramidEncoder(nn.Module):
    """Siamese encoder over the HEALPix hierarchy: one residual block per resolution,
    nested 4-to-1 pooling between levels (finest -> estimation grid).

    ``use_checkpoint`` gradient-checkpoints each per-resolution block (retina-depth
    chains store a [B, N, 9, C] SDPAConv gather per conv — ~0.9 GB fp16 at r8/C=16 —
    while the per-stage boundary features kept by checkpointing are only [B, N, C]).
    Off by default so existing models are byte-identical; the retina model turns it on.
    """

    use_checkpoint: bool = False

    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        kernel_size: int,
        resolutions: List[int],
    ):
        super().__init__()
        if len(channels) != len(resolutions):
            raise ValueError(
                f"need one channel width per resolution: {len(channels)} channels vs "
                f"{len(resolutions)} resolutions {resolutions}"
            )
        blocks = []
        c_prev = in_channels
        for c in channels:
            blocks.append(ResidualSphereBlock(c_prev, c, kernel_size))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.resolutions = list(resolutions)  # descending, e.g. [6, 5, 4]
        self.out_channels = c_prev

    def forward(self, x: torch.Tensor, pyramid: SpherePyramid) -> torch.Tensor:
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for i, res in enumerate(self.resolutions):
            level = pyramid.levels[res]
            if ckpt:
                x = checkpoint(
                    self.blocks[i], x, level.conv_index, level.conv_weight,
                    level.conv_valid, use_reentrant=False,
                )
            else:
                x = self.blocks[i](x, level.conv_index, level.conv_weight, level.conv_valid)
            if i < len(self.resolutions) - 1:
                # pool res -> res-1 (pool_index[res-1] holds the children at res)
                x = pool_features(x, pyramid.pool_index[res - 1])
        return x


# --------------------------------------------------------------------------- #
# §4.2 second-image correlation pyramid
# --------------------------------------------------------------------------- #
def build_correlation_pyramid(
    corr0: torch.Tensor, pyramid: SpherePyramid
) -> List[torch.Tensor]:
    """Pool the second-image axis of the r4 all-pairs correlation to each corr level.

    ``corr0`` is ``[B, N_est, N_est]``. Nested contiguity makes second-image columns
    ``4j..4j+3`` the children of coarse node ``j``, so each level is a 4-to-1 mean over the
    last axis. Returns ``[corr@r4, corr@r3, ...]`` aligned with ``pyramid.corr_resolutions``.
    """
    b, n1 = corr0.size(0), corr0.size(1)
    corr = corr0
    out: List[torch.Tensor] = []
    for res in pyramid.corr_resolutions:
        n_level = pyramid.levels[res].num_nodes
        if corr.size(2) != n_level:
            corr = corr.reshape(b, n1, n_level, -1).mean(dim=3)
        out.append(corr)
    return out


# --------------------------------------------------------------------------- #
# §4.3 multi-level spherical lookup
# --------------------------------------------------------------------------- #
def pyramid_lookup(
    corr_pyramid: List[torch.Tensor],
    flow: torch.Tensor,
    pyramid: SpherePyramid,
) -> torch.Tensor:
    """Exp-map once at the estimation grid, then gather each corr level's neighborhood.

    Returns ``[B, N_est, sum_k M_k]`` — the lookup features concatenated across levels.
    """
    est = pyramid.estimation_level
    endpoint = endpoint_from_tangent_flow(est.points, flow, est.basis_east, est.basis_north)
    feats: List[torch.Tensor] = []
    for corr, res in zip(corr_pyramid, pyramid.corr_resolutions):
        level = pyramid.levels[res]
        center = level.ang2pix(endpoint)        # [B, N_est] nearest node at this level
        nbr = level.lookup_index[center]        # [B, N_est, M]
        feats.append(torch.gather(corr, 2, nbr))
    return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------- #
# §4.5 convex upsampling weight head
# --------------------------------------------------------------------------- #
class UpsampleWeightHead(nn.Module):
    """Predict per-descendant softmax weights over the estimation 1-hop neighborhood."""

    def __init__(
        self,
        hidden_channels: int,
        n_descendants: int,
        n_neighbors: int,
        kernel_size: int,
        mid_channels: int = 128,
    ):
        super().__init__()
        self.n_descendants = n_descendants
        self.n_neighbors = n_neighbors
        self.conv1 = SDPAConv(hidden_channels, mid_channels, kernel_size=kernel_size, node_dim=1)
        self.conv2 = SDPAConv(mid_channels, n_descendants * n_neighbors, kernel_size=1, node_dim=1)

    def forward(self, h, index, weight, valid) -> torch.Tensor:
        x = F.relu(self.conv1(h, index, weight, valid))
        w = self.conv2(x)  # 1x1 — no neighbors needed
        b, n, _ = w.shape
        w = w.reshape(b, n, self.n_descendants, self.n_neighbors)
        return torch.softmax(w, dim=-1)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class OSLORAFTPyramid(nn.Module):
    """HEALPix-native iterative flow: estimate at the coarse grid, supervise at the fine grid.

    ``__init__`` reads only *shapes* from ``pyramid`` (correlation width, conv kernel,
    descendant/neighbor counts); ``forward`` is handed the (device-moved) pyramid. Predictions
    are returned at the fine (supervision) resolution.
    """

    def __init__(
        self,
        pyramid: SpherePyramid,
        in_channels: int = 3,
        feature_channels: Optional[Tuple[int, ...]] = None,
        context_channels: Optional[Tuple[int, ...]] = None,
        hidden_channels: int = 96,
        context_dim: int = 64,
        flow_scale: float = 0.5,
        upsample_mid: int = 128,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.flow_scale = flow_scale
        self.include_xyz = True

        resolutions = list(
            range(pyramid.fine_resolution, pyramid.estimation_resolution - 1, -1)
        )
        # Encoder widths ramp linearly to hidden_channels over the pyramid depth, so they
        # auto-size to any est/fine pair: depth 3 (est=4) -> (32,64,96) — the validated
        # default — depth 2 (est=5) -> (48,96), depth 1 -> (96,).
        n_res = len(resolutions)
        default_channels = tuple(round(hidden_channels * (i + 1) / n_res) for i in range(n_res))
        if feature_channels is None:
            feature_channels = default_channels
        if context_channels is None:
            context_channels = default_channels
        kernel_size = pyramid.estimation_level.conv_kernel_size

        enc_in = in_channels + (3 if self.include_xyz else 0)
        self.fnet = PyramidEncoder(enc_in, feature_channels, kernel_size, resolutions)
        self.cnet = PyramidEncoder(enc_in, context_channels, kernel_size, resolutions)
        self.context_head = SDPAConv(
            self.cnet.out_channels, hidden_channels + context_dim, kernel_size=1, node_dim=1
        )
        self.correlation = AllPairsCorrelation()

        corr_channels = sum(
            pyramid.levels[r].lookup_index.size(1) for r in pyramid.corr_resolutions
        )
        self.motion_encoder = MotionEncoder(corr_channels, kernel_size)
        self.gru = GraphConvGRU(
            hidden_channels, self.motion_encoder.out_channels + context_dim, kernel_size
        )
        self.flow_conv1 = SDPAConv(hidden_channels, hidden_channels, kernel_size=kernel_size, node_dim=1)
        self.flow_conv2 = SDPAConv(hidden_channels, 2, kernel_size=kernel_size, node_dim=1)
        self._zero_init(self.flow_conv2)

        self.upsample_head = UpsampleWeightHead(
            hidden_channels,
            pyramid.descendant_index.size(1),
            pyramid.upsample_neighbors.size(1),
            kernel_size,
            mid_channels=upsample_mid,
        )

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
        pyramid: SpherePyramid,
        iters: int = 8,
        flow_init: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return the list of fine-resolution tangent-flow predictions, one per iteration."""
        est = pyramid.estimation_level
        fine = pyramid.fine_level
        idx, wgt, val = est.conv_index, est.conv_weight, est.conv_valid

        f1 = self.fnet(self._prep_input(frame1, fine), pyramid)
        f2 = self.fnet(self._prep_input(frame2, fine), pyramid)
        corr0 = self.correlation(f1, f2)
        corr_pyramid = build_correlation_pyramid(corr0, pyramid)

        ctx = self.cnet(self._prep_input(frame1, fine), pyramid)
        ctx = self.context_head(ctx)  # 1x1
        h, context = torch.split(ctx, [self.hidden_channels, self.context_dim], dim=-1)
        h = torch.tanh(h)
        context = F.relu(context)

        b, n_est = frame1.size(0), est.num_nodes
        flow = (
            flow_init
            if flow_init is not None
            else torch.zeros(b, n_est, 2, device=frame1.device, dtype=frame1.dtype)
        )

        predictions: List[torch.Tensor] = []
        for _ in range(iters):
            flow = flow.detach()
            corr_feat = pyramid_lookup(corr_pyramid, flow, pyramid)
            motion = self.motion_encoder(corr_feat, flow, idx, wgt, val)
            gru_in = torch.cat([motion, context], dim=-1)
            h = self.gru(h, gru_in, idx, wgt, val)
            delta = self.flow_conv2(F.relu(self.flow_conv1(h, idx, wgt, val)), idx, wgt, val)
            flow = flow + self.flow_scale * delta
            weights = self.upsample_head(h, idx, wgt, val)
            predictions.append(convex_upsample(flow, weights, pyramid))
        return predictions
