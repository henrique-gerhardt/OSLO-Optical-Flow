"""OSLO-RAFT-R: the retina model — decouple the input grid from the estimation grid.

The post-mortem of the negative result (docs/OSLO_RAFT_RETINA_PLAN.md §1) reframed the
ladder's wall: every prior variant sampled its input *at* the estimation grid, so no
encoder — however good — could represent sub-node motion; and the `ang2pix`-snap lookup
made the correlation feature piecewise-constant in sub-node flow, mechanically consistent
with the correlation going inert. RAFT itself never has either problem: its retina is
full-res (only the corr grid is 1/8) and its lookup samples the volume *bilinearly at
continuous coordinates*.

This model ports both properties to the sphere (plan §2-§6). Three grids:

    retina  r_ret (7/8)   frames sampled here; PyramidEncoder runs r_ret -> r_est
    est     r_est (4/5)   correlation, interp lookup, ConvGRU, delta head
    sup     r_sup (6)     convex-upsampled predictions; loss + metrics (unchanged)

and one new operator, :func:`interp_pyramid_lookup` — a *lazy, interpolated* correlation
lookup that replaces `AllPairsCorrelation` + `build_correlation_pyramid` +
`pyramid_lookup`:

  - **lazy** like `local_correlation_lookup`: the [N, N] volume is never materialized,
    so estimation grids beyond r4 stay affordable;
  - **continuous in flow** like RAFT's bilinear sampling. Key identity (linearity of the
    dot product in f2): interpolating the correlation volume == correlating against
    interpolated features, ``sum_k w_k (f1 . f2_k) = f1 . (sum_k w_k f2_k)`` — so we
    interpolate `f2` features at continuous query directions and never need the volume.
    Likewise RAFT's corr-pyramid pooling == dotting f1 with child-averaged (normalized)
    f2 features, so the "correlation pyramid" is just 4-to-1 pooled f2 *without*
    re-normalizing (`build_feature_pyramid`).

Everything else is reused unchanged: `PyramidEncoder` (now checkpointable),
`GraphConvGRU`, `MotionEncoder`, the zero-init flow head, `UpsampleWeightHead` +
`convex_upsample`, and the `_update_step` iteration-checkpointing pattern validated for
`OSLORAFTLocal`. Existing models are untouched (house convention for every variant).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import endpoint_from_tangent_flow, tangent_basis
from .healpix_pyramid import SpherePyramid, convex_upsample, pool_features
from .models import SDPAConv
from .oslo_raft import GraphConvGRU, MotionEncoder
from .oslo_raft_pyramid import PyramidEncoder, UpsampleWeightHead

# One [chunk, N_level] fp32 block cap for the brute nearest-node resolve inside the
# lookup (64M floats = 256 MB). Chunking keeps est=5 (12k nodes x 17 offsets) affordable.
_NEAREST_BUDGET_ELEMS = 64 * 1024 * 1024

# Encoder channel ramps by pyramid depth (retina..est inclusive), for hidden=96. The
# linear auto-ramp of OSLORAFTPyramid gives too-fat early stages at depth 4-6 (retina
# levels have 16-64x the nodes of the estimation grid, so early widths dominate compute).
# Depths 1-3 reproduce the validated pre-retina defaults exactly.
_DEFAULT_RAMPS = {
    1: (96,),
    2: (48, 96),
    3: (32, 64, 96),
    4: (16, 32, 64, 96),
    5: (16, 32, 48, 64, 96),
    6: (16, 24, 32, 48, 64, 96),
}


# --------------------------------------------------------------------------- #
# Lookup stencil (§5.2)
# --------------------------------------------------------------------------- #
def lookup_stencil(rings: int = 2, ring_points: int = 8) -> torch.Tensor:
    """Unit tangent-plane stencil ``[M, 2]``: center + ``rings`` rings of ``ring_points``.

    Ring radii are 1..rings (in units of the level's node spacing); ring r is rotated by
    ``(r-1) * pi / ring_points`` so consecutive rings' directions interleave.
    """
    offsets = [(0.0, 0.0)]
    for rho in range(1, rings + 1):
        for j in range(ring_points):
            theta = 2.0 * math.pi * j / ring_points + (rho - 1) * math.pi / ring_points
            offsets.append((rho * math.cos(theta), rho * math.sin(theta)))
    return torch.tensor(offsets, dtype=torch.float32)


def build_lookup_offsets(
    pyramid: SpherePyramid, rings: int = 2, ring_points: int = 8
) -> torch.Tensor:
    """Per-corr-level stencils ``[L, M, 2]``, scaled by each level's node spacing
    ``s_l = sqrt(4 pi / N_l)`` (radians), aligned with ``pyramid.corr_resolutions``."""
    base = lookup_stencil(rings, ring_points)
    scales = torch.tensor(
        [
            math.sqrt(4.0 * math.pi / pyramid.levels[r].num_nodes)
            for r in pyramid.corr_resolutions
        ],
        dtype=torch.float32,
    )
    return scales.view(-1, 1, 1) * base.unsqueeze(0)


# --------------------------------------------------------------------------- #
# Feature pyramid (§5.1) — replaces the correlation pyramid, by linearity
# --------------------------------------------------------------------------- #
def build_feature_pyramid(f2: torch.Tensor, pyramid: SpherePyramid) -> List[torch.Tensor]:
    """Pool channel-normalized ``f2`` down the corr levels WITHOUT re-normalizing.

    ``f2`` is ``[B, N_est, C]``, already ``F.normalize``d once by the caller. Plain
    4-to-1 mean pooling of *normalized* features makes ``f1 . pooled(f2)`` exactly equal
    the 4-to-1 pooled all-pairs correlation (the linearity identity) — i.e. this list is
    the lazy twin of ``build_correlation_pyramid``, aligned with ``corr_resolutions``.
    """
    levels = [f2]
    for res in pyramid.corr_resolutions[1:]:
        levels.append(pool_features(levels[-1], pyramid.pool_index[res]))
    return levels


# --------------------------------------------------------------------------- #
# The interpolated lazy lookup (§5.3)
# --------------------------------------------------------------------------- #
def _expmap_batched(base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exp-map for arbitrary leading dims (``geometry.expmap`` is fixed to [B, N, 3])."""
    theta = tangent.norm(dim=-1, keepdim=True)
    scale = torch.sin(theta) / theta.clamp_min(eps)
    out = torch.cos(theta) * base + scale * tangent
    out = torch.where(theta < eps, base + tangent, out)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(eps)


def _chord_geodesic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic distance via the chord: ``2 asin(|a-b| / 2)``.

    Precise near zero, unlike ``acos(a.b)`` whose fp32 clamp floors distances at
    ~4.5e-4 rad — which would leak weight off exact-node queries and break the
    cold-start diagonal parity the smoke asserts.
    """
    chord = (a - b).norm(dim=-1)
    return 2.0 * torch.asin((0.5 * chord).clamp(max=1.0))


def _nearest_node(q: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Chunked brute nearest-node resolve (same argmax as ``level.ang2pix``).

    NOTE (est=r6 seam): this is the lookup's only non-local op. To estimate at r6
    later, replace it with the windowed candidate search from ``OSLORAFTLocal``
    (argmax among ``points[lookup_index[i]]``) — everything downstream is unchanged.
    """
    flat = q.reshape(-1, 3)
    n_level = points.size(0)
    chunk = max(1, _NEAREST_BUDGET_ELEMS // max(n_level, 1))
    if flat.size(0) <= chunk:
        idx = (flat @ points.t()).argmax(dim=-1)
    else:
        parts = [
            (flat[s : s + chunk] @ points.t()).argmax(dim=-1)
            for s in range(0, flat.size(0), chunk)
        ]
        idx = torch.cat(parts, dim=0)
    return idx.reshape(q.shape[:-1])


def interp_pyramid_lookup(
    f1: torch.Tensor,
    f2_levels: List[torch.Tensor],
    flow: torch.Tensor,
    pyramid: SpherePyramid,
    offsets_per_level: torch.Tensor,
    k_interp: int = 3,
) -> torch.Tensor:
    """Continuous-in-flow correlation lookup over the feature pyramid.

    Per node: exp-map the current tangent flow to its endpoint, place each level's
    stencil in the endpoint's tangent plane, and for every query direction interpolate
    ``f2`` from its ``k_interp`` nearest nodes (inverse-distance weights) before dotting
    with ``f1``. Because the interpolation weights vary smoothly with the query, the
    result is (piecewise-)smooth in sub-node flow — the property the old snap-gather
    lookup lacked, and the point of §5.

    Interpolation geometry (nearest nodes, distances, weights) is computed from the
    detached flow — the same contract as RAFT, which detaches its lookup coordinates
    each iteration; gradients flow through the gathered *feature values* only.

    Args:
        f1: ``[B, N_est, C]`` channel-normalized first-image features.
        f2_levels: per-corr-level normalized-then-pooled second-image features
            (:func:`build_feature_pyramid`).
        flow: ``[B, N_est, 2]`` current tangent flow (already detached by the caller).
        pyramid: geometry bundle.
        offsets_per_level: ``[L, M, 2]`` stencils from :func:`build_lookup_offsets`.
        k_interp: nodes per interpolated sample (3 ~ barycentric).

    Returns:
        ``[B, N_est, L * M]`` correlation features, levels concatenated.
    """
    est = pyramid.estimation_level
    b, n, c = f1.shape

    endpoint = endpoint_from_tangent_flow(
        est.points, flow, est.basis_east, est.basis_north
    )  # [B, N, 3]
    e_east, e_north = tangent_basis(endpoint.reshape(-1, 3))
    e_east = e_east.reshape(b, n, 1, 3)
    e_north = e_north.reshape(b, n, 1, 3)

    feats: List[torch.Tensor] = []
    for f2_l, offsets, res in zip(f2_levels, offsets_per_level, pyramid.corr_resolutions):
        level = pyramid.levels[res]
        m = offsets.size(0)

        # Query directions: stencil offsets exp-mapped from the endpoint.  [B, N, M, 3]
        t = offsets[:, 0].view(1, 1, m, 1) * e_east + offsets[:, 1].view(1, 1, m, 1) * e_north
        q = _expmap_batched(endpoint.unsqueeze(2), t)

        # Candidates: the nearest node's lookup ring contains the true k-nearest set
        # for any query inside that node's cell.  [B, N, M, K_c]
        center = _nearest_node(q, level.points)
        cand = level.lookup_index[center][..., : min(9, level.lookup_index.size(1))]
        cand_pts = level.points[cand]                                  # [B, N, M, K_c, 3]

        d = _chord_geodesic(q.unsqueeze(-2), cand_pts)                 # [B, N, M, K_c]
        k_int = min(k_interp, d.size(-1))
        d_sel, slot = torch.topk(d, k_int, dim=-1, largest=False)
        w = 1.0 / d_sel.clamp_min(1e-8)
        w = (w / w.sum(dim=-1, keepdim=True)).detach()                 # [B, N, M, K]
        sel = torch.gather(cand, -1, slot)                             # [B, N, M, K]

        # Flat-index gather of f2 (the OSLORAFTLocal trick — no [B, N, N, C] transient).
        n_l = f2_l.size(1)
        base = (torch.arange(b, device=f2_l.device) * n_l).view(b, 1, 1, 1)
        flat = (sel + base).reshape(-1)
        vals = f2_l.reshape(b * n_l, c).index_select(0, flat).reshape(b, n, m, k_int, c)
        f2_interp = (w.unsqueeze(-1) * vals).sum(dim=-2)               # [B, N, M, C]

        # Cosine units, deliberately NOT the /sqrt(C) of AllPairsCorrelation: with
        # channel-normalized features the dot is already in [-1, 1]; shrinking it by
        # another sqrt(96) leaves the motion encoder a ~0.1-max whisper against O(1)
        # context features (RAFT's 1/sqrt(C) tempers RAW dots, whose scale grows with
        # C — normalized dots need no temper).
        feats.append((f1.unsqueeze(2) * f2_interp).sum(dim=-1))       # [B, N, M]
    return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------- #
# Auxiliary stencil matching loss (the plan §10 "corr inert" fallback, promoted)
# --------------------------------------------------------------------------- #
def stencil_match_loss(
    f1: torch.Tensor,
    f2: torch.Tensor,
    gt_endpoint: torch.Tensor,
    level,
    valid: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    sigma_nodes: float = 0.5,
) -> torch.Tensor:
    """Soft-target InfoNCE over each node's lookup window at the estimation grid.

    Why this exists (measured 2026-07-02, the §9.2d gate): with the retina + the
    continuous lookup in place, end-to-end training STILL never discovers correlation
    on a from-scratch budget — the loss tracks the zero-flow baseline exactly while
    the corr features carry no learned meaning (the GRU's flow head has no reason to
    trust random-init correlations, and the encoder gets no gradient to make features
    matchable: a chicken-and-egg). This loss breaks the loop by supervising matching
    *directly*: for node ``i`` with ground-truth endpoint ``e_i``, the cosine logits of
    ``f1_i`` against ``f2`` over ``i``'s lookup window must follow the geometry —
    candidates near ``e_i`` are positives (a soft target ``exp(-d^2 / 2 sigma^2)``, so
    sub-node motion still splits mass between self and the toward-motion neighbor,
    keeping signal below one node where hard InfoNCE would degenerate to self-match).

    Args:
        f1, f2: ``[B, N, C]`` channel-normalized est-grid features (the same tensors
            the lookup consumes; ``f2`` is the finest entry of the feature pyramid).
        gt_endpoint: ``[B, N, 3]`` ground-truth correspondence directions at est nodes.
        level: the estimation :class:`SphereLevel` (uses ``points``/``lookup_index``).
        valid: optional ``[B, N]`` mask.
        temperature: softmax temperature over cosine logits.
        sigma_nodes: soft-target width in units of the node spacing.

    Returns:
        Scalar loss (0 when no node's endpoint lands inside its lookup window).
    """
    b, n, c = f1.shape
    cand = level.lookup_index                                       # [N, M]
    cand_pts = level.points[cand]                                   # [N, M, 3]
    spacing = math.sqrt(4.0 * math.pi / n)

    d = _chord_geodesic(gt_endpoint.unsqueeze(2), cand_pts.unsqueeze(0))  # [B, N, M]
    # Usable only where the endpoint actually lies inside the window (supra-window
    # motion has no positive to point at — skip those nodes, don't mislabel them).
    inside = d.min(dim=-1).values < spacing
    if valid is not None:
        inside = inside & valid.bool()
    if not inside.any():
        return f1.new_zeros(())

    sigma = sigma_nodes * spacing
    target = torch.softmax(-0.5 * (d / sigma) ** 2, dim=-1)          # [B, N, M]

    m = cand.size(1)
    f2c = f2[:, cand.reshape(-1)].reshape(b, n, m, c)                # [B, N, M, C]
    logits = (f1.unsqueeze(2) * f2c).sum(dim=-1) / temperature       # [B, N, M]
    loss_per_node = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return loss_per_node[inside].mean()


# --------------------------------------------------------------------------- #
# The model (§6)
# --------------------------------------------------------------------------- #
class OSLORAFTRetina(nn.Module):
    """Retina-decoupled OSLO-RAFT: ingest at r_ret, estimate at r_est, supervise at r_sup.

    Structure clones ``OSLORAFTPyramid`` (whose ``__init__`` reads only *shapes* from the
    pyramid); the deltas are the retina-deep encoders, the §5 lazy interpolated lookup in
    place of the all-pairs volume, and gradient checkpointing on both the encoder stages
    and the update iterations (the `OSLORAFTLocal` pattern — forward keeps only the
    per-iteration boundary tensors and recomputes the block in backward).
    """

    # Diagnostic ablations — identical semantics to OSLORAFT; these are the §8 gates
    # (Gate R1 = the corr ablation must finally HURT), so they must keep working here.
    ablate_corr: bool = False
    ablate_context: bool = False

    # Iteration-level checkpointing (the encoder has its own flag, set in __init__).
    use_checkpoint: bool = True

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
        lookup_rings: int = 2,
        lookup_ring_points: int = 8,
        k_interp: int = 3,
        use_checkpoint_encoder: bool = True,
    ):
        super().__init__()
        if pyramid.retina_resolution is None:
            raise ValueError("OSLORAFTRetina needs a pyramid built with retina_resolution")
        if not (
            pyramid.estimation_resolution
            < pyramid.fine_resolution
            <= pyramid.retina_resolution
        ):
            raise ValueError(
                "need estimation < fine (supervision) <= retina, got "
                f"est={pyramid.estimation_resolution} fine={pyramid.fine_resolution} "
                f"retina={pyramid.retina_resolution}"
            )
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.flow_scale = flow_scale
        self.k_interp = k_interp
        # Position channels and matching don't mix: xyz in the FEATURE encoder makes
        # every f1_i . f2_c logit peak at the spatially nearest candidate regardless of
        # texture (a built-in self-match bias — RAFT's fnet sees only image content for
        # this reason). The pre-retina models fed xyz to both nets; here the fnet is
        # position-free by design and only the context net keeps xyz (a legitimate
        # positional prior for the GRU).
        self.include_xyz_fnet = False
        self.include_xyz_cnet = True

        resolutions = list(
            range(pyramid.retina_resolution, pyramid.estimation_resolution - 1, -1)
        )
        depth = len(resolutions)
        if feature_channels is None or context_channels is None:
            if depth in _DEFAULT_RAMPS:
                base = _DEFAULT_RAMPS[depth]
            else:  # deeper than the table: pad the front with the narrowest stem
                base = (_DEFAULT_RAMPS[6][0],) * (depth - 6) + _DEFAULT_RAMPS[6]
            default = tuple(max(8, round(cw * hidden_channels / 96)) for cw in base)
            feature_channels = feature_channels or default
            context_channels = context_channels or default
        kernel_size = pyramid.estimation_level.conv_kernel_size

        fnet_in = in_channels + (3 if self.include_xyz_fnet else 0)
        cnet_in = in_channels + (3 if self.include_xyz_cnet else 0)
        self.fnet = PyramidEncoder(fnet_in, feature_channels, kernel_size, resolutions)
        self.cnet = PyramidEncoder(cnet_in, context_channels, kernel_size, resolutions)
        self.fnet.use_checkpoint = use_checkpoint_encoder
        self.cnet.use_checkpoint = use_checkpoint_encoder
        self.context_head = SDPAConv(
            self.cnet.out_channels, hidden_channels + context_dim, kernel_size=1, node_dim=1
        )

        # [L, M, 2] stencils; a buffer so .to(device)/state_dict handle it. Stencil shape
        # is architecture (it sets corr width), so it belongs in the checkpoint.
        self.register_buffer(
            "lookup_offsets", build_lookup_offsets(pyramid, lookup_rings, lookup_ring_points)
        )
        corr_channels = self.lookup_offsets.size(0) * self.lookup_offsets.size(1)
        self.motion_encoder = MotionEncoder(corr_channels, kernel_size)
        # The raw correlation features skip directly into the GRU input alongside the
        # encoded motion features. Measured necessity (2026-07-02): with matchable
        # features the corr stencil is LINEARLY decodable to flow (probe cos-sim 0.99),
        # yet through motion-encoder->GRU->head alone the decode never aligned within a
        # smoke budget — the skip shortens the corr->delta path to two hops.
        self.gru = GraphConvGRU(
            hidden_channels,
            self.motion_encoder.out_channels + context_dim + corr_channels,
            kernel_size,
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

    def _prep_input(self, frame, level, include_xyz):
        if include_xyz:
            xyz = level.points.unsqueeze(0).expand(frame.size(0), -1, -1)
            return torch.cat([frame, xyz], dim=-1)
        return frame

    def _update_step(
        self,
        h: torch.Tensor,
        flow: torch.Tensor,
        f1: torch.Tensor,
        f2_levels: List[torch.Tensor],
        context: torch.Tensor,
        pyramid: SpherePyramid,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One refinement iteration: interp lookup -> motion -> GRU -> heads.

        Self-contained over its tensor inputs so it can be wrapped in
        ``torch.utils.checkpoint.checkpoint`` (the OSLORAFTLocal pattern; the pyramid
        passes through untouched like `level` does there).
        """
        est = pyramid.estimation_level
        idx, wgt, val = est.conv_index, est.conv_weight, est.conv_valid
        corr_feat = interp_pyramid_lookup(
            f1, f2_levels, flow, pyramid, self.lookup_offsets, self.k_interp
        )
        if self.ablate_corr:
            corr_feat = torch.zeros_like(corr_feat)
        motion = self.motion_encoder(corr_feat, flow, idx, wgt, val)
        gru_in = torch.cat([motion, context, corr_feat], dim=-1)
        h = self.gru(h, gru_in, idx, wgt, val)
        delta = self.flow_conv2(F.relu(self.flow_conv1(h, idx, wgt, val)), idx, wgt, val)
        weights = self.upsample_head(h, idx, wgt, val)
        return h, delta, weights

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        pyramid: SpherePyramid,
        iters: int = 8,
        flow_init: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> List[torch.Tensor]:
        """frames ``[B, N_retina, 3]`` -> per-iteration flow predictions ``[B, N_sup, 2]``.

        ``return_features=True`` additionally returns the normalized est-grid feature
        pair ``(f1, f2)`` for :func:`stencil_match_loss`.
        """
        est = pyramid.estimation_level
        retina = pyramid.retina_level

        # Channel-normalize once at the estimation grid (all-pairs parity), then pool
        # WITHOUT re-normalizing (see build_feature_pyramid).
        f1 = F.normalize(
            self.fnet(self._prep_input(frame1, retina, self.include_xyz_fnet), pyramid), dim=-1
        )
        f2 = F.normalize(
            self.fnet(self._prep_input(frame2, retina, self.include_xyz_fnet), pyramid), dim=-1
        )
        f2_levels = build_feature_pyramid(f2, pyramid)

        ctx = self.cnet(self._prep_input(frame1, retina, self.include_xyz_cnet), pyramid)
        ctx = self.context_head(ctx)  # 1x1
        h, context = torch.split(ctx, [self.hidden_channels, self.context_dim], dim=-1)
        h = torch.tanh(h)
        context = F.relu(context)
        if self.ablate_context:
            h = torch.zeros_like(h)
            context = torch.zeros_like(context)

        b, n_est = frame1.size(0), est.num_nodes
        flow = (
            flow_init
            if flow_init is not None
            else torch.zeros(b, n_est, 2, device=frame1.device, dtype=frame1.dtype)
        )

        predictions: List[torch.Tensor] = []
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for _ in range(iters):
            flow = flow.detach()
            if ckpt:
                h, delta, weights = checkpoint(
                    self._update_step, h, flow, f1, f2_levels, context, pyramid,
                    use_reentrant=False,
                )
            else:
                h, delta, weights = self._update_step(h, flow, f1, f2_levels, context, pyramid)
            flow = flow + self.flow_scale * delta
            predictions.append(convex_upsample(flow, weights, pyramid))
        if return_features:
            return predictions, (f1, f2_levels[0])
        return predictions
