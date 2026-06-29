"""OSLO-RAFT with a memory-local correlation lookup — the r6 single-resolution model.

The all-pairs runs (single-res r4, multi-res est=4 and est=5) all stalled at the same
+2.8% active-subset ceiling. The diagnosis (see the shakeout notes / OSLO_RAFT_PLAN):
the cosine-correlation argmax is only *discriminative* once motion exceeds ~half a node;
below that it lands on self and yields no gradient. At r4 (3.67deg) and r5 (1.83deg)
every active subset is still sub-half-node, so neither crosses the threshold. It is first
crossed at **r6 (0.92deg)**, where active>0.5deg = 0.54 node and active>1.0deg = 1.09 node
become resolvable.

Estimating at r6 is impossible with the all-pairs volume — ``[B, N, N]`` at N=49152 is
9.66 GB at B=1. This model removes that wall: the correlation is never materialized as a
full volume. Instead the spherical lookup gathers ``f2`` over each node's flow-displaced
neighborhood and correlates with ``f1`` lazily, costing ``O(N*M*C)`` (~0.5 GB) instead of
``O(N^2)``. The gathered values are **identical** to gathering from the all-pairs volume
for bounded motion (the true nearest node lies inside the node's lookup neighborhood), so
this is a drop-in scaling of the validated single-resolution model — same architecture,
same parameters, same math, just affordable at r6.

Everything else (encoders, ConvGRU, motion encoder, flow head, the cold-start zero-init
contract) is reused unchanged from :mod:`spherical_flow.oslo_raft`; only the correlation
path differs. The all-pairs :class:`~spherical_flow.oslo_raft.OSLORAFT` is left untouched.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import endpoint_from_tangent_flow
from .oslo_raft import OSLORAFT, SphereLevel


def local_correlation_lookup(
    f1: torch.Tensor,
    f2: torch.Tensor,
    flow: torch.Tensor,
    level: SphereLevel,
    cand_points: torch.Tensor,
) -> torch.Tensor:
    """Lazy local correlation gather (the all-pairs lookup without the [B,N,N] volume).

    Mirrors :func:`spherical_flow.oslo_raft.spherical_lookup` exactly, but never builds the
    all-pairs correlation. For each node ``i`` with tangent flow ``f_i``:

      1. endpoint ``e_i = expmap(p_i, f_i)``;
      2. *local* ang2pix: pick the nearest node to ``e_i`` among ``i``'s own lookup
         candidates ``cand_points[i]`` (= ``points[lookup_index[i]]``) — no global
         ``[N, N]`` argmax. For bounded motion (the displaced node is one of ``i``'s
         nearest) this returns the same node the global ang2pix would; for the rare
         transient where flow exceeds the lookup radius it clamps to the window edge
         (graceful, no crash), exactly the regime the all-pairs model has no answer for
         either;
      3. gather ``f2`` over that center node's lookup neighborhood and correlate with
         ``f1[i]``.

    ``f1`` and ``f2`` must already be channel-normalized (done once by the caller), so the
    returned values match ``matmul(normalize(f1), normalize(f2).T) / sqrt(C)`` gathered at
    the same neighborhood — i.e. identical to the all-pairs path.

    Args:
        f1, f2: ``[B, N, C]`` channel-normalized features (first / second image).
        flow: ``[B, N, 2]`` current tangent flow at the estimation grid.
        level: the (single-resolution) geometry bundle; uses ``points``, the tangent basis
            and ``lookup_index`` ``[N, M]`` (M nearest incl. self at column 0).
        cand_points: ``[N, M, 3]`` = ``points[lookup_index]``, precomputed once per forward.

    Returns:
        ``[B, N, M]`` per-node correlation feature.
    """
    b, n, c = f1.shape
    lut = level.lookup_index                                   # [N, M]
    m = lut.size(1)

    endpoint = endpoint_from_tangent_flow(
        level.points, flow, level.basis_east, level.basis_north
    )                                                          # [B, N, 3]

    # Local ang2pix: nearest candidate node to the endpoint within each node's window.
    sim = (endpoint.unsqueeze(2) * cand_points.unsqueeze(0)).sum(-1)   # [B, N, M]
    slot = sim.argmax(-1)                                              # [B, N]
    center = torch.gather(lut.expand(b, n, m), 2, slot.unsqueeze(-1)).squeeze(-1)  # [B, N]
    nbr = lut[center]                                                  # [B, N, M] absolute

    # Gather f2 over the neighborhood and correlate with f1 (flat-index to avoid [B,N,N,C]).
    base = (torch.arange(b, device=f2.device) * n).view(b, 1, 1)       # [B, 1, 1]
    flat = (nbr + base).reshape(-1)                                    # [B*N*M]
    f2_nbr = f2.reshape(b * n, c).index_select(0, flat).reshape(b, n, m, c)
    corr = (f1.unsqueeze(2) * f2_nbr).sum(-1) / math.sqrt(c)           # [B, N, M]
    return corr


class OSLORAFTLocal(OSLORAFT):
    """Single-resolution OSLO-RAFT whose correlation lookup is local (so r6 fits).

    Identical ``__init__`` / parameters / submodules to
    :class:`~spherical_flow.oslo_raft.OSLORAFT`; the unused ``self.correlation`` all-pairs
    module is harmless (no parameters). Only :meth:`forward` changes — the precomputed
    ``[B, N, N]`` volume and :func:`spherical_lookup` are replaced by
    :func:`local_correlation_lookup`.

    Because the GRU/motion/flow stack now iterates over all ~49k r6 nodes, storing every
    iteration's activations for backward overflows a 24 GB card at B=2 (the 8 lookups alone
    are ~7 GB in fp32). :attr:`use_checkpoint` gradient-checkpoints each iteration's update
    (RAFT-style): the forward keeps only the per-iteration boundary tensors (``h``, ``flow``)
    and recomputes the block in backward, collapsing the 8x activation stack to ~1x. Eval
    (no grad) skips checkpointing automatically.
    """

    use_checkpoint: bool = True

    def _update_step(
        self,
        h: torch.Tensor,
        flow: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        context: torch.Tensor,
        cand_points: torch.Tensor,
        idx: torch.Tensor,
        wgt: torch.Tensor,
        val: torch.Tensor,
        level: SphereLevel,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One refinement iteration: local lookup -> motion -> GRU -> zero-init flow head.

        Returns the new hidden state and the flow delta (the caller adds ``flow_scale*delta``
        to the detached flow). Self-contained over its tensor inputs so it can be wrapped in
        :func:`torch.utils.checkpoint.checkpoint`.
        """
        corr_feat = local_correlation_lookup(f1, f2, flow, level, cand_points)
        motion = self.motion_encoder(corr_feat, flow, idx, wgt, val)
        gru_in = torch.cat([motion, context], dim=-1)
        h = self.gru(h, gru_in, idx, wgt, val)
        delta = self.flow_conv2(F.relu(self.flow_conv1(h, idx, wgt, val)), idx, wgt, val)
        return h, delta

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

        # Channel-normalize once (the all-pairs correlation normalizes the whole feature
        # map; gathering an already-normalized f2 reproduces those values exactly).
        f1 = F.normalize(self.fnet(self._prep_input(frame1, level), idx, wgt, val), dim=-1)
        f2 = F.normalize(self.fnet(self._prep_input(frame2, level), idx, wgt, val), dim=-1)
        cand_points = level.points[level.lookup_index]        # [N, M, 3], once per forward

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
        ckpt = self.use_checkpoint and torch.is_grad_enabled()
        for _ in range(iters):
            flow = flow.detach()
            if ckpt:
                # use_reentrant=False is the modern, AMP-safe variant; passes the non-grad
                # geometry tensors through untouched and recomputes only this block.
                h, delta = checkpoint(
                    self._update_step, h, flow, f1, f2, context,
                    cand_points, idx, wgt, val, level, use_reentrant=False,
                )
            else:
                h, delta = self._update_step(
                    h, flow, f1, f2, context, cand_points, idx, wgt, val, level
                )
            flow = flow + self.flow_scale * delta
            predictions.append(flow)
        return predictions
