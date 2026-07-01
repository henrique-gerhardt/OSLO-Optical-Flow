"""OSLO-RAFT-Diff: a spherical Lucas-Kanade (differential) flow head.

The full ablation ladder (resolution r4/r5/r6, loss reweighting, correlation- and
context-starvation) proved the correlation-argmax OSLO-RAFT does no correspondence: a
frame-1-only appearance prior already reaches its +2.9% active ceiling, because the
inter-frame motion is *sub-node* (p50 0.1deg) and a discrete argmax cannot resolve
sub-node displacement at any affordable HEALPix grid.

The right tool for *sub-pixel* motion is a **differential** estimator. Sub-node motion is
exactly the small-displacement regime where the Lucas-Kanade linearization is valid, so no
iterative feature-warping is needed (a discrete warp can't move sub-node anyway) — a single
linearized solve is the correct estimator. Per node, on learned features ``f = fnet(frame)``:

  1. spatial gradient of ``f1`` in the tangent basis, ``G = (OᵀO)⁻¹ Oᵀ Δf1`` — a
     parameter-free least-squares fit over the conv neighborhood (``O`` = neighbour east/north
     offsets, fixed geometry);
  2. temporal difference ``Δf = f2 − f1`` (same node);
  3. the feature-constancy solve ``G·flow ≈ −Δf`` → ``flow = (S + λI)⁻¹ r`` with structure
     tensor ``S = Σ_c w_c gᵀg`` and ``r = −Σ_c w_c g Δf_c`` (a differentiable 2×2 solve).

The flow depends on ``f2`` through ``Δf``, so it *structurally cannot* fall back to a frame-1
prior — the decisive test of whether frame 2 (correspondence) is usable at all here. The only
learnable pieces are ``fnet`` (which carries the capacity: it learns features whose gradients
and temporal differences yield good flow), a per-channel reliability weight ``w_c``, the
regularizer ``λ``, and an output scale.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .oslo_raft import SphereEncoder, SphereLevel


def tangent_gradient_operator(level: SphereLevel) -> torch.Tensor:
    """Per-node least-squares tangent-gradient operator ``P`` with shape ``[N, 2, K]``.

    For node ``i`` with conv neighbours ``j`` at tangent offsets ``o_j = (p_j·ê, p_j·n̂)``,
    the local gradient of any field is ``G = P_i · [f_j − f_i]`` where ``P_i = (OᵀO)⁻¹ Oᵀ``
    is the Moore-Penrose pseudo-inverse of the ``[K, 2]`` offset matrix. Depends only on
    geometry, so it is computed once and reused across features/iterations.
    """
    points = level.points                                   # [N, 3]
    east, north = level.basis_east, level.basis_north       # [N, 3]
    nbr = level.conv_index                                  # [N, K]
    p_nbr = points[nbr]                                     # [N, K, 3]
    o_e = (p_nbr * east.unsqueeze(1)).sum(-1)              # [N, K]
    o_n = (p_nbr * north.unsqueeze(1)).sum(-1)            # [N, K]
    o = torch.stack([o_e, o_n], dim=-1)                    # [N, K, 2]
    m = o.transpose(1, 2) @ o                               # [N, 2, 2] structure tensor of positions
    eye = torch.eye(2, device=points.device, dtype=points.dtype)
    m_inv = torch.linalg.inv(m + 1e-6 * eye)               # neighbourhoods always span 2D; eps for safety
    p = m_inv @ o.transpose(1, 2)                           # [N, 2, K]
    return p


class OSLORAFTDiff(nn.Module):
    """Spherical Lucas-Kanade differential flow model (feature encoder + one LK solve)."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 96),
        kernel_size: int = 9,
        lambda_init: float = 0.1,
        scale_init: float = 0.1,
    ):
        super().__init__()
        self.include_xyz = True
        enc_in = in_channels + (3 if self.include_xyz else 0)
        self.fnet = SphereEncoder(enc_in, feature_channels, kernel_size)
        c = self.fnet.out_channels
        # Per-channel reliability weight (softplus-positive), the LK regularizer, and an
        # output scale (the LK flow magnitude is only defined up to the feature scale).
        self.channel_logit = nn.Parameter(torch.zeros(c))
        self.log_lambda = nn.Parameter(torch.tensor(float(torch.log(torch.expm1(torch.tensor(lambda_init))))))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale_init))))
        self._grad_op: Optional[torch.Tensor] = None
        self._grad_op_key: Optional[Tuple[int, torch.device]] = None

    def _prep_input(self, frame, level):
        if self.include_xyz:
            xyz = level.points.unsqueeze(0).expand(frame.size(0), -1, -1)
            return torch.cat([frame, xyz], dim=-1)
        return frame

    def _gradient_operator(self, level: SphereLevel) -> torch.Tensor:
        key = (level.num_nodes, level.points.device)
        if self._grad_op_key != key:  # geometry is fixed; build once per level/device
            self._grad_op = tangent_gradient_operator(level)
            self._grad_op_key = key
        return self._grad_op

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        level: SphereLevel,
        iters: int = 1,
        flow_init: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return ``[flow]`` — a single differential tangent-flow estimate (LK is one-shot)."""
        idx, wgt, val = level.conv_index, level.conv_weight, level.conv_valid
        f1 = self.fnet(self._prep_input(frame1, level), idx, wgt, val)   # [B, N, C]
        f2 = self.fnet(self._prep_input(frame2, level), idx, wgt, val)

        p = self._gradient_operator(level).to(f1.dtype)                  # [N, 2, K]
        f1_nbr = f1[:, idx]                                              # [B, N, K, C]
        df1 = f1_nbr - f1.unsqueeze(2)                                  # [B, N, K, C]
        # G[b,n,a,c] = sum_k P[n,a,k] * df1[b,n,k,c]  (spatial gradient of f1, [B,N,2,C])
        grad = torch.einsum("nak,bnkc->bnac", p, df1)
        dt = f2 - f1                                                    # [B, N, C] temporal difference

        w = F.softplus(self.channel_logit)                              # [C] >= 0
        gw = grad * w.view(1, 1, 1, -1)                                 # [B, N, 2, C]
        s = gw @ grad.transpose(-1, -2)                                 # [B, N, 2, 2] structure tensor
        r = -(gw @ dt.unsqueeze(-1)).squeeze(-1)                        # [B, N, 2] rhs

        # The 2x2 solve runs in fp32 (linalg.solve is fragile / unsupported in fp16 under AMP),
        # then the result is cast back to the working dtype.
        lam = F.softplus(self.log_lambda.float())
        eye = torch.eye(2, device=f1.device, dtype=torch.float32)
        flow = torch.linalg.solve(s.float() + lam * eye, r.float().unsqueeze(-1)).squeeze(-1)
        flow = (flow * torch.exp(self.log_scale.float())).to(r.dtype)   # [B, N, 2]
        return [flow]
