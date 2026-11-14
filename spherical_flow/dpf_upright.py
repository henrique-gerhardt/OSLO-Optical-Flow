"""Angle path of the Dual-Projection Fusion upright estimator (arXiv 2512.00911), as an external baseline.

Minimal re-statement of the released model's angle-only path (ERP ConvNeXt-base encoder with
circular / zero padding and the high-frequency block, global average pool, one linear layer,
sigmoid): github.com/YuhaoShine/DualProjectionFusion, MIT License, Copyright (c) the authors.
The cubemap ViT branch, the fusion blocks and the upright-image decoder are not needed for the
angles (the authors' own ForTEST_ImplicitFusion_angle_only_runtime.py) and are not included. timm
is not required: DropPath is inactive at inference.

The network outputs pitch and roll in degrees in [-90, 90] for a 256x512 ERP image in [0, 1]. The
released code does not state how the two angles compose into a rotation, so `fit_convention`
picks, on frames with a known rotation, the composition that reproduces the known up vector.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_HW = (256, 512)
PREFIXES = (("ConvNext_branch.equi_encoder.", "encoder."), ("fuse_branch.fc.", "fc."))


class _HFM(nn.Module):
    def forward(self, x):
        low = F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear", align_corners=True)
        return 2 * x - low


class _LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, channels_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps, self.channels_first, self.shape = eps, channels_first, (dim,)

    def forward(self, x):
        if not self.channels_first:
            return F.layer_norm(x, self.shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim)
        self.norm = _LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.dim = dim
        self.hfm = _HFM()

    def forward(self, x):
        skip = x
        x = self.hfm(x)
        if self.dim < 256:
            x = F.pad(F.pad(x, (3, 3, 0, 0), mode="circular"), (0, 0, 3, 3))
        else:
            x = F.pad(x, (3, 3, 3, 3))
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x)))) * self.gamma
        return skip + x.permute(0, 3, 1, 2)


class _ConvNeXt(nn.Module):
    def __init__(self, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)):
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), _LayerNorm(dims[0], channels_first=True))]
            + [nn.Sequential(_LayerNorm(dims[i], channels_first=True), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2))
               for i in range(3)])
        self.stages = nn.ModuleList(nn.Sequential(*[_Block(dims[i]) for _ in range(depths[i])]) for i in range(4))
        self.hfm = _HFM()

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(self.hfm(x)))
        return x


class DPFAngles(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ConvNeXt()
        self.fc = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, 256, 512] in [0, 1] -> [B, 2] (pitch, roll) degrees."""
        return torch.sigmoid(self.fc(self.encoder(x).mean(dim=(-2, -1)))) * 180.0 - 90.0


def load_dpf(path: str, device) -> DPFAngles:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    own = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        for src, dst in PREFIXES:
            if key.startswith(src) and not key[len(src):].startswith(("norm.", "head.")):
                own[dst + key[len(src):]] = value
    model = DPFAngles()
    model.load_state_dict(own, strict=True)
    return model.to(device).eval()


def prepare(frame_hwc01: torch.Tensor) -> torch.Tensor:
    """ERP frame [H, W, 3] in [0, 1] -> [3, 256, 512]."""
    x = frame_hwc01.permute(2, 0, 1)[None]
    if tuple(x.shape[-2:]) != INPUT_HW:
        x = F.interpolate(x, size=INPUT_HW, mode="area")
    return x[0]


def _axis_rot(axis: int, deg: torch.Tensor) -> torch.Tensor:
    a = torch.deg2rad(deg)
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    rows = {0: [o, z, z, z, c, -s, z, s, c], 1: [c, z, s, z, o, z, -s, z, c], 2: [c, -s, z, s, c, z, z, z, o]}[axis]
    return torch.stack(rows, dim=-1).reshape(*a.shape, 3, 3)


# A convention: (first axis, second axis, sign of pitch, sign of roll, which angle goes first, transpose).
CONVENTIONS = [c for c in itertools.product((0, 1), (0, 1), (1, -1), (1, -1), (0, 1), (0, 1)) if c[0] != c[1]]


def up_from_angles(angles: torch.Tensor, convention: Sequence[int]) -> torch.Tensor:
    """(pitch, roll) [B, 2] degrees -> where the canonical up (0, 0, 1) appears in the frame, [B, 3]."""
    ax1, ax2, sp, sr, order, transpose = convention
    first, second = (sp * angles[:, 0], sr * angles[:, 1]) if order == 0 else (sr * angles[:, 1], sp * angles[:, 0])
    rot = _axis_rot(ax1, first) @ _axis_rot(ax2, second)
    if transpose:
        rot = rot.transpose(-1, -2)
    return rot[..., :, 2]


# Fitted on 96 tilted flowscape:train frames (2026-11-15): median up-vector error 2.06 deg, the
# same convention as on test frames.
DEFAULT_CONVENTION = (0, 1, -1, -1, 1, 1)


@torch.no_grad()
def estimate_rotation(model: DPFAngles, frame_hwc01: torch.Tensor, convention=DEFAULT_CONVENTION) -> torch.Tensor:
    """Tilt-only rotation of a frame (heading kept), in the convention of spherical_flow.canonizer."""
    from .canonizer import rotation_from_vectors

    device = next(model.parameters()).device
    angles = model(prepare(frame_hwc01)[None].to(device)).float().cpu()
    return rotation_from_vectors(up_from_angles(angles, convention)[0], None)


def fit_convention(angles: torch.Tensor, up_true: torch.Tensor) -> Tuple[tuple, float]:
    """Pick the convention with the lowest median up-vector error (degrees) on known rotations."""
    best = (None, math.inf)
    for conv in CONVENTIONS:
        err = torch.rad2deg(torch.acos((up_from_angles(angles, conv) * up_true).sum(-1).clamp(-1, 1))).median().item()
        if err < best[1]:
            best = (conv, err)
    return best
