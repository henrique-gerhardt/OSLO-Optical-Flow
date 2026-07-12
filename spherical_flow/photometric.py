"""Photometric jitter for spherical frame samples (plan P2C §2 / probe P0).

RAFT's robustness to real inter-frame appearance change was manufactured by
asymmetric photometric augmentation at training scale; the decisive triangle
(chapter §7.2) showed that exactly this nuisance axis separates our +80.6%
(resampled, photometrically clean) from −32% (real frame 2). This module is that
axis, made injectable: RAFT-parity jitter ranges scaled by a single ``scale`` knob
in [0, 1].

Operates pointwise on RGB, so applying it to node-sampled frames [N, 3] is
mathematically identical to applying it to the ERP raster before sampling
(brightness/contrast/saturation/hue touch no neighborhoods). Contrast blends with
the mean luma over the sample — on HEALPix nodes that mean is solid-angle-fair by
construction. Hue is a rotation of the (I, Q) chroma plane in YIQ space (linear,
deterministic; magnitude-equivalent to torchvision's HSV shift for small angles).
Ops apply in fixed order brightness -> contrast -> saturation -> hue; RAFT
randomizes the order, but parity in magnitude is what the nuisance axis needs.

Full scale (1.0) matches the RAFT recipe: brightness/contrast/saturation 0.4,
hue 0.5/pi ~ 0.159 (fraction of the hue circle).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

BRIGHTNESS = 0.4
CONTRAST = 0.4
SATURATION = 0.4
HUE = 0.5 / math.pi

_LUMA = (0.299, 0.587, 0.114)


def sample_jitter_params(gen: torch.Generator, scale: float) -> Dict[str, float]:
    """Draw one record's jitter factors from RAFT-parity ranges scaled by ``scale``."""

    def _factor(halfwidth: float) -> float:
        lo, hi = 1.0 - halfwidth * scale, 1.0 + halfwidth * scale
        return lo + float(torch.rand((), generator=gen)) * (hi - lo)

    hue_half = HUE * scale
    return {
        "brightness": _factor(BRIGHTNESS),
        "contrast": _factor(CONTRAST),
        "saturation": _factor(SATURATION),
        "hue": -hue_half + float(torch.rand((), generator=gen)) * 2.0 * hue_half,
    }


def apply_jitter(frame: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Apply jitter to an RGB float frame in [0, 1] of shape [..., 3]."""
    if frame.size(-1) != 3:
        raise ValueError("frame must have RGB channels last")
    r, g, b = frame.unbind(dim=-1)
    luma = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b

    out = frame * params["brightness"]
    mean_luma = luma.mean() * params["brightness"]
    out = mean_luma + params["contrast"] * (out - mean_luma)
    gray = (_LUMA[0] * out[..., 0] + _LUMA[1] * out[..., 1] + _LUMA[2] * out[..., 2])
    out = gray.unsqueeze(-1) + params["saturation"] * (out - gray.unsqueeze(-1))

    if params["hue"] != 0.0:
        theta = 2.0 * math.pi * params["hue"]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        r, g, b = out.unbind(dim=-1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        i, q = cos_t * i - sin_t * q, sin_t * i + cos_t * q
        out = torch.stack(
            [
                y + 0.956 * i + 0.621 * q,
                y - 0.272 * i - 0.647 * q,
                y - 1.106 * i + 1.703 * q,
            ],
            dim=-1,
        )
    return out.clamp(0.0, 1.0)


def _blur_wrap(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [H, W, C] with ERP horizontal wrap, vertical replicate."""
    import torch.nn.functional as F

    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).to(x.dtype)
    v = x.permute(2, 0, 1).unsqueeze(1)  # [C,1,H,W]
    v = torch.cat([v[..., -radius:], v, v[..., :radius]], dim=-1)
    v = F.conv2d(v, k.view(1, 1, 1, -1))
    v = torch.cat([v[..., :1, :].expand(-1, -1, radius, -1), v,
                   v[..., -1:, :].expand(-1, -1, radius, -1)], dim=-2)
    v = F.conv2d(v, k.view(1, 1, -1, 1))
    return v.squeeze(1).permute(1, 2, 0)


def edge_corruption(
    frame_erp: torch.Tensor,
    gen: torch.Generator,
    mean_delta_255: float,
    sigma_px: float = 0.9,
    sigma_broad_px: float = 6.0,
    broad_weight: float = 0.45,
    sigma_gate_px: float = 14.0,
    gate_sharpness: float = 3.0,
    gate_bias: float = 0.8,
    flat_floor: float = 0.35,
    chroma: float = 0.45,
) -> torch.Tensor:
    """Corrupt an ERP raster [H, W, 3] with the *measured* real-nuisance structure.

    The appearance-residual diagnostic (`analyze_appearance_residual.py`,
    flow360:val) showed the real inter-frame nuisance is sparse and edge-anchored
    (top-decile-gradient pixels carry 33% of the mass at 17x flat-region
    amplitude), mesoscale-correlated (~3 px), luma-dominant (0.84), with mean
    3.1/255 and p99/p50 ~ 24. This op reproduces that shape:

        delta = (|grad luma|/mean + flat_floor) * (N_shared + chroma * N_rgb)

    with N_* Gaussian-blurred white noise (corr length from ``sigma_px``); the
    heavy tail is inherited from the image-gradient distribution. Scaled so
    E|delta| = ``mean_delta_255``/255 pre-clamp. Draw count per call is fixed, so
    sweeps over the magnitude share raw draws (nested, like the other levers).
    Acceptance is match-the-table: the diagnostic run on (frame, corrupt(frame))
    pairs must reproduce the measured statistics.
    """
    height, width = frame_erp.shape[:2]
    y = _LUMA[0] * frame_erp[..., 0] + _LUMA[1] * frame_erp[..., 1] + _LUMA[2] * frame_erp[..., 2]
    gx = torch.roll(y, -1, dims=1) - torch.roll(y, 1, dims=1)
    yp = torch.cat([y[:1], y, y[-1:]], dim=0)
    gy = yp[2:] - yp[:-2]
    edge = torch.sqrt(gx.square() + gy.square())

    noise = torch.randn(height, width, 5, generator=gen, dtype=frame_erp.dtype)
    # Stochastic edge gate: only patches of edges light up (the real residual's
    # edge-amplitude relation is strong, 17x, but partial — corr 0.46, mass 0.33:
    # specularity/AA hits some surfaces, not all). Broad-scale sigmoid gate.
    gate_field = _blur_wrap(noise[..., 4:5], sigma_gate_px)[..., 0]
    gate_z = gate_field / gate_field.std().clamp_min(1e-8)
    gate = torch.sigmoid(gate_sharpness * (gate_z - gate_bias))
    envelope = (edge / edge.mean().clamp_min(1e-8)) * 3.0 * gate + flat_floor
    # Two correlation scales: the fine one sets lag-1 coherence, the broad one the
    # measured lag-4 tail (real residual: 0.70 -> 0.22).
    fine = _blur_wrap(noise, sigma_px)
    broad = _blur_wrap(noise, sigma_broad_px)
    smooth = (fine / fine.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
              + broad_weight * broad / broad.std(dim=(0, 1), keepdim=True).clamp_min(1e-8))
    smooth = smooth / smooth.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
    field = envelope.unsqueeze(-1) * (smooth[..., :1] + chroma * smooth[..., 1:4])
    delta = field * (mean_delta_255 / 255.0) / field.abs().mean().clamp_min(1e-12)
    return (frame_erp + delta).clamp(0.0, 1.0)


def apply_noise(frame: torch.Tensor, gen: torch.Generator, std_255: float) -> torch.Tensor:
    """Add per-pixel iid Gaussian noise of std ``std_255``/255 to a [0, 1] RGB frame.

    The *spatially-unstructured* counterpart of :func:`apply_jitter` (which shifts
    the whole frame coherently): probe P0 showed global jitter at real-magnitude
    mean delta costs ~2 points while the real leg costs ~110, so the P0b sweep uses
    this knob to test whether high-frequency appearance noise alone reproduces the
    damage. Mean |delta| of the noise is std * sqrt(2/pi) ~ 0.80 * std.
    """
    noise = torch.randn(frame.shape, generator=gen, dtype=frame.dtype)
    return (frame + noise * (std_255 / 255.0)).clamp(0.0, 1.0)
