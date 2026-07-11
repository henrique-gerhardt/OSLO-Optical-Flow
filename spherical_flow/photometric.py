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
