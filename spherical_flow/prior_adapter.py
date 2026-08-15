"""Adapter to score PriOr-RAFT checkpoints through OSLO's spherical metric.

Wraps the vendored PriOr-RAFT network (``prior_vendor.prior_raft``, ICCV 2025) so
its ERP pixel flow feeds the same ``erp_flow_to_tangent`` path used for
SLOF/PanoFlow/OSLO. The dual-branch machinery — the orthogonal view, the
Dual-Cost Collaborative Lookup and the Ortho-Driven Distortion Compensation — all
live inside the network's own ``forward``, so unlike PanoFlow's CFE there is no
inference protocol to re-implement here: ``test_mode=True`` returns the primitive
view's full-resolution flow directly.

Runs at the frame's native resolution (pad to /8, unpad back), which is the repo's
own eval protocol. No resize and no rescale.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from .prior_vendor.prior_raft import PriOr_RAFT
from .prior_vendor.utils import my_cycle_sample, projection_prim_ortho
from .raft_adapter import transform_flow

# Below this the 1/8 feature map cannot host a 4-level correlation pyramid and the
# network returns all-NaN rather than raising. Measured: 64x128 is 100% non-finite,
# 256x512 is clean. Guarded explicitly so a bad crop can never look like a result.
MIN_SIDE = 128


class _Args:
    """Namespace with exactly the fields the network reads."""

    def __init__(self, dropout: float = 0.0, mixed_precision: bool = False,
                 batch_size: int = 1, stage: str = "flowscape"):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        self.stage = stage


class _InputPadder:
    """Pad H,W up to a multiple of 8, sintel-style (same as the PanoFlow path)."""

    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_prior_checkpoint(checkpoint_path: str, device: torch.device,
                          eval_iters: int = 12) -> Tuple[torch.nn.Module, dict]:
    """Build PriOr-RAFT and load a checkpoint (``module.`` prefixes stripped).

    Hard-fails on any missing key, so a silent architecture mismatch can never
    masquerade as a valid run. The upstream checkpoints are saved from a
    ``DataParallel`` wrapper, hence the prefix.
    """
    # The vendored projection helpers allocate their grids on an explicit device
    # (upstream hardcodes .cuda()); pin it before the first forward.
    projection_prim_ortho.set_device(device)
    my_cycle_sample.set_device(device)

    model = PriOr_RAFT(_Args())

    raw = torch.load(checkpoint_path, map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]

    def norm_key(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module."):]
        return key

    state = {norm_key(k): v for k, v in raw.items()}
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    missing = sorted(model_keys - set(filtered.keys()))
    unexpected = sorted(set(state.keys()) - model_keys)
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} does not cover the PriOr-RAFT module tree; "
            f"{len(missing)} missing, first: {missing[:5]}; "
            f"{len(unexpected)} unexpected, first: {unexpected[:5]}"
        )
    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    meta = {
        "arch": "prior_raft",
        "eval_iters": eval_iters,
        "params": sum(p.numel() for p in model.parameters()),
        "loaded_keys": len(filtered),
        "unexpected_keys": len(unexpected),
        "unexpected_key_sample": unexpected[:5],
    }
    return model, meta


@torch.no_grad()
def predict_prior_flow(
    model: torch.nn.Module,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
    flow_transform: str,
    eval_iters: int = 12,
) -> torch.Tensor:
    """Run PriOr-RAFT; returns BCHW ERP pixel flow at the input size.

    Frames are BCHW floats in the 0-255 range: the network normalizes internally,
    exactly as its own ``evaluate.py`` feeds it.
    """
    image1 = frame1.to(device, non_blocking=True).float()
    image2 = frame2.to(device, non_blocking=True).float()
    if min(image1.shape[-2:]) < MIN_SIDE:
        raise ValueError(
            f"PriOr-RAFT needs both sides >= {MIN_SIDE} px (got {tuple(image1.shape[-2:])}); "
            "below that the correlation pyramid degenerates and the net returns NaN"
        )

    padder = _InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow = model(image1, image2, iters=eval_iters, test_mode=True)
    flow = padder.unpad(flow)

    if not torch.isfinite(flow).all():
        raise ValueError("PriOr-RAFT returned non-finite flow")
    return transform_flow(flow.detach().float().cpu(), flow_transform)
