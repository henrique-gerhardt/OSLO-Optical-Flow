"""Adapter to score PanoFlow(CSFlow) checkpoints through OSLO's spherical metric.

Wraps the vendored PanoFlow(CSFlow) network (``panoflow_vendor.panoflow_csflow``)
with a faithful re-implementation of the repo's CFE (Cyclic Flow Estimation)
inference path — a verbatim port of ``opticalflow/api/evaluate.py``'s
``validate_flow360_cfe`` split/merge (MasterHow/PanoFlow, MIT) — so the returned
ERP pixel flow feeds the same ``erp_flow_to_tangent`` path used for SLOF/OSLO.
Runs at the frame's native resolution (InputPadder pads to /8, unpads back), which
is PanoFlow's own eval protocol; no resize/rescale.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from .panoflow_vendor.panoflow_csflow import PanoCSFlow
from .raft_adapter import transform_flow


class _DotDict(dict):
    """dict with attribute access AND ``in`` membership.

    PanoFlow's net treats its ``args`` as a container (``'dcn' not in self.args``)
    and also as a namespace (``self.args.dcn = True``); a plain argparse Namespace
    supports neither pattern together, so mirror easydict semantics.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _InputPadder:
    """Pad H,W to a multiple of 8 (sintel mode), verbatim from PanoFlow utils."""

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


def _default_panoflow_args(eval_iters: int) -> _DotDict:
    # Only the fields the net reads at build/forward time; the net sets
    # corr_levels/corr_radius itself and fills dcn/dropout/alternate_corr/
    # mixed_precision if absent (we pin them to the eval defaults).
    return _DotDict(
        iters=20,
        eval_iters=eval_iters,
        train=True,        # only consulted for KITTI/Sintel init; Flow360 -> standard grid
        dataset="Flow360",
        dcn=True,
        dropout=0,
        alternate_corr=False,
        mixed_precision=False,
        small=False,
    )


def load_panoflow_checkpoint(checkpoint_path: str, device: torch.device, eval_iters: int = 12):
    """Build PanoFlow(CSFlow) and load a checkpoint (module./_model. prefixes stripped).

    Hard-fails on any missing/unexpected key so a silent architecture mismatch can
    never masquerade as a valid run (mirrors ``load_princeton_checkpoint``).
    """
    args = _default_panoflow_args(eval_iters)
    model = PanoCSFlow(args)

    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    def norm_key(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module."):]
        while key.startswith("_model."):
            key = key[len("_model."):]
        return key

    state = {norm_key(k): v for k, v in raw.items()}
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    missing = sorted(model_keys - set(filtered.keys()))
    unexpected = sorted(set(state.keys()) - model_keys)
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} does not cover the PanoFlow(CSFlow) module "
            f"tree; {len(missing)} missing, first: {missing[:5]}; "
            f"{len(unexpected)} unexpected, first: {unexpected[:5]}"
        )
    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    meta = {
        "arch": "panoflow_csflow",
        "eval_iters": eval_iters,
        "loaded_keys": len(filtered),
        "unexpected_keys": len(unexpected),
        "unexpected_key_sample": unexpected[:5],
    }
    return model, meta


@torch.no_grad()
def predict_panoflow_cfe_flow(
    model: torch.nn.Module,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
    flow_transform: str,
) -> torch.Tensor:
    """Run PanoFlow(CSFlow) under CFE; returns BCHW px flow at the input size.

    Port of ``validate_flow360_cfe`` (MasterHow/PanoFlow): encode both frames once
    (``gen_fmap``), split the feature maps at the ERP mid-meridian, decode the two
    cyclically-shifted halves (``skip_encode``), take the element-wise minimum of the
    two estimates per half, re-stitch and repair the two seam columns. Batched over
    the leading dim. Frames are CHW uint8-range floats (0-255); the net normalizes
    internally.
    """
    image1 = frame1.to(device, non_blocking=True).float()
    image2 = frame2.to(device, non_blocking=True).float()
    padder = _InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    image_pair = torch.stack((image1, image2))
    fmap1, fmap2, cnet1 = model(image_pair, test_mode=True, gen_fmap=True)

    half = fmap1.shape[3] // 2
    img_A1, img_B1 = fmap1[:, :, :, :half], fmap1[:, :, :, half:]
    img_A2, img_B2 = fmap2[:, :, :, :half], fmap2[:, :, :, half:]
    cnet_A1, cnet_B1 = cnet1[:, :, :, :half], cnet1[:, :, :, half:]

    img_pair_B1A1 = torch.stack((
        torch.cat([img_B1, img_A1], dim=3),
        torch.cat([img_B2, img_A2], dim=3),
        torch.cat([cnet_B1, cnet_A1], dim=3),
    ))
    img_pair_A1B1 = torch.stack((
        torch.cat([img_A1, img_B1], dim=3),
        torch.cat([img_A2, img_B2], dim=3),
        torch.cat([cnet_A1, cnet_B1], dim=3),
    ))

    _, flow_B1A1 = model(img_pair_B1A1, test_mode=True, skip_encode=True)
    _, flow_A1B1 = model(img_pair_A1B1, test_mode=True, skip_encode=True)

    wp = flow_B1A1.shape[3] // 2
    flow_A = torch.minimum(flow_B1A1[:, :, :, wp:], flow_A1B1[:, :, :, :wp])
    flow_B = torch.minimum(flow_B1A1[:, :, :, :wp], flow_A1B1[:, :, :, wp:])
    flow_pr = torch.cat([flow_A, flow_B], dim=3)

    mid = flow_pr.shape[3] // 2
    flow_pr[:, :, :, mid] = flow_pr[:, :, :, mid + 1]
    flow_pr[:, :, :, mid - 1] = flow_pr[:, :, :, mid - 2]

    flow = padder.unpad(flow_pr)
    return transform_flow(flow.detach().float().cpu(), flow_transform)
