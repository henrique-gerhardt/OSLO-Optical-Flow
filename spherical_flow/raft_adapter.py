from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from .flow360 import Flow360Pair, bilinear_sample_erp
from .geometry import equirectangular_pixels_to_unit_vectors, logmap


FLOW_TRANSFORMS = (
    "identity",
    "negated",
    "negate_x",
    "negate_y",
    "swap_xy",
    "swap_xy_negated",
    "swap_xy_negate_x",
    "swap_xy_negate_y",
)


def require_divisible_by_8(height: int, width: int) -> None:
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            "TorchVision RAFT requires frame height and width divisible by 8. "
            f"Got width={width}, height={height}. This v1 baseline does not resize frames "
            "because flow rescaling would change the spherical evaluation."
        )


def load_chw_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_frame_batch(pairs: Iterable[Flow360Pair]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    frames1: list[torch.Tensor] = []
    frames2: list[torch.Tensor] = []
    size: Optional[tuple[int, int]] = None

    for pair in pairs:
        frame1 = load_chw_uint8(pair.frame1)
        frame2 = load_chw_uint8(pair.frame2)
        if frame1.shape != frame2.shape:
            raise ValueError(f"Frame size mismatch: {pair.frame1} and {pair.frame2}")
        height, width = int(frame1.shape[-2]), int(frame1.shape[-1])
        require_divisible_by_8(height, width)
        if size is None:
            size = (height, width)
        elif size != (height, width):
            raise ValueError("All frames in a RAFT batch must share the same height and width.")
        frames1.append(frame1)
        frames2.append(frame2)

    if size is None:
        raise ValueError("Cannot build an empty RAFT batch")
    height, width = size
    return torch.stack(frames1, dim=0), torch.stack(frames2, dim=0), height, width


def load_raft_model(model_name: str, weights_name: str, device: torch.device):
    import torchvision
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if model_name == "raft_large":
        weights_enum = Raft_Large_Weights.DEFAULT if weights_name == "default" else None
        model = raft_large(weights=weights_enum).to(device)
    elif model_name == "raft_small":
        weights_enum = Raft_Small_Weights.DEFAULT if weights_name == "default" else None
        model = raft_small(weights=weights_enum).to(device)
    else:
        raise ValueError(f"Unsupported RAFT model: {model_name}")

    transforms = weights_enum.transforms() if weights_enum is not None else None
    model.eval()
    resolved_weights = weights_enum.name if weights_enum is not None else "none"
    return model, transforms, resolved_weights, torchvision.__version__


def normalize_untrained_inputs(frame1: torch.Tensor, frame2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return frame1.float().div(127.5).sub(1.0), frame2.float().div(127.5).sub(1.0)


def transform_flow(flow: torch.Tensor, name: str) -> torch.Tensor:
    x = flow[:, 0:1]
    y = flow[:, 1:2]
    if name == "identity":
        return torch.cat([x, y], dim=1)
    if name == "negated":
        return torch.cat([-x, -y], dim=1)
    if name == "negate_x":
        return torch.cat([-x, y], dim=1)
    if name == "negate_y":
        return torch.cat([x, -y], dim=1)
    if name == "swap_xy":
        return torch.cat([y, x], dim=1)
    if name == "swap_xy_negated":
        return torch.cat([-y, -x], dim=1)
    if name == "swap_xy_negate_x":
        return torch.cat([-y, x], dim=1)
    if name == "swap_xy_negate_y":
        return torch.cat([y, -x], dim=1)
    raise ValueError(f"Unsupported flow transform: {name}")


@torch.no_grad()
def predict_princeton_flow(
    model: torch.nn.Module,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
    flow_transform: str,
    iters: int,
    infer_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Run a princeton-tree RAFT checkpoint; returns BCHW px flow at the input size.

    ``infer_size`` runs the network at a different resolution (e.g. SLOF's native
    320x640) with the flow interpolated back and its components rescaled by the
    size ratio — the same convention the SLOF loader uses in reverse.
    """
    frame1 = frame1.to(device, non_blocking=True).float()
    frame2 = frame2.to(device, non_blocking=True).float()
    orig_h, orig_w = int(frame1.shape[-2]), int(frame1.shape[-1])
    if infer_size is not None and infer_size != (orig_h, orig_w):
        frame1 = torch.nn.functional.interpolate(
            frame1, size=infer_size, mode="bilinear", align_corners=False, antialias=True)
        frame2 = torch.nn.functional.interpolate(
            frame2, size=infer_size, mode="bilinear", align_corners=False, antialias=True)
    flow = model(frame1, frame2, iters=iters)
    if infer_size is not None and infer_size != (orig_h, orig_w):
        flow = torch.nn.functional.interpolate(
            flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        flow = torch.cat([
            flow[:, 0:1] * (orig_w / infer_size[1]),
            flow[:, 1:2] * (orig_h / infer_size[0]),
        ], dim=1)
    return transform_flow(flow.detach().float().cpu(), flow_transform)


@torch.no_grad()
def predict_raft_flow(
    model: torch.nn.Module,
    transforms,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
    flow_transform: str,
) -> torch.Tensor:
    frame1 = frame1.to(device, non_blocking=True)
    frame2 = frame2.to(device, non_blocking=True)
    if transforms is not None:
        frame1, frame2 = transforms(frame1, frame2)
    else:
        frame1, frame2 = normalize_untrained_inputs(frame1, frame2)
    predictions = model(frame1, frame2)
    return transform_flow(predictions[-1].detach().float().cpu(), flow_transform)


def erp_flow_to_tangent(
    flow_erp: torch.Tensor,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    sampled_flow = bilinear_sample_erp(flow_erp, u, v)
    endpoint_u = u + sampled_flow[:, 0]
    endpoint_v = v + sampled_flow[:, 1]
    endpoint = equirectangular_pixels_to_unit_vectors(endpoint_u, endpoint_v, height, width)
    return logmap(points, endpoint, basis_east, basis_north).squeeze(0)


def flow_cache_path(
    cache_dir: str | Path,
    split: str,
    sequence: str,
    direction: str,
    frame: str,
    resolution: int,
) -> Path:
    return Path(cache_dir) / split / sequence / direction / f"{frame}_r{resolution}.npz"
