from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import (
    equirectangular_pixels_to_unit_vectors,
    logmap,
    points_to_equirectangular_pixels,
    tangent_basis,
)


FlowDirection = Literal["forward", "backward", "both"]


@dataclass(frozen=True)
class Flow360Pair:
    sequence: str
    direction: str
    frame1: Path
    frame2: Path
    flow: Path


def _sorted_files(folder: Path, suffix: str) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def discover_flow360_pairs(
    root: str | Path,
    split: str,
    direction: FlowDirection = "forward",
    max_sequences: Optional[int] = None,
    max_pairs: Optional[int] = None,
) -> List[Flow360Pair]:
    """Discover adjacent-frame FLOW360 pairs from the SLOF folder layout."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"FLOW360 split not found: {split_dir}")

    pairs: List[Flow360Pair] = []
    sequences = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    for seq_dir in sequences:
        frames = _sorted_files(seq_dir / "frames", ".png")
        frame_by_stem = {p.stem: p for p in frames}
        stems = [p.stem for p in frames]

        if direction in ("forward", "both"):
            for idx, stem in enumerate(stems[:-1]):
                flow_path = seq_dir / "fflows" / f"{stem}.npy"
                if flow_path.is_file():
                    pairs.append(
                        Flow360Pair(
                            sequence=seq_dir.name,
                            direction="forward",
                            frame1=frame_by_stem[stem],
                            frame2=frame_by_stem[stems[idx + 1]],
                            flow=flow_path,
                        )
                    )

        if direction in ("backward", "both"):
            for idx, stem in enumerate(stems[1:], start=1):
                flow_path = seq_dir / "bflows" / f"{stem}.npy"
                if flow_path.is_file():
                    pairs.append(
                        Flow360Pair(
                            sequence=seq_dir.name,
                            direction="backward",
                            frame1=frame_by_stem[stem],
                            frame2=frame_by_stem[stems[idx - 1]],
                            flow=flow_path,
                        )
                    )

        if max_pairs is not None and len(pairs) >= max_pairs:
            return pairs[:max_pairs]

    if not pairs:
        raise RuntimeError(f"No FLOW360 pairs found in {split_dir} with direction={direction}")
    return pairs


def _load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _load_flow(path: Path, flow_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    flow = np.load(path)
    if flow.ndim != 3:
        raise ValueError(f"Expected flow with 3 dimensions, got shape={flow.shape} from {path}")
    if flow.shape[0] in (2, 3) and flow.shape[-1] not in (2, 3):
        flow = np.moveaxis(flow, 0, -1)
    if flow.shape[-1] < 2:
        raise ValueError(f"Expected flow last dimension >= 2, got shape={flow.shape} from {path}")

    flow_xy = torch.from_numpy(flow[..., :2].astype(np.float32)) * float(flow_scale)
    finite = torch.isfinite(flow_xy).all(dim=-1)
    flow_xy = torch.nan_to_num(flow_xy, nan=0.0, posinf=0.0, neginf=0.0)
    return flow_xy, finite


def _bilinear_sample_erp(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample HxWxC tensors at ERP coordinates with horizontal wrap."""
    if image.ndim != 3:
        raise ValueError("image must have shape [H, W, C]")
    height, width, _ = image.shape
    u_wrapped = torch.remainder(u, float(width))
    v_clamped = v.clamp(0.0, float(height - 1))

    u0f = torch.floor(u_wrapped)
    v0f = torch.floor(v_clamped)
    u0 = u0f.long()
    v0 = v0f.long()
    u1 = (u0 + 1) % width
    v1 = (v0 + 1).clamp(max=height - 1)

    du = (u_wrapped - u0f).unsqueeze(-1)
    dv = (v_clamped - v0f).unsqueeze(-1)

    top_left = image[v0, u0]
    top_right = image[v0, u1]
    bottom_left = image[v1, u0]
    bottom_right = image[v1, u1]
    top = top_left * (1.0 - du) + top_right * du
    bottom = bottom_left * (1.0 - du) + bottom_right * du
    return top * (1.0 - dv) + bottom * dv


class Flow360Dataset(Dataset):
    """FLOW360 pairs sampled onto HEALPix nodes for spherical optical flow."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        points: torch.Tensor,
        direction: FlowDirection = "forward",
        flow_scale: float = 1.0,
        max_sequences: Optional[int] = None,
        max_pairs: Optional[int] = None,
    ) -> None:
        if points.ndim != 2 or points.size(-1) != 3:
            raise ValueError("points must have shape [N, 3]")
        self.root = Path(root)
        self.split = split
        self.points = points.detach().cpu().float()
        self.direction = direction
        self.flow_scale = flow_scale
        self.pairs = discover_flow360_pairs(root, split, direction, max_sequences, max_pairs)
        self.basis_east, self.basis_north = tangent_basis(self.points)

    def __len__(self) -> int:
        return len(self.pairs)

    def describe(self) -> Dict[str, object]:
        first = self.pairs[0]
        image = Image.open(first.frame1)
        flow = np.load(first.flow)
        sequences = sorted({pair.sequence for pair in self.pairs})
        return {
            "split": self.split,
            "pairs": len(self.pairs),
            "sequences": len(sequences),
            "first_sequence": first.sequence,
            "image_size": image.size,
            "flow_shape": tuple(flow.shape),
            "direction": self.direction,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        pair = self.pairs[idx]
        frame1_erp = _load_rgb(pair.frame1)
        frame2_erp = _load_rgb(pair.frame2)
        flow_erp, finite_flow = _load_flow(pair.flow, self.flow_scale)

        height, width = frame1_erp.shape[:2]
        if frame2_erp.shape[:2] != (height, width):
            raise ValueError(f"Frame size mismatch: {pair.frame1} and {pair.frame2}")
        if flow_erp.shape[:2] != (height, width):
            raise ValueError(f"Flow size {tuple(flow_erp.shape[:2])} does not match frame size {(height, width)}")

        u, v = points_to_equirectangular_pixels(self.points, height, width)
        frame1 = _bilinear_sample_erp(frame1_erp, u, v)
        frame2 = _bilinear_sample_erp(frame2_erp, u, v)
        sampled_flow = _bilinear_sample_erp(flow_erp, u, v)
        sampled_finite = _bilinear_sample_erp(finite_flow.float().unsqueeze(-1), u, v).squeeze(-1) > 0.999

        endpoint_u = u + sampled_flow[:, 0]
        endpoint_v = v + sampled_flow[:, 1]
        inside_vertical = (endpoint_v >= 0.0) & (endpoint_v <= float(height - 1))
        valid = sampled_finite & inside_vertical

        endpoint = equirectangular_pixels_to_unit_vectors(endpoint_u, endpoint_v, height, width)
        flow = logmap(self.points, endpoint, self.basis_east, self.basis_north).squeeze(0)

        return {
            "frame1": frame1,
            "frame2": frame2,
            "flow": flow,
            "endpoint": endpoint,
            "valid": valid,
            "sequence": pair.sequence,
            "direction": pair.direction,
            "frame": pair.frame1.stem,
        }
