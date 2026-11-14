"""Canonizer: estimates the camera rotation of an ERP frame relative to the canonical
(upright, forward-facing) orientation of a benchmark, so the frame can be rotated back
before a flow model runs (docs/plans/ORIENTATION_ROBUSTNESS.md 4d-4e).

Convention (run_raft_shard_baseline.rotate_erp): the raster rotated by ``R`` shows at
direction ``d`` what the real raster holds at ``d @ R``. The canonical up ``u0`` and
forward ``f0`` therefore appear in the rotated raster at ``u0 @ R.T`` and ``f0 @ R.T``;
those two unit vectors are what the network regresses.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision

from spherical_flow.geometry import equirectangular_pixels_to_unit_vectors

INPUT_HW = (160, 320)
UP0 = torch.tensor([0.0, 0.0, 1.0])


def forward0() -> torch.Tensor:
    """Direction of the ERP image centre: the canonical heading."""
    h, w = INPUT_HW
    f = equirectangular_pixels_to_unit_vectors(torch.tensor([(w - 1) / 2.0]), torch.tensor([(h - 1) / 2.0]), h, w)[0]
    f = f - (f @ UP0) * UP0
    return f / f.norm()


def labels_from_rotation(rotation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Where the canonical up and forward appear in a raster rotated by ``rotation``."""
    return UP0 @ rotation.T, forward0() @ rotation.T


def rotation_from_vectors(up: torch.Tensor, fwd: torch.Tensor | None) -> torch.Tensor:
    """The rotation whose labels are (up, fwd). ``fwd=None`` keeps the heading as it is
    (tilt-only canonization: the smallest change of heading compatible with ``up``)."""
    up = up / up.norm()
    f0 = forward0().to(up)
    if fwd is None:
        fwd = f0
    fwd = fwd - (fwd @ up) * up
    if float(fwd.norm()) < 1e-6:  # heading parallel to up: any heading will do
        fwd = torch.linalg.cross(up, torch.tensor([0.0, 1.0, 0.0]).to(up))
    fwd = fwd / fwd.norm()
    u0 = UP0.to(up)
    basis0 = torch.stack([f0, torch.linalg.cross(u0, f0), u0])       # canonical, rows
    basis = torch.stack([fwd, torch.linalg.cross(up, fwd), up])      # observed, rows
    return basis.T @ basis0                                          # basis0 @ R.T == basis


class Canonizer(torch.nn.Module):
    """ResNet-18 on a 160x320 ERP frame -> unit up and forward vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 6)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # circular padding in longitude: the ERP has no left/right border
        x = torch.cat([x[..., -16:], x, x[..., :16]], dim=-1)
        out = self.net(x)
        return F.normalize(out[:, :3], dim=1), F.normalize(out[:, 3:], dim=1)


def prepare(frame_hwc01: torch.Tensor) -> torch.Tensor:
    """float [H, W, 3] in [0, 1] -> network input [3, 160, 320]."""
    x = frame_hwc01.permute(2, 0, 1)[None]
    if x.shape[-2:] != INPUT_HW:
        x = F.interpolate(x, size=INPUT_HW, mode="area")
    return x[0] - 0.5


@torch.no_grad()
def estimate_rotation(model: Canonizer, frame_hwc01: torch.Tensor, mode: str = "full") -> torch.Tensor:
    """Estimated rotation of a frame, on the CPU. mode: 'full' or 'up' (tilt only)."""
    device = next(model.parameters()).device
    up, fwd = model(prepare(frame_hwc01)[None].to(device))
    return rotation_from_vectors(up[0].float().cpu(), fwd[0].float().cpu() if mode == "full" else None)


def load_canonizer(path: str, device: torch.device) -> Canonizer:
    model = Canonizer()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.to(device).eval()
