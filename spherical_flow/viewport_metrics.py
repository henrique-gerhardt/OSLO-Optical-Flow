"""Viewport EPE: endpoint error measured in rectilinear (gnomonic) viewports.

Protocol of Cheng et al., arXiv 2610.32561, Sec. IV-A, carried over from image
quality to flow: the reference and every prediction are read out along the same
viewing rays, each from its own native format (ERP raster for planar methods,
HEALPix nodes for OSLO), so each method goes through exactly one mapping.

For every viewport pixel ray d:
  - GT endpoint: the GT ERP flow at the nearest ERP pixel, added to the ray's
    fractional ERP position, mapped back to the sphere;
  - ERP prediction: the same sampling of the predicted ERP flow;
  - node prediction: the node field interpolated at d (``nodes_to_endpoints``);
  - error: distance between the gnomonic projections of the two endpoints, in
    viewport pixels.

Rays whose GT endpoint leaves the viewport's front hemisphere by more than
``MIN_FORWARD`` are excluded for every method (GT-determined, so identical across
rows); a predicted endpoint behind that limit is clamped to it (finite, large
penalty) and counted.

Layouts:
  cube   six 90x90 deg faces, W/4 px square: partitions the sphere, equator = the 4
         side faces, poles = top/bottom faces (|lat| > 45 deg at the face centers)
  paper  the protocol of 2610.32561: four 150x60 deg viewports at yaw 0/90/180/270,
         pitch 0, and two at pitch +-90; W/3 x W/6 px (512x256 at 1536-wide ERP)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

MIN_FORWARD = 0.1  # cos(84.3 deg): endpoints further off-axis are behind the image plane

# name -> (fov_x deg, fov_y deg, width as a fraction of ERP width, height fraction,
#          [(group, yaw deg, pitch deg), ...])
LAYOUTS = {
    "cube": (90.0, 90.0, 1 / 4, 1 / 4, [
        ("equator", 0.0, 0.0), ("equator", 90.0, 0.0), ("equator", 180.0, 0.0),
        ("equator", 270.0, 0.0), ("poles", 0.0, 90.0), ("poles", 0.0, -90.0),
    ]),
    "paper": (150.0, 60.0, 1 / 3, 1 / 6, [
        ("equator", 0.0, 0.0), ("equator", 90.0, 0.0), ("equator", 180.0, 0.0),
        ("equator", 270.0, 0.0), ("poles", 0.0, 90.0), ("poles", 0.0, -90.0),
    ]),
}


def _camera(yaw_deg: float, pitch_deg: float) -> torch.Tensor:
    """Rows = (right, up, forward) in world coordinates (z up, lon = atan2(y, x))."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = torch.tensor([math.cos(pitch) * math.cos(yaw),
                            math.cos(pitch) * math.sin(yaw), math.sin(pitch)], dtype=torch.float64)
    right = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0], dtype=torch.float64)
    up = torch.linalg.cross(forward, right)
    return torch.stack([right, up, forward])


def erp_uv_to_dirs(u: torch.Tensor, v: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """ERP pixel-center coordinates -> unit vectors, fp64, lon wrapped, lat clamped.

    Same convention as ``geometry.points_to_equirectangular_pixels`` (inverse).
    """
    lon = (u + 0.5) / width * (2.0 * math.pi) - math.pi
    lat = (math.pi / 2.0 - (v + 0.5) / height * math.pi).clamp(-math.pi / 2, math.pi / 2)
    c = torch.cos(lat)
    return torch.stack([c * torch.cos(lon), c * torch.sin(lon), torch.sin(lat)], dim=-1)


def dirs_to_erp_uv(dirs: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y, z = dirs.unbind(-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z.clamp(-1.0, 1.0))
    u = (lon + math.pi) / (2.0 * math.pi) * width - 0.5
    v = (math.pi / 2.0 - lat) / math.pi * height - 0.5
    return u, v


def build_viewports(layout: str, height: int, width: int) -> dict:
    """Ray table for every viewport pixel of ``layout`` on an HxW ERP raster."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown viewport layout {layout!r}; choose from {sorted(LAYOUTS)}")
    fov_x, fov_y, fw, fh, views = LAYOUTS[layout]
    vw, vh = max(8, round(width * fw)), max(8, round(height * 2 * fh))
    fx = (vw / 2) / math.tan(math.radians(fov_x) / 2)
    fy = (vh / 2) / math.tan(math.radians(fov_y) / 2)
    jj, ii = torch.meshgrid(torch.arange(vw, dtype=torch.float64),
                            torch.arange(vh, dtype=torch.float64), indexing="xy")
    xn = ((jj + 0.5) - vw / 2) / fx
    yn = (vh / 2 - (ii + 0.5)) / fy
    local = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1).reshape(-1, 3)
    local = local / local.norm(dim=-1, keepdim=True)

    dirs, cams, view_ids, groups = [], [], [], []
    for k, (group, yaw, pitch) in enumerate(views):
        cam = _camera(yaw, pitch)
        dirs.append(local @ cam)  # world = local . rows
        cams.append(cam)
        view_ids.append(torch.full((local.shape[0],), k, dtype=torch.long))
        groups.append(group)
    dirs = torch.cat(dirs)
    u, v = dirs_to_erp_uv(dirs, height, width)
    nearest = (torch.remainder(torch.round(u), width).long()
               + width * torch.round(v).clamp(0, height - 1).long())
    return {
        "layout": layout, "dirs": dirs, "u": u, "v": v, "nearest": nearest,
        "view": torch.cat(view_ids), "cams": torch.stack(cams), "groups": groups,
        "fx": fx, "fy": fy, "size": (vh, vw),
    }


def erp_flow_endpoints(flow_erp: torch.Tensor, vp: dict) -> torch.Tensor:
    """ERP pixel flow [H, W, 2] -> endpoints [M, 3] at the viewport rays."""
    height, width = flow_erp.shape[:2]
    f = flow_erp.reshape(-1, 2).double()[vp["nearest"]]
    return erp_uv_to_dirs(vp["u"] + f[:, 0], vp["v"] + f[:, 1], height, width)


def project(endpoints: torch.Tensor, vp: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Endpoints [M, 3] -> (gnomonic pixel coords [M, 2], forward component [M])."""
    cam = vp["cams"][vp["view"]]  # [M, 3, 3]
    c = torch.einsum("mij,mj->mi", cam, endpoints.double())
    return c, c[:, 2]


def viewport_epe(pred_end: torch.Tensor, gt_end: torch.Tensor, vp: dict) -> Dict[str, torch.Tensor]:
    """Per-ray viewport EPE of a prediction and of zero flow, plus masks."""
    c_gt, fwd_gt = project(gt_end, vp)
    c_pr, fwd_pr = project(pred_end, vp)
    valid = fwd_gt > MIN_FORWARD
    clamped = valid & (fwd_pr <= MIN_FORWARD)
    fwd_pr = fwd_pr.clamp_min(MIN_FORWARD)
    scale = torch.tensor([vp["fx"], vp["fy"]], dtype=torch.float64)
    p_gt = c_gt[:, :2] / fwd_gt.clamp_min(MIN_FORWARD).unsqueeze(-1) * scale
    p_pr = c_pr[:, :2] / fwd_pr.unsqueeze(-1) * scale
    c_src, fwd_src = project(vp["dirs"], vp)
    p_src = c_src[:, :2] / fwd_src.unsqueeze(-1) * scale
    return {
        "epe": (p_pr - p_gt).norm(dim=-1),
        "zero": (p_src - p_gt).norm(dim=-1),
        "valid": valid,
        "clamped": clamped,
    }


def accumulate_viewport(maps: Dict[str, torch.Tensor], vp: dict,
                        totals: Dict[str, float], counts: Dict[str, float]) -> None:
    group_ids = torch.tensor([0 if g == "equator" else 1 for g in vp["groups"]])[vp["view"]]
    masks = {"all": maps["valid"],
             "equator": maps["valid"] & (group_ids == 0),
             "poles": maps["valid"] & (group_ids == 1)}
    for k in range(len(vp["groups"])):
        masks[f"view{k}"] = maps["valid"] & (vp["view"] == k)
    for name, mask in masks.items():
        for kind in ("epe", "zero"):
            key = f"{name}_{kind}"
            totals[key] = totals.get(key, 0.0) + maps[kind][mask].sum().item()
            counts[key] = counts.get(key, 0.0) + mask.sum().item()
    totals["excluded"] = totals.get("excluded", 0.0) + (~maps["valid"]).sum().item()
    totals["clamped"] = totals.get("clamped", 0.0) + maps["clamped"].sum().item()
    totals["rays"] = totals.get("rays", 0.0) + maps["valid"].numel()


def finalize_viewport(totals: Dict[str, float], counts: Dict[str, float], layout: str) -> Dict[str, float]:
    out = {}
    for key, total in totals.items():
        if key in ("excluded", "clamped", "rays"):
            continue
        if counts.get(key, 0) > 0:
            out[f"viewport_{layout}_{key}_px"] = total / counts[key]
    rays = max(totals.get("rays", 0.0), 1.0)
    out[f"viewport_{layout}_excluded_fraction"] = totals.get("excluded", 0.0) / rays
    out[f"viewport_{layout}_clamped_fraction"] = totals.get("clamped", 0.0) / rays
    return out
