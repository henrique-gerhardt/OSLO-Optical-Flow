import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from spherical_flow import (
    SphericalFlowMVP,
    endpoint_from_tangent_flow,
    geodesic_distance,
    healpix_grid_struct,
    healpix_unit_vectors,
    tangent_basis,
)
from spherical_flow.flow360 import Flow360Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the OSLO-style spherical flow MVP on FLOW360."
    )
    parser.add_argument("--data-root", default="/data/flow360", help="FLOW360 root with train/test folders.")
    parser.add_argument("--grid-dir", default="/data/oslo_data/neighbor_grids", help="OSLO HEALPix neighbor grids.")
    parser.add_argument("--output-dir", default="/outputs", help="Directory for checkpoints and metrics.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--resolution", type=int, default=5, help="HEALPix order; r=5 has 12,288 nodes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--max-flow-rad", type=float, default=1.2)
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Multiplier for FLOW360 pixel flow values.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true", help="Run one batch forward/loss pass and exit.")
    parser.add_argument("--checkpoint-out", default="", help="Optional checkpoint path.")
    parser.add_argument("--metrics-out", default="", help="Optional JSON metrics path.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def build_healpix(args: argparse.Namespace, device: torch.device) -> tuple:
    grid_file = Path(args.grid_dir) / f"healpix_grid_resolution_{args.resolution}.npz"
    if not grid_file.is_file():
        raise FileNotFoundError(f"Missing HEALPix grid file: {grid_file}")

    index, weight, valid_index = healpix_grid_struct(str(grid_file), args.resolution, num_hops=1)
    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)
    return (
        index.to(device),
        weight.to(device),
        valid_index.to(device),
        points.to(device),
        basis_east.to(device),
        basis_north.to(device),
    )


def build_region_masks(points: torch.Tensor, seam_width_deg: float = 15.0) -> Dict[str, torch.Tensor]:
    x, y, z = points.unbind(dim=-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z.clamp(-1.0, 1.0))
    seam_width = torch.tensor(np.deg2rad(seam_width_deg), dtype=points.dtype, device=points.device)
    return {
        "global": torch.ones(points.size(0), dtype=torch.bool, device=points.device),
        "poles": lat.abs() >= np.deg2rad(60.0),
        "equator": lat.abs() <= np.deg2rad(30.0),
        "seam": (torch.pi - lon.abs()) <= seam_width,
    }


def compute_maps(
    pred_flow: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_endpoint = endpoint_from_tangent_flow(points, pred_flow, basis_east, basis_north)
    target_endpoint = batch["endpoint"]
    geo = geodesic_distance(pred_endpoint, target_endpoint)
    tangent_epe = (pred_flow - batch["flow"]).norm(dim=-1)
    zero_endpoint = points.unsqueeze(0).expand_as(target_endpoint)
    zero_geo = geodesic_distance(zero_endpoint, target_endpoint)
    valid = batch.get("valid")
    if valid is None:
        valid = torch.ones_like(geo, dtype=torch.bool)
    return {
        "geo_rad": geo,
        "geo_deg": geo * (180.0 / torch.pi),
        "tangent_epe_rad": tangent_epe,
        "zero_geo_deg": zero_geo * (180.0 / torch.pi),
        "valid": valid.bool(),
    }


def compute_loss(
    pred_flow: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    maps = compute_maps(pred_flow, batch, points, basis_east, basis_north)
    loss = masked_mean(maps["geo_rad"], maps["valid"])
    return loss, maps


def summarize_maps(maps: Dict[str, torch.Tensor], region_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    valid = maps["valid"]
    for region_name, region_mask in region_masks.items():
        mask = valid & region_mask.unsqueeze(0)
        count = int(mask.sum().item())
        if count == 0:
            continue
        prefix = f"{region_name}_"
        out[prefix + "count"] = float(count)
        out[prefix + "geo_deg"] = float(masked_mean(maps["geo_deg"], mask).detach().cpu())
        out[prefix + "zero_geo_deg"] = float(masked_mean(maps["zero_geo_deg"], mask).detach().cpu())
        out[prefix + "tangent_epe_rad"] = float(masked_mean(maps["tangent_epe_rad"], mask).detach().cpu())
    return out


@torch.no_grad()
def evaluate(
    model: SphericalFlowMVP,
    loader: DataLoader,
    index: torch.Tensor,
    weight: torch.Tensor,
    valid_index: torch.Tensor,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    region_masks: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points)
        maps = compute_maps(pred, batch, points, basis_east, basis_north)
        summary = summarize_maps(maps, region_masks)
        for key, value in summary.items():
            if key.endswith("_count"):
                continue
            region = next((name for name in region_masks if key.startswith(f"{name}_")), "")
            if not region:
                continue
            count = summary.get(f"{region}_count", 0.0)
            totals[key] = totals.get(key, 0.0) + value * count
            counts[key] = counts.get(key, 0.0) + count

    return {key: totals[key] / max(counts[key], 1.0) for key in sorted(totals)}


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    keys = [
        "global_geo_deg",
        "global_zero_geo_deg",
        "poles_geo_deg",
        "poles_zero_geo_deg",
        "equator_geo_deg",
        "equator_zero_geo_deg",
        "seam_geo_deg",
        "seam_zero_geo_deg",
    ]
    items = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    print(f"{prefix} " + " ".join(items), flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    index, weight, valid_index, points, basis_east, basis_north = build_healpix(args, device)
    region_masks = build_region_masks(points)

    train_dataset = Flow360Dataset(
        args.data_root,
        split=args.train_split,
        points=points.detach().cpu(),
        direction=args.direction,
        flow_scale=args.flow_scale,
        max_sequences=args.max_train_sequences,
        max_pairs=args.max_train_pairs,
    )
    val_dataset = Flow360Dataset(
        args.data_root,
        split=args.val_split,
        points=points.detach().cpu(),
        direction=args.direction,
        flow_scale=args.flow_scale,
        max_sequences=args.max_val_sequences,
        max_pairs=args.max_val_pairs,
    )
    print(f"train_dataset={train_dataset.describe()}", flush=True)
    print(f"val_dataset={val_dataset.describe()}", flush=True)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = SphericalFlowMVP(
        hidden_channels=args.hidden_channels,
        feature_channels=args.feature_channels,
        max_flow_rad=args.max_flow_rad,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    batch = move_batch(next(iter(train_loader)), device)
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        pred, debug = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points, return_debug=True)
        loss, maps = compute_loss(pred, batch, points, basis_east, basis_north)
    smoke_metrics = summarize_maps(maps, region_masks)
    print(
        "smoke "
        f"device={device} "
        f"amp={amp_enabled} "
        f"nodes={points.size(0)} "
        f"pred_shape={tuple(pred.shape)} "
        f"cost_shape={tuple(debug['cost'].shape)} "
        f"loss_rad={float(loss.detach().cpu()):.6f}",
        flush=True,
    )
    print_metrics("smoke_metrics", smoke_metrics)
    if args.smoke_test:
        return

    start = time.time()
    model.train()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points)
            loss, maps = compute_loss(pred, batch, points, basis_east, basis_north)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step == args.steps or step % args.log_every == 0:
            metrics = summarize_maps(maps, region_masks)
            print(f"step={step:06d} loss_rad={float(loss.detach().cpu()):.6f}", flush=True)
            print_metrics("train", metrics)

    metrics = evaluate(model, val_loader, index, weight, valid_index, points, basis_east, basis_north, region_masks, device)
    metrics["elapsed_s"] = time.time() - start
    print_metrics("validation", metrics)
    print(f"elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    checkpoint_out = args.checkpoint_out or str(Path(args.output_dir) / "flow360_mvp.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, checkpoint_out)
    print(f"saved_checkpoint={checkpoint_out}", flush=True)

    metrics_out = args.metrics_out or str(Path(args.output_dir) / "flow360_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metrics": metrics}, handle, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
