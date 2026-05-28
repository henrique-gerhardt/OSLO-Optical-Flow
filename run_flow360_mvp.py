import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from spherical_flow import (
    SphericalFlowMVP,
    healpix_grid_struct,
    healpix_unit_vectors,
    logmap,
    tangent_basis,
)
from spherical_flow.flow360 import Flow360Dataset
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_loss,
    compute_maps,
    finalize_metrics,
    parse_thresholds,
    print_metrics,
    summarize_maps,
    target_sample_from_maps,
)


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
    parser.add_argument(
        "--cost-num-hops",
        type=int,
        default=1,
        help="HEALPix neighborhood radius used only by the local cost volume; encoder convolutions stay 1-hop.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--max-flow-rad", type=float, default=1.2)
    parser.add_argument("--no-zero-init-flow-head", dest="zero_init_flow_head", action="store_false")
    parser.set_defaults(zero_init_flow_head=True)
    parser.add_argument(
        "--use-displacement-prior",
        action="store_true",
        help="Use candidate tangent offsets to build a soft flow prior, then predict residual + gate.",
    )
    parser.add_argument(
        "--cost-prior-temperature",
        type=float,
        default=0.05,
        help="Softmax temperature for converting local cost-volume scores into the displacement prior.",
    )
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Multiplier for FLOW360 pixel flow values.")
    parser.add_argument("--loss-min-target-deg", type=float, default=0.0)
    parser.add_argument("--loss-motion-weight", type=float, default=0.0)
    parser.add_argument("--loss-motion-ref-deg", type=float, default=1.0)
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument(
        "--target-quantile-max-samples",
        type=int,
        default=2_000_000,
        help="Maximum validation samples used for target motion percentiles; keeps r=6+ quantiles bounded.",
    )
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


def num_neighbors_for_hops(num_hops: int) -> int:
    if num_hops <= 0:
        raise ValueError("num_hops must be positive")
    return 8 * num_hops * (num_hops + 1) // 2


def build_cost_candidates(
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    cost_index: torch.Tensor,
    cost_valid_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_nodes, num_neighbors = cost_index.shape
    neighbor_points = points.index_select(0, cost_index.reshape(-1)).reshape(n_nodes, num_neighbors, 3)
    base = points.unsqueeze(1).expand(-1, num_neighbors, -1).reshape(-1, 3)
    endpoints = neighbor_points.reshape(-1, 3).unsqueeze(0)
    east = basis_east.unsqueeze(1).expand(-1, num_neighbors, -1).reshape(-1, 3)
    north = basis_north.unsqueeze(1).expand(-1, num_neighbors, -1).reshape(-1, 3)
    neighbor_offsets = logmap(base, endpoints, east, north).squeeze(0).reshape(n_nodes, num_neighbors, 2)
    neighbor_offsets = torch.where(cost_valid_index.unsqueeze(-1), neighbor_offsets, torch.zeros_like(neighbor_offsets))

    center_offsets = torch.zeros(n_nodes, 1, 2, dtype=points.dtype)
    center_valid = torch.ones(n_nodes, 1, dtype=torch.bool)
    offsets = torch.cat([center_offsets, neighbor_offsets], dim=1)
    valid = torch.cat([center_valid, cost_valid_index], dim=1)
    return offsets, valid


def build_healpix(args: argparse.Namespace, device: torch.device) -> tuple:
    grid_file = Path(args.grid_dir) / f"healpix_grid_resolution_{args.resolution}.npz"
    if not grid_file.is_file():
        raise FileNotFoundError(f"Missing HEALPix grid file: {grid_file}")

    index, weight, valid_index = healpix_grid_struct(str(grid_file), args.resolution, num_hops=1)
    if args.cost_num_hops == 1:
        cost_index = index
        cost_valid_index = valid_index
    else:
        cost_index, _, cost_valid_index = healpix_grid_struct(
            str(grid_file),
            args.resolution,
            num_hops=args.cost_num_hops,
        )
    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)
    cost_offsets, cost_candidate_valid = build_cost_candidates(
        points,
        basis_east,
        basis_north,
        cost_index,
        cost_valid_index,
    )
    return (
        index.to(device),
        weight.to(device),
        valid_index.to(device),
        cost_index.to(device),
        cost_valid_index.to(device),
        cost_offsets.to(device),
        cost_candidate_valid.to(device),
        points.to(device),
        basis_east.to(device),
        basis_north.to(device),
    )


@torch.no_grad()
def evaluate(
    model: SphericalFlowMVP,
    loader: DataLoader,
    index: torch.Tensor,
    weight: torch.Tensor,
    valid_index: torch.Tensor,
    cost_index: torch.Tensor,
    cost_valid_index: torch.Tensor,
    cost_offsets: torch.Tensor,
    cost_candidate_valid: torch.Tensor,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    region_masks: Dict[str, torch.Tensor],
    active_thresholds: list[float],
    device: torch.device,
    target_quantile_max_samples: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    active_counts: Dict[str, float] = {}
    target_chunks: list[torch.Tensor] = []
    sample_per_batch = None
    if target_quantile_max_samples > 0:
        sample_per_batch = max(1, target_quantile_max_samples // max(len(loader), 1))

    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(
            batch["frame1"],
            batch["frame2"],
            index,
            weight,
            valid_index,
            points,
            cost_index,
            cost_valid_index,
            cost_offsets,
            cost_candidate_valid,
        )
        maps = compute_maps(pred, batch, points, basis_east, basis_north)
        target_chunks.append(target_sample_from_maps(maps, sample_per_batch))
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts)

    return finalize_metrics(totals, counts, active_counts, target_chunks)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)

    (
        index,
        weight,
        valid_index,
        cost_index,
        cost_valid_index,
        cost_offsets,
        cost_candidate_valid,
        points,
        basis_east,
        basis_north,
    ) = build_healpix(args, device)
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
        zero_init_flow_head=args.zero_init_flow_head,
        cost_num_neighbors=num_neighbors_for_hops(args.cost_num_hops),
        use_displacement_prior=args.use_displacement_prior,
        cost_prior_temperature=args.cost_prior_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    batch = move_batch(next(iter(train_loader)), device)
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        pred, debug = model(
            batch["frame1"],
            batch["frame2"],
            index,
            weight,
            valid_index,
            points,
            cost_index,
            cost_valid_index,
            cost_offsets,
            cost_candidate_valid,
            return_debug=True,
        )
        loss, maps = compute_loss(
            pred,
            batch,
            points,
            basis_east,
            basis_north,
            args.loss_min_target_deg,
            args.loss_motion_weight,
            args.loss_motion_ref_deg,
        )
    smoke_metrics = summarize_maps(maps, region_masks, active_thresholds)
    print(
        "smoke "
        f"device={device} "
        f"amp={amp_enabled} "
        f"nodes={points.size(0)} "
        f"cost_num_hops={args.cost_num_hops} "
        f"displacement_prior={args.use_displacement_prior} "
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
            pred = model(
                batch["frame1"],
                batch["frame2"],
                index,
                weight,
                valid_index,
                points,
                cost_index,
                cost_valid_index,
                cost_offsets,
                cost_candidate_valid,
            )
            loss, maps = compute_loss(
                pred,
                batch,
                points,
                basis_east,
                basis_north,
                args.loss_min_target_deg,
                args.loss_motion_weight,
                args.loss_motion_ref_deg,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step == args.steps or step % args.log_every == 0:
            metrics = summarize_maps(maps, region_masks, active_thresholds)
            print(f"step={step:06d} loss_rad={float(loss.detach().cpu()):.6f}", flush=True)
            print_metrics("train", metrics)

    metrics = evaluate(
        model,
        val_loader,
        index,
        weight,
        valid_index,
        cost_index,
        cost_valid_index,
        cost_offsets,
        cost_candidate_valid,
        points,
        basis_east,
        basis_north,
        region_masks,
        active_thresholds,
        device,
        args.target_quantile_max_samples,
    )
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
