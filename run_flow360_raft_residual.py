import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spherical_flow import RaftResidualCorrector, healpix_grid_struct, healpix_unit_vectors, tangent_basis
from spherical_flow.flow360 import Flow360Dataset
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_maps,
    finalize_metrics,
    masked_mean,
    parse_thresholds,
    print_metrics,
    summarize_maps,
    target_sample_from_maps,
)
from spherical_flow.raft_adapter import flow_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a HEALPix residual corrector on top of cached RAFT flow.")
    parser.add_argument("--data-root", default="/data/flow360")
    parser.add_argument("--grid-dir", default="/data/oslo_data/neighbor_grids")
    parser.add_argument("--raft-cache-dir", default="/outputs/raft_cache")
    parser.add_argument("--output-dir", default="/outputs/raft_residual_r6")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--residual-max-rad", type=float, default=0.05)
    parser.add_argument("--residual-reg-weight", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument("--target-quantile-max-samples", type=int, default=2_000_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--metrics-out", default="")
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


def build_healpix(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


class Flow360CachedRaftDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        points: torch.Tensor,
        raft_cache_dir: str | Path,
        resolution: int,
        direction: str = "forward",
        max_pairs: int | None = None,
    ) -> None:
        self.base = Flow360Dataset(root, split=split, points=points, direction=direction, max_pairs=max_pairs)
        self.raft_cache_dir = Path(raft_cache_dir)
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.base)

    @property
    def pairs(self):
        return self.base.pairs

    def describe(self) -> Dict[str, object]:
        desc = self.base.describe()
        desc["raft_cache_dir"] = str(self.raft_cache_dir)
        return desc

    def cache_path_for_index(self, idx: int) -> Path:
        pair = self.base.pairs[idx]
        return flow_cache_path(
            self.raft_cache_dir,
            self.base.split,
            pair.sequence,
            pair.direction,
            pair.frame1.stem,
            self.resolution,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.base[idx]
        cache_path = self.cache_path_for_index(idx)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing RAFT cache file: {cache_path}")
        with np.load(cache_path) as data:
            raft_flow = torch.from_numpy(data["flow_tangent"].astype(np.float32, copy=False))
        if raft_flow.shape != item["flow"].shape:
            raise ValueError(
                f"RAFT cache shape {tuple(raft_flow.shape)} does not match target flow "
                f"{tuple(item['flow'].shape)} for {cache_path}"
            )
        item["raft_flow"] = raft_flow
        item["raft_cache"] = str(cache_path)
        return item


def residual_loss(
    pred_flow: torch.Tensor,
    residual: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    residual_reg_weight: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    maps = compute_maps(pred_flow, batch, points, basis_east, basis_north)
    valid = maps["valid"]
    geo_loss = masked_mean(maps["geo_rad"], valid)
    reg_loss = masked_mean(residual.norm(dim=-1), valid)
    loss = geo_loss + float(residual_reg_weight) * reg_loss
    return loss, {**maps, "geo_loss": geo_loss, "residual_reg_loss": reg_loss}


@torch.no_grad()
def evaluate_flow(
    loader: DataLoader,
    pred_fn,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    region_masks: Dict[str, torch.Tensor],
    active_thresholds: list[float],
    device: torch.device,
    target_quantile_max_samples: int,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    active_counts: Dict[str, float] = {}
    target_chunks: list[torch.Tensor] = []
    sample_per_batch = None
    if target_quantile_max_samples > 0:
        sample_per_batch = max(1, target_quantile_max_samples // max(len(loader), 1))

    for batch in loader:
        batch = move_batch(batch, device)
        pred_flow = pred_fn(batch)
        maps = compute_maps(pred_flow, batch, points, basis_east, basis_north)
        target_chunks.append(target_sample_from_maps(maps, sample_per_batch))
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts)

    return finalize_metrics(totals, counts, active_counts, target_chunks)


def vs_raft_metrics(residual_metrics: Dict[str, float], raft_metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in residual_metrics.items():
        if key not in raft_metrics or not isinstance(value, (float, int)):
            continue
        delta = value - raft_metrics[key]
        out[f"{key}_delta"] = delta
        if key.endswith("_geo_deg") and abs(raft_metrics[key]) > 1e-12:
            out[f"{key}_improvement_pct"] = 100.0 * (raft_metrics[key] - value) / raft_metrics[key]
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)

    index, weight, valid_index, points, basis_east, basis_north = build_healpix(args, device)
    region_masks = build_region_masks(points)
    train_dataset = Flow360CachedRaftDataset(
        args.data_root,
        split=args.train_split,
        points=points.detach().cpu(),
        raft_cache_dir=args.raft_cache_dir,
        resolution=args.resolution,
        direction=args.direction,
        max_pairs=args.max_train_pairs,
    )
    val_dataset = Flow360CachedRaftDataset(
        args.data_root,
        split=args.val_split,
        points=points.detach().cpu(),
        raft_cache_dir=args.raft_cache_dir,
        resolution=args.resolution,
        direction=args.direction,
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

    model = RaftResidualCorrector(
        hidden_channels=args.hidden_channels,
        residual_max_rad=args.residual_max_rad,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    first_batch = move_batch(next(iter(train_loader)), device)
    with torch.no_grad():
        pred0, residual0 = model(
            first_batch["frame1"],
            first_batch["frame2"],
            first_batch["raft_flow"],
            index,
            weight,
            valid_index,
            points,
            return_residual=True,
        )
        diff0 = (pred0 - first_batch["raft_flow"]).abs().max()
        maps0 = compute_maps(pred0, first_batch, points, basis_east, basis_north)
        smoke_metrics = summarize_maps(maps0, region_masks, active_thresholds)
    print(
        "smoke "
        f"device={device} amp={amp_enabled} nodes={points.size(0)} "
        f"residual_max_rad={args.residual_max_rad} "
        f"initial_residual_absmax={float(residual0.abs().max().detach().cpu()):.8f} "
        f"initial_raft_diff_absmax={float(diff0.detach().cpu()):.8f}",
        flush=True,
    )
    print_metrics("smoke_metrics", smoke_metrics)

    start_time = time.time()
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
            pred, residual = model(
                batch["frame1"],
                batch["frame2"],
                batch["raft_flow"],
                index,
                weight,
                valid_index,
                points,
                return_residual=True,
            )
            loss, maps = residual_loss(
                pred,
                residual,
                batch,
                points,
                basis_east,
                basis_north,
                args.residual_reg_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step == args.steps or step % args.log_every == 0:
            metrics = summarize_maps(maps, region_masks, active_thresholds)
            print(
                f"step={step:06d} loss={float(loss.detach().cpu()):.6f} "
                f"geo_loss={float(maps['geo_loss'].detach().cpu()):.6f} "
                f"residual_reg={float(maps['residual_reg_loss'].detach().cpu()):.6f}",
                flush=True,
            )
            print_metrics("train", metrics)

    model.eval()
    raft_metrics = evaluate_flow(
        val_loader,
        lambda batch: batch["raft_flow"],
        points,
        basis_east,
        basis_north,
        region_masks,
        active_thresholds,
        device,
        args.target_quantile_max_samples,
    )
    residual_metrics = evaluate_flow(
        val_loader,
        lambda batch: model(
            batch["frame1"],
            batch["frame2"],
            batch["raft_flow"],
            index,
            weight,
            valid_index,
            points,
        ),
        points,
        basis_east,
        basis_north,
        region_masks,
        active_thresholds,
        device,
        args.target_quantile_max_samples,
    )
    elapsed_s = time.time() - start_time
    residual_metrics["elapsed_s"] = elapsed_s
    deltas = vs_raft_metrics(residual_metrics, raft_metrics)
    print_metrics("raft_baseline", raft_metrics)
    print_metrics("residual_validation", residual_metrics)
    print(f"elapsed_s={elapsed_s:.1f}", flush=True)

    checkpoint_out = args.checkpoint_out or str(Path(args.output_dir) / "raft_residual.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "raft_metrics": raft_metrics, "metrics": residual_metrics}, checkpoint_out)
    print(f"saved_checkpoint={checkpoint_out}", flush=True)

    metrics_out = args.metrics_out or str(Path(args.output_dir) / "raft_residual_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "train_dataset": train_dataset.describe(),
                "val_dataset": val_dataset.describe(),
                "raft_metrics": raft_metrics,
                "residual_metrics": residual_metrics,
                "vs_raft": deltas,
            },
            handle,
            indent=2,
        )
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
