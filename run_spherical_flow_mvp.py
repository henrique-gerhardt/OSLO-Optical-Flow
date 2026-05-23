import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from spherical_flow import (
    SyntheticRotationFlowDatasetFromPoints,
    SyntheticRotationFlowDataset,
    SphericalFlowMVP,
    directional_knn_graph,
    endpoint_from_tangent_flow,
    fibonacci_unit_vectors,
    healpix_grid_struct,
    geodesic_distance,
    healpix_unit_vectors,
    tangent_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic spherical optical-flow MVP using OSLO SDPA convolutions."
    )
    parser.add_argument(
        "--grid",
        default="fibonacci",
        choices=["fibonacci", "healpix"],
        help="Use exact OSLO HEALPix structures or a dependency-light Fibonacci graph.",
    )
    parser.add_argument("--resolution", type=int, default=3, help="HEALPix order for the smoke experiment.")
    parser.add_argument("--num-nodes", type=int, default=192, help="Number of nodes for --grid fibonacci.")
    parser.add_argument("--train-length", type=int, default=64, help="Number of synthetic train samples.")
    parser.add_argument("--val-length", type=int, default=16, help="Number of synthetic validation samples.")
    parser.add_argument("--max-angle-deg", type=float, default=5.0, help="Maximum random rotation angle.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--max-flow-rad", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--struct-dir", default="", help="Optional cache directory for HEALPix structures.")
    parser.add_argument(
        "--grid-dir",
        default="../oslo_data/neighbor_grids",
        help="Directory with OSLO precomputed HEALPix neighbor-grid npz files.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/loss pass and exit.")
    parser.add_argument("--checkpoint-out", default="", help="Optional path to save model state.")
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
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def build_experiment(args: argparse.Namespace, device: torch.device) -> tuple:
    if args.grid == "healpix":
        try:
            from healpix_sdpa_struct_loader import HealpixSdpaStructLoader
        except ImportError:
            grid_file = os.path.join(args.grid_dir, f"healpix_grid_resolution_{args.resolution}.npz")
            if not os.path.isfile(grid_file):
                raise FileNotFoundError(
                    f"HEALPix fallback grid not found: {grid_file}. "
                    "Install healpy or provide --grid-dir with OSLO neighbor-grid files."
                )
            index, weight, valid_index = healpix_grid_struct(grid_file, args.resolution, num_hops=1)
            label = f"healpix_grid_order={args.resolution}"
        else:
            struct_loader = HealpixSdpaStructLoader(
                weight_type="identity",
                use_geodesic=True,
                use_4connectivity=False,
                normalization_method="non",
                cutGraphForPatchOutside=True,
                load_save_folder=args.struct_dir or None,
            )
            index, weight, valid_index = struct_loader.getStruct(args.resolution, num_hops=1)
            label = f"healpix_oslo_order={args.resolution}"
        points_cpu = healpix_unit_vectors(args.resolution)
        train_dataset = SyntheticRotationFlowDataset(
            args.resolution,
            length=args.train_length,
            max_angle_deg=args.max_angle_deg,
            seed=args.seed * 1000,
        )
        val_dataset = SyntheticRotationFlowDataset(
            args.resolution,
            length=args.val_length,
            max_angle_deg=args.max_angle_deg,
            seed=args.seed * 2000,
        )
    else:
        points_cpu = fibonacci_unit_vectors(args.num_nodes)
        index, weight, valid_index = directional_knn_graph(points_cpu, num_neighbors=8)
        train_dataset = SyntheticRotationFlowDatasetFromPoints(
            points_cpu,
            length=args.train_length,
            max_angle_deg=args.max_angle_deg,
            seed=args.seed * 1000,
        )
        val_dataset = SyntheticRotationFlowDatasetFromPoints(
            points_cpu,
            length=args.val_length,
            max_angle_deg=args.max_angle_deg,
            seed=args.seed * 2000,
        )
        label = f"fibonacci_nodes={args.num_nodes}"

    return (
        index.to(device),
        weight.to(device),
        valid_index.to(device),
        points_cpu.to(device),
        train_dataset,
        val_dataset,
        label,
    )


def compute_losses(
    pred_flow: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
) -> dict:
    pred_endpoint = endpoint_from_tangent_flow(points, pred_flow, basis_east, basis_north)
    target_endpoint = batch["endpoint"]
    geo = geodesic_distance(pred_endpoint, target_endpoint).mean()
    tangent_epe = (pred_flow - batch["flow"]).norm(dim=-1).mean()
    zero_endpoint = points.unsqueeze(0).expand_as(target_endpoint)
    zero_geo = geodesic_distance(zero_endpoint, target_endpoint).mean()
    return {
        "loss": geo,
        "geo_rad": geo.detach(),
        "geo_deg": geo.detach() * (180.0 / torch.pi),
        "tangent_epe_rad": tangent_epe.detach(),
        "zero_geo_deg": zero_geo.detach() * (180.0 / torch.pi),
    }


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
    device: torch.device,
) -> dict:
    model.eval()
    totals = {"geo_deg": 0.0, "tangent_epe_rad": 0.0, "zero_geo_deg": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points)
        losses = compute_losses(pred, batch, points, basis_east, basis_north)
        bsz = batch["frame1"].size(0)
        for key in totals:
            totals[key] += float(losses[key]) * bsz
        count += bsz
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    index, weight, valid_index, points, train_dataset, val_dataset, grid_label = build_experiment(args, device)
    basis_east, basis_north = tangent_basis(points)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SphericalFlowMVP(
        hidden_channels=args.hidden_channels,
        feature_channels=args.feature_channels,
        max_flow_rad=args.max_flow_rad,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch = move_batch(next(iter(train_loader)), device)
    pred, debug = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points, return_debug=True)
    losses = compute_losses(pred, batch, points, basis_east, basis_north)
    print(
        "smoke "
        f"grid={grid_label} "
        f"pred_shape={tuple(pred.shape)} "
        f"cost_shape={tuple(debug['cost'].shape)} "
        f"geo_deg={float(losses['geo_deg']):.4f} "
        f"zero_geo_deg={float(losses['zero_geo_deg']):.4f}",
        flush=True,
    )

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
        pred = model(batch["frame1"], batch["frame2"], index, weight, valid_index, points)
        losses = compute_losses(pred, batch, points, basis_east, basis_north)
        losses["loss"].backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(
                f"step={step:04d} "
                f"loss_rad={float(losses['loss'].detach()):.6f} "
                f"geo_deg={float(losses['geo_deg']):.4f} "
                f"zero_geo_deg={float(losses['zero_geo_deg']):.4f}",
                flush=True,
            )

    metrics = evaluate(model, val_loader, index, weight, valid_index, points, basis_east, basis_north, device)
    elapsed = time.time() - start
    print(
        "validation "
        f"geo_deg={metrics['geo_deg']:.4f} "
        f"zero_geo_deg={metrics['zero_geo_deg']:.4f} "
        f"tangent_epe_rad={metrics['tangent_epe_rad']:.6f} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )

    if args.checkpoint_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_out)), exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint_out)
        print(f"saved checkpoint to {args.checkpoint_out}", flush=True)


if __name__ == "__main__":
    main()
