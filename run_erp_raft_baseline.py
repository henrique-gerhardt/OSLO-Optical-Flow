import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from spherical_flow import (
    equirectangular_pixels_to_unit_vectors,
    healpix_unit_vectors,
    logmap,
    points_to_equirectangular_pixels,
    tangent_basis,
)
from spherical_flow.flow360 import Flow360Dataset, Flow360Pair, bilinear_sample_erp
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_maps,
    finalize_metrics,
    parse_thresholds,
    print_metrics,
    target_sample_from_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchVision RAFT ERP baseline on FLOW360 with spherical metrics."
    )
    parser.add_argument("--data-root", default="/data/flow360", help="FLOW360 root with train/test folders.")
    parser.add_argument("--grid-dir", default="/data/oslo_data/neighbor_grids", help="OSLO HEALPix neighbor grids.")
    parser.add_argument("--output-dir", default="/outputs/raft_r6", help="Directory for metrics and predictions.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--resolution", type=int, default=6, help="HEALPix order used for spherical evaluation.")
    parser.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    parser.add_argument("--weights", default="default", choices=["default", "none"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Multiplier for FLOW360 target pixel flow values.")
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument(
        "--target-quantile-max-samples",
        type=int,
        default=2_000_000,
        help="Maximum validation samples used for target motion percentiles.",
    )
    parser.add_argument("--save-predictions", action="store_true", help="Save RAFT ERP pixel-flow predictions.")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_raft(args: argparse.Namespace, device: torch.device):
    import torchvision
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if args.model == "raft_large":
        weights_enum = Raft_Large_Weights.DEFAULT if args.weights == "default" else None
        model = raft_large(weights=weights_enum).to(device)
    else:
        weights_enum = Raft_Small_Weights.DEFAULT if args.weights == "default" else None
        model = raft_small(weights=weights_enum).to(device)

    transforms = weights_enum.transforms() if weights_enum is not None else None
    model.eval()
    return model, transforms, weights_enum, torchvision.__version__


def normalize_untrained_inputs(frame1: torch.Tensor, frame2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return frame1.float().div(127.5).sub(1.0), frame2.float().div(127.5).sub(1.0)


@torch.no_grad()
def predict_raft_flow(
    model: torch.nn.Module,
    transforms,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    frame1 = frame1.to(device, non_blocking=True)
    frame2 = frame2.to(device, non_blocking=True)
    if transforms is not None:
        frame1, frame2 = transforms(frame1, frame2)
    else:
        frame1, frame2 = normalize_untrained_inputs(frame1, frame2)
    predictions = model(frame1, frame2)
    return predictions[-1].detach().float().cpu()


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


def build_target_batch(items: list[dict]) -> dict:
    return {
        "flow": torch.stack([item["flow"] for item in items], dim=0),
        "endpoint": torch.stack([item["endpoint"] for item in items], dim=0),
        "valid": torch.stack([item["valid"] for item in items], dim=0),
    }


def save_prediction(output_dir: Path, split: str, pair: Flow360Pair, flow_erp: torch.Tensor) -> None:
    pred_dir = output_dir / "predictions" / split / pair.sequence / pair.direction
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{pair.frame1.stem}.npy", flow_erp.numpy().astype(np.float32, copy=False))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_file = Path(args.grid_dir) / f"healpix_grid_resolution_{args.resolution}.npz"
    if not grid_file.is_file():
        raise FileNotFoundError(f"Missing HEALPix grid file: {grid_file}")

    device = get_device(args.device)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)
    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)
    region_masks = build_region_masks(points)

    dataset = Flow360Dataset(
        args.data_root,
        split=args.split,
        points=points,
        direction=args.direction,
        flow_scale=args.flow_scale,
        max_pairs=args.max_pairs,
    )
    dataset_description = dataset.describe()
    print(f"dataset={dataset_description}", flush=True)

    model, transforms, weights_enum, torchvision_version = load_raft(args, device)
    weights_name = weights_enum.name if weights_enum is not None else "none"
    print(
        f"raft model={args.model} weights={weights_name} device={device} "
        f"batch_size={args.batch_size} torchvision={torchvision_version}",
        flush=True,
    )

    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    active_counts: Dict[str, float] = {}
    target_chunks: list[torch.Tensor] = []
    sample_per_batch = None
    if args.target_quantile_max_samples > 0:
        num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
        sample_per_batch = max(1, args.target_quantile_max_samples // max(num_batches, 1))

    pixel_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    start_time = time.time()

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="RAFT ERP", unit="batch"):
        end = min(start + args.batch_size, len(dataset))
        pair_batch = [dataset.pairs[idx] for idx in range(start, end)]
        target_items = [dataset[idx] for idx in range(start, end)]
        target_batch = build_target_batch(target_items)

        frame1, frame2, height, width = load_frame_batch(pair_batch)
        raft_flow = predict_raft_flow(model, transforms, frame1, frame2, device)

        cache_key = (height, width)
        if cache_key not in pixel_cache:
            pixel_cache[cache_key] = points_to_equirectangular_pixels(points, height, width)
        u, v = pixel_cache[cache_key]

        pred_flows: list[torch.Tensor] = []
        for batch_idx, pair in enumerate(pair_batch):
            flow_erp = raft_flow[batch_idx].permute(1, 2, 0).contiguous()
            pred_flows.append(erp_flow_to_tangent(flow_erp, points, basis_east, basis_north, u, v, height, width))
            if args.save_predictions:
                save_prediction(output_dir, args.split, pair, flow_erp)

        pred_flow = torch.stack(pred_flows, dim=0)
        maps = compute_maps(pred_flow, target_batch, points, basis_east, basis_north)
        target_chunks.append(target_sample_from_maps(maps, sample_per_batch))
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = finalize_metrics(totals, counts, active_counts, target_chunks)
    metrics["elapsed_s"] = time.time() - start_time
    print_metrics("raft_validation", metrics)
    print(f"elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    result = {
        "args": vars(args),
        "dataset": dataset_description,
        "model": {
            "name": args.model,
            "requested_weights": args.weights,
            "weights_enum": weights_name,
            "torchvision": torchvision_version,
        },
        "metrics": metrics,
    }
    metrics_out = output_dir / "raft_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
