import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from spherical_flow import healpix_unit_vectors, points_to_equirectangular_pixels, tangent_basis
from spherical_flow.flow360 import Flow360Dataset
from spherical_flow.raft_adapter import (
    FLOW_TRANSFORMS,
    erp_flow_to_tangent,
    flow_cache_path,
    load_frame_batch,
    load_raft_model,
    predict_raft_flow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cache TorchVision RAFT predictions as HEALPix tangent flow.")
    parser.add_argument("--data-root", default="/data/flow360")
    parser.add_argument("--grid-dir", default="/data/oslo_data/neighbor_grids")
    parser.add_argument("--cache-dir", default="/outputs/raft_cache")
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    parser.add_argument("--weights", default="default", choices=["default", "none"])
    parser.add_argument("--flow-transform", default="negated", choices=FLOW_TRANSFORMS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-out", default="")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    grid_file = Path(args.grid_dir) / f"healpix_grid_resolution_{args.resolution}.npz"
    if not grid_file.is_file():
        raise FileNotFoundError(f"Missing HEALPix grid file: {grid_file}")

    device = get_device(args.device)
    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)
    dataset = Flow360Dataset(
        args.data_root,
        split=args.split,
        points=points,
        direction=args.direction,
        max_pairs=args.max_pairs,
    )
    print(f"dataset={dataset.describe()}", flush=True)

    model, transforms, weights_name, torchvision_version = load_raft_model(args.model, args.weights, device)
    print(
        f"raft_cache model={args.model} weights={weights_name} transform={args.flow_transform} "
        f"resolution={args.resolution} device={device}",
        flush=True,
    )

    cache_dir = Path(args.cache_dir)
    pixel_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    written = 0
    skipped = 0
    manifest_records = []
    start_time = time.time()

    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"cache {args.split}", unit="batch"):
        end = min(start + args.batch_size, len(dataset))
        pairs = [dataset.pairs[idx] for idx in range(start, end)]
        outputs = [
            flow_cache_path(cache_dir, args.split, pair.sequence, pair.direction, pair.frame1.stem, args.resolution)
            for pair in pairs
        ]
        if not args.overwrite and all(path.is_file() for path in outputs):
            skipped += len(outputs)
            continue

        frame1, frame2, height, width = load_frame_batch(pairs)
        raft_flow = predict_raft_flow(model, transforms, frame1, frame2, device, args.flow_transform)
        key = (height, width)
        if key not in pixel_cache:
            pixel_cache[key] = points_to_equirectangular_pixels(points, height, width)
        u, v = pixel_cache[key]

        for batch_idx, pair in enumerate(pairs):
            out_path = outputs[batch_idx]
            if out_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            flow_erp = raft_flow[batch_idx].permute(1, 2, 0).contiguous()
            flow_tangent = erp_flow_to_tangent(flow_erp, points, basis_east, basis_north, u, v, height, width)
            np.savez_compressed(
                out_path,
                flow_tangent=flow_tangent.numpy().astype(np.float32, copy=False),
                model=np.array(args.model),
                requested_weights=np.array(args.weights),
                weights_enum=np.array(weights_name),
                flow_transform=np.array(args.flow_transform),
                torchvision=np.array(torchvision_version),
                resolution=np.array(args.resolution, dtype=np.int32),
                image_height=np.array(height, dtype=np.int32),
                image_width=np.array(width, dtype=np.int32),
                split=np.array(args.split),
                sequence=np.array(pair.sequence),
                direction=np.array(pair.direction),
                frame=np.array(pair.frame1.stem),
                frame1=np.array(str(pair.frame1)),
                frame2=np.array(str(pair.frame2)),
            )
            written += 1
            manifest_records.append(str(out_path))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.time() - start_time
    manifest = {
        "args": vars(args),
        "dataset": dataset.describe(),
        "model": {
            "name": args.model,
            "requested_weights": args.weights,
            "weights_enum": weights_name,
            "flow_transform": args.flow_transform,
            "torchvision": torchvision_version,
        },
        "written": written,
        "skipped": skipped,
        "elapsed_s": elapsed_s,
        "records": manifest_records,
    }
    manifest_out = Path(args.manifest_out) if args.manifest_out else cache_dir / f"{args.split}_r{args.resolution}_manifest.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"written={written} skipped={skipped} elapsed_s={elapsed_s:.1f}", flush=True)
    print(f"saved_manifest={manifest_out}", flush=True)


if __name__ == "__main__":
    main()
