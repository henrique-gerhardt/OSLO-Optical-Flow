"""Measure the ground-truth displacement distribution of each dataset.

Reports the same quantity the evaluation harness reports as ``target_geo_deg_p50``
(``spherical_flow.metrics.target_sample_from_maps`` + ``finalize_metrics``): the
geodesic distance, in degrees, between each supervision node and its ground-truth
arrival point, pooled over all valid nodes of all pairs, then quantiled. Frames are
not decoded and no model is loaded — only ``flow`` and ``valid`` are read.

Run from the OSLO repo root:
    python run_dataset_motion_stats.py --datasets flow360:test replica360:val mpf:val
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from spherical_flow.geometry import (
    equirectangular_pixels_to_unit_vectors,
    geodesic_distance,
    healpix_unit_vectors,
    points_to_equirectangular_pixels,
    set_geodesic_mode,
)
from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.shard_dataset import _import_sfprep


def node_displacement_deg(
    points: torch.Tensor,
    flow: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Per-node geodesic displacement, in degrees, over the valid nodes of one pair.

    Mirrors ``spherical_flow.shard_dataset``: sample the ERP flow at the node pixel,
    step in pixel space, and lift the arrival pixel back to a unit vector. A node whose
    arrival leaves the image vertically is dropped, exactly as the bridge drops it.
    """
    u, v = points_to_equirectangular_pixels(points, height, width)
    sampled_flow = bilinear_sample_erp(flow, u, v)
    node_valid = bilinear_sample_erp(valid, u, v).squeeze(-1) > 0.999

    endpoint_u = u + sampled_flow[:, 0]
    endpoint_v = v + sampled_flow[:, 1]
    inside = (endpoint_v >= 0.0) & (endpoint_v <= float(height - 1))
    endpoint = equirectangular_pixels_to_unit_vectors(
        endpoint_u, endpoint_v.clamp(0.0, float(height - 1)), height, width
    )

    keep = node_valid & inside
    return torch.rad2deg(geodesic_distance(points[keep], endpoint[keep]))


def measure(shards_dir: str, dataset: str, split: str, points: torch.Tensor, max_pairs: int) -> dict:
    iter_shard, list_shards = _import_sfprep()
    chunks: list[np.ndarray] = []
    pairs = 0
    t0 = time.time()
    for shard in list_shards(shards_dir, dataset, split):
        for record in iter_shard(shard):
            flow = torch.from_numpy(record["flow"]).float()
            valid = torch.from_numpy(record["valid"]).float().unsqueeze(-1)
            height, width = flow.shape[:2]
            d = node_displacement_deg(points, flow, valid, height, width)
            if d.numel():
                chunks.append(d.numpy().astype(np.float32))
            pairs += 1
            if pairs >= max_pairs:
                break
        if pairs >= max_pairs:
            break
    if not chunks:
        return {"pairs": 0}
    d = np.concatenate(chunks)
    p50, p90, p95 = np.quantile(d, [0.50, 0.90, 0.95])
    return {
        "pairs": pairs,
        "nodes": int(d.size),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "mean": float(d.mean()),
        "secs": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", default="/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards")
    p.add_argument("--resolution", type=int, default=6, help="HEALPix order of the supervision grid.")
    p.add_argument("--max-pairs", type=int, default=256, help="Pairs per dataset.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["flow360:test", "replica360:val", "mpf:val"],
        help="dataset:split entries.",
    )
    args = p.parse_args()

    set_geodesic_mode("haversine")
    points = healpix_unit_vectors(args.resolution, device=torch.device("cpu"))
    print(f"grid: healpix r={args.resolution} ({points.size(0)} nodes), metric: haversine\n")
    print(f"{'dataset':<18}{'pairs':>7}{'nodes':>12}{'p50':>10}{'p90':>10}{'p95':>10}{'mean':>10}")

    for entry in args.datasets:
        dataset, split = entry.split(":")
        s = measure(args.shards, dataset, split, points, args.max_pairs)
        if not s["pairs"]:
            print(f"{entry:<18}{'no shards found':>49}")
            continue
        print(
            f"{entry:<18}{s['pairs']:>7}{s['nodes']:>12,}"
            f"{s['p50']:>9.3f}°{s['p90']:>9.3f}°{s['p95']:>9.3f}°{s['mean']:>9.3f}°"
            f"   ({s['secs']:.0f}s)"
        )


if __name__ == "__main__":
    main()
