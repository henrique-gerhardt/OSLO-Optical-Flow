"""Validate the shard -> HEALPix-node bridge (spherical_flow.shard_dataset).

For each dataset it loads a small capped subset through the bridge and reports:
  - nodes, pairs, valid fraction, GT motion magnitude (geodesic degrees);
  - a node-space photometric check: sampling frame2 at the GT endpoint should
    match frame1 at the node far better than zero-flow (frame2 at the node).
    A clear positive improvement means the diagnosed flow convention survived the
    ERP -> node resampling and the tangent targets point the right way.

Run from the OSLO repo root:
    python run_shard_dataset_check.py \
        --shards "/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards"
"""

from __future__ import annotations

import argparse

import torch

from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.geometry import (
    fibonacci_unit_vectors,
    geodesic_distance,
    points_to_equirectangular_pixels,
)
from spherical_flow.shard_dataset import _import_sfprep, load_shard_subset


def _build_points(resolution: int, device: torch.device) -> tuple[torch.Tensor, str]:
    """HEALPix nodes if healpy/astropy is available, else a Fibonacci stand-in."""
    try:
        from spherical_flow.geometry import healpix_unit_vectors

        pts = healpix_unit_vectors(resolution, device=device)
        return pts, f"healpix r={resolution} ({pts.size(0)} nodes)"
    except ImportError:
        n = 12 * (1 << resolution) ** 2
        pts = fibonacci_unit_vectors(n, device=device)
        return pts, f"fibonacci ({n} nodes, healpy unavailable)"


def _photometric_node_check(
    shards_dir: str,
    points: torch.Tensor,
    dataset: str,
    split: str,
    max_pairs: int,
) -> dict:
    """Warp frame2 to the GT endpoint vs zero-flow, in node space."""
    iter_shard, list_shards = _import_sfprep()
    height = width = None
    warp_err = []
    zero_err = []
    seen = 0
    for shard in list_shards(shards_dir, dataset, split):
        for record in iter_shard(shard):
            f1 = torch.from_numpy(record["frame1"]).float() / 255.0
            f2 = torch.from_numpy(record["frame2"]).float() / 255.0
            flow = torch.from_numpy(record["flow"]).float()
            valid = torch.from_numpy(record["valid"])
            height, width = f1.shape[:2]

            u, v = points_to_equirectangular_pixels(points, height, width)
            f1_node = bilinear_sample_erp(f1, u, v)
            f2_node = bilinear_sample_erp(f2, u, v)
            sampled_flow = bilinear_sample_erp(flow, u, v)
            node_valid = (
                bilinear_sample_erp(valid.float().unsqueeze(-1), u, v).squeeze(-1) > 0.999
            )

            eu = u + sampled_flow[:, 0]
            ev = (v + sampled_flow[:, 1]).clamp(0.0, float(height - 1))
            f2_warp = bilinear_sample_erp(f2, eu, ev)

            m = node_valid
            if m.any():
                warp_err.append((f1_node[m] - f2_warp[m]).abs().mean().item())
                zero_err.append((f1_node[m] - f2_node[m]).abs().mean().item())
            seen += 1
            if seen >= max_pairs:
                break
        if seen >= max_pairs:
            break

    we = float(torch.tensor(warp_err).mean()) if warp_err else float("nan")
    ze = float(torch.tensor(zero_err).mean()) if zero_err else float("nan")
    improvement = 100.0 * (ze - we) / ze if ze > 0 else float("nan")
    return {"warp_err": we, "zero_err": ze, "improvement_pct": improvement, "pairs": seen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        default="/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards",
        help="Path to the materialized shards directory.",
    )
    parser.add_argument("--resolution", type=int, default=5, help="HEALPix order (or Fibonacci count).")
    parser.add_argument("--max-pairs", type=int, default=12, help="Pairs per dataset.")
    args = parser.parse_args()

    device = torch.device("cpu")
    points, points_desc = _build_points(args.resolution, device)
    print(f"points: {points_desc}\n")

    sources = [
        ("flow360", "train"),
        ("replica360", "val"),
        ("mpf", "train"),
    ]

    for dataset, split in sources:
        print(f"=== {dataset} [{split}] ===")
        samples = load_shard_subset(
            args.shards, points, (dataset, split), max_pairs=args.max_pairs
        )
        if not samples:
            print("  no samples found\n")
            continue

        valid_fracs = []
        motion_deg = []
        rgb_min, rgb_max = 1.0, 0.0
        for s in samples:
            valid = s["valid"]
            valid_fracs.append(valid.float().mean().item())
            motion = torch.rad2deg(geodesic_distance(points, s["endpoint"]))
            if valid.any():
                motion_deg.append(motion[valid].mean().item())
            rgb_min = min(rgb_min, s["frame1"].min().item())
            rgb_max = max(rgb_max, s["frame1"].max().item())

        vf = float(torch.tensor(valid_fracs).mean())
        md = float(torch.tensor(motion_deg).mean()) if motion_deg else float("nan")
        photo = _photometric_node_check(args.shards, points, dataset, split, args.max_pairs)

        print(f"  pairs loaded     : {len(samples)}")
        print(f"  nodes/pair       : {samples[0]['frame1'].shape[0]}")
        print(f"  valid fraction   : {vf:.3f}")
        print(f"  GT motion (deg)  : {md:.4f}  (mean over valid nodes)")
        print(f"  frame1 RGB range : [{rgb_min:.3f}, {rgb_max:.3f}]")
        print(
            f"  photometric node : warp {photo['warp_err']:.4f} vs zero {photo['zero_err']:.4f}"
            f"  -> {photo['improvement_pct']:+.1f}% better"
        )
        print(f"  first uid/dir    : {samples[0]['uid']} / {samples[0]['direction']}\n")


if __name__ == "__main__":
    main()
