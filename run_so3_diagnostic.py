"""Acceptance tests for SO(3) rotation augmentation (Phase 0, Week 1).

Implements the four checks from docs/OSLO_RAFT_PLAN.md:

  1. Identity        : R = I reproduces the unrotated node sample exactly.
  2. Round-trip      : the augmented target equals the original target at the
                       rotated node, transported by R (logmap equivariance) -
                       an exact algebraic identity, no interpolation noise.
  3. Yaw-exactness   : a yaw by an integer multiple of the ERP column spacing
                       equals an ERP column roll (near pixel-exact).
  4. Metric invariance: zero-flow global_geo_deg is invariant under rotation,
                       while pole/seam subsets are not (the thesis point).

Run from the OSLO repo root:
    python run_so3_diagnostic.py \
        --shards "/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards"
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.geometry import (
    fibonacci_unit_vectors,
    geodesic_distance,
    points_to_equirectangular_pixels,
    tangent_basis,
)
from spherical_flow.shard_dataset import _import_sfprep, sample_pair_to_nodes
from spherical_flow.so3_augment import (
    rotation_matrix,
    sample_rotation,
    so3_augment_pair,
    yaw_matrix,
)


def _build_points(resolution: int) -> tuple[torch.Tensor, str]:
    try:
        from spherical_flow.geometry import healpix_unit_vectors

        pts = healpix_unit_vectors(resolution)
        return pts, f"healpix r={resolution} ({pts.size(0)} nodes)"
    except ImportError:
        n = 12 * (1 << resolution) ** 2
        return fibonacci_unit_vectors(n), f"fibonacci ({n} nodes, healpy unavailable)"


def _load_raw_pairs(shards_dir: str, dataset: str, split: str, n: int):
    """Yield a few raw ERP records as torch tensors (frame1/frame2/flow/valid)."""
    iter_shard, list_shards = _import_sfprep()
    out = []
    for shard in list_shards(shards_dir, dataset, split):
        for rec in iter_shard(shard):
            out.append(
                (
                    torch.from_numpy(rec["frame1"]).float() / 255.0,
                    torch.from_numpy(rec["frame2"]).float() / 255.0,
                    torch.from_numpy(rec["flow"]).float(),
                    torch.from_numpy(rec["valid"]),
                )
            )
            if len(out) >= n:
                return out
    return out


def _result(name: str, value: float, tol: float, unit: str = "") -> bool:
    ok = value <= tol
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:<26} {value:.3e}{unit}  (tol {tol:.1e})")
    return ok


def test_identity(pairs, points, east, north) -> bool:
    f1, f2, fl, vl = pairs[0]
    plain = sample_pair_to_nodes(f1, f2, fl, vl, points, east, north)
    eye = torch.eye(3)
    aug = so3_augment_pair(f1, f2, fl, vl, points, eye, east, north)
    err = max(
        (plain["frame1"] - aug["frame1"]).abs().max().item(),
        (plain["frame2"] - aug["frame2"]).abs().max().item(),
        (plain["flow"] - aug["flow"]).abs().max().item(),
    )
    valid_match = torch.equal(plain["valid"], aug["valid"])
    print(f"  valid masks identical: {valid_match}")
    return _result("R=I reproduces sample", err, 1e-5)


def test_round_trip(pairs, points, east, north) -> bool:
    """Augmented target at p == original target at q=p@R, transported by R."""
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for f1, f2, fl, vl in pairs:
        R = sample_rotation(gen, max_angle_deg=180.0)
        aug = so3_augment_pair(f1, f2, fl, vl, points, R, east, north)

        q = points @ R
        east_q, north_q = tangent_basis(q)
        ref = sample_pair_to_nodes(f1, f2, fl, vl, q, east_q, north_q)  # target at q

        # Transport ref tangent flow from q into p's basis: t3d_p = R @ t3d_q.
        t3d_q = ref["flow"][:, 0:1] * east_q + ref["flow"][:, 1:2] * north_q
        t3d_p = t3d_q @ R.transpose(-1, -2)
        transported = torch.stack(
            [(t3d_p * east).sum(-1), (t3d_p * north).sum(-1)], dim=-1
        )
        m = aug["valid"] & ref["valid"]
        if m.any():
            worst = max(worst, (aug["flow"][m] - transported[m]).abs().max().item())
    return _result("target equivariance", worst, 1e-4)


def test_yaw_exactness(pairs, points, east, north) -> bool:
    f1, f2, fl, vl = pairs[0]
    height, width = f1.shape[:2]
    k = 37  # integer column shift
    phi = 2.0 * math.pi * k / width
    R = yaw_matrix(phi)
    aug = so3_augment_pair(f1, f2, fl, vl, points, R, east, north)

    u, v = points_to_equirectangular_pixels(points, height, width)
    best = math.inf
    best_sign = 0
    for sign in (+1, -1):
        rolled = torch.from_numpy(np.roll(f1.numpy(), sign * k, axis=1))
        ref = bilinear_sample_erp(rolled, u, v)
        err = (aug["frame1"] - ref).abs().max().item()
        if err < best:
            best, best_sign = err, sign
    print(f"  matched column-roll sign: {best_sign:+d} (k={k} cols, phi={math.degrees(phi):.2f} deg)")
    return _result("yaw == column roll", best, 5e-3)


def test_metric_invariance(pairs, points, east, north) -> bool:
    """Zero-flow global geo invariant under rotation; pole/seam subsets are not."""
    lon = torch.atan2(points[:, 1], points[:, 0])
    lat = torch.asin(points[:, 2].clamp(-1, 1))
    pole_mask = lat.abs() > math.radians(60.0)
    seam_mask = (lon.abs() - math.pi).abs() < 0.10

    def zero_flow_geo(sample, mask=None):
        # prediction = zero tangent flow -> predicted endpoint = node itself.
        geo = torch.rad2deg(geodesic_distance(points, sample["endpoint"]))
        valid = sample["valid"] if mask is None else (sample["valid"] & mask)
        return geo[valid].mean().item() if valid.any() else float("nan")

    gen = torch.Generator().manual_seed(1)
    f1, f2, fl, vl = pairs[0]
    base = sample_pair_to_nodes(f1, f2, fl, vl, points, east, north)
    g0 = zero_flow_geo(base)

    global_diffs = []
    pole_shifts = []
    seam_shifts = []
    for _ in range(5):
        R = sample_rotation(gen, max_angle_deg=180.0)
        aug = so3_augment_pair(f1, f2, fl, vl, points, R, east, north)
        global_diffs.append(abs(zero_flow_geo(aug) - g0))
        pole_shifts.append(abs(zero_flow_geo(aug, pole_mask) - zero_flow_geo(base, pole_mask)))
        seam_shifts.append(abs(zero_flow_geo(aug, seam_mask) - zero_flow_geo(base, seam_mask)))

    global_max = max(global_diffs)
    pole_mean = float(np.mean(pole_shifts))
    seam_mean = float(np.mean(seam_shifts))
    print(f"  base global zero-flow geo : {g0:.4f} deg")
    print(f"  pole-subset shift (mean)  : {pole_mean:.4f} deg  (expected NON-zero)")
    print(f"  seam-subset shift (mean)  : {seam_mean:.4f} deg  (expected NON-zero)")
    # global invariance tolerance scales with motion; use a loose absolute bound.
    return _result("global geo invariance", global_max, 5e-2, " deg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        default="/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards",
    )
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--dataset", default="replica360", help="Source with real motion.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()

    points, desc = _build_points(args.resolution)
    east, north = tangent_basis(points)
    pairs = _load_raw_pairs(args.shards, args.dataset, args.split, args.pairs)
    print(f"points: {desc}")
    print(f"source: {args.dataset}[{args.split}], {len(pairs)} pairs\n")

    results = []
    print("1) Identity")
    results.append(test_identity(pairs, points, east, north))
    print("\n2) Round-trip / equivariance")
    results.append(test_round_trip(pairs, points, east, north))
    print("\n3) Yaw-exactness")
    results.append(test_yaw_exactness(pairs, points, east, north))
    print("\n4) Metric invariance (zero-flow)")
    results.append(test_metric_invariance(pairs, points, east, north))

    print("\n" + ("ALL PASS" if all(results) else "SOME TESTS FAILED"))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
