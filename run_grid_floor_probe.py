"""Measure the estimation-grid floor: the best any r_est estimator could score at r_fine.

Motivation (2026-07-25, docs/plans/UNIVERSALITY_TABLE.md §7). On flowscape:test OSLO
scored 1.158° global vs PanoFlow's 0.251°. One hypothesis is the estimation grid:
OSLO decodes at ``estimation_resolution 4`` (3072 nodes, 3.66° mean spacing) while the
dataset's median motion is 2.47° — sub-node. But the decode-grid gap to PanoFlow is only
~1.6x linear (3.66° vs 2.24°), which does not obviously explain a 4.6x error gap. So the
grid is a HYPOTHESIS, not an established diagnosis, and training at r5 before testing it
would be a blind bet.

This probe settles it without training. It hands the estimation grid the PERFECT answer —
the ground-truth flow sampled directly at the estimation nodes — reconstructs the fine
grid through the model's own convex upsampler, and scores the result with the same
geodesic stack used for every published row. Whatever error remains is imposed purely by
the estimation grid's resolution; no estimator at that resolution can do better.

Three reconstructions bracket the achievable floor:

* ``pwc``     one-hot weights on the center node (piecewise-constant, parallel-transported)
              — what a naive upsampler gives; pessimistic.
* ``uniform`` equal weights over the 1-hop neighborhood — a smooth, untrained upsampler.
* ``oracle``  the best convex weights per descendant, solved by projected gradient on the
              simplex — no learned upsampler of this family can beat it. THE decisive bound.

Decision rule (pre-registered in §7):
  oracle floor >~ 1.0 deg  => OSLO (1.158°) is AT its grid ceiling; r5 is the fix.
  oracle floor <~ 0.4 deg  => the grid is NOT the bottleneck; r5 would be wasted compute.

Stated assumption: the coarse values are the GT *point-sampled at the estimation
nodes* — "a perfect estimator reports the true field at its own node". An
alternative is the per-cell average (the L2-optimal piecewise-constant field),
which would tighten the ``pwc`` row specifically; it does not bind the decision,
since ``oracle`` optimizes the reconstruction directly and is the number the rule
reads.

Example:
  python run_grid_floor_probe.py --shards /data/shards --sources flowscape:test \
      --resolution 6 --estimation-resolutions 4,5 --output-dir /outputs/grid_floor
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from spherical_flow.geometry import (
    healpix_unit_vectors,
    parallel_transport,
    set_geodesic_mode,
    tangent_basis,
)
from spherical_flow.healpix_pyramid import build_healpix_pyramid, load_pyramid, save_pyramid
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_maps,
    finalize_metrics,
    parse_bands,
    parse_thresholds,
    print_metrics,
    target_sample_from_maps,
)
from spherical_flow.shard_dataset import (
    _to_chw_free_float,
    sample_pair_to_nodes,
)
from run_raft_shard_baseline import iter_source_records, parse_sources


MODES = ("pwc", "uniform", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the estimation-grid floor with a perfect coarse estimator."
    )
    parser.add_argument("--shards", default="/data/shards")
    parser.add_argument("--sources", default="flowscape:test")
    parser.add_argument("--resolution", type=int, default=6,
                        help="Fine/supervision grid — must match the rows being compared.")
    parser.add_argument("--estimation-resolutions", default="4,5",
                        help="Comma-separated estimation grids to probe (e.g. '4,5').")
    parser.add_argument("--conv-neighbors", type=int, default=8,
                        help="Must match training so upsample_neighbors has the same K.")
    parser.add_argument("--oracle-iters", type=int, default=100,
                        help="Frank-Wolfe steps for the oracle convex combination.")
    parser.add_argument("--pyramid-cache", default="")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument("--motion-bands-deg", default="",
                        help="Disjoint motion bands (see run_raft_shard_baseline.py). "
                             "Empty = off.")
    parser.add_argument("--geodesic-metric", default="acos", choices=["acos", "haversine"],
                        help="Great-circle formula; 'haversine' removes the 0.028 deg "
                             "float32 floor that the r6 identity check exposed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="/outputs/grid_floor")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solve_oracle_combo(
    contrib: torch.Tensor, target_grouped: torch.Tensor, iters: int
) -> torch.Tensor:
    """Closest point to ``target`` in the convex hull of the ``K`` contributions.

    Frank-Wolfe with an exact line search: projection-free (so it never leaves the
    simplex), no step-size to tune, and each step is closed-form because the objective
    is a 2-D least-squares. Returns the best achievable combination ``[B, N, D, 2]``.
    """
    combo = contrib.mean(dim=3)                                   # feasible start
    for _ in range(iters):
        resid = combo - target_grouped                            # [B,N,D,2]
        # Linear minimization oracle over the simplex = pick the best single vertex.
        scores = (contrib * resid.unsqueeze(3)).sum(dim=-1)       # [B,N,D,K]
        best = scores.argmin(dim=-1, keepdim=True)                # [B,N,D,1]
        vertex = torch.gather(
            contrib, 3, best.unsqueeze(-1).expand(*best.shape, 2)
        ).squeeze(3)                                              # [B,N,D,2]
        direction = vertex - combo
        denom = (direction * direction).sum(dim=-1, keepdim=True)
        gamma = torch.where(
            denom > 1e-12,
            (-(resid * direction).sum(dim=-1, keepdim=True) / denom.clamp_min(1e-12)),
            torch.zeros_like(denom),
        ).clamp(0.0, 1.0)
        combo = combo + gamma * direction
    return combo


def transported_neighbor_flows(flow_est: torch.Tensor, pyramid) -> torch.Tensor:
    """Neighbor tangent flows transported to each descendant, in the FINE basis.

    Mirrors the transport half of :func:`convex_upsample` and returns the per-neighbor
    contributions before the convex combination, so different weightings can be compared
    on exactly the same geometry. Returns ``[B, N_est, D, K, 2]``.
    """
    est = pyramid.estimation_level
    fine = pyramid.fine_level
    nbr = pyramid.upsample_neighbors
    desc = pyramid.descendant_index

    b = flow_est.size(0)
    n_est, k = nbr.shape
    d = desc.size(1)

    nbr_flow = flow_est[:, nbr, :]
    t_nbr = nbr_flow[..., 0:1] * est.basis_east[nbr] + nbr_flow[..., 1:2] * est.basis_north[nbr]

    a = est.points[nbr].reshape(1, n_est, 1, k, 3)
    fine_dir = fine.points[desc]
    b_pt = fine_dir.reshape(1, n_est, d, 1, 3)
    t = t_nbr.reshape(b, n_est, 1, k, 3)
    transported = parallel_transport(t, a, b_pt)               # [B, N_est, D, K, 3]

    fe = fine.basis_east[desc].unsqueeze(2)                    # [N_est, D, 1, 3]
    fn = fine.basis_north[desc].unsqueeze(2)
    return torch.stack([(transported * fe).sum(-1), (transported * fn).sum(-1)], dim=-1)


def scatter_to_fine(flow_grouped: torch.Tensor, pyramid) -> torch.Tensor:
    desc = pyramid.descendant_index
    b = flow_grouped.size(0)
    out = flow_grouped.new_zeros(b, pyramid.fine_level.num_nodes, 2)
    out.index_copy_(1, desc.reshape(-1), flow_grouped.reshape(b, -1, 2))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    sources = parse_sources(args.sources)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)
    motion_bands = parse_bands(args.motion_bands_deg)
    set_geodesic_mode(args.geodesic_metric)
    est_resolutions = [int(t) for t in args.estimation_resolutions.split(",") if t.strip()]

    fine_points = healpix_unit_vectors(args.resolution)
    fine_east, fine_north = tangent_basis(fine_points)
    region_masks = build_region_masks(fine_points)

    print(f"fine grid r{args.resolution} nodes={fine_points.shape[0]} "
          f"estimation grids={est_resolutions} device={device}", flush=True)

    results: Dict[str, dict] = {}

    for r_est in est_resolutions:
        cache = Path(args.pyramid_cache) / f"pyr_r{r_est}_r{args.resolution}.pt" \
            if args.pyramid_cache else None
        if cache is not None and cache.exists():
            pyramid = load_pyramid(cache)
        else:
            pyramid = build_healpix_pyramid(
                estimation_resolution=r_est,
                fine_resolution=args.resolution,
                conv_neighbors=args.conv_neighbors,
            )
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                save_pyramid(pyramid, cache)
        pyramid = pyramid.to(device)

        est_points = pyramid.estimation_level.points.cpu()
        est_east = pyramid.estimation_level.basis_east.cpu()
        est_north = pyramid.estimation_level.basis_north.cpu()
        n_est = est_points.shape[0]
        spacing_deg = float(np.degrees(np.sqrt(4 * np.pi / n_est)))
        print(f"\n=== estimation r{r_est}: {n_est} nodes, mean spacing {spacing_deg:.3f}° "
              f"K={pyramid.upsample_neighbors.shape[1]} D={pyramid.descendant_index.shape[1]}",
              flush=True)

        acc = {m: ({}, {}, {}, []) for m in MODES}
        seen = 0
        start = time.time()

        for record in tqdm(iter_source_records(Path(args.shards), sources),
                           desc=f"r{r_est}", unit="pair"):
            frame1 = _to_chw_free_float(record["frame1"])
            frame2 = _to_chw_free_float(record["frame2"])
            flow_erp = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
            valid_erp = torch.from_numpy(np.ascontiguousarray(record["valid"]))

            # Perfect estimate at the ESTIMATION grid.
            coarse = sample_pair_to_nodes(
                frame1, frame2, flow_erp, valid_erp, est_points, est_east, est_north
            )
            # Reference target at the FINE grid (what every published row is scored on).
            fine = sample_pair_to_nodes(
                frame1, frame2, flow_erp, valid_erp, fine_points, fine_east, fine_north
            )

            flow_est = coarse["flow"].unsqueeze(0).to(device)
            target_batch = {
                "flow": fine["flow"].unsqueeze(0),
                "endpoint": fine["endpoint"].unsqueeze(0),
                "valid": fine["valid"].unsqueeze(0),
            }

            contrib = transported_neighbor_flows(flow_est, pyramid)      # [1,N,D,K,2]
            desc = pyramid.descendant_index
            target_grouped = target_batch["flow"].to(device)[:, desc, :]  # [1,N,D,2]

            recon = {
                "pwc": contrib[:, :, :, 0, :],
                "uniform": contrib.mean(dim=3),
                "oracle": solve_oracle_combo(contrib, target_grouped, args.oracle_iters),
            }

            for mode in MODES:
                pred = scatter_to_fine(recon[mode], pyramid).cpu()
                maps = compute_maps(pred, target_batch, fine_points, fine_east, fine_north)
                totals, counts, active_counts, chunks = acc[mode]
                chunks.append(target_sample_from_maps(maps, None))
                accumulate_maps(maps, region_masks, active_thresholds,
                                totals, counts, active_counts, motion_bands=motion_bands)

            seen += 1
            if args.max_pairs is not None and seen >= args.max_pairs:
                break

        if seen == 0:
            raise RuntimeError("no pairs evaluated")

        entry = {
            "estimation_resolution": r_est,
            "estimation_nodes": n_est,
            "mean_spacing_deg": spacing_deg,
            "pairs": seen,
            "elapsed_s": time.time() - start,
            "floors": {},
        }
        for mode in MODES:
            totals, counts, active_counts, chunks = acc[mode]
            metrics = finalize_metrics(totals, counts, active_counts, chunks)
            entry["floors"][mode] = metrics
            print_metrics(f"grid_floor_r{r_est}_{mode}", metrics, motion_bands)
        results[f"r{r_est}"] = entry

    print("\n================ GRID FLOOR SUMMARY ================", flush=True)
    print(f"{'est grid':<10} {'spacing':>9} {'pwc':>10} {'uniform':>10} {'oracle':>10}", flush=True)
    for key, entry in results.items():
        f = entry["floors"]
        print(f"{key:<10} {entry['mean_spacing_deg']:>8.3f}° "
              f"{f['pwc']['global_geo_deg']:>9.4f}° "
              f"{f['uniform']['global_geo_deg']:>9.4f}° "
              f"{f['oracle']['global_geo_deg']:>9.4f}°", flush=True)
    print("Decision rule (§7): oracle >~1.0° => grid IS the ceiling (r5 is the fix); "
          "oracle <~0.4° => grid is NOT the bottleneck (skip r5).", flush=True)

    out = output_dir / "grid_floor.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "sources": sources, "results": results},
                  handle, indent=2)
    print(f"saved_metrics={out}", flush=True)


if __name__ == "__main__":
    main()
