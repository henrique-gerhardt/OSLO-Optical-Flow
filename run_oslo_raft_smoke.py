"""Smoke + overfit tests for OSLO-RAFT (Phase 1, before any real training).

Three checks from docs/OSLO_RAFT_PLAN.md Section 4.6:
  1. Forward/backward pass at the estimation grid (shapes, gradients flow).
  2. Cold-start sanity: zero-init flow head => iteration 0 prediction is ~zero flow.
  3. Overfit a handful of real pairs with no augmentation; the geodesic loss must
     drop near zero. "If it cannot overfit 10 pairs, debug before scaling."

Runs on CPU with a healpy-free Fibonacci grid stand-in (real nested HEALPix comes
from the container). Real pairs are read through the shard bridge.

    python run_oslo_raft_smoke.py \
        --shards "/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards"
"""

from __future__ import annotations

import argparse

import torch

from spherical_flow.geometry import fibonacci_unit_vectors, geodesic_distance
from spherical_flow.oslo_raft import (
    OSLORAFT,
    build_knn_level,
    endpoint_from_tangent_flow,
    sequence_geodesic_loss,
)
from spherical_flow.shard_dataset import load_shard_subset


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _mean_geo_deg(flow, gt_endpoint, level, valid) -> float:
    endpoint = endpoint_from_tangent_flow(level.points, flow, level.basis_east, level.basis_north)
    geo = torch.rad2deg(geodesic_distance(endpoint, gt_endpoint))
    m = valid if valid.dim() == 2 else valid.unsqueeze(0)
    return geo[m].mean().item() if m.any() else float("nan")


def test_forward_backward(model, level, iters) -> bool:
    b, n = 2, level.num_nodes
    f1 = torch.rand(b, n, 3)
    f2 = torch.rand(b, n, 3)
    preds = model(f1, f2, level, iters=iters)
    ok_shapes = len(preds) == iters and all(p.shape == (b, n, 2) for p in preds)
    loss = preds[-1].abs().mean()
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
    ok_grads = all(grads)
    print(f"  predictions: {len(preds)} x {tuple(preds[-1].shape)}  shapes_ok={ok_shapes}")
    print(f"  finite grads on all params: {ok_grads}")
    model.zero_grad(set_to_none=True)
    return ok_shapes and ok_grads


def test_cold_start(model, level, iters) -> bool:
    b, n = 1, level.num_nodes
    with torch.no_grad():
        preds = model(torch.rand(b, n, 3), torch.rand(b, n, 3), level, iters=iters)
    iter0_mag = preds[0].abs().mean().item()
    print(f"  iter-0 flow magnitude (zero-init head): {iter0_mag:.2e}")
    return iter0_mag < 1e-4


def test_overfit(model, level, samples, args) -> bool:
    frame1 = torch.stack([s["frame1"] for s in samples])
    frame2 = torch.stack([s["frame2"] for s in samples])
    gt_endpoint = torch.stack([s["endpoint"] for s in samples])
    valid = torch.stack([s["valid"] for s in samples])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    preds = model(frame1, frame2, level, iters=args.iters)
    init_geo = _mean_geo_deg(preds[-1], gt_endpoint, level, valid)
    gt_motion = _mean_geo_deg(torch.zeros_like(preds[-1]), gt_endpoint, level, valid)
    print(f"  GT motion (zero-flow geo) : {gt_motion:.3f} deg")
    print(f"  initial pred geo          : {init_geo:.3f} deg")

    model.train()
    final_loss = float("nan")
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        preds = model(frame1, frame2, level, iters=args.iters)
        loss = sequence_geodesic_loss(preds, gt_endpoint, level, valid, gamma=0.8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = loss.item()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            with torch.no_grad():
                geo = _mean_geo_deg(model(frame1, frame2, level, iters=args.iters)[-1], gt_endpoint, level, valid)
            print(f"  step {step:4d}  loss(rad)={final_loss:.5f}  pred geo={geo:.3f} deg")

    with torch.no_grad():
        final_geo = _mean_geo_deg(model(frame1, frame2, level, iters=args.iters)[-1], gt_endpoint, level, valid)
    # Pass if the model drives prediction error well below the raw GT motion.
    ok = final_geo < 0.35 * gt_motion and final_geo < init_geo
    print(f"  final pred geo            : {final_geo:.3f} deg  (target < {0.35*gt_motion:.3f})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default="/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards")
    parser.add_argument("--dataset", default="replica360")
    parser.add_argument("--split", default="val")
    parser.add_argument("--nodes", type=int, default=768, help="Fibonacci node count (smoke speed).")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--lookup-neighbors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    points = fibonacci_unit_vectors(args.nodes)
    level = build_knn_level(points, conv_neighbors=8, lookup_neighbors=args.lookup_neighbors)
    model = OSLORAFT(lookup_neighbors=args.lookup_neighbors + 1)
    print(f"nodes={level.num_nodes}  params={_count_params(model):,}\n")

    print("1) Forward/backward")
    r1 = test_forward_backward(model, level, args.iters)
    print("\n2) Cold-start (zero-init head)")
    r2 = test_cold_start(model, level, args.iters)

    print(f"\n3) Overfit {args.pairs} {args.dataset}[{args.split}] pairs")
    samples = load_shard_subset(args.shards, points, (args.dataset, args.split), max_pairs=args.pairs)
    if len(samples) < args.pairs:
        print(f"  only {len(samples)} pairs available; using those")
    r3 = test_overfit(model, level, samples, args)

    results = {"forward_backward": r1, "cold_start": r2, "overfit": r3}
    print("\n" + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
