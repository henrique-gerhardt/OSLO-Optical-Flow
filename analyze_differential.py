"""Diagnostic: does the differential (LK) flow carry real motion signal where the structure
tensor is STRONG? Answers whether a confidence-gated / feature-constancy differential is
worth building, from the SAVED checkpoint — no retraining.

Two decisive read-outs on the val set:
  1. Direction agreement: median cos-sim(LK flow, GT flow) on active pixels, and on the
     high-confidence (top structure-tensor eigenvalue) subset. ~0 => no signal even where
     the gradient is reliable => the differential can't be rescued, write up.
  2. Simulated confidence gating: output the LK flow only on the top-X% most-reliable nodes
     (zero elsewhere) and recompute the active-subset geodesic improvement vs zero-flow. If
     the best gating fraction turns active POSITIVE, gating + a constancy loss are worth
     building; if every fraction stays <= 0, they are not.

Run in the container like run_oslo_raft.py, e.g.:
  python analyze_differential.py --grid healpix --resolution 4 \
    --checkpoint /outputs/oslo_raft_r4_diff/oslo_raft.pt --shards /data/shards \
    --val-sources flow360:val --max-val-pairs 512 --device cuda
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from spherical_flow.geometry import endpoint_from_tangent_flow, geodesic_distance
from spherical_flow.oslo_raft import build_knn_level
from spherical_flow.oslo_raft_diff import OSLORAFTDiff
from spherical_flow.shard_dataset import ShardFlowDataset
from run_oslo_raft import build_points, parse_sources, move_batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shards", default="/data/shards")
    p.add_argument("--val-sources", default="flow360:val")
    p.add_argument("--grid", default="healpix", choices=["fibonacci", "healpix"])
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--nodes", type=int, default=3072)
    p.add_argument("--conv-neighbors", type=int, default=8)
    p.add_argument("--lookup-neighbors", type=int, default=24)
    p.add_argument("--max-val-pairs", type=int, default=512)
    p.add_argument("--active-deg", type=float, default=0.25)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    points = build_points(args)
    level = build_knn_level(points, args.conv_neighbors, args.lookup_neighbors).to(device)

    model = OSLORAFTDiff(kernel_size=args.conv_neighbors + 1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  nodes={level.num_nodes} device={device}", flush=True)

    val_ds = ShardFlowDataset(args.shards, points, parse_sources(args.val_sources),
                              shuffle_shards=False, shuffle_buffer=0, seed=7)
    loader = DataLoader(val_ds, batch_size=2, num_workers=2)

    cols = {k: [] for k in ("lam", "geo_lk", "zero_geo", "cos", "lk_mag", "gt_mag")}
    seen = 0
    east, north, pts = level.basis_east, level.basis_north, level.points
    for batch in loader:
        batch = move_batch(batch, device)
        flow_lk, lam_min = model.confidence(batch["frame1"], batch["frame2"], level)  # [B,N,2],[B,N]
        gt_flow = batch["flow"]
        gt_end = batch["endpoint"]
        valid = batch.get("valid")
        valid = torch.ones(flow_lk.shape[:2], dtype=torch.bool, device=device) if valid is None else valid.bool()

        lk_end = endpoint_from_tangent_flow(pts, flow_lk, east, north)
        geo_lk = geodesic_distance(lk_end, gt_end)                       # [B,N] rad
        zero_geo = geodesic_distance(pts.unsqueeze(0).expand_as(gt_end), gt_end)
        lk_mag = flow_lk.norm(dim=-1)
        gt_mag = gt_flow.norm(dim=-1)
        cos = (flow_lk * gt_flow).sum(-1) / (lk_mag.clamp_min(1e-9) * gt_mag.clamp_min(1e-9))

        m = valid
        for k, t in [("lam", lam_min), ("geo_lk", geo_lk), ("zero_geo", zero_geo),
                     ("cos", cos), ("lk_mag", lk_mag), ("gt_mag", gt_mag)]:
            cols[k].append(t[m].detach().float().cpu().numpy())
        seen += flow_lk.size(0)
        if seen >= args.max_val_pairs:
            break

    d = {k: np.concatenate(v) for k, v in cols.items()}
    deg = 180.0 / np.pi
    motion_deg = d["zero_geo"] * deg
    active = motion_deg >= args.active_deg
    print(f"\nnodes={d['lam'].size:,}  active(>{args.active_deg}deg)={active.mean():.1%}")
    print(f"LK flow magnitude (deg): p50={np.median(d['lk_mag'])*deg:.3f}  "
          f"p90={np.quantile(d['lk_mag'],0.9)*deg:.3f}   GT motion p50={np.median(motion_deg):.3f}")

    # ---- 1. direction agreement (LK vs GT) by confidence, on active pixels ----------------
    print("\n[1] median cos-sim(LK, GT) on ACTIVE pixels, by structure-tensor confidence:")
    lam_a = d["lam"][active]
    cos_a = d["cos"][active]
    for q in (0.0, 0.5, 0.75, 0.9, 0.95):
        thr = np.quantile(lam_a, q)
        sel = lam_a >= thr
        print(f"    top {int((1-q)*100):3d}% conf (lam>={thr:.2e}): n={sel.sum():>7d}  "
              f"median cos={np.median(cos_a[sel]):+.3f}  mean cos={cos_a[sel].mean():+.3f}")

    # ---- 2. simulated confidence gating: active improvement vs zero-flow -------------------
    # Output LK flow only on the top-X% most-reliable nodes (zero elsewhere) and recompute the
    # active-subset geodesic improvement exactly as the training metric does.
    print(f"\n[2] simulated hard confidence gating -> active(>{args.active_deg}deg) improvement vs zero:")
    zero_geo_a = d["zero_geo"][active]
    geo_lk_a = d["geo_lk"][active]
    denom = zero_geo_a.sum()
    for keep in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        thr = np.quantile(lam_a, 1.0 - keep)
        kept = lam_a >= thr
        geo_used = np.where(kept, geo_lk_a, zero_geo_a)     # gated-out nodes fall back to zero-flow
        improvement = 100.0 * (zero_geo_a.sum() - geo_used.sum()) / denom
        # fraction of KEPT nodes where LK actually beats zero
        beat = (geo_lk_a[kept] < zero_geo_a[kept]).mean() if kept.any() else float("nan")
        print(f"    keep top {keep*100:5.1f}% conf: active_improvement={improvement:+6.2f}%   "
              f"(kept nodes where LK<zero: {beat:.1%})")

    print("\nREAD: [1] cos ~0 even at top conf => no signal, write up.  "
          "[2] if some keep-fraction is +, gating+constancy worth building; if all <=0, not.")


if __name__ == "__main__":
    main()
