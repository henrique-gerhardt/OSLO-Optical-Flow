"""Train / evaluate OSLO-RAFT on the standard shards (Phase 2 pipeline harness).

End-to-end wiring of everything Phase 0/1 built: the shard->node bridge
(`ShardFlowDataset`) with SO(3) augmentation, the OSLO-RAFT iterative model, the
geodesic sequence loss, and the shared `spherical_flow.metrics` pipeline — so the
output JSON has the same schema as the MVP/residual runners and flows straight into
`run_aggregate_results.py`.

This is grid-agnostic. ``--grid fibonacci`` (default) runs anywhere on CPU for
pipeline validation; ``--grid healpix`` uses real HEALPix node directions (needs
healpy) and is the path for the GPU box. The nested-HEALPix multi-resolution model
(pyramid + convex upsampler) is a later drop-in; this trains the single-resolution
core to exercise the whole loop before moving to the GPU.

    # CPU pipeline smoke (tiny):
    python run_oslo_raft.py --grid fibonacci --nodes 768 --steps 20 \
        --train-sources replica360:train --val-sources replica360:val \
        --max-val-pairs 16 --output-dir outputs/oslo_raft_smoke

    # GPU run (in container):
    python run_oslo_raft.py --grid healpix --resolution 4 --device cuda --amp \
        --train-sources flow360:train,replica360:train,mpf:train \
        --val-sources flow360:val --so3-prob 1.0 --steps 100000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from spherical_flow.geometry import fibonacci_unit_vectors, tangent_basis
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_maps,
    finalize_metrics,
    parse_thresholds,
    print_metrics,
    summarize_maps,
    target_sample_from_maps,
)
from spherical_flow.oslo_raft import OSLORAFT, build_knn_level, sequence_geodesic_loss
from spherical_flow.shard_dataset import ShardFlowDataset


def parse_sources(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"source '{token}' must be 'dataset:split'")
        dataset, split = token.split(":", 1)
        out.append((dataset.strip(), split.strip()))
    if not out:
        raise ValueError("no sources parsed")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", default="/Volumes/External SSD/Mestrado/sphereflow-dataprep/shards")
    p.add_argument("--train-sources", default="replica360:train,mpf:train,flow360:train")
    p.add_argument("--val-sources", default="flow360:val")
    p.add_argument("--output-dir", default="outputs/oslo_raft")
    # geometry
    p.add_argument("--grid", default="fibonacci", choices=["fibonacci", "healpix"])
    p.add_argument("--resolution", type=int, default=4,
                   help="HEALPix order (grid=healpix); the FINE/supervision order when --multi-res.")
    p.add_argument("--nodes", type=int, default=3072, help="Fibonacci node count (grid=fibonacci).")
    p.add_argument("--conv-neighbors", type=int, default=8)
    p.add_argument("--lookup-neighbors", type=int, default=24)
    # multi-resolution (estimate coarse, supervise fine via the convex upsampler)
    p.add_argument("--multi-res", action="store_true",
                   help="Use OSLORAFTPyramid: estimate at --estimation-resolution, supervise at "
                        "--resolution. Requires --grid healpix.")
    p.add_argument("--estimation-resolution", type=int, default=4,
                   help="Coarse estimation order for --multi-res (correlation/GRU live here).")
    p.add_argument("--corr-pool-levels", type=int, default=3,
                   help="Second-image correlation pyramid depth for --multi-res (r_est .. r_est-N).")
    # single-resolution local correlation (estimate AND supervise at --resolution, e.g. r6)
    p.add_argument("--local-corr", action="store_true",
                   help="Use OSLORAFTLocal: single-res model at --resolution with a memory-local "
                        "correlation lookup (no [N,N] volume), so the estimation grid can be r6. "
                        "Requires --grid healpix. Mutually exclusive with --multi-res.")
    p.add_argument("--differential", action="store_true",
                   help="Use OSLORAFTDiff: a spherical Lucas-Kanade differential flow head (flow from "
                        "the tangent-space spatial gradient of f1 and the temporal difference f2-f1) "
                        "— the sub-pixel estimator. Single-res; mutually exclusive with the others.")
    # model / optim
    p.add_argument("--hidden-channels", type=int, default=96)
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--flow-scale", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.8)
    # loss re-balancing (diagnose/break the static-majority ceiling)
    p.add_argument("--loss-motion-weight", type=float, default=0.0,
                   help="Up-weight moving pixels: weight = 1 + w*min(gt_motion/ref, 1).")
    p.add_argument("--loss-motion-ref-deg", type=float, default=1.0,
                   help="GT-motion (deg) at which --loss-motion-weight saturates.")
    p.add_argument("--loss-min-target-deg", type=float, default=0.0,
                   help="Drop pixels with GT motion below this (deg) from the loss (active-only).")
    p.add_argument("--ablate-corr", action="store_true",
                   help="Zero the correlation feature before the motion encoder (diagnostic: does "
                        "the model match, or just regress a motion prior?). Not for --multi-res.")
    p.add_argument("--ablate-context", action="store_true",
                   help="Zero the context-net contribution (GRU hidden init + per-iter context feed) "
                        "so only correlation + flow-recurrence drive flow (diagnostic: does correlation "
                        "carry usable signal the context shortcut masks?). Not for --multi-res.")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--onecycle", action="store_true", help="OneCycle LR schedule over --steps.")
    # augmentation
    p.add_argument("--so3-prob", type=float, default=1.0)
    p.add_argument("--so3-max-angle-deg", type=float, default=180.0)
    p.add_argument("--so3-uniform", action="store_true", help="Haar SO(3) angle density.")
    # data plumbing
    p.add_argument("--shuffle-buffer", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-pairs", type=int, default=None, help="Cap train pairs/epoch (debug/overfit).")
    p.add_argument("--max-val-pairs", type=int, default=None)
    p.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    # runtime
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=0, help="Periodic val (0 = only at end).")
    p.add_argument("--smoke-test", action="store_true", help="One train step + one eval, then exit.")
    p.add_argument("--checkpoint-out", default="")
    p.add_argument("--metrics-out", default="")
    return p.parse_args()


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


def git_hash() -> str:
    # OSLO_GIT_SHA lets the container (which has no .git) still record provenance;
    # the Dockerfile bakes it at build time. stderr is silenced so a missing repo
    # doesn't print "fatal: not a git repository".
    env_sha = os.environ.get("OSLO_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_points(args: argparse.Namespace) -> torch.Tensor:
    if args.grid == "healpix":
        from spherical_flow.geometry import healpix_unit_vectors

        return healpix_unit_vectors(args.resolution)
    return fibonacci_unit_vectors(args.nodes)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, geom, sup_level, region_masks, active_thresholds, device, eval_iters, max_pairs):
    # geom is what the model consumes (a SphereLevel single-res, or a SpherePyramid multi-res);
    # sup_level is the grid the loss/metrics live on (the level itself, or the pyramid's fine level).
    model.eval()
    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    active_counts: Dict[str, float] = {}
    target_chunks: List[torch.Tensor] = []
    seen = 0
    points = sup_level.points
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["frame1"], batch["frame2"], geom, iters=eval_iters)[-1]
        maps = compute_maps(pred, batch, points, sup_level.basis_east, sup_level.basis_north)
        target_chunks.append(target_sample_from_maps(maps, None))
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts)
        seen += batch["frame1"].size(0)
        if max_pairs is not None and seen >= max_pairs:
            break
    return finalize_metrics(totals, counts, active_counts, target_chunks)


def cycling_loader(dataset: ShardFlowDataset, loader: DataLoader):
    """Infinite stream over the (finite) IterableDataset, bumping epoch each pass.

    Note: with ``num_workers > 0`` under spawn, ``set_epoch`` updates the main-process
    copy only, so per-epoch shard reshuffle does not reach workers (the shuffle buffer
    + initial shard order still randomize within the long first epoch). For exact
    per-epoch reshuffle with workers, move the epoch into a shared value; not needed at
    100k-step training scale.
    """
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)
    train_sources = parse_sources(args.train_sources)
    val_sources = parse_sources(args.val_sources)

    _model_modes = sum([args.multi_res, args.local_corr, args.differential])
    if _model_modes > 1:
        raise SystemExit("--multi-res / --local-corr / --differential are mutually exclusive")
    if args.differential and (args.ablate_corr or args.ablate_context):
        raise SystemExit("--differential has no correlation/context path to ablate")
    if args.ablate_corr and args.multi_res:
        raise SystemExit("--ablate-corr is not supported with --multi-res (no corr ablation hook)")
    if args.ablate_context and args.multi_res:
        raise SystemExit("--ablate-context is not supported with --multi-res (no context ablation hook)")

    if args.differential:
        from spherical_flow.oslo_raft_diff import OSLORAFTDiff

        points = build_points(args)
        level = build_knn_level(points, args.conv_neighbors, args.lookup_neighbors).to(device)
        dataset_points = points
        geom, sup_level = level, level
        model = OSLORAFTDiff(kernel_size=args.conv_neighbors + 1).to(device)
        print(f"grid={args.grid} differential nodes={level.num_nodes} "
              f"device={device} git={git_hash()[:9]}", flush=True)
    elif args.local_corr:
        if args.grid != "healpix":
            raise SystemExit("--local-corr requires --grid healpix")
        from spherical_flow.geometry import healpix_unit_vectors
        from spherical_flow.healpix_pyramid import _build_level
        from spherical_flow.oslo_raft_local import OSLORAFTLocal

        # _build_level is the chunked SphereLevel builder: it never materializes the [N,N]
        # similarity that build_knn_level's topk/ang2pix would (9.66 GB at r6). OSLORAFTLocal
        # never calls level.ang2pix (it resolves the displaced node locally), so the brute
        # ang2pix closure the builder attaches is never exercised at r6.
        points = healpix_unit_vectors(args.resolution)
        level = _build_level(points, args.conv_neighbors, args.lookup_neighbors, knn_chunk=2048).to(device)
        dataset_points = points
        geom, sup_level = level, level
        model = OSLORAFTLocal(
            hidden_channels=args.hidden_channels, context_dim=args.context_dim,
            kernel_size=args.conv_neighbors + 1, lookup_neighbors=args.lookup_neighbors + 1,
            flow_scale=args.flow_scale,
        ).to(device)
        print(f"grid=healpix local-corr res={args.resolution} nodes={level.num_nodes} "
              f"device={device} git={git_hash()[:9]}", flush=True)
    elif args.multi_res:
        if args.grid != "healpix":
            raise SystemExit("--multi-res requires --grid healpix")
        if args.resolution <= args.estimation_resolution:
            raise SystemExit(
                "--multi-res needs --resolution (fine) > --estimation-resolution "
                "(e.g. --resolution 6 --estimation-resolution 4)"
            )
        from spherical_flow.healpix_pyramid import build_healpix_pyramid
        from spherical_flow.oslo_raft_pyramid import OSLORAFTPyramid

        pyramid = build_healpix_pyramid(
            fine_resolution=args.resolution,
            estimation_resolution=args.estimation_resolution,
            corr_pool_levels=args.corr_pool_levels,
            conv_neighbors=args.conv_neighbors,
            lookup_neighbors=args.lookup_neighbors,
        )
        dataset_points = pyramid.fine_level.points  # sample frames/GT at the supervision grid
        pyramid = pyramid.to(device)
        geom, sup_level = pyramid, pyramid.fine_level
        model = OSLORAFTPyramid(
            pyramid, hidden_channels=args.hidden_channels,
            context_dim=args.context_dim, flow_scale=args.flow_scale,
        ).to(device)
        print(f"grid=healpix multi-res est={args.estimation_resolution} fine={args.resolution} "
              f"nodes_est={pyramid.num_estimation_nodes} nodes_fine={pyramid.num_fine_nodes} "
              f"device={device} git={git_hash()[:9]}", flush=True)
    else:
        points = build_points(args)
        level = build_knn_level(points, args.conv_neighbors, args.lookup_neighbors).to(device)
        dataset_points = points
        geom, sup_level = level, level
        model = OSLORAFT(
            hidden_channels=args.hidden_channels, context_dim=args.context_dim,
            kernel_size=args.conv_neighbors + 1, lookup_neighbors=args.lookup_neighbors + 1,
            flow_scale=args.flow_scale,
        ).to(device)
        print(f"grid={args.grid} nodes={level.num_nodes} device={device} git={git_hash()[:9]}", flush=True)

    # Honored by OSLORAFT / OSLORAFTLocal (guarded above so it is never silently a no-op).
    model.ablate_corr = args.ablate_corr
    model.ablate_context = args.ablate_context
    if args.ablate_corr:
        print("ablate_corr=True (correlation feature zeroed before the motion encoder)", flush=True)
    if args.ablate_context:
        print("ablate_context=True (context net zeroed; only correlation + recurrence drive flow)", flush=True)
    if args.loss_motion_weight > 0.0 or args.loss_min_target_deg > 0.0:
        print(f"loss re-balance: motion_weight={args.loss_motion_weight} "
              f"motion_ref_deg={args.loss_motion_ref_deg} min_target_deg={args.loss_min_target_deg}",
              flush=True)

    region_masks = build_region_masks(sup_level.points)

    train_ds = ShardFlowDataset(
        args.shards, dataset_points, train_sources,
        shuffle_shards=True, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        max_pairs=args.max_train_pairs,
        so3_prob=args.so3_prob, so3_max_angle_deg=args.so3_max_angle_deg, so3_uniform=args.so3_uniform,
    )
    val_ds = ShardFlowDataset(
        args.shards, dataset_points, val_sources, shuffle_shards=False, shuffle_buffer=0, seed=args.seed,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=pin, persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=min(2, args.num_workers), pin_memory=pin)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (
        torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
        if args.onecycle else None
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    stream = cycling_loader(train_ds, train_loader)
    start = time.time()
    model.train()
    total_steps = 1 if args.smoke_test else args.steps
    for step in range(1, total_steps + 1):
        batch = move_batch(next(stream), device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            preds = model(batch["frame1"], batch["frame2"], geom, iters=args.iters)
            loss = sequence_geodesic_loss(
                preds, batch["endpoint"], sup_level, batch["valid"], gamma=args.gamma,
                motion_weight=args.loss_motion_weight, motion_ref_deg=args.loss_motion_ref_deg,
                min_target_deg=args.loss_min_target_deg,
            )
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()

        if step == 1 or step == total_steps or step % args.log_every == 0:
            with torch.no_grad():
                maps = compute_maps(preds[-1], batch, sup_level.points, sup_level.basis_east, sup_level.basis_north)
                tm = summarize_maps(maps, region_masks, active_thresholds)
            lr_now = opt.param_groups[0]["lr"]
            print(f"step={step:06d} loss_rad={loss.item():.6f} lr={lr_now:.2e} "
                  f"train_global={tm.get('global_geo_deg', float('nan')):.4f}", flush=True)
            model.train()

        if args.eval_every and step % args.eval_every == 0 and step != total_steps:
            metrics = evaluate(model, val_loader, geom, sup_level, region_masks, active_thresholds,
                               device, args.eval_iters, args.max_val_pairs)
            print_metrics(f"val@{step}", metrics)
            model.train()

    metrics = evaluate(model, val_loader, geom, sup_level, region_masks, active_thresholds,
                       device, args.eval_iters, args.max_val_pairs)
    metrics["elapsed_s"] = time.time() - start
    print_metrics("validation", metrics)
    print(f"elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    meta = {
        "args": vars(args), "git_hash": git_hash(), "params": n_params,
        "nodes": sup_level.num_nodes, "train_sources": train_sources, "val_sources": val_sources,
    }
    ckpt = args.checkpoint_out or str(Path(args.output_dir) / "oslo_raft.pt")
    torch.save({"model": model.state_dict(), **meta, "metrics": metrics}, ckpt)
    print(f"saved_checkpoint={ckpt}", flush=True)

    metrics_out = args.metrics_out or str(Path(args.output_dir) / "oslo_raft_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as fh:
        json.dump({**meta, "metrics": metrics}, fh, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
