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

    # OSLO-RAFT-R Stage A matching bootstrap (docs/OSLO_RAFT_RETINA_PLAN.md §8):
    python run_oslo_raft.py --grid healpix --retina --retina-resolution 7 \
        --resolution 6 --estimation-resolution 4 --device cuda --amp --onecycle \
        --train-sources replica360:train --val-sources replica360:val \
        --synth-rot-prob 0.5 --val-synth-rot-prob 0.5 --steps 5000
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
import torch.nn.functional as F
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
    # retina mode (OSLO-RAFT-R): ingest at --retina-resolution, estimate at
    # --estimation-resolution, supervise at --resolution (docs/OSLO_RAFT_RETINA_PLAN.md)
    p.add_argument("--retina", action="store_true",
                   help="Use OSLORAFTRetina: frames sampled at --retina-resolution, flow estimated "
                        "at --estimation-resolution, supervised at --resolution via the convex "
                        "upsampler, with the lazy interpolated correlation lookup. Requires "
                        "--grid healpix and est < resolution <= retina.")
    p.add_argument("--retina-resolution", type=int, default=7,
                   help="Retina (input) HEALPix order for --retina (8 = ERP-pixel parity, ~4x cost).")
    p.add_argument("--lookup-rings", type=int, default=2,
                   help="Interp-lookup stencil rings per corr level (--retina).")
    p.add_argument("--lookup-ring-points", type=int, default=8,
                   help="Interp-lookup points per stencil ring (--retina).")
    p.add_argument("--feature-channels", default="",
                   help="Comma list overriding the retina encoder channel ramp, finest->est "
                        "(e.g. '16,32,48,64,96'). Empty = the depth-keyed default.")
    p.add_argument("--pyramid-cache", default="outputs/pyramid_cache",
                   help="Directory for cached pyramids ('' disables). r7/r8 graphs take minutes "
                        "to build; the cache is keyed on the full geometry config.")
    p.add_argument("--no-encoder-checkpoint", action="store_true",
                   help="Disable encoder gradient checkpointing in --retina mode (on by default).")
    p.add_argument("--aux-match-weight", type=float, default=0.5,
                   help="Weight of the stencil matching loss (--retina only). Measured necessary "
                        "for correlation to bootstrap at all (see the retina plan §9.2 notes); "
                        "0 disables.")
    p.add_argument("--aux-warmup-steps", type=int, default=0,
                   help="Train ONLY the matching loss for the first N steps (--retina; the "
                        "Stage-A feature bootstrap). 0 = joint from step 1.")
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
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Accumulate gradients over N micro-batches per optimizer step "
                        "(retina-8 memory fallback: --batch-size 1 --grad-accum 2).")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--onecycle", action="store_true", help="OneCycle LR schedule over --steps.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Polyak/EMA weight averaging: shadow = d*shadow + (1-d)*weights after each "
                        "optimizer step (0 disables). Evals also report the shadow weights and the "
                        "EMA checkpoint is saved as oslo_raft_ema.pt (P1 consolidation: the raw "
                        "trajectory oscillates around the optimum at effective batch 8).")
    p.add_argument("--init-checkpoint", default="",
                   help="Load model weights from a saved checkpoint before training "
                        "(stage-to-stage init, e.g. Stage B from Stage A).")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training: evaluate --init-checkpoint (or a fresh model) on the val "
                        "sources and write metrics. Gate R1 = eval-only twice, with and without "
                        "--ablate-corr.")
    # augmentation
    p.add_argument("--so3-prob", type=float, default=1.0)
    p.add_argument("--so3-max-angle-deg", type=float, default=180.0)
    p.add_argument("--so3-uniform", action="store_true", help="Haar SO(3) angle density.")
    # synthetic-rotation motion source (the Stage-A matching bootstrap, plan §8)
    p.add_argument("--synth-rot-prob", type=float, default=0.0,
                   help="Probability a train record's motion is REPLACED by an exact rotation of "
                        "its own frame1 (perfect constancy, exact GT — the matching bootstrap).")
    p.add_argument("--synth-rot-min-deg", type=float, default=1.0)
    p.add_argument("--synth-rot-max-deg", type=float, default=15.0)
    p.add_argument("--val-synth-rot-prob", type=float, default=0.0,
                   help="Same knob for the val stream (Gate R1 uses a synth-rot val set).")
    p.add_argument("--synth-photo-scale", type=float, default=0.0,
                   help="Asymmetric photometric jitter on the synthetic frame 2, RAFT-parity "
                        "ranges x this scale in [0,1] (P2C nuisance axis; probe P0 sweeps it). "
                        "Applies to train and val synth records alike; 0 = bit-identical to "
                        "the pre-flag pipeline.")
    p.add_argument("--synth-photo-noise-std", type=float, default=0.0,
                   help="Per-pixel iid Gaussian noise on the synthetic frame 2, std in 1/255 "
                        "units (P0b: the spatially-unstructured nuisance axis; mean |delta| "
                        "= 0.8 x std).")
    p.add_argument("--real-resample-prob", type=float, default=0.0,
                   help="Probability a train record's frame 2 is REPLACED by frame 1 resampled "
                        "at the REAL GT endpoints (P0d: real motion structure, exact constancy; "
                        "GT untouched).")
    p.add_argument("--val-real-resample-prob", type=float, default=0.0,
                   help="Same knob for the val stream (the P0d probe uses 1.0).")
    p.add_argument("--synth-edge-corrupt-delta", type=float, default=0.0,
                   help="Edge-modulated structured corruption of the synth frame-2 raster, "
                        "target mean |delta| in 1/255 units (P0c: the measured real-nuisance "
                        "shape; real magnitude is ~3.1).")
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


def load_or_build_pyramid(args: argparse.Namespace):
    """Build the retina pyramid, disk-cached on the full geometry config (plan §3.4).

    The r7/r8 neighbor graphs take minutes to build; the cache pays that once. The
    filename keys every parameter that shapes the geometry, so a config change never
    reads a stale pyramid; a corrupt/old-version file is rebuilt, not trusted.
    """
    from spherical_flow.healpix_pyramid import build_healpix_pyramid, load_pyramid, save_pyramid

    cache_path = None
    if args.pyramid_cache:
        name = (f"pyramid_ret{args.retina_resolution}_sup{args.resolution}"
                f"_est{args.estimation_resolution}_cp{args.corr_pool_levels}"
                f"_cn{args.conv_neighbors}_ln{args.lookup_neighbors}.pt")
        cache_path = Path(args.pyramid_cache) / name
        if cache_path.exists():
            try:
                pyramid = load_pyramid(cache_path)
                print(f"pyramid cache hit: {cache_path}", flush=True)
                return pyramid
            except Exception as exc:  # version bump / partial write -> rebuild
                print(f"pyramid cache unusable ({exc}); rebuilding", flush=True)

    t0 = time.time()
    pyramid = build_healpix_pyramid(
        fine_resolution=args.resolution,
        estimation_resolution=args.estimation_resolution,
        corr_pool_levels=args.corr_pool_levels,
        conv_neighbors=args.conv_neighbors,
        lookup_neighbors=args.lookup_neighbors,
        retina_resolution=args.retina_resolution,
    )
    print(f"pyramid built in {time.time() - t0:.1f}s", flush=True)
    if cache_path is not None:
        save_pyramid(pyramid, cache_path)
        print(f"pyramid cached: {cache_path}", flush=True)
    return pyramid


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

    _model_modes = sum([args.multi_res, args.local_corr, args.differential, args.retina])
    if _model_modes > 1:
        raise SystemExit("--multi-res / --local-corr / --differential / --retina are mutually exclusive")
    if args.differential and (args.ablate_corr or args.ablate_context):
        raise SystemExit("--differential has no correlation/context path to ablate")
    if args.ablate_corr and args.multi_res:
        raise SystemExit("--ablate-corr is not supported with --multi-res (no corr ablation hook)")
    if args.ablate_context and args.multi_res:
        raise SystemExit("--ablate-context is not supported with --multi-res (no context ablation hook)")

    # OSLO-RAFT-R decouples the frame grid from the supervision grid; the dataset then
    # samples frames at the retina and targets at the fine (supervision) grid.
    target_points = None

    if args.retina:
        if args.grid != "healpix":
            raise SystemExit("--retina requires --grid healpix")
        if not (args.estimation_resolution < args.resolution <= args.retina_resolution):
            raise SystemExit(
                "--retina needs estimation < resolution (supervision) <= retina-resolution, got "
                f"est={args.estimation_resolution} sup={args.resolution} ret={args.retina_resolution}"
            )
        from spherical_flow.oslo_raft_retina import OSLORAFTRetina

        pyramid = load_or_build_pyramid(args)
        dataset_points = pyramid.retina_level.points   # frames at the retina
        target_points = pyramid.fine_level.points      # targets at the supervision grid
        pyramid = pyramid.to(device)
        geom, sup_level = pyramid, pyramid.fine_level
        feature_channels = (
            tuple(int(t) for t in args.feature_channels.split(",") if t.strip())
            or None
        )
        model = OSLORAFTRetina(
            pyramid, hidden_channels=args.hidden_channels, context_dim=args.context_dim,
            flow_scale=args.flow_scale, feature_channels=feature_channels,
            context_channels=feature_channels,
            lookup_rings=args.lookup_rings, lookup_ring_points=args.lookup_ring_points,
            use_checkpoint_encoder=not args.no_encoder_checkpoint,
        ).to(device)
        print(f"grid=healpix retina={args.retina_resolution} est={args.estimation_resolution} "
              f"sup={args.resolution} nodes_ret={pyramid.retina_level.num_nodes} "
              f"nodes_est={pyramid.num_estimation_nodes} nodes_sup={pyramid.num_fine_nodes} "
              f"device={device} git={git_hash()[:9]}", flush=True)
    elif args.differential:
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

    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
        print(f"loaded init checkpoint: {args.init_checkpoint}", flush=True)

    # Honored by OSLORAFT / OSLORAFTLocal / OSLORAFTRetina (guarded above so it is never
    # silently a no-op).
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
    if args.synth_rot_prob > 0.0 or args.val_synth_rot_prob > 0.0:
        print(f"synth-rot: train_prob={args.synth_rot_prob} val_prob={args.val_synth_rot_prob} "
              f"angle=[{args.synth_rot_min_deg}, {args.synth_rot_max_deg}] deg", flush=True)
    if args.real_resample_prob > 0.0 or args.val_real_resample_prob > 0.0:
        print(f"real-resample: train_prob={args.real_resample_prob} "
              f"val_prob={args.val_real_resample_prob} (frame2 = f1 @ real GT endpoints)",
              flush=True)
    if (args.synth_photo_scale > 0.0 or args.synth_photo_noise_std > 0.0
            or args.synth_edge_corrupt_delta > 0.0):
        print(f"synth-photo: jitter scale={args.synth_photo_scale} "
              f"noise_std={args.synth_photo_noise_std}/255 "
              f"edge_corrupt_delta={args.synth_edge_corrupt_delta}/255 on synth frame 2",
              flush=True)
    if args.retina and args.aux_match_weight > 0.0:
        print(f"aux stencil matching: weight={args.aux_match_weight} "
              f"warmup_steps={args.aux_warmup_steps}", flush=True)

    region_masks = build_region_masks(sup_level.points)

    train_ds = ShardFlowDataset(
        args.shards, dataset_points, train_sources,
        shuffle_shards=True, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        max_pairs=args.max_train_pairs,
        so3_prob=args.so3_prob, so3_max_angle_deg=args.so3_max_angle_deg, so3_uniform=args.so3_uniform,
        target_points=target_points,
        synth_rot_prob=args.synth_rot_prob,
        synth_rot_min_deg=args.synth_rot_min_deg, synth_rot_max_deg=args.synth_rot_max_deg,
        synth_photo_scale=args.synth_photo_scale,
        synth_photo_noise_std=args.synth_photo_noise_std,
        synth_edge_corrupt_delta=args.synth_edge_corrupt_delta,
        real_resample_prob=args.real_resample_prob,
    )
    val_ds = ShardFlowDataset(
        args.shards, dataset_points, val_sources, shuffle_shards=False, shuffle_buffer=0, seed=args.seed,
        target_points=target_points,
        synth_rot_prob=args.val_synth_rot_prob,
        synth_rot_min_deg=args.synth_rot_min_deg, synth_rot_max_deg=args.synth_rot_max_deg,
        synth_photo_scale=args.synth_photo_scale,
        synth_photo_noise_std=args.synth_photo_noise_std,
        synth_edge_corrupt_delta=args.synth_edge_corrupt_delta,
        real_resample_prob=args.val_real_resample_prob,
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

    # Auxiliary stencil matching loss (retina only): direct matching supervision at the
    # estimation grid — measured necessary for the correlation to bootstrap at all (the
    # retina plan §9.2 notes). Est-grid GT endpoints are the normalized mean of each est
    # node's supervision-grid descendants (exact for locally-smooth motion; the loss's
    # window mask drops the rest).
    aux_w = args.aux_match_weight if args.retina else 0.0
    if aux_w > 0.0:
        from spherical_flow.oslo_raft_retina import stencil_match_loss

    # Polyak/EMA shadow weights: averages over the oscillating raw trajectory so the
    # eval'd/saved point sits near the basin center instead of a random phase of the swing.
    ema_decay = 0.0 if args.eval_only else args.ema_decay
    ema_state = None
    if ema_decay > 0.0:
        ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        print(f"ema: decay={ema_decay} (shadow weights eval'd alongside; saved as oslo_raft_ema.pt)",
              flush=True)

    def evaluate_state(state: dict) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: s.to(backup[k].dtype) for k, s in state.items()})
        m = evaluate(model, val_loader, geom, sup_level, region_masks, active_thresholds,
                     device, args.eval_iters, args.max_val_pairs)
        model.load_state_dict(backup)
        return m

    stream = cycling_loader(train_ds, train_loader)
    start = time.time()
    model.train()
    accum = max(1, args.grad_accum)
    total_steps = 0 if args.eval_only else (1 if args.smoke_test else args.steps)
    for step in range(1, total_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        warmup_only = aux_w > 0.0 and step <= args.aux_warmup_steps
        for _ in range(accum):
            batch = move_batch(next(stream), device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                if aux_w > 0.0:
                    preds, (f1e, f2e) = model(
                        batch["frame1"], batch["frame2"], geom, iters=args.iters,
                        return_features=True,
                    )
                    est_end = F.normalize(
                        batch["endpoint"][:, geom.descendant_index].mean(dim=2), dim=-1
                    )
                    est_valid = batch["valid"][:, geom.descendant_index].float().mean(dim=2) > 0.5
                    aux = stencil_match_loss(
                        f1e, f2e, est_end, geom.estimation_level, valid=est_valid
                    )
                else:
                    preds = model(batch["frame1"], batch["frame2"], geom, iters=args.iters)
                    aux = None
                if warmup_only:
                    loss = aux / accum
                else:
                    loss = sequence_geodesic_loss(
                        preds, batch["endpoint"], sup_level, batch["valid"], gamma=args.gamma,
                        motion_weight=args.loss_motion_weight, motion_ref_deg=args.loss_motion_ref_deg,
                        min_target_deg=args.loss_min_target_deg,
                    ) / accum
                    if aux is not None:
                        loss = loss + aux_w * aux / accum
            scaler.scale(loss).backward()
            loss_total += loss.item()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    s = ema_state[k]
                    if s.dtype.is_floating_point:
                        s.mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                    else:
                        s.copy_(v)

        if step == 1 or step == total_steps or step % args.log_every == 0:
            with torch.no_grad():
                maps = compute_maps(preds[-1], batch, sup_level.points, sup_level.basis_east, sup_level.basis_north)
                tm = summarize_maps(maps, region_masks, active_thresholds)
            lr_now = opt.param_groups[0]["lr"]
            aux_txt = ""
            if aux_w > 0.0:
                aux_txt = f" aux={aux.item():.4f}{' (warmup: aux-only)' if warmup_only else ''}"
            print(f"step={step:06d} loss_rad={loss_total:.6f}{aux_txt} lr={lr_now:.2e} "
                  f"train_global={tm.get('global_geo_deg', float('nan')):.4f}", flush=True)
            model.train()

        if args.eval_every and step % args.eval_every == 0 and step != total_steps:
            metrics = evaluate(model, val_loader, geom, sup_level, region_masks, active_thresholds,
                               device, args.eval_iters, args.max_val_pairs)
            print_metrics(f"val@{step}", metrics)
            if ema_state is not None:
                print_metrics(f"val_ema@{step}", evaluate_state(ema_state))
            model.train()

    metrics = evaluate(model, val_loader, geom, sup_level, region_masks, active_thresholds,
                       device, args.eval_iters, args.max_val_pairs)
    metrics["elapsed_s"] = time.time() - start
    print_metrics("validation", metrics)
    metrics_ema = None
    if ema_state is not None:
        metrics_ema = evaluate_state(ema_state)
        print_metrics("validation_ema", metrics_ema)
    print(f"elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    meta = {
        "args": vars(args), "git_hash": git_hash(), "params": n_params,
        "nodes": sup_level.num_nodes, "train_sources": train_sources, "val_sources": val_sources,
    }
    if not args.eval_only:  # eval-only must never clobber the checkpoint it is judging
        ckpt = args.checkpoint_out or str(Path(args.output_dir) / "oslo_raft.pt")
        torch.save({"model": model.state_dict(), **meta, "metrics": metrics}, ckpt)
        print(f"saved_checkpoint={ckpt}", flush=True)
        if ema_state is not None:
            ema_ckpt = str(Path(args.output_dir) / "oslo_raft_ema.pt")
            ema_model = {k: s.detach().cpu().to(model.state_dict()[k].dtype)
                         for k, s in ema_state.items()}
            torch.save({"model": ema_model, **meta, "metrics": metrics_ema}, ema_ckpt)
            print(f"saved_checkpoint_ema={ema_ckpt}", flush=True)

    metrics_out = args.metrics_out or str(Path(args.output_dir) / "oslo_raft_metrics.json")
    payload = {**meta, "metrics": metrics}
    if metrics_ema is not None:
        payload["metrics_ema"] = metrics_ema
    with open(metrics_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
