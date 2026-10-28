"""ERP-pixel EPE evaluation on the sfprep shards (plan P2A).

Scores predictions in the unit the published 360-flow tables use — mean endpoint
error in ERP pixels — with the du error seam-wrapped and regions/quantiles reported
like the geodesic runners. Node-space predictions reach the raster through the
``spherical_flow.erp_readout`` interpolation readout (validated by
``run_epe_smoke.py``); RAFT is evaluated natively (its output already is ERP flow).

Predictors:
  zero        all-zero flow — G1: every ``*_epe_px`` must equal the ``*_zero_epe_px``
              columns exactly (they are computed from the same GT in the same pass)
  oracle      GT -> supervision nodes -> readout -> EPE vs the dense GT. This is the
              **grid floor**: the EPE unreachable by any model supervised at
              ``--resolution``, dominated by motion-discontinuity pixels
  raft        frozen TorchVision RAFT on the ERP frames, native EPE
  raft_nodes  the same RAFT flow pushed ERP -> nodes -> ERP first — G4: its gap to
              ``raft`` bounds what the node route costs any predictor
  oslo        an OSLO-RAFT-R checkpoint (frames sampled at the retina grid, node
              flow at the supervision grid, readout to ERP)

Example (grid floor, GPU box):

    SHARDS_HOST=../sfprep/shards \
    docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
      python run_epe_eval.py --shards /data/shards --sources replica360:val \
        --resolution 6 --predictor oracle --output-dir /outputs/epe_oracle_replica
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from spherical_flow import healpix_unit_vectors, points_to_equirectangular_pixels, tangent_basis
from spherical_flow.erp_readout import (
    accumulate_epe,
    bilinear_node_weights,
    build_pixel_region_masks,
    compute_epe_maps,
    erp_pixel_directions,
    finalize_epe,
    nodes_to_erp_flow,
    pixel_coslat,
)
from spherical_flow.panoflow_adapter import load_panoflow_checkpoint, predict_panoflow_cfe_flow
from spherical_flow.prior_adapter import load_prior_checkpoint, predict_prior_flow
from spherical_flow.raft_adapter import (
    FLOW_TRANSFORMS,
    erp_flow_to_tangent,
    load_raft_model,
    predict_raft_flow,
    require_divisible_by_8,
)
from spherical_flow.shard_dataset import _to_chw_free_float, sample_pair_to_nodes
from run_raft_shard_baseline import get_device, iter_source_records, parse_sources

NODE_PREDICTORS = ("oracle", "raft_nodes", "oslo")


# Published protocol (PriOr-Flow ``evaluate.py``, validate_FlowScape[_regions]):
# every pixel counts (no validity mask); GT du is wrapped to [-W/2, W/2), the
# difference is not; SEPE is the haversine great-circle distance between the
# predicted and GT endpoints in radians (endpoint x wrapped, y clamped to the
# raster); "Poles" are the rows outside the central 90-degree band.
def official_region_masks(height: int, width: int) -> Dict[str, torch.Tensor]:
    min_n = int(np.round((0.5 - 0.25) * height - 0.5))
    max_n = int(np.round((0.5 + 0.25) * height - 0.5))
    center = torch.zeros(height, width, dtype=torch.bool)
    center[min_n:max_n] = True
    return {"all": torch.ones(height, width, dtype=torch.bool),
            "equator": center, "poles": ~center}


def official_maps(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """pred, gt [H, W, 2] ERP pixel flow -> (EPE px, SEPE rad), both [H, W]."""
    height, width = gt.shape[:2]
    pred = pred.double()
    gt = gt.double().clone()
    gt[..., 0] = torch.remainder(gt[..., 0] + width / 2, width) - width / 2
    epe = (pred - gt).norm(dim=-1)
    xx = torch.arange(width, dtype=torch.float64).view(1, -1).expand(height, width)
    yy = torch.arange(height, dtype=torch.float64).view(-1, 1).expand(height, width)

    def sphere(flow):
        x = torch.remainder(xx + flow[..., 0] + 0.5, width) - 0.5
        y = (yy + flow[..., 1]).clamp(-0.5, height - 0.5)
        lon = (x + 0.5) / width * 2.0 * math.pi
        lat = (0.5 - (y + 0.5) / height) * math.pi
        return lon, lat

    lon_p, lat_p = sphere(pred)
    lon_g, lat_g = sphere(gt)
    hav = torch.sin((lat_g - lat_p) / 2) ** 2 + \
        torch.cos(lat_p) * torch.cos(lat_g) * torch.sin((lon_g - lon_p) / 2) ** 2
    sepe = 2.0 * torch.arcsin(hav.clamp(0.0, 1.0).sqrt())
    return epe, sepe


# Bins of GT great-circle displacement (degrees) for the per-magnitude breakdown. OSLO's
# search reach is rings x the coarsest corr level's node spacing: 29 / 59 / 117 degrees
# for 1 / 2 / 4 rings at r4..r1. RAFT's is 4 cells x 64 px = 90 degrees at 1024 px width.
DISP_BINS_DEG = (0, 5, 15, 30, 60, 90, 180)


def displacement_bin_masks(gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """gt [H, W, 2] ERP pixel flow -> {"disp{lo}-{hi}": mask} by GT angular displacement."""
    _, disp = official_maps(torch.zeros_like(gt), gt)
    deg = torch.rad2deg(disp)
    return {f"disp{lo}-{hi}": (deg >= lo) & (deg < hi)
            for lo, hi in zip(DISP_BINS_DEG[:-1], DISP_BINS_DEG[1:])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERP-pixel EPE evaluation on sfprep shards.")
    parser.add_argument("--shards", default="/data/shards")
    parser.add_argument("--sources", default="replica360:val")
    parser.add_argument("--resolution", type=int, default=6,
                        help="Supervision HEALPix order (node grid of the readout).")
    parser.add_argument("--predictor", default="oracle",
                        choices=["zero", "oracle", "raft", "raft_nodes", "oslo"])
    parser.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    parser.add_argument("--weights", default="default", choices=["default", "none"])
    parser.add_argument("--flow-transform", default="identity", choices=FLOW_TRANSFORMS)
    parser.add_argument("--infer-size", default="",
                        help="raft predictor (TorchVision): run the net at HxW, flow resampled "
                             "back to shard size (for checkpoints trained with --resize).")
    parser.add_argument("--tv-checkpoint", default="",
                        help="raft predictor: a run_erp_raft_train.py checkpoint (raft_erp.pt)")
    parser.add_argument("--prior-checkpoint", default="",
                        help="raft predictor: a released PriOr-RAFT checkpoint")
    parser.add_argument("--prior-eval-iters", type=int, default=12)
    parser.add_argument("--panoflow-checkpoint", default="",
                        help="raft predictor: a released PanoFlow(CSFlow) checkpoint (CFE)")
    parser.add_argument("--panoflow-eval-iters", type=int, default=12)
    parser.add_argument("--init-checkpoint", default="", help="oslo predictor checkpoint (.pt)")
    parser.add_argument("--grid", default="healpix", choices=["healpix", "equiangular"])
    parser.add_argument("--retina-resolution", type=int, default=7)
    parser.add_argument("--estimation-resolution", type=int, default=4)
    parser.add_argument("--corr-pool-levels", type=int, default=3)
    parser.add_argument("--conv-neighbors", type=int, default=8)
    parser.add_argument("--neighbor-sort-cut-deg", type=float, default=None)
    parser.add_argument("--fill-missing-neighbors", action="store_true")
    parser.add_argument("--lookup-neighbors", type=int, default=24)
    parser.add_argument("--lookup-rings", type=int, default=2)
    parser.add_argument("--lookup-ring-points", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--eval-iters", type=int, default=12)
    parser.add_argument("--pyramid-cache", default="/outputs/pyramid_cache")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default="/outputs/epe_eval")
    return parser.parse_args()


def _parse_size(text: str):
    if not text:
        return None
    h, w = (int(x) for x in text.lower().split("x"))
    if h % 8 or w % 8:
        raise SystemExit(f"--infer-size {text} must be divisible by 8")
    return (h, w)


def build_oslo(args: argparse.Namespace, device: torch.device):
    import run_oslo_raft as oslo_runner
    from spherical_flow.oslo_raft_retina import OSLORAFTRetina

    if not args.init_checkpoint:
        raise SystemExit("--predictor oslo requires --init-checkpoint")
    if not (args.estimation_resolution < args.resolution <= args.retina_resolution):
        raise SystemExit("need estimation < resolution (supervision) <= retina-resolution")
    pyramid = oslo_runner.load_or_build_pyramid(args).to(device)
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    # Architecture flags recorded by the training run win over this script's own
    # defaults; checkpoints that predate a flag fall back to its default.
    trained = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = OSLORAFTRetina(
        pyramid,
        hidden_channels=trained.get("hidden_channels", args.hidden_channels),
        context_dim=trained.get("context_dim", args.context_dim),
        flow_scale=trained.get("flow_scale", args.flow_scale),
        feature_channels=None, context_channels=None,
        lookup_rings=trained.get("lookup_rings", args.lookup_rings),
        lookup_ring_points=trained.get("lookup_ring_points", args.lookup_ring_points),
        use_checkpoint_encoder=True,
        interp_mode=trained.get("lookup_interp", "idw"),
        flow_unit=(
            math.sqrt(4.0 * math.pi / pyramid.num_estimation_nodes)
            if trained.get("flow_unit", "rad") == "node" else 1.0
        ),
        corr_norm=trained.get("corr_norm", "cosine"),
        update_block=trained.get("update_block", "small"),
        encoder_blocks_per_level=trained.get("encoder_blocks_per_level", 1),
        encoder_est_blocks=trained.get("encoder_est_blocks", 0),
    ).to(device)
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.ablate_corr = False
    model.ablate_context = False
    model.eval()
    print(f"oslo checkpoint loaded: {args.init_checkpoint} "
          f"(ret={args.retina_resolution} est={args.estimation_resolution} sup={args.resolution})",
          flush=True)
    return model, pyramid


# Grid and pyramid settings an OSLO checkpoint records; they override this script's
# defaults so every checkpoint is scored on the grids it was trained with.
CHECKPOINT_GRID_ARGS = ("grid", "retina_resolution", "estimation_resolution", "resolution",
                        "corr_pool_levels", "conv_neighbors", "lookup_neighbors",
                        "neighbor_sort_cut_deg", "fill_missing_neighbors")


def adopt_checkpoint_grids(args: argparse.Namespace) -> None:
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    trained = payload.get("args", {}) if isinstance(payload, dict) else {}
    for key in CHECKPOINT_GRID_ARGS:
        if key in trained and trained[key] != getattr(args, key):
            print(f"checkpoint sets {key}={trained[key]} (was {getattr(args, key)})", flush=True)
            setattr(args, key, trained[key])


def main() -> None:
    args = parse_args()
    if args.predictor == "oslo" and args.init_checkpoint:
        adopt_checkpoint_grids(args)
        if args.grid != "healpix":
            raise SystemExit("the ERP readout is built for HEALPix nodes; equiangular "
                             "checkpoints are not supported here")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_sources(args.sources)
    device = get_device(args.device) if args.predictor in ("raft", "raft_nodes", "oslo") \
        else torch.device("cpu")

    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)

    raft = raft_transforms = weights_name = torchvision_version = None
    external = [x for x in (args.tv_checkpoint, args.prior_checkpoint, args.panoflow_checkpoint) if x]
    if len(external) > 1:
        raise SystemExit("--tv-checkpoint, --prior-checkpoint and --panoflow-checkpoint "
                         "are mutually exclusive")
    if external and args.predictor != "raft":
        raise SystemExit("checkpoint flags need --predictor raft")
    if args.prior_checkpoint:
        raft, meta = load_prior_checkpoint(args.prior_checkpoint, device,
                                           eval_iters=args.prior_eval_iters)
        weights_name, torchvision_version = f"prior:{args.prior_checkpoint}", None
        print(f"prior_raft loaded {meta}", flush=True)
    elif args.panoflow_checkpoint:
        raft, meta = load_panoflow_checkpoint(args.panoflow_checkpoint, device,
                                              eval_iters=args.panoflow_eval_iters)
        weights_name, torchvision_version = f"panoflow:{args.panoflow_checkpoint}", None
        print(f"panoflow(csflow) loaded {meta}", flush=True)
    elif args.predictor in ("raft", "raft_nodes"):
        raft, raft_transforms, weights_name, torchvision_version = load_raft_model(
            args.model, "none" if args.tv_checkpoint else args.weights, device
        )
        if args.tv_checkpoint:
            payload = torch.load(args.tv_checkpoint, map_location=device)
            raft.load_state_dict(payload["model"])
            raft.eval()
            weights_name = f"trained:{args.tv_checkpoint}"
        print(f"raft model={args.model} weights={weights_name} "
              f"flow_transform={args.flow_transform} torchvision={torchvision_version}", flush=True)
    oslo_model = pyramid = None
    retina_points = retina_be = retina_bn = None
    if args.predictor == "oslo":
        oslo_model, pyramid = build_oslo(args, device)
        retina_points = pyramid.retina_level.points.cpu()
        retina_be, retina_bn = tangent_basis(retina_points)
    print(f"predictor={args.predictor} sources={sources} resolution={args.resolution} "
          f"nodes={points.shape[0]} device={device}", flush=True)

    # Per-raster caches: readout stencil, pixel dirs/masks/weights, node pixel coords.
    stencils: Dict[Tuple[int, int], dict] = {}

    def raster(height: int, width: int) -> dict:
        key = (height, width)
        if key not in stencils:
            stencils[key] = {
                "weights": bilinear_node_weights(args.resolution, height, width),
                "dirs": erp_pixel_directions(height, width),
                "masks": build_pixel_region_masks(height, width),
                "coslat": pixel_coslat(height, width),
                "uv": points_to_equirectangular_pixels(points, height, width),
            }
        return stencils[key]

    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    official_totals: Dict[str, float] = {}
    official_counts: Dict[str, float] = {}
    sample_chunks: List[torch.Tensor] = []
    seen = 0
    start_time = time.time()

    def readout(node_flow: torch.Tensor, cache: dict, height: int, width: int) -> torch.Tensor:
        return nodes_to_erp_flow(
            node_flow.cpu(), points, basis_east, basis_north, height, width,
            cache["weights"], pixel_dirs=cache["dirs"],
        )

    def process(items: List[dict]) -> None:
        height, width = items[0]["flow_erp"].shape[:2]
        cache = raster(height, width)
        u, v = cache["uv"]

        if args.predictor == "zero":
            preds = [torch.zeros_like(it["flow_erp"]) for it in items]
        elif args.predictor == "oracle":
            preds = []
            for it in items:
                node_gt = sample_pair_to_nodes(
                    it["frame1_f"], it["frame2_f"], it["flow_erp"], it["valid_erp"],
                    points, basis_east, basis_north,
                )["flow"]
                preds.append(readout(node_gt, cache, height, width))
        elif args.predictor in ("raft", "raft_nodes"):
            frames1 = torch.stack([it["frame1_u8"] for it in items], dim=0)
            frames2 = torch.stack([it["frame2_u8"] for it in items], dim=0)
            if args.prior_checkpoint:
                raft_flow = predict_prior_flow(raft, frames1, frames2, device,
                                               args.flow_transform,
                                               eval_iters=args.prior_eval_iters)
            elif args.panoflow_checkpoint:
                raft_flow = predict_panoflow_cfe_flow(raft, frames1, frames2, device,
                                                      args.flow_transform)
            else:
                raft_flow = predict_raft_flow(raft, raft_transforms, frames1, frames2,
                                              device, args.flow_transform,
                                              _parse_size(args.infer_size))
            raft_flow = raft_flow.detach().float().cpu()
            preds = []
            for i in range(len(items)):
                erp = raft_flow[i].permute(1, 2, 0).contiguous()
                if args.predictor == "raft_nodes":
                    node_flow = erp_flow_to_tangent(
                        erp, points, basis_east, basis_north, u, v, height, width
                    )
                    erp = readout(node_flow, cache, height, width)
                preds.append(erp)
        else:  # oslo
            frame1 = torch.stack([it["nodes"]["frame1"] for it in items], dim=0).to(device)
            frame2 = torch.stack([it["nodes"]["frame2"] for it in items], dim=0).to(device)
            with torch.no_grad():
                node_pred = oslo_model(frame1, frame2, pyramid, iters=args.eval_iters)[-1]
            preds = [readout(node_pred[i].float(), cache, height, width)
                     for i in range(len(items))]

        if "official" not in cache:
            cache["official"] = official_region_masks(height, width)
        for it, pred in zip(items, preds):
            maps = compute_epe_maps(pred, it["flow_erp"], it["valid_erp"], width)
            accumulate_epe(maps, cache["masks"], cache["coslat"], totals, counts, sample_chunks)
            epe, sepe = official_maps(pred.cpu(), it["flow_erp"])
            regions = dict(cache["official"])
            regions.update(displacement_bin_masks(it["flow_erp"]))
            for region, mask in regions.items():
                for name, vals in (("epe", epe), ("sepe", sepe)):
                    key = f"{region}_{name}"
                    official_totals[key] = official_totals.get(key, 0.0) + vals[mask].sum().item()
                    official_counts[key] = official_counts.get(key, 0.0) + mask.sum().item()

    pending: List[dict] = []
    for record in tqdm(iter_source_records(Path(args.shards), sources), desc="pairs", unit="pair"):
        flow_erp = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
        valid_erp = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        item = {"flow_erp": flow_erp, "valid_erp": valid_erp}
        if args.predictor == "oracle":
            item["frame1_f"] = _to_chw_free_float(record["frame1"])
            item["frame2_f"] = _to_chw_free_float(record["frame2"])
        if args.predictor in ("raft", "raft_nodes"):
            require_divisible_by_8(*flow_erp.shape[:2])
            item["frame1_u8"] = torch.from_numpy(
                np.ascontiguousarray(record["frame1"])).permute(2, 0, 1).contiguous()
            item["frame2_u8"] = torch.from_numpy(
                np.ascontiguousarray(record["frame2"])).permute(2, 0, 1).contiguous()
        if args.predictor == "oslo":
            item["nodes"] = sample_pair_to_nodes(
                _to_chw_free_float(record["frame1"]), _to_chw_free_float(record["frame2"]),
                flow_erp, valid_erp, retina_points, retina_be, retina_bn,
                target_points=points,
            )
        if pending and pending[0]["flow_erp"].shape != flow_erp.shape:
            process(pending)
            pending = []
        pending.append(item)
        if len(pending) >= args.batch_size:
            process(pending)
            pending = []
        seen += 1
        if args.max_pairs is not None and seen >= args.max_pairs:
            break
    if pending:
        process(pending)

    if seen == 0:
        raise RuntimeError("no pairs evaluated")
    metrics = finalize_epe(totals, counts, sample_chunks)
    for key, total in official_totals.items():
        if official_counts[key] > 0:
            metrics[f"official_{key}"] = total / official_counts[key]
    all_px = official_counts.get("all_epe", 0.0)
    for key, count in official_counts.items():
        if key.startswith("disp") and key.endswith("_epe") and all_px > 0:
            metrics[f"official_{key[:-4]}_fraction"] = count / all_px
    metrics["elapsed_s"] = time.time() - start_time
    metrics["pairs"] = seen
    print(f"{args.predictor}_epe " + json.dumps(
        {k: round(vv, 6) for k, vv in metrics.items()}, sort_keys=True), flush=True)

    result = {
        "args": vars(args),
        "sources": sources,
        "model": None if args.predictor not in ("raft", "raft_nodes") else {
            "name": args.model, "requested_weights": args.weights,
            "weights_enum": weights_name, "flow_transform": args.flow_transform,
            "torchvision": torchvision_version,
        },
        "metrics": metrics,
        "official_protocol": "PriOr-Flow evaluate.py: all pixels, GT du wrapped, "
                             "SEPE = haversine endpoint distance (rad), poles = |lat| > 45 deg",
    }
    metrics_out = output_dir / "epe_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
