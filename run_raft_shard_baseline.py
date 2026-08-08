"""Frozen TorchVision RAFT head-to-head on the sfprep shards (spherical metrics).

Runs a pretrained perspective RAFT directly on the shards' ERP frames and scores it
with exactly the metric pipeline used by ``run_oslo_raft.py`` — same supervision grid,
region masks, active thresholds, and zero-flow baselines — so its metrics JSON is
row-for-row comparable with an OSLO-RAFT(-R) eval on the same source/split. This is
the published-pipeline proxy: 360° flow via an unchanged pixel-raster RAFT on the ERP
projection. Note the asymmetry both ways when comparing: RAFT is zero-shot on this
data, but carries orders of magnitude more pretraining than an OSLO-RAFT run.

Predictors:
  raft    frozen TorchVision RAFT (default)
  oracle  the GT flow itself as the prediction — validates the whole shard -> ERP ->
          sphere conversion + metric path (expect geo error at fp roundtrip level,
          and the zero-flow/quantile columns must reproduce a run_oslo_raft eval of
          the same source/split)
  zero    all-zero ERP flow — must land exactly on the *_zero_geo_deg columns

Flow convention: shard flow is canonical ``[du_x, dv_y]`` (frame1 -> frame2) and the
GT endpoint is built as ``(u + du, v + dv)`` — the same construction
``erp_flow_to_tangent`` applies to predictions. TorchVision RAFT predicts ``(dx, dy)``
in those pixel axes, so ``--flow-transform identity`` is the default (FLOW360's native
files needed ``negated``; the materializer already canonicalized the shards). If in
doubt, ``--predictor raft --max-pairs 8`` swept over transforms settles it: the right
one wins by a wide margin.

Example (GPU box, from the repo root; TORCH_HOME persists the RAFT weights download):

    SHARDS_HOST=../sfprep/shards \
    docker compose -f docker-compose.oslo_raft.yml run --rm \
      -e TORCH_HOME=/outputs/torch_home oslo-raft \
      python run_raft_shard_baseline.py \
        --shards /data/shards --sources replica360:val --resolution 6 \
        --device cuda --output-dir /outputs/raft_erp_replica360_val
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from spherical_flow import (
    healpix_unit_vectors,
    points_to_equirectangular_pixels,
    tangent_basis,
)
from spherical_flow.geometry import set_geodesic_mode
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
from spherical_flow.princeton_raft import load_princeton_checkpoint
from spherical_flow.panoflow_adapter import (
    load_panoflow_checkpoint,
    predict_panoflow_cfe_flow,
)
from spherical_flow.raft_adapter import (
    FLOW_TRANSFORMS,
    erp_flow_to_tangent,
    load_raft_model,
    predict_princeton_flow,
    predict_raft_flow,
    require_divisible_by_8,
)
from spherical_flow.shard_dataset import (
    _import_sfprep,
    _to_chw_free_float,
    sample_pair_to_nodes,
)
from spherical_flow.so3_augment import sample_rotation, so3_augment_pair
from spherical_flow.geometry import equirectangular_pixels_to_unit_vectors
from spherical_flow.flow360 import bilinear_sample_erp


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
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen ERP RAFT baseline on sfprep shards with spherical metrics."
    )
    parser.add_argument("--shards", default="/data/shards", help="Directory with sfprep shard tars.")
    parser.add_argument("--sources", default="replica360:val", help="Comma-separated dataset:split list.")
    parser.add_argument("--resolution", type=int, default=6, help="HEALPix order for spherical evaluation.")
    parser.add_argument("--predictor", default="raft", choices=["raft", "oracle", "zero"])
    parser.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    parser.add_argument("--weights", default="default", choices=["default", "none"])
    parser.add_argument("--checkpoint", default="",
                        help="Path to a princeton-tree RAFT checkpoint (SLOF etc.). When set, "
                             "the vendored princeton RAFT replaces the TorchVision model; "
                             "--model picks large/small, --weights is ignored.")
    parser.add_argument("--iters", type=int, default=32,
                        help="Refinement iterations for the princeton path (SLOF evals used 64).")
    parser.add_argument("--princeton-input-scale", default="byte", choices=["byte", "unit"],
                        help="Frame range fed to a --checkpoint model. 'byte' is 0-255, which is "
                             "what the forward's 2*(x/255)-1 normalization expects. 'unit' is "
                             "0-1, which is what SLOF's own loader produces (ToTensor) and "
                             "therefore the range their checkpoints were trained and published "
                             "under; see run_slof_table1.py. Ignored for torchvision/PanoFlow.")
    parser.add_argument("--panoflow-checkpoint", default="",
                        help="Path to a PanoFlow(CSFlow) checkpoint. When set, the vendored "
                             "PanoFlow(CSFlow) net runs under CFE at native shard resolution "
                             "(its own eval protocol); --checkpoint/--model/--infer-size ignored.")
    parser.add_argument("--panoflow-eval-iters", type=int, default=12,
                        help="GRU eval iterations for the PanoFlow path (repo default 12).")
    parser.add_argument("--infer-size", default="",
                        help="Optional HxW (e.g. 320x640) to run princeton inference at; flow is "
                             "resampled+rescaled back to shard resolution. Empty = shard resolution.")
    parser.add_argument("--flow-transform", default="identity", choices=FLOW_TRANSFORMS,
                        help="Transform applied to RAFT pixel-flow before spherical evaluation.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument("--motion-bands-deg", default="",
                        help="A4: comma-separated edges for DISJOINT motion bands, e.g. "
                             "'0,0.125,0.25,0.5,1,2,4,8,16,inf'. Unlike the cumulative active_X "
                             "tails, bands locate the displacement at which a method starts "
                             "beating the zero baseline. Empty = off (default).")
    parser.add_argument("--directions", default="both",
                        choices=["both", "forward", "backward"],
                        help="Restrict to one flow direction. Needed because a dataset can carry "
                             "a different sign convention per direction, which no statistic of the "
                             "GT alone can reveal (the zero baseline is sign-invariant).")
    parser.add_argument("--gt-transform", default="identity",
                        choices=["identity", "negated", "negate_x", "negate_y"],
                        help="Sign/axis convention applied to the GT before scoring. Diagnostic: "
                             "the convention a strong independently-trained predictor prefers is "
                             "the physical one. Reproducing a paper's numbers does NOT settle "
                             "this, since a model trained against a reversed target reproduces "
                             "its own table exactly.")
    parser.add_argument("--geodesic-metric", default="acos", choices=["acos", "haversine"],
                        help="Great-circle formula. 'acos' reproduces every existing number and "
                             "has a 0.028 deg float32 floor; 'haversine' is exact at zero.")
    parser.add_argument("--val-so3-prob", type=float, default=0.0,
                        help="Probability of rotating each pair by a random SO(3) element before "
                             "prediction. Measures robustness to camera orientation: the ERP has a "
                             "privileged axis and a sphere-native grid does not, so a rotated scene "
                             "is ordinary for one and out-of-distribution for the other. The "
                             "predictor receives a re-rendered ERP, which costs exactly the one "
                             "bilinear resampling a node-sampling model also pays, so the "
                             "comparison carries no interpolation advantage either way.")
    parser.add_argument("--val-so3-max-angle-deg", type=float, default=180.0)
    parser.add_argument("--val-so3-uniform", action="store_true",
                        help="Draw from the Haar measure on SO(3) instead of the axis-uniform, "
                             "angle-uniform schedule the training augmentation uses.")
    parser.add_argument("--val-so3-seed", type=int, default=1234,
                        help="Rotations come from a dedicated generator advanced once per pair, so "
                             "every predictor run at this seed sees the same rotation sequence.")
    parser.add_argument("--output-dir", default="/outputs/raft_shard_baseline")
    return parser.parse_args()


def rotate_erp(frame: torch.Tensor, rotation: torch.Tensor,
               grid: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Re-render an ERP frame as seen by a camera rotated by ``R``.

    The convention matches :func:`so3_augment_pair`, which samples the real frames
    at ``p @ R``: the rotated raster at direction ``d`` shows whatever the real
    raster holds at ``d @ R``. Reading the rotated raster at an unrotated node
    therefore returns exactly what the node-sampling path returns, so the target
    that ``so3_augment_pair`` produces scores the two without any further
    bookkeeping.
    """
    height, width = frame.shape[:2]
    directions, _ = grid
    u, v = points_to_equirectangular_pixels(directions @ rotation, height, width)
    return bilinear_sample_erp(frame, u, v).reshape(height, width, frame.shape[2])


def erp_direction_grid(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    v, u = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                          torch.arange(width, dtype=torch.float32), indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)
    return equirectangular_pixels_to_unit_vectors(u, v, height, width), u


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


GT_TRANSFORMS = {
    "identity": lambda f: f,
    "negated": lambda f: -f,
    "negate_x": lambda f: torch.stack([-f[..., 0], f[..., 1]], dim=-1),
    "negate_y": lambda f: torch.stack([f[..., 0], -f[..., 1]], dim=-1),
}


def iter_source_records(shards_dir: Path, sources: List[Tuple[str, str]],
                        directions: str = "both"):
    iter_shard, list_shards = _import_sfprep()
    shards = []
    for dataset, split in sources:
        found = list_shards(shards_dir, dataset, split)
        if not found:
            raise FileNotFoundError(f"no shards for {dataset}:{split} under {shards_dir}")
        shards.extend(found)
    for shard in shards:
        for record in iter_shard(shard):
            if directions != "both" and record["meta"].get("direction") != directions:
                continue
            yield record


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_sources(args.sources)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)
    motion_bands = parse_bands(args.motion_bands_deg)
    set_geodesic_mode(args.geodesic_metric)
    points = healpix_unit_vectors(args.resolution)
    basis_east, basis_north = tangent_basis(points)
    region_masks = build_region_masks(points)

    infer_size = None
    if args.infer_size:
        try:
            h_txt, w_txt = args.infer_size.lower().split("x")
            infer_size = (int(h_txt), int(w_txt))
        except ValueError as exc:
            raise ValueError(f"--infer-size must be HxW, got '{args.infer_size}'") from exc
        require_divisible_by_8(*infer_size)

    model = transforms = None
    weights_name = torchvision_version = None
    princeton_meta = None
    panoflow_meta = None
    device = torch.device("cpu")
    if args.predictor == "raft":
        device = get_device(args.device)
        if args.panoflow_checkpoint:
            model, panoflow_meta = load_panoflow_checkpoint(
                args.panoflow_checkpoint, device, eval_iters=args.panoflow_eval_iters
            )
            print(
                f"panoflow(csflow) checkpoint={args.panoflow_checkpoint} "
                f"eval_iters={panoflow_meta['eval_iters']} resolution=native+CFE "
                f"flow_transform={args.flow_transform} device={device} "
                f"loaded_keys={panoflow_meta['loaded_keys']} "
                f"unexpected_keys={panoflow_meta['unexpected_keys']} "
                f"unexpected_sample={panoflow_meta['unexpected_key_sample']}",
                flush=True,
            )
        elif args.checkpoint:
            model, princeton_meta = load_princeton_checkpoint(
                args.checkpoint, device, small=(args.model == "raft_small")
            )
            print(
                f"princeton raft checkpoint={args.checkpoint} small={princeton_meta['small']} "
                f"iters={args.iters} input_scale={args.princeton_input_scale} "
                f"infer_size={infer_size or 'shard'} "
                f"flow_transform={args.flow_transform} device={device} "
                f"loaded_keys={princeton_meta['loaded_keys']} "
                f"unexpected_keys={princeton_meta['unexpected_keys']} "
                f"unexpected_sample={princeton_meta['unexpected_key_sample']}",
                flush=True,
            )
        else:
            model, transforms, weights_name, torchvision_version = load_raft_model(
                args.model, args.weights, device
            )
            print(
                f"raft model={args.model} weights={weights_name} flow_transform={args.flow_transform} "
                f"device={device} batch_size={args.batch_size} torchvision={torchvision_version}",
                flush=True,
            )
    print(f"predictor={args.predictor} sources={sources} resolution={args.resolution} "
          f"nodes={points.shape[0]}", flush=True)

    totals: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    active_counts: Dict[str, float] = {}
    target_chunks: List[torch.Tensor] = []
    pixel_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    direction_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    so3_gen = torch.Generator().manual_seed(args.val_so3_seed)
    if args.val_so3_prob > 0.0 and args.predictor == "oracle":
        raise ValueError("--predictor oracle scores the unrotated GT raster and cannot be "
                         "combined with --val-so3-prob")
    seen = 0
    start_time = time.time()

    def process(items: List[dict]) -> None:
        height, width = items[0]["flow_erp"].shape[:2]
        if (height, width) not in pixel_cache:
            pixel_cache[(height, width)] = points_to_equirectangular_pixels(points, height, width)
        u, v = pixel_cache[(height, width)]

        if args.predictor == "raft":
            frames1 = torch.stack([it["frame1_u8"] for it in items], dim=0)
            frames2 = torch.stack([it["frame2_u8"] for it in items], dim=0)
            if args.panoflow_checkpoint:
                raft_flow = predict_panoflow_cfe_flow(
                    model, frames1, frames2, device, args.flow_transform,
                )
            elif args.checkpoint:
                if args.princeton_input_scale == "unit":
                    frames1 = frames1.float() / 255.0
                    frames2 = frames2.float() / 255.0
                raft_flow = predict_princeton_flow(
                    model, frames1, frames2, device, args.flow_transform,
                    args.iters, infer_size,
                )
            else:
                raft_flow = predict_raft_flow(
                    model, transforms, frames1, frames2, device, args.flow_transform,
                )
            pred_erp = [raft_flow[i].permute(1, 2, 0).contiguous() for i in range(len(items))]
        elif args.predictor == "oracle":
            pred_erp = [it["flow_erp"] for it in items]
        else:  # zero
            pred_erp = [torch.zeros_like(it["flow_erp"]) for it in items]

        pred_flow = torch.stack(
            [erp_flow_to_tangent(f, points, basis_east, basis_north, u, v, height, width)
             for f in pred_erp],
            dim=0,
        )
        target_batch = {
            "flow": torch.stack([it["target"]["flow"] for it in items], dim=0),
            "endpoint": torch.stack([it["target"]["endpoint"] for it in items], dim=0),
            "valid": torch.stack([it["target"]["valid"] for it in items], dim=0),
        }
        maps = compute_maps(pred_flow, target_batch, points, basis_east, basis_north)
        target_chunks.append(target_sample_from_maps(maps, None))
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts,
                        motion_bands=motion_bands)

    pending: List[dict] = []
    for record in tqdm(iter_source_records(Path(args.shards), sources, args.directions),
                       desc="pairs", unit="pair"):
        frame1_erp = _to_chw_free_float(record["frame1"])
        frame2_erp = _to_chw_free_float(record["frame2"])
        flow_erp = GT_TRANSFORMS[args.gt_transform](
            torch.from_numpy(np.ascontiguousarray(record["flow"])).float())
        valid_erp = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        height, width = flow_erp.shape[:2]

        rotation = None
        if args.val_so3_prob > 0.0 and float(
                torch.rand((), generator=so3_gen)) < args.val_so3_prob:
            rotation = sample_rotation(so3_gen, max_angle_deg=args.val_so3_max_angle_deg,
                                       uniform_so3=args.val_so3_uniform)
        if rotation is None:
            target = sample_pair_to_nodes(frame1_erp, frame2_erp, flow_erp, valid_erp,
                                          points, basis_east, basis_north)
        else:
            target = so3_augment_pair(frame1_erp, frame2_erp, flow_erp, valid_erp,
                                      points, rotation, basis_east, basis_north)
            if (height, width) not in direction_cache:
                direction_cache[(height, width)] = erp_direction_grid(height, width)
            frame1_erp = rotate_erp(frame1_erp, rotation, direction_cache[(height, width)])
            frame2_erp = rotate_erp(frame2_erp, rotation, direction_cache[(height, width)])
        item = {"flow_erp": flow_erp, "target": target}
        if args.predictor == "raft":
            if infer_size is None and not args.panoflow_checkpoint:
                require_divisible_by_8(height, width)
            if rotation is None:                    # keep the byte-exact original path
                f1 = torch.from_numpy(np.ascontiguousarray(record["frame1"]))
                f2 = torch.from_numpy(np.ascontiguousarray(record["frame2"]))
            else:
                f1 = frame1_erp.mul(255.0).round().clamp(0, 255).to(torch.uint8)
                f2 = frame2_erp.mul(255.0).round().clamp(0, 255).to(torch.uint8)
            item["frame1_u8"] = f1.permute(2, 0, 1).contiguous()
            item["frame2_u8"] = f2.permute(2, 0, 1).contiguous()
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = finalize_metrics(totals, counts, active_counts, target_chunks)
    metrics["elapsed_s"] = time.time() - start_time
    metrics["pairs"] = seen
    print_metrics(f"{args.predictor}_validation", metrics, motion_bands)
    print(f"pairs={seen} elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    result = {
        "args": vars(args),
        "sources": sources,
        "model": None if args.predictor != "raft" else {
            "arch": (
                "panoflow_csflow" if args.panoflow_checkpoint
                else "princeton" if args.checkpoint
                else "torchvision"
            ),
            "name": args.model,
            "requested_weights": args.weights,
            "weights_enum": weights_name,
            "flow_transform": args.flow_transform,
            "torchvision": torchvision_version,
            "iters": args.iters if args.checkpoint else None,
            "input_scale": args.princeton_input_scale if args.checkpoint else None,
            "infer_size": list(infer_size) if (args.checkpoint and infer_size) else None,
            "princeton": princeton_meta,
            "panoflow": panoflow_meta,
        },
        "metrics": metrics,
    }
    metrics_out = output_dir / "raft_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
