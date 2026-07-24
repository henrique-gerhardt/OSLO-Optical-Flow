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
from spherical_flow.metrics import (
    accumulate_maps,
    build_region_masks,
    compute_maps,
    finalize_metrics,
    parse_thresholds,
    print_metrics,
    target_sample_from_maps,
)
from spherical_flow.princeton_raft import load_princeton_checkpoint
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
    parser.add_argument("--infer-size", default="",
                        help="Optional HxW (e.g. 320x640) to run princeton inference at; flow is "
                             "resampled+rescaled back to shard resolution. Empty = shard resolution.")
    parser.add_argument("--flow-transform", default="identity", choices=FLOW_TRANSFORMS,
                        help="Transform applied to RAFT pixel-flow before spherical evaluation.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--active-thresholds-deg", default="0.25,0.5,1.0")
    parser.add_argument("--output-dir", default="/outputs/raft_shard_baseline")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_source_records(shards_dir: Path, sources: List[Tuple[str, str]]):
    iter_shard, list_shards = _import_sfprep()
    shards = []
    for dataset, split in sources:
        found = list_shards(shards_dir, dataset, split)
        if not found:
            raise FileNotFoundError(f"no shards for {dataset}:{split} under {shards_dir}")
        shards.extend(found)
    for shard in shards:
        yield from iter_shard(shard)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_sources(args.sources)
    active_thresholds = parse_thresholds(args.active_thresholds_deg)
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
    device = torch.device("cpu")
    if args.predictor == "raft":
        device = get_device(args.device)
        if args.checkpoint:
            model, princeton_meta = load_princeton_checkpoint(
                args.checkpoint, device, small=(args.model == "raft_small")
            )
            print(
                f"princeton raft checkpoint={args.checkpoint} small={princeton_meta['small']} "
                f"iters={args.iters} infer_size={infer_size or 'shard'} "
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
            if args.checkpoint:
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
        accumulate_maps(maps, region_masks, active_thresholds, totals, counts, active_counts)

    pending: List[dict] = []
    for record in tqdm(iter_source_records(Path(args.shards), sources), desc="pairs", unit="pair"):
        frame1_erp = _to_chw_free_float(record["frame1"])
        frame2_erp = _to_chw_free_float(record["frame2"])
        flow_erp = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
        valid_erp = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        item = {
            "flow_erp": flow_erp,
            "target": sample_pair_to_nodes(
                frame1_erp, frame2_erp, flow_erp, valid_erp, points, basis_east, basis_north
            ),
        }
        if args.predictor == "raft":
            height, width = flow_erp.shape[:2]
            if infer_size is None:
                require_divisible_by_8(height, width)
            item["frame1_u8"] = torch.from_numpy(
                np.ascontiguousarray(record["frame1"])).permute(2, 0, 1).contiguous()
            item["frame2_u8"] = torch.from_numpy(
                np.ascontiguousarray(record["frame2"])).permute(2, 0, 1).contiguous()
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
    print_metrics(f"{args.predictor}_validation", metrics)
    print(f"pairs={seen} elapsed_s={metrics['elapsed_s']:.1f}", flush=True)

    result = {
        "args": vars(args),
        "sources": sources,
        "model": None if args.predictor != "raft" else {
            "arch": "princeton" if args.checkpoint else "torchvision",
            "name": args.model,
            "requested_weights": args.weights,
            "weights_enum": weights_name,
            "flow_transform": args.flow_transform,
            "torchvision": torchvision_version,
            "iters": args.iters if args.checkpoint else None,
            "infer_size": list(infer_size) if (args.checkpoint and infer_size) else None,
            "princeton": princeton_meta,
        },
        "metrics": metrics,
    }
    metrics_out = output_dir / "raft_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
