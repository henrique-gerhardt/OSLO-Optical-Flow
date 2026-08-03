"""Reproduce SLOF's published FLOW360 table in SLOF's own metric — plus the row
they never printed.

Every universality claim in this project so far is scoped to *our* protocol:
geodesic degrees on an equal-area grid, actives conditioned on true angular
displacement. A referee is entitled to answer "your metric is not the one the
field uses". This script removes that answer. It re-implements the evaluation in
``SLOF/evaluate_raft.py`` + ``SLOF/dataloader.py`` + ``SLOF/utils.py`` line for
line, runs it on their own test split with their own checkpoints, and adds one
row: **zero flow**. If our reproduction lands on their shipped CSVs, then the
zero row is a number from their table, not from ours.

Faithful to their code, including the parts that look wrong (each is flagged in
the JSON under ``protocol``):

* pairs = forward flows only, ``sorted(seq)`` x ``sorted(frames)[:-1]`` (1089 on
  test), which is half of what our shards carry for the same split;
* frames: PIL resize to 320x640 then ``ToTensor()`` -> **[0, 1]**, fed to a
  forward that normalizes with ``2*(x/255) - 1``. Their images therefore reach
  the network at 1/255 of the intended contrast. ``--input-scale`` runs it either
  way; ``unit`` is what produced the published numbers;
* flow: negated twice (``ReadData`` then ``Flow360Loader``), so the evaluated GT
  is the raw ``.npy`` — the same convention sfprep pins for ``flow360``;
* flow resize: normalize by (w, h), ``F.interpolate`` at its **nearest** default,
  rescale by the new size;
* EPE = mean over every pixel of the split, unweighted, no validity mask;
  magnitude buckets are CUMULATIVE and overlapping (lt5 c lt10 c lt20), taken on
  the resized GT;
* EPEd (``--weighted``) divides the per-pixel error by their density map, which
  is ~0.5 at the poles and ~1.0 at the equator, i.e. it **doubles** polar error;
* AE reproduces two defects: ``ugt`` is normalized before being reused to
  normalize ``vgt``, and the result is stretched to [-1, 1] by per-batch
  min/max — so AE depends on batch composition. Batch 16 = theirs.

Published reference values are embedded from the CSVs shipped in their release
(``quantitative_results/``), so a run prints its own agreement.

    # the row that does not need a GPU: zero flow, their metric, their split
    python run_slof_table1.py --raw-root /data/flow360_raw --mode test \
        --output-dir /outputs/slof_table1_zero

    # a published row, their setting (the shipped CSVs are iters=12)
    python run_slof_table1.py --raw-root /data/flow360_raw --mode test \
        --checkpoint /outputs/slof_weights/singlerotation.pt --iters 12 \
        --device cuda --output-dir /outputs/slof_table1_singlerotation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BUCKETS = ("all", "lt5", "lt10", "lt20", "gte20")

# |latitude| bands, in degrees, for the decomposition of where an ERP-pixel win
# actually comes from. ERP rows are linear in latitude, so these are row slabs.
LAT_EDGES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)

# The two conventions an ERP-pixel EPE bakes in, and the four ways to combine them.
#   plain      their metric: raw pixel displacement, every pixel weighted equally
#   area       raw pixel displacement, each pixel weighted by the solid angle it covers
#   sph        longitudinal component converted to angle-equivalent px (du * cos lat)
#   sph_area   both fixes
VARIANTS = ("plain", "area", "sph", "sph_area")

# SLOF/quantitative_results/*.csv, shipped with their release. The unweighted
# files are the plain EPE/AE run; the distortion_aware_* files are the same run
# with USE_DENSITY_MASK on. Order: all, lt5, lt10, lt20, gte20.
PUBLISHED = {
    "singlerotation": {
        "epe": (1.568181, 0.309022, 0.387124, 0.502485, 62.475649),
        "ae": (0.496864, 0.501493, 0.496714, 0.494935, 0.607081),
        "epe_weighted": (2.548246, 0.469749, 0.599812, 0.789643, 103.057231),
        "ae_weighted": (0.707644, 0.712147, 0.705931, 0.703802, 0.927230),
    },
    "switchrotation": {
        "epe": (1.615216, 0.325814, 0.400975, 0.511809, 64.677911),
        "ae": (0.484966, 0.488987, 0.483769, 0.481927, 0.658610),
        "epe_weighted": (2.625845, 0.496313, 0.623273, 0.806744, 106.592490),
        "ae_weighted": (0.691368, 0.695210, 0.688384, 0.686173, 0.988236),
    },
    "doublerotation": {
        "epe": (2.338830, 0.610784, 0.772102, 0.973909, 80.347806),
        "ae": (1.419504, 1.405884, 1.413653, 1.417047, 1.559905),
        "epe_weighted": (3.843414, 0.897590, 1.173338, 1.520335, 136.613772),
        "ae_weighted": (1.826634, 1.811022, 1.818972, 1.822870, 2.041745),
    },
    "raftfinetune": {
        "epe": (1.624326, 0.314285, 0.393218, 0.509497, 65.339836),
        "ae": (0.521822, 0.527298, 0.521875, 0.519633, 0.646911),
        "epe_weighted": (2.634517, 0.469523, 0.602262, 0.793780, 107.837684),
        "ae_weighted": (0.745454, 0.751455, 0.744281, 0.741456, 0.973926),
    },
    "raft": {
        "epe": (2.057559, 0.558382, 0.681999, 0.838401, 71.735776),
        "ae": (0.820225, 0.825486, 0.821170, 0.819381, 0.868470),
        "epe_weighted": (3.343966, 0.838946, 1.047068, 1.307689, 119.722794),
        "ae_weighted": (1.119672, 1.121675, 1.117512, 1.116266, 1.314313),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SLOF's FLOW360 table in SLOF's metric, with a zero-flow row.")
    parser.add_argument("--source", default="raw", choices=["raw", "shards"],
                        help="'raw' is the faithful path (their PNGs and .npy files). "
                             "'shards' reads the sfprep tars instead: same pairs, but the "
                             "frames went through JPEG and the flow through float16, so it "
                             "measures what our pipeline costs rather than their table.")
    parser.add_argument("--raw-root", default="/data/flow360_raw",
                        help="FLOW360_train_test root (contains train/ and test/).")
    parser.add_argument("--shards", default="/data/shards")
    parser.add_argument("--dataset", default="flow360", help="sfprep dataset name (shards source).")
    parser.add_argument("--mode", default="test", choices=["train", "val", "test"],
                        help="Their split. 'test' is the official held-out one (1089 forward pairs).")
    parser.add_argument("--resize", default="320x640",
                        help="HxW their loader resizes to. Their published setting is 320x640.")
    parser.add_argument("--checkpoint", default="",
                        help="Princeton-tree RAFT checkpoint (SLOF weights/*.pt). Empty = only "
                             "the zero row, which needs no GPU.")
    parser.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    parser.add_argument("--iters", type=int, default=12,
                        help="RAFT refinement iterations. The shipped CSVs are iters=12 "
                             "(train.py calls the evaluator with 12; evaluate_raft.py's own "
                             "__main__ defaults to 64).")
    parser.add_argument("--input-scale", default="unit", choices=["unit", "byte"],
                        help="'unit' = their ToTensor() [0,1] frames (reproduces the published "
                             "numbers). 'byte' = 0-255, the scale the forward's normalization "
                             "expects and the one our universality rows used.")
    parser.add_argument("--weighted", action="store_true",
                        help="Their distortion-density weighting (EPEd). Needs --density-map.")
    parser.add_argument("--density-map", default="",
                        help="Path to their distortiondensity.npy (320x640). Required with "
                             "--weighted.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Theirs is 16. EPE is batch-invariant; their AE is not.")
    parser.add_argument("--resample", default="bicubic", choices=["bicubic", "bilinear", "nearest"],
                        help="PIL frame resampling. Their code takes the PIL default (bicubic).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default="/outputs/slof_table1")
    return parser.parse_args()


def git_hash() -> str:
    env_sha = os.environ.get("OSLO_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Their transforms (SLOF/utils.py ReadData, SLOF/dataloader.py Flow360Loader)
# ---------------------------------------------------------------------------

RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}


def transform_image(image: Image.Image, size_hw: Tuple[int, int], resample: str) -> torch.Tensor:
    """PIL resize -> ToTensor(): float in [0, 1], CHW. Their transform_image + ToTensor."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = image.resize(size_hw[::-1], resample=RESAMPLE[resample])
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)


def transform_flow(flow: np.ndarray, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Their ReadData.transform_flow: normalize by the old size, resample at
    F.interpolate's *nearest* default, rescale by the new size. Returns 2HW float64."""
    flow = np.array(flow, dtype=np.float64, copy=True)
    height, width, _ = flow.shape
    flow[:, :, 0] /= width
    flow[:, :, 1] /= height
    tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size_hw)[0]
    tensor[0] *= size_hw[1]
    tensor[1] *= size_hw[0]
    return tensor


def maprange(x: torch.Tensor, minfrom, maxfrom, minto, maxto):
    """Their utils.maprange, verbatim."""
    return minto + ((maxto - minto) * (x - minfrom)) / (maxfrom - minfrom)


# ---------------------------------------------------------------------------
# Pair sources
# ---------------------------------------------------------------------------


def raw_pairs(root: Path, mode: str) -> List[Tuple[Path, Path, Path]]:
    """Their ReadData index: every sequence, every frame but the last, forward only."""
    sequences = sorted(p for p in (root / mode).glob("*") if p.is_dir())
    if not sequences:
        raise FileNotFoundError(f"no sequences under {root / mode}")
    firsts = list(chain.from_iterable(sorted(s.glob("frames/*"))[:-1] for s in sequences))
    pairs = []
    for f1 in firsts:
        f2 = f1.parent / f"{str(int(f1.stem) + 1).zfill(4)}{f1.suffix}"
        flow = Path(f1.as_posix().replace("frames", "fflows").replace(".png", ".npy"))
        for path in (f2, flow):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing (their loader asserts the same)")
        pairs.append((f1, f2, flow))
    return pairs


def iter_raw(pairs, size_hw, resample) -> Iterator[dict]:
    for f1, f2, flow_path in pairs:
        yield {
            "uid": f"{f1.parent.parent.name}/{f1.stem}",
            "frame1": transform_image(Image.open(f1), size_hw, resample),
            "frame2": transform_image(Image.open(f2), size_hw, resample),
            "flow": transform_flow(np.load(flow_path), size_hw),
        }


def iter_shards(shards_dir: Path, dataset: str, mode: str, size_hw, resample) -> Iterator[dict]:
    """Same pairs from the sfprep tars. Stream order, not their sorted order: EPE is
    order-invariant, their AE is not, so AE from this source is not comparable."""
    from spherical_flow.shard_dataset import _import_sfprep

    iter_shard, list_shards = _import_sfprep()
    found = list_shards(shards_dir, dataset, mode)
    if not found:
        raise FileNotFoundError(f"no shards for {dataset}:{mode} under {shards_dir}")
    for shard in found:
        for record in iter_shard(shard):
            if record["meta"].get("direction") != "forward":
                continue          # their loader only ever reads fflows
            yield {
                "uid": record["meta"]["uid"],
                "frame1": transform_image(Image.fromarray(record["frame1"]), size_hw, resample),
                "frame2": transform_image(Image.fromarray(record["frame2"]), size_hw, resample),
                "flow": transform_flow(record["flow"], size_hw),
            }


# ---------------------------------------------------------------------------
# Their metrics (SLOF/evaluate_raft.py)
# ---------------------------------------------------------------------------


class Accumulator:
    """Per-key weighted sums. With unit weights this is their concatenate-then-
    ``np.mean``, since every selected pixel then carries the same weight; with
    ``cos(lat)`` weights it is the per-solid-angle mean over the same pixels."""

    def __init__(self, keys) -> None:
        self.keys = tuple(keys)
        self.total: Dict[str, float] = {k: 0.0 for k in self.keys}
        self.weight: Dict[str, float] = {k: 0.0 for k in self.keys}
        self.count: Dict[str, int] = {k: 0 for k in self.keys}

    def add(self, values: torch.Tensor, masks: Dict[str, Optional[torch.Tensor]],
            weights: Optional[torch.Tensor] = None) -> None:
        for key, mask in masks.items():
            # Magnitude masks arrive per pixel of the batch; latitude masks are one
            # raster shared by it.
            if mask is not None and mask.shape != values.shape:
                mask = mask.expand_as(values)
            selected = values if mask is None else values[mask]
            if weights is None:
                self.total[key] += float(selected.sum().item())
                self.weight[key] += float(selected.numel())
            else:
                w = weights.expand_as(values)
                w = w if mask is None else w[mask]
                self.total[key] += float((selected * w).sum().item())
                self.weight[key] += float(w.sum().item())
            self.count[key] += int(selected.numel())

    def means(self) -> Dict[str, float]:
        return {k: (self.total[k] / self.weight[k] if self.weight[k] else float("nan"))
                for k in self.keys}

    def counts(self) -> Dict[str, int]:
        return dict(self.count)


def pixel_latitude(height: int) -> torch.Tensor:
    """Row-centre latitude in degrees, [H, 1]. ERP rows are linear in latitude."""
    rows = torch.arange(height, dtype=torch.float64)
    return (90.0 - (rows + 0.5) * 180.0 / height).unsqueeze(1)


def latitude_masks(latitude: torch.Tensor) -> Dict[str, torch.Tensor]:
    """|lat| slabs, keyed ``lo-hi``. Disjoint, unlike their magnitude buckets."""
    absolute = latitude.abs()
    masks = {}
    for lo, hi in zip(LAT_EDGES[:-1], LAT_EDGES[1:]):
        key = f"{int(lo)}-{int(hi)}"
        masks[key] = (absolute >= lo) & (absolute < hi if hi < 90.0 else absolute <= hi)
    return masks


LAT_KEYS = tuple(f"{int(lo)}-{int(hi)}" for lo, hi in zip(LAT_EDGES[:-1], LAT_EDGES[1:]))


def bucket_masks(magnitude: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """Their five predicates. ``all`` is unmasked; the rest are cumulative tails."""
    return {
        "all": None,
        "lt5": magnitude < 5.0,
        "lt10": magnitude < 10.0,
        "lt20": magnitude < 20.0,
        "gte20": magnitude >= 20.0,
    }


def angular_error(gt: torch.Tensor, pred: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Their angularError, defects preserved: ``ugt`` is overwritten before it is
    reused to normalize ``vgt`` (same for u/v), and the cosine is stretched to
    [-1, 1] by the min/max of this batch."""
    ugt, vgt = gt.select(1, 0), gt.select(1, 1)
    u, v = pred.select(1, 0), pred.select(1, 1)

    ugt = ugt / (ugt ** 2 + vgt ** 2 + epsilon).sqrt()
    vgt = vgt / (ugt ** 2 + vgt ** 2 + epsilon).sqrt()
    u = u / (u ** 2 + v ** 2 + epsilon).sqrt()
    v = v / (u ** 2 + v ** 2 + epsilon).sqrt()

    var = ((ugt * u + v * vgt + 1)
           / ((u ** 2 + v ** 2 + 1).sqrt() * (ugt ** 2 + vgt ** 2 + 1).sqrt()))
    var = maprange(var, minfrom=var.min(), maxfrom=var.max(), minto=-1, maxto=1)
    return torch.acos(var.clamp(-1.0, 1.0))


def compare(measured: Dict[str, float], reference: Optional[Tuple[float, ...]]) -> Optional[dict]:
    if reference is None:
        return None
    return {b: {"published": ref, "reproduced": measured[b],
                "abs_diff": measured[b] - ref,
                "rel_diff_pct": 100.0 * (measured[b] - ref) / ref if ref else float("nan")}
            for b, ref in zip(BUCKETS, reference)}


def main() -> None:
    args = parse_args()
    height, width = (int(x) for x in args.resize.lower().split("x"))
    size_hw = (height, width)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density = None
    if args.weighted:
        if not args.density_map:
            raise SystemExit("--weighted needs --density-map (their distortiondensity.npy)")
        raw = np.load(args.density_map)
        if raw.shape != size_hw:
            raise SystemExit(f"density map is {raw.shape}, expected {size_hw}")
        # getdensitymask(): invert, then map the full range onto [0.5, 1.0].
        raw = 1.0 - raw
        raw = maprange(torch.from_numpy(raw).double(), raw.min(), raw.max(), 0.500, 1.000)
        density = raw.unsqueeze(0)

    device = get_device(args.device) if args.checkpoint else torch.device("cpu")
    model = princeton_meta = None
    if args.checkpoint:
        from spherical_flow.princeton_raft import load_princeton_checkpoint

        model, princeton_meta = load_princeton_checkpoint(
            args.checkpoint, device, small=(args.model == "raft_small"))
        print(f"checkpoint={args.checkpoint} iters={args.iters} input_scale={args.input_scale} "
              f"device={device} loaded_keys={princeton_meta['loaded_keys']} "
              f"unexpected_keys={princeton_meta['unexpected_keys']}", flush=True)

    if args.source == "raw":
        pairs = raw_pairs(Path(args.raw_root), args.mode)
        print(f"source=raw root={args.raw_root} mode={args.mode} pairs={len(pairs)}", flush=True)
        stream = iter_raw(pairs, size_hw, args.resample)
    else:
        print(f"source=shards {args.dataset}:{args.mode} (frames are JPEG, flow float16)",
              flush=True)
        stream = iter_shards(Path(args.shards), args.dataset, args.mode, size_hw, args.resample)

    latitude = pixel_latitude(height)                       # [H, 1] degrees
    coslat = torch.cos(torch.deg2rad(latitude)).unsqueeze(0)  # [1, H, 1] solid-angle weight
    lat_masks = {k: m.expand(height, width) for k, m in latitude_masks(latitude).items()}

    names = ["zero"] + (["model"] if model is not None else [])
    rows = {name: {
        "epe": {variant: Accumulator(BUCKETS) for variant in VARIANTS},
        "lat": {variant: Accumulator(LAT_KEYS) for variant in VARIANTS},
        "ae": Accumulator(BUCKETS),
    } for name in names}

    seen = 0
    nonfinite_px = 0
    start = time.time()
    batch: List[dict] = []

    def flush(batch: List[dict]) -> None:
        nonlocal nonfinite_px
        gt = torch.stack([item["flow"] for item in batch], dim=0)          # B2HW float64
        nonfinite_px += int((~torch.isfinite(gt)).sum().item())
        magnitude = gt.pow(2).sum(1).sqrt()
        masks = bucket_masks(magnitude)

        predictions = {"zero": torch.zeros_like(gt)}
        if model is not None:
            frames1 = torch.stack([item["frame1"] for item in batch], dim=0)
            frames2 = torch.stack([item["frame2"] for item in batch], dim=0)
            if args.input_scale == "byte":
                frames1 = frames1 * 255.0
                frames2 = frames2 * 255.0
            with torch.no_grad():
                flow = model(frames1.to(device), frames2.to(device), iters=args.iters)
            predictions["model"] = flow.detach().cpu().double()

        for name, pred in predictions.items():
            residual = gt - pred
            plain = residual.pow(2).sum(1).sqrt()
            # Longitudinal pixels shrink with cos(lat) on the sphere; leaving that out
            # is what lets a raster metric price polar error at up to 1/cos(lat).
            spherical = ((residual.select(1, 0) * coslat) ** 2
                         + residual.select(1, 1) ** 2).sqrt()
            error = angular_error(gt, pred)
            if density is not None:
                plain = plain / density
                spherical = spherical / density
                error = error / density
            for variant, values, weights in (
                    ("plain", plain, None), ("area", plain, coslat),
                    ("sph", spherical, None), ("sph_area", spherical, coslat)):
                rows[name]["epe"][variant].add(values, masks, weights)
                rows[name]["lat"][variant].add(values, lat_masks, weights)
            rows[name]["ae"].add(error, masks)

    for item in stream:
        batch.append(item)
        seen += 1
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            print(f"  pairs={seen} elapsed_s={time.time() - start:.1f}", flush=True)
        if args.max_pairs is not None and seen >= args.max_pairs:
            break
    if batch:
        flush(batch)
    if seen == 0:
        raise RuntimeError("no pairs evaluated")

    suffix = "_weighted" if args.weighted else ""
    stem = Path(args.checkpoint).stem if args.checkpoint else ""
    published = PUBLISHED.get(stem)
    result = {
        "args": vars(args),
        "git_hash": git_hash(),
        "pairs": seen,
        "elapsed_s": time.time() - start,
        "nonfinite_gt_px": nonfinite_px,
        "buckets": list(BUCKETS),
        "protocol": {
            "source": "SLOF/evaluate_raft.py + dataloader.py + utils.py",
            "pair_selection": "forward flows only, sorted sequences x sorted frames[:-1]",
            "gt_sign": "negated twice by their loaders = raw .npy (sfprep pins the same)",
            "flow_resize": "normalize by old size, F.interpolate nearest default, rescale",
            "frame_scale": args.input_scale,
            "buckets": "cumulative and overlapping: lt5 c lt10 c lt20; gte20 is the complement",
            "no_validity_mask": True,
            "ae_defects": "ugt overwritten before normalizing vgt; per-batch min/max stretch",
            "ae_deviation": "acos argument clamped to [-1,1]; their code would emit nan if "
                            "the stretch overshot by one ulp",
            "published_csv_iters": 12,
        },
        "model": None if model is None else {
            "checkpoint": args.checkpoint, "arch": "princeton", "name": args.model,
            "iters": args.iters, "princeton": princeton_meta,
        },
        "metrics": {},
        "published_reference": published,
    }

    result["latitude_bands"] = list(LAT_KEYS)
    result["variants"] = {
        "plain": "their metric: raw ERP px, every pixel weighted equally",
        "area": "raw ERP px, each pixel weighted by cos(lat) (its solid angle)",
        "sph": "du scaled by cos(lat) before the norm, pixels weighted equally",
        "sph_area": "both corrections",
    }

    for name in names:
        for variant in VARIANTS:
            result["metrics"][f"{name}_epe_{variant}{suffix}"] = rows[name]["epe"][variant].means()
            result["metrics"][f"{name}_lat_{variant}{suffix}"] = rows[name]["lat"][variant].means()
        result["metrics"][f"{name}_ae{suffix}"] = rows[name]["ae"].means()
    result["pixel_counts"] = {"buckets": rows[names[0]]["epe"]["plain"].counts(),
                              "latitude": rows[names[0]]["lat"]["plain"].counts()}

    def table(title: str, keys, key_of) -> None:
        print(f"\n{title}", flush=True)
        print(f"{'row':<10}{'variant':<10}" + "".join(f"{k:>14}" for k in keys), flush=True)
        for name in names:
            for variant in VARIANTS:
                means = result["metrics"][key_of(name, variant)]
                print(f"{name:<10}{variant:<10}"
                      + "".join(f"{means[k]:>14.6f}" for k in keys), flush=True)

    table("EPE by GT-magnitude bucket (px @ %dx%d)" % size_hw, BUCKETS,
          lambda n, v: f"{n}_epe_{v}{suffix}")
    table("EPE by |latitude| band", LAT_KEYS, lambda n, v: f"{n}_lat_{v}{suffix}")

    print("\nAE (their formula, defects included)", flush=True)
    for name in names:
        means = result["metrics"][f"{name}_ae{suffix}"]
        print(f"{name:<10}{'ae':<10}" + "".join(f"{means[b]:>14.6f}" for b in BUCKETS), flush=True)

    if published is not None and model is not None:
        result["metrics"]["published_vs_reproduced_epe"] = compare(
            result["metrics"][f"model_epe_plain{suffix}"], published.get(f"epe{suffix}"))
        result["metrics"]["published_vs_reproduced_ae"] = compare(
            result["metrics"][f"model_ae{suffix}"], published.get(f"ae{suffix}"))
        print("\npublished vs reproduced (epe, plain):", flush=True)
        for bucket, entry in result["metrics"]["published_vs_reproduced_epe"].items():
            print(f"  {bucket:<6} published={entry['published']:.6f} "
                  f"reproduced={entry['reproduced']:.6f} "
                  f"diff={entry['abs_diff']:+.6f} ({entry['rel_diff_pct']:+.2f}%)", flush=True)

    if model is not None:
        print("\nimprovement over zero flow (%, positive = beats doing nothing)", flush=True)
        for label, keys, key_of in (
                ("bucket", BUCKETS, lambda n, v: f"{n}_epe_{v}{suffix}"),
                ("|lat|", LAT_KEYS, lambda n, v: f"{n}_lat_{v}{suffix}")):
            print(f"{'variant':<10}" + "".join(f"{k:>14}" for k in keys), flush=True)
            for variant in VARIANTS:
                zero = result["metrics"][key_of("zero", variant)]
                pred = result["metrics"][key_of("model", variant)]
                gain = {k: (100.0 * (zero[k] - pred[k]) / zero[k] if zero[k] else float("nan"))
                        for k in keys}
                result["metrics"][f"improvement_pct_{label.strip('|')}_{variant}"] = gain
                print(f"{variant:<10}" + "".join(f"{gain[k]:>+14.2f}" for k in keys), flush=True)

    metrics_out = output_dir / "slof_table1.json"
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    csv_out = output_dir / "slof_table1.csv"
    with open(csv_out, "w", encoding="utf-8") as handle:
        handle.write(",".join(BUCKETS) + ",metric,mode,row\n")
        for name in names:
            for variant in VARIANTS:
                means = result["metrics"][f"{name}_epe_{variant}{suffix}"]
                handle.write(",".join(f"{means[b]:.10f}" for b in BUCKETS)
                             + f",epe_{variant}{suffix},2DRawFlow,{name}\n")
            means = result["metrics"][f"{name}_ae{suffix}"]
            handle.write(",".join(f"{means[b]:.10f}" for b in BUCKETS)
                         + f",ae{suffix},2DRawFlow,{name}\n")
    print(f"\nsaved_metrics={metrics_out}\nsaved_csv={csv_out}", flush=True)


if __name__ == "__main__":
    main()
