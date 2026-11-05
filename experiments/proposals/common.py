"""Shared pieces of the five proposal probes (reports/Formulações de fluxo para HEALPix.md).

Data: FlowScape pairs from the raw release (local Mac) or from sfprep shards (box).
Scoring: the PriOr-Flow protocol of run_epe_eval.py (ERP-px EPE, SEPE, All / Equator /
Poles at 45 deg, GT-displacement bins) plus the cube viewport EPE, for any number of
named rows, each given as a dense ERP flow [H, W, 2].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_epe_eval as ee  # noqa: E402
from spherical_flow.erp_readout import erp_pixel_directions, wrap_px  # noqa: E402
from spherical_flow.geometry import points_to_equirectangular_pixels  # noqa: E402
from spherical_flow.viewport_metrics import (  # noqa: E402
    accumulate_viewport,
    build_viewports,
    erp_flow_endpoints,
    erp_uv_to_dirs,
    finalize_viewport,
    viewport_epe,
)

RAW_ROOT = os.environ.get("FLOWSCAPE_RAW", "/Volumes/External SSD/Mestrado/Datasets/FlowScape(Flow360)")
REFERENCE = "matched-recipe flowscape:test EPE (RAFT_PARITY 11.13): OSLO 8.63 px, RAFT-small 6.32 px"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shards", default="", help="sfprep shard root (box); empty = raw FlowScape")
    p.add_argument("--sources", default="flowscape:test")
    p.add_argument("--raw-root", default=RAW_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--weathers", default="sunny,cloud,fog,rain")
    p.add_argument("--stride", type=int, default=50, help="keep every k-th pair")
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, 1)[0]
        if abs(magic - 202021.25) > 1e-3:
            raise ValueError(f"bad .flo magic in {path}")
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)


def _raw_pairs(root: Path, split: str, weathers: List[str]) -> Iterator[dict]:
    from PIL import Image

    for weather in weathers:
        base = root / split / weather
        if not (base / "img").is_dir():
            continue
        for seq in sorted(p for p in (base / "img").iterdir() if p.is_dir()):
            stems = sorted(p.stem for p in seq.iterdir() if p.suffix.lower() == ".jpg")
            for a, b in zip(stems[:-1], stems[1:]):
                flo = base / "flow" / seq.name / f"{a}.flo"
                if not flo.is_file():
                    continue

                def load(rel, a=a, b=b, seq=seq, flo=flo, base=base):
                    rec = {
                        "frame1": np.array(Image.open(seq / f"{a}.jpg").convert("RGB")),
                        "frame2": np.array(Image.open(seq / f"{b}.jpg").convert("RGB")),
                        "flow": read_flo(flo),
                        "name": f"{split}/{weather}/{seq.name}/{a}",
                    }
                    h, w = rec["flow"].shape[:2]
                    rec["valid"] = np.ones((h, w), dtype=bool)  # official protocol: all pixels
                    sem = base / "semantic" / seq.name / f"{a}.jpg"
                    rec["semantic"] = np.array(Image.open(sem).convert("RGB")) if sem.is_file() else None
                    return rec

                yield load


def iter_pairs(args: argparse.Namespace) -> Iterator[dict]:
    """Yields record dicts: frame1/frame2 uint8 [H, W, 3], flow [H, W, 2], valid, name, semantic."""
    kept = 0
    if args.shards:
        from run_raft_shard_baseline import iter_source_records, parse_sources

        for k, r in enumerate(iter_source_records(Path(args.shards), parse_sources(args.sources))):
            if k % args.stride:
                continue
            r = dict(r)
            r.setdefault("name", f"{args.sources}#{k}")
            r.setdefault("semantic", None)
            yield r
            kept += 1
            if kept >= args.max_pairs:
                return
    else:
        for k, load in enumerate(_raw_pairs(Path(args.raw_root), args.split, args.weathers.split(","))):
            if k % args.stride:
                continue
            yield load(None)
            kept += 1
            if kept >= args.max_pairs:
                return


def gt_endpoints(flow: np.ndarray) -> torch.Tensor:
    """Exact GT endpoint directions [H*W, 3] (fp64) of every ERP pixel."""
    h, w = flow.shape[:2]
    f = torch.from_numpy(np.ascontiguousarray(flow)).double().reshape(-1, 2)
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    return erp_uv_to_dirs(u.reshape(-1) + f[:, 0], v.reshape(-1) + f[:, 1], h, w)


def pixel_dirs64(h: int, w: int) -> torch.Tensor:
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    return erp_uv_to_dirs(u.reshape(-1), v.reshape(-1), h, w)


def endpoints_to_erp_flow(end: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Endpoint directions [H*W, 3] -> ERP pixel flow [H, W, 2] (du seam-wrapped)."""
    u2, v2 = points_to_equirectangular_pixels(end.double(), h, w)
    u1 = torch.arange(w, dtype=torch.float64).repeat(h)
    v1 = torch.arange(h, dtype=torch.float64).repeat_interleave(w)
    return torch.stack([wrap_px(u2 - u1, w), v2 - v1], -1).reshape(h, w, 2).float()


class Scorer:
    """Accumulates the benchmark metrics for named rows over pairs."""

    def __init__(self, layouts=("cube",)):
        self.layouts = layouts
        self.sums: Dict[str, Dict[str, list]] = {}
        self.vp: Dict[str, Dict[str, tuple]] = {}
        self.per_pair: Dict[str, List[float]] = {}
        self.cache = {}

    def add(self, row: str, pred: torch.Tensor, gt_flow: np.ndarray, extra_masks: Dict[str, torch.Tensor] = None):
        h, w = gt_flow.shape[:2]
        key = (h, w)
        if key not in self.cache:
            self.cache[key] = {"official": ee.official_region_masks(h, w),
                               "vp": {L: build_viewports(L, h, w) for L in self.layouts}}
        c = self.cache[key]
        gt = torch.from_numpy(np.ascontiguousarray(gt_flow)).float()
        epe, sepe = ee.official_maps(pred.cpu().float(), gt)
        regions = dict(c["official"])
        regions.update(ee.displacement_bin_masks(gt))
        regions.update(ee.overshoot_masks(gt))
        if extra_masks:
            regions.update(extra_masks)
        acc = self.sums.setdefault(row, {})
        for region, m in regions.items():
            for kind, vals in (("epe", epe), ("sepe", sepe)):
                s = acc.setdefault(f"{region}_{kind}", [0.0, 0.0])
                s[0] += vals[m].sum().item()
                s[1] += m.sum().item()
        for L in self.layouts:
            vp = c["vp"][L]
            tot = self.vp.setdefault(row, {}).setdefault(L, ({}, {}))
            accumulate_viewport(viewport_epe(erp_flow_endpoints(pred.cpu(), vp), erp_flow_endpoints(gt, vp), vp),
                                vp, *tot)
        self.per_pair.setdefault(row, []).append(epe.mean().item())

    def summary(self, base_row: str = None) -> Dict[str, dict]:
        out = {}
        for row, acc in self.sums.items():
            m = {f"official_{k}": s / n for k, (s, n) in acc.items() if n > 0}
            m["mask_fraction"] = {k[:-4]: n for k, (s, n) in acc.items() if k.endswith("_epe")}
            for L, tot in self.vp.get(row, {}).items():
                m.update(finalize_viewport(*tot, L))
            if base_row and base_row in self.per_pair and len(self.per_pair[row]) == len(self.per_pair[base_row]):
                d = np.array(self.per_pair[row]) - np.array(self.per_pair[base_row])
                m["paired_delta_vs_" + base_row] = float(d.mean())
                m["paired_delta_se"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
                m["paired_improved_fraction"] = float((d < 0).mean())
            out[row] = m
        return out

    def print_table(self, summary: Dict[str, dict], extra_cols=()):
        cols = [("EPE all", "official_all_epe"), ("eq", "official_equator_epe"), ("pol", "official_poles_epe"),
                ("SEPE", "official_all_sepe"), ("0-5deg", "official_disp0-5_epe"),
                ("5-15deg", "official_disp5-15_epe"), ("inraster", "official_inraster_epe"),
                ("overshoot", "official_overshoot_epe"), ("vp cube", "viewport_cube_all_epe_px")] + list(extra_cols)
        print(f"{'row':<26}" + "".join(f"{c:>10}" for c, _ in cols))
        for row, m in summary.items():
            vals = []
            for c, k in cols:
                v = m.get(k, float("nan"))
                vals.append(v * 1e3 if k.endswith("sepe") else v)
            print(f"{row:<26}" + "".join(f"{v:>10.3f}" for v in vals))


def save(out_dir: str, payload: dict, name: str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    path = path / name
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"saved={path}", flush=True)
    return path
