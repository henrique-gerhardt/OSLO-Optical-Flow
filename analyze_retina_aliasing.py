"""Does the retina alias the ERP raster at the poles? (polar-deficit diagnostic)

The retina samples the ERP frame at HEALPix node directions with bilinear
interpolation. Bilinear reads four taps, so it *point-samples* whatever the
raster holds at that position, and how much raster each node stands for depends
on latitude:

    ERP pixels per r=7 node  =  1.7 at the equator,  6.6 averaged over |lat|>60,
                                20 at 85 deg,  97 at 89 deg

That is a 3.7x more aggressive decimation inside the polar mask than inside the
equatorial one, done with a 4-tap kernel. Whatever the raster carries above the
node Nyquist rate does not disappear: it folds back as aliasing, and the model
ingests it as if it were signal.

This probe measures the folded energy directly, with no model involved. For each
frame it samples the node directions twice:

    v_raw    from the frame as stored
    v_band   from the frame low-passed to the node spacing AT EACH LATITUDE
             (a longitudinal box whose width is the node spacing in pixels,
             which grows as 1/cos(lat), plus the matching vertical box)

``v_raw - v_band`` is the above-Nyquist content the retina folds in. Reported per
latitude band, in levels of 255, next to the band-limited signal's own contrast
so the ratio is readable as a signal-to-alias figure.

The control that makes it a measurement rather than an illustration: the same
quantity is computed for an equiangular grid of the same node count, whose
spacing follows the raster instead of the sphere. If aliasing were an artefact of
the probe, both grids would show it.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from spherical_flow.geometry import healpix_unit_vectors
from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.shard_dataset import _to_chw_free_float
from run_raft_shard_baseline import iter_source_records, parse_sources

BANDS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90))


def equiangular_unit_vectors(count: int) -> torch.Tensor:
    """Lat/lon grid with ~`count` nodes, 2:1 aspect, matching HEALPix nesting."""
    rows = int(round((count / 2) ** 0.5))
    cols = 2 * rows
    lat = (90.0 - (torch.arange(rows, dtype=torch.float64) + 0.5) * (180.0 / rows)).deg2rad()
    lon = ((torch.arange(cols, dtype=torch.float64) + 0.5) * (360.0 / cols) - 180.0).deg2rad()
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1).float()


def node_uv(points: torch.Tensor, height: int, width: int):
    x, y, z = points.double().unbind(dim=-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z.clamp(-1.0, 1.0))
    u = (lon / (2 * np.pi) + 0.5) * width - 0.5
    v = (0.5 - lat / np.pi) * height - 0.5
    return u.float(), v.float(), lat.float()


def _box_blur_rows(frame: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Horizontal box blur with a per-row width, wrapping in longitude.

    Rows sharing a width are blurred together, so the cost is one cumulative sum
    per distinct width rather than one per row.
    """
    H, W, C = frame.shape
    out = frame.clone()
    for w in widths.unique():
        k = int(w.item())
        if k < 2:
            continue
        rows = (widths == w).nonzero(as_tuple=True)[0]
        sel = frame[rows]                                   # [R, W, C]
        pad = torch.cat([sel[:, -k:], sel, sel[:, :k]], dim=1)
        csum = torch.cumsum(pad.double(), dim=1)
        csum = torch.cat([torch.zeros_like(csum[:, :1]), csum], dim=1)
        off = k - k // 2                                    # centre the k-wide window
        out[rows] = ((csum[:, off + k:off + k + W] - csum[:, off:off + W]) / k).float()
    return out


def band_limit(frame: torch.Tensor, node_spacing_deg: float) -> torch.Tensor:
    """Low-pass the ERP frame to the node spacing, latitude by latitude."""
    H, W, _ = frame.shape
    lat = 90.0 - (torch.arange(H, dtype=torch.float64) + 0.5) * (180.0 / H)
    px_per_deg_lon = W / 360.0
    # Node spacing in *pixels* along the row: constant on the sphere, so it spans
    # more pixels the closer the row is to a pole.
    widths = (node_spacing_deg * px_per_deg_lon / lat.deg2rad().cos().clamp_min(1e-6))
    widths = widths.round().clamp(1, W).long()
    out = _box_blur_rows(frame, widths)
    kv = max(int(round(node_spacing_deg * H / 180.0)), 1)
    if kv >= 2:                                              # vertical box, clamped edges
        pad = torch.cat([out[:1].expand(kv, -1, -1), out, out[-1:].expand(kv, -1, -1)], dim=0)
        csum = torch.cumsum(pad.double(), dim=0)
        csum = torch.cat([torch.zeros_like(csum[:1]), csum], dim=0)
        off = kv - kv // 2
        out = ((csum[off + kv:off + kv + H] - csum[off:off + H]) / kv).float()
    return out


def probe(points: torch.Tensor, spacing_deg: float, frame: torch.Tensor) -> Dict[str, List[float]]:
    H, W, _ = frame.shape
    u, v, lat = node_uv(points, H, W)
    raw = bilinear_sample_erp(frame, u, v)
    band = bilinear_sample_erp(band_limit(frame, spacing_deg), u, v)
    alias = (raw - band).abs().mean(dim=-1) * 255.0
    contrast = (band - band.mean()).abs().mean(dim=-1) * 255.0
    lat_deg = lat.rad2deg().abs()
    out: Dict[str, List[float]] = {}
    for lo, hi in BANDS:
        sel = (lat_deg >= lo) & (lat_deg < hi + (1e-6 if hi == 90 else 0.0))
        if sel.any():
            out[f"{lo}_{hi}"] = [float(alias[sel].mean()), float(contrast[sel].mean())]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Retina aliasing by latitude.")
    ap.add_argument("--shards", default="/data/shards")
    ap.add_argument("--sources", default="flow360:test")
    ap.add_argument("--resolution", type=int, default=7)
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--output-dir", default="/outputs/retina_aliasing")
    args = ap.parse_args()

    hp = healpix_unit_vectors(args.resolution)
    n = hp.size(0)
    spacing = float(np.degrees(np.sqrt(4 * np.pi / n)))
    grids = {"healpix": (hp, spacing), "equiangular": (equiangular_unit_vectors(n), spacing)}
    print(f"nos {n}  espacamento {spacing:.3f} deg")

    acc: Dict[str, Dict[str, List[List[float]]]] = {g: {} for g in grids}
    seen = 0
    start = time.time()
    for record in iter_source_records(Path(args.shards), parse_sources(args.sources)):
        frame = _to_chw_free_float(record["frame1"])
        for name, (pts, sp) in grids.items():
            for band, vals in probe(pts, sp, frame).items():
                acc[name].setdefault(band, []).append(vals)
        seen += 1
        if seen >= args.max_pairs:
            break

    report: Dict[str, object] = {"nodes": n, "spacing_deg": spacing, "frames": seen}
    print(f"\n{'faixa de latitude':<20}{'HEALPix alias':>15}{'equiang. alias':>16}{'sinal':>10}")
    for lo, hi in BANDS:
        key = f"{lo}_{hi}"
        row = {}
        for name in grids:
            vals = np.array(acc[name].get(key, [[np.nan, np.nan]]))
            row[name] = [float(vals[:, 0].mean()), float(vals[:, 1].mean())]
        report[key] = row
        print("%-20s%15.3f%16.3f%10.2f"
              % (f"{lo}-{hi} graus", row["healpix"][0], row["equiangular"][0], row["healpix"][1]))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retina_aliasing.json").write_text(json.dumps(report, indent=2))
    print(f"\nelapsed {time.time() - start:.1f}s  saved={out / 'retina_aliasing.json'}")


if __name__ == "__main__":
    main()
