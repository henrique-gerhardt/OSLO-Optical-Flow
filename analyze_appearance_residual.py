"""Structure diagnostic of the REAL inter-frame appearance change (plan P2C, post-P0).

P0/P0b eliminated photometric *magnitude* as the wall: at the real pairs' mean
photometric delta (0.42/255), coherent global jitter costs 2.65 improvement points
and iid per-pixel noise 0.33 — the real leg costs 112.8. This script measures what
the real nuisance actually looks like, so P1's structured augmentation imitates the
measured enemy.

Per pair, on the ERP raster, the brightness-constancy residual in frame-1 coords:

    delta(x) = frame2[ x + gt_flow(x) ]  -  frame1[x]      (valid pixels only)

(forward GT only — no backward flow needed). Reported statistics, per pair then
aggregated mean +- std across pairs (quantiles pooled):

  magnitude   mean |delta| x255, pooled p50/p90/p99; and the no-warp baseline
              mean |f2 - f1| (how much motion compensation explains)
  edges       Pearson corr(|delta|, |grad f1|); fraction of total |delta| mass in
              the top-10% gradient pixels (iid -> 0.10); mean |delta| ratio
              top-decile-edge vs bottom-half-edge pixels
  spatial     lag-1 / lag-4 horizontal and lag-1 vertical autocorrelation of the
              delta luma (global jitter -> ~1, iid noise -> ~0)
  color       luma share of delta variance in YIQ (exposure/shading -> high,
              chromatic/specular shifts -> lower)
  motion      corr(|delta|, |gt_flow| px) — does the residual ride motion edges?
  regions     mean |delta| equator vs poles vs seam

``--self-test`` validates the metrics on constructed cases (copy / global jitter /
iid noise on a real frame) before trusting the real numbers.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from spherical_flow.erp_readout import build_pixel_region_masks
from spherical_flow.flow360 import bilinear_sample_erp
from spherical_flow.shard_dataset import _to_chw_free_float
from run_raft_shard_baseline import iter_source_records, parse_sources


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def grad_mag(frame: torch.Tensor) -> torch.Tensor:
    """|grad| of luma with horizontal wrap (ERP) and vertical clamp."""
    y = luma(frame)
    gx = torch.roll(y, -1, dims=1) - torch.roll(y, 1, dims=1)
    yp = torch.cat([y[:1], y, y[-1:]], dim=0)
    gy = yp[2:] - yp[:-2]
    return torch.sqrt(gx.square() + gy.square())


def masked_pearson(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
    a, b = a[m].double(), b[m].double()
    if a.numel() < 16:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def lag_corr(d: torch.Tensor, m: torch.Tensor, lag: int, dim: int) -> float:
    ds, ms = torch.roll(d, -lag, dims=dim), torch.roll(m, -lag, dims=dim)
    if dim == 0:  # vertical: rolled-over rows are invalid
        ms = ms.clone()
        ms[-lag:, :] = False
    return masked_pearson(d, ds, m & ms)


def pair_stats(f1: torch.Tensor, f2: torch.Tensor, flow: torch.Tensor,
               valid: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    H, W = f1.shape[:2]
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    eu, ev = u + flow[..., 0], v + flow[..., 1]
    m = valid.bool() & (ev >= 0) & (ev <= H - 1)
    warped = bilinear_sample_erp(f2, eu.flatten(), ev.flatten()).reshape(H, W, 3)

    delta = warped - f1
    dmag = delta.abs().mean(dim=-1)
    dmag_nowarp = (f2 - f1).abs().mean(dim=-1)
    dl = luma(delta)

    edge = grad_mag(f1)
    thr_hi = torch.quantile(edge[m].float(), 0.9)
    thr_lo = torch.quantile(edge[m].float(), 0.5)
    hi, lo = m & (edge >= thr_hi), m & (edge <= thr_lo)

    r, g, b = delta.unbind(dim=-1)
    dy = 0.299 * r + 0.587 * g + 0.114 * b
    di = 0.596 * r - 0.274 * g - 0.322 * b
    dq = 0.211 * r - 0.523 * g + 0.312 * b
    var_y, var_c = float(dy[m].var()), float(di[m].var() + dq[m].var())

    out = {
        "mean_abs_delta_255": float(dmag[m].mean()) * 255.0,
        "mean_abs_nowarp_255": float(dmag_nowarp[m].mean()) * 255.0,
        "edge_corr": masked_pearson(dmag, edge, m),
        "edge_top10_mass": float(dmag[hi].sum() / dmag[m].sum().clamp_min(1e-12)),
        "edge_hi_lo_ratio": float(dmag[hi].mean() / dmag[lo].mean().clamp_min(1e-12)),
        "autocorr_h1": lag_corr(dl, m, 1, 1),
        "autocorr_h4": lag_corr(dl, m, 4, 1),
        "autocorr_v1": lag_corr(dl, m, 1, 0),
        "luma_share": var_y / max(var_y + var_c, 1e-12),
        "motion_corr": masked_pearson(dmag, flow.norm(dim=-1), m),
    }
    for region in ("equator", "poles", "seam"):
        sel = m & masks[region]
        out[f"{region}_mean_abs_delta_255"] = float(dmag[sel].mean()) * 255.0 if sel.any() else float("nan")
    return out


def self_test() -> None:
    from spherical_flow.photometric import apply_jitter, apply_noise, sample_jitter_params

    torch.manual_seed(7)
    H, W = 256, 512
    base = torch.rand(H // 8, W // 8, 3)
    f1 = torch.nn.functional.interpolate(
        base.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).contiguous()
    zero = torch.zeros(H, W, 2)
    ones = torch.ones(H, W)
    masks = build_pixel_region_masks(H, W)

    s_copy = pair_stats(f1, f1.clone(), zero, ones, masks)
    gen = torch.Generator().manual_seed(3)
    s_jit = pair_stats(f1, apply_jitter(f1, sample_jitter_params(gen, 0.5)), zero, ones, masks)
    s_noise = pair_stats(f1, apply_noise(f1, gen, 4.0), zero, ones, masks)

    print(f"copy : delta {s_copy['mean_abs_delta_255']:.3f}/255")
    print(f"jit  : delta {s_jit['mean_abs_delta_255']:.2f}/255 autocorr_h1 {s_jit['autocorr_h1']:.3f} "
          f"luma {s_jit['luma_share']:.2f} edge_corr {s_jit['edge_corr']:.3f}")
    print(f"noise: delta {s_noise['mean_abs_delta_255']:.2f}/255 autocorr_h1 {s_noise['autocorr_h1']:.3f} "
          f"edge_top10_mass {s_noise['edge_top10_mass']:.3f}")
    assert s_copy["mean_abs_delta_255"] < 1e-4
    assert s_jit["autocorr_h1"] > 0.85, "global jitter must be spatially coherent"
    assert abs(s_noise["autocorr_h1"]) < 0.1, "iid noise must be spatially white"
    assert abs(s_noise["edge_top10_mass"] - 0.10) < 0.03, "iid noise mass must be ~uniform"
    print("self-test PASSED (copy floor, jitter coherent, noise white/uniform).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Structure diagnostic of real inter-frame appearance change.")
    ap.add_argument("--shards", default="/data/shards")
    ap.add_argument("--sources", default="flow360:val")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--output-dir", default="/outputs/appearance_residual")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    sources = parse_sources(args.sources)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_cache: Dict[tuple, Dict[str, torch.Tensor]] = {}
    per_pair: List[Dict[str, float]] = []
    dmag_samples: List[torch.Tensor] = []
    start = time.time()

    for record in iter_source_records(Path(args.shards), sources):
        f1 = _to_chw_free_float(record["frame1"])
        f2 = _to_chw_free_float(record["frame2"])
        flow = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
        valid = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        key = f1.shape[:2]
        if key not in masks_cache:
            masks_cache[key] = build_pixel_region_masks(*key)
        per_pair.append(pair_stats(f1, f2, flow, valid, masks_cache[key]))
        H, W = key
        v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                              torch.arange(W, dtype=torch.float32), indexing="ij")
        warped = bilinear_sample_erp(
            f2, (u + flow[..., 0]).flatten(), (v + flow[..., 1]).flatten()
        ).reshape(H, W, 3)
        dmag = (warped - f1).abs().mean(dim=-1)
        m = valid.bool() & ((v + flow[..., 1]) >= 0) & ((v + flow[..., 1]) <= H - 1)
        dmag_samples.append(dmag[m].flatten()[::101] * 255.0)
        if args.max_pairs is not None and len(per_pair) >= args.max_pairs:
            break

    keys = per_pair[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([p[k] for p in per_pair], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    pooled = torch.cat(dmag_samples)
    for q, name in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
        agg[f"abs_delta_255_{name}"] = float(torch.quantile(pooled, q))
    agg["pairs"] = len(per_pair)
    agg["elapsed_s"] = time.time() - start

    print(json.dumps({k: round(v, 4) for k, v in sorted(agg.items())}, indent=2))
    with open(out_dir / "appearance_residual.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "sources": sources, "stats": agg}, fh, indent=2)
    print(f"saved={out_dir / 'appearance_residual.json'}")


if __name__ == "__main__":
    main()
