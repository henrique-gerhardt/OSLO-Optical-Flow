"""Model-free arbiter of a dataset's flow sign convention (plan §16.16b).

The convention defect in FLOW360 (`fflows` is the negative of the N->N+1 motion,
§16.12-16.18) was diagnosed with predictors: frozen RAFT-large says one thing,
SLOF's checkpoints say the opposite, and the 2x2 in §16.23 separates them by 60
to 106 points. That is strong, but every row of it is a neural network, and the
claim being made is about a public dataset. Before asserting the defect in print
this project needs one measurement that no model touches.

Brightness constancy is that measurement. If g is the N->N+1 flow at pixel x of
frame 1, then frame2[x + g(x)] should equal frame1[x]. So sweep a scalar along
the stored flow,

    R(alpha) = mean | frame2[ x + alpha * g(x) ] - frame1[x] |

and read off where the minimum sits. alpha = +1 means the stored flow is the
motion. alpha = -1 means it is the negative of the motion. Nothing here is
learned; it is photometry and interpolation.

Two things make or break the power of this test on real video:

* **Sub-pixel motion has no power.** The P0 diagnostic measured it: on flow360
  the constancy residual barely responds to the GT warp at all (3.10/255 unwarped
  vs 3.12 warped), because the median displacement is ~0.23 ERP px and the
  appearance change between real frames is an order of magnitude larger than the
  motion signal. The test must therefore run on the movers only
  (`--min-motion-px`) and on pixels that carry gradient (`--edge-quantile`),
  where |g| is large enough that +g and -g land on genuinely different content.
* **The pixel set must not move with alpha.** Warping by different alphas pushes
  different pixels out of bounds, so R(-1) and R(+1) would be averaged over
  different populations. Every alpha is scored on the intersection mask.

The per-pair sign vote is the headline statistic: for each pair, does R(+1) or
R(-1) win? That is a paired, distribution-free comparison over thousands of
pairs, and it is reported per direction because the whole point is that FLOW360's
two directions disagree.

``--self-test`` builds a pair with a KNOWN convention (frame2 synthesised so that
frame2[x+g] == frame1[x]) and asserts the arbiter returns +1, then feeds it -g
and asserts it returns -1. Run it every time. This project already published one
inverted diagnosis (§16.15) by reasoning about a sign instead of measuring it.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

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


def select_pixels(f1: torch.Tensor, flow: torch.Tensor, valid: torch.Tensor,
                  min_motion_px: float, edge_quantile: float,
                  max_abs_lat_deg: float) -> torch.Tensor:
    """Pixels where constancy can actually discriminate a sign."""
    H, W = f1.shape[:2]
    lat = 90.0 - (torch.arange(H, dtype=torch.float32) + 0.5) * (180.0 / H)
    sel = valid.bool() & (lat.abs() <= max_abs_lat_deg).unsqueeze(1)
    sel = sel & (flow.norm(dim=-1) >= min_motion_px)
    if not sel.any():
        return sel
    edge = grad_mag(f1)
    thr = torch.quantile(edge[sel].float(), edge_quantile)
    return sel & (edge >= thr)


def alpha_residuals(f1: torch.Tensor, f2: torch.Tensor, flow: torch.Tensor,
                    sel: torch.Tensor, alphas: List[float]) -> Optional[np.ndarray]:
    """R(alpha) in 1/255, scored on the SAME pixels for every alpha."""
    H, W = f1.shape[:2]
    idx = sel.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() < 64:
        return None
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    us, vs = u.reshape(-1)[idx], v.reshape(-1)[idx]
    gu, gv = flow[..., 0].reshape(-1)[idx], flow[..., 1].reshape(-1)[idx]
    ref = f1.reshape(-1, 3)[idx]

    inb = torch.ones_like(us, dtype=torch.bool)
    for a in alphas:                       # common support across the whole sweep
        ev = vs + a * gv
        inb &= (ev >= 0) & (ev <= H - 1)
    if int(inb.sum()) < 64:
        return None

    out = np.empty(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        warped = bilinear_sample_erp(f2, (us + a * gu)[inb], (vs + a * gv)[inb])
        out[i] = float((warped - ref[inb]).abs().mean()) * 255.0
    return out


def parabola_argmin(alphas: List[float], curve: np.ndarray) -> float:
    """Sub-grid minimum from the three points around the grid minimum."""
    k = int(np.argmin(curve))
    if k == 0 or k == len(curve) - 1:
        return float(alphas[k])
    x0, x1, x2 = alphas[k - 1], alphas[k], alphas[k + 1]
    y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-12:
        return float(x1)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    return float(x1) if abs(a) < 1e-12 else float(-b / (2 * a))


def sign_test_z(wins: int, total: int) -> float:
    """Normal approximation with continuity correction; report counts too."""
    if total == 0:
        return float("nan")
    return (abs(wins - total / 2) - 0.5) / math.sqrt(total / 4)


def summarize(curves: List[np.ndarray], alphas: List[float]) -> Dict[str, object]:
    stack = np.stack(curves)
    mean = stack.mean(axis=0)
    i_pos, i_neg = alphas.index(1.0), alphas.index(-1.0)
    i_zero = alphas.index(0.0)
    wins_pos = int((stack[:, i_pos] < stack[:, i_neg]).sum())
    total = stack.shape[0]
    r0, rp, rn = mean[i_zero], mean[i_pos], mean[i_neg]
    best = min(rp, rn)
    return {
        "pairs": total,
        "curve_255": {str(a): round(float(m), 4) for a, m in zip(alphas, mean)},
        "argmin_grid": float(alphas[int(np.argmin(mean))]),
        "argmin_parabolic": round(parabola_argmin(alphas, mean), 4),
        "R_nowarp_255": round(float(r0), 4),
        "R_plus1_255": round(float(rp), 4),
        "R_minus1_255": round(float(rn), 4),
        "explained_by_best_pct": round(float((r0 - best) / max(r0, 1e-9) * 100.0), 3),
        "pairs_favouring_plus1": wins_pos,
        "win_rate_plus1": round(wins_pos / total, 4) if total else float("nan"),
        "sign_test_z": round(sign_test_z(wins_pos, total), 2),
        "verdict": "identity (stored flow IS the motion)" if rp < rn
                   else "negated (stored flow is the NEGATIVE of the motion)",
    }


def self_test(alphas: List[float]) -> None:
    torch.manual_seed(7)
    H, W = 128, 256
    base = torch.rand(H // 4, W // 4, 3)
    f1 = torch.nn.functional.interpolate(
        base.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).contiguous()
    f1[40:60, 80:140] = 0.05                      # hard edges, so the residual responds
    f1[70:90, 30:70] = 0.95

    gx, gy = 4.0, 2.0
    flow = torch.zeros(H, W, 2)
    flow[..., 0], flow[..., 1] = gx, gy
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32), indexing="ij")
    # frame2[y] = frame1[y - g]  =>  frame2[x + g] == frame1[x]  (identity convention)
    f2 = bilinear_sample_erp(f1, (u - gx).reshape(-1), (v - gy).reshape(-1)).reshape(H, W, 3)
    valid = torch.ones(H, W)

    for name, stored, expect in (("identity", flow, 1.0), ("negated", -flow, -1.0)):
        sel = select_pixels(f1, stored, valid, min_motion_px=1.0,
                            edge_quantile=0.9, max_abs_lat_deg=90.0)
        curve = alpha_residuals(f1, f2, stored, sel, alphas)
        assert curve is not None, f"{name}: no usable pixels"
        found = alphas[int(np.argmin(curve))]
        print(f"{name:9s} stored flow -> argmin alpha = {found:+.2f} "
              f"(R(-1)={curve[alphas.index(-1.0)]:.2f}, R(0)={curve[alphas.index(0.0)]:.2f}, "
              f"R(+1)={curve[alphas.index(1.0)]:.2f})")
        assert found == expect, f"{name}: arbiter returned {found}, expected {expect}"
    print("self-test PASSED (a known convention is recovered, both signs).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-free flow sign-convention arbiter.")
    ap.add_argument("--shards", default="/data/shards")
    ap.add_argument("--sources", default="flow360:test")
    ap.add_argument("--directions", default="both", choices=["both", "forward", "backward"])
    ap.add_argument("--alphas", default="-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5")
    ap.add_argument("--min-motion-px", type=float, default=2.0,
                    help="Skip pixels whose stored |flow| is below this. Sub-pixel "
                         "motion cannot discriminate a sign (P0: warp explains ~nothing).")
    ap.add_argument("--edge-quantile", type=float, default=0.9,
                    help="Keep only the top (1-q) of |grad frame1| within the motion gate.")
    ap.add_argument("--max-abs-lat-deg", type=float, default=75.0,
                    help="Drop extreme ERP rows, where a px-space warp is degenerate.")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--output-dir", default="/outputs/constancy_arbiter")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for required in (-1.0, 0.0, 1.0):
        if required not in alphas:
            raise ValueError(f"--alphas must contain {required}")

    if args.self_test:
        self_test(alphas)
        return

    sources = parse_sources(args.sources)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves: Dict[str, List[np.ndarray]] = defaultdict(list)
    skipped = 0
    seen = 0
    start = time.time()

    for record in iter_source_records(Path(args.shards), sources, args.directions):
        f1 = _to_chw_free_float(record["frame1"])
        f2 = _to_chw_free_float(record["frame2"])
        flow = torch.from_numpy(np.ascontiguousarray(record["flow"])).float()
        valid = torch.from_numpy(np.ascontiguousarray(record["valid"]))
        sel = select_pixels(f1, flow, valid, args.min_motion_px,
                            args.edge_quantile, args.max_abs_lat_deg)
        curve = alpha_residuals(f1, f2, flow, sel, alphas)
        seen += 1
        if curve is None:
            skipped += 1
        else:
            curves[record["meta"].get("direction", "unknown")].append(curve)
        if args.max_pairs is not None and seen >= args.max_pairs:
            break

    if not curves:
        raise RuntimeError(f"no pair carried >=64 gated pixels (seen {seen}); "
                           "lower --min-motion-px or --edge-quantile")

    report = {d: summarize(c, alphas) for d, c in sorted(curves.items())}
    report["_meta"] = {
        "args": vars(args), "sources": sources, "pairs_seen": seen,
        "pairs_skipped_no_gated_pixels": skipped,
        "elapsed_s": round(time.time() - start, 1),
    }
    print(json.dumps(report, indent=2))
    with open(out_dir / "constancy_arbiter.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved={out_dir / 'constancy_arbiter.json'}")


if __name__ == "__main__":
    main()
