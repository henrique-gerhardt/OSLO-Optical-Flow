"""Measure the gauge inconsistency of the east/north frame, before spending GPU on a fix.

`tangent_basis` builds an east/north frame per node from `cross(z, p)`. That frame is
not parallel-transported between neighbours, and `MotionEncoder.flow_conv`
(`SDPAConv(2, 32, ...)`) aggregates the 2-channel tangent flow over the conv
neighbourhood **without** transporting it, even though `parallel_transport` exists in
the codebase and is used inside `convex_upsample`. So type-1 features from nodes with
different frames are summed as if the frames agreed.

The open question is whether that costs anything measurable, and specifically whether
it can explain the 29.3% SO(3) degradation that is currently attributed to in-domain
training. It can only explain a *rotation* effect if the damage is **latitude
dependent**: rotation moves scene content to different latitudes, so a defect that is
flat in latitude cannot produce a rotation gap, however large it is in absolute terms.

Two measurements, both training-free:

  A. **Frame holonomy.** For every conv edge i->j, transport the neighbour's east
     vector to node i and measure the angle it makes with node i's own east. This is
     the primitive: if it is ~0 everywhere the whole concern is void, and the
     correlation-stencil orientation (which inherits the same frame) is fine too.

  B. **Aggregation error.** Take a tangent field, aggregate it over the conv
     neighbourhood the way the model does (raw component sum) and the way geometry
     says it should be done (transport, then sum), and report the discrepancy
     relative to the local field magnitude. With `--shards` the field is the real
     ground-truth flow; without it, a global-rotation field, which is smooth and
     therefore the *favourable* case for untransported aggregation.

Both are reported per latitude band. The pre-registered reading is in `verdict()`:
the expensive two-arm training (east/north head vs extrinsic R^3 head) is justified
only if the damage is both material and latitude-dependent.

    python run_gauge_diagnostic.py                                  # geometry only
    python run_gauge_diagnostic.py --shards /data/shards \
        --dataset flowscape --split test --max-pairs 64             # + real fields
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from spherical_flow.geometry import parallel_transport, tangent_basis
from spherical_flow.healpix_pyramid import build_healpix_pyramid

BANDS = ((0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 75.0), (75.0, 90.0))


def latitude_deg(points: torch.Tensor) -> torch.Tensor:
    return torch.asin(points[:, 2].clamp(-1.0, 1.0)) * (180.0 / math.pi)


def by_band(values: torch.Tensor, lat: torch.Tensor) -> list[dict]:
    """Per-|latitude| summary. HEALPix cells are equal area, so a plain mean is area weighted."""
    out = []
    a = lat.abs()
    for lo, hi in BANDS:
        m = (a >= lo) & (a < hi) if hi < 90.0 else (a >= lo) & (a <= hi)
        if not bool(m.any()):
            continue
        v = values[m]
        out.append({
            "band": f"{lo:.0f}-{hi:.0f}",
            "nodes": int(m.sum()),
            "mean": float(v.mean()),
            "p50": float(v.median()),
            "p95": float(v.quantile(0.95)),
        })
    return out


def frame_holonomy(level) -> torch.Tensor:
    """Angle, in degrees, between node i's east and its neighbours' east transported to i."""
    p = level.points                                  # [N, 3]
    idx = level.conv_index                            # [N, K]
    valid = level.conv_valid                          # [N, K]
    e_i = level.basis_east.unsqueeze(1)               # [N, 1, 3]
    n_i = level.basis_north.unsqueeze(1)
    p_i = p.unsqueeze(1)
    e_j = level.basis_east[idx]                       # [N, K, 3]
    p_j = p[idx]

    e_ji = parallel_transport(e_j, p_j, p_i.expand_as(e_j))
    x = (e_ji * e_i).sum(-1)
    y = (e_ji * n_i).sum(-1)
    ang = torch.atan2(y, x).abs() * (180.0 / math.pi)  # [N, K]
    ang = torch.where(valid, ang, torch.zeros_like(ang))
    denom = valid.sum(-1).clamp_min(1)
    return ang.sum(-1) / denom                         # mean over a node's edges


def aggregation_error(level, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrepancy between the model's untransported neighbourhood sum and the transported one.

    `flow` is [B, N, 2] tangent components. Uses the SDPAConv weights so the operation
    matches what `flow_conv` actually computes, minus the learned channel mixing (which
    is a per-channel linear map and cannot repair a frame mismatch).
    """
    p = level.points
    idx = level.conv_index
    w = level.conv_weight * level.conv_valid.float()   # [N, K]
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)

    e_j = level.basis_east[idx]                        # [N, K, 3]
    n_j = level.basis_north[idx]
    f_j = flow[:, idx, :]                              # [B, N, K, 2]
    t_j = f_j[..., 0:1] * e_j + f_j[..., 1:2] * n_j    # [B, N, K, 3] ambient tangent

    # What the model does: sum the raw (east, north) components, call the result a
    # vector in node i's frame.
    raw = (w.unsqueeze(0).unsqueeze(-1) * f_j).sum(dim=2)          # [B, N, 2]

    # What geometry says: transport each neighbour's tangent vector to i, then sum.
    p_i = p.reshape(1, -1, 1, 3)
    t_ji = parallel_transport(t_j, p[idx].unsqueeze(0), p_i.expand_as(t_j))
    pooled = (w.unsqueeze(0).unsqueeze(-1) * t_ji).sum(dim=2)      # [B, N, 3]
    e_i, n_i = level.basis_east.unsqueeze(0), level.basis_north.unsqueeze(0)
    pt = torch.stack([(pooled * e_i).sum(-1), (pooled * n_i).sum(-1)], dim=-1)

    diff = (raw - pt).norm(dim=-1)                                 # [B, N]
    scale = flow.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-8)
    return diff.mean(0), (diff / scale).mean(0)


def rotation_field(points: torch.Tensor, axis=(0.0, 0.0, 1.0), rate_deg: float = 2.0):
    """A global-rotation tangent field: smooth everywhere, the easy case for aggregation."""
    ax = torch.tensor(axis, dtype=points.dtype).reshape(1, 3)
    v = torch.cross(ax.expand_as(points), points, dim=-1) * (rate_deg * math.pi / 180.0)
    e, n = tangent_basis(points)
    return torch.stack([(v * e).sum(-1), (v * n).sum(-1)], dim=-1).unsqueeze(0)


def real_fields(args, points: torch.Tensor) -> torch.Tensor | None:
    from torch.utils.data import DataLoader
    from spherical_flow.shard_dataset import ShardFlowDataset
    ds = ShardFlowDataset(args.shards, points, (args.dataset, args.split),
                          shuffle_shards=False, max_pairs=args.max_pairs)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    chunks = []
    seen = 0
    for batch in loader:
        chunks.append(batch["flow"].float())
        seen += batch["flow"].size(0)
        if seen >= args.max_pairs:
            break
    return torch.cat(chunks, dim=0) if chunks else None


def verdict(holonomy_bands, agg_bands) -> dict:
    """Pre-registered reading, declared before the run.

    The two-arm training is justified only if BOTH hold:
      1. material   -- polar (|lat| > 60) median relative aggregation error >= 0.10;
      2. structured -- that polar median is >= 2x the equatorial (|lat| < 30) median.
    Condition 2 is the one that matters for the SO(3) attribution: a flat defect
    cannot produce a rotation gap.
    """
    def pick(bands, names):
        v = [b["p50"] for b in bands if b["band"] in names]
        return sum(v) / len(v) if v else float("nan")
    pol = pick(agg_bands, {"60-75", "75-90"})
    eqt = pick(agg_bands, {"0-15", "15-30"})
    ratio = pol / eqt if eqt else float("inf")
    material = pol >= 0.10
    structured = ratio >= 2.0
    return {
        "polar_rel_agg_error_p50": pol,
        "equatorial_rel_agg_error_p50": eqt,
        "polar_over_equatorial": ratio,
        "polar_holonomy_deg_p50": pick(holonomy_bands, {"60-75", "75-90"}),
        "equatorial_holonomy_deg_p50": pick(holonomy_bands, {"0-15", "15-30"}),
        "material": bool(material),
        "structured": bool(structured),
        "two_arm_training_justified": bool(material and structured),
    }


def table(title, rows, unit):
    print(f"\n{title}")
    print(f"  {'|lat|':>8}  {'nós':>7}  {'média':>9}  {'p50':>9}  {'p95':>9}   ({unit})")
    for r in rows:
        print(f"  {r['band']:>8}  {r['nodes']:>7}  {r['mean']:>9.4f}  {r['p50']:>9.4f}  {r['p95']:>9.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimation-resolution", type=int, default=4)
    ap.add_argument("--fine-resolution", type=int, default=6)
    ap.add_argument("--conv-neighbors", type=int, default=8)
    ap.add_argument("--shards", default="", help="shards dir; omit to use a synthetic rotation field")
    ap.add_argument("--dataset", default="flowscape")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-pairs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", default="outputs/gauge_diagnostic")
    args = ap.parse_args()

    torch.manual_seed(0)
    pyr = build_healpix_pyramid(fine_resolution=args.fine_resolution,
                                estimation_resolution=args.estimation_resolution,
                                conv_neighbors=args.conv_neighbors)
    est = pyr.estimation_level
    lat = latitude_deg(est.points)
    print(f"grade de estimação r={args.estimation_resolution}: {est.points.size(0)} nós, "
          f"{est.conv_index.size(1)} vizinhos por nó")

    hol = frame_holonomy(est)
    hol_bands = by_band(hol, lat)
    table("A. Holonomia do referencial (ângulo entre o leste local e o leste do vizinho transportado)",
          hol_bands, "graus")

    if args.shards:
        flow = real_fields(args, est.points)
        source = f"{args.dataset}:{args.split} ({0 if flow is None else flow.size(0)} pares)"
        if flow is None:
            raise SystemExit("nenhum par lido dos shards")
    else:
        flow = rotation_field(est.points)
        source = "campo sintético de rotação global (caso favorável)"

    _, rel = aggregation_error(est, flow)
    agg_bands = by_band(rel, lat)
    table(f"B. Erro de agregação sem transporte, relativo à magnitude do campo — {source}",
          agg_bands, "fração")

    v = verdict(hol_bands, agg_bands)
    print("\nLeitura pré-registrada")
    print(f"  erro relativo polar (p50)      {v['polar_rel_agg_error_p50']:.4f}   "
          f"(material se >= 0,10: {'sim' if v['material'] else 'não'})")
    print(f"  erro relativo equatorial (p50) {v['equatorial_rel_agg_error_p50']:.4f}")
    print(f"  razão polos/equador            {v['polar_over_equatorial']:.2f}    "
          f"(estruturado se >= 2,0: {'sim' if v['structured'] else 'não'})")
    print(f"\n  => treino em dois braços justificado: "
          f"{'SIM' if v['two_arm_training_justified'] else 'NAO'}")
    if not v["two_arm_training_justified"]:
        print("     o gauge não é carga; a atribuição atual do SO(3) sobrevive sem o experimento.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "source": source,
               "holonomy_deg": hol_bands, "relative_aggregation_error": agg_bands,
               "verdict": v}
    (out / "gauge_diagnostic.json").write_text(json.dumps(payload, indent=2))
    print(f"\nescrito em {out / 'gauge_diagnostic.json'}")


if __name__ == "__main__":
    main()
