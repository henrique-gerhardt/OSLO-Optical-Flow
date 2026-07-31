from typing import Dict, Optional

import numpy as np
import torch

from .geometry import endpoint_from_tangent_flow, geodesic_distance


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    if weights is None:
        weights = torch.ones_like(values)
    weights = weights.to(device=values.device, dtype=values.dtype) * mask_f
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def parse_thresholds(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_bands(text: str) -> list[tuple[float, float]]:
    """Parse ``"0,0.25,0.5,inf"`` into consecutive half-open [lo, hi) motion bands.

    Bands complement the ``active_X`` thresholds, which are *cumulative tails*
    (``zero_geo_deg >= X``) and therefore mix regimes: ``active_0.25`` on a
    large-motion set is dominated by the 70 px pairs, not by the 0.25 deg ones.
    Disjoint bands are the instrument for locating the displacement at which a
    method starts beating the zero baseline (the "crossing point").
    """
    if not text.strip():
        return []
    edges = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(edges) < 2:
        raise ValueError("motion bands need at least two edges, e.g. '0,0.25,inf'")
    if any(hi <= lo for lo, hi in zip(edges, edges[1:])):
        raise ValueError(f"motion band edges must be strictly increasing: {edges}")
    return list(zip(edges, edges[1:]))


def band_key(lo: float, hi: float) -> str:
    def fmt(value: float) -> str:
        return "inf" if value == float("inf") else str(value).replace(".", "_")

    return f"band_{fmt(lo)}_{fmt(hi)}"


def build_region_masks(points: torch.Tensor, seam_width_deg: float = 15.0) -> Dict[str, torch.Tensor]:
    # Boundary decisions in float64 with an include-the-boundary tolerance: HEALPix
    # grids place node columns exactly ON region boundaries (44 nodes at the r6 seam
    # edge, fp32 margin ~1e-7 rad; one full ring exactly at |lat| = 30 deg, fp64 margin
    # ~1e-16), where a one-ulp CPU-vs-CUDA atan2/asin difference flips them all at once
    # (measured: ~0.03 deg seam-mean skew between devices). All true margins are either
    # < 1e-15 (mathematically on the boundary) or > 7e-11 across r4-r7, so eps = 1e-12
    # decides every node deterministically on any device.
    eps = 1e-12
    x, y, z = points.double().unbind(dim=-1)
    lon = torch.atan2(y, x)
    lat = torch.asin(z.clamp(-1.0, 1.0))
    return {
        "global": torch.ones(points.size(0), dtype=torch.bool, device=points.device),
        "poles": lat.abs() >= np.deg2rad(60.0) - eps,
        "equator": lat.abs() <= np.deg2rad(30.0) + eps,
        "seam": (torch.pi - lon.abs()) <= np.deg2rad(seam_width_deg) + eps,
    }


def compute_maps(
    pred_flow: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_endpoint = endpoint_from_tangent_flow(points, pred_flow, basis_east, basis_north)
    target_endpoint = batch["endpoint"]
    geo = geodesic_distance(pred_endpoint, target_endpoint)
    tangent_epe = (pred_flow - batch["flow"]).norm(dim=-1)
    zero_endpoint = points.unsqueeze(0).expand_as(target_endpoint)
    zero_geo = geodesic_distance(zero_endpoint, target_endpoint)
    valid = batch.get("valid")
    if valid is None:
        valid = torch.ones_like(geo, dtype=torch.bool)
    return {
        "geo_rad": geo,
        "geo_deg": geo * (180.0 / torch.pi),
        "tangent_epe_rad": tangent_epe,
        "zero_geo_deg": zero_geo * (180.0 / torch.pi),
        "valid": valid.bool(),
    }


def compute_loss(
    pred_flow: torch.Tensor,
    batch: dict,
    points: torch.Tensor,
    basis_east: torch.Tensor,
    basis_north: torch.Tensor,
    loss_min_target_deg: float = 0.0,
    loss_motion_weight: float = 0.0,
    loss_motion_ref_deg: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    maps = compute_maps(pred_flow, batch, points, basis_east, basis_north)
    valid = maps["valid"]
    if loss_min_target_deg > 0.0:
        active = maps["zero_geo_deg"] >= loss_min_target_deg
        valid = valid & active
        if int(valid.sum().detach().cpu()) == 0:
            valid = maps["valid"]

    weights = None
    if loss_motion_weight > 0.0:
        ref = max(loss_motion_ref_deg, 1e-6)
        weights = 1.0 + loss_motion_weight * (maps["zero_geo_deg"] / ref).clamp(max=1.0)
    loss = weighted_masked_mean(maps["geo_rad"], valid, weights)
    return loss, maps


def active_key(threshold: float) -> str:
    return f"active_{str(threshold).replace('.', '_')}"


def add_improvement_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    for key, value in list(metrics.items()):
        if not key.endswith("_geo_deg") or key.endswith("_zero_geo_deg") or key.startswith("target_"):
            continue
        prefix = key[: -len("_geo_deg")]
        zero_key = f"{prefix}_zero_geo_deg"
        if zero_key not in metrics:
            continue
        zero = metrics[zero_key]
        improvement = zero - value
        metrics[f"{prefix}_improvement_deg"] = improvement
        metrics[f"{prefix}_improvement_pct"] = 100.0 * improvement / zero if abs(zero) > 1e-12 else 0.0
    return metrics


def summarize_maps(
    maps: Dict[str, torch.Tensor],
    region_masks: Dict[str, torch.Tensor],
    active_thresholds: list[float],
    motion_bands: list[tuple[float, float]] = (),
    node_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """One-shot twin of :func:`accumulate_maps`; ``node_weights`` has the same meaning."""
    out: Dict[str, float] = {}
    valid = maps["valid"]

    def mass(mask: torch.Tensor) -> float:
        if node_weights is None:
            return float(mask.sum().item())
        w = node_weights.to(device=mask.device, dtype=maps["geo_deg"].dtype)
        return float((mask.to(dtype=w.dtype) * w).sum().item())

    total = max(mass(valid), 1e-12)
    for region_name, region_mask in region_masks.items():
        mask = valid & region_mask.unsqueeze(0)
        count = mass(mask)
        if count == 0.0:
            continue
        prefix = f"{region_name}_"
        out[prefix + "count"] = count
        for name in ("geo_deg", "zero_geo_deg", "tangent_epe_rad"):
            out[prefix + name] = float(
                weighted_masked_mean(maps[name], mask, node_weights).detach().cpu())
    selections = [(active_key(t), valid & (maps["zero_geo_deg"] >= t)) for t in active_thresholds]
    selections += [
        (band_key(lo, hi), valid & (maps["zero_geo_deg"] >= lo) & (maps["zero_geo_deg"] < hi))
        for lo, hi in motion_bands
    ]
    for key, mask in selections:
        count = mass(mask)
        prefix = key + "_"
        out[prefix + "count"] = count
        out[prefix + "frac"] = count / total
        if count > 0.0:
            for name in ("geo_deg", "zero_geo_deg"):
                out[prefix + name] = float(
                    weighted_masked_mean(maps[name], mask, node_weights).detach().cpu())
    return add_improvement_metrics(out)


def target_sample_from_maps(
    maps: Dict[str, torch.Tensor],
    sample_count: Optional[int],
) -> torch.Tensor:
    target = maps["zero_geo_deg"][maps["valid"]].detach().float().flatten().cpu()
    if sample_count is not None and target.numel() > sample_count:
        sample_idx = torch.linspace(0, target.numel() - 1, steps=sample_count, dtype=torch.long)
        target = target.index_select(0, sample_idx)
    return target


def accumulate_maps(
    maps: Dict[str, torch.Tensor],
    region_masks: Dict[str, torch.Tensor],
    active_thresholds: list[float],
    totals: Dict[str, float],
    counts: Dict[str, float],
    active_counts: Dict[str, float],
    motion_bands: list[tuple[float, float]] = (),
    node_weights: Optional[torch.Tensor] = None,
) -> None:
    """Stream masked sums into ``totals``/``counts``.

    ``node_weights`` is a ``[N]`` per-node solid angle normalized to mean 1, which turns
    every mean and every ``_frac`` into a per-area quantity. Pass it whenever the grid is
    not equal-area: an unweighted node mean answers "average over this grid's nodes",
    which is a different question on each grid and is not comparable across them. ``None``
    keeps the plain node mean, which already equals the per-area mean on HEALPix.
    """
    valid = maps["valid"]

    def weigh(mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(dtype=maps["geo_deg"].dtype)
        if node_weights is None:
            return w
        return w * node_weights.to(device=w.device, dtype=w.dtype)

    for region_name, region_mask in region_masks.items():
        w = weigh(valid & region_mask.unsqueeze(0))
        count = float(w.sum().item())
        if count == 0.0:
            continue
        for metric_name in ("geo_deg", "zero_geo_deg", "tangent_epe_rad"):
            key = f"{region_name}_{metric_name}"
            totals[key] = totals.get(key, 0.0) + float((maps[metric_name] * w).sum().detach().cpu())
            counts[key] = counts.get(key, 0.0) + count

    selections = [(active_key(t), valid & (maps["zero_geo_deg"] >= t)) for t in active_thresholds]
    selections += [
        (band_key(lo, hi), valid & (maps["zero_geo_deg"] >= lo) & (maps["zero_geo_deg"] < hi))
        for lo, hi in motion_bands
    ]
    for prefix, mask in selections:
        w = weigh(mask)
        count = float(w.sum().item())
        active_counts[prefix] = active_counts.get(prefix, 0.0) + count
        if count == 0.0:
            continue
        for metric_name in ("geo_deg", "zero_geo_deg", "tangent_epe_rad"):
            key = f"{prefix}_{metric_name}"
            totals[key] = totals.get(key, 0.0) + float((maps[metric_name] * w).sum().detach().cpu())
            counts[key] = counts.get(key, 0.0) + count


def finalize_metrics(
    totals: Dict[str, float],
    counts: Dict[str, float],
    active_counts: Dict[str, float],
    target_chunks: list[torch.Tensor],
) -> Dict[str, float]:
    metrics = {key: totals[key] / max(counts[key], 1.0) for key in sorted(totals)}
    global_count = max(counts.get("global_geo_deg", 0.0), 1.0)
    for prefix, count in active_counts.items():
        metrics[f"{prefix}_frac"] = count / global_count
    if target_chunks:
        # np.quantile (not torch.quantile) — torch hard-errors above 2**24 elements, which
        # the r=6 supervision grid (~49k nodes x hundreds of val pairs) blows past; numpy
        # has no such cap, so this stays exact at any resolution / pair count.
        target = torch.cat(target_chunks).numpy()
        p50, p90, p95 = np.quantile(target, [0.50, 0.90, 0.95])
        metrics["target_geo_deg_p50"] = float(p50)
        metrics["target_geo_deg_p90"] = float(p90)
        metrics["target_geo_deg_p95"] = float(p95)
        metrics["target_geo_deg_quantile_samples"] = float(target.size)
    return add_improvement_metrics(metrics)


def print_metrics(
    prefix: str,
    metrics: Dict[str, float],
    motion_bands: list[tuple[float, float]] = (),
) -> None:
    keys = [
        "global_geo_deg",
        "global_zero_geo_deg",
        "global_improvement_pct",
        "target_geo_deg_p50",
        "target_geo_deg_p90",
        "poles_geo_deg",
        "poles_zero_geo_deg",
        "poles_improvement_pct",
        "equator_geo_deg",
        "equator_zero_geo_deg",
        "equator_improvement_pct",
        "seam_geo_deg",
        "seam_zero_geo_deg",
        "seam_improvement_pct",
        "active_0_25_frac",
        "active_0_25_geo_deg",
        "active_0_25_zero_geo_deg",
        "active_0_25_improvement_pct",
        "active_0_5_frac",
        "active_0_5_geo_deg",
        "active_0_5_zero_geo_deg",
        "active_0_5_improvement_pct",
        "active_1_0_frac",
        "active_1_0_geo_deg",
        "active_1_0_zero_geo_deg",
        "active_1_0_improvement_pct",
    ]
    # Bands are emitted in edge order (not sorted), so the log line reads as a curve.
    for lo, hi in motion_bands:
        band = band_key(lo, hi)
        keys += [f"{band}_frac", f"{band}_zero_geo_deg", f"{band}_geo_deg",
                 f"{band}_improvement_pct"]
    items = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    print(f"{prefix} " + " ".join(items), flush=True)
