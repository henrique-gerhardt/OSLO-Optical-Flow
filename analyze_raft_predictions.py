import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Flow360Pair:
    sequence: str
    direction: str
    frame1: Path
    frame2: Path
    flow: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved RAFT ERP predictions against FLOW360 ground-truth pixel flow."
    )
    parser.add_argument("--data-root", default="/data/flow360", help="FLOW360 root with train/test folders.")
    parser.add_argument("--output-dir", default="/outputs/raft_r6_forward_debug", help="RAFT baseline output dir.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--metrics-out", default="", help="Optional diagnostic JSON path.")
    return parser.parse_args()


def load_flow(path: Path, flow_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    flow = np.load(path)
    if flow.ndim != 3:
        raise ValueError(f"Expected 3D flow array, got shape={flow.shape} from {path}")
    if flow.shape[0] in (2, 3) and flow.shape[-1] not in (2, 3):
        flow = np.moveaxis(flow, 0, -1)
    if flow.shape[-1] < 2:
        raise ValueError(f"Expected flow last dimension >= 2, got shape={flow.shape} from {path}")
    flow = flow[..., :2].astype(np.float32, copy=False) * float(flow_scale)
    finite = np.isfinite(flow).all(axis=-1)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return flow, finite


def sorted_files(folder: Path, suffix: str) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == suffix)


def discover_pairs(
    root: str | Path,
    split: str,
    direction: str = "forward",
    max_pairs: Optional[int] = None,
) -> list[Flow360Pair]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"FLOW360 split not found: {split_dir}")

    pairs: list[Flow360Pair] = []
    for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        frames = sorted_files(seq_dir / "frames", ".png")
        frame_by_stem = {path.stem: path for path in frames}
        stems = [path.stem for path in frames]

        if direction in ("forward", "both"):
            for idx, stem in enumerate(stems[:-1]):
                flow_path = seq_dir / "fflows" / f"{stem}.npy"
                if flow_path.is_file():
                    pairs.append(
                        Flow360Pair(
                            sequence=seq_dir.name,
                            direction="forward",
                            frame1=frame_by_stem[stem],
                            frame2=frame_by_stem[stems[idx + 1]],
                            flow=flow_path,
                        )
                    )

        if direction in ("backward", "both"):
            for idx, stem in enumerate(stems[1:], start=1):
                flow_path = seq_dir / "bflows" / f"{stem}.npy"
                if flow_path.is_file():
                    pairs.append(
                        Flow360Pair(
                            sequence=seq_dir.name,
                            direction="backward",
                            frame1=frame_by_stem[stem],
                            frame2=frame_by_stem[stems[idx - 1]],
                            flow=flow_path,
                        )
                    )

        if max_pairs is not None and len(pairs) >= max_pairs:
            return pairs[:max_pairs]

    if not pairs:
        raise RuntimeError(f"No FLOW360 pairs found in {split_dir} with direction={direction}")
    return pairs


def pred_path(output_dir: Path, split: str, pair: Flow360Pair) -> Path:
    return output_dir / "predictions" / split / pair.sequence / pair.direction / f"{pair.frame1.stem}.npy"


def candidate_transforms(flow: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    x = flow[..., 0]
    y = flow[..., 1]
    yield "identity", np.stack([x, y], axis=-1)
    yield "negated", np.stack([-x, -y], axis=-1)
    yield "negate_x", np.stack([-x, y], axis=-1)
    yield "negate_y", np.stack([x, -y], axis=-1)
    yield "swap_xy", np.stack([y, x], axis=-1)
    yield "swap_xy_negated", np.stack([-y, -x], axis=-1)
    yield "swap_xy_negate_x", np.stack([-y, x], axis=-1)
    yield "swap_xy_negate_y", np.stack([y, -x], axis=-1)


def sample_values(values: np.ndarray, max_samples: Optional[int]) -> np.ndarray:
    values = values.reshape(-1)
    if max_samples is not None and max_samples > 0 and values.size > max_samples:
        idx = np.linspace(0, values.size - 1, num=max_samples, dtype=np.int64)
        values = values[idx]
    return values.astype(np.float32, copy=False)


def empty_fit() -> Dict[str, float]:
    return {"dot": 0.0, "pred_sq": 0.0, "gt_sq": 0.0}


def update_fit(acc: Dict[str, float], pred: np.ndarray, gt: np.ndarray) -> None:
    acc["dot"] += float((pred * gt).sum(dtype=np.float64))
    acc["pred_sq"] += float((pred * pred).sum(dtype=np.float64))
    acc["gt_sq"] += float((gt * gt).sum(dtype=np.float64))


def fitted_scale(acc: Dict[str, float]) -> float:
    if acc["pred_sq"] <= 1e-12:
        return 0.0
    return acc["dot"] / acc["pred_sq"]


def empty_metric() -> Dict[str, float]:
    return {
        "count": 0.0,
        "epe_sum": 0.0,
        "pred_mag_sum": 0.0,
        "gt_mag_sum": 0.0,
        "dot": 0.0,
        "pred_sq": 0.0,
        "gt_sq": 0.0,
    }


def update_metric(acc: Dict[str, float], pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = pred - gt
    epe = np.linalg.norm(diff, axis=-1)
    pred_mag = np.linalg.norm(pred, axis=-1)
    gt_mag = np.linalg.norm(gt, axis=-1)
    acc["count"] += float(epe.size)
    acc["epe_sum"] += float(epe.sum(dtype=np.float64))
    acc["pred_mag_sum"] += float(pred_mag.sum(dtype=np.float64))
    acc["gt_mag_sum"] += float(gt_mag.sum(dtype=np.float64))
    update_fit(acc, pred, gt)
    return epe


def finalize_metric(acc: Dict[str, float], zero_mean: float, samples: list[np.ndarray]) -> Dict[str, float]:
    count = max(acc["count"], 1.0)
    mean_epe = acc["epe_sum"] / count
    cosine = 0.0
    denom = np.sqrt(max(acc["pred_sq"], 0.0) * max(acc["gt_sq"], 0.0))
    if denom > 1e-12:
        cosine = acc["dot"] / denom
    out = {
        "mean_epe_px": mean_epe,
        "improvement_px": zero_mean - mean_epe,
        "improvement_pct": 100.0 * (zero_mean - mean_epe) / zero_mean if zero_mean > 1e-12 else 0.0,
        "pred_mag_mean_px": acc["pred_mag_sum"] / count,
        "gt_mag_mean_px": acc["gt_mag_sum"] / count,
        "cosine": cosine,
    }
    if samples:
        values = np.concatenate(samples)
        out["epe_p50_px"] = float(np.quantile(values, 0.50))
        out["epe_p90_px"] = float(np.quantile(values, 0.90))
        out["epe_p95_px"] = float(np.quantile(values, 0.95))
        out["sample_count"] = float(values.size)
    return out


def load_pairs(args: argparse.Namespace, output_dir: Path) -> list[Flow360Pair]:
    pairs = discover_pairs(args.data_root, args.split, args.direction, max_pairs=args.max_pairs)
    existing = [pair for pair in pairs if pred_path(output_dir, args.split, pair).is_file()]
    if not existing:
        raise FileNotFoundError(f"No saved predictions found under {output_dir / 'predictions'}")
    return existing


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pairs = load_pairs(args, output_dir)
    sample_per_pair = None
    if args.max_samples > 0:
        sample_per_pair = max(1, args.max_samples // max(len(pairs), 1))

    fit: dict[str, Dict[str, float]] = {}
    zero_acc = empty_metric()
    zero_samples: list[np.ndarray] = []

    for pair in pairs:
        pred, pred_valid = load_flow(pred_path(output_dir, args.split, pair), flow_scale=1.0)
        gt, gt_valid = load_flow(pair.flow, flow_scale=args.flow_scale)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape} for {pair.flow}")
        mask = pred_valid & gt_valid
        pred_valid_flow = pred[mask]
        gt_valid_flow = gt[mask]
        zero_epe = update_metric(zero_acc, np.zeros_like(gt_valid_flow), gt_valid_flow)
        zero_samples.append(sample_values(zero_epe, sample_per_pair))
        for name, transformed in candidate_transforms(pred):
            fit.setdefault(name, empty_fit())
            update_fit(fit[name], transformed[mask], gt_valid_flow)

    zero_mean = zero_acc["epe_sum"] / max(zero_acc["count"], 1.0)
    scales = {name: fitted_scale(acc) for name, acc in fit.items()}
    raw_metrics = {name: empty_metric() for name in fit}
    scaled_metrics = {name: empty_metric() for name in fit}
    raw_samples = {name: [] for name in fit}
    scaled_samples = {name: [] for name in fit}

    for pair in pairs:
        pred, pred_valid = load_flow(pred_path(output_dir, args.split, pair), flow_scale=1.0)
        gt, gt_valid = load_flow(pair.flow, flow_scale=args.flow_scale)
        mask = pred_valid & gt_valid
        gt_valid_flow = gt[mask]
        for name, transformed in candidate_transforms(pred):
            pred_valid_flow = transformed[mask]
            raw_epe = update_metric(raw_metrics[name], pred_valid_flow, gt_valid_flow)
            scaled_epe = update_metric(scaled_metrics[name], scales[name] * pred_valid_flow, gt_valid_flow)
            raw_samples[name].append(sample_values(raw_epe, sample_per_pair))
            scaled_samples[name].append(sample_values(scaled_epe, sample_per_pair))

    zero = finalize_metric(zero_acc, zero_mean, zero_samples)
    candidates = {}
    for name in sorted(fit):
        candidates[name] = finalize_metric(raw_metrics[name], zero_mean, raw_samples[name])
        candidates[f"{name}_scaled"] = finalize_metric(scaled_metrics[name], zero_mean, scaled_samples[name])
        candidates[f"{name}_scaled"]["scale"] = scales[name]

    best_raw_name = min(fit, key=lambda name: candidates[name]["mean_epe_px"])
    best_scaled_name = min(fit, key=lambda name: candidates[f"{name}_scaled"]["mean_epe_px"])
    result = {
        "args": vars(args),
        "pairs": len(pairs),
        "zero_flow": zero,
        "best_raw": {"name": best_raw_name, **candidates[best_raw_name]},
        "best_scaled": {"name": f"{best_scaled_name}_scaled", **candidates[f"{best_scaled_name}_scaled"]},
        "candidates": candidates,
    }

    metrics_out = Path(args.metrics_out) if args.metrics_out else output_dir / "raft_prediction_diagnostic.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"pairs={len(pairs)}")
    print(f"zero_flow mean_epe_px={zero['mean_epe_px']:.4f}")
    print(
        "identity "
        f"mean_epe_px={candidates['identity']['mean_epe_px']:.4f} "
        f"improvement_pct={candidates['identity']['improvement_pct']:.2f} "
        f"cosine={candidates['identity']['cosine']:.4f}"
    )
    print(
        "identity_scaled "
        f"scale={candidates['identity_scaled']['scale']:.4f} "
        f"mean_epe_px={candidates['identity_scaled']['mean_epe_px']:.4f} "
        f"improvement_pct={candidates['identity_scaled']['improvement_pct']:.2f}"
    )
    print(
        "best_raw "
        f"name={result['best_raw']['name']} "
        f"mean_epe_px={result['best_raw']['mean_epe_px']:.4f} "
        f"improvement_pct={result['best_raw']['improvement_pct']:.2f}"
    )
    print(
        "best_scaled "
        f"name={result['best_scaled']['name']} "
        f"scale={result['best_scaled']['scale']:.4f} "
        f"mean_epe_px={result['best_scaled']['mean_epe_px']:.4f} "
        f"improvement_pct={result['best_scaled']['improvement_pct']:.2f}"
    )
    print(f"saved_metrics={metrics_out}")


if __name__ == "__main__":
    main()
