"""Aggregate metric JSONs from training/eval runs into one comparison table.

Every runner writes a JSON whose metric blocks contain ``global_geo_deg`` (see
spherical_flow/metrics.py): the MVP under ``metrics``, the residual run under
``raft_metrics`` + ``residual_metrics``, and future OSLO-RAFT runners under their
own keys. This script discovers those blocks generically, collapses repeated seeds
into mean +/- std, derives the shared zero-flow reference, and prints the headline
table the plan's gate decisions need (Markdown, plus optional CSV).

Usage:
    python run_aggregate_results.py --root outputs
    python run_aggregate_results.py --root outputs --csv results.csv --markdown results.md
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default headline columns (the plan's reference-baseline metrics).
HEADLINE = [
    "global_geo_deg",
    "poles_geo_deg",
    "seam_geo_deg",
    "active_0_5_geo_deg",
    "global_improvement_pct",
]

# Friendly method names for well-known block keys; anything else falls back to the
# run's directory name.
BLOCK_METHODS = {
    "raft_metrics": "frozen RAFT",
    "residual_metrics": "RAFT+residual",
}

_SEED_TOKEN = re.compile(r"[_-](?:seed|s)[_-]?\d+", re.IGNORECASE)


def _clean_label(name: str) -> str:
    return _SEED_TOKEN.sub("", name).strip("_-") or name


def _find_blocks(payload: dict) -> List[Tuple[str, dict]]:
    """Return (block_key, metrics_dict) for every metric block in a loaded JSON."""
    blocks: List[Tuple[str, dict]] = []
    if "global_geo_deg" in payload:  # the whole file is a metrics dict
        blocks.append(("metrics", payload))
    for key, val in payload.items():
        if isinstance(val, dict) and "global_geo_deg" in val:
            blocks.append((key, val))
    return blocks


def _seed_of(args: dict, path: Path) -> Optional[int]:
    if isinstance(args, dict) and args.get("seed") is not None:
        return int(args["seed"])
    m = re.search(r"(?:seed|_s)[_-]?(\d+)", path.parent.name, re.IGNORECASE)
    return int(m.group(1)) if m else None


class Record:
    def __init__(self, method: str, resolution: str, seed: Optional[int], metrics: dict, src: str):
        self.method = method
        self.resolution = resolution
        self.seed = seed
        self.metrics = metrics
        self.src = src


def collect(root: Path, pattern: str) -> List[Record]:
    records: List[Record] = []
    for path in sorted(root.rglob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", {})
        resolution = str(args.get("resolution", "?")) if isinstance(args, dict) else "?"
        seed = _seed_of(args, path)
        for block_key, metrics in _find_blocks(payload):
            if block_key == "args":
                continue
            if block_key in BLOCK_METHODS:
                method = BLOCK_METHODS[block_key]
            elif block_key == "metrics":
                method = _clean_label(path.parent.name)
            else:
                method = f"{_clean_label(path.parent.name)}:{block_key}"
            records.append(
                Record(method, resolution, seed, metrics, str(path.relative_to(root)))
            )
    return records


def _agg(values: List[float]) -> Tuple[float, float, int]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), 0.0, 0
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), std, len(vals)


def _zero_flow_rows(records: List[Record], headline: List[str]) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """Derive a zero-flow reference per resolution from any run's *_zero_geo_deg."""
    by_res: Dict[str, dict] = {}
    for r in records:
        z = {}
        for col in headline:
            if col.endswith("_geo_deg"):
                zkey = col[: -len("geo_deg")] + "zero_geo_deg"
                if zkey in r.metrics:
                    z[col] = r.metrics[zkey]
            elif col.endswith("_improvement_pct"):
                z[col] = 0.0
        if z:
            by_res.setdefault(r.resolution, z)  # first wins; identical across runs
    return {
        res: {col: (val, 0.0, 1) for col, val in z.items()}
        for res, z in by_res.items()
    }


def aggregate(records: List[Record], headline: List[str]):
    groups: Dict[Tuple[str, str], List[Record]] = {}
    for r in records:
        groups.setdefault((r.method, r.resolution), []).append(r)

    rows = []
    for (method, res), recs in groups.items():
        seeds = sorted({r.seed for r in recs if r.seed is not None})
        row = {"method": method, "r": res, "n": len(recs), "seeds": seeds}
        for col in headline:
            row[col] = _agg([r.metrics.get(col) for r in recs])
        rows.append(row)

    # zero-flow reference rows
    for res, cols in _zero_flow_rows(records, headline).items():
        row = {"method": "zero-flow (derived)", "r": res, "n": 1, "seeds": []}
        for col in headline:
            row[col] = cols.get(col, (float("nan"), 0.0, 0))
        rows.append(row)

    rows.sort(key=lambda x: (x["r"], _sort_key(x, headline[0])))
    return rows


def _sort_key(row, col):
    val = row[col][0]
    return val if not math.isnan(val) else float("inf")


def _fmt(agg: Tuple[float, float, int]) -> str:
    mean, std, n = agg
    if n == 0 or math.isnan(mean):
        return "-"
    return f"{mean:.4f}" if n <= 1 or std == 0.0 else f"{mean:.4f}±{std:.4f}"


def render_markdown(rows, headline: List[str]) -> str:
    head = ["method", "r", "n"] + headline
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for row in rows:
        cells = [row["method"], row["r"], str(row["n"])] + [_fmt(row[col]) for col in headline]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows, headline: List[str], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csvmod.writer(fh)
        writer.writerow(["method", "r", "n", "seeds"] + [f"{c}_mean" for c in headline] + [f"{c}_std" for c in headline])
        for row in rows:
            writer.writerow(
                [row["method"], row["r"], row["n"], ";".join(map(str, row["seeds"]))]
                + [f"{row[c][0]:.6f}" for c in headline]
                + [f"{row[c][1]:.6f}" for c in headline]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs", help="Directory to scan for metric JSONs.")
    parser.add_argument("--pattern", default="*.json", help="Glob for metric files (rglob).")
    parser.add_argument(
        "--metrics", nargs="*", default=HEADLINE, help="Headline metric columns to report."
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    records = collect(root, args.pattern)
    if not records:
        print(f"No metric blocks found under {root} (pattern {args.pattern}).")
        print("Runs write to /outputs in the container; copy them under this root to aggregate.")
        raise SystemExit(0)

    rows = aggregate(records, args.metrics)
    table = render_markdown(rows, args.metrics)
    print(f"Aggregated {len(records)} metric block(s) from {root}\n")
    print(table)

    if args.markdown:
        args.markdown.write_text(table + "\n")
        print(f"\nwrote {args.markdown}")
    if args.csv:
        write_csv(rows, args.metrics, args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
