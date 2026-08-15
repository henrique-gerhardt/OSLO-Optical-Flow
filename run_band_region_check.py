"""Self-check for region x motion-band cross-tabulation (`--motion-band-regions`).

Region masks and band masks were independent selections, so "polar error restricted
to nodes moving more than 2 degrees" could not be read from any run. `_selections`
now crosses them. This script proves the crossing on synthetic maps where every
answer is known in closed form, and — the part that actually matters — proves the
streaming path (`accumulate_maps`, used by every real run) agrees with the one-shot
path (`summarize_maps`, used by the training log).

Run inside Docker; no GPU needed.
"""

import sys

import numpy as np
import torch

from spherical_flow.metrics import (
    accumulate_maps,
    add_improvement_metrics,
    band_key,
    build_region_masks,
    finalize_metrics,
    parse_bands,
    summarize_maps,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def synthetic(n_lat: int = 60, n_lon: int = 120, seed: int = 0):
    """Points on a lat/lon grid plus maps with known per-node values."""
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(np.linspace(-89.0, 89.0, n_lat))
    lon = np.deg2rad(np.linspace(-179.0, 179.0, n_lon))
    lat, lon = np.meshgrid(lat, lon, indexing="ij")
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    points = torch.tensor(
        np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1),
        dtype=torch.float32)
    n = points.shape[0]
    # Two "pairs" in the batch so the streaming path has something to accumulate over.
    zero = torch.tensor(rng.uniform(0.0, 8.0, size=(2, n)), dtype=torch.float32)
    geo = zero * torch.tensor(rng.uniform(0.2, 1.4, size=(2, n)), dtype=torch.float32)
    maps = {
        "geo_deg": geo,
        "zero_geo_deg": zero,
        "tangent_epe_rad": geo * float(np.pi / 180.0),
        "valid": torch.ones(2, n, dtype=torch.bool),
    }
    return points, maps


def main() -> int:
    points, maps = synthetic()
    regions = build_region_masks(points)
    thresholds = [0.25, 0.5, 1.0]
    bands = parse_bands("0,1,2,4,inf")
    band_regions = ("poles", "equator")

    print("1. off by default — no new keys, existing keys untouched")
    base = summarize_maps(maps, regions, thresholds, bands)
    with_cross = summarize_maps(maps, regions, thresholds, bands, band_regions=band_regions)
    new_keys = set(with_cross) - set(base)
    check("no existing key was dropped", not (set(base) - set(with_cross)))
    check("existing values unchanged",
          all(close(base[k], with_cross[k]) for k in base),
          f"{len(base)} keys compared")
    expected_new = {f"{r}_{band_key(lo, hi)}_{suffix}"
                    for r in band_regions for lo, hi in bands
                    for suffix in ("count", "frac", "geo_deg", "zero_geo_deg",
                                   "improvement_deg", "improvement_pct")}
    check("cross keys are exactly the expected set", new_keys == expected_new,
          f"{len(new_keys)} new keys")

    print("2. cross means match a direct numpy computation")
    zero_np = maps["zero_geo_deg"].numpy()
    geo_np = maps["geo_deg"].numpy()
    for region_name in band_regions:
        rmask = regions[region_name].numpy()[None, :].repeat(2, axis=0)
        for lo, hi in bands:
            sel = rmask & (zero_np >= lo) & (zero_np < hi)
            key = f"{region_name}_{band_key(lo, hi)}"
            if not sel.any():
                check(f"{key} absent when empty", f"{key}_geo_deg" not in with_cross)
                continue
            check(f"{key}_geo_deg", close(with_cross[f"{key}_geo_deg"], geo_np[sel].mean(), 1e-6))
            check(f"{key}_count", close(with_cross[f"{key}_count"], float(sel.sum())))

    print("3. the bands partition each region")
    for region_name in band_regions:
        total = float(regions[region_name].numpy().sum()) * 2.0
        got = sum(with_cross.get(f"{region_name}_{band_key(lo, hi)}_count", 0.0) for lo, hi in bands)
        check(f"{region_name} bands sum to the region", close(got, total),
              f"{got:.0f} vs {total:.0f}")

    print("4. streaming path agrees with the one-shot path")
    totals, counts, active_counts = {}, {}, {}
    for i in range(2):  # feed one "pair" at a time, as a real run does
        chunk = {k: v[i:i + 1] for k, v in maps.items()}
        accumulate_maps(chunk, regions, thresholds, totals, counts, active_counts,
                        motion_bands=bands, band_regions=band_regions)
    streamed = finalize_metrics(totals, counts, active_counts, [])
    shared = [k for k in with_cross if k in streamed and k.endswith(("_geo_deg", "_frac"))]
    worst = max(((abs(streamed[k] - with_cross[k]), k) for k in shared), default=(0.0, "-"))
    check("streaming == one-shot on every shared key",
          all(close(streamed[k], with_cross[k], 1e-6) for k in shared),
          f"{len(shared)} keys, worst delta {worst[0]:.2e} at {worst[1]}")

    print("5. improvement metrics are derived for cross keys")
    key = f"poles_{band_key(2.0, 4.0)}"
    if f"{key}_geo_deg" in with_cross:
        expect = 100.0 * (with_cross[f"{key}_zero_geo_deg"] - with_cross[f"{key}_geo_deg"]) \
            / with_cross[f"{key}_zero_geo_deg"]
        check(f"{key}_improvement_pct", close(with_cross[f"{key}_improvement_pct"], expect, 1e-6))

    print("6. a typo'd region name stops the run")
    try:
        summarize_maps(maps, regions, thresholds, bands, band_regions=("poless",))
        check("unknown region raises", False)
    except KeyError as exc:
        check("unknown region raises", "poless" in str(exc))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
