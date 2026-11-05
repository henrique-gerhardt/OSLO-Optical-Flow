"""Known-answer checks for the viewport EPE (spherical_flow.viewport_metrics).

V1  zero prediction: viewport EPE equals the zero-flow column exactly.
V2  GT as prediction (ERP route): viewport EPE is exactly 0.
V3  yaw 2 deg: zero-flow EPE at the center of an equator cube face is fx*tan(2 deg),
    and the sampled GT endpoints match the analytic rotation.
V4  node route (native, one mapping) on rotation fields: error well below the
    motion in every viewport group, poles included.
"""

from __future__ import annotations

import math

import torch

from run_epe_smoke import H, RES, W, analytic_erp_flow, rotation_matrix
from spherical_flow.erp_readout import erp_pixel_directions, node_weights_at_dirs, nodes_to_endpoints
from spherical_flow.geometry import healpix_unit_vectors, logmap, tangent_basis
from spherical_flow.viewport_metrics import (
    accumulate_viewport,
    build_viewports,
    erp_flow_endpoints,
    finalize_viewport,
    viewport_epe,
)


def score(pred_end, gt_end, vp):
    totals, counts = {}, {}
    accumulate_viewport(viewport_epe(pred_end, gt_end, vp), vp, totals, counts)
    return finalize_viewport(totals, counts, vp["layout"])


def main() -> None:
    dirs = erp_pixel_directions(H, W)
    points = healpix_unit_vectors(RES)
    be, bn = tangent_basis(points)
    failures = []

    for layout in ("cube", "paper"):
        vp = build_viewports(layout, H, W)
        pre = f"viewport_{layout}"
        rot = rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), 2.0)
        gt = analytic_erp_flow(rot, dirs)
        gt_end = erp_flow_endpoints(gt, vp)

        m = score(vp["dirs"], gt_end, vp)  # V1: endpoints = sources
        v1 = abs(m[f"{pre}_all_epe_px"] - m[f"{pre}_all_zero_px"])
        m = score(erp_flow_endpoints(gt, vp), gt_end, vp)  # V2
        v2 = m[f"{pre}_all_epe_px"]
        analytic = vp["dirs"] @ rot.double().T
        v3_end = (gt_end - analytic).norm(dim=-1).max().item()
        print(f"[{layout}] V1 |epe-zero| {v1:.2e}  V2 epe {v2:.2e}  "
              f"V3 sampled-vs-analytic endpoint max {math.degrees(v3_end):.4f} deg")
        if v1 > 1e-9:
            failures.append(f"V1 {layout}")
        if v2 > 1e-9:
            failures.append(f"V2 {layout}")
        if math.degrees(v3_end) > 0.01:
            failures.append(f"V3 endpoints {layout}")

        if layout == "cube":
            vh, vw = vp["size"]
            center = (vh // 2) * vw + vw // 2  # face 0 (yaw 0, pitch 0)
            maps = viewport_epe(vp["dirs"], gt_end, vp)
            got = maps["zero"][center].item()
            expect = vp["fx"] * math.tan(math.radians(2.0))
            print(f"[cube] V3 face-center yaw flow {got:.4f} px, expected {expect:.4f} px")
            if abs(got - expect) > 0.02 * expect:
                failures.append("V3 face center")

        weights = node_weights_at_dirs(RES, vp["dirs"].numpy())
        for axis, angle in (([0.0, 0.0, 1.0], 2.0), ([1.0, 2.0, 0.5], 0.5), ([-0.3, 1.0, 0.8], 0.1)):
            rot = rotation_matrix(torch.tensor(axis), angle)
            gt_end = vp["dirs"] @ rot.double().T
            node_flow = logmap(points, points @ rot.T, be, bn)[0]
            pred_end = nodes_to_endpoints(node_flow, be, bn, weights, vp["dirs"].float())
            m = score(pred_end, gt_end, vp)
            rows = {g: (m[f"{pre}_{g}_epe_px"], m[f"{pre}_{g}_zero_px"])
                    for g in ("all", "equator", "poles")}
            print(f"[{layout}] V4 rot {angle} deg: " + "  ".join(
                f"{g} {e:.4f}/{z:.3f} px" for g, (e, z) in rows.items()))
            if any(e > 0.05 * z for e, z in rows.values()):
                failures.append(f"V4 {layout} {angle}")

    if failures:
        raise SystemExit(f"viewport smoke FAILED: {failures}")
    print("viewport smoke PASSED (V1 zero, V2 identity, V3 geometry, V4 node route).")


if __name__ == "__main__":
    main()
