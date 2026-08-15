# P2A: HEALPix→ERP readout and ERP-pixel EPE

<!-- SCOPE-BANNER -->
> **STATUS 2026-08-14 — read `docs/plans/LITERATURE_SCOPE.md` before quoting anything here.**
> Two independent corrections apply across this project's documents. (1) The FLOW360
> forward-flow convention was inverted, so every FLOW360 result recorded before
> 2026-08-04 is void. (2) A literature check on 2026-08-14 found that several things
> treated here as ours already exist in print: the geodesic (SEPE) metric, polar/equatorial
> stratification, rotation-robustness evaluation in panoramic vision, and matched-backbone
> comparison across panoramic representations. `LITERATURE_SCOPE.md` is the register of what
> we may and may not claim.
>
> **This file specifically:** The FLOW360 EPE floors and the flow360 rows are VOID; the readout instrument itself was validated independently and holds.


**Status: implemented, all local gates passed (2026-07-11).** Code:
`spherical_flow/erp_readout.py` + `run_epe_eval.py` + `run_epe_smoke.py`. Measured
(container, r6, 512×1024): G2 seam 0.0010 px; G5 poles/equator 1.49× at 1e-4 px;
G1 exact; G4 node-route cost ≈ 0.01 px at the median (RAFT 2-pair sample); G3 floor
samples — replica p50 0.47 px / global ≈ 2 px, flow360 p50 0.0001 px / equator
0.016 px / global 0.60 px unweighted (0.077 cos-lat) — pole-row 1/cos(lat) px
inflation + discontinuities dominate the unweighted mean. One design addition over
§1.1: polar-cap stencils (rings 1–9, where the 4i-node rings span ≥5° of longitude
and bilinear attenuates the cos(lon) harmonic by cos(gap/2) — measured 0.707× at
ring 1) use min-norm affine-reproducing weights (Σw=1, Σw·pᵢ=p_pixel), exact for
rotation fields.

**FULL-VAL BOX RUNS DONE + VERDICT (2026-07-11).** Floors: replica global 2.02 px
(p50 0.47, 2.2% of zero); flow360 global 0.50 px unweighted / 0.076 cos-lat
(equator 0.038 = 6% of zero-equator; p50 0.00013). G4 at scale: raft vs raft_nodes
= +0.004 px median, −0.75 px global (round-trip transparent-to-helpful).
**Acceptance verdict (§4): interpolation readout suffices for the P2C campaign**
— the floor is 14% of OSLO's current replica error and the node route costs RAFT
nothing measurable; the learned ERP head stays deferred. One carve-out: on flow360
*unweighted global* EPE the r6 floor is 35% of the zero baseline (pole-row 1/cos
px inflation + discontinuities), so leaderboard-grade flow360 global numbers will
eventually need r7 supervision or the learned head — equator/active numbers do not.
Headline findings logged in chapter §5.1.2 (EPE flips the replica global winner to
OSLO: 14.25 vs 31.28 px, cos-lat 7.33 vs 9.12) and §7.4 (frozen RAFT loses to
zero-flow on flow360:val global EPE, −20.5%; RAFT/B′ equator noise floors agree to
0.4%: 0.8172 vs 0.8204 px — the wall is universal).

**Goal.** Report OSLO-RAFT in the unit every published 360° method reports — mean
endpoint error in ERP pixels — without building the learned upsampling head yet. A
geometric interpolation readout is enough to (a) put an EPE column in the thesis §5.1
head-to-head, (b) measure the **grid floor**: how much EPE is unreachable by *any*
model supervised at r6, which is the number that decides whether P2C needs r7
supervision or a learned ERP head.

**Non-goal.** Competing on a leaderboard. The learned convex-upsampling-to-ERP head is
deliberately deferred until the round-trip floor (G3) proves interpolation is the
bottleneck.

---

## 1. Design

### 1.1 The readout is the exact inverse of `erp_flow_to_tangent`

`raft_adapter.erp_flow_to_tangent` already goes ERP→nodes: sample flow at node pixel
coords → endpoint pixel → endpoint direction → `logmap` at the node. The readout goes
nodes→ERP:

```
node tangent flow (N,2) at grid points
  → tangent_components_to_3d(points, flow, basis_east, basis_north)   # (N,3) ambient
  → HEALPix bilinear interpolation of the 3D field at each pixel direction  # (H·W,3)
  → project onto each pixel's tangent plane (remove radial component)
  → expmap(pixel_dirs, projected)                                     # endpoint dirs
  → points_to_equirectangular_pixels(endpoints, H, W)                 # (u2, v2)
  → du = wrap_lon(u2 − u),  dv = v2 − v                               # ERP flow, px
```

Interpolating the **3D ambient vectors** (not per-node tangent components) sidesteps
basis alignment: tangent bases at neighboring nodes differ, so averaging components is
wrong near the poles exactly where we claim to win; ambient vectors average cleanly and
the radial residual after averaging is O(spacing²) — removed by the projection step.

Interpolation weights: `astropy_healpix.HEALPix(nside, order="ring"/"nested")
.bilinear_interpolation_weights(lon, lat)` → 4 neighbor indices + weights per pixel.
Match the ordering convention used by `geometry.healpix_unit_vectors` (check its
`nest` argument — the weights must index the same node layout the model outputs).
Precompute once per (H, W, resolution) and cache next to the pyramid cache.

### 1.2 EPE definition (comparability first)

- `epe = sqrt((du_pred − du_gt)² + (dv_pred − dv_gt)²)` per valid pixel, plain
  unweighted mean over pixels — this is what FLOW360/PanoFlow/MPF papers report.
  Secondary column: `cos(lat)`-weighted mean (solid-angle-fair; ours to argue with,
  never substituted for the standard one).
- **Seam wrap:** compare *endpoint pixels*, not raw du: `du_err = wrap_to_half_width(
  (u2_pred − u2_gt))` where wrap maps to (−W/2, W/2]. GT `du` in the shards is already
  canonical `[du_x, dv_y]`; still wrap the *error*, since pred and GT may land on
  opposite sides of the seam.
- Region masks in pixel space: same definitions as `build_region_masks` (poles ≥60°,
  equator ≤30°, seam ±15° of ±180°) computed from pixel lat/lon — reuse the fp64+eps
  pattern verbatim (pixels don't sit on boundaries the way HEALPix nodes do, but the
  convention should be one convention).
- Validity: the shard `valid` mask, plus finite-GT.

### 1.3 Files

| file | change |
| --- | --- |
| `spherical_flow/erp_readout.py` | **new**: `bilinear_node_weights(points_meta, H, W, device)` (cached), `nodes_to_erp_flow(node_flow, points, basis_east, basis_north, H, W, weights)` → `(H,W,2)` px flow; `erp_epe_maps(pred_px, gt_px, valid)` → per-pixel EPE + region/valid reductions (sums/counts, accumulate-then-finalize like `spherical_flow.metrics`) |
| `run_epe_eval.py` | **new runner**, mirroring `run_raft_shard_baseline.py`'s structure: `--predictor {oslo,raft,zero,oracle}`; `oslo` loads a checkpoint via the same construction path as `run_oslo_raft.py --eval-only`, runs node inference, reads out to ERP; `raft` uses `predict_raft_flow` natively (no node round trip); `oracle` = GT→nodes (`sample_pair_to_nodes`) → readout → EPE vs GT = **the grid floor**; `zero` = zeros. Saves `epe_metrics.json` (same args/sources/metrics envelope as the baselines) |
| `run_raft_shard_baseline.py` | untouched (geodesic metrics stay its job; EPE lives in the new runner so neither script grows two responsibilities) |

The oslo predictor must accept every geometry flag the eval path accepts (`--retina`,
resolutions, pyramid cache) — lift the model-construction block of `run_oslo_raft.py`
into a helper if copying it would drift.

---

## 2. Validation gates (all CPU, all in Docker, before any box run)

| gate | test | pass criterion |
| --- | --- | --- |
| G1 zero-consistency | `--predictor zero` on any source | EPE mean == mean GT magnitude in px (matches the raw-file stats table: flow360 p50 0.64 px) and geodesic zero columns convert consistently (°→px at 0.352°/px eq.) |
| G2 wrap | synthetic pair: uniform +3 px du crossing the seam, GT built accordingly | EPE ≈ 0 (< 1e-3 px); breaking the wrap on purpose must blow it up to ~W/2 |
| G3 **grid floor** | `--predictor oracle` at r6 on replica360:val and flow360:val | *measured, not asserted* — record mean/p50/p90 EPE. Expectation: ≪ 2.6 px (the naive 0.92° node spacing) on smooth field, dominated by motion-discontinuity pixels. This number goes in the paper |
| G4 round-trip vs native | RAFT flow evaluated natively vs RAFT flow pushed through nodes (ERP→nodes→ERP) | difference ≈ G3 floor; bounds what the node route costs any predictor |
| G5 basis sanity | pure-rotation synthetic pair (exact GT from `synth_rotation_record` machinery) | readout EPE ≈ G3 floor at the same scale, uniformly in latitude (no pole blowup — this specifically tests the ambient-interpolation choice of §1.1) |

G2/G5 are unit-test-shaped: put them in `run_epe_smoke.py` (CPU, seconds, no shards)
following the `run_oslo_raft_retina_smoke.py` convention.

## 3. Box runs (after gates)

1. Grid floor: `--predictor oracle` on replica360:val and flow360:val (30 s each).
2. EPE columns for §5.1: `--predictor raft` and `--predictor oslo
   --init-checkpoint …stageA/oslo_raft.pt` on replica360:val.
3. flow360:val EPE for both predictors — the first number comparable in *unit* to the
   published FLOW360 tables. **Comparability caveat to carry into any text:** published
   numbers are on their test protocol/split with their eval code; ours is val-split
   with our code. Same unit ≠ same benchmark until we run their protocol.

## 4. Acceptance

P2A is done when: the five gates pass; the two grid-floor numbers exist; §5.1 gains an
EPE row; and the G3/G4 outcome is written down as a design verdict for P2C — if the
oracle floor at r6 is small relative to the gap we're chasing (< ~0.3 px on flow360),
interpolation readout suffices for the campaign and the learned head stays deferred;
if not, P2C's stage P3 must add r7 supervision or the learned ERP head.
