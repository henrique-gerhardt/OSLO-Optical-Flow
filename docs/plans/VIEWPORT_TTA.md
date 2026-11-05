# Viewport EPE (plan A) and per-pair spherical refinement (probe B)

Opened 2026-11-03. Motivation: Cheng et al., arXiv 2610.32561 (HEALPix overfitted codec,
beats OSLO-IC). Their ablation (Table III): ERP with 8 levels costs only +4.2 % rate over
HEALPix with 8 levels, while a single level costs +44.5 % (HEALPix) / +57.6 % (ERP). The
hierarchy is the lever, equal-area sampling is an efficiency property. They evaluate in
six rectilinear viewports because ERP-domain metrics carry the planar prior: rows with
|lat| > 60 deg hold one third of the ERP samples and cover 13.4 % of the sphere.

Our ERP-px EPE has the same weighting. 95 % of OSLO's EPE is polar (COMPARISON_PARTNERS
§9.1), the r6 output-grid floor alone is 7.60 px at the poles, and at matched budget OSLO
wins SEPE everywhere but loses polar px (RAFT_PARITY §11.8).

## A. Viewport EPE

`spherical_flow/viewport_metrics.py`, wired into `run_epe_eval.py` (`viewport_*` keys;
the `official_*` columns are unchanged).

- Each method is read along the same viewport rays from its own format: ERP methods by
  nearest-pixel flow, node methods directly from the nodes (`nodes_to_endpoints` with
  `node_weights_at_dirs`, the ERP readout stencil generalized to arbitrary directions).
  One mapping per method, as in the paper.
- Error = distance between the gnomonic projections of predicted and GT endpoints, in
  viewport px. Rays whose GT endpoint is more than 84.3 deg off-axis are excluded for every
  row; a prediction past that limit is clamped and counted.
- Layouts: `cube` (six 90 deg faces, W/4 px, partitions the sphere) and `paper` (4 equator
  150x60 deg + 2 poles, W/3 x W/6 px, the paper's protocol).
- Known-answer checks: `run_viewport_smoke.py` (zero = zero column exactly, GT-as-pred = 0,
  face-center yaw flow = fx tan(2 deg) to 0.01 %, node route on three rotations < 0.05 % of
  motion in every group, poles included). `run_epe_smoke.py` still passes after the
  stencil refactor.

Local, replica360:val, 8 pairs, Stage A: ERP-px EPE 18.47, viewport cube 5.05 / paper 5.31
px; oracle r6 2.27 ERP px vs 2.32 / 1.43 viewport px.

Pre-registered reading (matched pairs, two-seed means, `benchmark_viewport`):
- OSLO-small ≤ RAFT-small-320 in viewport cube all → the ERP-px gap is the metric's polar
  weighting; report viewport EPE next to ERP-px EPE as the sphere-uniform column.
- RAFT still wins viewport cube all by > seed noise → the gap is real, not metric.
- Pretrained pair (oslo_pretrain_ft vs raft_small_c360_ft) read the same way, with the
  caveat that OSLO there trained on erp_px loss.

## B. Per-pair spherical refinement (`run_tta_refine_probe.py`)

Training-free. Starts from the OSLO field, moves it to r7 and minimizes, per pair:
census distance (frame 1 at node x vs frame 2 at e(x), read from the full-resolution ERP)
plus edge-aware first-order smoothness plus an anchor to the network field, with a
forward-backward occlusion mask from a second network pass. Coarse-to-fine by blurring
both frames (sigma 2, 1, 0 px, 60 Adam steps each).

The residual is parametrized as a sum over quad-tree levels (nearest parent through NESTED
indexing), the paper's hierarchy. Per-node Adam random-walks in textureless regions:
self-test (frame 2 = frame 1 rotated 1 deg, init zero, r7) 0.785 -> 0.59 deg per-node vs
0.785 -> 0.104 deg with the quad-tree residual.

Local, replica360:val, 4 pairs, Stage A (large motion, not the target regime):
- coarsest r3, no anchor, no occlusion: helps 0-5 deg (25.1 -> 19.5 px), hurts overall
  (equator 4.0 -> 6.1 px): coarse levels drag regions across motion boundaries;
- coarsest r5, anchor 0.05, fb occlusion, smooth 0.2: 21.07 -> 20.10 px, 4/4 pairs better,
  SEPE and viewport also better.

Box run (`scripts/viewport_tta_probe.sh`, flowscape:test, 100 pairs at stride 13,
oslo_pretrain_ft s7, variants `anchor05_fb_c5` and `anchor02_fb_c4` x smooth 0.05/0.2/0.5).
Pre-registered reading:
- worth a full benchmark row if the best config improves paired EPE by >= 0.5 px with
  >= 70 % of pairs improved and does not worsen SEPE or viewport EPE;
- the 0-5 deg bin (77 % of pixels, OSLO 10.7 vs RAFT 4.1 px) is the main signal;
- a gain only in ERP-px at the poles with worse SEPE is the metric again, not a fix.

The probe is head-agnostic; if it passes, run it on the RAFT rows too before claiming any
spherical advantage.

## Seam defect found on the way (training targets)

replica360 stores GT du unwrapped (up to +-(W-1) px next to the seam). `sample_pair_to_nodes`
interpolates du bilinearly, so nodes whose stencil mixes wrapped and unwrapped pixels get an
endpoint on the far side of the sphere. replica360:val, 50 pairs, r6: 3.9 % of pixels have
|du| > W/2, 0.16 % of valid nodes get a target more than 90 deg wrong, mean target change
0.25 deg. The oracle row shows it too (replica seam EPE 11.2 vs 2.0 px global; viewport face
at yaw 180 deg 10.8 vs 0.4 px). mpf:val is clean. The ERP-px and viewport metrics score
against the dense GT and are unaffected; training targets and the oracle row are affected.

`run_seam_target_check.py` measures it per source on the box (first step of the probe job).
Default unchanged. Fix if flowscape or chairs360 show > 0.05 % of nodes off by > 90 deg:
wrap du to [-W/2, W/2) before sampling, behind a flag, A/B on replica360.
