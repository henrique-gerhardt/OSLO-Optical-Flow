# Minimal experiments for the five proposals

Source: `reports/Formulações de fluxo para HEALPix.md` (deep research, 2026-11-04). One isolated
script per proposal; nothing here changes the training code. Shared data and scoring:
`common.py` (FlowScape raw locally, sfprep shards on the box; PriOr-Flow protocol + cube
viewport EPE + near-nadir overshoot split). Box runner: `scripts/proposals_probes.sh`
(queue job `proposals_probes`).

| # | Script | Kind | Question | Pre-registered reading |
|---|---|---|---|---|
| 1 | `p1_rigid_ceiling.py` | oracle, no training | How much of the GT flow is camera rotation + translation + per-node inverse depth? | go if rigid_r6 puts >= 85-90 % of pixels below 1 px |
| 2 | `p2_cascade_probe.py` | classical, no network | Does r4 -> r5 -> r6 estimation beat a single grid with the same matcher? | cascade beats both single grids, esp. 0-5 deg and poles |
| 3 | `p3_gate_probe.py` | eval of a checkpoint | Ceiling of a static gate (zero where GT is static) and a training-free proxy | kill if oracle gate < 5 % EPE; go if >= 10 % with room above the proxy |
| 4 | `p4_faces_probe.py` | frozen pretrained RAFT | Does a planar network run on HEALPix faces as well as on ERP? | faces within ~10 % of ERP through the node route |
| 5 | `p5_waft_toy.py` | small isolated training | Does a cost-volume-free (warp + NESTED window attention) update learn on HEALPix? | waft >= corr within noise, esp. 0-5 deg and poles |

## Local results (Mac, raw FlowScape test, small samples; box runs pending)

Reference: matched-recipe flowscape:test EPE, OSLO 8.63 px vs RAFT-small 6.32 px.

**P1, 40 pairs. Go criterion NOT met.** rigid_r6 puts 61.8 % of pixels below 1 px
(equator 85 %, poles 39 %); EPE 14.99 px, worse than OSLO. In degrees the residual is small
in the median (0.03-0.3 deg per pair) but heavy-tailed at 60-85 deg latitude (p90 3-7 deg).
Some pairs reach 0.2-0.3 deg median: fit failures, moving objects or a GT that is not exactly
rigid (camera rig) are not separated yet.

**Found on the way: near-nadir overshoot.** About 3 % of FlowScape pixels (46 % of the
bottom 32 rows) have a GT endpoint below the raster (v + dv > H). They hold 23.6 % of the
zero-flow error mass (mean motion 175 px). Neither a pole-crossing nor a clamped reading of
them fits a rigid motion (median residual 5.2 / 1.75 deg vs 0.18 deg elsewhere), so they are
not geometric. `sample_pair_to_nodes` marks them invalid, so node models are never
supervised there, while the official protocol scores them. `run_epe_eval.py` now reports
them as `official_overshoot_*` / `official_inraster_*`. Hypothesis to test on the box: part
of OSLO's polar ERP-px gap is this region. A sphere-native writing of the exact GT costs only
0.39 px (the official protocol clamps the same way), so this is not a metric penalty on
sphere methods in general.

**P2, 12 pairs: cascade_r4_r6 best in-raster (15.6 vs single grids 17.5-17.9, zero 18.0) and 5-15 deg (42.8 vs 55), worse at 0-5 deg (14.2 vs 12.4 single_r6); the classical matcher barely beats zero overall. Earlier 2-pair notes:** Self-test (frame 2 = frame 1 rotated 2 deg):
single_r6 recovers half the error (12.6 -> 7.3 px), single_r4 is worse than zero; coarse
levels inject error the fine level does not fully undo. Real pairs: cascade_r4_r6 best
global (17.4 vs zero 18.8, single_r6 18.0) and much better at 5-15 deg (55 -> 40 px), worse
at 0-5 deg (11.8 vs 9.1). With a classical matcher the cascade buys reach and costs small-
motion precision.

**P3, smoke only** (Stage A is out of domain on FlowScape): oracle gate -12.2 %, magnitude
gate 0.5 deg -4.9 %. The reading needs the box checkpoints.

**P4, 6 pairs, frozen RAFT-small C_T_V2, r8 faces, pad 64.** Self-test on a 2 deg
rotation: geometry correct, faces better at the poles (6.7 vs 12.8 px). Real: faces 14.12 vs
ERP-through-nodes 10.56 px overall, but 3.1 of the 3.6 px gap is the overshoot region
(266 vs 136 px there). In-raster faces 8.04 vs 7.43 px (+8 %), equator 0.76 vs 0.94 px
(-20 %), cube viewport 1.23 vs 1.47 px (-16 %), with no training. Promising for E2
(FlyingThings weights in a face-based model).
20-pair run: in-raster faces 10.48 vs 10.30 px (+1.7 %, a tie), equator 1.27 vs 1.57 (-19 %), cube viewport 2.09 vs 2.59 (-19 %); the overall gap (18.2 vs 14.4) is the overshoot region (261 vs 145 px).

**P5, 30-step smoke only**: both arms train; the reading needs the box run (3000 steps).
