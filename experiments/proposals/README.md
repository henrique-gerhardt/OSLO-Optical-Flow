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
| 6 | `p6_polar_band_head.py` | frozen OSLO + small trained head | Can the near-nadir overshoot pixels (whole matched-30k gap) be learned with an ERP band head on the H/16 polar rows? | success if overshoot <= 40 px and in-raster within 0.05 px; > 80 px = not learnable from these inputs |
| 7 | `p7_learned_gate.py` | frozen OSLO + small trained gate | How much of the P3 oracle static-gate ceiling does a learned per-pixel gate recover? | on scale_fs30k s7: success if EPE <= -9 % (half the oracle -18.7 %) with moving-pixel EPE up <= 2 %; kill if no better than mag_gate_0.5 (-3.3 %) |
| 8 | `p8_mag_sweep.py` | frozen predictor, no training | Is the p7 learned gate just a tuned threshold on the predicted motion? | f = share of the p7 gate_soft gain recovered by mag_train (tau picked on train); f >= 0.75 = report the threshold; f <= 0.5 = the network does work |
| 9 | `p9_band_gate_stack.py` | frozen predictor + trained heads, no training | Who is ahead once both models carry the P6 band head and the P7 gate? | on `stack`, fs30k s7/s11: OSLO ahead if its EPE < RAFT-small's on both seeds; RAFT ahead if the reverse; else tie |
| 10 | `p10_prediction_probe.py` | frozen predictor, no training | Is the model's flow a good motion field for predicting a frame on HEALPix nodes (on-sphere codec direction)? | residual MSE over r8 nodes, fs30k s7/s11: alive if OSLO <= RAFT-small on both seeds; dead if RAFT-small lower by >= 5 % on both; else tie. Same thresholds at matched rate: OSLO `pred` (native r6) vs RAFT-small `pred_r6` |
| 11 | `p11_so3_finetune_probe.py` | released PriOr-RAFT, short fine-tune | Does rotation in training make a released in-domain model orientation-robust? | rotated geodesic error after the `so3` arm: falls >= 50 % with upright up <= 25 % = works; falls < 20 % = a short fine-tune is not enough |

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

**P5, box 3000 steps, 106 pairs: uninformative.** Neither arm beats zero flow (zero 22.40, corr 22.47, waft 22.45 px); the toy did not learn, so it says nothing about corr vs waft.

**Box 2026-11-06, flowscape:test (P1 279 pairs; P2 100; P3, P4 107).**

- **P1: kill.** rigid_r6 puts 58.1 % of pixels below 1 px (need 85-90 %; equator 80 %, poles 36 %). The oracle rigid fit is worse than the learned OSLO in-raster (9.06 vs 5.1-5.5 px), and in overshoot it is worse than zero (198 vs 175 px).
- **P2: inconclusive, leaning kill.** cascade_r4_r6 22.22 px vs zero 23.15 and single grids 23.39-23.99. It wins at 5-15 deg (41.8 vs 44.8-55.2) and at the poles (39.4 vs 40.8-42.8), and loses at 0-5 deg (15.16 vs single_r6 12.81, zero 12.42). The classical matcher barely beats zero, so the probe cannot separate the cascade from the matcher.
- **P3: go on the 30k checkpoint, between thresholds on the pretrained one.** Oracle gate: -18.7 % (scale_fs30k s7: 8.11 -> 6.59 px; in-raster 5.16 -> 3.80; SEPE 12.70 -> 9.36; static 8.38 -> 3.15) and -8.0 % (oslo_pretrain_ft s7). The training-free magnitude proxy reaches only -3.3 % / -2.4 % at 0.5 deg, so the room above the proxy is about 15 points on the 30k checkpoint. AUROC of -|pred| for static is 0.87-0.90. Static fraction 0.29.
- **P4: criterion not met overall, met in-raster for RAFT-small only.** Overall faces vs ERP through nodes: +21 % for small (17.91 vs 14.84), +41 % for large (17.23 vs 12.18). The overshoot region drives this (254-264 vs 123-150 px). In-raster: small -3.5 % (10.28 vs 10.65), large +13 % (9.88 vs 8.73). RAFT-small gains at the equator (-26 %, 1.20 vs 1.62) and on the cube viewport (-26 %, 2.02 vs 2.74). RAFT-large ties on both (1.00 vs 0.97, 1.78 vs 1.73). The ERP-to-nodes route costs 0.07-0.14 px overall.

**P6, local smoke only (Stage A, 3 pairs)**: pipeline runs; gt_clamped floor 0.28 px overall, ~10 px in overshoot. Overshoot |du| median ~130 px: the loss is longitude near the pole, not the vertical clamp (clamp alone costs 13.5 px there). Box job `overshoot_band_head`.

**P7, local smoke only (Stage A, 3 pairs, 40 steps)**: pipeline runs; the numbers carry no signal. Read the `moving` column. A soft gate can also shrink moving flow, and that gain is magnitude calibration, not static detection. Box job `static_gate_learned`.

**P7, box (full flowscape:test). Pre-registered success on scale_fs30k s7, replicated on s11.**

| checkpoint | base | gate_soft | gate_hard | oracle_gate | mag_gate_0.5 | moving (soft) | static base -> soft (oracle) |
|---|---|---|---|---|---|---|---|
| scale_fs30k s7 | 8.55 | 6.60 (-22.8 %) | 6.67 (-22.0 %) | 7.19 (-15.9 %) | 8.29 (-3.1 %) | -10.5 % | 8.71 -> 3.78 (3.50) |
| scale_fs30k s11 | 8.88 | 6.88 (-22.5 %) | 7.16 (-19.4 %) | 7.33 (-17.5 %) | 8.59 (-3.2 %) | -9.8 % | 9.44 -> 4.18 (3.50) |
| oslo_pretrain_ft s7 | 8.52 | 7.03 (-17.5 %) | 7.52 (-11.8 %) | 7.95 (-6.7 %) | 8.36 (-1.9 %) | -13.0 % | 5.68 -> 3.48 (3.50) |

- Two effects add up. The gate recovers almost all of the oracle's static gain. It also zeroes or shrinks moving pixels where OSLO's prediction is worse than zero (gate_hard alone: moving -6 to -10 %). The second effect is abstention or confidence, not static detection.
- fs30k s7/s11 gated: in-raster 3.73 / 3.62 px, SEPE 9.44 / 9.10, overshoot still 94 / 106 px.
- The proxy comparison is weak. The learned gate's AUROC (0.876) barely beats -|pred| (0.865-0.872), and the proxy was only swept up to 0.5 deg. A tuned magnitude threshold could recover much of the static part without a network.
- Not yet a claim about OSLO vs RAFT: RAFT-small gets the same head in `static_gate_raft_control`.
- P6 first run: both checkpoints stop at 1200 cached pairs (the OOM). The rerun `overshoot_band_head_r2` has no log yet.

**P8, local smoke only (Stage A, 3 pairs)**: pipeline runs, tau* = 1 deg on 4 train pairs; the numbers carry no signal. Box job `static_gate_mag_sweep` (after the RAFT control, so the p7r JSONs exist).

**Box 2026-11-09: P6 r2, P7r, P8 (full flowscape:test; details in `docs/plans/RAFT_PARITY.md` §11.25).**

- **P6 band head (OSLO fs30k s7 / s11):**
  - EPE 8.55 -> 5.15 / 8.88 -> 4.87 px; overshoot 103 -> 23 / 123 -> 24 px (criterion <= 40 met).
  - In-raster also falls by 0.9 px (5.46 -> 4.57 / 5.14 -> 4.25), all of it in the band rows (band in-raster 27.4 -> 18.2). The equator is unchanged. The "within 0.05 px" guard was meant to catch damage; the shift is an improvement.
  - OSLO only: the same head on RAFT-small is still missing.
- **P7r gate on RAFT-small:** fs s7 8.09 -> 4.41 (-45.5 %), fs s11 6.76 -> 3.94 (-41.8 %), c360_ft s7 6.54 -> 3.66 (-44.0 %). Matched-pair gap OSLO - RAFT, mean of the fs30k seeds: 1.29 px ungated, 2.57 px gated. The gate is a generic head and helps RAFT more; RAFT's static-pixel error is larger (9.1-11.4 px).
- **P8 threshold control:**
  - f for mag_train: 0.24 / 0.21 / 0.06 on OSLO and negative on RAFT (-0.09 to -0.12). Even tau picked on test gives at most 0.40.
  - s* = 1 almost everywhere, so a global shrink does not help either.
  - Pre-registered reading f <= 0.5: the learned gate does work a |pred| threshold cannot.

**P10, local smoke only (3 pairs; not the matched pair, no reading).** Pipeline runs for the three predictors. The GT closes the zero-flow residual (MSE 178.6 -> 62.7, 8-bit units); a half-pixel offset in either axis and the negated flow are both worse at the equator, so the sampling and the flow direction are right. Stage A OSLO (out of domain) closes 84.6 % of the zero-to-GT gap; frozen pretrained RAFT-small C_T_V2 closes 106.9 %, i.e. it goes below the GT residual. The GT is not a strict ceiling: at the poles it leaves 53.7 against 27.6 for RAFT-small and 39.4 for OSLO, and the offset test is flat there. OSLO's native node read and its ERP readout give the same residual (80.54 vs 80.53). Overshoot nodes are 0.2 % of the sphere area (3.17 % of ERP pixels). Box job `prediction_probe_both_models`.

Matched-rate rows (same 3 pairs): each field is reduced to an r6 / r5 / r4 HEALPix motion field (49152 / 12288 / 3072 vectors) and read back at the grid nodes. The reduction costs almost nothing: GT 62.7 -> 64.2 / 66.1 / 67.9, RAFT-small 54.6 -> 55.0 / 55.3 / 54.8, OSLO 80.5 -> 80.5 / 80.2 / 79.7. On these pairs the motion field is smooth enough that 3072 vectors carry it, so field density is not what separates the two models, and the operator itself is not lossy (OSLO pred_r6 = pred). Still not the matched pair.

**Box 2026-11-09 `band_gate_both_models`: RAFT-small rows only (the OSLO p9 and p6r tables were outside the report tail).**

| RAFT-small fs30k | base | band | gate | stack | stack_all | overshoot base -> band | in-raster base -> band |
|---|---|---|---|---|---|---|---|
| s7 | 8.09 | 5.00 (-38.2 %) | 4.41 | **3.93** (-51.4 %) | 4.29 | 28.5 -> 18.1 | 7.42 -> 4.57 |
| s11 | 6.76 | 4.37 (-35.4 %) | 3.94 | **3.62** (-46.4 %) | 3.90 | 27.1 -> 15.9 | 6.10 -> 3.99 |

- The band head helps RAFT-small almost as much as OSLO (OSLO -39.8 / -45.2 %), mostly in-raster band rows (42.1 -> 17.4 px). Polar ERP rows are hard for both models, not only for OSLO's readout.
- Band-only gap OSLO - RAFT: s7 5.15 vs 5.00 (+0.15), s11 4.87 vs 4.37 (+0.50). It was 0.47 / 2.12 ungated, so the band head narrows the gap but RAFT stays ahead on both seeds.
- stack_all is worse than stack on RAFT: the gate on the band rows undoes part of the band head.
- Reading pending: OSLO stack < 3.93 (s7) and < 3.62 (s11) is needed for "OSLO ahead".

**P9 verdict (2026-11-10): RAFT ahead on both seeds (pre-registered reading on `stack`).**

| fs30k | OSLO base | OSLO stack | RAFT-small base | RAFT-small stack | gap base | gap stack |
|---|---|---|---|---|---|---|
| s7 | 8.55 | 4.26 | 8.09 | 3.93 | +0.47 | +0.33 |
| s11 | 8.88 | 4.22 | 6.76 | 3.62 | +2.12 | +0.60 |

- With both heads on both models, RAFT-small is ahead on almost every column:
  - in-raster 3.47 / 3.22 vs 3.65 / 3.58;
  - overshoot 18.1 / 15.9 vs 22.9 / 23.8;
  - equator, SEPE, cube viewport and both displacement bins.
- Static pixels tie (OSLO 1.95 / 2.10, RAFT 2.07 / 1.98). The remaining gap is in moving pixels (OSLO 5.08 / 4.97, RAFT 4.59 / 4.21).
- Before the heads, OSLO had the lower in-raster error in the band rows (27.4 / 27.7 vs 46.6 / 38.6 px). The band head brings both to about 17-18 px, so that edge disappears.
- The heads narrow the matched gap (mean 1.29 -> 0.47 px) but do not flip it. Both are generic post-hoc corrections, not OSLO properties. This line is closed: report them as evaluation/engineering findings, not as a method contribution.

**P10 verdict (box 2026-11-10): dead on both seeds (pre-registered reading on `pred`, MSE over r8 nodes, full flowscape:test, 1386 pairs).**

| fs30k | zero | gt | OSLO pred | RAFT-small pred | RAFT lower by | RAFT pred_r6 (matched rate) | RAFT lower by |
|---|---|---|---|---|---|---|---|
| s7 | 258.03 | 86.92 | 104.08 | **86.29** | 17.1 % | 86.03 | 17.3 % |
| s11 | 258.03 | 86.92 | 105.51 | **90.44** | 14.3 % | 90.09 | 14.6 % |

- Both readings fire "dead": full resolution and matched rate (OSLO native r6 against RAFT-small reduced to r6). The reduction operator is not lossy: OSLO pred_r6 is 103.79 / 105.30, against pred 104.08 / 105.51.
- Gap closed (zero to GT): OSLO 90.0 / 89.1 %, RAFT-small 100.4 / 97.9 %.
- Field density does not separate the models. RAFT-small down to r4 (3072 vectors) is 86.32 / 90.23, still 17 / 14 % below OSLO at its native r6.
- Poles tie: OSLO 74.4 / 75.1, RAFT-small 73.5 / 75.8. Both are below the GT residual (79.4).
- The gap is at the equator (116.3 / 118.0 vs 91.6 / 96.5), in moving nodes (125.2 / 126.4 vs 104.2 / 109.3) and in static nodes (39.7 / 41.9 vs 31.7 / 32.9). On static nodes OSLO is above the GT residual (36.5) and RAFT-small below it, so OSLO's spurious small motion costs residual where the official EPE showed a tie.
- Overshoot nodes are 0.3 % of the sphere area, so this metric does not reward the readout fix of P6/P9.
- Verdict: the on-sphere codec direction is closed. On a HEALPix-native consumer that weights equal-area samples, the matched planar model still gives the better motion field, and rate does not rescue OSLO.

**P11, local (Mac CPU, 16 test pairs, 300 batch-1 steps; plan in `docs/plans/ORIENTATION_ROBUSTNESS.md`).**

- Released PriOr-RAFT: 0.131 deg upright, 0.791 deg under a Haar rotation (6.0x).
- After the `so3` arm: rotated 0.568 deg (-28.3 %), upright 0.142 deg (+8.3 %). Between the two thresholds: right direction, criterion not met at this budget.
- Control arm without rotation, 200 steps: rotated -6.1 %, upright +2.6 % (the `so3` arm at 200: -28.2 % / +7.2 %). The gain comes from the rotation.
- Head ceilings on PriOr-RAFT (24 pairs): the oracle static gate makes it worse (2.24 -> 2.73 px), the oracle band head gains 10 % at most. The heads do not transfer to a released model.
- Box job `so3_finetune_prior`: both arms at 5000 steps, 100 test pairs.
