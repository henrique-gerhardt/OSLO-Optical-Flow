# Orientation robustness — the direction after the parity study

Opened 2026-11-11. Purpose: fix the scientific contribution the master's work continues with,
now that the architecture question is closed (`RAFT_PARITY.md` 10.1, 11.19, 11.26, 11.27).
Check `LITERATURE_SCOPE.md` before turning any line here into a novelty sentence.

## 1. Where the parity study left the project

| Question | Verdict | Source |
|---|---|---|
| Does the sphere-native model beat a planar RAFT at matched data, steps and pairs per step? | No. RAFT-small is ahead on two seeds; RAFT-large SO(3) is ahead even at the replica360 poles | 9.3, 10.1, 11.19 |
| Is the gap budget, capacity, retina, search reach or estimation grid? | No to each | 11.5, 11.6, 11.8, 11.13, 11.19 |
| Do the post-hoc heads (polar band, static gate) restore OSLO's standing? | No. They are generic and help RAFT as much or more | 11.25, 11.26 |
| Is OSLO's flow a better motion field for a HEALPix-native consumer? | No. RAFT-small residual is 14-17 % lower, also at matched rate | 11.27 |

The architecture is not the lever. The same controls showed what is: the distribution of camera
orientations seen in training.

## 2. Directions assessed

| Direction | Evidence in hand | Literature | Verdict |
|---|---|---|---|
| **Orientation robustness of 360 flow** (benchmark + remedy) | Sections 3 and 4 | No 360-flow paper scores rotated test inputs (section 5) | **Chosen** |
| Model-agnostic heads (static gate, polar band) on released models | -42 to -51 % on our weak RAFT-small rows | Flow confidence/abstention exists in perspective flow | Section 6 |
| Controlled negative study alone | Complete | Matched-backbone tables exist (PriOr-Flow Table 5) | Kept as a chapter, not as the contribution |
| Hierarchy as the flow representation | Never run; P2 inconclusive | Codec ablation 2610.32561 | Dropped: gated on P10, which came back dead |
| On-sphere codec motion | P10 dead | - | Closed |
| Faces + pretrained planar weights | P4 mixed | Tangent images exist | Control only |
| Native HEALPix data (climate winds) | None | WindFlow uses planar RAFT | Use case, not a method; sub-node motion risk |

## 3. The claim and the evidence already measured

Claim: **in-domain 360 flow accuracy is contingent on the canonical camera orientation of the
benchmark, the cause is the training distribution and not the representation, and rotation in
training removes it.**

Released models, flowscape:test, 1386 pairs, sphere-weighted geodesic error in degrees, Haar
rotation of the camera (`UNIVERSALITY_TABLE.md` 16.45). Zero flow is 3.40 deg in both columns.

| Model | Trained on the benchmark | Upright | Rotated | Change |
|---|---|---|---|---|
| PriOr-RAFT (ICCV 2025) | yes | 0.2415 | 1.2335 | +410.7 % |
| PanoFlow | yes | 0.2512 | 1.0388 | +313.6 % |
| OSLO-RAFT (SO(3) in training) | yes | 1.1577 | 1.4977 | +29.4 % |
| frozen RAFT-large (perspective weights) | no | 0.8722 | 0.9268 | +6.3 % |

The ranking inverts: the 2020 perspective model is first under rotation.

Our matched trainings, improvement over zero flow in %, 10k steps (`RAFT_PARITY.md` 9.3, 10.1):

| Model | Rotation in training | flowscape upright -> rotated | replica360 upright -> rotated |
|---|---|---|---|
| RAFT-small | no | 86.5 -> 53.4 | 83.9 -> 80.0 |
| RAFT-small | SO(3) | 81.7 -> 79.7 (s11: 80.5 -> 78.2) | 87.1 -> 87.9 (s11: 87.8 -> 89.2) |
| OSLO | no | 83.5 -> 64.7 | 81.3 -> 80.8 |
| OSLO | SO(3) | 76.2 -> 74.7 | 86.3 -> 86.5 |
| RAFT-large | SO(3) | 83.5 -> 81.9 | 90.7 -> 90.9 |

- Without rotation in training both architectures lose 18-33 points on FlowScape under rotation. With it, both lose 1.5-2.
- The cost is on the upright benchmark: RAFT-small loses 4.8 points (86.5 -> 81.7), and 8.7 px of official EPE (4.74 -> 13.46 px, 11.8).
- So no model in hand is both accurate upright and robust. Released models: 0.24 deg upright, 1.0-1.2 deg rotated. Our best robust model (RAFT-large SO(3), scratch, 10k): about 0.56 / 0.62 deg.

## 4. Minimal validation of the remedy on a released model (P11)

`experiments/proposals/p11_so3_finetune_probe.py`. Released PriOr-RAFT FlowScape weights,
fine-tuned on the Mac CPU with the ERP re-render rotation of `run_erp_raft_train.py`
(Haar, probability 0.5), batch 1, lr 4e-5, 8 training iterations, primitive-view loss only.
16 held-out test pairs, each scored upright and under one fixed Haar rotation.

Pre-registered reading on the rotated error after the `so3` arm: falls by >= 50 % with upright
up by <= 25 % = the remedy works on a released model; falls by < 20 % = a short fine-tune is
not enough.

Result, `so3` arm (2026-11-11, Mac CPU, about 11 s per step). Zero flow on these 16 pairs: 1.98 deg upright, 2.04 deg rotated.

| Step | Upright (deg) | Rotated (deg) | Rotated / upright |
|---|---|---|---|
| 0 | 0.1311 | 0.7913 | 6.0 |
| 100 | 0.1483 | 0.6026 | 4.1 |
| 200 | 0.1405 | 0.5682 | 4.0 |
| 300 | 0.1420 | 0.5676 | 4.0 |

- The step-0 row reproduces the box finding on this subset (6.0x here, 5.1x on the full test set).
- Rotated error falls 28.3 %, upright rises 8.3 %. This is between the two pre-registered thresholds: the direction is right, and 300 batch-1 steps (about 150 rotated pairs) are not enough to meet the 50 % criterion.
- Most of the gain comes in the first 100 steps; steps 200-300 are flat at this learning rate and batch.
- Departures from the released recipe: batch 1 instead of 6, constant lr, 8 iterations, no orthogonal-view loss.
- That rotation in training reaches rotated = upright at full budget is shown by our from-scratch rows in section 3, not by this probe.

Control arm `ctrl` (same script, no rotation, 200 steps):

| Step | Upright (deg) | Rotated (deg) |
|---|---|---|
| 0 | 0.1311 | 0.7913 |
| 100 | 0.1315 | 0.7494 |
| 200 | 0.1345 | 0.7431 |

At step 200 the rotated error is down 6.1 % without rotation and 28.2 % with it; upright is up 2.6 % and 7.2 %. The gain comes from the rotation, not from the extra steps. The two arms do not draw the same training pairs.

Box job `so3_finetune_prior` (`PROBES=p11`): the same two arms at 5000 steps, 100 test pairs. Reading on the `so3` arm at 5000: rotated error <= 1.5x the step-0 upright error with upright up by <= 10 % = the remedy works; rotated falls by < 50 % = fine-tuning alone is not enough.

## 4b. Dose-response and the lossless control (2026-11-12, Mac CPU, 8 flowscape:test pairs)

Same harness as P11 (sphere-weighted geodesic degrees). Zero flow is 2.12-2.17 deg in every view.

| View | PriOr-RAFT | Change | frozen RAFT-large | Change |
|---|---|---|---|---|
| upright | 0.130 | - | 0.561 | - |
| exact roll of W/4 (yaw 90 deg, no resampling) | 0.225 | +73 % | 0.590 | +3 % |
| exact roll of W/2 (yaw 180 deg, no resampling) | 0.652 | +401 % | 0.601 | +5 % |
| yaw, random angle, resampled | 0.250 | +92 % | 0.627 | +12 % |
| tilt 15 deg | 0.613 | +370 % | 0.639 | +14 % |
| tilt 30 deg | 0.830 | +537 % | 0.664 | +18 % |
| tilt 45 deg | 0.931 | +615 % | 0.663 | +18 % |
| tilt 90 deg | 0.970 | +644 % | 0.657 | +17 % |
| Haar | 0.934 | +617 % | 0.670 | +19 % |

The roll rows of frozen RAFT-large come from a separate run on the raw frames (its upright there is 0.571); the other rows use the re-render path.

- **Resampling is not the cause.** A circular shift by half the width is lossless and keeps gravity. It multiplies PriOr-RAFT's error by 5 and puts it behind frozen RAFT-large (0.652 vs 0.601). The identity re-render reproduces the raw number (0.1303 vs 0.1300).
- **The pre-registered yaw control failed for PriOr-RAFT** (I expected <= 10 %). The dependence is on the camera heading as well as on the tilt. In FlowScape the vehicle always drives towards the image centre.
- **15 deg of tilt is enough**: PriOr-RAFT goes from 4.3x better than frozen RAFT-large to level with it. From 30 deg on it is worse. The curve saturates by 45 deg.
- A half-pixel resampled yaw costs PriOr-RAFT about 0.09 deg (0.130 -> 0.221). This bounds the resampling share of the rotated numbers at about a tenth of the Haar loss.
- Mechanism not established: a learned prior on where the motion is, or the model's seam handling when the direction of travel sits on the seam. WP2.
- 8 pairs, one machine. Box jobs `orient_main`, `orient_sweep`, `orient_cross` (`scripts/orientation_suite.sh`) repeat this on the full test set, on PanoFlow and our rows, and on replica360 / mpf / flow360.

## 4c. Box results at full scale (2026-11-12/13, `orient_main`, `orient_sweep`, `orient_cross`, `so3_finetune_prior`)

Sphere-mean geodesic error in degrees, r6 nodes, haversine. Zero flow is 3.39-3.40 deg in every flowscape view.

flowscape:test, 1386 pairs (change vs upright in %):

| Row | Upright | Roll W/4 | Roll W/2 | Yaw | Haar |
|---|---|---|---|---|---|
| PriOr-RAFT | 0.2412 | 0.3259 (+35) | 0.6869 (+185) | 0.3846 (+59) | 1.2342 (+412) |
| PanoFlow | 0.2511 | 0.3055 (+22) | 0.2583 (+3) | 0.2754 (+10) | 1.0396 (+314) |
| frozen RAFT-large | 0.8724 | 0.8777 (+1) | 0.8902 (+2) | 0.8669 (-1) | 0.9277 (+6) |
| PriOr-RAFT, SO(3) fine-tune (P11) | 0.2589 | 0.3758 (+45) | 0.6410 (+148) | 0.4135 (+60) | 0.7353 (+184) |
| PriOr-RAFT, control fine-tune | 0.2393 | 0.3380 (+41) | 0.7746 (+224) | 0.4140 (+73) | 1.1698 (+389) |
| RAFT-small, no rotation (ours) | 0.4818 | 1.5674 (+225) | 1.4845 (+208) | 1.2977 (+169) | 1.7527 (+264) |
| RAFT-small, SO(3) (ours) | 0.6245 | 0.6673 (+7) | 0.6701 (+7) | 0.6513 (+4) | 0.7721 (+24) |
| RAFT-large, SO(3) (ours) | 0.5288 | 0.5720 (+8) | 0.5494 (+4) | 0.5429 (+3) | 0.6766 (+28) |

Exact tilt, 400 pairs:

| Row | Upright | 15 deg | 30 deg | 45 deg | 90 deg |
|---|---|---|---|---|---|
| PriOr-RAFT | 0.2475 | 0.8712 (+252) | 1.2127 (+390) | 1.3323 (+438) | 1.2621 (+410) |
| PanoFlow | 0.2569 | 0.7846 (+205) | 1.0534 (+310) | 1.1314 (+340) | 1.0890 (+324) |
| frozen RAFT-large | 0.8880 | 0.9131 (+3) | 0.9433 (+6) | 0.9537 (+7) | 0.9333 (+5) |
| PriOr-RAFT, SO(3) fine-tune | 0.2666 | 0.6855 (+157) | 0.8056 (+202) | 0.7871 (+195) | 0.7209 (+170) |
| RAFT-small, no rotation | 0.5030 | 1.0091 (+101) | 1.3499 (+168) | 1.4383 (+186) | 1.5962 (+217) |
| RAFT-large, SO(3) | 0.5454 | 0.5872 (+8) | 0.6314 (+16) | 0.6543 (+20) | 0.6892 (+26) |

Readings against what was pre-registered:

- **Tilt: confirmed for both released models.** Monotone to 45 deg, frozen RAFT-large within 7 %. At 15 deg the two released models have lost their whole margin (0.78-0.87 vs 0.91); from 30 deg they are behind.
- **Heading (exact roll): confirmed for PriOr-RAFT and for our no-rotation RAFT-small, refuted for PanoFlow** (+3 % at W/2). The heading dependence is not general. PanoFlow's cyclic estimation is the candidate reason. PriOr-RAFT's loss is polar (poles 0.22 -> 1.92 deg, equator +33 %).
- **Yaw control (<= 10 % for every row): failed** for PriOr-RAFT (+59 %) and RAFT-small no rotation (+169 %); passed for PanoFlow, frozen RAFT and the SO(3)-trained rows.
- **Remedy (P11, 5000 batch-1 steps, probability 0.5): partial.** Haar 1.234 -> 0.735 deg (-40 %), upright +7 %. Below the pre-registered 50 %, so "fine-tuning alone is not enough" at this budget. The control arm moves Haar by -5 %. The fine-tune does not fix the heading (roll W/2 still +148 %, poles worse).
- The fine-tuned model is still the best upright/Haar compromise in hand: mean 0.50 deg, against 0.60 (RAFT-large SO(3)), 0.74 (PriOr-RAFT) and 0.90 (frozen RAFT-large). It is ahead of frozen RAFT-large in every view.
- **Other datasets: the loss does not follow the weights.** flow360:test Haar: PriOr-RAFT +7 %, PanoFlow +2 %, frozen RAFT-large +2 %. mpf:val: every row within 1 % (and every row is only about 25 % under zero flow there, so that set says little). The replica360 block was cut from the report (job `orient_cross_table` reprints it). Pre-registered reading "scoped to FlowScape content" holds on the two sets seen.
- Out of domain and upright, frozen RAFT-large is already ahead of both released models on flow360:test (0.236 vs 0.292 and 0.275; zero 0.398).

Scope the claim accordingly: **the accuracy margin of the released FlowScape models over a generic perspective model exists only in-domain and in the benchmark's canonical orientation.** Leaving either removes it. Our in-domain replica360 rows lose 1-4 points under rotation without rotation in training (section 3), so the contingency needs a training set with a canonical orientation; a driving benchmark has one.

## 4d. Canonization ceiling (2026-11-13, Mac CPU, 8 flowscape:test pairs)

Each pair is Haar-rotated, stored as 8-bit, rotated back with the known rotation, and given to released PriOr-RAFT.

| View | PriOr-RAFT (deg) |
|---|---|
| upright | 0.1300 |
| Haar | 0.9337 |
| Haar, rotated back (oracle) | 0.1337 (+3 % over upright) |

- Two resamplings cost 3 %. With the orientation known, the released model keeps its accuracy under any rotation.
- So a constructive method exists in principle: estimate the rotation to the benchmark's canonical frame, undo it, run the released model, rotate the flow back. The labels for such an estimator are free (rotate training images by known rotations).
- Open: how accurate the estimate must be (15 deg of tilt error already costs +252 %; job `orient_tol` measures 2 / 5 / 10 deg), and whether the heading can be estimated at all from a pair.
- 8 pairs, one model. Box job `orient_canon` repeats it on the full test set with PanoFlow and frozen RAFT-large. The `--val-so3-derotate` path of `run_raft_shard_baseline.py` has not been run; this local number comes from a scratch script on the P11 harness.

## 5. Position against the literature (searched 2026-11-11)

- PriOr-Flow (ICCV 2025): no rotated evaluation, no rotation or other augmentation described, no zero-flow row, horizontal wrap only.
- SLOF (ECCV 2022 / journal 2023): rotation is used in training (siamese pairs). The test is the standard FLOW360 test set. Rotation invariance is measured for the classification model, not for flow.
- SVI360 (ECCV 2026): dual-branch with a rotated orthogonal view for interpolation; the abstract reports no rotated flow evaluation.
- Sphere-Depth (arXiv 2604.23432, April 2026): pitch/roll sweeps of -40 to +40 deg for 360 monocular depth; spherical-aware models degrade. This is the precedent: a benchmark paper for another task.
- RobustSpring (ICLR 2026): corruption robustness of perspective flow, stereo and scene flow. No camera-orientation axis, no 360.
- Upright adjustment of panoramas is an established preprocessing step (pitch/roll regression + remap). It is the obvious baseline for the remedy and must be a row.
- Three searches for "360 optical flow" + rotation robustness / orientation benchmark returned no flow benchmark with rotated test inputs.

Second pass, 2026-11-12 (seven more searches, three full-text reads):

- SVI360, full text: rotation appears only as training augmentation ("random rotation roll/pitch/yaw, p=0.3, rotation ratio=0.1"). Flow metrics only on upright FlowScape. No PriOr-Flow row.
- SLOF journal version (arXiv 2301.11880), full text: Table I is the standard FLOW360 test set; the rotation table (Table II) is activity-recognition accuracy, not flow error.
- MPF-Net (ECCV 2022): the data generation states the camera does not rotate about the x and z axes. The benchmark is gravity-aligned by construction.
- Unified Spherical Frontend (CVPR 2026, arXiv 2511.18174): trains without rotation and tests under random rotation for classification, detection and segmentation (planar DeepLab 35.0 -> 12.1 mIoU, spherical 28.8 -> 28.1). No flow, no depth. Second precedent for the protocol, and the opposite architectural reading to ours: there the spherical model is the robust one.
- Spherical-GOF (arXiv 2603.08503) is Gaussian opacity fields for reconstruction, not optical flow.
- Panoramic survey (arXiv 2606.27745): lists optical flow under open problems and says no benchmark uses spherical-area-weighted metrics.
- No 2026 paper found that scores 360 flow under rotated inputs, tilt, or non-upright capture.

Rotation in training exists in two of the five flow papers (SLOF, SVI360). None scores flow under rotation. The claim stays as written below.

What may be claimed, per `LITERATURE_SCOPE.md`: the application of the rotated evaluation to
360 flow, the in-domain vs out-of-domain discriminator, and the measured trade-off and remedy.
Not the rotated evaluation itself, and not SO(3) augmentation itself (SLOF trains with rotations).

## 6. The second candidate: heads on released models

Ceilings on released PriOr-RAFT, 24 flowscape:test pairs, official EPE (2026-11-11, Mac CPU):

| Row | EPE all | In-raster | Overshoot | Static | Moving |
|---|---|---|---|---|---|
| PriOr-RAFT | 2.236 | 1.799 | 16.02 | 1.533 | 2.469 |
| oracle static gate (zero where GT < 0.25 deg) | 2.734 | 1.852 | 30.54 | 3.534 | 2.469 |
| oracle band (GT on the top/bottom H/16 rows) | 2.007 | 1.838 | 7.31 | 0.489 | 2.510 |
| both | 2.021 | 1.853 | 7.31 | 0.548 | 2.510 |

- A perfect static gate makes PriOr-RAFT worse (+22 %): it is already more accurate on "static" pixels (1.53 px) than zero flow is (3.53 px).
- A perfect band head is worth 10 % at most, and a learned head recovers only part of that.
- Verdict: the heads fix errors of our short-budget models, not of a released model. This direction is closed for the published state of the art.

## 7. Work packages for the continuation

| WP | Content | Status | Pre-registered reading |
|---|---|---|---|
| 1 | Protocol: fixed Haar rotation per test pair, sphere-weighted geodesic + official EPE, zero row. Rows: PriOr-RAFT, PanoFlow, SLOF (rotation-trained), frozen RAFT-large, OSLO, our RAFT rows. Angle sweep (pitch/roll 0-90 deg) | Harness and four rows done | - |
| 2 | Mechanism: the upright prior. Yaw-only control (must be harmless), per-latitude error maps before and after rotation, a second benchmark (MPF) | Partly done (9.3) | Yaw-only change <= 10 % |
| 3 | Remedy at scale: SO(3) fine-tune of released PriOr-RAFT and PanoFlow on the box, rotation probability in {0.25, 0.5, 1.0}, 10-20k steps | P11 is the probe | Rotated error <= 1.5x the original upright error, upright loss <= 10 % |
| 4 | If WP3 leaves an upright cost: upright-adjustment front end as a baseline; orientation-conditioned variant | Not started | Beats WP3 on the upright/rotated mean |
| 5 | Real video without labels: frame-prediction residual on HEALPix nodes (the P10 instrument) on non-upright footage, released vs fine-tuned weights | Instrument done | Residual falls on non-upright footage |
| 6 | Writing: dissertation and one article | - | - |

The role of HEALPix in this plan is the evaluation: an equal-area node mean is invariant to the
camera orientation, an ERP-pixel mean is not (the polar gap is 5x larger in px than in geodesic
units in PriOr-Flow's own table).

## 8. Risks

- "Use an upright-adjustment network or the IMU": legitimate. It is a row in WP4, and the rotation test also probes reliance on where motion usually is, which upright adjustment does not remove when the content itself is unusual.
- The fine-tune may trade upright accuracy for robustness with no setting that keeps both. Then the result is the measured trade-off curve, still unpublished for flow.
- FlowScape is synthetic and gravity-aligned; WP5 is the only real-footage evidence.
