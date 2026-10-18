# RAFT parity — where OSLO-RAFT-R departs from RAFT, and what each departure costs

Opened 2026-10-09. Question: is OSLO-RAFT-R under-adapting RAFT (Teed & Deng, ECCV 2020,
arXiv 2003.12039)? Each difference found by reading `oslo_raft_retina.py` against the
paper and `princeton_raft.py` is put behind a flag (default = the validated behaviour, so
every existing checkpoint and number is untouched) and measured one at a time.

## 1. Differences and their flags

| # | RAFT | OSLO-RAFT-R default | Flag (RAFT side) | Arm |
|---|---|---|---|---|
| 1 | flow in estimation-grid cells | radians (one r4 node = 0.064) | `--flow-unit node` | `unit` |
| 2 | bilinear lookup | inverse distance, 3 nearest nodes | `--lookup-interp barycentric` | `bary` |
| 3a | raw dot / sqrt(C) | cosine (normalized features) | `--corr-norm raw` | `raw` |
| 3b | radius 4 (81 samples/level) | 2 rings x 8 (17/level) | `--lookup-rings 4` (33/level) | `rings4` |
| 3c | no matching loss | stencil InfoNCE, weight 0.5 | `--aux-match-weight 0` | `noaux` |
| 4a | 2 residual units per scale | 1 block per level | `--encoder-blocks-per-level 2` | `enc2` |
| 4b | 2-layer motion encoder, SepConvGRU, 2x flow head | 1 layer, 1 GRU | `--update-block raft` | `raftblock` |
| 5a | 12 training iterations | 8 | `--iters 12` | `iters12` |
| 5b | OneCycle 5% warm-up, linear; wd 1e-4 | 30%, cosine; wd 1e-5 | `--onecycle-pct-start 0.05 --onecycle-anneal linear --weight-decay 1e-4` | `sched` |
| all | — | — | all of the above + hidden/context 128 | `raftfull` |

Augmentation also departs from RAFT and is not measured: training pairs get a random
SO(3) rotation with probability 1 (`--so3-prob 1.0`, up to 180°); there is no
photometric jitter, eraser or scale augmentation on real pairs. Arm `noso3` =
`--so3-prob 0`.

Not implemented: crop/patch training with batch 12 (RAFT and OSLO-IC both train on
patches; ours is the full sphere at batch 2), instance/batch norm split between the
encoders, strided-conv downsampling, the 0.25 gradient scale on the upsample mask.

All architecture flags must be repeated at eval of a checkpoint trained with them.

## 2. Protocol

`scripts/raft_parity_ab.sh` on the box. From scratch, `replica360:train -> replica360:val`,
the A1 matched-pair recipe: retina r7 / supervision r6 / estimation r4, 2000 steps,
one-cycle, haversine, AMP, batch 2, seeds 7/11/19. One arm = one difference against `base`.
Metric: improvement over zero flow (%), evaluated once at the final step.

## 3. Round 1 — items 1 and 2 (2026-10-10, git `89abd4e5a`)

Mean ± population std over 3 seeds:

| Arm | Global | Poles | Equator |
|---|---|---|---|
| `base` | 79.96 ± 1.01 | 64.77 ± 3.09 | 83.33 ± 0.64 |
| `unit` | 81.44 ± 0.19 | 62.26 ± 0.21 | 86.17 ± 0.06 |
| `bary` | 81.79 ± 2.53 | 65.23 ± 5.18 | 85.98 ± 1.98 |
| `both` | 80.43 ± 1.90 | 59.73 ± 3.53 | 85.27 ± 1.81 |

Paired difference against `base` at the same seed (7 / 11 / 19):

| Arm | Region | Δ per seed | Mean | Paired t (2 df) |
|---|---|---|---|---|
| `unit` | global | +3.06 / +0.97 / +0.42 | +1.48 | 1.8 |
| `unit` | equator | +3.38 / +3.22 / +1.93 | +2.84 | 6.2 |
| `unit` | poles | +1.84 / −4.77 / −4.60 | −2.51 | −1.2 |
| `bary` | global | −0.18 / +2.19 / +3.49 | +1.83 | 1.7 |
| `bary` | equator | +0.43 / +3.75 / +3.78 | +2.65 | 2.4 |
| `bary` | poles | −1.55 / −1.48 / +4.41 | +0.46 | 0.2 |
| `both` | global | +3.98 / −2.27 / −0.28 | +0.48 | 0.3 |
| `both` | equator | +4.57 / +0.13 / +1.13 | +1.94 | 1.4 |
| `both` | poles | +2.53 / −11.96 / −5.67 | −5.03 | −1.2 |

Reading:

- `base` reproduces the recorded numbers for this recipe (+77.8% A1, +80.9% P1 control).
- No arm separates from `base` on the global metric. The only difference that clears
  noise is `unit` at the equator (p ≈ 0.025, uncorrected for the number of comparisons).
- `unit` lands on nearly the same point at every seed (global 81.67 / 81.19 / 81.46,
  poles 62.2 / 62.0 / 62.5): it trades poles for equator, stably.
- `bary` widens the seed spread (global 78.4 – 84.5).
- The two changes do not add up; `both` at seed 11 has poles at 54.8.
- Neither flag becomes a default on this evidence.

Scope: replica360 is a supra-node regime (target p50 11.9°, r4 node spacing 3.66°). The
measured advantage of the barycentric lookup is sub-node (response to a 0.1-node shift
against the true directional change, cos-sim 0.996 vs 0.811 for inverse distance at r4),
so the deciding test for items 1–2 is corrected flow360, not this one.

Correction to the opening hypothesis: inverse-distance weights are not direction-blind
near a node (cos-sim 0.81); they are about 10× less accurate on a linear field (error
0.056 vs 0.005 node spacings at r4).

## 4. Round 2 — items 3 to 5 (pending)

Arms `raw rings4 noaux enc2 raftblock iters12 sched raftfull`, same protocol; the round-1
runs are skipped by the script. Parameter counts at the round-1 configuration: `enc2`
+0.5M, `raftblock` +1.3M, `rings4` +0.2M, `raftfull` 6.6M (base 1.56M). `raftfull` has
not been run on the box; its 24 GB fit at retina r7 / batch 2 / 12 iterations is unknown.

## 5. Top-down — small against the RAFT-large layout (replaces round 2 as next; pending)

Decision 2026-10-10: the one-difference-at-a-time round is deferred. Round 1 showed
seed noise of 1–2.5 points and non-additive effects, so isolated arms mostly return
nulls; and the model (1.56M) is RAFT-S class while every comparison row is RAFT-large
(RAFT's own small/large gap: Sintel clean 2.21 vs 1.43, KITTI 7.51 vs 5.04). The whole
set is tested first; single items are ablated out of it only if it wins.

`scripts/raft_parity_large.sh`:

- Arms: `base` (1.56M) vs `raftfull` (every §1 flag + hidden/context 128; 6.6M at the
  round-1 configuration). Optional `large` = the architecture flags only (rings 4,
  2 encoder blocks, RAFT update block, hidden/context 128), separating capacity from
  recipe.
- Data: from scratch on `replica360:train,flowscape:train`. flowscape alone is
  sub-node-dominated (p50 2.47° vs 3.66° node spacing), the composition on which
  chairs360-only training failed to bootstrap; replica360 supplies the supra-node mass.
- Scored on `flowscape:val` (during and after training) and `replica360:val`
  (eval-only pass). flowscape has the headroom replica360 lacks (OSLO +66.0 vs
  PanoFlow +92.6 on flowscape:test, §7 of `UNIVERSALITY_TABLE.md`); test is not touched.
- 10 000 steps, seed 7, same grids and optimizer settings as §2.
- A one-step `raftfull` preflight runs first, so a memory failure shows in the first
  minute. Fallback: `BATCH_SIZE=1 GRAD_ACCUM=2`, applied to both arms.

One seed decides only a gap well outside the round-1 noise; a gap of a few points
needs seeds 11 and 19 (`SEEDS="7 11 19"`, finished runs are skipped).

### 5.1 Box result (2026-10-13, git `ccbaa8746`, seed 7, 10 000 steps)

| Arm | Params | flowscape:val global | act₀.₅ | poles | equator | replica360:val global | s/step |
|---|---|---|---|---|---|---|---|
| `base` | 1,558,768 | **+75.8** | +77.4 | +72.8 | +64.5 | **+85.7** | 1.0 |
| `raftfull` | 6,625,982 | +0.1 | +0.0 | +0.1 | +0.2 | 0.0 | 2.7 |

`raftfull` never left the zero-flow predictor. Its train error repeats the zero-flow
error of the same batches every 3100 steps (one pass over the data), while its aux
matching loss fell normally (3.1 → 1.3): the features learned to match, but the flow
head never moved. This is a bootstrap failure, not a capacity verdict. It fits in
24 GB at batch 2.

`base` was still improving at the end (69.9 → 74.1 → 75.8 over the last 4k steps),
and its late train error (0.4–0.8°) is in the range of its val error (0.60°). There is
no train/val gap that would indicate a data shortage at this size.

### 5.2 Local bisection (CPU, replica360, ret5/sup4/est3, batch 2, 300 steps, seed 7)

Reference: `base` at this configuration = 69.75 (§7.3).

| Arm | Params | val@100 | val@200 | final |
|---|---|---|---|---|
| `sched` | 1.53M | 45.8 | 68.7 | 69.3 |
| `raw` | 1.53M | 44.3 | 63.4 | 67.9 |
| `raftblock` | 2.88M | 33.8 | 67.8 | 69.2 |
| `iters12` | 1.53M | 22.7 | 66.2 | 69.3 |
| `enc2` | 2.05M | 61.6 | 62.6 | 66.2 |
| `rings4` | 1.75M | 1.0 | 62.7 | 68.0 |
| `large` | 6.57M | 0.1 | 56.0 | 65.5 |
| `large` + `sched` | 6.57M | 0.1 | 0.6 | 0.5 |
| `large` + `raw` | 6.57M | 0.1 | 0.8 | 0.7 |
| `large` + `iters12` | 6.57M | −0.4 | 0.8 | 0.6 |
| `large` + `unit` | 6.57M | −4.6 | 63.2 | 65.0 |
| `large` + `bary` | 6.57M | 14.5 | 57.6 | 64.4 |
| `raftfull` | 6.57M | −0.1 | 0.0 | 0.0 |

- The failure reproduces locally. Every component trains on its own, so it is not an
  implementation defect in the round-2 code.
- `large` trains but bootstraps late (still at zero at step 100). `rings4` is the only
  single item with the same late start.
- Three of the five recipe items each block the `large` bootstrap within 300 steps:
  the RAFT schedule (`sched`), raw correlation (`raw`) and 12 iterations (`iters12`).
  Flow unit and barycentric lookup do not.
- Limit: 300 steps cannot separate "bootstraps later" from "never bootstraps" for the
  three pairwise arms; only the full combination was observed for 10k steps.
- Consequence: the capacity question is answered by `large` against `base`, not by
  `raftfull`. The RAFT recipe items stay off for the large layout until a bootstrap
  aid (longer warmup, iteration curriculum, warm start) is tested.

### 5.3 Box result for `large` (2026-10-15, git `9b7a436a5`, seed 7, 10 000 steps)

| Arm | Params | flowscape:val global | act₀.₅ | poles | equator | replica360:val global | replica poles | s/step |
|---|---|---|---|---|---|---|---|---|
| `base` | 1,558,768 | **+75.8** | **+77.4** | **+72.8** | +64.5 | **+85.7** | **+78.9** | 1.0 |
| `large` | 6,625,982 | +72.9 | +74.2 | +67.8 | +63.2 | +76.3 | +63.3 | 2.1 |

flowscape:val trajectory (global): `large` 54.9 / 65.1 / 67.5 / 71.6 / 72.9 at
2k/4k/6k/8k/10k, `base` 52.8 / 63.8 / 69.9 / 74.1 / 75.8.

- `large` bootstraps normally on the box and leads up to 4k steps; `base` overtakes it
  by 6k and stays ahead.
- At the end `large` is behind on every region of both val sets. The flowscape gap
  (−2.9) is at the edge of the round-1 seed noise (1–2.5); the replica360 gap (−9.4
  global, −15.6 poles) is well outside it.
- Train loss on the same batches is slightly lower for `large` near the end (e.g.
  0.696 vs 0.728 at step 10k), while its val error is higher. This is consistent with
  a small generalization gap at 6.6M parameters on this data, but one seed does not
  establish it.
- Verdict at this budget (10k steps, batch 2, one seed): the RAFT-large layout does
  not help, at equal steps and at 2.1× the cost per step. `base` stays the
  architecture. Not tested: longer schedules, where a larger model may catch up.

## 6. EPE table against frozen RAFT-small and RAFT-large (pending)

`scripts/raft_parity_epe.sh` scores, on the same pairs and in ERP pixels
(`run_epe_eval.py`, the P2A readout): zero flow, the grid floor (`oracle`), frozen
TorchVision `raft_small` and `raft_large`, and OSLO-RAFT-R checkpoints (default: the
three round-1 `base` seeds, on `replica360:val`). `run_epe_eval.py` now rebuilds a
checkpoint's architecture from the flags saved in it, so `raftfull` checkpoints score
without extra arguments.

What the table can and cannot show. The RAFT rows are zero-shot perspective models
(TorchVision small = Chairs + Things; large = + Sintel/KITTI/HD1K fine-tuning); the
OSLO rows are trained in-domain. The small/large ratio measured here against RAFT's own
(Sintel clean 2.21 / 1.43 = 1.55×) says how much of a gap to RAFT-large is model size.
It is not a matched-backbone comparison: that needs RAFT-small trained on the same
ERP pairs for the same steps.

## 7. Patch training (implemented 2026-10-10; local validation only)

RAFT and OSLO-IC train on crops; OSLO-RAFT-R trained on the whole sphere, which pins
the batch at 2 and the grids at retina r7 / estimation r4 / supervision r6. Against
RAFT on a 1024×512 ERP (0.35°/px at the equator; shards are 1024×512 or 1280×640) that
is 1.3× coarser at the input (0.46°), 1.3× coarser at the estimation grid (3.66° vs
2.81°) and 2.6× coarser at the output (0.92°, ×4 upsampling where RAFT does ×8 back to
native). Parity is retina r8 / estimation r5 / supervision r8, at 4× the nodes — the
cost a quarter-sphere patch cancels.

### 7.1 Mechanism

- `crop_pyramid` (`healpix_pyramid.py`): a patch is a cap of pixels of the coarsest
  correlation level plus all their descendants. Nested ordering keeps each parent's
  four children contiguous, so the patch is a nested pyramid over re-indexed nodes and
  the model runs on it unchanged (any architecture flag).
- Border: out-of-patch conv neighbors are invalid slots (zero padding); lookup queries
  farther than one whole-sphere node spacing from every patch node return zero
  correlation (RAFT's out-of-frame behaviour); nearer ones interpolate with the border
  node standing in for missing neighbors.
- Dataset: samples only the patch's nodes; `batch-size` consecutive samples share a
  patch. Flags `--patch-coarse-pixels K` (0 = whole sphere) and `--patch-bank N`.
  Validation always runs on the whole sphere.

### 7.2 Correctness (CPU, real HEALPix, ret6/sup5/est4, 12-of-48 patches at 5 centers)

Structure identical to the whole sphere (nodes, valid neighbors, pooling, descendants).
Error exactly 0 against the whole-sphere computation for: one SDPAConv at nodes with all
8 neighbors inside; convex upsampling at the same nodes; the finest-level lookup at
nodes more than 6 spacings from the border (both interpolation modes). No point on the
sphere sits farther than 0.80 spacings from its nearest node, so the 1.0 cutoff has
margin.

Coarse correlation levels see little real context on a quarter-sphere patch — share of
lookup entries equal to the whole-sphere value / zeroed as outside: r4 79.9% / 4.2%,
r3 64.5% / 7.6%, r2 33.5% / 12.8%, r1 6.6% / 22.9%.

### 7.3 Does a patch-trained model transfer to the whole sphere? (local, one seed)

replica360, retina r5 / supervision r4 / estimation r3, 300 steps, one-cycle, seed 7,
CPU; validation on the whole sphere (first 64 pairs). Same nodes per step in both arms.

| Arm | val@100 | val@200 | final global | poles | equator |
|---|---|---|---|---|---|
| whole sphere, batch 2 | 59.25 | 66.79 | 69.75 | 53.13 | 73.85 |
| patches of 1/3 sphere, batch 6 | 42.90 | 66.27 | 68.71 | 51.25 | 72.76 |

Reading: the patch-trained model reaches the whole-sphere model's level on whole-sphere
validation (−1.0 point, inside the 1–2.5 point seed noise of §3), after a slower start.
The coarse-level context loss of §7.2 does not prevent transfer at this scale. One seed,
300 steps, small grids: it shows the mechanism works, not that patches are better.
(The two arms' zero baselines differ by 0.5% — 13.379 vs 13.315 — because the 64-pair
cap lands on a batch boundary: 64 pairs at batch 2, 66 at batch 6.)

### 7.4 What a patch costs: samples per step, not nodes (local, one seed)

Same protocol as §7.3 (replica360, estimation r3, 300 steps, seed 7, whole-sphere
validation on the first 64 pairs); patches are 1/3 of the sphere.

| Retina / sup | Training | Batch | val@100 | val@200 | Final global | Poles | Equator |
|---|---|---|---|---|---|---|---|
| r5 / r4 | whole sphere | 2 | 59.25 | 66.79 | 69.75 | 53.13 | 73.85 |
| r5 / r4 | patches | 6 | 42.90 | 66.27 | 68.71 | 51.25 | 72.76 |
| r5 / r4 | patches | 2 | 0.83 | 22.68 | 24.81 | — | — |
| r6 / r5 | patches | 2 | −0.26 | 0.36 | 0.50 | — | — |
| r6 / r6 | patches | 2 | −0.17 | 0.43 | 1.05 | 0.69 | 1.17 |
| r6 / r6 | whole sphere | 2 | 38.98 | 71.16 | 73.63 | 52.49 | 79.55 |
| r6 / r6 | patches | 6 | 61.88 | 62.39 | 72.89 | 51.02 | 79.36 |

Reading:

- **Estimation nodes supervised per optimizer step decide the outcome at this budget.**
  Whole sphere at batch 2 and third-sphere patches at batch 6 supervise the same 1536
  estimation nodes per step and land within a point of each other at both resolutions;
  patches at batch 2 supervise a third of that and reach 25% or ~1%.
- **So patches buy memory, not compute.** For equal learning per step the batch must
  grow by the inverse of the patch fraction; node count per step, and so time, is then
  unchanged. What drops is the memory of one micro-batch (with `--grad-accum`) and the
  largest tensor of a single sample — the room a larger model or a finer grid needs.
- **Supervision = retina (×8 convex upsampling, 64 descendants) is validated.** r6/r5
  and r6/r6 fail identically at batch 2 and r6/r6 learns on the whole sphere and on
  patches at batch 6; the new path behaves like the old one.
- **The finer grids helped here:** r6/r6 beats r5/r4 by about 4 points global and 6 at
  the equator, on the whole sphere and on patches alike, with the estimation grid held
  at r3. One seed each; retina and supervision changed together.

### 7.5 Box run — `scripts/raft_parity_patch.sh`

replica360, the §2 recipe at 2000 steps, seed 7; whole-sphere reference = the round-1
`base` (79.96 ± 1.01 over three seeds).

| Arm | Retina / est / sup | Training | Samples per step |
|---|---|---|---|
| `r7_patch` | r7 / r4 / r6 | patches 12 of 48 coarse pixels | batch 8 |
| `r8_native` | r8 / r5 / r8 | patches 48 of 192, bank 24 | batch 4 × accum 2 |

`r7_patch` costs what `base` costs and tests the transfer at production scale.
`r8_native` is RAFT's resolution on a 1024×512 ERP at 4× the nodes per step (expect
about 4× the time); a one-step preflight covers its two unknowns, training memory and
whole-sphere validation at 786k nodes per frame.

Result (2026-10-16, git `9b7a436a5`, seed 7). Seed-7 whole-sphere reference from
round 1 (derived from §3: `unit` seed-7 values minus the paired Δ).

| Arm | Global | Poles | Equator | Time for 2000 steps |
|---|---|---|---|---|
| `base`, whole sphere r7/r4/r6 (round 1) | 78.6 | 60.4 | 82.8 | — |
| `r7_patch` | 79.9 | 50.3 | 86.9 | 41 min |
| `r8_native` | **81.3** | **64.6** | 85.9 | 107 min (+18 min one-time pyramid build) |

- Both unknowns of `r8_native` clear: training fits at batch 4 and whole-sphere
  validation at 786k nodes per frame fits on 24 GB.
- Patch training transfers at production scale: `r7_patch` matches the whole-sphere
  global score (+1.3, inside the round-1 spread of ±1.0).
- `r7_patch` poles are 10 points below the seed-7 reference. Poles are the noisiest
  region (round-1 std 3.1, single-seed drops of −12 seen in `both`); `r8_native`, also
  patch-trained, does not show the drop. Unexplained on one seed.
- `r8_native` is ahead in all three regions (+2.7 global, +4.3 poles, +3.1 equator),
  at 2.6× the time of `r7_patch`. One seed; the global gap is about 2.7 std of the
  round-1 spread.
- Scope: replica360 is supra-node (target p50 11.9°). The finer estimation grid
  (r5, 1.83°) and output grid (r8, 0.23°) are expected to matter most below the r4
  node spacing, which is flowscape's regime (p50 1.28°).

### 7.6 Native resolution on the top-down recipe

`scripts/raft_parity_native_mix.sh`: `r8_native` settings on replica360 + flowscape,
10 000 steps, seed 7, scored on flowscape:val and replica360:val. Reference:
`raft_parity_large/base_s7` (+75.8 / +85.7). Expected time about 9 h (3.2 s/step).

Result (2026-10-17, git `28bb83bae`, seed 7, 10 000 steps, 8.3 h = 3.0 s/step):

| Arm | flowscape:val global | act₀.₅ | poles | equator | replica360:val global | replica poles | replica equator |
|---|---|---|---|---|---|---|---|
| r7/r4/r6 whole sphere (`raft_parity_large/base_s7`) | 75.8 | 77.4 | **72.8** | 64.5 | **85.7** | **78.9** | **87.8** |
| r8/r5/r8 quarter-sphere patches | **78.2** | **79.3** | 63.5 | **75.4** | 81.7 | 69.2 | 85.6 |

Absolute flowscape:val error: global 0.525° vs 0.598°, equator 0.293° vs 0.457°,
poles 2.26° vs 1.68°.

flowscape:val trajectory (global): r8 38.1 / 71.8 / 75.6 / 77.4 / 78.2 at 2k…10k,
r7 52.8 / 63.8 / 69.9 / 74.1 / 75.8. r8 starts slower, leads from 4k, and the gap
narrows (8.0 → 5.7 → 3.3 → 2.4).

- r8 wins where motion is small: flowscape global +2.4, equator +10.9 points (error
  −36%).
- r8 loses where motion is large: poles −9.4 (flowscape) and −9.8 (replica360), and
  replica360 global −4.1. Polar zero-flow error is 5× the equatorial on flowscape
  (6.19° vs 1.19°); replica360 is large-motion throughout (p50 11.9°).
- Patch coverage is not the cause. Measured on the seed-7 bank of 24: polar nodes are
  covered by 30% of patches (equator 23%), none uncovered, mean depth from the patch
  border 19° (equator 16°).
- Two candidate mechanisms, both of which cut large displacements:
  1. Search reach. With 3 pooling levels the coarsest correlation level is r1
     (29.3° spacing) at est r4 but r2 (14.7°) at est r5, so the 2-ring lookup reaches
     about 58° in the r7 configuration and about 29° in the r8 one.
  2. Patch border. Mean depth from the border (16–19°) is of the order of the large
     displacements, so their endpoints often fall outside the patch, where
     correlation is zeroed.
- Confound: resolution and patch training change together; the r8 run cannot
  separate them.

### 7.7 Displacement bands, and next

1. `scripts/raft_parity_bands.sh` (eval only, minutes): error per displacement band
   crossed with poles/equator for both checkpoints, on flowscape:val and
   replica360:val. If the r8 deficit sits in the large-displacement bands at every
   latitude, the mechanism is reach or border; if it is at the poles in every band,
   it is something polar.
2. If reach/border: r8 with `--corr-pool-levels 4` (coarsest level r1, reach restored
   to about 58°) and patches of 12/48 r1 pixels, same recipe.

Band result (2026-10-18, git `032a488b7`, `scripts/raft_parity_bands.sh`). Improvement
over zero flow, %, r7 whole → r8 native; node fraction in parentheses.

flowscape:val

| Band (deg) | all | poles | equator |
|---|---|---|---|
| 0–0.25 (0.22) | −62.3 → −24.4 | −331 → −136 | −42.1 → −11.5 |
| 0.25–0.5 (0.08) | 32.7 → 54.6 | 59.1 → 57.4 | 31.2 → 54.8 |
| 0.5–1 (0.15) | 66.0 → 76.4 | 31.4 → 39.2 | 66.1 → 76.6 |
| 1–2 (0.15) | 77.5 → 84.5 | 55.5 → 65.5 | 78.4 → 85.3 |
| 2–4 (0.17) | 84.8 → 89.3 | 70.0 → 69.8 | 85.7 → 89.7 |
| 4–8 (0.21) | 86.3 → 84.2 | **79.6 → 70.1** | 85.6 → 88.2 |
| 8–16 (0.03) | 69.4 → 60.0 | **66.7 → 56.2** | 71.2 → 77.2 |

replica360:val

| Band (deg) | all | poles | equator |
|---|---|---|---|
| 1–2 (0.02) | 52.2 → 59.9 | 32.1 → 30.0 | 54.9 → 62.0 |
| 2–4 (0.07) | 74.8 → 79.2 | 65.5 → 68.4 | 77.6 → 81.5 |
| 4–8 (0.20) | 86.8 → 89.2 | 80.9 → 80.9 | 89.3 → 91.6 |
| 8–16 (0.40) | 91.5 → 92.6 | 86.4 → 83.0 | 93.3 → 94.8 |
| 16–32 (0.28) | 89.7 → 83.0 | **81.3 → 68.2** | **92.0 → 86.8** |
| 32+ (0.03) | 51.6 → 29.0 | **50.1 → 24.4** | **48.9 → 33.0** |

- r8 is better in every band below 4° (flowscape) or 16° (replica360), at every
  latitude, including the static band (r8 predicts less spurious motion).
- Above 16° r8 loses at the equator too (replica360 −5.2 and −15.9). This part is a
  displacement effect, consistent with the halved search reach or the patch border.
- At 4–16° r8 loses at the poles only (flowscape −9.5 and −10.5) while gaining at the
  equator in the same bands (+2.6, +6.0). That part is latitude-specific, and neither
  reach nor patch border explains it: both are latitude-uniform on HEALPix.
  Candidate (not tested): the encoders keep their layer count, so their angular
  receptive field halves at r8, which would hurt low-texture polar content most.
- flowscape's global gain comes from the 0–4° bands (76% of nodes); its loss from
  4–16° (24%), which is mostly polar.

Next: the `--corr-pool-levels 4` run (item 2 above) tests the reach. Prediction: it
recovers the bands above 16° at the equator. If the polar 4–16° deficit remains, the
receptive-field candidate is next.

Patch-border loss, measured (2026-10-18, local, replica360:train first 40 pairs,
seed-7 banks of 24, r6 nodes): percentage of supervised nodes whose GT endpoint
leaves the patch.

| Band (deg) | cp3 bank (48/192 r2): poles / equator | cp4 bank (12/48 r1): poles / equator |
|---|---|---|
| 2–4 | 3.1 / 4.0 | 2.6 / 3.9 |
| 4–8 | 6.5 / 6.9 | 9.1 / 9.0 |
| 8–16 | 11.2 / 13.2 | 16.3 / 14.1 |
| 16–32 | 13.6 / 18.6 | 24.3 / 21.3 |
| 32+ | 22.1 / 98.9 | 21.2 / 95.5 |

- Latitude-uniform at 4–16° (poles ≈ equator), so the border does not explain the
  polar 4–16° deficit; the receptive-field candidate stands.
- At 16–32° the border removes 15–19% of the supervision: a minority, so a
  contributor, not the whole equatorial loss there. Above 32° nearly every
  equatorial endpoint leaves the patch.
- The cp4 bank's caps (12 r1 pixels, blockier) lose more at 8–32° than cp3's. The
  cp4 run therefore tests reach with a slightly worse border: an improvement above
  16° is attributable to reach; no improvement does not rule reach out.


### 7.8 Encoder receptive field at r8 (flag implemented 2026-10-18; box run pending)

`--encoder-est-blocks N` adds N residual blocks at the estimation level of both
encoders (default 0; the state dict of existing checkpoints is unchanged). ab arm
`estblocks` = `--encoder-est-blocks 2`.

Geometric receptive field of the fnet output (gradient of one output node over the
input, random init, GroupNorm statistics removed because they couple the whole
sphere; local pyramids ret5/est3 and ret6/est4, the same one-resolution shift as
r7/r4 → r8/r5):

| Encoder | Max radius | 90% of gradient mass |
|---|---|---|
| ret5/est3 | 43.4° | 22.5° |
| ret6/est4 | 21.8° | 11.4° |
| ret6/est4 + 1 est block | 33.0° | 11.5° |
| ret6/est4 + 2 est blocks | 44.3° | 11.5° |
| ret6/est4 + 3 est blocks | 55.5° | 11.6° |

- The one-resolution shift halves the receptive field exactly (HEALPix is
  self-similar: same node counts, half the angle).
- Two blocks at the estimation level restore the maximum radius (44.3° vs 43.4°).
  The 90% radius does not move at initialization; whether training uses the wider
  field is what the run measures.
- Cost at the small local config: +666k parameters (1.53M → 2.20M).

Run: `ARMS="base estblocks"` on `scripts/raft_parity_native_mix.sh` (`base` is
skipped as done). Prediction: recovers the polar 4–16° bands of §7.7 and leaves the
bands above 16° at the equator unchanged.
