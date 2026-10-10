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
