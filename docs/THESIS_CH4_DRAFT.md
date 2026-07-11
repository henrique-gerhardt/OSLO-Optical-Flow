# Ch. 4 (draft) — OSLO-RAFT-R: Decoupling the Retina from Estimation, and the Real-Correspondence Wall

**Status: first draft, 2026-07-07.** This chapter integrates two experimental arcs:

- **Act I** — the single-grid OSLO-RAFT campaign, consolidated in
  [`OSLO_RAFT_DOCS.md`](OSLO_RAFT_DOCS.md): every variant plateaus at a **+2.9%**
  appearance-prior ceiling on FLOW360, correlation is inert, a differential head is
  anti-correlated with the true motion.
- **Acts II–III** — the OSLO-RAFT-R (retina) campaign, designed in
  [`OSLO_RAFT_RETINA_PLAN.md`](OSLO_RAFT_RETINA_PLAN.md) and run 2026-07-05 → 07-07:
  the redesign makes spherical correlation **load-bearing for the first time**
  (Gate R1, +88.4% on real large motion), the FLOW360 gate still fails twice
  (Gate R2), and three controlled probes isolate the remaining cause.

All numbers below are measured (checkpoints and `oslo_raft_metrics.json` files under
`/outputs/` on the GPU box; commands in §9). Cells marked **(pending)** await the three
bookkeeping runs listed in §9.4. Where this chapter and the running logs disagree, the
logs are authoritative for dates/commands, this chapter for the narrative.

---

## 1. Chapter summary: three acts, one variable left standing

**Act I (the wall).** A HEALPix-native RAFT — SDPAConv spatial operator, spherical
exp-map correlation lookup, ConvGRU updater, geodesic loss, full-SO(3) augmentation —
trains stably, overfits perfectly, and plateaus at **+2.9% active-subset improvement
over zero-flow on FLOW360** no matter what: resolution r4→r6, supervision grids, loss
reweighting, and, decisively, with correlation ablated (frame-1-only) or with context
ablated. Two disjoint input pathways converge to the same flow field, so +2.9% is the
ceiling of a *frame-1 appearance prior*; frame 2 contributes nothing. A differential
(Lucas–Kanade) head built for sub-pixel motion underperforms zero-flow (−4.4% active)
and is *anti-correlated* with the true motion exactly where it is most confident
(cos −0.07 overall → −0.18 in the top-5% confidence set). Act I's diagnosis:
median inter-frame motion (0.099°) is far below the resolving power of any affordable
spherical grid — "the sub-node motion wall."

**Act II (the wall moves).** That diagnosis conflated three separable defects: (i) the
input "retina" grid was tied to the estimation grid, discarding image detail the
correlation could have used; (ii) the `ang2pix`-snap lookup is piecewise-constant in
sub-node flow, so correlation *could not* express sub-node motion; (iii) nothing
bootstraps matchable features. OSLO-RAFT-R fixes all three (retina r7 input grid,
interpolated continuous lookup, auxiliary matching loss) plus four amendments that
gate-driven debugging proved individually necessary (§3.3). The result: on real
Replica360 motion (p50 ≈ 12°), the model reaches **1.56° mean geodesic error vs a
13.53° zero-flow baseline (+88.4%)** — three-seed mean **1.74° ± 0.25 (+87.1 ±
1.9%)** — and ablating correlation collapses it to 27.5° — correlation is finally,
demonstrably load-bearing (**Gate R1 passed**).

**Act III (the wall, correctly named).** On real FLOW360 pairs (p50 0.097°) the gate
still fails: **−2.4%** after the standard mix (Stage B) and **−2.9%** after a
flow360-only, motion-weighted fine-tune with a small-angle curriculum (Stage B′) —
never positive at any evaluation in 15k combined steps. But three controlled probes
dismantle every candidate explanation except one. The centerpiece: the *same* B′
checkpoint, on the *same* FLOW360 frames, at the *same* motion scale, scores
**+80.6% (0.046° residual ≈ 0.13 ERP pixels) when frame 2 is a resample of frame 1
at the true motion, and −32.2% when frame 2 is the real rendered next frame.** The
architecture resolves sub-pixel motion an order of magnitude below its input grid —
when the brightness-constancy assumption holds exactly. What remains is therefore
neither grid resolving power, nor training signal, nor motion scale per se, but the
**robustness of sub-pixel correspondence to real inter-frame appearance change**
(occlusion, shading, independently rendered images).

The chapter's contribution is this elimination chain, each link a measured,
controlled experiment (§7.3).

---

## 2. Background: the single-grid result being revised

Full account in [`OSLO_RAFT_DOCS.md`](OSLO_RAFT_DOCS.md); the rows needed here:

| rung | variant | active₀.₂₅ | active₀.₅ | active₁.₀ | global | reading |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | single-res r4 baseline | +2.90% | +2.70% | +1.20% | −1.10% | learns; capped |
| 4 | `--ablate-corr` (frame-1-only) | +2.90% | +2.69% | +1.19% | −1.10% | correlation inert |
| 5 | `--ablate-context` (corr-only) | +2.90% | +2.70% | +1.19% | −1.10% | frame 2 unused |
| 7 | differential (LK) head | −4.4% | — | — | −16% | actively wrong |

(FLOW360 val, 512-pair subset; motion p50 0.099°, p90 0.580°, p95 0.773°.)

The retina plan re-read this result against published 360° pipelines, which wrap an
*unchanged pixel-raster* RAFT whose correlation grid (8 px ≈ 2.81° on ERP) is
**coarser** than our r5/r6 grids — yet frozen RAFT beats zero-flow by +44% on the
FLOW360 test active₀.₅ subset. So grid coarseness alone could not be the wall. The
plan identified three architectural defects instead:

1. **Retina = estimation grid.** The frames were sampled at the estimation
   resolution, throwing away the image detail that sub-node matching needs. RAFT
   itself estimates at 1/8 resolution but *sees* full-resolution pixels.
2. **Snap lookup.** The `ang2pix` nearest-node gather makes the correlation response
   piecewise-constant in the flow: for sub-node flow updates the lookup literally
   cannot change its answer, so gradients through it are zero almost everywhere.
3. **No matching bootstrap.** End-to-end training must *discover* matchable features
   through the inert lookup — a chicken-and-egg failure.

---

## 3. OSLO-RAFT-R: design and the four measured amendments

### 3.1 Three grids

| role | HEALPix res. | nodes | spacing | carries |
| --- | --- | --- | --- | --- |
| retina | r7 | 196,608 | 0.46° | frames, feature/context encoders |
| supervision | r6 | 49,152 | 0.92° | loss targets, metrics, upsampled output |
| estimation | r4 | 3,072 | 3.66° | correlation stencil, ConvGRU state |

Constraint: est < sup ≤ retina. For reference, the shards' ERP source is
≈ 0.35°/pixel, so r7 slightly oversamples the source raster and r8 would be past it.
The shard format needed no change: shards store full-resolution ERP and node sampling
is reader-side, so retina and supervision grids are just two query sets against the
same record.

### 3.2 Interpolated lazy correlation lookup

Instead of snapping query directions to nodes, frame-2 features are interpolated at
the continuous query direction (inverse-distance weights over the k=3 nearest of the
node's lookup window, chord geodesic `2·asin(‖a−b‖/2)` for fp32 stability). By
linearity, interpolating features then dotting equals bilinearly sampling the
correlation volume (Σwₖ(f1·f2ₖ) = f1·(Σwₖ f2ₖ)) — RAFT's own trick, done lazily so
the full N×N volume is never materialized. A 17-point stencil (center + two staggered
rings of 8) is evaluated at each of the frame-2 feature-pyramid levels, scaled by each
level's node spacing. Unit tests confirm the old snap lookup is bit-identical under
0.3-node flow while the interpolated lookup responds smoothly — defect (2) fixed by
construction.

### 3.3 Four amendments, each individually necessary

Each was forced by a measured failure of the decisive CPU gate (§3.4) and is recorded
with its evidence in `OSLO_RAFT_RETINA_PLAN.md` §5.3/§6/§9.2:

1. **Position-free matching features.** Node xyz stays in the context encoder only.
   With xyz in the matching features, every node prefers *itself* — a built-in
   self-match (zero-flow) bias, and a plausible contributor to Act I's inert
   correlation.
2. **Cosine correlation units.** The inherited `/√C` on already-normalized dot
   products left the correlation a ≤0.1-magnitude whisper against O(1) context
   features; the GRU never heard it.
3. **Auxiliary stencil-matching loss** (soft-target InfoNCE over the lookup window,
   weight 0.5, warmed up alone for the first quarter of training). End-to-end
   training *never* develops matchable features on any budget tested; with direct
   matching supervision, features reach their entropy floor in ~300 steps
   (match accuracy 0.90–0.99). Defect (3) fixed — by supervision, not by hope.
4. **Correlation skip into the GRU input.** Even with perfect features — a linear
   ridge probe decodes the correlation stencil to flow at cos 0.99 — the
   motion-encoder→GRU path alone never aligns to the signal on small budgets; the
   skip makes the decode learn in ~450 steps.

The final model is 1,558,768 parameters (hidden 96, context 64, 8 train / 12 eval
GRU iterations), with encoder and iteration gradient checkpointing.

### 3.4 The decisive CPU gate

Before any GPU run: recover a *held-out sub-node* rotation (0.3× the estimation-node
spacing) from synthetic multi-band texture. Full model: direction cosine similarity
**0.997**; correlation-ablated control: **−0.019**. The first OSLO-RAFT variant in
which correlation is load-bearing at all.

---

## 4. Experimental setup

**Data.** Sharded full-resolution ERP pairs (reader-side node sampling), sources
`replica360`, `mpf`, `flow360`. Motion regimes measured on the validation splits:

| dataset (val) | zero-flow global | target p50 | p90 | regime |
| --- | --- | --- | --- | --- |
| replica360 | 13.53° | 11.88° | 24.18° | large (≈3× est-node spacing) |
| flow360 | 0.211° | 0.097° | 0.584° | sub-pixel (≈0.3 ERP px, 0.2 retina spacings) |

**Synthetic-rotation motion source.** With probability p a training/val record is
replaced by an *exact* rotation pair: frame 2 is frame 1 resampled at a random
rotation R, GT endpoint = R p exactly. This provides (a) dense exact supervision at a
controllable motion scale, and (b) — used in reverse in §7 — a *clean-correspondence
control*: same frames, same motion statistics, but perfect brightness constancy and a
rigid field.

**Metrics.** Mean geodesic error (degrees) at the supervision grid, reported as %
improvement over the zero-flow baseline; subsets: global, active>{0.25°,0.5°,1.0°},
poles (|lat|>60°), equator (|lat|<30°), ERP seam. Act I numbers used a 512-pair val
subset; retina-era numbers use the full val (~790 pairs for flow360). Same metric
schema throughout.

**Protocol.** Staged training with pre-registered gates (from the retina plan):
Stage A (large motion, replica360 + synth-rot 0.5 @ 1–15°) gated by **R1** =
correlation must be load-bearing (ablation gap on held-out data); Stage B (standard
mix, synth-rot 0.1) gated by **R2** = flow360 active₀.₅ improvement must beat +5.2%
(the best transient any Act-I model ever touched) with the R1 ablation gap persisting
on flow360. AdamW + OneCycle, AMP, batch 2, seed 7, budgets 5k–10k steps
(82–190 min/run on the single-GPU box).

**Honesty box.** Single seed; smoke-scale budgets; all numbers are validation, no
test split touched; eval-time correlation ablation of a correlation-trained model is
out-of-distribution and *overstates* the correlation contribution (the fair
prior-ceiling control is a from-scratch `--ablate-corr` training — **(pending)**,
§9.4). Metric-infrastructure note: HEALPix places node columns exactly on the seam
and 30°-latitude mask boundaries, making fp32 region masks device-dependent (a
one-ulp CPU/CUDA atan2 difference flips 44 r6 seam-edge nodes at once, ±0.026° on
seam means); found while cross-validating the RAFT baseline's oracle mode and fixed
2026-07-07 (fp64 + 1e-12 tolerance in `build_region_masks`). Seam numbers quoted
before the fix (Stage B/B′ logs) differ from the fixed convention by ≤0.03°; the
§5/§5.1 tables use the fixed masks.

---

## 5. Stage A and Gate R1: spherical correlation works

Training (5,000 steps, 82 min): aux-only warmup drops the matching loss 2.56 → 1.65;
in the joint phase it settles at its measured entropy floor (~1.3–1.6) — features at
their theoretical matchability optimum — while validation improves monotonically
(58 → 78 → 87 → 91%).

| eval (Stage A ckpt) | global err | zero baseline | improvement |
| --- | --- | --- | --- |
| mixed val (real + synth-rot 0.5) | 0.850° | 9.62° | **+91.2%** |
| same, `--ablate-corr` at eval | 27.54° | 9.62° | −186% |
| **real pairs only** | **1.56°** | **13.53°** | **+88.4%** |
| real only, `--ablate-corr` at eval | 28.38° | 13.53° | −110% |
| from-scratch `--ablate-corr` **control**, mixed val | 9.63° | 9.62° | **−0.1%** |

Regions (real only): equator 1.21° (+90.9%), poles 2.88° (+79.4%), seam 4.40°
(+74.1%) — a real but bounded pole/seam penalty. (Seam values here and in §5.1 use
the fp-robust region masks; see the honesty box.)

**Gate R1: PASSED.** The +88.4% on *real* rendered motion is the headline positive
result: it cannot be an appearance prior (Act I bounded that at +2.9%), and the
ablation collapse shows the estimate rides on correlation. The **from-scratch
control** (identical recipe, `--ablate-corr` from step 0) settles the eval-time
ablation caveat (§4): after 5k steps it converges to zero-flow parity, −0.1% at every
region. Two consequences: (i) the fair no-correlation baseline is not the
catastrophic 27.5° (an OOD artifact of eval-time ablation) but "exactly nothing" —
100% of Stage A's improvement flows through correlation; (ii) the control lands
*below* Act I's +2.9% prior ceiling, which resolves the prediction in §4: that
ceiling was measured on sub-pixel flow360, where near-zero predictions are almost
right and an appearance prior can shave the residual; on replica's p50 ≈ 12° motion a
frame-1-only model has no information about displacement and its best strategy is
zero flow — which is precisely what it learns.

**Seed robustness.** Stage A was retrained from scratch with seeds 11 and 19
(identical recipe; the run above is seed 7). All three runs share the training
signature (aux loss at its entropy floor, monotone mixed-val improvement, ~82 min),
and all three checkpoints are evaluated on the *identical* real-pairs val stream
(seed-7 val order, post-fix masks — the zero-flow columns agree to 6 decimals):

| real pairs, r6 | seed 7 | seed 11 | seed 19 | mean ± std (n = 3) |
| --- | --- | --- | --- | --- |
| global | 1.564° | 2.029° | 1.621° | **1.74° ± 0.25 (+87.1 ± 1.9%)** |
| equator | 1.212° | 1.581° | 1.301° | 1.36° ± 0.19 (+89.7 ± 1.5%) |
| poles | 2.876° | 3.740° | 2.768° | 3.13° ± 0.53 (+77.6 ± 3.8%) |
| seam | 4.400° | 4.940° | 4.205° | 4.52° ± 0.38 (+73.4 ± 2.2%) |
| poles/equator ratio | 2.37× | 2.37× | 2.13× | 2.13–2.37× |

Gate R1's conclusion is seed-stable: every seed clears +85% global on real pairs.
Seed 11 is a mild negative outlier (worst at every region under the same recipe —
ordinary training variance at this budget, and the reason the headline is reported
as a mean with spread rather than the single best run).

### 5.1 Head-to-head against the published-pipeline proxy (frozen ERP-RAFT)

The published 360° pipelines wrap an unchanged pixel-raster RAFT on the ERP
projection; `run_raft_shard_baseline.py` evaluates exactly that — TorchVision
RAFT-large (C_T_SKHT_V2 weights, zero-shot) on the shards' ERP frames — through the
*identical* spherical metric pipeline (validated: an oracle predictor scores 0.028°
and reproduces every GT column of the OSLO eval bit-compatibly). Same 162 replica360
val pairs, same r6 grid, same masks:

| replica360:val, r6 | frozen RAFT-large (ERP) | OSLO-RAFT-R (Stage A, seed 7) | zero-flow |
| --- | --- | --- | --- |
| params / training | 5.3M, Chairs→Things→Sintel/KITTI/HD1K, zero-shot | 1.56M, from scratch, 5k steps (82 min) | — |
| global | **1.158° (+91.4%)** | 1.564° (+88.4%) | 13.53° |
| equator | **0.681° (+94.9%)** | 1.212° (+90.9%) | 13.28° |
| poles | 3.649° (+73.9%) | **2.876° (+79.4%)** | 13.96° |
| seam | **4.012° (+76.4%)** | 4.400° (+74.1%) | 16.99° |
| poles / equator error ratio | 5.36× | **2.37×** | — |
| seam / equator error ratio | 5.89× | **3.63×** | — |

Three readings, in decreasing order of confidence:

1. **The native model wins the poles** — 21% lower error than a 3.4× larger,
   massively pretrained model, on the region where the ERP projection is most
   distorted. The poles row is the cleanest comparison in the table: its zero-flow
   baseline (13.96°) matches the global one (13.53°), so unlike the seam it is not
   confounded by harder motion — the difference is projection handling.
2. **Native processing buys spatial uniformity.** RAFT's error grows 5.4–5.9× from
   equator to poles/seam; OSLO-RAFT-R's grows 2.4–3.6×. The ERP wrapper pays a
   projection tax exactly where the sphere stops looking like a plane.
3. **RAFT wins globally (1.35×)** — expected given the capacity and pretraining gap,
   and stated as-is. The comparison is asymmetric in both directions (RAFT never saw
   this domain; OSLO is 3.4× smaller with 82 minutes of training), so the defensible
   claim is not "beats RAFT" but: *a small, from-scratch native-spherical model is in
   the same league as the standard pipeline and better in the polar caps.* The seam
   row is a near-tie and conflates projection artifact with harder content (its
   zero baseline is 16.99° vs 13.53° global); the poles row is the claim-bearing one.

Seed robustness of the comparison (three-seed table in §5): the poles column reads
2.88°/3.74°/2.77° across seeds — two of three individually beat RAFT's 3.649°, the
seed mean (3.13°) beats it by 14%, and the worst seed is a near-tie (2.5% behind).
The uniformity reading is seed-invariant: every seed's poles/equator ratio
(2.13–2.37×) is under half of RAFT's 5.36×. So reading 2 holds unconditionally;
reading 1 holds in the mean and for the typical seed.

#### 5.1.1 Transfer leg: mpf:val, zero-shot for both models

Same protocol on MPF's city-driving scenes (2,211 pairs, motion p50 4.2° — between
flow360 and replica; zero-flow baseline nearly flat across regions, 4.56–4.63°):

| mpf:val, r6 | frozen RAFT-large (ERP) | OSLO-RAFT-R (Stage A, seed 7) | zero-flow |
| --- | --- | --- | --- |
| global | **3.650° (+21.2%)** | 4.191° (+9.6%) | 4.634° |
| equator | **3.593° (+22.4%)** | 4.096° (+11.5%) | 4.630° |
| poles | **3.791° (+18.0%)** | 4.538° (+1.8%) | 4.624° |
| seam | **3.541° (+22.4%)** | 4.045° (+11.3%) | 4.561° |

Stated straight: **RAFT leads every region here, including the poles — the §5.1
poles win does not transfer zero-shot to this domain at this training budget.** Both
models degrade sharply out of domain (+21% and +10% against +91% and +88% on
replica), and the comparison is asymmetric in a new direction: MPF is outdoor
driving footage, and RAFT's fine-tune includes KITTI (driving) — for content, mpf is
*near RAFT's training distribution* and maximally far from OSLO's (indoor Replica
rooms, 5k steps, one dataset). Two structural observations survive the loss: (i)
RAFT shows no polar tax on mpf at all (poles/equator 1.06×) — consistent with mpf's
polar content being sky/road with little texture or motion, so mpf's poles row does
not probe projection distortion the way replica's does; (ii) OSLO's regional profile
stays flat too (1.11×) — the uniformity property holds off-domain even when absolute
accuracy drops. The honest summary for the chapter: the poles advantage is
demonstrated in-domain; whether it survives domain shift is exactly what the
Phase-2 multi-dataset training (P2C) is designed to test.

## 6. Stage B/B′ and Gate R2: the FLOW360 gate fails twice

**Stage B** (init from A; mix replica360+mpf+flow360; synth-rot 0.1; 5k steps): on
real flow360 val, active₀.₅ = **−2.4%** final, never positive at any of 5 evals (best
−1.0%); global −40%. The model carries an output noise floor ≈ 0.3° — larger than the
median target. Confound identified: the loss is an unweighted mean, and flow360's
~0.1° targets contribute ≈1% of the gradient against replica's ~13° motion.

**Stage B′** (init from B; **flow360 only**; motion-weighted loss
`--loss-motion-weight 1.0 --loss-motion-ref-deg 1.0`; small-angle exact-GT curriculum
`--synth-rot-prob 0.5`, 0.1–2°; 10k steps, lr 1e-4): the confound is removed — and
the gate still fails. Active₀.₅ trajectory −7.8 → −2.9%, monotone-ish but asymptoting
to a zero-flow tie, nowhere near +5.2%. Calibration improves substantially (global
−78% → −32%, poles −214% → −60%), but the error floor settles at 0.278° vs the 0.211°
zero baseline: **after 10k steps in which the loss rewards predicting zero on the 76%
of near-static nodes, the model still cannot tell "moved 0.1°" from "didn't move" in
real imagery.**

Honest comparison row: on real FLOW360 the retina model (−2.9% active₀.₅) is below
both zero-flow and the Act I prior (+2.7% active₀.₅). Trained under a matching loss
and a motion-weighted objective, it makes confident small predictions instead of
retreating to the prior — and at this motion scale, on real pairs, confident is
wrong. **Gate R2: FAILED** (twice, with the training-signal explanation removed the
second time).

**What B′ paid for the attempt (retention eval).** The B′ checkpoint back on
replica360:val real pairs: 13.55° global vs the 13.53° zero baseline — **−0.1%**,
against Stage A's +88.4% from the same lineage. The flow360-only fine-tune did not
partially degrade the large-motion capability; it erased it completely, collapsing
the model to a near-zero-flow predictor everywhere. This is catastrophic forgetting
in its textbook form, and it sharpens §6's reading: B′ didn't fail to *learn* — it
successfully specialized into a domain where the optimal real-pair behavior it could
find was "predict almost nothing," and that policy overwrote the one that worked.
Any Phase-2 schedule must therefore mix domains (or replay) rather than fine-tune
sequentially.

## 7. The isolation probes: naming the wall

### 7.1 Resolving power (probe on the Stage B checkpoint)

Zero-shot eval on flow360 val frames with exact-GT synthetic rotations of 0.1–0.5°
(displacement p50 0.22° ≈ 0.5 retina spacings ≈ 0.6 ERP px): **+32.3% global,
+44.0% active₀.₂₅**, positive in every region. A model trained at 1–15° rotations and
real motion generalizes *down* to sub-pixel scales — the "sub-node motion is below
the grid's resolving power" hypothesis (Act I's root cause) is disproved for this
architecture.

### 7.2 The triangle (probe on the Stage B′ checkpoint)

One checkpoint, one frame source (flow360 val), one motion scale, one variable:

| frame 2 is… | global | active₀.₂₅ | residual error |
| --- | --- | --- | --- |
| frame 1 resampled at the true motion (0.1–0.5°) | **+80.6%** | **+86.1%** | **0.046°** ≈ 0.13 ERP px ≈ 0.10 retina spacing |
| the real rendered next frame | −32.2% | −4.2% | 0.278° |

When brightness constancy holds exactly, this 1.5M-parameter model tracks motion to a
*tenth of a retina node* — an order of magnitude below its input grid, and ~7× below
the ERP source pixel. On real pairs of the same scenes at the same scale, it cannot
beat doing nothing. (B′'s curriculum trained on this synthetic family, so 7.1 vs 7.2
also shows the family is learnable to near-saturation — +32% → +81% — while the real
objective, trained simultaneously and motion-weighted, stayed negative.)

### 7.3 The elimination chain

| hypothesis for the FLOW360 failure | experiment | verdict |
| --- | --- | --- |
| spherical correlation can't work on HEALPix | Act II: CPU gate; Stage A + R1 (+88.4% real, ablation collapse) | **eliminated** |
| sub-node/sub-pixel motion below resolving power | §7.1 probe: +32% zero-shot; §7.2: 0.046° residual | **eliminated** |
| flow360 starved of training signal in the mix | Stage B′: flow360-only, motion-weighted, curriculum → still −2.9% | **eliminated** |
| GT quality / metric artifact | same GT & metric score +80.6% under resampled frame 2 | **eliminated** |
| real inter-frame appearance change swamps sub-pixel correspondence | §7.2 triangle: only frame 2's image-formation differs; 0.28° static/moving indistinguishability floor (§6) | **standing, by direct demonstration** |

Corr-gap footnote: even in the failing regime, correlation carries what output there
is. Eval-ablating correlation on the B′ checkpoint (flow360:val real) degrades the
error floor 10.8× — 0.278° → 3.00° global (−1326%) — so the −2.9% verdict is a
property of *correlation-driven* estimation hitting the appearance wall, not of the
model having quietly fallen back to its context path.

### 7.4 What "appearance change" contains, and why the wall is not absolute

The resample-vs-real contrast bundles: photometric inconsistency (shading, exposure,
anti-aliasing of two independent renders), occlusion/disocclusion, and the
non-rigidity of real motion (depth parallax, discontinuities) versus a global
rotation field. This chapter does not decompose the bundle further. Two boundary
markers: (i) the smooth-field advantage is real — the GRU can aggregate a rigid
rotation globally — so §7.2 bounds resolving power, not scene difficulty; (ii)
frozen perspective-RAFT beats zero-flow by +44% on FLOW360's test active₀.₅ subset,
so the correspondence signal *exists* in real pairs at that scale and is extractable
by a 5.3M-parameter model pretrained on large corpora with heavy photometric
augmentation. The wall is therefore best stated as: **within this model scale and
training budget, sub-pixel correspondence on real 360° imagery was not learnable,
while the same correspondence under exact brightness constancy was learned to 0.046°
— robustness to real image formation is the missing ingredient, and it is a data/
robustness problem, no longer an architecture problem.**

## 8. Conclusions

**Contributions.**
1. A HEALPix-native RAFT in which spherical correlation is *demonstrably
   load-bearing* — to our knowledge the first, achieved by decoupling the retina from
   the estimation grid, an interpolation-continuous lookup, and directly supervised
   matching features; each design element carries measured necessity evidence.
2. A positive, *comparative* result: +88.4% over zero-flow on real Replica360 motion
   (1.56° mean geodesic error; three-seed mean 1.74° ± 0.25, +87.1 ± 1.9%), and a
   head-to-head against the published-pipeline proxy (frozen ERP RAFT-large) in
   which the 3.4×-smaller native model wins the polar caps (seed mean 3.13° vs
   3.65°) and shows less than half the equator-to-pole degradation on every seed
   (2.13–2.37× vs 5.36×) (§5.1).
3. Demonstrated sub-pixel resolving power on the sphere: 0.046° residual (0.13 ERP
   px) on exact-correspondence pairs.
4. A three-act, five-hypothesis elimination chain that revises Act I's "sub-node
   motion wall" into a sharper claim: the wall is robustness of sub-pixel
   correspondence to real inter-frame appearance change.

**Limitations.** Single seed for Stages B/B′ and the probes (Stage A: three seeds,
§5); 5–10k-step budgets; validation-only numbers; the
active₀.₅ Act-I/retina comparison crosses a 512-pair-subset vs full-val boundary;
the appearance-change bundle (§7.4) is not decomposed; the poles win is
demonstrated in-domain only — it does not transfer zero-shot to mpf (§5.1.1), where
RAFT (KITTI-fine-tuned, i.e. near-domain for driving footage) leads every region.

**Future work** (each attacks exactly the isolated variable): photometric and
appearance augmentation on real pairs; occlusion-aware matching; multi-frame
aggregation (temporal redundancy against per-pair appearance noise); model scale +
perspective-pretraining transfer onto the retina architecture; a decomposition study
inserting occlusion/shading separately into the synthetic-rotation source.

---

## 9. Reproducibility

All runs from the repo root on the GPU box via
`docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft python run_oslo_raft.py …`
with `SHARDS_HOST` pointing at the shards. Git hash for every result above:
`7de09fecb55d4179fc1c94fcf26bef075b0fa2f4`. Metrics JSONs live next to each
checkpoint under `/outputs/`.

### 9.1 Stage A (+ Gate R1 evals)

- train: `--grid healpix --retina --retina-resolution 7 --resolution 6
  --estimation-resolution 4 --device cuda --amp --onecycle --steps 5000
  --batch-size 2 --so3-prob 1.0 --train-sources replica360:train
  --val-sources replica360:val --synth-rot-prob 0.5 --synth-rot-min-deg 1
  --synth-rot-max-deg 15 --val-synth-rot-prob 0.5 --aux-match-weight 0.5
  --aux-warmup-steps 500 --output-dir /outputs/oslo_raft_retina_stageA`
- R1 evals: `--eval-only --init-checkpoint …stageA/oslo_raft.pt` with
  `--val-synth-rot-prob 0.5` (mixed) / `0` (real), ± `--ablate-corr`.
- seed spread: same train command with `--seed 11` / `--seed 19`
  (`…stageA_seed11`, `…stageA_seed19`), each followed by the real-only eval above
  (default seed 7 at eval so all three share the val stream).

### 9.2 Stage B and B′ (+ Gate R2)

- B: init from A; `--train-sources replica360:train,mpf:train,flow360:train
  --val-sources flow360:val --synth-rot-prob 0.1 --val-synth-rot-prob 0
  --lr 2e-4 --aux-warmup-steps 0 --steps 5000`.
- B′: init from B; `--train-sources flow360:train --loss-motion-weight 1.0
  --loss-motion-ref-deg 1.0 --synth-rot-prob 0.5 --synth-rot-min-deg 0.1
  --synth-rot-max-deg 2 --val-synth-rot-prob 0 --lr 1e-4 --steps 10000`.

### 9.3 Probes (§7)

Eval-only on flow360:val with `--val-synth-rot-prob 1.0 --synth-rot-min-deg 0.1
--synth-rot-max-deg 0.5`, init from the B (§7.1) / B′ (§7.2) checkpoint.

### 9.3b Head-to-head (§5.1)

- RAFT column: `python run_raft_shard_baseline.py --shards /data/shards
  --sources replica360:val --resolution 6 --device cuda --batch-size 2`
  (TorchVision `raft_large`, `C_T_SKHT_V2`, `--flow-transform identity`;
  set `TORCH_HOME=/outputs/torch_home` to cache the weights). Validation modes:
  `--predictor zero` must land on the `*_zero_geo_deg` columns exactly;
  `--predictor oracle` (GT as prediction) scores 0.028° and reproduces the OSLO
  eval's GT columns bit-compatibly.
- OSLO column: the §9.1 real-only eval re-run under the fixed region masks
  (`/outputs/oslo_raft_retina_stageA_R1_real_full_v2`).

### 9.4 Bookkeeping runs (filled 2026-07-09)

| run | purpose | result |
| --- | --- | --- |
| B′ ckpt, flow360:val real, `--ablate-corr` | R2 corr-gap record | 3.00° vs 0.278° unablated (10.8×) → §7.3 footnote |
| B′ ckpt, replica360:val real | retention after flow360-only fine-tune | 13.55° = −0.1%, total forgetting → §6 |
| Stage A from scratch, `--ablate-corr` | fair no-correlation control | mixed val 9.63° = −0.1% (zero-flow parity) → §5 table |
| mpf:val head-to-head leg (RAFT + OSLO zero-shot) | transfer of the poles win | RAFT leads all regions → §5.1.1 |
| ↳ real-only eval of the ablated control | completes the §5 control row on real pairs | **pending** (expected ≈ 13.53°, ~0%) |
