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
13.53° zero-flow baseline (+88.4%)**, and ablating correlation collapses it to 27.5°
— correlation is finally, demonstrably load-bearing (**Gate R1 passed**).

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
§9.4).

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

Regions (real only): equator 1.21° (+90.9%), poles 2.88° (+79.4%), seam 4.39°
(+74.1%) — a real but bounded pole/seam penalty.

**Gate R1: PASSED.** The +88.4% on *real* rendered motion is the headline positive
result: it cannot be an appearance prior (Act I bounded that at +2.9%), and the
ablation collapse shows the estimate rides on correlation. The eval-time-ablation
caveat (§4) applies to the 27.5°/28.4° rows — the from-scratch control **(pending)**
will replace them as the fair no-correlation baseline; Act I's result predicts it
lands near the prior ceiling.

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
2. A positive result: +88.4% over zero-flow on real Replica360 motion (1.56° mean
   geodesic error), with pole/seam behavior quantified.
3. Demonstrated sub-pixel resolving power on the sphere: 0.046° residual (0.13 ERP
   px) on exact-correspondence pairs.
4. A three-act, five-hypothesis elimination chain that revises Act I's "sub-node
   motion wall" into a sharper claim: the wall is robustness of sub-pixel
   correspondence to real inter-frame appearance change.

**Limitations.** Single seed; 5–10k-step budgets; validation-only numbers; the
active₀.₅ Act-I/retina comparison crosses a 512-pair-subset vs full-val boundary;
eval-time ablation overstates (from-scratch control pending); the appearance-change
bundle (§7.4) is not decomposed.

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

### 9.4 Pending bookkeeping runs (fill these cells)

| run | purpose | fills |
| --- | --- | --- |
| B′ ckpt, flow360:val real, `--ablate-corr` | R2 corr-gap record | §7.3 footnote |
| B′ ckpt, replica360:val real | retention after flow360-only fine-tune | §6 |
| Stage A retrained from scratch with `--ablate-corr` (+ real-only eval of it) | fair prior-ceiling control | §5 table |
