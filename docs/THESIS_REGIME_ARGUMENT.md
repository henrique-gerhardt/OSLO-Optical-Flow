# The Two-Regime Argument: why the thesis targets real-video 360° flow

Research summary for thesis writing (2026-07-24). This note consolidates the
framing argument developed at the close of the P2C campaign: what regime the
thesis actually studies, why the field's benchmarks do not contain it, why
flow360 is the measuring instrument rather than the problem, and what the
correct deliverables and honest caveats are. Numbers cite campaign results
recorded in `docs/plans/P2C_TRAINING_CAMPAIGN.md`.

## 1. 360° optical flow has two operating regimes, not one

**The large-displacement regime.** Motion of tens of pixels everywhere on the
sphere. This is the regime every published 360° flow benchmark was constructed
to probe: FlowScape/Flow360 (PanoFlow) is CARLA driving footage — fast
ego-motion, wide baselines (p90 motion 70–93 px in our resharded copy);
MPFDataset and EquirectFlyingThings-style sets are synthetic pairs built to
have large motion by construction; our own chairs360 (p90 13.6°) and
replica360 follow the same design. The construction is inherited from
perspective flow (FlyingChairs/Things → Sintel/KITTI), where classical
matching difficulty — large search ranges, occlusion — is the object of study.

**The real-video regime.** Consecutive frames of 360° video at ordinary frame
rates (30–60 fps). The frame-rate arithmetic makes its character universal:
anything not both fast and close to the camera moves sub-pixel between frames.
On FLOW360 (Bhandari et al., ECCV 2022 — naturalistic rendered video, the only
GT-labelled instance of this regime we are aware of): median motion 0.097°
(≈0.28 px at 1024×512 ERP), ~76% of the sphere below 0.25°, ~88% below 0.5°
(only 12.4% of nodes above), 1.7% above 1°. The field is mostly-static with sparse,
spatially-structured movers (parallax, articulated motion), under genuine
inter-frame appearance change.

These are different problems wearing the same name. The thesis studies the
second, and the central empirical claim is that the second is (a) where real
captured 360° video actually lives, and (b) unsolved — by anyone.

## 2. The difficulty is the regime, not the flow360 dataset

Three campaign results pin the difficulty to the regime itself and rule out
"flow360 is just a hard/badly-labelled dataset":

1. **Published-lineage methods lose to doing nothing.** Frozen RAFT-large
   (5.3M params) scores −20.5% EPE against the zero-flow baseline on
   flow360:val. Zero-flow is not a strawman in this regime; it is the
   strongest baseline measured, because the first job on a mostly-static
   sphere is to not hallucinate motion.
2. **The decisive triangle.** The same checkpoint on the same frames scores
   +80.6% when frame 2 is resampled from GT (0.046°) and −32% with the real
   frame 2. The labels are fine and the signal is learnable; the wall is
   exactly the real inter-frame component at sub-pixel motion.
3. **The P0 decomposition.** Controlled swaps show the dominant term is the
   *motion-field structure* (real GT field + clean appearance: −72.5%; the
   field swap costs 4× the appearance swap), with structured, edge-anchored
   appearance change a bounded secondary term (~1/3 of the wall at matched
   magnitude; photometric *magnitude* and iid noise are ruled out by measured
   nuisance curves). Mostly-static + sparse parallax is not aggregatable the
   way benchmark fields (global rotations, dense synthetic motion) are.

Corollary for writing: flow360 should be presented as the *instrument* that
samples the regime, not as a benchmark being chased. Any deployment on
captured video sits in this regime — and captured footage is likely harsher
than FLOW360, which is naturalistic-synthetic (renderer-originated appearance
change; no rolling shutter, stitching seams, or sensor noise).

## 3. Application relevance: which regime do applications live in?

- **Real-video regime** (this thesis): telepresence and monitoring (moving-
  object detection), VR capture, 360° video stabilization, frame
  interpolation and compression, temporal consistency in editing — anything
  that consumes consecutive frames of a real camera at normal frame rates.
- **Large-displacement regime** (existing benchmarks): autonomous platforms
  (driving, drones) with fast ego-motion, and any pipeline processing at
  keyframe stride rather than frame rate.

Neither regime is "wrong"; the point for the motivation chapter is that the
first family is served by no current benchmark or method, while the second
has a saturated leaderboard culture. The thesis targets the underserved
regime that ordinary 360° cameras occupy the moment they record.

## 4. The regime dictates the deliverables (and the metrics)

Global EPE is application-irrelevant in the real-video regime: zero-flow is
near-unbeatable on the static ~88%, so global averages measure calibration
noise, not usefulness. The deliverables that applications actually consume
are:

1. **Detect and measure the sparse genuine movers** — our active-subset
   metrics (improvement over zero-flow restricted to nodes with GT motion
   above 0.25°/0.5°/1°). This is where the campaign's headline lives:
   **+4.5 ± 0.9% consolidated on active₀.₅** (EMA-measured basin center;
   first consolidated positive real actives; the Act-I appearance-prior
   ceiling of +2.9% falls at act₀.₂₅ +3.5 consolidated), with correlation
   proven load-bearing (ablation collapses 0.276° → 54.0°: genuine matching,
   not an appearance prior).
2. **Confidently report zero on the static majority** — static-confidence
   calibration. Our global −16.8% (best ever measured in the campaign, still
   negative) quantifies that current models, ours included, fail this;
   it is the identified engineering frontier, not a footnote.

A method delivering both converts directly into the applications of §3 in a
way a FlowScape leaderboard position does not.

## 5. Honest caveats to carry into the text

- **Large-motion capability still matters** even for video applications:
  keyframe-stride processing, fast near-field movers, and composing flow over
  frames (where sub-pixel errors accumulate — itself an argument for
  sub-pixel accuracy). This motivates the *mix* recipe over specialization;
  quantified by the retention stamp: the final model keeps +15.5% on
  out-of-mix replica360 where sequential fine-tuning retained −0.1%.
- **The gate was approached, not met** (+4.5 vs +5.2, 86%), and the miss is
  precisely factored: variance solved by EMA (9×); the level shown to be
  noise-sustained (reducing gradient noise degrades it monotonically ⇒ the
  training objective's minimum ≠ the gate metric); remaining levers are data
  scale and the calibration objective of §4.2.
- **Positioning against published methods** uses comparisons aligned with the
  claims: the universality table (frozen RAFT-large done; PriOr-RAFT,
  PanoFlow(CSFlow), SLOF public weights available) on the flow360 real leg —
  SLOF is the load-bearing row, being FLOW360's in-domain home method; the
  SO(3) rotation protocol, which ERP methods structurally cannot win; and
  honest placement of OSLO on flowscape/mpf against published numbers, with
  the P2A grid-floor analysis stating what a 3,072-node estimation grid can
  resolve. A raw EPE shootout on large-motion benchmarks is reported for
  honesty but measures capacity and training scale, not the contribution.

## 6. One-paragraph thesis-ready statement

> Published 360° optical-flow benchmarks probe a large-displacement regime
> inherited from perspective flow; real 360° video at ordinary frame rates
> occupies a different one — a mostly-static sphere with sparse sub-pixel
> movers under genuine inter-frame appearance change — in which the strongest
> published baseline is surpassed by predicting no motion at all. This thesis
> characterizes that regime with controlled decompositions (locating the
> difficulty in the structure of the real motion field, not photometric
> nuisance), shows the field is learnable, and reports the first consolidated
> positive result over the zero-flow baseline on the moving subset of real
> 360° video (+4.5% active₀.₅, matching proven load-bearing by ablation),
> while identifying static-confidence calibration and data scale as the
> remaining, precisely-measured obstacles.
