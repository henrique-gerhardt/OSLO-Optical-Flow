# Decision-first flow: the zero-flow baseline and the static/motion gate

**Status: REGISTERED PROPOSAL, NOT EXECUTED (2026-07-31).** Nothing here has been
run. This document exists so the idea, its evidence base, its precedent in a
neighbouring field, and its falsifiers are dated and versioned *before* any number
exists. The active campaign continues to be making OSLO-RAFT competitive
(`ROADMAP_SEMINARIO.md` §8, `UNIVERSALITY_TABLE.md` §14.6).

---

## 1. The measurement that motivates this

Every number below is already in the repository.

**The trivial predictor is unbeaten in the sub-pixel regime.** On `flow360:test`,
no published 360° flow method beats zero-flow on the active subsets: SLOF,
PanoFlow/CSFlow and OSLO-RAFT all land negative (`UNIVERSALITY_TABLE.md` §10).
Three architectures, three labs, on the benchmark those labs published on.

**The failure belongs to the regime, not to the models.** The same weights, with
only the dataset changing, swing 80 to 103 percentage points: PanoFlow +92.6 →
−10.1, RAFT-large +74.4 → −14.8, OSLO +66.0 → −14.0. This kills the "your methods
are weak" reading.

**The error mass sits on the static majority.** On FLOW360 the band below
0.0625° holds **42.8% of nodes**, whose true displacement is 0.0103° (0.029 ERP
pixel). OSLO-RAFT's error on those same nodes is 0.151° (0.43 pixel) — a **15×
over-claim across nearly half the sphere**.

**The final model is already positive where motion exists.** `P1proper_ema6k`
consolidates at `act₀.₂₅` **+3.5%** and `act₀.₅` **+4.5 ± 0.9%**, while global sits
at **−16.8%**. The model wins on movers and loses on the static majority, which is
a calibration failure, not an estimation failure.

**The cause is field structure, not appearance or magnitude.** The P0d
decomposition puts the motion-structure swap at 3.84× the appearance swap, and A3
shows the real-field leg *degrading* monotonically (−32.1 → −70.4%) as magnitude
grows while the coherent-rotation leg climbs (+1.7 → +84.1%).

**The static error is spatial leakage.** In the rotation experiment, nodes near the
axis have constant true displacement, yet model error grows 2.4× as the rest of the
sphere accelerates. Motion bleeds in from neighbours through the recurrent
aggregation.

---

## 2. The zero-flow baseline, and where it comes from

**The baseline itself is not a contribution.** Predicting zero everywhere is
trivial and anyone can compute it. Claiming it as an invention would be weak and
correctly attacked.

**Its role is to legitimise the question.** The identical failure mode is already
documented in **video prediction** (next-frame prediction), where the "copy the last
frame" baseline is notoriously strong. The published reasoning matches ours exactly:
copying the last frame gets the background perfectly right, so on videos where the
foreground occupies a small fraction of the image the trivial baseline scores high
regardless of whether the learned model is actually better
([Villegas et al., *Decomposing Motion and Content for Natural Video Sequence
Prediction*](https://arxiv.org/pdf/1706.08033);
[Szeto et al., *A Temporally-Aware Interpolation Network for Video Frame
Inpainting*](https://arxiv.org/pdf/1803.07218)). That community absorbed the lesson
and now reports the baseline.

A second precedent exists in **activity progress prediction**, where learned methods
are exceeded by non-learned frame-counting baselines, which put the reported
benchmark progress in doubt ([Is there progress in activity progress
prediction?](https://arxiv.org/pdf/2308.05533)).

**The lesson never crossed into optical flow, and least of all into 360° flow.**
PanoFlow, SLOF and OmniFlowNet report EPE and percentage reduction against the
previous best — PanoFlow, for instance, reports 27.3% EPE reduction on FLOW360 and
55.5% on OmniFlowNet — and never against a trivial predictor
([PanoFlow](https://arxiv.org/pdf/2202.13388),
[SLOF](https://link.springer.com/chapter/10.1007/978-3-031-20074-8_32),
[Revisiting Optical Flow Estimation in 360
Videos](https://www.semanticscholar.org/paper/Revisiting-Optical-Flow-Estimation-in-360-Videos-Bhandari-Zong/3fe75c5f79a6a37d3cc9082ce74048cc48128fc4)).

This is *better* than pure novelty: because the argument is already legitimate in a
neighbouring field, a committee can verify the precedent and will not treat it as an
eccentricity. What is new is the application and the measurement.

---

## 3. What is actually new here, in layers

1. **The measurement.** Nobody had the number for 360° flow.
2. **The scope of the failure.** Three architectures from three labs, not one weak
   method, which turns a per-method result into a property of the regime.
3. **The regime characterisation.** The 80-to-103-point swing answers *when* the
   trivial baseline wins and *why*. The video-prediction critique never needed this:
   it sufficed there to note the baseline is strong on low-motion data.
4. **The causal isolation.** P0d attributes the failure to field *structure*, not
   magnitude and not appearance. That level of diagnosis does not exist in the
   neighbouring critique.
5. **The architectural response.** See §4 — this is the step beyond.

### The step that matters: evaluation fix vs method fix

When video prediction recognised the problem, the field's response was an
**evaluation fix**: report the trivial baseline alongside. That corrected the table,
not the model.

The response proposed here is a **method fix**: an estimator whose output space
contains exact zero as an explicit, supervised decision. With the gate closed the
output is exactly the zero field, so the model **contains the baseline it must
beat** and can only lose to it by deciding wrongly.

**The mechanism is not new and must not be claimed as such** — video codecs have
done exactly this since the 1990s under the name SKIP mode, and classical flow
abstained via confidence thresholds. See §8.1, which rewrites the claim: what is new
is the diagnosis that **the flow objective contains no price for abstaining**, so a
dense network cannot learn it, plus the measurement of what that costs.

---

## 4. Design

**A gate is a head, not an architecture.** It attaches to whatever produces the
flow, which makes the natural experiment much stronger than a single-model demo.

**Apply it to all three predictors.** Demonstrating the gate only on OSLO-RAFT
invites the reading "a crutch for a weak model". Demonstrating that the same gate
also corrects PanoFlow — the state of the art — makes the defect a property of how
the field builds flow estimators, and the fix a general contribution. The
competitors' own strong models become evidence for the claim.

**Post-hoc form, for feasibility.** A small network taking (frame 1, frame 2,
predicted field) and emitting a per-node keep-or-zero decision, trained with the
predictor frozen. Works for any estimator, and the repository already vendors the
PanoFlow and RAFT-large weights.

**Native form, for OSLO-RAFT.** The gate as part of the architecture rather than
bolted on, which is where the spherical representation lets it operate uniformly
over the sphere.

**Supervision.** Per-node binary classification on `|GT| > τ`, with τ = 0.25°. The
label is free: the ground-truth field is already sampled at the supervision nodes.
This is a far easier learning problem than the regression it guards.

**Pairs naturally with a structure-matched decomposition.** P0d says the field is
mostly-static plus sparse parallax, and B′ reached 0.044° on coherent rotation
fields. That argues for a coherent global-motion head plus a sparse residual that is
only evaluated where the gate opens. Each piece is motivated by a measurement rather
than by intuition. Not required for the first probe.

---

## 5. The arithmetic of the payoff

With a perfect gate zeroing every node below 0.25°, the static nodes inherit the
zero-flow error and the aggregate becomes

```
err_new − err_zero = f_a · (err_active − zero_active)
```

Using the measured `act₀.₂₅` frac 0.3136, `zero_active` 0.9226°, `zero_global`
0.3400° and the model's `act₀.₂₅` +3.5%:

```
0.3136 × 0.035 × 0.9226 / 0.3400 = +3.0%
```

**Global moves from −16.8% to roughly +3%**, which by §10 of
`UNIVERSALITY_TABLE.md` would make OSLO-RAFT the first 360° flow method to beat the
trivial baseline in the sub-pixel regime.

**Do not headline the +3%.** It is the wrong number to sell: a 3% aggregate gain
reads as marginal. Two better framings, both true:

- It is a **sign flip on a universal failure**, not a 3% gain. Every published
  method sits on the wrong side of that line.
- On the 43% of the sphere that is static, the over-claim drops from 0.151° to
  ~0.0103°, a **~93% error reduction on those nodes**. That is the number that maps
  to the visible artefact.

The honest position is that neither is the selling number. The selling number is
downstream task quality, and this project has not measured it yet (§7).

---

## 6. First probe, and what kills the idea

**Probe: is the static/moving decision learnable at all?** Train only the gate as
per-node binary classification. One day of work, before building anything.

**Pre-registered falsifier:** the gate must beat the trivial "everything is static"
classifier on F1 at τ = 0.25°. If it does not, the direction ends there.

**This risk is concrete and has precedent inside this project.** The differential
head's confidence came out *anti-correlated* with motion — more wrong exactly where
features were sharpest, with no gating fraction beating zero
(`oslo-raft-shakeout-r4`, `analyze_differential.py`). A confidence-like signal has
already failed here once. Better to know in a day than in a trimester.

**Second falsifier, if the gate trains but does not pay:** the end-to-end swap must
show the gated model beating the ungated one on `flow360:val` global *and* not
losing on the active subsets. A gate that buys global by suppressing real movers is
not a contribution.

---

## 7. Where to verify, beyond the metric

Ordered cheapest first.

1. **Gate learnability probe** (§6). One day.
2. **Multi-frame stacking for noise reduction.** No training. Align N consecutive
   frames with the predicted field, stack, measure PSNR against the clean frame.
   Three conditions: no alignment, OSLO, PanoFlow. **The prediction is
   uncomfortable, which is why it is worth running: alignment should make stacking
   worse than not aligning.** If that shows up in PSNR, the finding stops being a
   metric curiosity and becomes a visible consequence, measured across three
   methods. The same experiment is then the demonstration vehicle for the gate.
3. **Frame interpolation.** Same data, predict the middle frame, PSNR against the
   real one. Second independent task, same prediction.
4. **Widen the empirical base.** The "nobody beats zero" claim currently rests on
   `flow360:test` alone. The project already has MPF City shards. Running the same
   table there turns one dataset into two, which is what separates a finding from a
   property of the regime.

### Why these tasks and not tracking or surveillance

Dense flow is not what surveillance uses. The applications where sub-pixel accuracy
is the binding requirement are **temporal alignment**: multi-frame noise reduction,
multi-frame super-resolution, frame interpolation, and stabilisation (whose
ego-rotation component is exactly the coherent field B′ estimates at 0.044°). In all
of them, asserting motion where there is none produces ghosting and background blur.
The metric failure and the product defect are the same failure seen twice.

**This industrial framing is inference, not measurement.** The project marks that
distinction elsewhere and must mark it here. Experiment 2 is what converts it.

---

## 8. Literature sweeps still owed

Two, before any novelty claim is written down. Each is about half a day and each
becomes a table in the paper.

1. ~~**Baselines reported by 360° flow papers.**~~ **DONE — see §8.2.** Six papers
   read in full text, none reports a trivial predictor. The sweep also mapped the
   abstention design space and exposed a real gap in our own claim (§8.2, "the gap
   this sweep exposed").
2. ~~Uncertainty and validity masks in optical flow.~~ **DONE — see §8.1. The
   hypothesis survives for the uncertainty class, but the sweep found real prior art
   elsewhere and the novelty claim has been rewritten accordingly.**

---

## 8.1 Sweep 2 result (2026-08-01): the claim must be rewritten

### Finding A — uncertainty methods weight, they do not abstain. Hypothesis holds.

ProbFlow jointly predicts flow and its uncertainty as a constrained mixture model,
and the multi-hypotheses line estimates local uncertainty in a single forward pass.
Ilg et al. frame the purpose explicitly: uncertainty is *"vital information when
building decisions on top of the estimated optical flow"* — the decision is
downstream and external to the model, which still emits a full dense field every
time. **For this class the proposed distinction is confirmed.**

### Finding B — motion segmentation is a different decision than ours

Joint depth/pose/flow work does produce static/dynamic masks: Competitive
Collaboration learns depth, camera motion, flow and motion segmentation together,
and related work uses the mask to route supervision, with the static area feeding
depth-pose training and the dynamic area feeding flow training.

Two reasons this is not the same decision:

1. The mask **routes the loss**, it does not zero the output field.
2. More fundamentally, "static" there means *static in the world*, and world-static
   scene content has **non-zero** flow whenever the camera moves — it carries the
   ego-motion field. Our gate is a **magnitude** decision (is the image-space
   displacement below what is resolvable and worth estimating), not a semantic one.
   The two coincide only when the camera is also nearly still, which is the FLOW360
   condition and is not the driving-video condition.

This distinction is sharper than the one originally written in §8 and should be the
one stated in the paper.

### Finding C — THE PRIOR ART IS VIDEO COMPRESSION, and it is 30 years old

This is the finding that matters. Block-based codecs have **SKIP mode**: residuals
are inferred to be zero and no motion vector is encoded, and when a segment has an
insignificant level of motion the encoder assigns the zero motion vector for skip
coding. Encoders check SKIP **first**, with early termination if its rate-distortion
cost is good enough. Learned codecs carry the idea over, with block-wise
content-driven mode selection in feature space.

So "decide that nothing moved and emit exactly zero" is **standard engineering
practice, not a new idea.** A committee member from video or multimedia will know
this. Writing "first estimator to contain the baseline" would be wrong and would be
caught.

### Finding D — classical optical flow abstained, and the field discarded it

Barron, Fleet and Beauchemin's canonical 1994 evaluation compared methods on
accuracy, reliability **and density** of the velocity measurements, and discussed
confidence measures while noting the lack of reliable ones. Density was an axis
because classical estimators *only reported flow where they were confident* —
Lucas-Kanade with an eigenvalue threshold on the structure tensor, phase-based
methods with a confidence test.

The deep era made flow **dense by construction** and dropped both the abstention and
the density axis from the evaluation. The capability existed, was standard, and was
lost — not because it was solved, but because dense regression architectures had
nowhere to put it.

### The rewritten claim, which is stronger than the original

Not "we invented the decision". The diagnosis is:

> **Compression prices abstention and flow does not.** A codec optimises
> rate-distortion, which explicitly charges for signalling motion, so emitting
> nothing is rewarded whenever it is nearly as good. The optical flow objective —
> EPE, or our geodesic error — contains no such price: it only rewards matching the
> ground truth and never rewards emitting nothing. A dense flow network therefore
> **cannot learn to abstain, because no term in its loss pays for it.** The field
> did not notice, because it never measured against the abstention baseline.

That reframes the contribution from "we add a gate" to "the flow objective is
missing a term that compression has had since the 1990s and that classical flow had
until the deep era, and here is what it costs on 360° video".

Three consequences for how this gets written:

- **Cite the codec precedent up front, do not hide it.** It converts the strongest
  available objection into supporting evidence, and it shows the decision is known
  to be the right engineering answer in an adjacent field with an explicit cost
  function.
- **Cite Barron et al. for density.** Restoring an abandoned evaluation axis is a
  defensible framing that a reviewer can check.
- **The novelty is the diagnosis plus the measurement**, not the mechanism.

### What the sweep did NOT establish

No learned *dense optical flow* method was found that emits exact zero as a
supervised per-pixel decision. **That absence is from four targeted searches, not
from a systematic review**, and it is the one claim still resting on incomplete
evidence. Sweep 1 (§8 item 1) remains owed, and the same protocol should cover the
flow-with-confidence literature between 1994 and the deep era, which these searches
only touched through Barron et al.

### Sources

- [ProbFlow: Joint Optical Flow and Uncertainty Estimation](https://www.semanticscholar.org/paper/ProbFlow:-Joint-Optical-Flow-and-Uncertainty-Wannenwetsch-Keuper/4a4a312d4aa1265afc8706102ed8588294db37cb)
- [Ilg et al., Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow (ECCV 2018)](https://arxiv.org/pdf/1802.07095)
- [Ranjan et al., Competitive Collaboration: Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation](https://arxiv.org/pdf/1805.09806)
- [Joint Self-supervised Depth and Optical Flow Estimation towards Dynamic Objects](https://arxiv.org/pdf/2310.00011)
- [Barron, Fleet & Beauchemin, Performance of Optical Flow Techniques (IJCV 1994)](https://link.springer.com/article/10.1007/BF01420984)
- [Motion Vector Coding and Block Merging in the Versatile Video Coding Standard](https://www.researchgate.net/publication/353593424_Motion_Vector_Coding_and_Block_Merging_in_Versatile_Video_Coding_Standard)
- [Method for coding motion in a video sequence (skip mode, zero motion vector)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7532808)
- [MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection](https://arxiv.org/pdf/2304.02273)

---

## 8.2 Systematic review (2026-08-01): sweep 1 closed, and the design space mapped

### Protocol, so this is auditable

Sweep 1 asked one question of each paper: **does any quantitative table contain a row
that is not a learned or hand-designed flow estimator** — a zero, identity, constant
or copy predictor. Inclusion was every 360°/panoramic optical flow paper that
introduces a method or a benchmark and reports EPE. Verification was **full text**,
not abstract: each PDF was downloaded, text-extracted, and searched for `zero flow`,
`zero motion`, `no motion`, `trivial`, `naive`, `identity`, `constant flow`,
`static baseline`, `copy`, `density`, `abstain`, `reject`. Every hit was read in
context, because the words appear in unrelated senses.

### Sweep 1 result — six papers, full text, zero trivial baselines

| Paper | Venue | Baselines actually reported | Trivial baseline |
|---|---|---|---|
| LiteFlowNet360 (Bhandari et al.) | ICPR 2020 | LiteFlowNet, LiteFlowNet+, own stages | none |
| 360° Flow w/ Tangent Images (Yuan, Richardt) | BMVC 2021 | OmniFlowNet, PWC-Net, RAFT, DIS | none |
| PanoFlow (Shi et al.) | T-ITS 2023 | RAFT, CSFlow, OmniFlowNet, own variants | none |
| SLOF / FLOW360 (Bhandari et al.) | ECCV 2022 | RAFT, finetuned RAFT, RAFT+KTN | none |
| MPF-Net / MPFDataset | ECCV 2022 | projection-fusion variants, PWC-Net | none |
| PriOr-Flow (Liu et al.) | ICCV 2025 | SphereNet+RAFT, TanImg+DIS/RAFT, MPF-net+PWC, SLOF+RAFT, PanoFlow+RAFT/CSFlow, GMA, SKFlow | none |

PriOr-Flow is the strongest case: **not one** of the search terms appears anywhere in
the full text. The claim in §2 now rests on read evidence rather than memory.

One qualifier that must be written down. SLOF does something closer to our protocol
than anyone else: its Table 1 decomposes EPE by ground-truth speed
(`s<5`, `s<10`, `s<20`, `s≥20`). So the tooling to see this exists in the field. What
is missing is the row to compare against, not the decomposition.

### Sweep 3 result — the design space, and the one empty cell

The right way to state the novelty is not "nobody has done a gate". It is a taxonomy
of *what happens when a model declines*, which has six occupied cells and one empty
one.

| Family | Decision unit | When it fires | Is that region still scored? | Priced in the objective? |
|---|---|---|---|---|
| Flow confidence / uncertainty (Mac Aodha, ProbFlow, Ilg, PDC-Net) | pixel | flow still emitted, confidence reported beside it | evaluated by *sparsification*, which **removes** pixels | no |
| Selective prediction (SelectiveNet, El-Yaniv & Wiener) | sample or pixel | prediction **withheld**, coverage drops | no, withheld predictions leave the metric | **yes** — risk/coverage |
| Video codec SKIP (H.264/HEVC/VVC) | block | zero MV, zero residual | yes, distortion is still measured | **yes** — rate/distortion |
| Skip-Convolutions (CVPR 2021) | feature location | features **copied from the previous frame** | yes, task accuracy | a FLOP budget, not an error price |
| Event-flow masked evaluation | pixel | prediction masked **at evaluation time** | no | no |
| Motion segmentation (Competitive Collaboration) | pixel | mask **routes the loss** | yes | no |
| **This proposal** | node | **emits exact zero** | **yes, full field** | **no — and that is the hole** |

Read the "still scored" column. This is the discriminator, and it is checkable by a
reviewer in one pass. The uncertainty line and the selective-prediction line both
**shrink the evaluation set**: sparsification plots re-score after deleting the
least-confident pixels, and selective prediction reports risk at a coverage below
100%. Both make the number better by looking at less. A gate that emits zero and is
still scored on every node cannot do that — abstention has to *earn* its place
against the ground truth in the region where it fired.

### The theoretical anchor we were missing

Selective prediction is the formal name for abstention in machine learning, and it
**does** price it: the risk-coverage curve is the objective, and SelectiveNet
optimises prediction and rejection jointly end to end. Compression prices it too, in
bits. So the diagnosis in §8.1 sharpens into something citable:

> Two mature fields price abstention explicitly — machine learning as *coverage*, and
> compression as *rate*. Dense optical flow inherited neither. Its objective charges
> only for disagreeing with the ground truth, so abstention is unrepresentable: there
> is no coverage term to trade against, and no bit cost to save. The gap is not that
> nobody thought of the mechanism. It is that the flow objective has no slot for it,
> and the evaluation never exposed the cost of that.

### Prior art that must be cited and distinguished, not hidden

**Skip-Convolutions** is the closest neural precedent and needs handling with care.
It couples every layer to a binary gate that decides whether a residual matters, and
when the gate closes, *"output features are copied from the previous time step"*.
Three differences, all verifiable in their text: the gate acts on **features**, not
on the output field; closing it means **copy**, not **zero**; and the goal is stated
as reducing cost *"without any accuracy drop"* — accuracy is a constraint, not the
thing being improved. They also cite HEVC residual coding as the inspiration, which
ties this cell to the codec cell.

There is a genuine gift in that paper. Their **Norm gate has no trainable parameters
at all** — it thresholds the residual magnitude — and it works. That is independent
evidence that a magnitude criterion is enough to find static regions in real video,
which is exactly what our §6 learnability probe has to establish.

**E-RAFT and the event-flow evaluation debate** is the closest precedent for the
*argument*, not the method. That community found that masking predictions to pixels
with events makes the reported error about 11% lower than dense evaluation, and it
has been arguing about whether masked numbers are honest. That is our argument with
the sign reversed, from a different subfield, and it shows a community can be moved
by this class of evidence.

### The gap this sweep exposed in OUR OWN claim

Sweep 1 turned up a problem worth more than everything else here. SLOF's Table 1
reports, in plain 2D EPE, RAFT at **0.558** and SLOF v1 at **0.309** in the `s<5`
bucket. Our zero-flow measurement lives in a different metric (geodesic degrees on
the node grid, actives thresholded at 0.25°/0.5°/1.0°) at a different resolution.
Nothing published tells us what a zero predictor scores **inside SLOF's own buckets,
in SLOF's own metric, at SLOF's own resolution.**

So the sentence "nobody beats zero on FLOW360" is currently scoped to our protocol,
and a reviewer with the FLOW360 data and an afternoon can test it in theirs. If the
zero row lands above 0.309 in the `s<5` bucket, the universal claim narrows to a
statement about our metric, and the paper must say so.

**Action, and it outranks the learnability probe: reproduce SLOF Table 1 with one
extra row.** Same data, same resolution, same buckets, same EPE definition, plus
`zero`. It is cheap, we already have FLOW360, and it is the first thing an examiner
would ask for. Either it produces the single most persuasive table in the thesis, or
it tells us to scope the claim before we publish it — and finding that out ourselves
is much better than being told.

### What this review still does not establish

The absence claim is now *"no method in the surveyed families emits a supervised
exact zero on a still-scored dense field"*, over six 360° papers read in full and the
five adjacent families above. It is not a claim about all of optical flow. Two known
holes remain: the confidence-measure literature between 1994 and the deep era was
sampled (Bruhn & Weickert 2006, Kondermann 2008, Mac Aodha 2013) rather than
enumerated, and no search covers non-English or pre-1994 work.

### Sources

- [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html)
- [Mac Aodha et al., Learning a Confidence Measure for Optical Flow (PAMI)](https://people.inf.ethz.ch/pomarc/pubs/MacAodhaPAMI12.pdf)
- [PDC-Net+: Enhanced Probabilistic Dense Correspondence Network](https://arxiv.org/pdf/2109.13912)
- [Kondermann et al., A Statistical Confidence Measure for Optical Flows (ECCV 2008)](https://cvg.cit.tum.de/_media/spezial/bib/nieuwenhuis-et-al-eccv08.pdf)
- [Habibian et al., Skip-Convolutions for Efficient Video Processing (CVPR 2021)](https://arxiv.org/pdf/2104.11487)
- [Gehrig et al., E-RAFT: Dense Optical Flow from Event Cameras](https://arxiv.org/pdf/2108.10552)
- [Bhandari et al., Revisiting Optical Flow Estimation in 360 Videos (ICPR 2020)](https://arxiv.org/pdf/2010.08045)
- [Yuan & Richardt, 360° Optical Flow using Tangent Images (BMVC 2021)](https://arxiv.org/pdf/2112.14331)
- [Shi et al., PanoFlow (T-ITS 2023)](https://arxiv.org/pdf/2202.13388)
- [Bhandari et al., SLOF / FLOW360 (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680546.pdf)
- [Liu et al., PriOr-Flow (ICCV 2025)](https://arxiv.org/pdf/2506.23897)

---

## 9. Relationship to the existing work

### Survives, load-bearing

- The spherical evaluation protocol: geodesic error by region and by motion band,
  always against zero-flow. It is the instrument that makes the finding possible.
- The metric corrections: the `arccos` float32 floor (`UNIVERSALITY_TABLE.md` §12)
  and the per-area node weighting (§14.3).
- The universality table (§10) and the regime-contrast figure.
- The P0d causal decomposition, which motivates the gate.
- The static calibration measurement, which is its direct evidence.
- The grid-floor probe (§13), which excludes estimation resolution as the cause.
- The whole data pipeline: sfprep, the shard format, the chairs360 generator.
- A1 (§14) as a controlled negative result on equal-area sampling.

### Dies

Two claims, not the work: "equal-area sampling buys polar accuracy" (killed by A1)
and "OSLO-RAFT is good at large motion" (killed by PanoFlow on flowscape). Neither
was the central result.

### Demoted

OSLO-RAFT stops being *the proposal* and becomes two things: the native
instantiation of the gate, and the vehicle for the spherical-geometry investigation.
This is a real demotion of role and should be stated plainly rather than dressed up.
It is not a discard.

### Measured levers still open for OSLO-RAFT itself

1. **The equiangular grid.** 33 to 35% lower global error, matched pair, two seeds
   with non-overlapping ranges, at zero parameter cost. The production-scale test is
   specified in §14.6 and not yet run.
2. **Capacity and training recipe.** The remaining untested hypothesis for the gap
   to PanoFlow. Training uses batch 2 and the gradient-accumulation compensation
   failed; a genuinely larger batch has never been run.
3. **The gate**, per §5.

**Honest ceiling.** Stacking all three, OSLO-RAFT probably still does not beat
PanoFlow at large motion: that gap is 4.6× and comes substantially from perspective
pretraining a spherical graph architecture cannot inherit. The realistic outcome is
competitive in the sub-pixel regime and behind in the large-motion regime.

---

## 10. If this direction holds

The thesis question stops being "did we build a better 360° flow estimator" and
becomes **when is estimating flow on 360° video better than not estimating, and how
do you build an estimator that knows the difference**.

Three-sentence form for a committee:

> The 360° optical flow literature never compares its methods against the trivial
> zero-motion predictor. We compared, and no published method beats it on
> consecutive-frame video, including ours. The proposal is an estimator that decides
> whether motion occurred before estimating it, and which by construction contains
> that predictor, so it cannot lose to it.

Related: `UNIVERSALITY_TABLE.md`, `ROADMAP_SEMINARIO.md`, `P2C_TRAINING_CAMPAIGN.md`
