# P2C: the RAFT-recipe training campaign on the sphere

**Goal.** Train OSLO-RAFT-R with the recipe that actually produced RAFT's robustness —
staged curriculum (Chairs→Things→domain), asymmetric photometric augmentation,
occlusion erasing, ~100k-step budgets — and measure whether it moves the two numbers
that matter: (1) the real-frame-2 leg of the decisive triangle (the wall itself), and
(2) the head-to-head vs frozen ERP-RAFT, now in EPE (P2A). The rotation-protocol
evaluation (§4) is the paper's third, uncontestable table.

**The core hypothesis, stated falsifiably.** The triangle showed +80.6% on
photometrically-clean pairs vs −32% on real pairs at the same motion scale. RAFT
crosses that gap; its training manufactured the robustness via nuisance augmentation
at scale. If OSLO trained with the same nuisance curriculum still fails the real leg,
the wall is *not* recipe — that is a publishable negative sharper than Phase 1's. If
it passes, that's the paper.

---

## 1. Prerequisites

- P2A merged (EPE reporting; grid-floor verdict decides §3 Stage P3's readout).
- P2B shards available (`chairs360:{train,val}`).
- Throughput anchor: Stage A measured ~1 step/s on visco3 (5k steps = 82 min) ⇒
  100k steps ≈ 28 h. Budget below assumes batch 2 × `--grad-accum 3` = effective 6
  (RAFT used 6–12); re-measure with grad-accum before committing the schedule.

## 2. New training-time augmentation (this repo)

All applied in `ShardFlowDataset` on the ERP frames *before* node sampling, gated by
flags, default-off (existing runs stay reproducible). RAFT-parity values as defaults.

| flag | mechanism | RAFT parity |
| --- | --- | --- |
| `--photo-prob 0.8` | color jitter: brightness 0.4, contrast 0.4, saturation 0.4, hue 0.16 (torch ops on uint8→float frames) | yes (their aug pipeline) |
| `--photo-asym-prob 0.2` | with this prob, re-draw jitter *independently for frame 2* — the direct attack on the triangle wall | yes |
| `--eraser-prob 0.5` | 1–3 **geodesic caps** on frame 2 (center uniform on sphere, radius U[2°, 12°]) filled with the frame-2 mean color; GT untouched | rectangles in RAFT; caps are the raster-honest spherical analog |
| `--photo-noise-std 0.02` | additive Gaussian on both frames (optional, off in parity runs) | no (ours; for the decomposition study) |

Implementation: `spherical_flow/photometric.py` — pure-torch `jitter(frame, params)`,
`erase_caps(frame, dirs_grid, params)` (reuse the cached pixel-direction grid from
`equirectangular_pixels_to_unit_vectors`); wired into `_augment_record` after SO(3),
before node sampling; params drawn from the per-record generator so `--seed`
reproduces augmentation exactly.

**CPU gates (Docker, before any training):** (a) jitter/erase are no-ops at prob 0;
(b) asym draw actually differs between frames (hash check); (c) eraser caps land
uniform-on-sphere (latitude histogram); (d) GT/valid tensors bit-identical with and
without photometric aug (it must never touch geometry).

## 3. Stages and gates

Every stage: seeds 7/11/19 for anything headline-bound; AdamW + OneCycle + clip 1.0
(unchanged); `--eval-every` ≥ 5 evals/stage; post-fix masks; EPE reported alongside
geodesic from day one.

### Stage P0 — nuisance-sensitivity probe (cheap, before everything)

Re-run the triangle's *resampled* leg on the existing B′ checkpoint but with
increasing asymmetric jitter injected into the resampled frame 2 (eval-only, ~30 s per
point). Output: the degradation curve +80.6% → ? as photometric nuisance approaches
real-Δf levels. **This curve is the paper's Figure 1 regardless of outcome** — it
locates the wall on a continuous axis and predicts how much of the real gap is
photometric (vs occlusion/parallax). Zero training cost.

**Status: lever implemented + gated (2026-07-11).** `spherical_flow/photometric.py`
(RAFT-parity ranges × scale; brightness→contrast→saturation→hue, hue = YIQ chroma
rotation) applied to the synthetic frame 2 via `--synth-photo-scale` (dataset knob
`synth_photo_scale`, train+val synth records alike). Jitter draws use a *dedicated*
generator, so runs differing only in scale share bit-identical rotations/frames/GT
(gated in Docker: scale-0 no-op & deterministic; isolation — only frame2 moves;
nested determinism at scale>0; monotone delta). **Measured calibration
(flow360:val, 4 pairs):** mean |Δframe2| = 3.9 / 7.8 / 11.6 / 15.4 per 255 at
scales 0.25/0.5/0.75/1.0 (≈ 15.6·scale); context: real pairs |f2−f1| = 3.03/255 vs
synth motion-only 2.61/255 — the *mean* photometric excess of real imagery is
≈ 0.4/255, i.e. scale ≈ 0.03. The sweep therefore concentrates low:
scales {0, 0.05, 0.1, 0.25, 0.5, 1.0}. Interpretation guide: if the model survives
scale ≥ 0.1 (≈ 4× the real mean excess) while the real leg sits at −32%, plain
photometric *magnitude* does not explain the wall — its spatially-structured
component (specularity, shading, occlusion edges) does, and the eraser/occlusion
levers outrank global jitter for stage P1.

**P0 RESULTS (2026-07-11, box, B′ ckpt, flow360:val, synth-rot 0.1–0.5°, 791 pairs;
scale-0 re-anchors +80.6% exactly under post-fix masks):**

| scale | mean Δ /255 | global | active₀.₂₅ | poles | equator |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0 | +80.6% | +86.1% | +67.4% | +83.3% |
| 0.05 | 0.8 | +75.7% | +82.3% | +61.4% | +79.3% |
| 0.10 | 1.6 | +69.0% | +77.0% | +53.6% | +73.5% |
| 0.25 | 3.9 | +51.1% | +62.2% | +33.3% | +56.7% |
| 0.50 | 7.8 | +26.1% | +40.4% | +6.1% | +32.0% |
| 1.00 | 15.4 | −6.2% | +10.0% | −28.9% | +0.1% |

Near-linear: global ≈ 77.4 − 5.71·Δ₂₅₅ (graceful, no cliff; poles degrade ~1.4×
faster than equator). **Verdict: photometric magnitude is ruled out as the wall.**
At the real-pair photometric excess (Δ ≈ 0.42/255) the curve predicts **+75%**;
the real leg measures **−32.2%** — a 107-point structured-damage gap. To cause
−32% with global jitter takes Δ = 19.2/255 = **46× the real mean excess**. ⇒ the
decision tree's first branch fires: the operative nuisance is spatially structured
(per-pixel render/AA noise, specularity, shading, occlusion edges, parallax), and
eraser/occlusion + per-pixel noise levers outrank global jitter for P1. This
table + the real-leg point below it = paper Figure 1.

**P0b RESULTS (2026-07-11; per-pixel iid Gaussian via `--synth-photo-noise-std`,
std in 1/255 units, mean Δ = 0.8·std):** std 0.5/1/2/4/8 → global
+80.3/+79.6/+77.7/+73.0/+62.5% (slope −2.92 pts per 1/255 — *half* the jitter
slope; iid noise averages out in pooled correlation features). **At the
real-magnitude delta (0.42/255): iid noise costs 0.33 pts, global jitter 2.65 pts,
the real leg costs 112.8 pts.** Combined P0+P0b verdict — the decomposition triad
is measured: spatially-coherent photometric shifts (eliminated), spatially-
unstructured per-pixel noise (eliminated, even more robust), leaving
**spatially-STRUCTURED appearance change** as the wall: view-dependent
specular/shading shifts tied to scene surfaces, render/AA differences concentrated
at image edges, occlusion/disocclusion bands, plus the §7.4 residual confound
(non-rigid parallax field vs global rotation — a GT-structure, not appearance,
difference). Paper Figure 1 = both curves + the real-leg point 100+ points below.

**P1 design implication (binding):** global jitter and iid noise are cheap
regularizers but demonstrably cannot carry robustness to the real nuisance; the
levers that model *structured* nuisance move to the front, and Chairs-360's
layered occlusion (P2B) gains weight as the primary training signal.

**Δf-STRUCTURE DIAGNOSTIC RESULTS (2026-07-11, `analyze_appearance_residual.py`,
local Docker; self-test: copy→0, jitter→autocorr .98, noise→white/uniform).**
Constancy residual Δ(x) = f2[x + GT(x)] − f1(x), full flow360:val (791) and
replica360:val (162):

| stat | flow360:val | replica360:val |
| --- | --- | --- |
| mean abs Δ (warped) | **3.12/255** | 4.17/255 |
| mean abs (no-warp) | 3.10/255 | 31.6/255 (GT explains 87%) |
| p50 / p90 / p99 | 1.33 / 6.7 / **32** | 1.60 / 7.9 / 52 |
| edge corr / top-10% mass / hi-lo ratio | 0.46 / **0.33** / **17×** | 0.43 / 0.35 / 9× |
| autocorr lag-1 / lag-4 | 0.70 / 0.22 | 0.67 / 0.39 |
| luma share | 0.84 | 0.95 |
| corr with motion magnitude | 0.10 | −0.01 |

**Correction to the P0 anchor.** The earlier "real excess ≈ 0.42/255" subtracted
two |f2−f1| means with different motion contents — methodologically inferior. The
direct measurement: the motion-compensated residual is **3.12/255** (warping with
exact GT reduces nothing on flow360 — appearance change is ~100% of the
inter-frame difference at sub-pixel motion). Revised bookkeeping: at Δ = 3.12 the
jitter curve gives ≈ +57%, the noise curve ≈ +73%, the real leg −32.2% ⇒
magnitude explains ~10–20% of the 113-point wall; **structure still carries
80–90%**. Qualitative P0 verdict unchanged, numbers corrected.

**The measured enemy:** sparse (p99/p50 ≈ 24×), edge-anchored (top-decile-gradient
pixels carry 33% of the mass at 17× the flat-region amplitude), mesoscale-
correlated (coherence length ~2–4 px), luma-dominant (84–95%). This also
*explains the Act-I LK anti-correlation* (LK more wrong where features are
sharpest): the nuisance is 17× stronger exactly where the sub-pixel motion signal
(|∇f|·displacement) lives. The wall in one sentence: **at sub-pixel motion, the
appearance residual is concentrated on the same edge pixels that carry all the
correspondence signal, and is larger than that signal.**

**P1 augmentation spec (from the measurement):** an edge-modulated corruption op —
`field = envelope(|∇f1|) × smooth_noise(corr_len≈3px) × heavy_tail_amplitude`,
luma-dominant, calibrated to mean Δ ≈ 3–4/255 with p99/p50 ≈ 25× — applied
asymmetrically to frame 2 (synth and, in training, real pairs too). Gate: re-run
this diagnostic on augmented synth pairs and match the table above; then the P0
sweep with *this* op is the honest predictor of the real leg.

**IMPLEMENTED + MATCH-THE-TABLE PASSED (2026-07-11):**
`photometric.edge_corruption` — envelope = 3·(|∇luma|/mean)·sigmoid-gate + 0.35
floor, where the gate is broad-scale (σ 14 px, bias 0.8: only ~20% of edge patches
light up, matching the real residual's *partial* edge coverage), noise = two-scale
blur mix (σ 0.9 + 6.0 px, weight 0.45) shared-luma + 0.45·per-channel chroma;
corrupts the **raster** frame 2 is sampled from (`synth_rotation_record` gained
`frame2_erp`), flag `--synth-edge-corrupt-delta` (target mean Δ in 1/255). Match
vs the measured real table (8 flow360 frames, diagnostic metrics):
mean 2.96 vs 3.12 ✓; edge_corr 0.49 vs 0.46 ✓; top-10% mass 0.38 vs 0.33 ✓;
autocorr h1/h4/v1 0.73/0.18/0.71 vs 0.70/0.22/0.69 ✓; luma 0.83 vs 0.84 ✓;
hi/lo ratio 8× vs 17× (**known gap** — the real ratio has ±27 cross-pair std;
chasing it would over-fit a high-variance statistic). Isolation/determinism/
nesting gates pass through the dataset (GT bit-identical across deltas).
**P0c RESULTS (2026-07-11):** delta 1.5/3.1/6.2 → global +64.8/+44.4/+12.3%
(anchor +80.6, real −32.2). **Damage ladder at the real magnitude Δ=3.1/255:**
iid noise 7 pts, global jitter 23 pts, edge-structured 36 pts, real 113 pts —
structure ordering confirmed (edge-op 1.6× jitter, 5× noise per unit delta), but
the matched op explains only **~1/3 of the wall**; even at 2× real magnitude it
sits 44 pts above the real leg. Magnitude-equivalents to reach −32%: edge-op ~3×,
jitter ~6×, noise ~12× real. **Un-modeled remainder (~2/3): (a) extreme cross-pair
events (the hi/lo 17× ± 27 heterogeneity), (b) occlusion/disocclusion bands, and
(c) the motion-field-structure axis** — the val protocol's global rotations are
GRU-aggregatable; real parallax is not, and no appearance op touches that.
**P0d RESULTS (2026-07-11) — THE DOMINANT AXIS IS THE MOTION FIELD, NOT
APPEARANCE.** Real GT motion + perfectly clean appearance (`--val-real-resample-
prob 1.0`, zero columns identical to the real eval's): global **−72.5%** (error
0.363° vs zero 0.211°), active₀.₂₅ −57.6%, poles −93.4% — *worse than the real
leg's −32.2%*. The decomposition square (B′, flow360:val):

| | clean appearance | real / matched appearance |
| --- | --- | --- |
| rotation motion (0.1–0.5°) | **+80.6%** | +44.4% (edge-op @ 3.1) |
| real motion field | **−72.5%** | −32.2% (real) |

Motion-structure swap at clean appearance: **−153 pts** — 4× the appearance swap
(−36 pts at rotation motion). And the appearance effect *flips sign* at real
motion (+40 pts): appearance noise damps the model's confidence, accidentally
helping when the true field is near-zero. Reading: B′ resolves sub-pixel
*coherent* fields to 0.046° but cannot read the real field's structure
(mostly-static with sparse sub-pixel parallax); clean inputs raise its confidence
in a rotation-like prior (its synth curriculum's family) and errors grow.
**Controls (2026-07-12): the failure is checkpoint-universal.** Same P0d leg on
Stage B (synth exposure only prob 0.1 @ 1–15°): **−57.6%** (0.332°, poles −112%);
on Stage A (never saw flow360): **−118%** (0.459°). Ordering B (−58) > B′ (−73) >
A (−118): every trained variant fails on the clean-appearance real field, all
with errors above their own real-leg floors (B 0.332 vs 0.295; B′ 0.363 vs
0.278 — the sign-flip is consistent). The B′-vs-B gap (~15 pts) isolates the
rotation-curriculum prior misfire as a *minor* component; the shared −58%+ is
the field-structure wall itself. Even active₁.₀ nodes (GT ≥ 1°) are negative on
all three. **P0 is closed: 6-cell decomposition + 3-checkpoint robustness.**

**P1a RESULTS (2026-07-12): GATE PASSED — the real field IS learnable.** Init B′,
flow360:train, `--real-resample-prob 1.0 --loss-motion-weight 1.0`, 5k steps
(95 min): clean-leg val −72.5% → **+31.7% @ 1k → +42.9% @ 2k → plateau +42.7% @
5k** (equator +46, poles +23, seam +29; active₀.₂₅ **+63.4%**, active₀.₅ +61.1%,
active₁.₀ +32.9%; error 0.121° vs zero 0.211°; aux at its floor throughout). The
field-structure wall is *trainable*, not architectural — 95 GPU-minutes settled
it. Plateau by 2k steps suggests the next gains need lr schedule/steps/data, not
a redesign.

**P1a transfer eval (real pairs, the money leg): −72.8%** — worse than B′'s
−32.2% globally, BUT the active subsets are the best any variant has scored on
real pairs: active₀.₂₅ **−11.0%**, active₀.₅ −6.6%, active₁.₀ −4.2% (B′: −2.9%
active₀.₅ with a zero-collapsed global). Reading: the field-trained model now
*commits* — on static nodes its confidence costs it globally, and its clean-field
reading strategy leans on exactly the sub-pixel edge cues where the measured
nuisance is 17× concentrated. **The appearance gap at fixed (learned) motion
structure is now cleanly measured: +42.7 → −72.8 = 115 points — the axes
interact; peeling the field layer exposed the appearance layer at full size.**

**⇒ P1b (the composition, ~95 min): train on real fields WITH the nuisance ramp**
— real-resample pairs + edge-corruption (+ jitter) on the resampled frame 2, val
on *plain real pairs* so the eval-every trajectory tracks Gate R2 directly.
Requires wiring the photometric levers into `_real_resample_record` (they
currently only touch `_synth_record`). If the trained-with-nuisance model holds
part of its active-subset gain on real pairs, the recipe (field + nuisance
robustness co-trained) is the campaign core and Chairs-360 scales it; if nuisance
training destroys the field skill, the un-modeled nuisance components
(cross-pair events) become the bottleneck to model next.

**Campaign-design consequence (binding for P1):** the coherence prior is the
enemy as much as the nuisance. (i) `--real-resample-prob` is now a TRAINING lever
— real GT fields with exact constancy are a new exact-supervision data source
that teaches real field structure nuisance-free; (ii) Chairs-360's independent
sprites (locally coherent, globally incoherent, static background) are the
synthetic family that breaks the global-aggregation shortcut — promoted from
"occlusion source" to the campaign's central asset; (iii) the P0a–c nuisance
levers remain the robustness axis, layered on top. P1 curriculum: real-resample
+ Chairs-360 first (learn the field), nuisance ramp second (keep it under
noise).

### Stage P1 — Chairs-360 bootstrap (~28 h)

`chairs360:train`, 100k steps, full augmentation, `--so3-prob 1.0`.
- **Gate P1a:** chairs360:val geodesic improvement ≥ +85% AND `--ablate-corr`
  eval-collapse (matching, not prior, carries it — the G-gate P2B §5 previewed).
- **Gate P1b (the wall moves or it doesn't):** flow360:val real pairs, zero-shot from
  P1: active₀.₅ vs the Phase-1 best (−2.4%). *Any* positive value is the first
  crossing ever; even −1% → 0 movement tells us nuisance training transfers.
- **Gate P1c:** replica360:val real ≥ +80% zero-shot (no catastrophic domain gap).

### Stage P2 — Things-analog mix (~28 h)

`chairs360:train,replica360:train,mpf:train` (real parallax + rendered scenes), 100k
steps, full augmentation, init from P1.
- **Gate P2a:** replica360:val ≥ Stage-A three-seed mean (1.74°) — the bootstrap must
  not cost the large-motion result.
- **Gate P2b:** head-to-head vs frozen RAFT-large re-run (geodesic + EPE): poles win
  must hold; global gap ≤ 1.2× (Stage A was 1.35×).

### Stage P3 — domain fine-tune (~14 h)

Mix weighted toward flow360 (`--loss-motion-weight 1.0` machinery from B′), 50k steps,
lr 1e-4, init from P2.
- **Gate P3 (= Gate R2, third attempt, now earned):** flow360:val real active₀.₅
  > +5.2%. Pass ⇒ paper headline. Fail after P1b showed movement ⇒ decompose with the
  P0 curve (photometric solved, occlusion/parallax remains) — still a strong paper
  with the negative precisely factored.

### Stage P4 (conditional) — distillation to close the equator

Only if P2b's global gap stays > 1.2×: frozen RAFT-large pseudo-labels on unlabeled
360° video, confidence-masked to the equator band, mixed 1:1 with P2 data, 50k steps.
Target: match RAFT globally while keeping the poles win ⇒ strict dominance table.

## 4. The rotation protocol (evaluation contribution, ~zero cost)

Evaluate every checkpoint AND frozen RAFT-large on replica360:val + flow360:val under
`k` fixed random SO(3) rotations of the *evaluation* pairs (rotate frames + GT with
the existing so3 machinery; publish the seeds). Report mean and worst-rotation
metrics. ERP methods train and eval on equator-biased content; OSLO is
rotation-equivariant by construction. Expected: RAFT's global advantage inverts under
rotation; OSLO ≈ invariant (Stage A already trained at `--so3-prob 1.0`). This table
needs no new training and no new code beyond an eval flag (`--eval-so3-seeds 0,1,2`),
and it reframes "loses global" as "wins the unbiased protocol."

## 5. Decision tree

```
P0 curve flat (jitter doesn't degrade resampled leg)
   → wall is occlusion/parallax, not photometric → prioritize eraser/occlusion work, P1 unchanged
P1b positive → recipe transfers → full send P2/P3
P1b zero-ish AND P1a passed → nuisance transfers only in-domain → P3 becomes
   flow360-with-augmentation retry (B′ + new aug), before P2
P3 pass → paper: first native-spherical RAFT to beat zero-flow on FLOW360 real pairs
   + poles win + rotation protocol
P3 fail → paper: recipe-complete negative + P0 decomposition + poles win + rotation
   protocol (still submittable; the triangle argument gets its final edge)
```

## 6. Cost summary

| item | GPU time |
| --- | --- |
| P0 probe | ~5 min |
| P1 + P2 + P3, single seed | ~70 h |
| headline stages ×3 seeds | +~56 h (P1 can stay single-seed; seed the P2/P3 headlines) |
| evals/head-to-heads/rotation protocol | ~2 h |
| **total** | **~5–6 GPU-days** |
