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

**P0b (next 5-min probe, lever implemented+gated 2026-07-11):** per-pixel iid
Gaussian noise via `--synth-photo-noise-std` (std in 1/255 units; mean Δ = 0.8·std,
verified to 3 decimals; same dedicated-generator nesting). Discriminates the two
remaining suspects: if std 0.5–2/255 (real-magnitude, spatially *unstructured*)
reproduces the damage, the wall is high-frequency appearance noise (render/AA) and
P1 centers on per-pixel-noise robustness; if the model shrugs it off like global
jitter, the wall is *structural* (occlusion/specular edges) and the eraser +
real-pair emphasis wins. Sweep: std {0.5, 1, 2, 4, 8}.

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
