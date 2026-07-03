# OSLO-RAFT: A HEALPix-Native RAFT and the Sub-Node Motion Wall

**A consolidated final report.** This document folds the chronological development log in
[`OSLO_RAFT_PLAN.md`](OSLO_RAFT_PLAN.md) into a coherent, thesis-ready account of a single
result: a HEALPix-native reformulation of RAFT, trained end-to-end on real 360° optical-flow
data, **degenerates to a frame-1 appearance prior because the inter-frame motion is sub-node
at every affordable spherical grid.** Both correspondence paradigms fail from opposite
directions — discrete matching goes *inert*, and a differential (Lucas–Kanade) estimator goes
*actively wrong* — and a six-rung ablation ladder pins the mechanism at every step. This is a
complete, well-characterized negative result and is the intended basis for the thesis Ch. 4.

`OSLO_RAFT_PLAN.md` remains the primary source and running log; this document is the report
built from it. Where the two disagree, the log is authoritative for dates/commands and this
document is authoritative for the narrative.

---

## 1. Summary

We built **OSLO-RAFT**, a spherical recast of RAFT native to the HEALPix grid: OSLO's
`SDPAConv` as the spatial operator, a spherical exp-map + neighbor-grid correlation lookup in
place of RAFT's planar bilinear lookup, an SDPAConv `ConvGRU` iterative updater, a convex
HEALPix upsampler, and a geodesic sequence loss — all trained with full-SO(3) rotation
augmentation on three independent 360° flow datasets. The architecture is sound: it overfits
10 pairs to near-zero geodesic error, and at scale it learns a real signal that beats
zero-flow on every moving-pixel subset.

But it plateaus. Across resolutions (r4/r5/r6), supervision grids, loss reweightings, and two
entirely different correspondence mechanisms, the model lands on the **same +2.9% active-subset
improvement**. The ablation ladder shows why: the correlation mechanism does no work
(zeroing it changes nothing), the model never uses frame 2 (a frame-1-only model gets the full
+2.9%), and a differential estimator built specifically for sub-pixel motion underperforms
zero-flow and is anti-correlated with the true motion exactly where it is most confident.

**Root cause:** the median inter-frame motion is 0.099° — about **0.027 of a node at r4** and
still sub-node at r6 (the finest grid that fits a 24 GB GPU). Discrete matching cannot resolve
displacement below ~½ a node; a differential estimator drowns the ~3% motion signal in the ~97%
non-motion feature difference. The correspondence signal this data affords is below the
resolving power of both paradigms on any affordable HEALPix grid, so the network falls back to
the only thing that helps — an appearance prior over frame 1.

---

## 2. The model

New module `spherical_flow/oslo_raft.py` (+ variants), runner `run_oslo_raft.py`. Parameter
budget 1–3M (RAFT-large is 5.3M; far less data ⇒ stay smaller). All variants share the same
building blocks; they differ only in how correspondence is computed.

| Component | Implementation |
| --- | --- |
| **Feature / context encoders** | Siamese `SDPAConv` residual blocks on the nested HEALPix hierarchy; 4-to-1 child pooling r6→r5→r4, channels 32→64→96, GroupNorm. Context encoder runs on frame 1 only, split into GRU hidden init (tanh) + context features (relu), as in RAFT. |
| **Correlation volume** | All-pairs cosine correlation at the estimation grid (`[N,N]`); correlation pyramid by pooling the *second-image* axis via nested HEALPix averaging (r4→r3→r2→r1). |
| **Spherical lookup (§4.3, the central geometric claim)** | Given flow `fᵢ`: endpoint `eᵢ = expmap(pᵢ, fᵢ)`, center `cᵢ = ang2pix(eᵢ)` at each corr level, gather the correlation over `cᵢ`'s k-hop HEALPix neighborhood, concat across levels. Replaces RAFT's planar bilinear grid lookup. |
| **Iterative updater (§4.4)** | SDPAConv `GraphConvGRU`; inputs = correlation feature + tangent flow + context; output = **zero-initialized** delta tangent flow `[east, north]`. Flow composes in the tangent plane (valid for the small per-step magnitudes here). 8–12 iters train, up to 24 eval. |
| **Convex upsampler (§4.5)** | Each r4 parent predicts softmax weights over its 1-hop neighborhood for each of its 16 r6 descendants; upsampled flow = weighted combination via **parallel transport of the tangent flow** (endpoint-averaging would break the cold-start contract). |
| **Loss** | γ-weighted geodesic sequence loss (γ=0.8), the spherical analogue of RAFT's sequence loss, masked by validity, computed at the supervision grid: `L = Σₜ γ^(T−t) · mean_valid( geodesic(expmap(p, fₜ), gt_endpoint) )`. |

**Cold-start contract (RAFT-preserving):** the zero-init delta head ⇒ `flow = 0` at every
iteration at init ⇒ `convex_upsample(0) = 0` ⇒ `preds[0]` is exactly zero end-to-end. Verified
in every smoke test.

### 2.1 Model variants built

| Variant | File | What it changes | Params |
| --- | --- | --- | --- |
| `OSLORAFT` | `oslo_raft.py` | Single-resolution baseline (estimation == supervision). | 1.2M |
| `OSLORAFTPyramid` | `oslo_raft_pyramid.py` | Estimate at r4, convex-upsample + supervise at r6. | 1.40M |
| `OSLORAFTLocal` | `oslo_raft_local.py` | Single-res r6 with a **lazy local** correlation (`O(N·M·C)`) replacing the 9.66 GB all-pairs volume; gradient checkpointing on the per-iter update. | 1.20M |
| `OSLORAFTDiff` | `oslo_raft_diff.py` | Differential (spherical Lucas–Kanade) head: `fnet` + one linearized solve, no correlation/GRU/context. | 214k |

### 2.2 Diagnostic levers built (all CPU-validated before any GPU run)

- `--ablate-corr` — zero the correlation feature before the motion encoder.
- `--ablate-context` — zero the GRU hidden-init and the per-iteration context feed.
- `--loss-motion-weight` / `--loss-motion-ref-deg` — up-weight pixels by GT motion magnitude
  (`weight = 1 + w·min(gt_motion/ref, 1)`).
- `--loss-min-target-deg` — drop near-static pixels from the loss (active-only supervision).
- `analyze_differential.py` (+ `OSLORAFTDiff.confidence()`) — measures, from a saved
  checkpoint with no retraining, whether the LK flow carries any signal where the structure
  tensor is strong, and simulates hard confidence gating.

---

## 3. Experimental setup

- **Data:** GT mix of FLOW360 (primary, matches the benchmark), Replica 360 (static
  camera-motion, "easy"), MPF City (urban rendered) — three independent renderers, ~6–7k
  labeled full-360 pairs before augmentation. Conventions diagnosed in the sibling
  `sphereflow-dataprep` project (flow360 `identity`, replica360 `identity`, mpf `negated`).
- **Augmentation:** full-SO(3) rotation at sampling time, prob 1.0 (exact; no small-angle
  approximation — validated by `run_so3_diagnostic.py`).
- **Protocol (shakeout standard):** 5000 steps, AdamW, OneCycle peak LR 4e-4, batch 2, AMP,
  RTX 3090. ~33–43 min at r4, ~67 min at r5 all-pairs, ~3.2 h at r6 local.
- **Metrics:** the shared `spherical_flow.metrics` geodesic-degree pipeline, evaluated on a
  512-pair val subset. Reported as **% improvement of `global`/`active` geodesic error vs the
  zero-flow reference**, where `active>θ°` is the subset of pixels whose GT motion exceeds θ.

### 3.1 The motion-scale facts (this is the whole story)

The GT motion distribution over 25.2M samples: **p50 = 0.099°, p90 = 0.580°, p95 = 0.773°**.
HEALPix node spacing is `≈ √(4π/N)`: **r4 = 3.67°, r5 = 1.83°, r6 = 0.92°.** Motion in node
units:

| | p50 motion | p90 motion | active>0.5° | active>1.0° |
| --- | ---: | ---: | ---: | ---: |
| **r4** (3.67°) | 0.027 node | 0.16 node | 0.14 node | 0.27 node |
| **r5** (1.83°) | 0.054 node | 0.32 node | 0.27 node | 0.55 node |
| **r6** (0.92°) | 0.11 node | 0.63 node | 0.54 node | 1.09 node |

Only **~24%** of pixels move more than 0.25°; the other ~76% are near-static, for which
zero-flow is the correct answer. This is why the *global* metric is slightly negative for any
non-zero prediction while the *active* subsets are the honest signal.

**The half-node discriminability threshold:** a correlation argmax can only localize a match
when the displacement exceeds ~½ a node; below that, the argmax lands on the node itself and
produces no gradient. The active subsets only cross this threshold at **r6** — which is the
falsifiable prediction the ladder went on to test and refute.

---

## 4. The ablation ladder (the core result)

Every row is trained/evaluated under the same protocol; `active` numbers are % improvement over
zero-flow. The striking feature is the **invariance of the +2.9% ceiling** across radically
different geometry, until the two rows that break it (motion-weighting slides a bias tradeoff;
the differential collapses).

| # | Experiment | active>0.25° | active>0.5° | active>1.0° | global | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | Single-res **r4** (baseline) | **+2.90%** | +2.70% | +1.20% | −1.10% | learns; sub-node capped |
| 1 | Multi-res **est r4 → sup r6** | +2.82% | +2.58% | +1.03% | −1.05% | upsampler only interpolates |
| 2 | **est r5** all-pairs corr | +2.81% | +2.57% | +1.02% | −1.06% | half-node threshold uncrossed |
| 3 | **Local r6** correlation | +2.80% | +2.57% | +1.02% | −1.05% | **falsifies resolution hypothesis** |
| 4 | **`--ablate-corr`** (r4) | +2.90% | +2.69% | +1.19% | −1.10% | **correlation is INERT** |
| 5 | **`--ablate-context`** (r4) | +2.90% | +2.70% | +1.19% | −1.10% | **frame 2 is unused** |
| 6 | `--loss-motion-weight 4` (r4) | +3.33% | +3.12% | +1.38% | −2.71% | bias-tradeoff slide, not matching |
| 7 | **Differential (LK)** (r4) | **−4.43%** | −4.37% | −3.25% | −16.2% | **noise amplification** |

Rows 0–3 and 4–5 agree to within ~0.02 points despite different architectures (1.2M/1.40M/
1.26M/1.20M params) and different correlation grids — even the **per-step loss trajectory** is
near-identical across resolutions (step 100 = 0.2607 at both est5 and r6; step 2800 = 0.07420
vs 0.07422). The +2.9% ceiling is a property of the loss+data, not of any correspondence
mechanism.

### Reading each rung

**0 — Single-res r4 learns.** Marginally worse than zero-flow globally (−1.1%, expected: 76%
of pixels are static) but consistently beats it on active subsets over ~380k active pixels. It
is *not* collapsing to zero (that would give 0% on active). The architecture works; it is
resolution-capped because median motion is 0.027 node.

**1 — Multi-res doesn't lift it.** Estimating at r4 and convex-upsampling to r6 reproduces the
r4 ceiling exactly. The convex upsampler is a spatial *interpolator*: it redistributes the r4
flow to r6, it cannot *synthesize* sub-r4-node detail the r4 correlation never captured. The
ceiling is set by the estimation grid, not the supervision grid.

**2 — Finer correlation (r5) doesn't lift it.** Halving the spacing (3.67°→1.83°) moves
nothing, because at r5 the active subsets are *still* sub-node (>0.5° = 0.27 node, >1.0° = 0.55
node) — the same non-discriminative regime as r4. This sharpened the half-node threshold theory
and its falsifiable prediction: r6 should finally lift active>0.5°/>1.0°.

**3 — r6 refutes the resolution hypothesis.** With a lazy local `O(N·M·C)` correlation
(the 9.66 GB all-pairs volume is out of budget) and gradient checkpointing, r6 puts active>0.5°
at 0.54 node and >1.0° at 1.09 node — **above** the half-node threshold — and *still* moves
nothing. The prediction is refuted. The ceiling is not correlation resolution or locality; it
is shared by all three runs, so it lives in the loss + data.

**4 — The correlation is inert.** Zeroing the correlation feature entirely leaves
`active_0_25_geo` **bit-identical** (0.580 vs 0.580), same loss curve, same everything. The
model's central geometric claim — the exp-map + neighbor-grid lookup — does *no work*. It is not
matching frame 1 → frame 2; it regresses a smooth motion prior from the context net and flow
recurrence.

**5 — Frame 2 is unused.** Context = `cnet(frame1)` never sees frame 2, so the corr-off model
(row 4) is a **frame-1-only** model — and it already scores the full +2.9%. The mirror image
(context off, correlation is the only frame-2 signal) *also* collapses to +2.9%. Two disjoint
input pathways converge to the same flow field. **⇒ +2.9% is the ceiling of an appearance
prior; frame 2 — the entire basis of correspondence — contributes nothing.** This also kills the
"context shortcut masks a usable correlation" hypothesis: with the shortcut removed, correlation
alone still just reproduces the prior. The lever is not a training trick.

**6 — Motion-weighting only slides a bias tradeoff.** Up-weighting moving pixels buys +0.4 pts
of active (+2.9→+3.3) at the cost of global (−1.1→−2.7) — the classic global↔active bias slide
(a bigger prior helps big-GT pixels, hurts the static majority), not recovered correspondence.
Given row 4, this +0.4 is prior-scaling.

**7 — The differential estimator goes actively wrong.** See §5.

---

## 5. The differential head, and its independent confirmation

Discrete matching (correlation-argmax) needs multi-node displacement and went inert. The
*right* tool for sub-pixel motion is a **differential** estimator, where the linearization is
valid precisely because motion is small. `OSLORAFTDiff` implements a spherical Lucas–Kanade
solve on learned features `f = fnet(frame)`:

1. tangent-space spatial gradient of `f1` over the conv neighborhood, `G = (OᵀO)⁻¹Oᵀ Δf1`
   (parameter-free least-squares, fixed geometry);
2. temporal difference `Δf = f2 − f1`;
3. the feature-constancy solve `flow = (S + λI)⁻¹ r`, `S = Σ_c w_c gᵀg` (differentiable 2×2,
   run in fp32 for AMP safety).

Because flow depends on `f2` through `Δf`, the head **structurally cannot** fall back to a
frame-1 prior — it is the decisive test of whether frame 2 is usable at all.

**Synthetic recovery passes:** with linear features so `f2` is `f1` at a known sub-node
displacement, the operator recovers it to **0.55% median error, cos-sim 1.0000** — the math and
the sign (`G·flow = −Δf`, matching the dataset's `endpoint = p + flow`) are correct.

**On real data it underperforms zero-flow** (row 7: active −4.43%, global −16.2%, worse on
every eval, never recovered). The operator is provably correct, so the failure is the real-data
assumption: **one-shot LK amplifies noise.** The motion-induced part of `Δf` is only
`≈ (motion/feature-scale) ≈ 0.027` of the signal at r4; ~97% of `Δf` is non-motion difference
(content change, encoder noise, un-enforced feature constancy). The solve turns that into
spurious flow. The synthetic test passed only because linear features have perfect constancy and
no noise.

### 5.1 The confidence-gating diagnostic (closes the paradigm)

Before spending a retrain on confidence gating (the natural rescue — output flow only where the
structure tensor is well-conditioned), `analyze_differential.py` measured, from the saved
checkpoint, whether the LK flow carries *any* signal where the aperture is best-conditioned —
the best case gating could ever exploit. The answer is **worse than "no signal": the flow is
anti-correlated with motion, and most anti-correlated where it is most confident.**

**[1] Direction agreement — median cos-sim(LK, GT) on active pixels, by structure-tensor
confidence:**

| confidence subset | n | median cos | mean cos |
| --- | ---: | ---: | ---: |
| all active (top 100%) | 380,744 | **−0.072** | −0.023 |
| top 50% (λ ≥ 6.58) | 190,372 | −0.130 | −0.032 |
| top 25% (λ ≥ 15.8) | 95,186 | −0.145 | −0.037 |
| top 9% (λ ≥ 38.8) | 38,075 | −0.142 | −0.040 |
| top 5% (λ ≥ 64.4) | 19,038 | **−0.181** | −0.044 |

**[2] Simulated hard confidence gating — active improvement vs zero-flow (emit LK only on the
top-X% nodes, zero elsewhere):**

| keep top | active improvement | kept nodes that beat zero |
| --- | ---: | ---: |
| 100% | −4.43% | 45.4% |
| 50% | −1.32% | 46.3% |
| 25% | −0.65% | 46.1% |
| 10% | −0.24% | 45.8% |
| 5% | −0.11% | 45.6% |
| 2% | −0.04% | 46.0% |

Every gating fraction is negative, asymptoting to zero-flow *from below*; the kept nodes beat
zero-flow only 45–46% of the time — worse than a coin flip. LK magnitude (p50 0.069°) is *under*
GT (p50 0.099°), so the failure is **direction, not over-shoot**.

**Interpretation — an independent confirmation of the whole thesis.** Where features are sharp
(high λ_min), `Δf` is dominated by *appearance change, not motion*, so the confident solve fits
that structured non-motion difference to a flow anti-correlated with the truth. A completely
different estimator, on the same data, reaches the same conclusion the matching ladder did:
**frame 2 carries no usable correspondence at this grid.** Confidence gating is dead (no reliable
subset exists), and the remaining differential shots (r6, feature-constancy loss) would only be
pushing a paradigm whose confident predictions point the wrong way.

---

## 6. Conclusion

**Both correspondence paradigms fail, from opposite directions:**

- **Discrete matching goes inert** — the correlation is never used; a frame-1-only appearance
  prior reaches the full +2.9% active ceiling, invariant to grid resolution from r4 to r6.
- **Differential estimation goes actively wrong** — a Lucas–Kanade head underperforms zero-flow
  and is anti-correlated with the true motion, most strongly where it is most confident.

**Because the inter-frame motion this data affords is sub-node** (p50 0.099° ≈ 0.027 node at r4,
still sub-node at the finest affordable r6): below the resolving power of a discrete argmax
(needs > ½ node) and below the SNR floor of a one-shot differential (motion is ~3% of the feature
difference). A HEALPix-native RAFT therefore degenerates to a frame-1 appearance prior on this
data. This is a clean, mechanistically complete negative result, evidenced at every rung of the
ladder.

### 6.1 Why this is a thesis result, not a dead end

The negative result is *characterized*, not merely observed. Every alternative explanation was
tested and eliminated:

- not the supervision grid (row 1), not the correlation resolution or locality (rows 2–3),
- not a loss that under-weights motion (row 6), not a context shortcut hiding a usable
  correlation (row 5),
- not a bug in the differential operator (0.55% synthetic recovery), not a lack of confidence
  gating (§5.1).

Two independent estimators (matching, differential) converge on the same wall, and the wall is
quantified in node units. That convergence *is* the contribution: it establishes that the
bottleneck for spherical optical flow on this class of data is the ratio of inter-frame motion
to affordable grid spacing — a geometry-and-data limit, not an architecture-tuning problem.

### 6.2 Thesis mapping

Slots into the Ch. 4 three-act structure already in `OSLO_RAFT_PLAN.md` §8: **MVP negative
result → residual positive result → OSLO-RAFT.** OSLO-RAFT is the third act as an *analyzed
negative*: the frozen-RAFT + HEALPix residual remains the guaranteed positive contribution;
OSLO-RAFT contributes the ablation ladder, the sub-node characterization, and the SO(3)
equivariance / pole-seam geometry evidence. Recommended figures: the motion-in-node-units table
(§3.1); the ceiling-invariance table (§4); the cos-sim-vs-confidence curve (§5.1) as the money
figure for "the differential is anti-correlated where it is most sure."

---

## 7. Reproducibility

**Container:** `Dockerfile.oslo_raft` (base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) +
`docker-compose.oslo_raft.yml`; the `sfprep` shard reader is baked in, only the shard *data* is
mounted. Provenance: `OSLO_GIT_SHA=$(git rev-parse HEAD)`, `SHARDS_HOST=../sfprep/shards`.
Every run writes config + git hash + metrics JSON to `/outputs`.

Representative run commands (all under `docker compose ... run --build --rm oslo-raft`):

```bash
# Row 0 — single-res r4 baseline
python run_oslo_raft.py --grid healpix --resolution 4 ...

# Row 1 — multi-res est r4 → sup r6
python run_oslo_raft.py --grid healpix --resolution 6 --estimation-resolution 4 --multi-res

# Row 2 — est r5 all-pairs
python run_oslo_raft.py --grid healpix --resolution 6 --estimation-resolution 5 --multi-res

# Row 3 — local r6 correlation
python run_oslo_raft.py --grid healpix --resolution 6 --local-corr

# Rows 4/5 — ablations (r4)
python run_oslo_raft.py --grid healpix --resolution 4 --ablate-corr
python run_oslo_raft.py --grid healpix --resolution 4 --ablate-context

# Row 6 — motion-weighted loss
python run_oslo_raft.py --grid healpix --resolution 4 --loss-motion-weight 4

# Row 7 — differential head
python run_oslo_raft.py --grid healpix --resolution 4 --differential

# §5.1 — confidence-gating diagnostic (eval-only, from the saved diff checkpoint)
python analyze_differential.py --grid healpix --resolution 4 \
  --checkpoint /outputs/oslo_raft_r4_diff/oslo_raft.pt \
  --shards /data/shards --val-sources flow360:val --max-val-pairs 512 --device cuda
```

CPU-testable throughout via the healpy-free Fibonacci/`build_knn_level` path
(`--grid fibonacci`); every lever was validated on CPU before its GPU run.

---

## 8. What was not tried (and why it wouldn't change the conclusion)

- **r6 differential + feature-constancy loss.** Would raise the differential SNR ~4× and enforce
  the constancy the encoder violates — but §5.1 shows the confident predictions are
  *anti-correlated* with motion, so a better-conditioned version of the same solve pushes a
  paradigm already pointing the wrong way. Low expected value; high cost (r6 ≈ 3.2 h/run).
- **Auxiliary matching loss (InfoNCE on the lookup neighborhood).** Would give features a direct
  matchability gradient, but needs supra-half-node targets to be non-trivial — and r6 (row 3)
  already showed crossing the threshold moves nothing.
- **The full T1/T2 distillation + GT fine-tuning schedule** (`OSLO_RAFT_PLAN.md` §5). Every one
  of these runs starts from the same correspondence mechanism the ladder proved inert on this
  data; scaling steps does not create a signal the grid cannot resolve.

The productive next direction, if the model is to be revived, is **not** more architecture on
this data — it is *data with supra-node motion* (higher frame rate → larger displacement, or a
coarser effective grid) so that correspondence becomes resolvable in the first place. That is a
data-regime change, and it is out of scope for the negative result being reported here.
