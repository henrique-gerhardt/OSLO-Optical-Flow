# OSLO-RAFT Development Plan

This document is the working plan for the next stage of the project: building and training a complete spherical optical-flow model with its own weights ("OSLO-RAFT"), starting from SO(3) data augmentation and the multi-dataset foundation, and training it to perform well against the frozen corrected-RAFT baseline.

It assumes the current validated state described in [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md): the frozen-RAFT + HEALPix residual result is robust across seeds and is preserved as the project's safe baseline and fallback contribution. Nothing in this plan deletes or retrains that pipeline.

## 1. Goal

Train a HEALPix-native optical-flow model that:

- uses OSLO's SDPAConv as its core spatial operator;
- uses RAFT's estimation paradigm (feature correlation + iterative refinement), re-formulated on the sphere;
- is trained with SO(3) rotation augmentation, RAFT distillation, and the multi-dataset ground truth now on disk;
- is evaluated with the existing spherical metrics against zero-flow, the standalone MVP, frozen corrected RAFT, and RAFT + residual.

### Reference baselines (FLOW360 test, r=6)

| metric | zero-flow | MVP (3-seed mean) | frozen RAFT fwd | RAFT + residual fwd |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.4513 | 0.4423 | 0.2698 | 0.2669 |
| poles_geo_deg | 0.4831 | 0.4569 | 0.3420 | 0.3396 |
| seam_geo_deg | 1.0793 | 1.0859 | 0.8537 | 0.8466 |
| active_0_5_geo_deg | 1.8704 | 1.7696 | 1.0404 | 1.0368 |

### Decision gates

- **Gate 1 (after distillation pretraining, r=5):** the standalone OSLO-RAFT model, trained only on RAFT pseudo-labels plus SO(3) augmentation, must clearly beat the best standalone MVP (global 0.4423) and be within striking distance of frozen RAFT on active-motion subsets. If it cannot beat the MVP, the architecture is wrong; fix it before adding GT data.
- **Gate 2 (after GT fine-tuning, r=6):** approach frozen RAFT globally (within ~5-10%) while winning on poles and seam. Beating fine-tuned planar SOTA is explicitly not the target; the thesis claim is geometric, not leaderboard-based.

If Gate 2 fails, the thesis falls back to the residual result plus the geometry ablations, with OSLO-RAFT reported as a partially successful extension. The plan is ordered so that every phase produces thesis-usable material even if a later phase fails.

## 2. Data inventory

All paths under `/Volumes/External SSD/Mestrado/Datasets`.

| dataset | dir | content | role |
| --- | --- | --- | --- |
| FLOW360 (SLOF) | `FLOW360_train_test/` | 18 train + 11 test seqs, frames + fflows/bflows `.npy`, dynamic scenes | primary train + the held-out benchmark |
| Replica 360 (tangent-images dataset, arXiv 2301.11880) | `released/` | 54 seqs (18 scenes x circ/line/rand), RGB pano + fwd/bwd `.flo` + depth `.dpt` + masks | static camera-motion GT; curriculum "easy" data; cross-dataset eval |
| MPF City | `ECCV2022MPF-net_dataset/` | City_100_r, City_200_r, City_2000_r (images re-downloaded) | additional GT diversity (urban, rendered) |
| OmniFlow 512 | `omniflow_512/` | 320 seqs, 180-degree FOV, EXR flow | optional auxiliary only, behind hemisphere validity mask; skip unless data-starved |
| RAFT pseudo-labels | existing cache pipeline | unlimited (any 360 video) | distillation pretraining |

Approximate full-360 GT pair budget: FLOW360 ~1,500 x2 directions, Replica ~2,000, MPF City ~2,400. Total ~6-7k labeled pairs across three independent renderers, before SO(3) augmentation.

Known convention trap: every dataset may have its own flow sign/axis convention (FLOW360 needed `negated` forward / `identity` backward for RAFT). **No dataset enters training before passing a convention diagnostic** (Phase 0, Week 2).

## 3. Phase 0 — Data foundation (Weeks 1-3)

### Week 1: SO(3) rotation augmentation — IMPLEMENTED & VALIDATED (2026-06-14)

Done so far:
- `spherical_flow/shard_dataset.py` — the bridge from the standard shards to
  HEALPix-node samples (reuses the validated FLOW360 sampling path). `ShardFlowDataset`
  (streaming `IterableDataset`, worker-aware) + `load_shard_subset` (eager, for
  overfit/val). `sample_pair_to_nodes` exposes the `query_points` / `endpoint_rotation`
  seam. Verified: node-space photometric warp vs zero-flow — replica360 +86.4%,
  mpf +47.9% (conventions survive ERP→node resampling); flow360 inconclusive at
  0.28° motion (as expected, pinned). `run_shard_dataset_check.py`.
- `spherical_flow/so3_augment.py` — exact SO(3) augmentation (`rotation_matrix`,
  `yaw_matrix`, `sample_rotation`, `so3_augment_pair`). `run_so3_diagnostic.py`
  passes all four acceptance tests: identity 0.0, equivariance 9.5e-7, yaw-exactness
  4.6e-5, global metric invariance 0.034° while pole 1.46° / seam 6.12° shift.

Reference (original design below):


New module `spherical_flow/so3_augment.py`. Building blocks already exist in `spherical_flow/geometry.py`: `rotate_points`, `random_rotation`, `expmap`, `logmap`, `tangent_basis`, `points_to_equirectangular_pixels`.

Operation, for HEALPix node directions `p` ([N, 3]), ERP frames, ERP-derived target endpoints, and a rotation `R`:

1. Source directions: `q = p @ R` (apply inverse rotation to the sampling grid; fix one convention and test it, do not reason it twice).
2. Sample frame1/frame2 at `q` via `points_to_equirectangular_pixels` + bilinear interpolation (this is the existing HEALPix sampling path with rotated query directions).
3. Targets: take the GT spherical endpoints `e(q)` at the source directions, rotate them back: `e' = e(q) @ R.T`, then re-express as tangent flow at the unrotated nodes: `logmap(p, e')` in the standard basis at `p`. This is exact; no small-angle approximation.
4. Validity masks: nearest-neighbor resample at `q`.

The augmentation must be applied at dataset-sampling time (it needs ERP frames and GT, not pre-sampled HEALPix caches). Sampling distribution: uniform random axis, angle uniform in [0, 180] deg (full SO(3)), with a configurable `--so3-prob` (default 0.5-1.0) and `--so3-max-angle-deg`.

Acceptance criteria (write these as a runnable diagnostic, `run_so3_diagnostic.py`):

- **Identity test:** R = I reproduces the unrotated sample within float tolerance.
- **Round-trip test:** augment with R, then with R.T composed; tangent flow matches the original within interpolation tolerance (report max/mean geodesic discrepancy; expect < interpolation noise of one ERP pixel).
- **Yaw-exactness test:** a pure yaw by an integer multiple of the ERP column spacing equals an ERP column roll; should be near pixel-exact.
- **Metric invariance test:** evaluating zero-flow and cached-RAFT predictions on a rotated sample leaves `global_geo_deg` invariant within tolerance. `poles_*` and `seam_*` subsets are *not* invariant under rotation — that is expected and is itself a thesis figure (it demonstrates why ERP models have pole/seam pathologies and HEALPix+SO(3) does not).

### Week 2: dataset adapters + convention diagnostics — DONE (separate project)

Built as a standalone, citable project: **`../sphereflow-dataprep`** (not mixed
into this repo). It is ERP-space and `healpy`-free, normalizes N datasets into one
standard sharded format, and is config-driven (`datasets.toml`) so a new dataset
(e.g. FlowScape) is one adapter + one config block. Two layers keep the large data
local and regenerable while only a ~5 MB manifest is portable.

Pipeline: `python -m sfprep build` (index → manifest) → `diagnose --apply`
(detect flow convention) → `materialize` (normalized tar shards). The OSLO-RAFT
trainer consumes it via `sfprep.iter_split`, which yields ERP
`frame1/frame2/flow/valid/meta` — the boundary the HEALPix node sampling + Week-1
SO(3) augmentation wrap, producing `(frame1_nodes, frame2_nodes,
target_tangent_flow, valid_mask, meta)` so the trainer stays dataset-agnostic.

**Diagnosed conventions (built 2026-06-14, 14,009 pairs):** flow360 `identity`
(pinned — GT motion too small to diagnose photometrically), replica360 `identity`
(+83% vs zero, crisp), mpf `negated` (+14% vs zero). Splits: train 8,278 /
val 3,164 / test 2,567 (FLOW360 official test = the benchmark).

See `sphereflow-dataprep/README.md` for the format spec, the read contract, and
the "how to add a dataset" walkthrough.

### Week 3: distillation caches

- Reuse `run_flow360_cache_raft.py` to cache corrected-RAFT tangent flow for: FLOW360 train (exists), Replica 360 (all 54 seqs), MPF City. Use each dataset's diagnosed RAFT transform.
- These caches are the pseudo-label set for pretraining. GT is *not* used in this stage's labels, which also future-proofs the recipe for unlabeled real 360 video later.
- Storage estimate: r=6 tangent flow is 49,152 x 2 float32 ≈ 0.4 MB/pair; ~7k pairs ≈ 3 GB. Trivial.

Deliverable for Phase 0: a single mixed `SphericalFlowDataset` with per-dataset weights, SO(3) augmentation, and both GT and pseudo-label targets available per sample.

## 4. Phase 1 — OSLO-RAFT architecture (Weeks 3-6)

**Core IMPLEMENTED & smoke-passed (2026-06-14).** `spherical_flow/oslo_raft.py`:
siamese SDPAConv feature/context encoders, all-pairs cosine correlation, the
spherical exp-map + neighbor-grid lookup (§4.3), an SDPAConv `GraphConvGRU` with
zero-init delta head and tangent-plane flow composition (§4.4), and the γ-weighted
geodesic sequence loss. Pure tensor ops over a precomputed `SphereLevel`; the
healpy-free `build_knn_level` (directional-knn grid + brute-force ang2pix) makes it
CPU-testable now. `run_oslo_raft_smoke.py` passes all three §4.6 checks: 1.2M params
(in budget), forward/backward + finite grads, cold-start flow exactly 0.0 (zero-init
head), and **overfit of 10 replica360 pairs drove geodesic error 13.07°→2.39°**
(loss 0.84→0.13) — the model can fit, so lookup/GRU are wired correctly.

Still single-resolution (estimation == supervision grid). The explicit next increment
(needs the CUDA container + healpy): the nested-HEALPix multi-level builder for the
encoder downsampling pyramid (§4.1), the second-image correlation pyramid (§4.2), and
the convex HEALPix upsampler (§4.5). The model's forward signature already leaves room.

**Pyramid geometry foundation DONE (2026-06-17).** `spherical_flow/healpix_pyramid.py`:
exact nested index arithmetic (parent `i>>2`, children `4i..4i+3`, descendant blocks,
4-to-1 `pool_features`), a memory-bounded `chunked_directional_knn_graph` /
`chunked_nearest` (so r5/r6 conv/lookup grids build without a 9.7 GB `[N,N]`),
`SpherePyramid` + `build_healpix_pyramid` (per-resolution `SphereLevel`s + pooling /
descendant / upsample-neighbor maps), and `convex_upsample` (§4.5). CPU-validated via
`run_healpix_pyramid_smoke.py` (index bijections, chunked==unblocked kNN, convex-upsample
cold-start-zero / finite grads / transport consistency 2e-3 rad) **and validated on the box
(2026-06-17): tier 1.5 of `scripts/container_smoke.sh` builds the real r1-r4 pyramid and
asserts node counts `12·4^r`, nested-child proximity, and descendant sanity — green on the
RTX 3090.** (`_build_level` clamps the neighbor request to the level's node count so very
coarse correlation levels don't over-request.)

**Multi-resolution model WIRED (2026-06-17).** `spherical_flow/oslo_raft_pyramid.py`:
`OSLORAFTPyramid` consumes the pyramid — §4.1 `PyramidEncoder` (one `ResidualSphereBlock`
per resolution, nested `pool_features` between: r6→r5→r4, ch 32→64→96), §4.2
`build_correlation_pyramid` (all-pairs at r4, pool the second-image axis to r3/r2/r1),
§4.3 `pyramid_lookup` (one r4 endpoint, gather + concat each corr level's neighborhood),
§4.4 the reused SDPAConv `GraphConvGRU` + zero-init delta head at r4, §4.5
`UpsampleWeightHead` → `convex_upsample` lifting r4 flow to r6. **The loss is now genuinely
computed at r6 after upsampling.** Reuses all building-block modules from `oslo_raft.py`;
the single-res `OSLORAFT` is untouched. `run_oslo_raft.py` gains a `--multi-res` path
(`--estimation-resolution`, `--corr-pool-levels`; `--resolution` = fine grid) that samples
data at r6 and threads the pyramid through train/eval. 1.36M params (in budget), cold-start
flow exactly zero at r6 (RAFT contract end-to-end). CPU-validated via
`run_oslo_raft_pyramid_smoke.py` (synthetic nested pyramid: forward/cold-start/grads/budget)
and the single-res runner path re-checked for no regression; real-geometry + CUDA
forward/backward runs in-container as tier 1.6 of `scripts/container_smoke.sh`. Next: a
multi-res training run (the first test that estimate-coarse/supervise-r6 lifts the r=4
resolution ceiling).

Original design (reference):


New module `spherical_flow/oslo_raft.py`, runner `run_oslo_raft.py`, scripts `scripts/oslo_raft_*.sh`. Target parameter budget: **1-3M params** (RAFT-large is 5.3M; we have far less data, so stay smaller).

### 4.1 Feature and context encoders

- Siamese feature encoder: SDPAConv residual blocks on the HEALPix hierarchy, downsampling by nested 4-to-1 child pooling: r6 (input RGB + node xyz) -> r5 -> r4. Channels 32 -> 64 -> 96. GroupNorm.
- Context encoder: same topology on frame1 only; output split into GRU initial hidden state (tanh) and context features (relu), as in RAFT.
- Flow is estimated at r=4 (3,072 nodes) and upsampled to r=6 for the loss/metrics.

### 4.2 Correlation volume

- All-pairs correlation at r=4: `[3072, 3072]` = 9.4M entries — trivial memory.
- Correlation pyramid: pool the *second* image axis by HEALPix child averaging to r3, r2, r1 (RAFT pools 4 levels; nested HEALPix gives this for free).
- If estimating at r=5 instead (12,288 nodes): 151M entries ≈ 600 MB fp32 / 300 MB fp16 — fits the RTX 3090. r=6 all-pairs (9.7 GB) is out of budget; r=6 quality comes from the upsampler, not from a finer correlation grid (same trade RAFT makes with its 1/8-resolution volume).

### 4.3 Lookup operator (the spherical part)

Given current flow estimate `f_i` at node `i`:

1. endpoint `e_i = expmap(p_i, f_i)`;
2. center pixel `c_i = ang2pix(e_i)` at the correlation level's resolution;
3. gather the correlation values of `c_i`'s k-hop HEALPix neighborhood (reuse the OSLO neighbor grids; k=3 hops ≈ 7x7 RAFT lookup), at each pyramid level;
4. concatenate across levels -> per-node correlation feature.

This replaces RAFT's planar bilinear grid lookup with an exp-map + neighbor-grid lookup and is the central geometric claim of the model. Document it carefully; it is a thesis figure.

### 4.4 Iterative update block

- ConvGRU where every conv is an SDPAConv on the r=4 (or r=5) grid.
- Input: correlation feature, current tangent flow, context features. Output: delta tangent flow `[east, north]`, zero-initialized final layer (reuse the convention from `RaftResidualCorrector`).
- Iterations: 8-12 at train, up to 24 at eval (report eval-iters sweep).
- Flow update composes on the sphere: new endpoint = expmap from node through accumulated tangent flow (keep flow stored as tangent at the source node; deltas add in the tangent plane — valid for the small per-step magnitudes here, and consistent with how targets are encoded).

### 4.5 Upsampling head

- RAFT's convex upsampling adapted to nested HEALPix: each r=4 parent predicts softmax weights over its 1-hop neighborhood (9 nodes) for each of its 4^2 = 16 r=6 descendants; upsampled flow = weighted combination of coarse tangent flows, re-expressed in the fine node's tangent basis.
- **Implementation note (`convex_upsample`, 2026-06-17):** the transport is **parallel transport of the tangent flow**, not averaging absolute endpoints. Endpoint-averaging breaks the RAFT cold-start contract — at flow=0 the coarse endpoints are the coarse node directions, whose weighted average is *not* the fine node, injecting spurious flow. Transporting the tangent flow gives `parallel_transport(0)=0`, so a zero coarse flow upsamples to exactly zero for any weights (asserted in the smoke).

### 4.6 Smoke tests (before any real training)

- Forward/backward pass at r=4 on the synthetic-rotation generator (`run_spherical_flow_mvp.py` data path), batch 1, CPU-tolerable.
- Overfit test: 10 FLOW360 pairs, no augmentation; the model must drive training geodesic loss near zero. If it cannot overfit 10 pairs, debug before scaling.
- Equivariance check: predictions on an SO(3)-rotated input should approximately equal rotated predictions (SDPAConv is not exactly rotation-equivariant; measure and report the gap — also thesis material).

## 5. Phase 2 — Training (Weeks 6-10)

**Training harness READY & verified end-to-end (2026-06-15).** `run_oslo_raft.py`
wires `ShardFlowDataset` (streaming + SO(3) augmentation at sampling time) → OSLO-RAFT
→ `sequence_geodesic_loss` → the shared `spherical_flow.metrics` pipeline, writing an
aggregator-compatible metrics JSON + checkpoint with git hash. Grid-agnostic
(`--grid fibonacci` for CPU, `--grid healpix` for the GPU box). Has `--smoke-test`,
`--max-train-pairs`/`--max-val-pairs`, `--onecycle`, AMP, OneCycle, periodic eval.
Verified: full-batch overfit of 10 replica pairs at 768 nodes through the complete
runner drove val `global_geo_deg` 13.07°→2.53° (**+80.6%**, poles +70.9 / seam +76.5),
confirming the loop trains. (Note: on a too-coarse grid, e.g. 512 nodes over the full
set, the model correctly converges to ≈zero-flow because correspondences can't form —
not a bug; use a real estimation grid.) The zero-init flow head stalls step-0 upstream
gradient by construction (RAFT contract / cold-start = 0 flow) and unfreezes from step 1.

Caveat for the GPU box: `--num-workers>0` under spawn doesn't propagate per-epoch
`set_epoch` reshuffle into workers (shuffle buffer + shard order still randomize); fine
at 100k-step scale. Real HEALPix `--grid healpix` still uses the interim knn neighbor
grid + brute-force ang2pix until the nested-HEALPix builder lands.

**GPU container READY (2026-06-15).** Reproducible image so the project can move to
the box: `Dockerfile.oslo_raft` (base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`)
+ `requirements-oslo-raft.txt` (astropy-healpix — the cfitsio-free nested backend
`healpix_unit_vectors` already falls back to — pillow, opencv-headless, tqdm) +
`docker-compose.oslo_raft.yml` + `scripts/container_smoke.sh` + `OSLO_RAFT_DOCKER.md`.
The `sfprep` shard reader is **baked in** (Dockerfile clones
`github.com/henrique-gerhardt/sfprep` pinned to `SFPREP_REF`, no Python packaging so
clone not pip; `shard_dataset.py` finds it via `SPHEREFLOW_DATAPREP`); only the large,
regenerable shard *data* is mounted at runtime. The smoke is
two-tier: (1) data-free — build a real HEALPix level + one OSLO-RAFT forward/backward
on random frames (proves image/HEALPix/SDPAConv/CUDA), (2) if shards mounted, the full
`run_oslo_raft.py --grid healpix --smoke-test`. Validated locally as far as possible
without a GPU: `docker compose config` OK, requirements resolve OK, shell syntax OK,
tier-1 model wiring OK via the fibonacci stand-in (1.2M params, cold-start zero, finite
grads). The image build + GPU run themselves are validated on the box.

**Single-resolution shakeout DONE (2026-06-17, RTX 3090).** A GO/NO-GO de-risk run on the
current single-resolution model *before* investing in the multi-resolution pyramid: 5000
steps, r=4 (estimation == supervision), GT mix (replica360 + flow360 + mpf), SO(3) prob
1.0, OneCycle peak 4e-4, batch 2, AMP, ~36 min. **Result: the architecture learns on real
data at scale.** Final val (max 512 pairs): the model is marginally *worse* than zero-flow
globally (global_geo_deg 0.2093 vs 0.2070, −1.1%) but consistently *beats* zero-flow on
every active subset — active>0.25° +2.9% (0.580 vs 0.597), >0.5° +2.7%, >1.0° +1.2% —
stable across the last evals over ~380k active pixels. The negative global number is
expected, not a failure: `active_0_25_frac≈0.24`, so ~76% of pixels barely move and
zero-flow is the correct answer there; the model is **not** collapsing to zero-flow (that
would give 0% on active). **Key diagnostic — the case for the pyramid:** motion is sub-node
at r=4 (node spacing ≈3.67°; GT motion p50=0.099°, p90=0.58° → median flow ≈0.03 of a
node), so the single-resolution model physically cannot represent the fine flow that
dominates the GT. Node spacing vs p90 motion: r4 0.16 node, r5 0.32, r6 0.63 — only at r=6
does motion become node-resolvable, and the convex upsampler regresses the sub-node
remainder. This quantitatively motivates the estimate-coarse / supervise-at-r6 design.
Side note: val spiked transiently at the OneCycle LR peak (~4e-4) and recovered — add LR
warmup / lower peak for the long T1 run. (Possible later tuning: the geodesic loss is
dominated by near-static pixels, biasing toward zero-flow — consider a motion-weighted /
active-emphasized loss.)

**Multi-resolution shakeout DONE (2026-06-18, RTX 3090) — NEGATIVE RESULT.** Same protocol,
now `--multi-res --resolution 6 --estimation-resolution 4` (estimate at r=4, convex-upsample
+ supervise at r=6), 1.40M params, ~43 min. **The estimate-coarse/supervise-fine hypothesis
is falsified in this form: it reproduced the single-resolution r=4 ceiling almost exactly.**
Final val (512 pairs): active>0.25° **+2.82%** (vs single-res +2.9%), >0.5° **+2.58%** (vs
+2.7%), >1.0° **+1.03%** (vs +1.2%), global **−1.05%** (vs −1.1%) — statistically the same
point, if anything a hair worse. **Why:** the convex upsampler is a spatial *interpolator* —
it redistributes the r=4 flow field to r=6, it cannot *synthesize* sub-r4-node detail the r=4
correlation never captured. The resolution ceiling is set by the **estimation/correlation
grid (r=4), not the supervision grid (r=6)**; moving the loss to r=6 changes what is scored,
not what r=4 correlation can resolve. The r=4 all-pairs argmax lands on *self* for sub-node
motion (p50 0.099° = 0.027 node), so there is no discriminative gradient to upsample.
**Confirming evidence:** (a) the r=6 target distribution reads correctly (p50 0.099°, p90
0.580°, p95 0.773° over 25.2M samples) so the fine pipeline is sound; (b) the val@2000 spike
at the OneCycle LR peak (global −86%, active +5%) shows the model *can* emit large flow but
the static majority pulls it back to a near-zero solution as LR decays — the model is
signal-starved, not capacity-starved. **Implication — the next lever is the correlation
resolution, not the upsampler.** Motion as fraction of node spacing: even at r=6 the *median*
motion is only 0.11 node (sub-node everywhere affordable); only the active subsets (≥0.25°)
reach ~0.3–0.7 node at r5/r6. Next experiments, cheapest first: (1) `--estimation-resolution
5` — one flag, r5 all-pairs ≈1.2 GB (B=2, fits the 3090), 1.83° spacing; direct test of "does
a finer correlation grid lift the active-subset ceiling." (2) If r5 helps, the real fix is to
estimate at r=6 with a **local** correlation (O(N·K) neighborhood volume, not O(N²) all-pairs)
— since the match is always within a node's local neighborhood for sub-node motion, a local
r=6 correlation is both affordable *and* finer than all-pairs at r=4. (3) Orthogonal lever: an
active-emphasized loss so the static 76% stop dominating the gradient toward zero.

**est=5 follow-up DONE (2026-06-29, RTX 3090) — NEGATIVE, and it sharpens the theory.** Same
protocol, `--estimation-resolution 5` (r5 all-pairs correlation, 1.83° spacing), 1.26M params,
~67 min. Halving the estimation/correlation spacing (3.67° → 1.83°) moved nothing: final val
active>0.25° **+2.81%** (est=4 was +2.82%), >0.5° **+2.57%** (+2.58%), >1.0° **+1.02%** (+1.03%),
global **−1.06%** (−1.05%) — two independently-trained models on the *same* ceiling to <0.02
points, with the identical val@2000 LR-peak transient (global −86%, active_0_5 +5.2%). **Why r5
didn't help — the half-node discriminability threshold.** The all-pairs correlation argmax is
only discriminative when motion exceeds ~½ node; below that it lands on *self* and gives no
gradient. Active-subset motion as a fraction of node spacing: at r5 (1.83°) active>0.25° = 0.14
node, >0.5° = 0.27 node, >1.0° = 0.55 node — *all still sub-node*. So r5 never crosses the
threshold for the active subsets; it's the same non-discriminative regime as r4, hence the same
+2.8% (which is a small-mean-motion bias, not matching). The threshold is first crossed at **r6
(0.92°)**: there active>0.5° = 0.54 node and >1.0° = 1.09 node become resolvable (only >0.25° =
0.27 node stays sub-node). **Falsifiable prediction for experiment (2):** estimating at r6 should
lift active>0.5° and active>1.0° above their +2.6%/+1.0% ceilings while active>0.25° stays capped
— a signature no resolution ≤ r5 can produce. r6 all-pairs is out of budget (9.66 GB at B=1), so
this *requires* the local O(N·K) correlation (each node vs its ~K neighbors, [N,K] ≈ 9 MB —
trivial). That is now the clear next build; the est-grid sweep (r4, r5) is exhausted as a cheap
lever. Caveat on the orthogonal active-loss lever: the val@2000 +5% is the OneCycle *overshoot*
(large global bias happens to help the large-GT tail), not recovered matching — an active-emphasized
loss would likely just slide along the same global↔active bias tradeoff, not add correspondence, so
it is a diagnostic, not the fix. The fix is the local r6 correlation.

**Local r6 correlation BUILT + CPU-validated (2026-06-29).** `spherical_flow/oslo_raft_local.py`:
`OSLORAFTLocal` is the single-resolution `OSLORAFT` (same architecture/params, est==sup==r6 so
*no upsampler*) with one surgical change — the `[B,N,N]` all-pairs volume (9.66 GB at r6) is
replaced by a **lazy local lookup** (`local_correlation_lookup`): per node, exp-map to the
endpoint, a *windowed* ang2pix (argmax over the node's own `M`-nearest candidates — no global
`[N,N]` argmax), gather `f2` over the displaced neighborhood and dot with `f1`. Cost `O(N·M·C)`
≈ 0.5 GB instead of `O(N²)`; for bounded motion the gathered values are *identical* to
all-pairs-then-gather (the true nearest node lies in the lookup window). Wired as
`run_oslo_raft.py --local-corr` (single-res, builds the r6 level with the chunked `_build_level`,
never calls the brute `[N,N]` ang2pix). CPU checks (Fibonacci level, healpy-free): param parity
with `OSLORAFT` (byte-identical state_dict), cold-start flow exactly 0, **local lookup ≡ all-pairs
to 3e-8 wherever the two ang2pix agree (100% at flow=0, 99.7% at small flow)**, full
encoder→corr→GRU→head chain differentiable (after the zero-init head trains), and the end-to-end
loss+metrics loop runs. **Gradient checkpointing** (`use_checkpoint`, RAFT-style, on the per-iteration
update) is required: iterating the GRU/lookup over all ~49k r6 nodes stored an 8x activation stack
that OOM'd a 24 GB 3090 at B=2 (the 8 fp32 lookups alone are ~7 GB); checkpointing keeps only the
per-iteration boundary tensors and recomputes the block in backward, collapsing 8x→~1x — verified
bit-identical forward + gradients (78 param tensors, 0.00e+00) to the non-checkpointed path, and a
no-op under eval/no_grad. GPU run: `--grid healpix --resolution 6 --local-corr` (drop `--multi-res`).
**Decisive test:** if r6 lifts active>0.5°/>1.0° above the +2.6%/+1.0% all-pairs ceiling while
active>0.25° stays capped, correlation discriminability was the bottleneck — confirmed.

**r6 local correlation DONE (2026-06-30, RTX 3090, B=2, ~3.2 h) — NEGATIVE, and it falsifies the
resolution-ceiling hypothesis outright.** r4, r5 and r6 land on the *same* point: active>0.25°
**+2.80%** (r4 +2.82, r5 +2.81), >0.5° **+2.57%** (+2.58/+2.57), >1.0° **+1.02%** (+1.03/+1.02),
global **−1.05%**. Not just the same ceiling — the **per-step loss trajectory is near-identical**
across all three resolutions (step 100 = 0.2607 in est=5 and r6; step 2800 = 0.07420 vs 0.07422)
despite three different architectures (1.40M / 1.26M / 1.20M params) on three different correlation
grids, AND the val@2000 LR-peak overshoot is resolution-invariant too (active>0.5° = +5.19%/+5.19%/
+5.20%). Pushing the correlation grid to r6 — where active>0.5° = 0.54 node and active>1.0° = 1.09
node ARE above the half-node threshold — moved nothing. **Conclusion: the ceiling is NOT correlation
resolution or locality. It is shared by all three runs → it lives in the LOSS + DATA.** The geodesic
sequence loss is dominated by the ~76% near-static majority (active>0.25° is only 24% of pixels), so
every model — regardless of what its correlation can resolve — is pulled to the same conservative
near-zero solution. The val@2000 overshoot is the proof of headroom: at the LR peak the model emits
larger flow and reaches +5.2% on active>0.5° (with correct-enough direction to beat zero), but the
global loss penalizes the static majority and anneals it back to +2.6%; and that +5.2% peak being
*resolution-invariant* shows the global↔active operating curve is fixed by the loss, not the corr.

**Practical corollary: the ceiling is resolution-invariant, so all further experiments run at r4**
(~43 min, vs r6's 3.2 h) and transfer. **Next lever = the LOSS, not geometry.** `metrics.compute_loss`
ALREADY has the knobs (`loss_motion_weight` → up-weight pixels by GT motion magnitude;
`loss_min_target_deg` → drop near-static pixels from the loss) but `run_oslo_raft.py` calls
`sequence_geodesic_loss`, which ignores them — so the change is to fold a motion weighting into the
sequence loss + expose the flags. Hypothesis: a motion-weighted loss lets the model sit at the
+5%-active operating point the overshoot already reached, instead of annealing back. Cheap
confirmatory ablation first (one eval, no retrain): zero `corr_feat` at eval on the saved r6
checkpoint — if +2.8% active survives without correlation, the trained model is regressing a smooth
motion prior, not matching, and the lever is features/architecture; if it collapses, correlation is
load-bearing and the loss is the right amplifier.

**Ablation + loss runs DONE (2026-06-30, RTX 3090, r4 single-res, ~33 min each) — THE CORRELATION IS
INERT. This is the finding that reframes the project.** Run A (`--ablate-corr`, corr feature zeroed
before the motion encoder): active>0.25° **+2.90%**, >0.5° **+2.69%**, >1.0° **+1.19%**, global
**−1.10%** — *identical to the full model* (+2.9/+2.7/+1.2/−1.1), same per-step loss curve, same
val@2000 overshoot (active>0.5° +5.40%), `active_0_25_geo` bit-identical (0.580 vs 0.580). **Zeroing
the model's central mechanism — the spherical exp-map + neighbor-grid correlation lookup — changes
nothing. The model is not matching frame1→frame2; it regresses a smooth motion PRIOR from the context
net + flow recurrence.** This retroactively explains every prior negative: r4=r5=r6, est4=est5, the
near-identical loss curves, the resolution-invariant overshoot — the correlation grid is irrelevant
because the correlation is never used. Run B (`--loss-motion-weight 4`, corr on): active>0.25°
**+3.33%**, >0.5° **+3.12%**, >1.0° **+1.38%**, global **−2.71%**, overshoot +7.95% — a *modest*
active lift (+0.4 pts) bought with a global drop (−1.1→−2.7); the classic global↔active bias-tradeoff
slide (bigger prior helps big-GT pixels, hurts the static majority), NOT recovered correspondence.
**Root cause (hypothesis): sub-node motion + a cold-start bootstrap failure + an easy context
shortcut.** Correlation-argmax needs multi-node displacement to localize, but the motion is sub-node
at every affordable grid (p50 0.1° = 0.027 node at r4), so at flow=0 the corr profile is flat →
near-zero gradient to sharpen features via the corr path → features never become matchable, while the
context path offers an immediate "predict the mean motion" solution the optimizer locks onto. **The
lever is no longer loss or geometry — it is MAKING THE CORRELATION/MATCHING PATHWAY LEARN.** Immediate
airtight confirm: re-run B *with* `--ablate-corr` (2×2 completion) — if `B+ablate ≈ B` (+3.3%), even a
loss that demands motion accuracy leaves the correlation inert → the +0.4 is pure prior-scaling.
Then the real fork (attack "make the model estimate sub-node motion"): (a) **context-starvation
diagnostic** — shrink/remove the context feed to the GRU so the prior shortcut is unavailable; if the
model then engages correlation the shortcut was the blocker (fixable), if it collapses the features
fundamentally can't match (need arch); (b) **auxiliary matching loss** — supervise the correlation to
peak at the GT-displaced node (InfoNCE over the lookup neighborhood) so features get a *direct*
gradient to become matchable, independent of the flow-head bootstrap (needs r6 for non-trivial targets
since sub-node targets collapse to self); (c) a **differential/feature-difference flow head** — regress
sub-node displacement from local feature gradients rather than a correlation argmax (the right tool for
sub-pixel motion).

**Context-starvation DONE (2026-06-30, r4, `--ablate-context`) — it sharpens the finding to its final
form: the model does not use frame 2 at all.** Zeroing the entire context net (GRU hidden init + feed)
so only correlation + recurrence drive flow gave active>0.25° **+2.90%**, >0.5° **+2.70%**, >1.0°
**+1.19%**, global **−1.10%** — *identical to the full model and to the corr-ablated Run A*
(`active_0_25_geo` = 0.5799 across all three, matching to 5–6 sig figs). Two disjoint input pathways
converge to the *same flow field*. **The decisive point: context = `cnet(frame1)` never sees frame 2,
so Run A (corr off, context on) is a FRAME-1-ONLY model — and it already scores the full +2.9%.** A
model with zero frame-2 information matches the complete result; and the mirror (context off, so
correlation is the only frame-2 signal) *also* collapses to that same +2.9%. **⇒ The +2.9% is the
ceiling of an appearance PRIOR ("given frame 1, predict where things usually move"); frame 2 — the
entire basis of correspondence — contributes nothing.** This kills the "context shortcut masks usable
correlation" hypothesis (we removed the shortcut; correlation-only still just reproduced the prior), so
the lever is NOT training tricks. Root cause, now firmly: the inter-frame motion (p50 0.1°) is sub-node
at every affordable grid, and correlation-argmax cannot resolve sub-node displacement — no correspondence
signal is extractable at any resolution (why r6 tied r4). The 2×2 is effectively closed; `B+ablate` is
now a redundant confirm (would give ≈+3.3%, motion-loss's prior-scaling with corr off).

**Where this leaves the project — one principled lever, and a clean fallback thesis result.** The only
remaining mechanism that can extract SUB-node motion is a **differential (Lucas–Kanade-style) flow head**:
estimate flow from the temporal feature difference `f2 − f1` and the tangent-space spatial gradient of
`f1` (`flow ≈ −(∂f/∂t)/(∂f/∂x)`), which resolves sub-pixel displacement via continuous gradients rather
than a discrete argmax — the right tool for exactly this regime. It is the decisive test of "is sub-node
motion extractable at all here": if it beats +2.9%, correspondence is recoverable and the differential
operator is the contribution; if it too ties the frame-1 prior, that CONFIRMS the motion is fundamentally
sub-resolution for affordable HEALPix grids — itself a legitimate, well-characterized thesis result
(a HEALPix-native RAFT degenerates to an appearance prior because the data's motion is sub-node, with
the full ablation ladder — resolution r4/r5/r6, loss reweighting, corr/context starvation — as evidence).

**Differential head BUILT + CPU-validated (2026-06-30).** `spherical_flow/oslo_raft_diff.py`:
`OSLORAFTDiff` = the `fnet` feature encoder + a spherical Lucas–Kanade solve, no correlation / GRU /
context. Per node: parameter-free least-squares tangent gradient of `f1` over the conv neighborhood
(`tangent_gradient_operator`, fixed geometry), temporal difference `Δf = f2 − f1`, and the
feature-constancy solve `flow = (S + λI)⁻¹ r` (`S = Σ_c w_c gᵀg`, differentiable 2×2, run in fp32 for
AMP safety). Learnable: `fnet`, per-channel weight `w_c`, `λ`, output scale (~214k params — far leaner
than the ~1.2M correlation model; `fnet` carries the capacity). One-shot (sub-node ⇒ LK linearization
valid, no warping). Wired as `run_oslo_raft.py --differential` (single-res, mutually exclusive with the
other modes; nothing to ablate). CPU checks: **synthetic RECOVERY — with linear features so `f2` is
`f1` sampled at a known sub-node displacement, the operator returns it to 0.55% median error, direction
cos-sim 1.0000** (proves the math; the dataset convention `endpoint = p + flow` ⇒ `G·flow = −Δf` sign
is correct); end-to-end grads into all params; train+metrics loop runs; AMP/autocast forward finite.
GPU run: `--grid healpix --resolution 4 --differential`. Decisive: beats +2.9% ⇒ frame 2 is finally
used, sub-pixel correspondence recovered (the contribution); ties/underperforms ⇒ the sub-node
limitation is intrinsic (the characterized negative result stands).

Loss: iteration-weighted geodesic endpoint loss, the spherical analogue of RAFT's sequence loss:

```text
L = sum_t  gamma^(T-t) * mean_valid( geodesic(expmap(p, f_t), gt_endpoint) ),  gamma = 0.8
```

masked by validity, computed at r=6 after upsampling.

### Stage T1 — distillation pretraining (Gate 1)

```text
data:        RAFT pseudo-label caches (FLOW360 train + Replica + MPF), no GT
aug:         SO(3) full-sphere, prob 1.0; photometric jitter; fwd/bwd both
resolution:  r=5 estimation (r=4 first if memory/debug demands), r=6 supervision
steps:       ~100k, AdamW, OneCycle LR peak 4e-4, weight decay 1e-5, grad clip 1.0
batch:       2 (fp16 correlation if needed)
seeds:       7 (gate decision), then 11/19 for the kept config
budget:      ~1-1.5 days on the RTX 3090 per run
```

Evaluate on FLOW360 test with the standard runner. **Gate 1 decision here.** Expected outcome: distillation alone should land between the MVP (0.4423) and frozen RAFT (0.2698) globally; the model cannot beat its teacher globally from distillation alone, and is not expected to.

### Stage T2 — ground-truth fine-tuning (Gate 2)

```text
data:        mixed GT — FLOW360 train, Replica 360, MPF City; sampling weights
             roughly proportional to sqrt(pairs) per dataset, FLOW360 boosted 2x
             (it matches the test domain)
curriculum:  first 20% of steps oversample Replica (static, camera-only = easy),
             then anneal to the standard mix
aug:         SO(3) prob 0.7-1.0 (ablate), photometric jitter
steps:       ~50-100k, LR peak 1.25e-4 (RAFT's fine-tune ratio), init from T1
seeds:       7/11/19 for the final config
```

Evaluate r=6 on FLOW360 test, all metric groups, direction-split. **Gate 2 decision here.**

### Stage T3 (optional, only if Gate 2 passes) — hybrid head

Condition OSLO-RAFT's initial flow on the cached RAFT tangent flow instead of zero-init (i.e., the model becomes a deep iterative residual over RAFT). This unifies the thesis: the residual corrector is the 1-step special case, OSLO-RAFT-from-zero is the standalone case, and this is the full hybrid. Cheap to run (same weights, warm-started flow) and gives the thesis a clean three-way comparison.

### Engineering notes

- Pre-sample HEALPix node frames into a cache (like the RAFT cache) if external-SSD I/O bottlenecks the GPU; node sampling of ERP frames is the per-step cost most worth amortizing. Caveat: SO(3) augmentation needs ERP-space sampling, so cache the *decoded ERP frames* (e.g., as fp16 npy or via an LMDB) rather than pre-sampled nodes.
- Log per-dataset validation curves separately; a mixed metric hides a dataset whose convention is wrong.
- Keep every run's config + git hash in the output dir (existing convention).

## 6. Phase 3 — Evaluation and ablations (Weeks 10-12)

Headline table — all on FLOW360 test, r=6, seeds 7/11/19, direction forward and both:

```text
zero-flow | MVP | frozen RAFT | RAFT+residual | OSLO-RAFT (T1) | OSLO-RAFT (T2) | hybrid (T3)
```

Required ablations (each one answers a referee question):

1. **No SO(3) augmentation** in T2 — quantifies the augmentation's contribution; expected to be the largest single factor given ~7k pairs.
2. **No distillation** (T2 recipe from random init) — quantifies the pseudo-label pretraining.
3. **Planar operator swap**: replace SDPAConv with a plain per-node MLP over the same neighbor grid (or a 2D conv on the unfolded grid) — proves the gain is the spherical operator, not parameter count. This is the thesis-critical geometry ablation and applies to the residual corrector too.
4. **Iteration sweep** at eval (1, 4, 8, 12, 24) — shows the recurrent update matters.
5. **Resolution**: r=4 vs r=5 estimation grid.
6. **Cross-dataset holdout**: train T2 without Replica, evaluate on Replica (and the same for MPF) — generalization evidence no current 360-flow paper using a single dataset can show.

Plus the Phase-1-consolidation items inherited from the handoff doc, which serve this phase directly: result aggregation script (build it in Week 1 alongside the augmentation — it is needed for every gate decision), direction-split evaluation, error-map visualization (pole/seam maps comparing ERP RAFT vs OSLO-RAFT are the money figures).

Standard-metric parity: also report pixel EPE and spherical EPE (SEPE) alongside the geodesic-degree metrics so results are comparable to SLOF/PanoFlow/PriOr-Flow numbers.

## 7. Risks and contingencies

| risk | signal | contingency |
| --- | --- | --- |
| Dataset convention bug | per-dataset val loss anomalous; diagnostic ambiguous | Week-2 diagnostics are blocking; never train on an undiagnosed dataset |
| Correlation memory at r=5 | OOM at batch 2 | fp16 correlation; batch 1 + grad accumulation; fall back to r=4 |
| Model can't overfit 10 pairs | smoke test fails | bug hunt in lookup/upsampler before any schedule tuning; check expmap/ang2pix round trip |
| Gate 1 fails (worse than MVP) | T1 eval | inspect correlation lookup first (most novel part); try r=4; only then revisit architecture |
| Gate 2 fails (far from RAFT) | T2 eval | ship thesis on residual + ablations; report OSLO-RAFT as analyzed negative/partial result with the equivariance and pole/seam evidence |
| SSD I/O starves GPU | low GPU util | ERP frame cache (engineering note above) |
| Time overrun | calendar | T3 and ablations 4-6 are cuttable; ablations 1-3 are not |

## 8. Thesis mapping

- Ch. 3 (method): geometry (Sec 4.3 lookup, SO(3) augmentation derivation), architecture.
- Ch. 4 (experiments): MVP negative result -> residual positive result -> OSLO-RAFT, the three-act structure already in the docs.
- Ch. 5 (analysis): ablations 1-6, equivariance measurement, pole/seam error maps, cross-dataset generalization.
- The residual result stays the guaranteed contribution; OSLO-RAFT is the headline if Gate 2 passes.

## 9. Immediate next actions (this week)

1. `run_aggregate_results.py` — DONE. Discovers any metric block (`global_geo_deg`)
   under a root, collapses seeds to mean±std, derives the zero-flow reference, emits
   the headline Markdown/CSV. Future OSLO-RAFT runners appear automatically.
2. `spherical_flow/so3_augment.py` + `run_so3_diagnostic.py` — DONE, four tests pass.
3. Verify the re-downloaded `City_2000_r/image` pairing — DONE in sphereflow-dataprep.
4. Replica 360 adapter + convention diagnostic — DONE in sphereflow-dataprep.

Remaining before Phase 1: Week 3 distillation caches (needs GPU/container), then the
OSLO-RAFT architecture (`spherical_flow/oslo_raft.py`) + overfit-10-pairs smoke test.
