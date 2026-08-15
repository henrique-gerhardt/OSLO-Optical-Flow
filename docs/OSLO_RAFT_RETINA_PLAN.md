# OSLO-RAFT-R: Decoupling the Retina from the Estimation Grid

<!-- SCOPE-BANNER -->
> **STATUS 2026-08-14 — read `docs/plans/LITERATURE_SCOPE.md` before quoting anything here.**
> Two independent corrections apply across this project's documents. (1) The FLOW360
> forward-flow convention was inverted, so every FLOW360 result recorded before
> 2026-08-04 is void. (2) A literature check on 2026-08-14 found that several things
> treated here as ours already exist in print: the geodesic (SEPE) metric, polar/equatorial
> stratification, rotation-robustness evaluation in panoramic vision, and matched-backbone
> comparison across panoramic representations. `LITERATURE_SCOPE.md` is the register of what
> we may and may not claim.
>
> **This file specifically:** FLOW360-derived conclusions are VOID.


**Status: implemented (2026-07-02) — §11 steps 1–9 landed, and the decisive §9.2d gate
PASSED: held-out sub-node (0.3·s_est) rotation recovery at direction cos-sim 0.997 with the
corr-ablated control failing at −0.02 — the first variant in this project whose correlation
is load-bearing.** CPU validation runs via `run_oslo_raft_retina_smoke.py`; real-geometry +
data validation via `scripts/container_smoke.sh` tiers 1.65/1.7/2.7; the Stage A training
command is in `OSLO_RAFT_DOCKER.md`. This is the improvement plan that follows from the
post-mortem of the negative result in [`OSLO_RAFT_DOCS.md`](OSLO_RAFT_DOCS.md). It specifies a
new model variant, **OSLO-RAFT-R** ("retina"), plus the data-seam, geometry, runner, and test
changes needed to build it. It is written to be implementable by an agent without further
design decisions; every section names the files, functions, and acceptance criteria involved.

Nothing in this plan deletes or invalidates the ablation ladder. The negative result stands —
*for models whose input grid is their estimation grid*. This plan tests the sharpened
interpretation of that result (§1) by removing the confound.

---

## 1. Why: the post-mortem in one page

The ladder concluded "sub-node inter-frame motion is below the resolving power of
correspondence." Comparing against how the successful 360° pipelines actually work
(PanoFlow, PriOr-Flow, tangent-images, MPF-Net — all wrap an *unchanged pixel-raster RAFT*)
exposes a confound in that conclusion:

**RAFT's correlation grid is coarse, but its retina is not.** RAFT ingests every full-res
pixel; the 1/8-grid features are deep nonlinear functions of full-res content, so a sub-cell
shift of the underlying texture measurably changes them. Its lookup samples the correlation
volume *bilinearly at continuous coordinates*, and the GRU regresses continuous deltas.
RAFT is a sub-pixel regressor, not an argmax matcher.

The numbers on our own benchmark (measured 2026-07-01 from the raw dataset files; 20 random
flow files per dataset):

| dataset | native raster | \|flow\| px p50 | p90 | p99 |
| --- | --- | ---: | ---: | ---: |
| FLOW360 (the val/benchmark set) | 512×1024 (0.352°/px eq.) | 0.64 | 3.7 | 32 |
| Replica 360 | 640×1280 (0.281°/px eq.) | 36.5 | 136 | ~1234 (pole/seam inflated) |
| MPF City | 512×1024 | 6.2 | 42 | 254 |

- RAFT's 1/8 correlation cell on FLOW360 is **8 px ≈ 2.81°** — *coarser in angle than our r5
  (1.83°) and r6 (0.92°) correlation grids*. Median FLOW360 motion is **0.08 of a RAFT corr
  cell**, p90 is 0.46 cell — below the "half-node threshold" — and frozen RAFT still improves
  active_0_5 by 44% (1.87°→1.04°) on our metrics. The correlation grid was never the
  bottleneck.
- What differs is the **input**: RAFT's retina is 512×1024 = 524k samples (≈ HEALPix **r8**
  density; r8 = 786,432 nodes, 0.229° spacing). Our models ingested 3,072 (r4) to 49,152 (r6)
  node samples — a 10–170× input downsample applied *before* the encoder could encode
  sub-cell phase.
- The SNR ladder matches: the motion share of the temporal feature difference was measured at
  ~2.7% (r4) and ~11% (r6); at ~1-px pitch the same ratio is ~60% at the median and >100% on
  active pixels. The LK head that is hopeless at 3% SNR is classical vision's workhorse at
  60%+.
- Our §4.3 lookup does `ang2pix` (nearest-node snap) + discrete gather: the correlation
  feature is **piecewise-constant in sub-node flow** — mechanically consistent with the
  correlation going inert.
- Every published net also arrives with matching *pre-bootstrapped* on large-motion synthetic
  data (FlyingChairs/Things, tens of px); ours trained matching from scratch on a benchmark
  where matching is never obviously useful, so it never bootstrapped.

**Reframed root cause:** not "sub-node motion is unresolvable", but "*a spherical model whose
retina is its estimation grid cannot see sub-node motion*." RAFT decouples retina (1/1) from
correlation grid (1/8); OSLO-RAFT fused them.

**Scoping fix to carry into the thesis:** the §3.1 motion table in `OSLO_RAFT_DOCS.md`
(p50 0.099°) is `flow360:val` only. Replica (p50 ≈ 13° in node space — the overfit smoke's
initial error) and MPF (p50 ≈ 2.2° eq.) are supra-node at r4/r5. (Also: the Replica renders
come from the tangent-images paper arXiv **2112.14331**; arXiv 2301.11880 is the FLOW360
journal paper — fix the citation in `OSLO_RAFT_PLAN.md` §2.)

### The three fixes, mapped to RAFT

| RAFT mechanism | OSLO-RAFT (ladder) | OSLO-RAFT-R (this plan) |
| --- | --- | --- |
| Full-res retina → coarse features | input sampled *at* the estimation grid | **§3 retina grid r7/r8**, encoder pools to r_est |
| Bilinear corr-volume sampling | `ang2pix` snap + discrete gather | **§5 interpolated lazy lookup** (continuous in flow) |
| Chairs/Things large-motion pretraining | from scratch on sub-node data | **§8 matching bootstrap** (synthetic rotations + Replica first) |

---

## 2. Architecture overview

Three grids instead of two:

```
retina r_ret (default 7; stretch 8)   frames sampled here; encoder stem runs here
        │  PyramidEncoder: SDPAConv block per level + nested 4-to-1 pool_features
        ▼
estimation r_est (default 4; later 5) correlation, lookup, ConvGRU, delta head
        │  UpsampleWeightHead + convex_upsample (unchanged)
        ▼
supervision r_sup (stays 6)           loss + metrics (unchanged; benchmark-comparable)
```

Constraints: `r_est < r_sup <= r_ret`. With the defaults (7/4/6) the encoder chain
r7→r6→r5→r4 passes through r_sup, so all needed levels exist in one pyramid.

| component | status | source |
| --- | --- | --- |
| Encoder over r_ret..r_est | **reuse** | `PyramidEncoder` (`oslo_raft_pyramid.py`) + per-stage gradient checkpointing (new flag) |
| Correlation | **new** | lazy interpolated pyramid lookup (§5) — replaces `AllPairsCorrelation` + `build_correlation_pyramid` + `pyramid_lookup` |
| GRU / motion encoder / zero-init flow head | reuse | `GraphConvGRU`, `MotionEncoder`, `flow_conv1/2` (`oslo_raft.py`) |
| Upsampler r_est→r_sup | reuse | `UpsampleWeightHead` + `convex_upsample` (`healpix_pyramid.py`) |
| Loss / metrics | reuse | `sequence_geodesic_loss` at r_sup, `spherical_flow.metrics` |
| Iteration checkpointing | reuse pattern | `OSLORAFTLocal._update_step` + `torch.utils.checkpoint` |
| `--ablate-corr` / `--ablate-context` | **must keep working** | they are the decisive diagnostics (§8 gates) |

New module: `spherical_flow/oslo_raft_retina.py` (`OSLORAFTRetina`). Existing models stay
untouched (same convention as every previous variant).

Cold-start contract (non-negotiable, asserted in smoke): zero-init delta head ⇒ flow ≡ 0 at
every iteration at init ⇒ `convex_upsample(0) = 0` ⇒ `preds[0]` exactly zero at r_sup.

---

## 3. Geometry: pyramid with a retina range (`spherical_flow/healpix_pyramid.py`)

### 3.1 `build_healpix_pyramid` extension

Add `retina_resolution: Optional[int] = None` (default `None` → behaves exactly as today,
`retina == fine`). Changes when set:

- `encoder_resolutions = range(estimation_resolution, retina_resolution + 1)` (the encoder
  chain now spans est..retina instead of est..fine).
- `corr_resolutions`, `descendant_index` (est→fine), `upsample_neighbors`: **unchanged** —
  supervision still lives at `fine_resolution`.
- `SpherePyramid` gains a `retina_resolution: int` field (set = `fine_resolution` when the
  arg is None, so existing callers/`.to()` code stay valid) and a `retina_level` property.
- `pool_index` already covers "every r with r+1 present" — no change needed; the encoder
  chain provides r_ret..r_est contiguously.

### 3.2 Cheap levels above `fine_resolution`

Levels used only by the encoder (r7, r8) need `points`, `basis_*`, `conv_*` — **not**
`lookup_index`/`ang2pix`. Add a `lookup_neighbors_override: dict[int, int]` or simpler: in
`_build_level`, accept `lookup_neighbors=0` meaning "trivial lookup" and set
`lookup_index = arange(N).unsqueeze(1)` ([N,1] self-index — keeps the dataclass shape and
skips a `chunked_nearest` pass and ~150 MB of int64 at r8). `build_healpix_pyramid` passes 0
for every level above `fine_resolution`.

### 3.3 Fast neighbor graph at r7/r8

`chunked_directional_knn_graph` is O(N²) and its default `chunk_size=2048` allocates a
`[2048, N]` similarity block — **6.4 GB at r8**. Two changes:

1. **Auto-scale the chunk**: `chunk = max(64, int(mem_budget_bytes / (4 * N)))` with a 256 MB
   default budget. This alone makes r8 *possible* (but slow: O(N²) ≈ 6×10¹¹ MACs).
2. **Preferred fast path** — `healpix_neighbor_graph(resolution)` (new function): use
   `astropy_healpix.neighbours(idx, nside, order='nested')` to get each pixel's 8 HEALPix
   neighbors in O(N), then sort them by local tangent angle exactly as
   `directional_knn_graph` does (project neighbor directions onto the node's
   `tangent_basis`, `atan2(north, east).argsort`). A handful of pixels have only 7 neighbors
   (`neighbours` returns −1 there): replace −1 with the node's own index and set
   `conv_valid=False` for that slot — `SDPAConv` already honors `valid_index`.
   Use this path for any HEALPix level with `N > 100_000`; keep chunked kNN as the generic
   fallback and for Fibonacci grids. **Parity test:** at r4 the angle-sorted 8-neighborhood
   from `neighbours` must be compared against `chunked_directional_knn_graph` — allow set
   differences (kNN's 8 nearest ≠ HEALPix's 8 topological neighbors at some pixels) but
   assert ≥95% of nodes have identical neighbor *sets*, and run one training-smoke parity
   check (loss curve within noise) before adopting it silently for lower levels. If parity
   is poor, restrict the fast path to r≥7 (where the kNN path is unaffordable anyway).
   *Measured 2026-07-02 (container tier 1.7a): identical sets 77.6% at r4 — kNN and the
   topology genuinely disagree on the marginal 8th neighbor — so the fallback clause
   applies: the fast path stays restricted to `N > 100_000` (r7+), and the tier asserts
   the properties SDPAConv actually needs (valid rows, ≥7 neighbors/pixel, ≥85% per-node
   overlap with the nearest set, all neighbors within 2.5× the cell scale).*

### 3.4 Pyramid disk cache

Building the r7/r8 graphs takes minutes; don't pay it per run. Add:

- `save_pyramid(pyramid, path)`: `torch.save` a plain-tensor dict — per level
  `{points, basis_east, basis_north, conv_index, conv_weight, conv_valid, lookup_index}`
  plus the pyramid maps and a config dict `{resolutions, conv_neighbors, lookup_neighbors,
  version}`. **Do not pickle `SphereLevel`** (the `ang2pix` closure is unpicklable).
- `load_pyramid(path) -> SpherePyramid`: rebuild each `SphereLevel`, reconstructing the
  standard brute `ang2pix` closure over the loaded `points` (same code as `_build_level`).
- Runner flag `--pyramid-cache PATH` (default `outputs/pyramid_cache/`): key the filename on
  the config tuple; build+save on miss, load on hit, rebuild if `version` mismatches.

---

## 4. Data seam: frames at the retina, targets at the supervision grid

### 4.1 `sample_pair_to_nodes` (`spherical_flow/shard_dataset.py`)

Frames and targets currently share one `points` grid. Add optional target-grid arguments
(backward compatible — `None` reproduces today's behavior bit-for-bit):

```python
def sample_pair_to_nodes(
    frame1_erp, frame2_erp, flow_erp, valid_erp,
    points, basis_east, basis_north,           # FRAME grid (retina)
    query_points=None, endpoint_rotation=None,
    target_points=None,                         # TARGET grid (supervision); None -> points
    target_basis_east=None, target_basis_north=None,
    target_query_points=None,                   # rotated targets under SO(3); None -> target_points
) -> Dict[str, torch.Tensor]:
```

Implementation: compute `u, v` twice — frame dirs → sample `frame1`/`frame2` only; target
dirs → sample `flow`/`valid`, build `endpoint` (+ rotation) and `logmap` at `target_points`.
Returned dict: `frame1`/`frame2` are `[N_ret, 3]`; `flow`/`endpoint`/`valid` are `[N_sup, ·]`.
The batch collater needs no change (per-key stacking already works with different N per key).

### 4.2 `so3_augment_pair` (`spherical_flow/so3_augment.py`)

Gains the same optional `target_points/target_basis_*` args and passes
`target_query_points = target_points @ rotation` (the same `R` used for the frame grid).
The equivariance property is per-grid; the existing diagnostic still applies.

### 4.3 `ShardFlowDataset` / `load_shard_subset`

Both gain `target_points: Optional[Tensor] = None`; precompute both tangent bases; thread
through `_sample_record` and `_augment_record`. `run_so3_diagnostic.py` and all existing
callers pass nothing and are unaffected.

### 4.4 Synthetic-rotation motion source (the bootstrap data, §8 Stage A)

New helper in `shard_dataset.py` (pattern from `synthetic.py`'s
`SyntheticRotationFlowDataset`, but with *real ERP texture*):

```python
def synth_rotation_record(frame1_erp, frame_points, frame_basis, target_points,
                          target_basis, rotation) -> Dict[str, torch.Tensor]
```

- `frame1` = sample `frame1_erp` at `frame_points` (as usual).
- `frame2` = sample `frame1_erp` at `frame_points @ rotation` — i.e. frame 2 *is* frame 1
  seen after the world rotates by `R`: perfect brightness constancy, exact correspondence.
- targets: `endpoint = rotate(target_points, R)` (careful: match the convention used by
  `SyntheticRotationFlowDataset` — endpoint is the rotated node, frame2 sampled at the
  inversely-rotated direction; reuse `rotate_points` and copy its sign conventions
  verbatim), `flow = logmap(target_points, endpoint, target_basis...)`, `valid` = all True.
- `ShardFlowDataset` gains `synth_rot_prob: float = 0.0`, `synth_rot_min_deg`,
  `synth_rot_max_deg`: with that probability a record's frame2/targets are *replaced* by a
  synthetic rotation of its frame1 (angle uniform in [min, max], axis uniform). Composes
  with (applies after) the SO(3) *viewpoint* augmentation — they are different things:
  SO(3) aug rotates the sampling of a real pair; this synthesizes the *motion* itself.
- This gives unlimited, exact, arbitrary-magnitude motion on real imagery — our
  FlyingChairs. It exercises matching by construction (appearance identical across frames).

### 4.5 Throughput note

Per pair at r8: 2 × [786k, 3] frame tensors ≈ 19 MB, ~1.6M bilinear ERP lookups on a CPU
worker (r7: ¼ of that). Expected fine with `--num-workers 4-8` + `pin_memory`; **measure
loader throughput in the tier-1 container smoke** (steps/s with a trivial model). If it
starves the GPU, the documented fallback is GPU-side node sampling (move
`points_to_equirectangular_pixels` + `bilinear_sample_erp` to the training loop on decoded
ERP tensors) — out of scope for v1.

---

## 5. The interpolated lazy correlation lookup (`spherical_flow/oslo_raft_retina.py`)

This replaces all three of: `AllPairsCorrelation` ([N,N] volume),
`build_correlation_pyramid` (volume pooling), and the snap-gather `pyramid_lookup`. It is
lazy like `local_correlation_lookup` (no [N,N] ever, so it also unlocks r5/r6 estimation
later) and *continuous in flow* like RAFT's bilinear sampling.

**Key identity (document in the docstring):** because the dot product is linear in `f2`,
interpolating the correlation volume ≡ correlating against interpolated features:
`Σₖ wₖ (f1·f2ₖ) = f1 · (Σₖ wₖ f2ₖ)`. So we interpolate `f2` features and never build a
volume. Likewise RAFT's corr-pyramid pooling ≡ dotting with child-averaged normalized
features, so:

### 5.1 Feature pyramid (replaces the corr pyramid)

After the encoder: `f1, f2` at r_est, `F.normalize(·, dim=-1)` **once** (parity with
`AllPairsCorrelation` / `OSLORAFTLocal`). Then
`f2_levels[r] = pool_features(f2_levels[r+1], pyramid.pool_index[r])` for each
`r in corr_resolutions[1:]` — pooled **without renormalizing** (matches all-pairs volume
pooling exactly, by the linearity identity).

Note `corr_resolutions` go *below* r_est (r4→r3→r2→r1): `pool_index` must therefore also be
built for those coarse pairs. `build_healpix_pyramid` already builds levels for the corr
range; extend the `pool_index` comprehension to cover them (it currently does, since it
covers "every r with r+1 present" — verify with a unit test at est=4, corr levels 4..1).

### 5.2 Lookup stencil

Per corr level `l` with node spacing `s_l = sqrt(4π/N_l)` (radians), a fixed tangent-plane
stencil (configurable `--lookup-rings R` (default 2), `--lookup-ring-points P` (default 8)):

```
offsets_l = s_l · ( {(0,0)} ∪ { ρ·(cos θⱼ, sin θⱼ) : ρ ∈ 1..R, θⱼ = 2πj/P + (ρ−1)·π/P } )
```

→ `M = 1 + R·P = 17` samples/level; corr feature width `Σ_l M = 68` at 4 levels (vs 100
today — same order; `MotionEncoder(corr_channels=...)` already takes the width as an arg).
The ring-2 stagger (`+π/P`) decorrelates the two rings' directions.

### 5.3 The lookup

```python
def interp_pyramid_lookup(f1, f2_levels, flow, pyramid, offsets_per_level) -> Tensor:
    # [B, N_est, Σ_l M_l]
```

Per iteration (flow already detached by the caller, matching RAFT's semantics):

1. `e = endpoint_from_tangent_flow(est.points, flow, est.basis_east, est.basis_north)`
   — once. `e_east, e_north = tangent_basis(e)` (works on arbitrary unit vectors; the
   existing pole convention applies).
2. Per level `l`, per offset `δ = (δe, δn)`:
   `q = endpoint_from_tangent_flow(e, δ, e_east, e_north)` → query directions `[B, N, 3]`.
3. **Candidates:** `c = level_l.ang2pix(q)` (the brute closure is a `[B·N, N_l]` matmul —
   fine for `N_l ≤ 12,288`; chunk the queries if needed). Candidate set =
   `level_l.lookup_index[c][:, :9]` (center + its 8 ring — contains the true nearest nodes
   for any query within the cell).
4. **Interpolation:** geodesic distances `d_k = geodesic(q, points[cand_k])`; take the
   `K_int = 3` smallest; weights `w_k = (1/max(d_k, 1e-4)) / Σ` (inverse-distance,
   normalized). Weights are computed from detached geometry (no grad through indices —
   same contract as RAFT, which detaches lookup coords each iteration).
5. **Value:** gather `f2_l[cand_k]` with the flat-index trick from
   `local_correlation_lookup`, then `corr = f1 · (Σₖ wₖ f2ₖ)`. *(Implemented in cosine
   units, deliberately dropping the plan's original `/sqrt(C)`: with channel-normalized
   features the dot already lies in [−1, 1], and the extra shrink left the motion
   encoder a ≤0.1-max whisper against O(1) context features. RAFT's 1/√C tempers RAW
   dots whose scale grows with C — normalized dots need no temper.)*

Cost: `O(B · N_est · ΣM · K_int · C)` — at B=2, N=3072, ΣM=68, K=3, C=96 ≈ 0.5 GFLOP/iter
and ~48 MB gathered fp16 per iteration. Trivial at est=4/5.

**Estimation at r6 later:** step 3's brute `ang2pix` is the only non-local op; swap it for
the windowed candidate search from `OSLORAFTLocal` (`cand_points = points[lookup_index]`,
argmax within the window). Leave a `# NOTE` marking the seam; do not build it in v1.

### 5.4 Properties to unit-test (CPU, Fibonacci/synthetic-nested pyramid)

1. **Continuity (the point of the whole section):** for flow perturbations
   `‖δ‖ = 0.25·s_est` and `0.125·s_est`, `‖lookup(f+δ) − lookup(f)‖` is > 0 and roughly
   halves with δ (finite-difference smoothness). Contrast assert: the *old*
   `pyramid_lookup` returns **bit-identical** output under the same sub-node perturbation
   (this pair of asserts documents the bug being fixed).
2. **Snap parity anchor:** with `K_int=1` and `offsets = {(0,0)}` the new lookup equals the
   old snap lookup's center column exactly (regression tie to the validated path).
3. Cold start: at `flow = 0`, `q` at δ=0 equals the node direction; center-column corr
   equals the all-pairs diagonal to 1e-6.

---

## 6. The model (`spherical_flow/oslo_raft_retina.py`)

`class OSLORAFTRetina(nn.Module)` — clone the `OSLORAFTPyramid` skeleton with these deltas:

- `resolutions = range(pyramid.retina_resolution, pyramid.estimation_resolution − 1, −1)`
  for both `fnet`/`cnet` (`PyramidEncoder` reused as-is — it already runs one
  `ResidualSphereBlock` per resolution with `pool_features` between).
- **Channel ramp:** the linear auto-ramp in `OSLORAFTPyramid` gives too-fat early stages at
  depth 5–6. Default explicit ramps (overridable via `--feature-channels` comma list):
  retina 7 → est 4: `(16, 32, 48, 64, 96)`; retina 8 → est 4: `(16, 24, 32, 48, 64, 96)`.
  Params stay ~1.5–2.5M (SDPAConv params scale with C², not N — verify against the 1–3M
  budget in the smoke).
- **Encoder gradient checkpointing** (`use_checkpoint_encoder: bool = True`): wrap each
  per-resolution block call in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
  when `torch.is_grad_enabled()`. The big transient is SDPAConv's neighbor gather
  `[B, N, 9, C]` — at r8/C=16, B=2 that is ~0.9 GB fp16 *per conv* if stored; checkpointing
  keeps only the per-stage boundary features (`[B, N, C]`, ~50 MB).
- **Iteration checkpointing:** copy the `_update_step` + checkpoint pattern from
  `OSLORAFTLocal` verbatim (needed for est=5+; harmless at est=4).
- Correlation path: §5 (`f2_levels` built once per forward; `interp_pyramid_lookup` per
  iteration). `self.correlation` / `build_correlation_pyramid` are not used.
- `ablate_corr` / `ablate_context` class attributes with **identical semantics** to
  `OSLORAFT` (zero `corr_feat` after lookup / zero `h` and `context`). These must work in
  this model — they are the §8 gates.
- Context head, GRU, motion encoder, flow head (zero-init), `UpsampleWeightHead`,
  `convex_upsample` est→r_sup: unchanged reuse. Predictions returned at r_sup.

**Three §9.2-gate-driven amendments (measured 2026-07-02; each was individually
necessary for the decisive sub-node recovery test to pass — full write-up in the smoke's
docstrings):**

- **Position-free fnet.** The pre-retina models concatenated node xyz into *both*
  encoders. Position channels and matching don't mix — they bias every `f1ᵢ·f2_c` logit
  toward the spatially nearest candidate regardless of texture (a built-in self-match
  bias; RAFT's feature encoder sees only image content for this reason). `OSLORAFTRetina`
  keeps xyz in the context net only.
- **`stencil_match_loss` (auxiliary matching supervision), promoted from the §10
  fallback to a core ingredient.** End-to-end training NEVER developed matchable
  features on any tested budget (the flow loss tracked the zero-flow baseline to 3
  decimals); with direct soft-target InfoNCE over each node's lookup window, matching
  reaches its entropy floor in ~300 steps (argmax accuracy 0.90–0.99 across angles).
  Runner: `--aux-match-weight` (default 0.5) + `--aux-warmup-steps` (aux-only feature
  bootstrap phase); est-grid GT endpoints are pooled from the supervision grid's
  descendants.
- **Corr skip into the GRU input.** With matching solved, the corr stencil is *linearly*
  decodable to flow (ridge-probe cos-sim 0.99, correct magnitude) — yet through
  motion-encoder→GRU→head alone the decode never aligned within a smoke budget. Feeding
  the raw corr features into `gru_in` alongside the encoded motion features (two hops
  from the delta head) lets the flow loss fall to ~0.1× the zero baseline within ~450
  joint steps.

---

## 7. Runner wiring (`run_oslo_raft.py`)

New mode, mutually exclusive with `--multi-res` / `--local-corr` / `--differential`:

```
--retina                    use OSLORAFTRetina
--retina-resolution 7       retina grid order (8 = ERP-pixel parity, ~4x cost)
--resolution 6              supervision order (unchanged meaning)
--estimation-resolution 4   correlation/GRU order (unchanged meaning)
--lookup-rings 2 --lookup-ring-points 8
--pyramid-cache outputs/pyramid_cache
--no-encoder-checkpoint     (checkpointing on by default in retina mode)
--synth-rot-prob 0.0 --synth-rot-min-deg 1.0 --synth-rot-max-deg 15.0
```

Wiring (mirror the `--multi-res` branch):

- `pyramid = load-or-build` via §3.4 with `retina_resolution=args.retina_resolution`;
  validate `est < sup <= retina`.
- **Datasets:** `dataset_points = pyramid.retina_level.points` (frames) and
  `target_points = pyramid.fine_level.points` (targets) — both `ShardFlowDataset`s get the
  new kwarg. `sup_level = pyramid.fine_level`; `geom = pyramid`.
- `--ablate-corr` / `--ablate-context` are **allowed** in retina mode (remove them from the
  multi-res-style guard for this mode).
- Everything else (loss flags, AMP, OneCycle, eval, metrics JSON schema, checkpoint saving)
  unchanged — the aggregator must keep discovering these runs automatically.

---

## 8. Training plan and decision gates

(Recipe outline — exact mixes/schedules to be settled before the first GPU run; the gates
are the load-bearing part. All shakeouts use the established protocol: 5000 steps, AdamW,
OneCycle peak 4e-4 **with warmup** (the val@2000 spike), batch 2, AMP, RTX 3090.)

### Stage A — matching bootstrap (retina 7, est 4)

Data: `replica360:train` only (real supra-node motion, p50 ≈ 13° node-space) +
`--synth-rot-prob 0.5 --synth-rot-min-deg 1 --synth-rot-max-deg 15` (exact supra-node
motion on real texture). SO(3) prob 1.0 as usual. ~5000 steps.

**Gate R1 — "is the correlation finally load-bearing?"** Evaluate the trained checkpoint on
a synthetic-rotation val set + `replica360:val`, twice: full model vs `--ablate-corr` at
eval. The ablated model *cannot* predict a random rotation from frame 1 alone, so:
**PASS = the corr-ablated eval is dramatically worse** (synthetic-rotation val should
approach zero error un-ablated). FAIL = they tie again → the lookup/retina is broken;
debug §5 before touching more data (nothing else in this plan matters until R1 passes).

### Stage B — fine-motion transfer

Init from Stage A. Standard mix (`replica360,mpf,flow360`), anneal `--synth-rot-prob`
0.5→0.1. 5000-step shakeout, then a longer run if the shakeout moves.

**Gate R2 — "does it beat the appearance prior where it counts?"** On `flow360:val`:
active>0.5° improvement must exceed **+5.2%** (the best transient the old models ever
touched) and clearly beat the +2.9% ceiling; and the corr-ablation gap from R1 must
*persist on flow360 val* (if the gap collapses on flow360 while holding on Replica, the
model matches large motion but still can't use frame 2 at FLOW360's scale — that is itself
a publishable refinement of the negative result: the retina fixed matching, and the
remaining wall is the data's motion scale, cleanly isolated).

### Stage C — headline (only if R2 passes)

Retina 8 and/or est 5; longer schedule (~50–100k steps per the original T2 recipe); seeds
7/11/19. Reference points on FLOW360 test r6: zero-flow active_0_5 = 1.8704, frozen RAFT =
1.0404 (+44%), RAFT+residual = 1.0368. A defensible thesis win = landing meaningfully
between the +2.9% prior ceiling and frozen RAFT on active subsets while keeping the
pole/seam story; matching frozen RAFT is a stretch goal, not the gate.

### Cost estimates (verify at tier-1 smoke; these are planning numbers)

| config | encoder input | est. wall-clock per 5k-step shakeout |
| --- | --- | --- |
| r4 single-res (old baseline) | 3k nodes | ~33–43 min (measured) |
| retina 7 / est 4 | 196k nodes | ~1.5–3 h |
| retina 8 / est 4 | 786k nodes | ~4–8 h, B=1 + grad-accum likely |

Memory at retina 8 if B=2 OOMs: drop stem to C=16 (default), B=1 + `--grad-accum 2`
(add the flag if needed — trivial), keep encoder checkpointing on.

---

## 9. Tests and smokes (build before any GPU run — house rule)

1. **Unit (CPU, no healpy):** §5.4 lookup continuity/parity/cold-start; §3.2 trivial-lookup
   levels; §4.1 seam backward-compat (target_points=None reproduces current samples
   bit-for-bit on a fixed record); §4.4 synth-rotation record (flow of a pure yaw at the
   equator ≈ yaw angle; conventions match `SyntheticRotationFlowDataset`).
2. **`run_oslo_raft_retina_smoke.py` (CPU, the decisive one):** synthetic nested pyramid
   (extend the `run_oslo_raft_pyramid_smoke.py` builder with one extra retina level).
   Checks: (a) param budget 1–3M; (b) cold-start preds[0] == 0 at r_sup; (c) finite grads
   end-to-end incl. both checkpointing paths (and checkpointed == non-checkpointed grads,
   as validated for `OSLORAFTLocal`); (d) **sub-est-node recovery:** train on
   `analytic_sphere_texture` rotation pairs (random axis per sample), **held-out eval at
   angle = 0.3·s_est** (sub-node at est, multi-sample at the retina); full model must
   reach direction cos-sim > 0.9 while the same run with `ablate_corr=True` must fail
   (< 0.3). This is the entire thesis of the plan in one CPU test — if (d) fails, stop
   and debug before the box.
   *Implementation notes (2026-07-02): training at the fixed sub-node angle alone never
   escaped the context-only optimum (predict ≈ 0 flow; a fixed single rotation overfits
   in 25 steps, so the wiring was fine) — the implemented protocol trains on angles
   uniform in 0.2–1.2·s_est (the §8 Stage-A bootstrap insight in miniature) with OneCycle
   peak 2.5e-3 over 600 steps, and keeps the eval strictly at held-out 0.3·s_est.*
3. **Container tiers (`scripts/container_smoke.sh`):** tier 1.7 = build/load the r7 pyramid
   (cache round-trip), one forward/backward `--retina` B=1 AMP on random frames, assert
   VRAM < 20 GB and report loader steps/s (§4.5); then `--smoke-test` on real shards.
4. **Overfit:** 10 replica pairs through `--retina --max-train-pairs 10` → geodesic error
   near zero (the standing sanity bar for every variant).

---

## 10. Risks / fallbacks

| risk | signal | fallback |
| --- | --- | --- |
| r8 graph build cost | minutes-scale rebuild per run | §3.3 fast path + §3.4 cache (mandatory, not optional) |
| retina encoder OOM | tier-1 smoke | C=16 stem, B=1 + accum, retina 7 |
| loader starves GPU | low util at tier 1.7 | more workers; then GPU-side sampling (§4.5, v2) |
| corr still inert after Stage A | R1 fails | debug lookup continuity first; then add InfoNCE aux matching loss over the stencil (supra-node targets exist now, unlike before); if *still* inert → the negative result extends to retina-decoupled models: a stronger, well-instrumented conclusion — write that up |
| R1 passes, R2 fails | flow360 active stuck ≤ +2.9% w/ corr gap collapsed on flow360 | the refined negative result of §8-B: matching works, FLOW360's motion scale is the wall — quantified with the R1/R2 contrast |
| `tangent_basis` degeneracy at ±z for endpoint bases | NaNs in lookup near poles | reuse the geometry module's existing pole convention; add a pole-node unit test in §5.4 |

Every branch produces thesis-usable material: PASS-PASS = the method chapter's positive
result; PASS-FAIL and FAIL both sharpen Ch. 4's negative result with the confound removed.

---

## 11. Implementation order (agent checklist)

Each step lands with its tests green before the next; CPU-first throughout.
**All steps landed 2026-07-02**; notes record where each item lives.

1. ✅ `healpix_pyramid.py`: `retina_resolution` + trivial-lookup levels
   (`lookup_neighbors=0` → `[[i]]`) + auto-chunk (`_effective_chunk`, 256 MB budget).
2. ✅ `healpix_pyramid.py`: `healpix_neighbor_graph` fast path (used automatically for
   HEALPix levels with N > 100k and `conv_neighbors=8`; r4 parity asserted in container
   tier 1.7a); `save_pyramid`/`load_pyramid` (version-tagged plain-tensor dict; round-trip
   asserted in tier 1.7c).
3. ✅ `shard_dataset.py`: `sample_pair_to_nodes` target-grid args (backward-compat
   bit-for-bit; guard: `endpoint_rotation` + `target_points` requires
   `target_query_points`); `so3_augment_pair` passthrough; `ShardFlowDataset` /
   `load_shard_subset` kwargs. Tests in `run_oslo_raft_retina_smoke.py` §2.
4. ✅ `shard_dataset.py`: `synth_rotation_record` (+ optional `view_rotation` composing
   the SO(3) viewpoint aug) + `synth_rot_prob/min_deg/max_deg` dataset knobs. Convention
   tests (endpoint = R p; frame2 = texture(R⁻¹ p); equator yaw) in smoke §3; the
   exact-rotation Gram invariant over real shards was verified during implementation.
5. ✅ `oslo_raft_retina.py`: `interp_pyramid_lookup` + `build_feature_pyramid` +
   `build_lookup_offsets`. §5.4 tests in smoke §4 — with one refinement discovered during
   implementation: distances use the **chord form** `2·asin(‖q−p‖/2)` (fp32 `acos` floors
   at ~4.5e-4 rad and would leak interpolation weight off exact-node queries), and on the
   *irregular synthetic* grid the snap-constancy contrast is asserted per-node
   (perturbation scaled to each node's nearest-neighbor distance, est-level slice
   bit-identical); the strict global form runs on real HEALPix in container tier 1.7b.
6. ✅ `oslo_raft_retina.py`: `OSLORAFTRetina` — depth-keyed channel ramps
   (`_DEFAULT_RAMPS`, depths 1–3 reproduce the validated pre-retina defaults), encoder
   per-stage checkpointing (`PyramidEncoder.use_checkpoint`), iteration checkpointing
   (`_update_step`, the OSLORAFTLocal pattern), `ablate_corr`/`ablate_context`.
   Cold-start/grad/checkpoint-parity/param-budget tests in smoke §5.
7. ✅ `run_oslo_raft.py`: `--retina` wiring + `--pyramid-cache` (load-or-build) + the
   stage-recipe plumbing the gates need: `--init-checkpoint`, `--eval-only`,
   `--val-synth-rot-prob`, `--grad-accum`. Old modes regression-smoked on real shards.
8. ✅ `run_oslo_raft_retina_smoke.py` incl. the sub-node recovery test (§9.2d) — the
   go/no-go for GPU work.
9. ✅ `scripts/container_smoke.sh` tiers 1.65 (CPU wiring suite), 1.7 (real geometry:
   parity/snap-constancy/cache/model+VRAM), 2.7 (loader throughput + `--retina
   --smoke-test` on shards). Stage A command in `OSLO_RAFT_DOCKER.md` §Real training.
