# Universality table: published 360° methods vs zero-flow on the sub-pixel leg

Workstream opened 2026-07-24 (see `docs/THESIS_REGIME_ARGUMENT.md` §5 for why this
table is load-bearing: the claim "the sub-pixel regime is unsolved by anyone"
currently rests on one perspective-model data point, frozen RAFT-large −20.5%).

Rows planned: frozen RAFT-large (DONE, P2A), SLOF (in-domain home method of
FLOW360 — load-bearing row), PriOr-RAFT (ICCV 2025 SOTA), PanoFlow(CSFlow).
Columns: geodesic global + actives (0.25/0.5/1.0°) vs zero-flow, on flow360:val
real pairs, our metrics stack.

## 1. Literature pre-check (2026-07-24) — DONE

Published FLOW360 test numbers (journal version, arXiv 2301.11880, Table I;
eval at **320×640**, iters=64, unweighted ERP px, magnitude-bucketed):

| method | All | s<5 px | s≥20 px |
| --- | --- | --- | --- |
| SLOF (v1) | 2.548 | 0.309 | 62.476 |
| SLOF (v2) | 2.626 | 0.326 | 64.678 |
| RAFT fine-tuned | 2.635 | 0.314 | 65.340 |
| RAFT (trained) | 3.344 | 0.558 | 71.736 |
| RAFT + KTN | 3.899 | 0.598 | 76.426 |

Zero-flow "All" EPE (= mean |GT|): **never reported by them.** From our P2A
numbers on our val split: unweighted global zero ≈ 1.43 px @1024-wide ⇒ **≈0.89
px @640-wide**. Their own bucket arithmetic (All ≈ mix of lt20 body ~0.5 and
gte20 tail ~62) implies a zero baseline of ~1.3–1.9 px on their test split
(their split carries more large motion than our val, and raw-ERP-px bucketing
pole-inflates "large motion"). **Either way: every published FLOW360 row sits
~1.3–2.9× ABOVE the zero-flow baseline they never report.** Globally, the
literature's own tables lose to doing nothing — universality supported at the
global level from reading alone.

**What the pre-check CANNOT settle:** the s<5 body bucket (SLOF 0.309) is in the
same range as the zero baseline restricted to that bucket (~0.35–0.9 px
depending on split stats and pole inflation) — px-magnitude buckets conflate
latitude with motion (1/cos inflation) and mix the static 88% with small movers,
so they neither confirm nor exclude SLOF beating zero on genuine movers. The
geodesic actives-conditioned run remains necessary and is a real risk in both
directions. This is exactly the due-diligence the run must close BEFORE the
thesis prints "unsolved by anyone".

## 2. SLOF tarball inspection (2026-07-24) — DONE

Source: `SLOF.tar.gz` (163 MB, Dropbox, siamlof.github.io), Nov 2021.

- **At inference SLOF = vanilla RAFT-large forward pass.** The siamese/
  contrastive machinery (simsiam.py, rotation schemes) is training-only. Their
  `RAFT/core` is the princeton-vl code (hdim 128, corr 4/4, BN in cnet) with a
  `simsiam` flag; forward() takes uint8-range BCHW, normalizes internally,
  `test_mode=True` returns (lowres, upsampled px flow).
- **Six checkpoints shipped in `weights/`**: `raft.pt`, `raftfinetune.pt` (20 MB
  — plain RAFT-large state dicts), `ktn.pt`, and the SLOF variants
  `singlerotation.pt` / `switchrotation.pt` / `doublerotation.pt` (36 MB — saved
  with the simsiam projector; extra keys). All saved via DataParallel
  (`module.` prefix). Which variant is the paper's "v1" must be pinned by a
  quick check (main.py default is singlerotation); plan: run all three, report
  best, footnote the mapping.
- **Their eval protocol**: 320×640, iters 64, flow resize with correct magnitude
  rescaling (`ReadData.transform_flow` normalizes by old dims, re-scales by
  new), EPE = unweighted ERP px + buckets + optional distortion-density
  weighting ("EPEd"). Data layout: `FLOW360_train_test/{train,test}/NNN/
  {frames,fflows,bflows}` — same raws sphereflow-dataprep ingested.
- **Deps ancient but harmless**: torch 1.7/cu101/py3.8; the inference path is
  standard ops only (no custom CUDA needed; alt_cuda_corr optional). Should run
  under a modern torch in a small `baselines` image.

## 3. Integration plan — **harness DONE + Docker-validated 2026-07-24**

Implemented: `spherical_flow/princeton_raft.py` (vendored inference-only
princeton RAFT, BSD-3; loader strips DataParallel `module.` and SLOF's SimSiam
`encoder.` prefixes, drops the 20 `predictor.*` projector keys, and hard-fails
on any missing model key); `predict_princeton_flow` in `raft_adapter.py`
(optional `infer_size` resize with flow magnitude rescale, mirroring SLOF's
loader convention); `run_raft_shard_baseline.py` gains `--checkpoint`,
`--iters`, `--infer-size` (metadata recorded in the JSON). Local Docker checks:
all 5 SLOF ckpts load exactly (179/179 keys) and forward finite at real sizes
(the 64×128 smoke NaN is the known RAFT pyramid degeneracy, not a bug);
end-to-end on 2 local flow360:val pairs through the full spherical metric path
passes. Box protocol: iters 64 @ 320×640 (their published setting), full
flow360:val, one output dir per checkpoint (`universality_slof_<ck>`).

Wire their checkpoints into OUR harness (not their loader): the existing
RAFT-shard baseline / P2A EPE path already runs princeton RAFT-large on
flow360:val. Needed: checkpoint arg + `module.` strip + `strict=False` for the
projector-bearing SLOF variants; eval at their native 320×640 AND our standard
resolution (fairness both ways; BN is eval-mode so batch size irrelevant).
Outputs: geodesic global/regions/actives vs zero (the universality columns) +
ERP EPE for continuity with published numbers. Then PriOr-RAFT and
PanoFlow(CSFlow) ride the same harness (both PyTorch RAFT-family, public
weights; PanoFlow needs its `--CFE` wrap-around treatment at inference).

Decision rule for the thesis: all rows negative on actives ⇒ "unsolved by
anyone" stands and +4.5% consolidated is *the first positive, full stop*; any
row positive ⇒ scope the claim to "first native-spherical" and report the
finding honestly.

## 4. SLOF round 1 on flow360:val (2026-07-24) — RESULTS + a split-leakage discovery

Five checkpoints, 791 pairs, iters 64 @ 320×640, geodesic vs zero-flow:

| ckpt | global | act₀.₂₅ | act₀.₅ | act₁.₀ |
| --- | --- | --- | --- | --- |
| raft (scratch on FLOW360) | −465.2 | −112.5 | −65.1 | −33.9 |
| raftfinetune | −23.0 | −9.2 | −8.2 | +0.4 |
| **singlerotation** | −19.4 | **+5.0** | **+4.5** | **+6.2** |
| switchrotation | −58.3 | −21.2 | −19.5 | −7.0 |
| **doublerotation** | **+0.7** | +1.1 | +1.6 | +0.4 |
| (OSLO EMA, held-out val) | −16.8 | +3.5 | +4.2 (basin +4.5±0.9) | −1.3 |

**Leakage caveat that changes the reading**: sfprep's flow360 adapter carves
our `val` out of FLOW360's *official train* split (every `val_every`-th train
sequence; official test preserved as `test`). SLOF trained on the full official
train ⇒ **our val sequences were in SLOF's training set**. The singlerotation
positives are therefore in-training-set numbers — an upper bound on their
capability, not generalization. Even so, three observations stand: (a) the
regime's actives/calibration TRADE-OFF is now visible across their own
variants — singlerotation buys actives at −19 global, doublerotation is the
"confidently predict zero" solution (+0.7 global, first non-negative global
ever on this leg, ~0 actives), nobody gets both; (b) **with a train-set
advantage, their best variant only ties OSLO's held-out consolidated actives**;
(c) from-scratch in-domain training is catastrophic (−465) — calibration does
not come free even with supervision.

**Round 2 (decisive, leakage-free): flow360:test** — official FLOW360 test,
unseen by BOTH SLOF (their held-out split) and OSLO (sfprep preserved it; our
training used sfprep train only). Same five SLOF rows + the OSLO EMA final
model, outputs `universality_slof_<ck>_test` and `P1final_test_flow360`. This
is the publishable table; the val round is kept as the leakage illustration.

## 5. Round 2 COMPLETE (flow360:test, 2026-07-25) — universality holds SYMMETRICALLY

flow360:test (2567 pairs) is a different, harder regime mix than val: zero global
0.4368° (2× val's 0.2105), active fracs **34.6/18.0/5.8%** (val: 24.1/12.4/1.7),
zero actives 1.10/1.79/4.08° — the official test videos carry ~3× the mover
mass and much larger motions. The full leakage-free table:

| row (test) | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles |
| --- | --- | --- | --- | --- | --- |
| SLOF raft (scratch) | −217.0 | −59.7 | −25.9 | −7.1 | −128.8 |
| SLOF raftfinetune | −9.2 | −6.2 | −5.2 | −4.2 | −11.0 |
| SLOF singlerotation | −8.1 | −2.2 | **−3.2** | −6.0 | −16.7 |
| SLOF switchrotation | −20.2 | −11.1 | −9.0 | −6.9 | −27.4 |
| SLOF doublerotation | +0.03 | +0.02 | +0.12 | +0.36 | −0.63 |
| **PanoFlow(CSFlow)+CFE** | −10.1 | −5.8 | **−4.7** | −4.2 | −12.0 |
| **OSLO EMA final** | −14.0 | **−4.9** | **−4.0** | **−5.7** | −24.8 |

**PanoFlow(CSFlow) (T-ITS'23), run with its own CFE cyclic wrap at native
resolution (2026-07-25, ckpt strict-loaded 207/207 keys), is a genuine
predictor — global −10.1, not zero-parity, it moves flow — and loses to zero on
every actives bucket (act₀.₅ −4.7), landing in the same negative band as SLOF
singlerotation (−3.2) and OSLO (−4.0).** A method whose entire design point is
360° cyclic flow estimation for the wrap-around setting still cannot beat doing
nothing on the sub-pixel actives. Three architectures, three labs (Princeton-
lineage SLOF, cross-strip-correlation PanoFlow, native-spherical OSLO): all
negative.

**Domain status of the PanoFlow row — PINNED 2026-07-25 (naming collision
resolved).** PanoFlow's README calls *their own CARLA-rendered dataset*
"**FlowScape (Flow360)**" (8 city maps × 4 weathers, 1024×512, 6400 frames);
their `--validation Flow360` flag points at **that**, not at SLOF's FLOW360. Two different datasets, confusingly similar names:
- sfprep `flowscape` = PanoFlow's FlowScape/Flow360 (CARLA, large motion)
- sfprep `flow360`  = SLOF's FLOW360 (naturalistic RENDERED video, sub-pixel)

Their public `PanoFlow(CSFlow)-wo-CFE.pth` is trained with `--dataset Flow360`
= FlowScape ⇒ **the row above is OUT-OF-DOMAIN zero-shot**, in the same class as
the frozen RAFT-large row, not an in-domain row. This does not weaken the
universality conclusion (SLOF remains the load-bearing *in-domain* row, and an
out-of-domain method losing to zero is fully consistent), but the strength must
be attributed correctly: **in-domain evidence = SLOF; out-of-domain corroboration
= PanoFlow + frozen RAFT.** It also means PanoFlow's checkpoint is *in-domain on
flowscape*, which is what makes §7 a fair fight.

**The decisive line — singlerotation: +4.5 (val, leaked) → −3.2 (test, clean).**
Its val positive was an in-training-set artifact (our val ⊂ SLOF's train). On
the clean cross-pool test **no SLOF variant beats zero on actives**; the only
non-negative row is doublerotation, and it is a trivial zero-predictor
(global +0.03, actives ~0, poles −0.6 — it predicts ≈nothing and inherits the
zero baseline). Every variant that actually moves flow (raft/raftfinetune/
single/switch rotation) is strictly worse than doing nothing on movers.

**OSLO is symmetric to this**: +4.2 act₀.₅ (val) → −4.0 (test). Not leakage on
our side (val sequences were held out from our training) — two honest causes:
(a) SELECTION pressure: every campaign decision (gates, EMA basin, checkpoint
choice) was made on val; (b) POOL shift: val is carved from the same official
train pool our training sequences come from; test is a disjoint video pool with
3× the mover mass. Our val number is same-pool generalization; test is
cross-pool. Calibration transfers better than actives: global −16.8→−14.0,
poles −56→−24.8 (more movers = smaller static penalty).

### Verdict for the thesis

On clean, leakage-free, cross-pool flow360:test, **nobody beats zero-flow on
actives — not SLOF (the in-domain home method, even with a train-set advantage
on val), not OSLO.** The "sub-pixel regime is unsolved by anyone" claim holds
symmetrically and is now backed by the field's own SOTA-lineage method rather
than one perspective-model data point. Three things this table buys the thesis:

1. **The wall is universal, not an OSLO artifact.** SLOF singlerotation −3.2,
   PanoFlow(CSFlow)+CFE −4.7, OSLO −4.0 — three independent architectures from
   three labs (Princeton-lineage siamese, cross-strip-correlation with a 360°
   cyclic wrap, native-spherical) land in the same negative act₀.₅ band. The
   only non-negative rows are trivial zero-predictors (SLOF doublerotation).
2. **The actives↔calibration trade-off is real and visible inside SLOF's own
   variants**: singlerotation buys actives (val +4.5) at −19 val global;
   doublerotation is the confident-zero corner (global ~0, actives ~0). Nobody
   gets both — exactly OSLO's dilemma, reproduced in a second lab's model.
3. **OSLO's honest scope**: the +4.5 act₀.₅ is "consolidated on same-pool
   held-out validation"; on cross-pool test it does not transfer, and neither
   does anyone's. The publishable positive is the *characterization* (regime
   split, universal wall, trade-off curve, matcher-genuine correspondence), not
   a cross-pool actives win — which no published method has either.

**Scope discipline going forward**: any actives number in the thesis must be
labeled val (same-pool) vs test (cross-pool). The test table is the load-bearing
universality evidence; the val round is retained only as the leakage
illustration and the trade-off-curve visualization.

### Remaining published rows (optional)

PriOr-RAFT (ICCV 2025) and PanoFlow(CSFlow) would strengthen the table from
"FLOW360's own method loses" to "the last two years of published 360° flow
loses", but are not required for the core claim — SLOF, as FLOW360's home
method, already carries it. **Neither is drop-in like SLOF** (which is vanilla
RAFT at inference): PriOr-RAFT is dual-branch (primitive ERP + orthogonal view,
DCCL/ODDC modules) and ships only large-motion-regime weights (MPF/FlowScape) →
out-of-domain zero-shot on flow360, weaker row; PanoFlow(CSFlow) has cross-strip
correlation + the CFE cyclic-inference wrap and a native Flow360 eval → the
stronger row. Decision (2026-07-25): **integrate PanoFlow(CSFlow) only**; PriOr
stays cite-only.

## 6. PanoFlow(CSFlow) integration — HARNESS BUILT + locally Docker-validated 2026-07-25

Same design as the SLOF/princeton integration (vendor the net, add an adapter,
one flag on the existing runner → identical spherical metric → byte-comparable
row). PanoFlow's inference net is **one self-contained file** (only torch +
`torchvision.ops.DeformConv2d`), so no second image / no two-stage pipeline:

- `spherical_flow/panoflow_vendor/panoflow_csflow.py` — verbatim copy of
  MasterHow/PanoFlow `opticalflow/core/model/external/panoflow_csflow.py` (MIT,
  © 2022 Hao; only a docstring header added). PanoCSFlow net, 5.63M params,
  DCN encoder, cross-strip correlation.
- `spherical_flow/panoflow_adapter.py` — `load_panoflow_checkpoint` (DotDict
  args mirroring easydict semantics the net needs; strips `module.`/`_model.`
  prefixes; **strict** load, hard-fails on any missing key) +
  `predict_panoflow_cfe_flow` (verbatim port of `evaluate.py::validate_flow360_cfe`
  CFE: encode once with `gen_fmap`, split feature maps at the ERP mid-meridian,
  decode the two cyclically-shifted halves with `skip_encode`, element-wise
  `minimum` per half, re-stitch + repair the 2 seam columns; batched; runs at
  native shard resolution with InputPadder — PanoFlow's own eval protocol).
- `run_raft_shard_baseline.py` — `--panoflow-checkpoint` + `--panoflow-eval-iters`
  (default 12); native-res + CFE, skips the /8 divisibility check (InputPadder
  handles it); records `model.panoflow` metadata in the JSON.

Local Docker validation (our `oslo-raft:cuda` image, CPU): net builds; loader
round-trip strict-loads `module._model.` / `_model.` / bare-key checkpoints and
rejects a wrong-architecture one; a non-CFE forward runs through
fnet/cnet/DCN/corr/GRU with the correct `(1,2,H,W)` shape. The CFE path itself
uses PanoFlow's hardcoded `.cuda()` (skip_encode branch) → validated on the box
smoke, not locally.

**Box run — DONE 2026-07-25.** Full flow360:test (2567 pairs, 827s, eval_iters
12, native res + CFE): ckpt strict-loaded **207/207 keys, 0 unexpected** (the
public `-wo-CFE.pth` matched the vendored net exactly); printed zero-baseline
**global 0.4368°, active fracs 34.6/18.0/5.8%** = identical to the SLOF/OSLO test
rows ⇒ same split+resolution, row is byte-comparable. Result: global −10.1,
act₀.₂₅/₀.₅/₁.₀ = **−5.8 / −4.7 / −4.2**, poles −12.0 — a genuine predictor that
loses to zero everywhere. Row folded into §5; harness path validated end-to-end
on GPU (the CFE `.cuda()` branch works, no NaN). PriOr-RAFT remains cite-only.

## 7. LARGE-MOTION cross-lab table on flowscape:test — OPENED 2026-07-25

**Why this section exists.** The universality table (§1–6) hardened the *negative*
claim with cross-lab, same-metric, self-run rows. The *positive* claim — "OSLO is
good in the large-motion regime" — rested entirely on **internal** baselines
(zero-flow and a frozen RAFT-large we ran ourselves): replica360 +88.4%, poles
2.88° vs RAFT 3.65°, 2.3× flatter, EPE 14.2 vs 31.3 px, chairs360 +16–17%. No
published 360° method was ever run in the large-motion regime. That asymmetry is
the weakest point in the thesis and this section closes it.

**Why flowscape:test is the right arena — clean and in-domain for BOTH sides:**
- It IS PanoFlow's own published benchmark (their "FlowScape (Flow360)"), so
  their public checkpoint is **in-domain** there — no out-of-domain excuse in
  either direction.
- sfprep's flowscape adapter **preserves the official test split**
  (`split = "test" if official == "test"`, adapter l.59); train/val are carved
  from official train only, MAP-grouped across weathers so no scene straddles the
  split via a weather variant. So flowscape:test is unseen by PanoFlow (their
  held-out) and by OSLO (our training used the `:train` carve).
- Motion is genuinely large: p90 70–93 px — the regime where OSLO's correlation
  actually resolves, unlike the sub-pixel leg.

### RESULTS 2026-07-25 — OSLO LOSES DECISIVELY. Pre-registered branch 2 applies.

**Split verification PASSED**: the final model's saved args show
`train_sources: "chairs360:train,flowscape:train,flow360:train"` — `flowscape:train`,
never `:test`. **Consistency check PASSED**: both runs report
`quantile_samples 44213334.0`, `target p90 7.290164947509766`, `p95 8.578697204589844`
bit-identically and `global_zero` 3.4024975 vs 3.4024978 (Δ 2.7e-7°, fp/AMP noise)
⇒ same split, same nodes, same data. 1386 pairs.

Regime confirmation: zero global **3.402°**, target p50 **2.467°**, p90 **7.290°**,
active fracs **92.3/85.0/73.7%** — genuinely large motion (vs flow360:test's
p50 0.131°, fracs 34.6/18.0/5.8).

| row (flowscape:test) | domain | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles | equator | seam |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero-flow (°) | baseline | 3.402 | 3.676 | 3.958 | 4.454 | 8.065 | 1.804 | 4.386 |
| **PanoFlow(CSFlow)+CFE** | in-domain | **0.251° (+92.6%)** | +92.8 | **+93.0** | +93.2 | **+95.3** | **+85.5** | **+63.9** |
| frozen RAFT-large | out-of-domain | 0.872° (+74.4%) | +74.5 | +74.6 | +74.5 | 3.650° (+54.7) | 0.351° (+80.5) | +49.7 |
| **OSLO EMA final (mix)** | in-domain | 1.158° (+66.0%) | +66.5 | +67.0 | +67.8 | **3.147° (+61.0)** | 0.727° (+59.7) | +39.9 |

(RAFT-large row: torchvision `C_T_SKHT_V2`, native 1024×512; both new runs
bit-match the zero baseline and active fracs of their respective splits.)

**PanoFlow wins on every single axis**: global 4.6× lower error (0.251° vs
1.158°), poles 8.4× lower (0.377° vs 3.147°), every actives bucket, equator, and
seam. There is no region and no threshold where OSLO leads it. Uniformity also
goes to PanoFlow: poles/equator ratio **1.44× (PanoFlow) vs 4.33× (OSLO)**.
OSLO also loses **global** to out-of-domain frozen RAFT-large (1.158° vs 0.872°).

### But the OSLO-vs-RAFT head-to-head REPLICATES on an independent dataset

The frozen RAFT-large row is not just a third data point for the swing figure —
it independently reproduces the replica360 head-to-head (§thesis 5.1), on a
different dataset, with the same qualitative split:

| OSLO vs frozen RAFT-large | replica360 (earlier) | flowscape:test (new) |
| --- | --- | --- |
| global | RAFT wins (1.16° vs 1.56°) | RAFT wins (0.872° vs 1.158°) |
| **poles** | **OSLO wins (2.88° vs 3.65°)** | **OSLO wins (3.147° vs 3.650°)** |
| **uniformity (poles/equator)** | **OSLO 2.3× flatter** | **OSLO 2.4× flatter** (4.33× vs 10.4×) |

Two independent large-motion datasets, same verdict: **the perspective model is
more accurate on average, the native-spherical model is more accurate at the
poles and markedly more uniform across the sphere.** This is a genuine,
replicated architectural result and it survives the scope-down — it is the
strongest *positive* claim the thesis can make about OSLO's geometry, and it is
now backed by replication rather than a single dataset.

**The honest boundary of that claim**: it holds **vs a perspective model**. A
well-trained ERP-native specialist (PanoFlow, +95.3% at the poles, 1.44×
uniformity) beats OSLO on exactly those axes. So the correct statement is
"spherical geometry buys polar accuracy and uniformity *relative to applying a
perspective architecture to ERP*", **not** "spherical geometry is the best way to
handle the poles". Training an ERP method on panoramic data at scale buys more.

**Diagnosis — the estimation grid, not the training.** OSLO estimates at
`estimation_resolution 4` = 3072 nodes ⇒ mean node spacing ≈ **3.66°**, while the
dataset's median motion is 2.47°. Most of the flow is *sub-node at the estimation
grid* even in the large-motion regime, recovered only by the r6 upsample.
PanoFlow decodes at 1/8 of 1024×512 with convex upsampling to per-pixel output ≈
**0.35°/px** — roughly a 10× output-resolution advantage. This is the same grid
floor identified in P2A, now shown to be costly even where correlation works.
It is an architectural cost of the coarse estimation grid, not an eval artifact:
both methods were scored at the identical r6 (49152-node) supervision grid.

**Verdict (pre-registered branch 2 — honest scope-down).** The claim "OSLO is
good in the large-motion regime" does **not** survive as a cross-lab claim.
OSLO is a *working* large-motion model (+66.0% global, +67.0% act₀.₅ — far from
broken) but it is **not competitive with a published 360° specialist on that
specialist's own benchmark**. What survives, stated precisely:
- **Polar accuracy + cross-sphere uniformity vs a perspective model** — now
  REPLICATED on two independent datasets (replica360: poles 2.88° vs 3.65°,
  2.3× flatter; flowscape:test: poles 3.147° vs 3.650°, 2.4× flatter), with 3.4×
  fewer parameters than RAFT-large. This is the surviving positive architectural
  claim and replication makes it stronger than it was before §7.
- Large-motion competence in absolute terms: +66.0% global / +67.0% act₀.₅ on a
  third-party benchmark — a working model, just not a leading one.
- **NOT claimable**: SOTA; "beats PanoFlow/PriOr/MPF-Net"; best-at-poles in the
  field (PanoFlow is better at the poles AND more uniform); global large-motion
  accuracy (both PanoFlow and even out-of-domain frozen RAFT-large beat OSLO).

### The scientific payoff: this loss makes the regime argument airtight

**THE REGIME-CONTRAST FIGURE — three checkpoints, unchanged weights, same
harness, same geodesic metric, only the dataset changes:**

| checkpoint | flowscape:test (large motion) | flow360:test (sub-pixel) | swing |
| --- | --- | --- | --- |
| | global / act₀.₅ | global / act₀.₅ | global |
| PanoFlow(CSFlow)+CFE | **+92.6% / +93.0%** | **−10.1% / −4.7%** | **103 pts** |
| frozen RAFT-large | +74.4% / +74.6% | −14.8% / −7.6% | **89 pts** |
| OSLO EMA final | +66.0% / +67.0% | −14.0% / −4.0% | **80 pts** |

Every checkpoint crosses from *strongly positive* to *negative*. Three
architectures — an ERP-native 360° specialist, a perspective model, and a
native-spherical model — three labs, three training regimes, one identical
pattern. The zero baselines and active fractions are bit-identical within each
column, so the only variable is the data regime.

**Why this closes the argument.** Before these runs a skeptic could answer the
universality table with "your methods are just weak." That reply is now dead: the
method that *nearly saturates* the large-motion benchmark (+92.6%, 0.251° — a
14× error reduction over zero) is **worse than predicting nothing** on real
video. Weakness cannot explain a 103-point swing in the *same weights*. The
failure is a property of the **regime**: mostly-static spheres with sparse
sub-pixel movers, where the motion field's structure — not appearance, not
architecture, not parameter count, not training budget — defeats correspondence.
This is the strongest single piece of evidence in the thesis, and it exists only
because the large-motion comparison we were missing got run.

**Table complete.** All planned rows are in: flowscape:test (4 rows) and
flow360:test (7 rows), every one self-run through the identical harness.

Protocol: same harness, same geodesic metric, HEALPix r6, native 1024×512 (already
/8, no padding), PanoFlow under CFE at eval_iters 12 — identical to the §6 run so
the two tables are directly comparable. Consistency check: the printed
`global_zero_geo_deg` and active fracs must match across all rows (proves same
split+resolution). Outputs: `largemotion_<row>_flowscape_test`.

**Pre-registered reading.** This is a real test with a real chance of losing, and
the answer is reportable either way:
- OSLO ≥ PanoFlow on actives ⇒ the positive claim upgrades from "beats our own
  baselines" to "competitive with a published 360° method on that method's own
  large-motion benchmark" — the cross-lab evidence the thesis currently lacks.
- OSLO < PanoFlow ⇒ scope the positive claim honestly to the axes we *did*
  measure and win on (polar accuracy, cross-sphere uniformity, parameter
  efficiency: 1.56M vs 5.63M), and state plainly that on global large-motion EPE a
  published method leads. Still a far stronger thesis than an unscoped claim.

**Verification step before trusting the OSLO row**: confirm the P1-proper run's
saved args used `flowscape:train` (not `:test`) — read
`/outputs/P1proper_mix20k/*.json` on the box. If it trained on any test-split
flowscape, this row is leaked and must be re-run from a clean checkpoint.

## 8. Grid-floor probe — TOOL BUILT + Docker-validated 2026-07-26

**Why.** §7 attributed OSLO's flowscape loss (1.158° vs PanoFlow 0.251°) to the
estimation grid. Two premises behind that attribution turned out to be wrong on
inspection:

1. *"OSLO lacks learned convex upsampling"* — **false**. `oslo_raft_retina.py:415`
   instantiates `UpsampleWeightHead` and `:519` applies `convex_upsample`; OSLO has
   the same mechanism as RAFT/PanoFlow, adapted to the sphere with parallel
   transport. Nothing to add. (The stale "next increment" comment is in the base
   `oslo_raft.py:19-21`, not the retina model we ran — cosmetic only.)
2. *"PanoFlow's decode grid is ~10× finer"* — **false**. That compared PanoFlow's
   per-pixel *output* (0.35°) against OSLO's *estimation* grid (3.66°), but OSLO
   convex-upsamples to r6 and **both were scored at r6** (0.916°), so output
   resolution beyond r6 never enters the metric. Decode grid vs decode grid:
   OSLO r4 = 3072 cells / 3.665° vs PanoFlow 128×64 = 8192 cells / 2.244° —
   **1.63× linear**, not 10×.

A 1.63× grid gap does not obviously explain a 4.6× error gap ⇒ the grid is a
**hypothesis, not a diagnosis**, and training at r5 before testing it is a blind bet.

**The probe** (`run_grid_floor_probe.py`) settles it without training: hand the
estimation grid the *perfect* answer (GT sampled directly at the estimation nodes
via `sample_pair_to_nodes`'s `target_*` path — no cross-tangent-plane pooling),
reconstruct r6 through the model's own transport, and score with the identical
geodesic stack every published row uses. Whatever error remains is imposed purely
by the estimation resolution. Three reconstructions bracket the floor:

| mode | weights | meaning |
| --- | --- | --- |
| `pwc` | one-hot on the center node (`upsample_neighbors[:,0]`) | naive upsampler; pessimistic |
| `uniform` | 1/K over the 1-hop neighborhood | smooth untrained upsampler |
| `oracle` | best convex combination per descendant | **no learned upsampler of this family can beat it** |

The oracle solves `min_w ||Σ w_k c_k − g||²` over the simplex by **Frank-Wolfe with
exact line search** (projection-free, no step size, closed-form 2-D steps). A first
attempt with projected gradient did **not** converge (residual 2.0 on an exactly
representable target) and was replaced.

**Docker validation.** Solver: recovers representable targets (mean residual 4e-5),
satisfies `oracle ≤ uniform` and `oracle ≤ pwc`, stays **inside the convex hull**
(support-function violation 2e-7 ⇒ a genuine bound, not cheating), and converges
monotonically in iters. End-to-end on 3 real flow360:test pairs, r4 and r5:

| est grid | spacing | pwc | uniform | oracle |
| --- | --- | --- | --- | --- |
| r4 | 3.665° | 0.0411° | 0.0534° | **0.0293°** |
| r5 | 1.832° | 0.0343° | 0.0412° | **0.0289°** |

Machinery confirmed (ordering holds everywhere; `pwc` beats `uniform` because
averaging over neighbors blurs while the center value does not). This also
retroactively explains the Act-I r4/r5/r6 null: on flow360 the floor is ~0.03°,
two orders below the error being measured, so the grid was never the binding
constraint there — the ladder *had* to tie.

**The decisive run is flowscape:test** (median motion 2.47°, where r4 = 0.67 nodes
is sub-node but r5 = 1.35 nodes is supra-node — the threshold is crossable, unlike
flow360 where p50 = 0.13° is below every affordable grid). Pre-registered rule:

| oracle floor @r4 | reading | action |
| --- | --- | --- |
| ≳ 1.0° | OSLO (1.158°) is **at** its grid ceiling | r5 is the fix; expected gain ≈ floor(r4) − floor(r5) |
| ≲ 0.4° | grid is **not** the bottleneck | capacity/training; skip r5, save the compute |

Box command (no rebuild needed only if the image already carries this file —
otherwise commit+push+rebuild first):

```
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_grid_floor_probe.py \
    --shards /data/shards --sources flowscape:test \
    --resolution 6 --estimation-resolutions 4,5 \
    --pyramid-cache /outputs/pyramid_cache \
    --output-dir /outputs/grid_floor_flowscape --device cuda
```

**Thesis note independent of the outcome.** ERP oversamples the poles by 1/cos(lat),
so PanoFlow's *angular* decode density at the poles is far above its average —
consistent with the poles being its strongest region (0.377°, +95.3%). HEALPix is
uniform by construction and gets no such windfall. This reframes the uniformity
claim honestly: **uniform sampling is an equity property, not an accuracy-maximizing
one.** HEALPix guarantees uniform error across the sphere; ERP buys polar accuracy
by spending density there. Worth a paragraph in the architecture chapter regardless
of whether any further run happens.

### 8b. Side-finding: the geodesic metric has a hard numerical floor of 0.028°

Found while validating the probe: running it with `--estimation-resolutions 6`
(estimation == supervision) must score **exactly zero** — the reconstruction is
provably an identity in that case (verified: `pwc` reproduces the source flow to
7.4e-9). It scores **0.0280°** instead. The residual is not the grid and not
invalid nodes; it is `geodesic_distance` itself:

```python
def geodesic_distance(a, b, eps=1e-7):
    dot = (a * b).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dot)          # <-- catastrophic cancellation near dot=1
```

`arccos` loses half the mantissa near 1. In float32 the floor is
`sqrt(2 * eps_f32) = 0.02798°` — matching the observed 0.0280° to four digits,
and uniform across every node (p50 = p99 = max = 0.0280). The explicit
`clamp(1 - 1e-7)` contributes a similar 0.0256° on its own. Confirmed on 200k
random identical unit-vector pairs: `arccos` gives mean 0.0073° / max 0.0442°,
while the standard stable form `2*asin(|a-b|/2)` gives **exactly 0**.

**Scope of impact — the headline results are safe:**

| quantity | value | floor as % | status |
| --- | --- | --- | --- |
| replica360 zero / OSLO | 13.53° / 1.57° | 0.2 / 1.8% | negligible |
| flowscape zero / OSLO / PanoFlow | 3.40° / 1.158° / 0.251° | 0.8 / 2.4 / 11% | safe; PanoFlow mildly inflated |
| flow360:test zero / OSLO | 0.437° / 0.498° | 6.4 / 5.6% | inflated at the low end, partly cancelling in the ratio |
| **actives (motion ≥ 0.25°)** | ≥ 9× the floor | ≤ 11% | **clean — the universality columns are unaffected** |
| grid floor on flow360 | 0.037° | ~76% | **unmeasurable — it is essentially all floor** |
| **B′ "0.046° = 0.13 ERP px"** | 0.046° | **61%** | **contaminated — do not print without re-measuring** |
| P2A "node route +0.004 px median" | — | > 100% | below the floor; not a real measurement |

**Correction (2026-07-28) — the "floor as %" column above is a linear-ratio
heuristic, not a measurement, and it is wrong.** The floor does not add linearly.
Measured contamination at controlled true angles (40k pairs each, pair built by
rotating `a` toward an orthonormal `t` so the haversine form is ground truth by
construction; run in Docker, see `docs/plans/ROADMAP_SEMINARIO.md` §5):

| true angle | `acos` reading | bias | rms |
| --- | --- | --- | --- |
| 0.010° | 0.02809° | +180.87% | 181.10% |
| 0.028° | 0.03095° | +10.55% | 19.68% |
| 0.050° | 0.04818° | −3.63% | 11.53% |
| 0.100° | 0.10085° | +0.85% | 2.71% |
| **0.250°** | 0.25023° | **+0.09%** | **0.42%** |
| 0.500° | 0.50007° | +0.01% | 0.10% |
| 1.000° | 0.99995° | −0.01% | 0.03% |
| ≥ 3° | exact | 0.00% | 0.00% |

So the actives conclusion above is now confirmed *by measurement* (+0.09% bias at
the 0.25° threshold), but the B′ "61%" figure should not be quoted: the true
contamination depends on the error *distribution*, not on a single angle — a
distribution with mass near zero is inflated far more than the fixed-angle row
suggests (identical vectors read 0.028° from a true 0). **Re-measure with
`--geodesic-metric haversine` before printing any sub-0.1° number** (roadmap A1.3).

Both formulas are now selectable at run time via `--geodesic-metric {acos,haversine}`
on `run_oslo_raft.py`, `run_raft_shard_baseline.py` and `run_grid_floor_probe.py`
(default `acos`, so every existing number is reproduced bit-for-bit).

**Consequence for the doublerotation row.** Measured end-to-end on real
flow360:test (40 pairs, both metrics): `--predictor zero` — which is *by
construction* identical to the baseline and must therefore score exactly 0 —
scores **+0.12% global** under `acos` and exactly 0.0000 under `haversine`. The
SLOF doublerotation row's **+0.03 global is inside that artefact**: its positive
sign there is not distinguishable from the metric's own numerical bias. Its
act₀.₅ (+0.12) is ~30× the corresponding artefact (+0.0036) and so is real,
though still negligible. The defensible sentence is therefore: *the only
non-negative row is a trivial zero-predictor, and even its global positive is a
metric artefact.* To be confirmed at full scale by roadmap A1.2 — the effect is a
bias, not zero-mean noise, so it should not shrink with more pairs, but that must
be verified rather than assumed.

Band occupancy on flow360:test also localises the floor precisely: the
`[0, 0.0625°)` band holds **44.5%** of all nodes and is inflated **+60.6%**
(0.0208° → 0.0334°), while every band at or above 0.0625° agrees to within 1.6%
and every band at or above 0.25° to within 0.06%. That is why the global metric
moves (0.2482 → 0.2537) while the actives columns do not.

**Second `acos` site — `logmap`, inside GT construction (found + fixed
2026-07-28).** Running the identity check under `haversine` on the box left a
0.00066° residual, and the bands showed it was **100% in the `[0, 0.0625°)` band**
(50.3% of nodes, 0.0013°; every other band 2.1e-6°), with that band carrying the
*lowest* tangent EPE of all — i.e. the tangent flow was exact and the loss was in
the conversion. Root cause was `geometry.py:207-208`:

```python
dot = (base * endpoints).sum(-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)  # eps=1e-8
theta = torch.acos(dot)
tangent_3d = endpoints - dot * base     # catastrophic cancellation at small theta
```

The `clamp(1 - 1e-8)` put a **hard 0.0081° floor on every GT flow magnitude**, and
`endpoints - dot*base` is a difference of near-equal O(1) vectors. Replaced by the
chord form plus a projection built from `delta = e - p` (identity
`delta - (delta·p)p = sin θ · u`): round-trip error improves 392×–5544× below
0.03° and is now flat at ~1e-6° from 0.001° to 90°, with no change above 0.0625°.

**Scope — verified, not assumed.** The published zero baseline never went through
`logmap` (`metrics.py:90` builds `zero_endpoint` from the node points directly),
and training uses `sequence_geodesic_loss(preds, batch["endpoint"], …)`. A repo
sweep shows the only consumer of the derived `batch["flow"]` inside
`spherical_flow/` is `metrics.py:89` (`tangent_epe_rad`, a secondary column).
**No headline number and no training run is affected.** What was contaminated:
`tangent_epe_rad`, the grid-floor probe's reconstruction target, and the
`--predictor zero` control row — which produced ERP zeros that `logmap` turned
into a spurious 0.0081° displacement instead of exact zero (now verified
`logmap(p, p) → max |flow| = 0.000e+00`).

**Harness measurement floor to declare in the thesis: ~2e-6°** — not 0.028° (old
metric), not 0.00066° (old `logmap`). Five orders of magnitude below the flow360
signal; the metric is no longer a limiting factor in any regime.

The mechanism is a *clamp from below on per-node error*, not additive noise: nodes
whose true error exceeds ~0.05° are measured correctly, while every node below
0.028° reports 0.028°. On flow360 that inflates both the zero baseline and the
model error (partly cancelling in the improvement ratio); on the actives, which
condition on ≥ 0.25° motion, it is immaterial. **No conclusion in §1–§7 changes.**

One-line fix if the sub-0.05° numbers are wanted (changes every number slightly,
so it is a deliberate call, not a silent edit):

```python
return 2.0 * torch.asin((a - b).norm(dim=-1).clamp(max=2.0) / 2.0)
```

**Recommendation:** keep the current metric for all published rows (consistency
with everything already run), state the 0.028° floor explicitly in the methods
chapter as the measurement resolution, and re-measure only the B′ sub-floor claim
before it appears in the thesis.

## 9. HAVERSINE RE-RUN + BAND DECOMPOSITION (2026-07-28) — the regime effect, with magnitude controlled

All 11 rows re-run under the exact metric (`--geodesic-metric haversine`, after the
`logmap` fix) with disjoint motion bands. Same checkpoints, same protocols
(SLOF at `--iters 64 --infer-size 320x640`, PanoFlow native+CFE, RAFT-large
native), replayed from each run's stored `args` via `rerun_from_json.py` so no
flag could drift.

### 9.1 flow360:test — the sole non-negative row was a metric artefact

Zero baseline 0.4314° (was 0.4368 under `acos` — inflated 1.2%), p50 0.1321°.

| row | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles |
| --- | --- | --- | --- | --- | --- |
| SLOF raft (scratch) | −220.97 | −59.66 | −25.85 | −7.11 | −133.26 |
| SLOF raftfinetune | −10.38 | −6.19 | −5.26 | −4.19 | −12.81 |
| SLOF singlerotation | −9.46 | −2.25 | −3.24 | −6.05 | −18.86 |
| SLOF switchrotation | −21.66 | −11.07 | −9.04 | −6.92 | −29.80 |
| SLOF doublerotation | **−0.57** | **−0.10** | +0.03 | +0.33 | −1.16 |
| PanoFlow(CSFlow)+CFE | −11.32 | −5.79 | −4.75 | −4.16 | −13.78 |
| frozen RAFT-large | −15.98 | −7.97 | −7.59 | −7.46 | −23.85 |
| **OSLO EMA final** | −15.12 | −4.92 | −4.04 | −5.70 | −26.89 |

**The predicted sign flip happened.** doublerotation's global went **+0.03 →
−0.57** and its act₀.₂₅ **+0.02 → −0.10**: §8b predicted exactly this, having
measured a spurious +0.12% global that `acos` grants an identity predictor. Under
the exact metric **every row is negative on global and on act₀.₂₅**; doublerotation
retains only +0.03 act₀.₅ / +0.33 act₁.₀, which is the trivial zero-predictor
inheriting the baseline.

The claim hardens accordingly: *on clean cross-pool flow360:test, under a
numerically exact metric, no published method beats zero-flow globally or on the
actives — and the one apparent exception was an artefact of the metric.*

### 9.2 flowscape:test — unchanged, as expected

Zero 3.4024°, p50 2.467°. Large motion is far above any floor, so `haversine`
reproduces the `acos` numbers (+92.6 / +74.4 / +66.0):

| row | global | act₀.₅ | poles | equator | seam |
| --- | --- | --- | --- | --- | --- |
| PanoFlow(CSFlow)+CFE | +92.62 | +92.97 | +95.33 | +85.54 | +63.86 |
| frozen RAFT-large | +74.36 | +74.61 | +54.74 | +80.50 | +49.65 |
| OSLO EMA final | +65.97 | +67.04 | +60.98 | +59.70 | +39.89 |

### 9.3 THE RESULT — same method, same displacement, opposite sign

Bands put both datasets on one displacement axis, so magnitude is **controlled**
rather than aggregated away. Improvement % at matched GT displacement,
flowscape:test / flow360:test:

| band | PanoFlow | frozen RAFT-large | OSLO |
| --- | --- | --- | --- |
| [0,25; 0,5) | +71.5 / **−11.4** | +64.2 / **−10.0** | +2.5 / −9.7 |
| [0,5; 1) | +84.2 / **−6.4** | +78.4 / **−8.0** | +35.7 / **+0.5** |
| [1; 2) | +91.4 / **−6.9** | +87.1 / **−8.9** | +56.5 / −7.5 |
| [2; 4) | +95.2 / **−7.3** | +88.5 / **−11.6** | +73.8 / −8.7 |
| [4; 8) | +97.5 / **−8.5** | +79.7 / **−11.4** | +76.0 / −8.3 |
| [8; 16) | +96.7 / **−3.2** | +60.3 / **−10.0** | +61.4 / −8.7 |

**Swings of 80–105 points at identical displacement.** The original regime-contrast
figure (§7, 103/89/80 pts) compared *aggregates* over datasets whose magnitude
distributions differ, so "it is just the motion magnitude" remained an available
objection. It is no longer available: PanoFlow at 0.5–1° scores **+84.2%** on
flowscape and **−6.4%** on flow360. Magnitude is held fixed; the sign still flips.

**Why band-matching is not full context-matching — and why that is the point.** A
node displaced 0.7° on flowscape sits in a scene whose median is 2.47°: its
neighbours move too, a coherent ego-motion field. The same 0.7° node on flow360
sits in a scene whose median is 0.13°: it is a sparse mover on a static sphere.
Band-matching equalises the node's *own* displacement and deliberately leaves the
*neighbourhood* free — which is exactly the field-structure hypothesis P0d
measured (−153 pts structure vs −36 appearance). This is the same conclusion
reached by a second, independent route.

**PROVENANCE CORRECTION 2026-07-28 — both datasets are RENDERED.** An earlier
draft of this section called flow360 "real video" and offered the real-vs-synthetic
appearance gap as the remaining confound. That is wrong, and the repo already said
so: `docs/THESIS_REGIME_ARGUMENT.md` §1 records FLOW360 (Bhandari et al., ECCV
2022) as **naturalistic *rendered* video**, and §2 spells out what it therefore
lacks — "no rolling shutter, stitching seams, or sensor noise". The adapter
confirms it structurally: `sfprep/adapters/flow360.py` reads dense per-frame GT
(`fflows/NNNN.npy`, `bflows/NNNN.npy`), which no captured 360° footage can supply.

Consequences, in order of importance:

1. **The band-matched result gets STRONGER.** Both legs are renderer output, so
   the sign flip at matched displacement cannot be charged to a real-vs-synthetic
   appearance gap. What still differs is renderer, scene content, and — the
   variable of interest — the **motion-field structure**.
2. **The residual confound is narrower but real:** different renderers and
   different scene content (CARLA driving vs naturalistic interiors/exteriors).
   A3 still earns its keep, because it holds *dataset, renderer and content fixed*
   and varies only magnitude with real structure. Sharpened prediction: the
   real-structure leg stays flat/negative across scales while the rotation leg
   climbs.
3. **The regime must not be named "real-video".** The distinguishing property is
   *consecutive frames at native frame rate ⇒ sub-pixel displacement over most of
   the sphere*, not provenance. Renamed throughout to the **sub-pixel regime**.
   The claim that captured video also lives in this regime is a **frame-rate
   argument, not a measurement** — sound geometry (at 30–60 fps anything not both
   fast and close moves sub-pixel), but it must be labelled as inference. We have
   measured no captured footage, and `THESIS_REGIME_ARGUMENT.md` §2 notes real
   capture would be *harsher* than FLOW360, not easier.

### 9.4 Crossing points

Nobody crosses zero on flow360 except in a narrow window, and only the two
in-domain-ish rows manage it at all:

- **SLOF singlerotation**: positive in [0,25; 2) — +3.1 / +4.5 / +1.5, its best region.
- **OSLO**: positive only in [0,5; 1) — +0.53.
- **PanoFlow and frozen RAFT-large**: never cross zero at any band.

On flowscape all three cross early (PanoFlow between 0.03° and 0.10°, RAFT-large
~0.10–0.19°, OSLO ~0.19–0.37°) and stay positive to 32°.

⇒ **The crossing is a window in the sub-pixel regime and a threshold in the
large-motion regime.** That asymmetry is itself the finding; the original expectation
("all methods cross at roughly the same displacement") is refuted.

**Flag on the top band.** `[32, ∞)` on flowscape has zero-baseline **115.7°** —
beyond 90°, i.e. degenerate/wrap GT — at 0.08% of nodes, and every method scores
≈0 there. Treat that band as unusable rather than as a result.

## 10. A2 (2026-07-28) — the val→test gap is BOTH composition and generalization, ~42/58

Same checkpoint (`P1proper_ema6k`), same metric (`haversine`), same bands, only the
split changes. flow360:val: zero global 0.2017°, p50 0.0976°, act₀.₅ **+4.17**,
global −21.21. flow360:test: zero 0.4314°, p50 0.1321°, act₀.₅ **−4.04**, global −15.12.

### 10.1 Per-band, at matched displacement

| band | val frac | val | test frac | test | Δ (val − test) |
| --- | --- | --- | --- | --- | --- |
| [0; 0,0625) | 42.8% | −1018.1 | 32.8% | −570.3 | *not comparable* |
| [0,0625; 0,125) | 12.7% | −14.3 | 15.8% | −47.5 | +33.2 |
| [0,125; 0,25) | 20.4% | −4.2 | 16.9% | −27.8 | +23.6 |
| [0,25; 0,5) | 11.7% | **+1.8** | 16.6% | −9.7 | +11.5 |
| [0,5; 1) | 10.7% | **+6.3** | 12.2% | **+0.5** | +5.8 |
| [1; 2) | 1.5% | −1.3 | 3.5% | −7.5 | +6.2 |
| [2; 4) | 0.10% | −6.7 | 1.0% | −8.7 | +1.9 |
| [4; 8) | 0.03% | −3.0 | 0.7% | −8.3 | +5.3 |
| [8; 16) | 0.02% | +1.6 | 0.3% | −8.7 | +10.3 |

**The per-band curves are not the same** — val beats test in every comparable band
by 2–33 points. So the gap is **not pure composition**: at the same GT
displacement the model is genuinely better on the pool it was trained near.

Validity of the matching, checked rather than assumed: the within-band mean of the
zero baseline agrees to **0.6%** in the dominant `[0,5; 1)` band, and to 2.7–9.6%
in the sparse higher bands. The lowest band is **excluded** — val's mean there is
0.0103° vs test's 0.0167° (62% apart), so its −1018 vs −570 is not a like-for-like
comparison.

### 10.2 Quantifying the split

Counterfactual: apply val's *per-band* improvements to test's *band composition*
for the act₀.₅ pool (recomposition verified against the JSON: 1.790183 vs
1.7901829).

| | act₀.₅ |
| --- | --- |
| val, actual | **+4.17** |
| counterfactual (val skill, test mix) | **+0.76** |
| test, actual | **−4.04** |

⇒ of the 8.21-point gap, **3.41 pts (42%) is composition** (test carries far more
mass above 1°, where nothing works: 5.8% of nodes vs val's 1.7%) and **4.80 pts
(58%) is generalization** (same-pool advantage at matched displacement).

First-order caveat: the counterfactual assumes per-band skill transfers, which is
the very thing under test; and the higher bands' within-band means differ by up to
10%. Treat 42/58 as an estimate with a few points of slack, not a precise split.

### 10.3 What must be written

The `+4.5% act₀.₅` headline cannot be reported as "held-out val" and left there.
The honest sentence is:

> Our best consolidated result, +4.5% act₀.₅, is measured on a validation split
> carved from the same sequence pool as training. On the disjoint official test
> pool the same checkpoint scores −4.0%. Decomposing by displacement band, ~42% of
> that swing is composition — the test pool carries 3.4× the mass above 1°, where
> no method beats zero — and ~58% is a genuine same-pool generalization advantage.

Note also that OSLO's positive window is **wider and higher on val** ([0,25; 1),
peaking +6.3) than on test ([0,5; 1), +0.5): the crossing window itself shifts with
the pool, which is worth one sentence in the limitations section.

## 11. A3 real leg (2026-07-28) — magnitude swept with real structure held fixed

Same checkpoint (`P1proper_ema6k`), same split (flow360:test), same content, same
renderer. `--val-real-resample-prob 1.0` replaces frame 2 by frame 1 resampled at
the GT endpoints (**photometrically perfect**), and `--real-resample-flow-scale k`
multiplies the real GT field. Only magnitude varies; the field's structure —
support, sparsity, bimodality — is invariant by construction.

| k | measured p50 | global | act₀.₅ | poles | equator |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.132° | −32.08 | −32.64 | −36.10 | −28.49 |
| 2 | 0.264° | −42.99 | −42.54 | −37.84 | −43.41 |
| 5 | 0.657° | −58.95 | −57.07 | −46.24 | −63.97 |
| 10 | 1.307° | −65.44 | −62.57 | −53.34 | −69.65 |
| 20 | 2.587° | **−70.40** | **−67.27** | −57.68 | −73.38 |

**Registered prediction was "flat or negative". The result is stronger: monotone
degradation**, −32% → −70% across a 20× magnitude sweep. More magnitude does not
rescue the real field; it makes things worse.

### 11.1 The decisive pair

| | p50 | structure | appearance | global |
| --- | --- | --- | --- | --- |
| flowscape:test | 2.467° | coherent ego-motion | real render | **+65.97** |
| a3_real_k20 | 2.587° | real (sparse movers) | **perfect (resampled)** | **−70.40** |

Same model, matched median displacement, **136 points apart** — and the *failing*
leg is the one holding the photometric advantage. Magnitude is not the variable.
Structure is.

### 11.2 The appearance sign flip REPLICATES

Compare k=1 against the ordinary test row (identical GT, identical model; the only
difference is whether frame 2 is the real next frame or a perfect resampling):

| frame 2 | global |
| --- | --- |
| real | **−15.12** |
| perfect resampling | **−32.08** |

**Making appearance perfect made the model 17 points worse.** This reproduces
P0d's sign flip (real+real −32.2 vs real+clean −72.5, a 40-point gap) on a
different checkpoint and a different split. Mechanism as diagnosed there: real
inter-frame change damps the correlation's confidence, and on a mostly-static
sphere a damped prediction is closer to zero, hence better. Clean appearance lets
the model commit confidently to the wrong field.

### 11.3 Checks run before drawing the conclusion

- **Not the ERP latitude clamp.** `bilinear_sample_erp` clamps latitude, so a
  resampling artefact would hit the poles hardest. It does not: poles degrade
  −36 → −58 (22 pts) while the equator degrades −28 → −73 (45 pts). The clamp is
  not driving the trend.
- **Do NOT read the band curve across k.** At scale k, band `[a, b)` contains
  nodes whose *original* displacement was `[a/k, b/k)` — k=1's `[0.5, 1°)` are
  genuine movers, k=20's are near-static nodes scaled up. The node population
  changes with k, so cross-k band comparisons are confounded. Only the aggregate
  columns above are used.
- **Degenerate mass grows with k**, as expected from an idealised scaling: the
  `[32, ∞)` fraction rises 0.0011 → 0.0178 and the lowest band's mean falls to
  0.0011°. Both are reported, neither is load-bearing.

### 11.4 The rotation leg — the within-dataset control (2026-07-29)

`--val-synth-rot-prob 1.0 --synth-rot-{min,max}-deg d`, same checkpoint, same
split, same frames. `synth_rotation_record` builds frame 2 with the **same**
`bilinear_sample_erp` call the real-resample leg uses — so both legs have exactly
the same appearance treatment (perfect brightness constancy, resampled frame 2).
Content, renderer, model, metric and appearance are all held fixed. **The only
thing that differs between the two legs is the structure of the motion field.**

| d | measured p50 | global | act₀.₅ | poles | equator |
| --- | --- | --- | --- | --- | --- |
| 0.13° | 0.113° | +1.72 | — | −6.49 | +2.68 |
| 0.26° | 0.225° | +11.25 | — | +8.37 | +11.31 |
| 0.66° | 0.572° | +43.10 | +47.08 | +24.55 | +50.19 |
| 1.32° | 1.143° | +62.16 | +62.80 | +36.08 | +71.78 |
| 2.64° | 2.286° | **+84.13** | **+84.26** | +63.19 | +90.34 |

Sanity check on the harness: measured p50 is `0.866·d` at every rung, to four
digits. That is exactly the median of `sin φ` over a uniform sphere (`median|cos φ| =
0.5 ⇒ √0.75 = 0.866`), i.e. the rotation is being applied as specified.

### 11.5 THE RESULT — the two curves diverge, and the gap grows with magnitude

| p50 | real field | rotation field | gap |
| --- | --- | --- | --- |
| ~0.12° | −32.08 | +1.72 | **34 pts** |
| ~0.25° | −42.99 | +11.25 | **54 pts** |
| ~0.6° | −58.95 | +43.10 | **102 pts** |
| ~1.2° | −65.44 | +62.16 | **128 pts** |
| ~2.4° | −70.40 | **+84.13** | **155 pts** |

The registered prediction ("the rotation leg climbs while the real leg falls")
holds at every rung, and the divergence is monotone: 34 → 155 points. Note the
pairing is **conservative** — the rotation p50 runs ~13% *below* the real p50 at
each rung, and the rotation leg improves with magnitude, so matching exactly would
widen the gap further.

This is the cleanest isolation of the field-structure hypothesis in the project.
Everything the §11.1 cross-dataset pair left open — different renderer, different
content, different scene statistics — is closed here, because both legs are the
same 126M nodes of flow360:test differing only in which vector field frame 1 is
warped by. P0d measured the same thing once (a 4-cell square on one checkpoint);
this measures it at five magnitudes on a different checkpoint and a different
split, and adds the magnitude axis P0d did not have.

Three-way ordering at matched displacement (~2.3–2.6°):

| leg | structure | appearance | global |
| --- | --- | --- | --- |
| a3_rot_d2.64 | coherent rotation | perfect resample | **+84.13** |
| flowscape:test | coherent ego-motion + parallax | real render | **+65.97** |
| a3_real_k20 | real (sparse movers) | perfect resample | **−70.40** |

The two *coherent* legs land 18 points apart across different datasets, different
content and different appearance treatments. The real-field leg is 136–155 points
from both. Structure dominates every other variable measured.

### 11.6 A second, unplanned finding: motion bleeds into static nodes

Every rotation run is **negative in the `[0, 0.0625°)` band**, and gets worse as
the scene moves more:

| d | scene mean motion | near-static band: true | predicted error | band improvement |
| --- | --- | --- | --- | --- |
| 0.13° | 0.102° | 0.0422° | 0.0473° | −12.1 |
| 0.66° | 0.518° | 0.0417° | 0.0479° | −14.9 |
| 1.32° | 1.037° | 0.0417° | 0.0536° | −28.5 |
| 2.64° | 2.073° | 0.0416° | 0.1137° | **−173.1** |

The true displacement at those nodes is constant (~0.042°, they sit near the
rotation axis) while the model's error there grows **2.4×** as the surrounding
sphere speeds up 20×. So motion leaks spatially from moving neighbours into static
nodes — the GRU aggregates over the neighbourhood and the near-axis nodes inherit
it. The leak is real but bounded: if the model simply predicted the ambient
rotation everywhere, the error there would be ~2.07° at d=2.64, not 0.11° (~5% of
ambient).

This matters for **B1 (the static/motion gate)**: it is a direct, magnitude-resolved
measurement of the failure the gate is designed to remove, obtained in a regime
where the model is otherwise working at +84%. The same failure mode that costs a
few points here is the *dominant* term on real flow360 pairs, where near-static
nodes are the majority rather than 0.03% of the sphere.

### 11.7 Caveats carried forward

- **Asymmetric warp artefacts.** A global rotation is a smooth, invertible warp;
  the real-field warp is discontinuous at motion boundaries, so the real leg's
  frame 2 carries disocclusion artefacts the rotation leg does not, and those grow
  with k. Part of that is genuinely field structure (occlusion boundaries *are*
  structure), part is synthesis artefact — this bounds how much of the *growth*
  from 34 → 155 pts is attributable to structure alone. It does not touch the
  qualitative result: at k=1, the least artefact-prone rung and the real
  magnitudes, the sign is already opposite (−32.08 vs +1.72).
- **Polar behaviour under pure rotation.** Poles trail the equator by 26/36/27
  points at d = 0.66/1.32/2.64, and the absolute polar/equatorial error ratio
  climbs 1.5× → 2.3× → 3.8×. This does not contradict the §7 head-to-head (that is
  a *relative* claim: OSLO is 2.3–2.4× flatter than RAFT on the same data, twice
  replicated), but "OSLO is uniform in absolute terms" is not supported at large
  coherent motion. ERP source rasters are heavily oversampled near the poles, so
  part of this may be resampling rather than the model. Not diagnosed.

## 12. A1.3 (2026-07-29) — the B′ claim re-measured, and the acos floor was flattering every model

Five evals on `flow360:val` under `haversine` + bands. The headline claim moves a
little; the P0d decomposition moves a lot, and in a direction that matters.

| cell | ckpt | `acos` | `haversine` | Δ |
| --- | --- | --- | --- | --- |
| rot + clean (`probe_smallrot`) | B′ | +80.63 | **+81.54** | +0.91 |
| real + real (`stageBprime`) | B′ | −32.15 | **−37.97** | −5.82 |
| real + clean (`P0d_realresample`) | B′ | −72.51 | **−80.03** | −7.52 |
| real + clean | A | −118.0 | **−127.69** | −9.69 |
| real + clean | B | −57.61 | **−64.14** | −6.53 |

### 12.1 Why these move 5–10 pts when flow360:test moved 1.1

Entirely the **denominator**. On `flow360:val` the zero baseline reads 0.2105°
under `acos` and **0.20167°** under `haversine` — inflated **4.2%**. The reason is
visible in the bands: `[0, 0.0625°)` holds **42.8% of all nodes** at a true
displacement of **0.01026°**, which is *below* the `acos` floor of 0.02798°. Nearly
half the split sat under the floor, so the baseline every model is scored against
was systematically too large.

The models' own errors (0.28–0.46°) are 10–16× the floor and barely move. That
asymmetry — contaminated numerator, clean denominator, or here the reverse — is
exactly what made the percentage uncorrectable on paper.

**Proof it is the baseline and not the checkpoint:** `a1_bprime_realeval_hav`
returns `global_geo_deg = 0.278230`, against **0.2782** recorded at the end of B′
training under `acos`. Identical to four significant figures. The checkpoint
reproduces exactly; **100% of the −5.82 pt move is the denominator.**

The registered control band ("−32 to −34") was **wrong**, and instructively so: it
was sized from flow360:**test**, where the baseline is inflated only 1.2%. `val`
has far more near-static mass — the same composition difference A2 measured
(val's lowest band 0.0103° vs test's 0.0167°, 62% apart). The control that
actually mattered — the raw predicted error — passed to four digits.

### 12.2 The headline claim, re-measured

| | `acos` | `haversine` |
| --- | --- | --- |
| residual | 0.0457° | **0.04357°** |
| improvement | +80.63% | **+81.54%** |
| ERP px (512×1024, 0.352°/px eq.) | 0.130 px | **0.124 px** |

**Publish 0.044° = 0.12 ERP px, +81.5%.** The claim survives in magnitude — B′ does
resolve a coherent sub-pixel field to about an eighth of an ERP pixel.

Note the direction: `acos` read **4.7% high** here, while the controlled-angle
table predicted ~3.6% *low* at 0.05°. Second time a distribution-level bias failed
to follow the single-angle table — the bias oscillates in sign across the support
(+181% @0.01°, +10.6% @0.028°, −3.6% @0.05°, +0.85% @0.1°), so an error
*distribution* spanning that range averages to something unreadable off the table.
**Never estimate this correction; measure it.**

### 12.3 The P0d square under one exact metric — conclusion holds, both effects bigger

| swap | from → to | Δ |
| --- | --- | --- |
| **motion structure** | rot+clean +81.54 → real+clean −80.03 | **−161.6 pts** |
| **appearance** | real+clean −80.03 → real+real −37.97 | **+42.1 pts** |

Ratio **3.84×** (it was 3.80× under `acos`). The field dominates appearance ~4:1,
and the appearance effect keeps its counter-intuitive **positive** sign: degrading
frame 2 to the real one *helps* by 42 points. Checkpoint ordering on real+clean is
also preserved — A −127.69, B −64.14, B′ −80.03, with B′ worse than B.

So the metric correction changes no conclusion in §11 or P0d. It makes the
negative results **more** negative, which is the honest direction: the old floor
was inflating the trivial baseline and therefore flattering every method scored
against it, ours included.

### 12.4 The static-calibration number, now uncontaminated

From `a1_bprime_realeval_hav`, band `[0, 0.0625°)` on real pairs:

| | value | in ERP px |
| --- | --- | --- |
| share of sphere | **42.8%** | — |
| true displacement | 0.01026° | 0.029 px |
| model error | 0.15135° | 0.43 px |
| improvement | **−1375%** | — |

**On 43% of the sphere the scene moves 0.03 of a pixel and the model asserts 15×
that.** Under `acos` this was hidden — the baseline read at the floor rather than
at 0.010°, so the ratio was compressed. This is the cleanest statement of the
static-calibration failure in the project, it is the same failure A3 §11.6 caught
leaking spatially under pure rotation, and it is the direct target of **B1**.

## 13. Grid-floor probe RESULT (2026-07-29) — the grid is NOT the bottleneck, in either regime

The §8 probe finally ran, on **full test sets** under **haversine** (§8's own
pre-registered decision run, never executed until now).

| source | pairs | est | spacing | pwc | uniform | **oracle** |
| --- | --- | --- | --- | --- | --- | --- |
| flowscape:test | 1386 | r4 | 3.665° | 0.3657° | 0.4624° | **0.0616°** |
| flowscape:test | 1386 | r5 | 1.832° | 0.1853° | 0.2870° | **0.0431°** |
| flow360:test | 2567 | r4 | 3.665° | 0.0773° | 0.1142° | **0.0254°** |
| flow360:test | 2567 | r5 | 1.832° | 0.0505° | 0.0757° | **0.0219°** |

**VERDICT — the pre-registered rule fires on the "skip r5" branch, by 16×.** The
rule was: oracle @r4 ≳ 1.0° ⇒ grid IS the ceiling; ≲ 0.4° ⇒ grid is NOT the
bottleneck. Measured **0.0616°** on flowscape. Machinery consistent (pwc < uniform
< … and oracle far below both, as §8 predicted).

**The grid explains ~5% of OSLO's error in BOTH regimes**, which is the headline:

| | OSLO real error | oracle floor @r4 | floor / error |
| --- | --- | --- | --- |
| flowscape:test | 1.158° | 0.0616° | **5.3%** |
| flow360:test | 0.497° | 0.0254° | **5.1%** |

**The number that closes the argument:** the r4 floor (0.0616°) is *below*
PanoFlow's error (0.251°). OSLO's own estimation grid would support **4× better**
accuracy than OSLO achieves ⇒ the grid is not what separates the two methods.
Remaining hypothesis for the 4.6× gap = **capacity + training recipe** (batch 2).

Metric note: flow360's old `acos` oracle was 0.0293° vs **0.0254°** under
haversine — 13% inflated, consistent with §8b's warning that it was sitting on the
0.028° floor. The number is now a real measurement (harness floor ~2e-6°).

### 13.1 NEW FINDING — the seam floor is 11–12× the global floor

| source | seam oracle @r4 | global oracle @r4 | ratio |
| --- | --- | --- | --- |
| flowscape:test | 0.7273° | 0.0616° | **11.8×** |
| flow360:test | 0.2725° | 0.0254° | **10.7×** |

On flowscape the seam floor is **28% of OSLO's seam error** (2.636°, from the
+39.89% seam row of §9.2) ⇒ **the seam column of every published row carries a
large reconstruction floor and must be read with that caveat.** Likely cause is
wrap discontinuity in the source ERP GT rather than the grid (cf. the degenerate
`[32,∞)` band already flagged in §9.3), but this is **not diagnosed**. Poles show
no clean story: flowscape poles floor 0.0823° is *above* its global floor while
flow360 poles 0.0182° is *below* — do not build a claim on it.

### 13.2 The gap this opens: where does OSLO's ACTUAL upsampler sit?

The probe bounds the *family*: one-hot (`pwc`) at 0.3657° and the best possible
convex combination (`oracle`) at 0.0616° on flowscape. **OSLO's own trained
`UpsampleWeightHead` was not measured** — it sits somewhere in that 6× interval.

That interval is worth up to **0.30° on flowscape, i.e. 26% of OSLO's error** — a
larger prize than anything grid refinement offered, and cheaper. Requires a fourth
probe mode that loads a checkpoint and applies the *learned* weights instead of
pwc/uniform/oracle. Small addition to `run_grid_floor_probe.py`; needs
`--init-checkpoint` plumbing.

**Action:** this becomes the cheapest high-value experiment in §8.4 of
`ROADMAP_SEMINARIO.md`, ahead of the equiangular control (which needs a new grid
builder).

### 13.3 `learned` mode IMPLEMENTED + Docker-validated 2026-07-30

Two files changed. `OSLORAFTRetina.forward` gains `return_upsample_weights=False`,
returning the last iteration's weights `[B, N_est, D, K]`. `run_grid_floor_probe.py`
gains a fourth mode plus `--init-checkpoint`, the model-config flags (defaults set to
the trained config: retina 7 / cp 3 / ln 24 / hidden 96 / ctx 64 / flow-scale 0.5 /
rings 2x8 / eval-iters 12) and `--amp`.

**Design point.** The head is conditioned on the GRU hidden state, so the weights do
not exist without a real forward pass. The probe therefore runs the model on the true
pair, takes the last iteration's weights, and applies them to the **perfect** coarse
field. That isolates the upsampler from the estimator: same weights the model would
use, but handed a flawless input. When `--init-checkpoint` is given the probe builds
the **retina** pyramid (same cache filename as `run_oslo_raft.py`, so
`pyramid_ret7_sup6_est4_cp3_cn8_ln24.pt` hits) and uses it for the contributions too,
removing any chance of geometry mismatch between the weights and the transport.

**Validated in Docker (5 properties):**

| check | result |
| --- | --- |
| `(w * contrib).sum(K)` reproduces `convex_upsample` | max dev **7.5e-9** (fp32 noise) — also verified for the `pwc` and `uniform` weight sets |
| `return_upsample_weights=True` does not perturb predictions | **bit-identical** |
| weights shape / simplex | `[1, N_est, D, K]`, sum over K = 1 (max dev 1.8e-7) |
| base columns unchanged by adding the mode | pwc/uniform/oracle **identical** with and without |
| untrained head end-to-end | lands **exactly on `uniform`** (0.1196°) — correct, since small init ⇒ near-uniform softmax; confirms the plumbing, not just the shapes |
| checkpoint/geometry mismatch | prints `learned mode OFF at rN`, drops to 3 modes, **exit 0** (so probing r4 and r5 with an r4 checkpoint will not crash) |

The equivalence check is the load-bearing one: without it the learned column could be
measuring something the model never emits.

### 13.4 RESULT (2026-07-30) — the trained upsampler is WORSE than one-hot, on both datasets

Full test sets, r4 estimation, EMA final checkpoint, haversine.

| mode | flowscape global | vs oracle | flow360 global | vs oracle |
| --- | --- | --- | --- | --- |
| `oracle` | 0.0616° | — | 0.0254° | — |
| `pwc` | 0.3657° | 5.9× | 0.0773° | 3.0× |
| `uniform` | 0.4624° | 7.5× | 0.1142° | 4.5× |
| **`learned`** | **0.5099°** | **8.3×** | **0.1236°** | **4.9×** |

**The trained head loses to one-hot by 39% (flowscape) and 60% (flow360)**, and loses
to plain uniform averaging by 10% and 8%. It captures none of the available gain.

**Diagnosis — the head barely discriminates.** On both datasets `learned` sits within
8–10% of `uniform`, and the local validation showed an *untrained* head lands exactly
on `uniform` (small init ⇒ near-uniform softmax). So the trained head behaves close to
an initialized one: it spreads weight over the neighborhood, which blurs the field,
which is why it loses even to taking the center node alone. Likely cause is gradient
pressure — the model's error (1.158° / 0.497°) is 3–6× the `pwc` floor, so the
upsampling term was never the binding term in the loss.

**Scale of the opportunity, stated carefully.** On flowscape the learned
reconstruction is 0.5099°, i.e. 44% of OSLO's 1.158°, and **2.0× PanoFlow's entire
error (0.251°)** — with the current head, no estimator improvement can reach PanoFlow.

**⚠ NOT a strict decomposition.** At the seam `learned` reads 3.875° while OSLO's
actual seam error is 2.636° — the measured value *exceeds* the model's error, so this
column is **not a lower bound** on model error. It measures upsampler fidelity given
perfect input. Where the true GT carries wrap discontinuities, faithfully
reconstructing it is worse than the smooth field the model emits. The defensible claim
is narrower: **the upsampler discards information the grid demonstrably carries**,
since the oracle recovers it with weights from the same convex family.

**Alternative explanation not yet excluded:** co-adaptation. The head was trained
jointly against the estimator's *imperfect* coarse field and may be compensating a
systematic bias that becomes distortion on a perfect field.

### 13.5 Two measurements that separate the explanations

1. **Weight entropy / max-weight.** Near-uniform softmax ⇒ mean max weight ≈ 1/K =
   0.11. Concentrated ⇒ higher. Separates "did not learn" from "learned something
   strange". One diagnostic added to the probe.
2. **End-to-end one-hot swap (decisive).** Evaluate the model normally but replace the
   learned weights with `pwc` inside `convex_upsample`. Model **improves** ⇒ the head is
   actively hurting and the fix is trivial. Model **degrades** ⇒ co-adaptation is real
   and the perfect-input probe does not see the work the head is doing.

Given `pwc` beats `learned` by 39% on perfect input, (2) is expected to improve — but
it is the only test that settles it, because it is the only one run on the field the
head was actually trained against.
