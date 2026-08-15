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

> **CORRECTION (2026-08-03, §16.1).** The "All" column above is their *weighted*
> one (`Weighted s≥0*` = EPEd, poles counted double). Table I also carries an
> unweighted `s≥0` column, which is the one the buckets belong to: SLOF v1
> **1.568**, v2 1.615, RAFT-ft 1.624, RAFT 2.058, KTN 2.222. Every global
> statement in this section must be re-read against those numbers, which sit
> much closer to the zero baseline — that is what §16 measures. The header's
> "iters=64" is also wrong: the shipped CSVs are iters=12.

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

### 13.6 END-TO-END SWAP (2026-07-30) — §13.4's actionable claim is REFUTED; the head is a denoiser

`--upsample-weights {learned,pwc}` on the EMA checkpoint, haversine, full test sets.

| | `learned` | `pwc` | delta |
| --- | --- | --- | --- |
| flowscape global | **1.1578° (+65.97)** | 1.2077° (+64.50) | pwc −1.47 pts |
| flowscape poles | **3.1471° (+60.98)** | 3.3819° (+58.07) | pwc −2.91 pts |
| flowscape act₀.₅ | **+67.04** | +65.56 | pwc −1.48 pts |
| flow360 global | **0.4967° (−15.12)** | 0.5098° (−18.17) | pwc −3.04 pts |
| flow360 poles | **0.5781° (−26.89)** | 0.6091° (−33.69) | pwc −6.80 pts |

**Swapping to one-hot makes the model WORSE everywhere. Co-adaptation is the answer.**

Control passed: the `learned` leg reproduces the published rows exactly (flowscape
+65.97 / poles 3.147 / act₀.₅ +67.04; flow360 −15.12 / act₀.₅ −4.04 / poles −26.89),
so the swap machinery does not perturb the baseline.

**Correction to §13.4.** That section called the upsampler "the single largest
identified component of OSLO's error" (44% on flowscape) and "2.0× PanoFlow's entire
error … a self-imposed ceiling". **The actionable part is wrong.** The head is not
hurting: replacing it degrades the model. §13.4's own co-adaptation caveat was the
correct reading and this settles it.

**What the two measurements together actually say.** On a *perfect* coarse field the
learned weights lose to one-hot, because smoothing an exact field only blurs it. On the
*model's* coarse field they win, because that field is noisy and the near-uniform
softmax is a smoothing operator. So the head did not "fail to learn to interpolate" —
**it learned to denoise**, which is correct given the estimator feeding it. Entropy
≈ log K is the signature of a filter, not of an untrained layer.

**Consequence for the plan:** the oracle headroom (0.062° vs 0.510°) is **not reachable
by changing the upsampler**. It is gated by coarse-field quality. This does not redirect
§8.0 — it reinforces it: the bottleneck is the estimator (capacity + training recipe).
A retrained upsampler only pays off *after* the estimator improves, since the current
head is tuned to the current noise level.

**Side finding — part of the polar advantage is the upsampler, not the sampling.**
`pwc` costs more at the poles than on average (flowscape +7.5% error at the poles vs
+2.5% at the equator; flow360 +5.4% vs +1.6%). The smoothing is load-bearing for the
polar/uniformity result, which is the article's headline positive. That is a caveat to
declare, and another reason the equiangular control (§8.4 A1) matters: it separates
sampling from architecture, and this smoothing sits on the architecture side.

---

## 14. A1 — equiangular control, round 1 (2026-07-30): the instrument was not grid-invariant

Matched-pair from-scratch training, `replica360:train → replica360:val`, 2000 steps,
one-cycle, seed 7, haversine, git `5c94e3e52`. Only `--grid` differs. Both legs are
1,558,768 parameters, retina r7 / supervision r6 / estimation r4.

### 14.1 What the run reported (node-weighted metrics — superseded)

| | HEALPix | equiangular |
|---|---|---|
| global | 3.005° (77.80%) | 2.477° (81.72%) |
| poles | 5.654° (59.49%) | 4.337° (68.97%) |
| equator | 2.390° (82.00%) | 1.315° (89.90%) |
| poles/equator | 2.37× | 3.30× |

Read naively this fires the pre-registered "polos comparáveis" branch and refutes
"equal-area sampling buys polar accuracy". Two things block that reading.

### 14.2 The band decomposition: HEALPix wins nine of eleven bands

Contribution to the global mean, `Σ frac × error`:

| faixa | HEALPix | equiangular | winner |
|---|---|---|---|
| 0–0.0625° | −3200% | −3897% | HP |
| 0.0625–0.125° | −828% | −1290% | HP |
| 0.125–0.25° | −372% | −602% | HP |
| 0.25–0.5° | −131% | −226% | HP |
| 0.5–1° | −11.6% | −58.4% | HP |
| 1–2° | +41.5% | +17.4% | HP |
| 2–4° | +70.3% | +57.9% | HP |
| 4–8° | +82.9% | +77.8% | HP |
| 8–16° | +88.4% | +85.2% | HP |
| 16–32° | +82.0% | +86.4% | EQ |
| 32°+ | +22.6% | +58.4% | EQ |

Bands below 16° give HEALPix **+0.261°**; bands above give equiangular **+0.789°**;
net +0.528° for equiangular. The **32°+ band alone (3.3% of nodes) is 0.505°, i.e. 96%
of the net global difference**. The region-level result is that one tail band leaking
through the aggregate.

### 14.3 The blocking defect: node-weighted means are not comparable across grids

`accumulate_maps` averaged over nodes. That equals the per-area average only on an
equal-area grid. The `poles` mask (`|lat| ≥ 60°`) covers 13.4% of HEALPix nodes and
**32.8%** of equiangular nodes, so every aggregate — global, regional, per band —
asked a different question on each leg.

Direction of the bias: equiangular over-weights the poles, where error is highest, so
its numbers were **penalised**. Reconstructing the per-area global from the three region
means flips the flow360 verdict (0.933° vs HEALPix 0.970°) and widens the replica360
one (1.889° vs 3.005°). The tail-band decomposition in §14.2 carries the same defect.

**Fixed** (`--metric-node-weights {area,uniform}`, default `area`): each node is weighted
by its exact cell solid angle, `(2π/n_lon)(cos θ_k − cos θ_{k+1})`, normalized to mean 1.
On HEALPix the weights are uniform and `node_weights` stays `None`, so every recorded
number reproduces bit-for-bit — the HEALPix re-run is a regression test. Validated: the
weights sum to 4π; the pole/equator ratio equals the `sin θ` ratio; the mean of `z²` over
the sphere reads 0.33335 weighted against 0.5 unweighted (truth 1/3); streamed and
one-shot paths agree; `_frac` becomes an area fraction.

**Residual, not fixable by weighting.** Region masks snap to whole cell rows, so the
`poles` mask covers 13.0% of the sphere on equiangular against 13.4% on HEALPix (equator
49.3% vs 50.0%). Region comparisons across grids stay approximate to ~3% relative; the
global mask has no edge and is exact.

**Methodological note.** This is the third time a component measured out of its operating
regime produced an actionable-looking claim (§13.4 upsampler; the "10× handicap" in §8.0;
now this). The pattern: before comparing two configurations, check that the *measurement*
means the same thing in both.

### 14.4 Pre-registered reading for the corrected re-run

Metrics of interest, per-area weighted, each leg against its own zero baseline:

- **`poles_improvement_pct` and the poles/equator error ratio.** Equiangular collapsing
  at the poles ⇒ equal-area sampling is the cause of the polar result. Comparable or
  better ⇒ the architectural claim must be reformulated, and the polar advantage over
  RAFT belongs to something else (spherical convolution, geodesic loss, tangent-space
  flow, or the smoothing upsampler of §13.6).
- **The band table.** If HEALPix still wins the sub-16° bands after weighting, the grid
  choice is a range/resolution trade rather than an accuracy claim, and equal-area is
  justified for the regimes the thesis targets.
- **Uniformity** is the reading most exposed to the mask-snapping residual above; treat
  a difference under ~5% as noise.

Falsifiers standing: one seed (project spread is ±14% relative, the observed global gap
was 21%), one dataset per regime, 2000 steps. The seed-11 pair is blocking before any
claim leaves this document.

### 14.5 Corrected result (per-area metric, git `d2961ce9b`)

Regression check first: the HEALPix leg re-evaluated from its checkpoint reproduces
`global_geo_deg = 3.005125431360787` — all sixteen digits of the original run, every band
identical. The refactor changed nothing that was already recorded.

**replica360:val** (in-domain, p50 ≈ 12°)

| | HEALPix | equiangular |
|---|---|---|
| global | 3.005° (+77.80%) | **1.816° (+86.42%)** |
| poles | 5.654° (+59.49%) | **3.699° (+73.49%)** |
| equator | 2.390° (+82.00%) | **1.312° (+89.92%)** |
| seam | 5.251° (+69.09%) | 2.190° (+85.43%) |
| poles/equator | **2.37×** | 2.82× |

**flow360:test** (zero-shot, p50 ≈ 0.11°). The HEALPix leg is unchanged by construction
(`node_weights` stays `None` on an equal-area grid), so §14.1's numbers already are the
per-area numbers.

| | HEALPix | equiangular |
|---|---|---|
| global | 0.970° (−185.2%) | **0.909° (−162.3%)** |
| poles | 1.728° (−373.3%) | **1.448° (−287.3%)** |
| equator | **0.672° (−112.2%)** | 0.701° (−118.0%) |
| poles/equator | 2.57× | **2.07×** |

**The §14.4 prediction failed.** The equiangular leg wins global and poles in *both*
regimes, and on flow360 it wins uniformity too. What survives for HEALPix: uniformity on
replica360, the equator on flow360, and the sub-4° bands on replica360 — but that last
advantage totals **0.009°**, negligible against a 1.189° global gap of which 94% sits in
the two bands above 16°.

Taken at face value this refutes the thesis' central geometric premise: equal-area
sampling does not buy polar accuracy, and does not reliably buy uniformity either.

### 14.6 Before that stands — the raster-alignment confound

The shards store ERP images and ERP flow, and the equiangular grid **is** the ERP
geometry. At zero rotation its nodes resample source pixels near 1:1, most tightly at the
poles where ERP is densest, while HEALPix nodes always fall between ERP pixels and take
a bilinear blur. That is a data-pipeline advantage, not a geometric one, and it would be
largest exactly where the measured gaps are largest.

**Control** (`--val-so3-prob`, default 0 so every recorded number is untouched): rotate
the validation stream. HEALPix keeps its sampling geometry under rotation; the
equiangular grid loses its alignment with the raster. The rotation draw is seeded
`seed + epoch*131 + worker_id`, independent of the grid, so both legs see an identical
rotation sequence on identical records — the pair stays matched.

Pre-registered reading, comparing the two legs *within* the rotated run only (rotated
numbers are not comparable to the unrotated ones, since the polar mask then holds
different scene content):

- Equiangular advantage **survives** ⇒ alignment is not the cause; the grid genuinely
  helps and §14.5 stands as written.
- Equiangular advantage **vanishes or inverts** ⇒ §14.5 measured the data pipeline, the
  equal-area claim is rehabilitated, and the correct statement becomes that equal-area
  sampling costs nothing while removing a dependence on the source raster's own grid.

Still blocking either way: seed 11. The project's seed spread is ±14% relative; the
replica360 gap here is 40%, comfortably outside it, but flow360's global gap is 6.2% and
would not survive a seed swing on its own.

### 14.7 Seed replication — the A1 result is not a seed artefact

replica360:val, per-area, seeds 7 and 11, same recipe, only `--grid` differs.

| | HP s7 | HP s11 | EQ s7 | EQ s11 |
|---|---|---|---|---|
| global | 3.005 | 2.648 | **1.816** | **1.848** |
| poles | 5.654 | 4.923 | **3.699** | **3.608** |
| equator | 2.390 | 2.114 | **1.312** | **1.371** |
| poles/equator | **2.37×** | **2.33×** | 2.82× | 2.63× |

The seed ranges do **not overlap in any region**. Worst-case equiangular (1.848°) still
beats best-case HEALPix (2.648°) by 30% global, 25% at the poles, 35% at the equator.
Two-seed means: 2.826° vs 1.832°, a 35% error reduction.

The HEALPix uniformity advantage also replicates (2.37/2.33 vs 2.82/2.63, a 14% gap,
above the ~5% mask-snapping noise threshold declared in §14.4). It is now the only
property the equal-area grid is measured to buy.

**Side finding: the equiangular leg is far more seed-stable** — global spread 1.8% vs
HEALPix 12.6%. The project's ±14% seed band was measured on HEALPix and does not
transfer to the equiangular grid.

§14.6's raster-alignment control remains the open question; the seed falsifier is closed.

### 14.8 A1 CLOSED — equal-area sampling is not the mechanism

Raster-alignment control (`--val-so3-prob 1.0`, seed 7, per-area, matched rotations).

**replica360:val**

| | HEALPix | equiangular | gap |
|---|---|---|---|
| global | 2.946° (+78.03%) | **1.966° (+85.34%)** | −33.3% |
| poles | 5.670° (+59.54%) | **3.564° (+74.57%)** | −37.1% |
| equator | 2.316° (+82.38%) | **1.588° (+87.92%)** | −31.4% |
| poles/equator | 2.45× | **2.24×** | |

**flow360:test**

| | HEALPix | equiangular | gap |
|---|---|---|---|
| global | 1.082° (−216.0%) | **0.909° (−165.5%)** | −16.0% |
| poles | 1.550° (−355.8%) | **1.232° (−262.8%)** | −20.6% |
| equator | 0.955° (−176.4%) | **0.846° (−144.5%)** | −11.4% |
| poles/equator | 1.62× | **1.46×** | |

**The confound is eliminated, and the evidence runs opposite to its prediction.**
Destroying the ERP alignment cost the equiangular leg nothing on flow360 (0.9094° →
0.9092°, 0.02%) and cost HEALPix **11.6%** (0.970° → 1.082°); the gap widened from 6.2%
to 16.0%. The mechanism for that inversion is not established and is left open — what
matters is that the alignment hypothesis predicted the reverse.

**Uniformity falls too.** Under rotation the scene content is statistically uniform
across regions, which isolates *grid* anisotropy from the *scene's* vertical structure.
In that reading equiangular is flatter on both datasets (2.24× vs 2.45×; 1.46× vs 1.62×).
The HEALPix 2.37×/2.33× advantage exists only in the axis-aligned replica360 view.

**Verdict.** Same architecture, same 1,558,768 parameters, same recipe, only node
placement differs. Equal-area sampling buys neither polar accuracy nor uniformity; it
costs ~33% global error on replica360 (two seeds, non-overlapping) and 6–16% on flow360,
and it trains less reproducibly (seed spread 12.6% vs 1.8%). Both readings — axis-aligned
and content-averaged — agree, so neither raster alignment nor scene structure explains it.

**What this does NOT say.** The equiangular grid here is still a graph on the sphere:
geodesic loss, tangent-space flow, wrap-around neighborhoods, no ERP-projection distortion
inside the convolution kernel. This is not "ERP + CNN", and the OSLO-vs-RAFT polar result
is untouched as a *measurement*. What falls is the *attribution*: the advantage over an
ERP-raster CNN comes from somewhere else in the architecture — spherical convolution,
geodesic loss, tangent-space representation, or the smoothing upsampler that §13.6 already
showed is load-bearing for the polar number. Sphere-native beats raster-native; equal-area
node placement is not the reason, and is measurably the worse choice.

**Scope.** 2000 steps, replica360 training only, one architecture. Whether the ordering
survives a full-length run on the production mix is untested.

## 15. A1 at production scale (2026-08-02) — the 20k equiangular leg

`P1proper_mix20k_a1eq`, git `5d7fc34aa`. The P1-proper recipe replayed with `--grid
equiangular` as the only change: same 1,558,768 parameters, same mix
(`chairs360:train,flowscape:train,flow360:train`), same seed 7, same 20000 steps, same
warm start from the HEALPix-trained `oslo_raft_retina_stageA`. Metrics are haversine and
per-area. Wall clock **22.8 h**, roughly four times my estimate — the equiangular
neighbour graph at retina level 7 is materially slower per step than HEALPix, and any
future scheduling should use the measured rate, not the HEALPix one.

### 15.1 The endpoint, and why it is not yet the answer

| flow360:val | model | zero | improvement |
|---|---|---|---|
| global | 0.2785° | 0.2066° | **−34.85%** |
| active ≥ 0.25° | 0.6692° | 0.6354° | **−5.32%** |
| active ≥ 0.5° | 0.9380° | 0.8943° | **−4.89%** |
| active ≥ 1.0° | 2.1581° | 2.0933° | **−3.09%** |
| poles | 0.2754° | 0.1576° | −74.77% |
| equator | 0.2627° | 0.2133° | −23.16% |
| seam | 0.4306° | 0.3578° | −20.33% |

Negative at every threshold and every region, and the pre-registered gate (act₀.₅ > +5.2%)
is not met. Three things block reading that as a falsification, in descending order of
how cheaply they can be removed.

**One: this is not the metric the HEALPix record was measured in.** The `+4.0` raw and
`+4.5±0.9` EMA numbers for HEALPix date to 2026-07-23/24, before the haversine switch
(§12) and before per-area weighting (§14.3). §12 showed the acos floor inflated the zero
baseline and therefore flattered every improvement figure by 5–10 points on these splits.
Comparing `−4.89` against `+4.0` compares two different rulers. Removing this costs one
eval-only pass over the HEALPix 20k endpoint and is the single highest-value next action.

**Two: the endpoint is one draw from a noisy trajectory.** The per-eval act₀.₅ series runs
−38.0 … +7.4, and the run crossed the gate twice on the way through (+6.75 at 5k, +7.37 at
6k). Over the deep-anneal half (11k–20k) the centre is **−4.6 ± 6.6** (s.e. 2.1), so the
endpoint sits essentially at the basin centre rather than on an excursion — but the raw
spread here (σ 11.1 over all evals) is the same pathology that made the EMA stage necessary
for HEALPix in the first place, where EMA cut σ from 8.6 to 0.9. The decision variable for
this recipe has been the EMA point, not the raw endpoint, since 2026-07-23.

**Three: the warm start is a HEALPix Stage A**, declared in advance as a handicap on the
equiangular leg. That was registered before the run precisely so a loss would not be
over-read: a loss under this design licenses "does not transfer from a HEALPix
initialisation at this scale", not "equiangular is worse at scale".

### 15.2 An unplanned reading: the error budget is flat, the signal is not

In absolute degrees the equiangular error is nearly isotropic — poles 0.2754° against
equator 0.2627°, a ratio of **1.05×**. The *signal* is not: the zero baseline is 0.1576° at
the poles against 0.2133° at the equator, ratio 0.74×. The model therefore spends about the
same error budget per unit solid angle everywhere, regardless of how much motion is locally
present, which is exactly why the polar percentage is the worst region while being the most
accurate region in degrees. This is the static-calibration problem of §12.4 in its cleanest
form, and it is orthogonal to the grid question.

### 15.3 The EMA point (`P1proper_ema6k_a1eq`, 7.5 h)

The 6k polish stage replayed on the equiangular grid, warm-started from the 20k endpoint
above. Same recipe, `--ema-decay 0.999`, `--lr 3e-05`, no one-cycle.

| flow360:val | raw @6k | **EMA @6k** | zero |
|---|---|---|---|
| act ≥ 0.25° | −0.65% | **−0.22%** | 0.6354° |
| act ≥ 0.5° | −0.63% | **+0.33%** | 0.8943° |
| act ≥ 1.0° | −2.38% | **−0.93%** | 2.0933° |
| global | −23.60% | **−21.22%** | 0.2066° |
| poles | −47.06% | **−49.98%** | 0.1576° |
| equator | −14.46% | **−11.59%** | 0.2133° |

**The EMA instrument transfers to the equiangular grid intact.** Raw act₀.₅ over the twelve
evals of this stage has σ 3.79 around a centre of −0.71; the EMA series has **σ 0.65** around
−0.33 over the last six, a 5.8× variance reduction of the same order as the 9× recorded for
HEALPix on 2026-07-23. §15.1's second objection is therefore removed: the equiangular basin
is well-defined, not a noise band.

**The EMA series is monotone-ish upward and had not plateaued** — −3.41, −2.00, −1.70,
−1.23, −0.97, −0.56, −0.61, −1.39, −0.15, −0.47, +0.30, **+0.33**, with the last two points
the best of the run. That rise is about +0.30 per 1000 steps from 3k onward. Reaching +5.2
from +0.33 at that rate needs roughly 16000 further polish steps under an assumption of
linearity that nothing supports. A longer polish is not a path to the gate.

**The gate as literally written is not met.** +0.33 ± 0.65 against a +5.2 threshold is
7.5 sigma. What is *not* settled is whether +5.2 is still the right threshold, because that
number was fixed under the acos metric and §12 showed acos inflated the zero baseline and
so flattered every improvement figure by 5–10 points on these splits. The correction moves
both legs the same direction, which is exactly why the paired reading decides this and the
absolute number cannot:

- if the HEALPix EMA re-reads near +4.5 under haversine, equiangular loses by ~4 points;
- if it re-reads near +0.5, the two grids tie and the §14 ordering simply does not replicate
  at production scale;
- if it re-reads negative, equiangular wins at scale.

All three are live. **A gate threshold stated as an absolute percentage is not portable
across a metric change** — restating it, either in the metric it will be adjudicated in or
as a margin over the HEALPix leg measured the same way, is a methodological debt this run
exposed and did not create.

### 15.4 The isotropy result, and its caveat

At the EMA point the equiangular error field is isotropic to within one percent: poles
0.2363° against equator 0.2380°, ratio **0.99×**. The §15.2 observation from the 20k
endpoint holds and tightens.

This is the flattest error field the project has measured, and it replicates §14.8's
finding that equal-area node placement is not what buys uniformity. The caveat from §15.2
stands and must travel with the number: the *signal* is not isotropic (zero baseline 0.158°
polar against 0.213° equatorial, 0.74×), so a model that tracked local motion strength would
show *less* error at the poles, not equal error. A flat error field is partly skill and
partly the static-calibration failure of §12.4. The same caveat applies to the OSLO-vs-RAFT
uniformity claim, which has never carried it.

### 15.5 THE PAIRED READING (2026-08-02) — A1 does not replicate at production scale

`P1proper_mix20k_havbase` and `P1proper_ema6k_havbase`, eval-only, 150 s each. All four
columns are now flow360:val under haversine + per-area, so the comparison is finally
like-for-like.

| improvement % | HP 20k raw | EQ 20k raw | **HP EMA** | **EQ EMA** |
|---|---|---|---|---|
| act ≥ 0.25° | +1.05 | −5.32 | **+3.49** | −0.22 |
| act ≥ 0.5° | +4.00 | −4.89 | **+4.17** | +0.33 |
| act ≥ 1.0° | −0.64 | −3.09 | −1.28 | **−0.93** |
| global | −36.65 | −34.85 | −21.21 | −21.22 |
| poles | −118.88 | −74.77 | −69.62 | **−49.98** |
| equator | −21.02 | −23.16 | **−11.15** | −11.59 |
| seam | −21.77 | −20.33 | −14.57 | **−12.71** |

**First: the metric change did NOT move the active metrics.** The shipped model re-reads at
act₀.₅ **+4.17%** under haversine + per-area against the **+4.5 ± 0.9** recorded on
2026-07-24 under acos + node-mean — inside the EMA band. My §15.3 expectation of a 5–10
point deflation was wrong for this metric on this split. It was right for *global*, which
moves from −16.8 to −21.21, a 4.4-point loss. The reason is structural: the acos floor bites
at tiny angles, which dominate global (43% of the sphere sits below 0.0625°) and which the
active thresholds exclude by construction. **The R2 threshold therefore needs no restating
for the active metrics**, and the 2026-07-24 closure of P1 — Gate R2 approached, not met,
at ~80% — survives the metric fix intact. The §15.3 debt is discharged, not carried.

**Second: the §14 ordering does not replicate, and it does not simply invert either.**

- Where there *is* motion, HEALPix wins clearly: **+4.17 vs +0.33** on act₀.₅, **+3.49 vs
  −0.22** on act₀.₂₅. This is the headline metric of the whole P1 campaign, and equiangular
  loses it by ~4 points.
- Global is a dead tie: −21.21 vs −21.22, with HEALPix 2.4% better in degrees.
- At the poles equiangular wins by **20 points** of improvement (0.2363° vs 0.2533°, 6.7%
  better in degrees), and it wins the seam by 1.9 points.
- Equiangular is flatter, but far less dramatically than §14.8 suggested: poles/equator in
  degrees **0.99× vs 1.08×**, against 2.24× vs 2.45× on replica360. flow360's motion is
  small everywhere, which compresses the regional spread for both grids.

So §14.8's "equiangular beats HEALPix everywhere including the poles" holds **only at the
poles** once the run is full-length, on the production mix, at the metric P1 is judged on.
The pre-registered criterion fired against the hypothesis that motivated it.

**Third, a defect this comparison exposed: the two legs are not scored against an identical
target field.** The zero baselines differ — act₀.₅ zero **0.8556° (HP) vs 0.8943° (EQ)**,
4.5% apart, and these are area-weighted, so §14.3's fix does not explain it. The active
*area* fractions match to three decimals (0.12382 vs 0.12394), so set membership is not the
cause. The residual is how each grid samples the ERP ground truth: 49152 nodes arranged
128×384 on equiangular against nside-64 equal-area on HEALPix, with different interpolation
stencils. Smoothing suppresses exactly the fine structure that is hardest to predict, so the
direction of this bias plausibly favours HEALPix — the same direction as the result. **It is
not quantified and it is not removed by area weighting.** Any use of this table must carry
that caveat.

**Fourth, a smaller leftover of §14.3:** `target_geo_deg_p50/p90/p95` are still node-weighted
and therefore not grid-comparable (p50 0.0976 HP vs 0.0849 EQ, 15% apart, entirely explained
by equiangular's polar node oversampling). The headline metrics are unaffected because the
active thresholds are absolute in degrees, but the quantile readout should not be compared
across grids until it is weighted.

### 15.6 Verdict

At 2000 steps on replica360, equiangular beat HEALPix everywhere (§14.8). At 20000 steps on
the production mix it loses the active subsets by ~4 points, ties global, and keeps only the
polar and seam advantage. The honest statement is **partial replication with a sign flip on
the headline metric**, under a warm start that was declared in advance to handicap the
equiangular leg and a target-sampling residual that runs the same way.

What this does *not* touch: §14.8's negative claim. Equal-area node placement still is not
the mechanism behind OSLO's polar accuracy — the equiangular grid still wins the poles here,
at production scale, with the same parameters. What falls is the stronger reading that
equiangular is simply the better grid.

**No further runs are warranted on this question.** Removing the two remaining confounds
costs a from-scratch equiangular Stage A plus a matched-sampling target readout, which is a
new campaign, not a control. The result as it stands is reportable with its scope attached.

## 16. SLOF's Table I reproduced in SLOF's own metric (2026-08-03) — OPENED

Every universality row so far is scored in *our* metric: geodesic degrees on an
equal-area grid, actives conditioned on true angular displacement. A referee is
entitled to reply "that is not the metric the field uses, and your baseline is a
consequence of your metric". This activity removes that reply. It re-implements
their evaluation from their released code, runs it on their split with their
checkpoints, and adds the one row their table does not have: **zero flow**. If
the reproduction lands on their published numbers, the zero row is a number from
*their* table.

Tool: `run_slof_table1.py` (source `SLOF/evaluate_raft.py` + `dataloader.py` +
`utils.py`, read line by line from the released tarball). Published values are
embedded from the CSVs shipped in `SLOF/quantitative_results/`, so every run
prints its own agreement.

### 16.1 The protocol audit — five deviations, three of them ours

Reading their code before running it turned up more than expected.

1. **The pair set is half of ours.** `ReadData` indexes forward flows only
   (`sorted(sequences) × sorted(frames)[:-1]`): **1289 pairs** on the official
   test split. Our universality rows ran 2567 — the same frames plus every
   backward flow. Not an error on either side, but the two pair sets are not the
   same population.
2. **The published runs are `iters=12`, not 64.** `train.py:294` calls
   `validate_flow360(..., iters=12)`, and the shipped
   `TEMP001_test_iters_12_rotation_False_final.csv` is numerically identical to
   `single_rotation.csv`. `evaluate_raft.py`'s own `__main__` defaults to 64,
   which is where our §3 protocol note came from. Our rows therefore gave their
   checkpoints *more* refinement than the paper did.
3. **Their loader feeds frames in [0, 1] to a forward that normalizes for
   [0, 255].** `Flow360Loader.transform_frame` ends in `ToTensor()`; `RAFT.forward`
   opens with `2*(image/255) - 1`. The network sees a near-constant −1 field with
   the image content compressed by 1/255. It is scale-invariant in `fnet`
   (instance norm) but **not** in `cnet` (batch norm, eval mode, running stats
   frozen at whatever scale training used) — and their training used the same
   loader, so the checkpoints are *fitted to that range*. **Our rows fed 0–255**,
   i.e. they ran their checkpoints outside the range they were trained in. First
   4 test pairs, `singlerotation`, iters 12, EPE in the `lt5` bucket: **0.288
   (unit) vs 0.524 (byte)** — an 82% penalty that is ours, not theirs.
   `run_slof_table1.py --input-scale` and `run_raft_shard_baseline.py
   --princeton-input-scale` now expose it; the default of the latter stays `byte`
   so no existing number silently changes.
4. **The GT sign agrees with ours.** `ReadData` negates the `.npy` and
   `Flow360Loader` negates it again, so the evaluated GT is the raw file — which
   is exactly what sfprep pins for `flow360` (`default_convention = "identity"`).
   One less thing to worry about.
5. **Our §1 transcription of Table I mixed two columns** (corrected in place
   above). The paper reports `Weighted s≥0*` (EPEd) *and* `s≥0` (plain EPE); §1
   took the weighted number as the global and compared it against an unweighted
   zero estimate. The unweighted global for SLOF v1 is **1.568**, not 2.548, and
   §1's "1.3–2.9× above zero" reading was inflated by that mix. What the true
   margin is, is the measurement below.

Reproduced faithfully, including three things that are defects on their side and
are flagged in the JSON under `protocol`: the magnitude buckets are **cumulative
and overlapping** (`lt5 ⊂ lt10 ⊂ lt20`, `gte20` the complement) rather than
disjoint; there is **no validity mask** anywhere in the evaluation; and the
angular error normalizes `ugt` before reusing it to normalize `vgt`, then
stretches the cosine to [−1, 1] by the **min/max of the current batch**, so AE is
a function of batch composition. EPE is unaffected by all three and is what the
comparison rests on.

### 16.2 The harness

`run_slof_table1.py` streams their pair list, applies their transforms (PIL
resize to 320×640 for frames, normalize/`F.interpolate` at its *nearest* default/
rescale for flow), and accumulates per-bucket sums and counts — algebraically the
same as their concatenate-then-`np.mean`, since every selected pixel carries the
same weight. It always scores **both** the checkpoint and zero flow in the same
pass, on the same pixels, so the two rows cannot drift apart. `--source shards`
runs the identical protocol off the sfprep tars instead, which measures what our
own pipeline costs (frames went through JPEG, flow through float16) rather than
their table.

Docker-validated locally (4 pairs, CPU, both input scales, `singlerotation`);
full-split runs follow.

### 16.3 Pre-registered reading (written before the full-split numbers landed)

**Gate first.** The reproduction is only usable if `singlerotation` at
`--input-scale unit --iters 12` lands on their shipped CSV: 1.568181 / 0.309022 /
62.475649 (s≥0, lt5, gte20) and 2.548246 on the weighted global. Agreement within
a few percent means the remaining gap is PIL/torch version drift and the zero row
is authoritative. A large gap means the audit missed something and no conclusion
may be drawn from the zero row at all.

Then, on the unweighted global (`s≥0`) and on `lt5`, where the mass is:

- **A — zero beats every published row.** The universality claim holds *in the
  field's own metric*, on the home method's own split, and stops depending on
  our protocol. This is the cheapest possible answer to "your metric made the
  baseline win".
- **B — the published rows beat zero on `lt5` but not globally.** Then the honest
  statement is narrower: doing nothing is unbeaten *globally*, while a raw-pixel
  small-motion bucket does show a win. That bucket mixes latitude with motion
  (1/cos inflation) and mixes the static majority with genuine movers, which is
  exactly the confound the geodesic actives readout was built to remove — so the
  thesis reports both and lets the metric section carry the difference. This
  branch strengthens §4.2 of the article rather than weakening the work.
- **C — the published rows beat zero everywhere.** The universality claim as
  currently written falls. What survives is the geodesic reading on genuine
  movers, and the thesis must print both tables and say plainly which protocol
  each conclusion belongs to.

No branch is a reason not to publish the number, and the branch is decided by the
measurement, not by preference.

**Second, an unrelated correction this audit forces regardless of the branch.**
The five SLOF rows in §5 and §9.1 were run at `--princeton-input-scale byte`,
outside the range their checkpoints were trained in (§16.1 item 3). They must be
replayed at `unit` before any of them is quoted again. `rerun_from_json.py`
carries the original args across, so only the two flags change:

```bash
python rerun_from_json.py /outputs/universality_slof_*_test \
    --set geodesic_metric=haversine --set princeton_input_scale=unit \
    --output-suffix _hav_unit --run
```

Iterations stay at their recorded 64 on purpose: changing the input scale and the
refinement budget in the same step would make the delta unattributable, and 64 is
the more generous setting for them. First indication, 8 pairs, `singlerotation`:
global **−10.96 → −4.01**, poles −20.98 → −14.45, act₀.₂₅ +1.44 → +1.91, act₀.₅
−1.34 → −2.47. Eight pairs decide nothing; the direction is that the scale bug
cost them global and polar accuracy, and the actives barely moved.

### 16.4 RESULT (2026-08-03) — the reproduction is bit-exact, and branch C fired

`singlerotation`, official test, 1289 pairs, `--iters 12 --input-scale unit`,
raw PNG/`.npy` (not our shards), 58 min on a laptop CPU under emulation.

| EPE px @640 | s≥0 | s<5 | s<10 | s<20 | s≥20 |
| --- | --- | --- | --- | --- | --- |
| published (`single_rotation.csv`) | 1.568181 | 0.309022 | 0.387124 | 0.502485 | 62.475649 |
| **reproduced** | 1.568184 | 0.309023 | 0.387125 | 0.502486 | 62.475752 |
| difference | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0002% |

Their AE reproduces to the same precision (0.496856 vs 0.496864 published), which
means both of its defects — the overwritten `ugt` and the per-batch min/max
stretch — were reproduced correctly too, at their batch size of 16. **The gate is
passed at six significant figures.** Every item of the §16.1 audit is confirmed by
construction: 1289 forward pairs, iters 12, `[0, 1]` frames, nearest flow resize,
cumulative buckets, no validity mask. There is no remaining degree of freedom in
which our reading of their protocol could differ from theirs.

So the zero row is authoritative:

| EPE px @640 | s≥0 | s<5 | s<10 | s<20 | s≥20 |
| --- | --- | --- | --- | --- | --- |
| SLOF v1 | 1.568 | 0.309 | 0.387 | 0.502 | 62.476 |
| **zero flow** | **2.338** | **0.609** | **0.771** | **0.973** | **80.345** |
| improvement | **+32.9%** | **+49.3%** | **+49.8%** | **+48.3%** | **+22.2%** |

**Branch C.** In their own metric, on their own split, at their own settings, SLOF
beats the trivial baseline everywhere, and not narrowly. The claim "no published
360° method beats zero-flow" is **false as stated for FLOW360 in the ERP-pixel
metric**, and every place the thesis says it must be rewritten. This was the
pre-registered risk of running this activity at all, and it fired.

**A second finding, from their own table.** The zero row's `s≥0` is **2.3375**.
Their `doublerotation` row publishes **2.3388** — 0.06% away — and its buckets
track the zero row just as closely (0.6108 vs 0.6091 on `s<5`). Our §5 already
called doublerotation "the confidently-predict-zero solution" from the geodesic
side; the ERP metric now says it numerically. **The trivial baseline is already
printed in Table I of the paper, unlabelled, as the authors' worst variant.**
Nobody, including the authors, appears to have noticed that a row of their own
table is doing nothing. That is a cleaner illustration of the missing-baseline
problem than any number we could have produced ourselves, and it survives every
metric objection because it is their table.

**What is now open.** Two things, and they are separable:

1. Why our geodesic rows disagree. §9.1 has singlerotation at −9.46% global on the
   same dataset. Three candidate causes, in order of expected size: the input-scale
   bug (§16.1 item 3, ours); the pair set (2567 with backward flows vs their 1289);
   and the metric itself (area-weighted geodesic degrees on an equal-area grid
   versus pixel-uniform ERP displacement, which prices a polar longitudinal error
   at up to 1/cos φ). The first is settled by the replay in §16.3 and must be run
   before anything else is concluded.
2. Whether the ERP-pixel margin survives an area-correct reading. The run now in
   flight decomposes the same predictions four ways — raw px, cos-weighted px, du
   scaled by cos φ, and both — by GT-magnitude bucket and by |latitude| band. On a
   4-pair probe the corrections *increased* SLOF's margin (+18% → +55% global),
   which is the opposite of what a "their metric flatters them" story predicts, so
   that story should not be told until the full split has been read.

### 16.5 The decomposition — the margin is NOT a metric artefact

Same predictions, four conventions, and a split by |latitude|. `plain` is theirs;
`area` weights each pixel by cos φ (the solid angle it covers); `sph` scales the
longitudinal residual by cos φ before the norm (an ERP pixel of `du` is worth
cos φ of angle); `sph_area` does both. `singlerotation`, 1289 pairs.

| improvement over zero (%) | s≥0 | s<5 | s<20 | s≥20 |
| --- | --- | --- | --- | --- |
| plain (their metric) | +32.91 | +49.26 | +48.34 | +22.24 |
| area | +29.93 | +54.40 | +49.44 | +8.14 |
| sph | +30.28 | +54.16 | +48.88 | +7.71 |
| sph_area | +30.32 | +54.73 | +48.75 | +5.80 |

| improvement over zero (%) | 0–15° | 15–30° | 30–45° | 45–60° | 60–75° | 75–90° |
| --- | --- | --- | --- | --- | --- | --- |
| plain | +30.38 | +30.56 | +31.70 | +26.49 | +28.15 | +36.64 |
| sph_area | +30.40 | +30.69 | +32.24 | +27.84 | +29.17 | +32.51 |

**The corrections move the global margin by 2.6 points and never change its sign,
in any band.** The hypothesis that the ERP-pixel metric manufactures the win —
that it prices polar longitudinal error at 1/cos φ and hands the model a cheap
victory where the sphere is small — is **refuted**. It is refuted twice over:
band by band the margin is flat at ~+30%, including the equatorial bands where no
projection inflation exists at all.

What the corrections *do* change is the size of the problem, not who wins it. The
zero baseline at 75–90° falls from 7.10 px (plain) to 0.89 px (`sph_area`), and
globally from 2.338 to 0.967 — **59% of the raw ERP "motion" in this dataset is
polar longitude inflation rather than displacement on the sphere.** That is worth
reporting on its own: it means the `s≥20` bucket, which their table presents as
the large-motion regime, is 1.72% of the pixels and is mostly a projection
artefact — under `sph_area` its zero baseline is 64.4 px of raw displacement that
corresponds to far less angle, and the margin there collapses from +22.2% to
+5.8%. The *only* place the metric convention materially flatters them is the
large-motion tail.

`raftfinetune` behaves the same way (+30.51 plain, +26.62 sph_area globally), and
reproduces to the same six figures, so this is not a property of one checkpoint.

**The disagreement with §9.1 is therefore ours to explain, not theirs.** Three
candidates remain, and the metric is now the least likely of them:

1. **Input scale** (§16.1 item 3). Settled by the box replay in §16.3. The 8-pair
   preview moved global −10.96 → −4.01, so this is large but, on that evidence,
   not large enough on its own to cross +33%.
2. **Our shards are JPEG.** `sfprep materialize` re-encoded the FLOW360 PNGs as
   JPEG; the flow is stored float16. This project's own P0c/P0d results say that a
   *structured* appearance perturbation is precisely what destroys sub-pixel
   matching, and JPEG ringing is structured and edge-anchored. If our own pipeline
   injected the nuisance the thesis studies, every row of the universality table
   measured a degraded model. Test in flight: the identical protocol run with
   `--source shards` against the raw run above, same checkpoint, same pairs.
3. **Pair set**: 2567 (both directions) versus their 1289 forward-only.

None of these is a defence of the original claim. They determine *how* it has to
be rewritten, not whether.

### 16.6 Blast radius in the progress-report article

Audited against `src/main.tex` at the state of 2026-08-03. Three kinds of claim,
and they are affected very differently.

**Falls, pending the §16.3 replay.** `\subsection{Regime sub-pixel: nenhum método
supera fluxo nulo}` and the four SLOF rows of `tab:universalidade` (−221,0 /
−10,4 / −9,5 / −0,6 on global). Those rows ran at the wrong input scale and are
not a measurement of SLOF. The section *heading* is a universal quantifier over
methods, and one of the methods in it is mis-run.

**False now, independent of any replay.** Section~4.4's bullet says the SLOF rows
use "o protocolo de inferência dos autores". They do not: iters 64 against their
12, and frames in 0–255 against their [0, 1]. This sentence has to change even if
every number survives.

**Unaffected.** The PanoFlow, RAFT-large and OSLO-RAFT rows: PanoFlow(CSFlow) and
TorchVision RAFT both receive frames through their own repositories' conventions,
which our adapters follow, so the scale defect is confined to the five princeton
checkpoints. The regime-contrast figure keeps all three of its curves, and
`tab:flowscape`, the head-to-head, the A1 control, the metric section and the
motion-structure isolation are untouched.

**Strengthened.** The claim that no cited work reports a trivial baseline on its
own test set (Section~2.2) now has a bit-exact reproduction behind it, and a
sharper example than any argument: their `doublerotation` row *is* the trivial
baseline, to 0.06%, printed in their Table I and read as their worst variant.

### 16.7 THE REPLAY (2026-08-03) — the universality claim falls, and the ranking inverts

Five SLOF rows re-run on the box through `rerun_from_json.py`, flow360:test, 2567
pairs, haversine, iters 64, everything identical to §9.1 except
`princeton_input_scale=unit`. 350 s each.

| improvement % | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles |
| --- | --- | --- | --- | --- | --- |
| SLOF raft (scratch) | **+7.08** | **+15.22** | **+12.16** | **+5.10** | **+12.47** |
| SLOF raftfinetune | **+0.63** | **+3.96** | **+3.08** | **+1.12** | **+2.36** |
| SLOF singlerotation | −0.72 | **+3.32** | **+2.39** | −0.73 | −3.32 |
| SLOF switchrotation | −14.42 | −5.39 | −3.57 | −1.81 | −12.84 |
| SLOF doublerotation | −0.46 | −0.10 | +0.01 | +0.24 | −0.50 |
| PanoFlow(CSFlow)+CFE | −11.32 | −5.79 | −4.75 | −4.16 | −13.78 |
| frozen RAFT-large | −15.98 | −7.97 | −7.59 | −7.46 | −23.85 |
| OSLO EMA final | −15.12 | −4.92 | −4.04 | −5.70 | −26.89 |

Against §9.1, the same rows at `byte`: raft −220.97 → **+7.08**, act₀.₂₅ −59.66 →
**+15.22**; raftfinetune −10.38 → +0.63; singlerotation −9.46 → −0.72;
switchrotation −21.66 → −14.42; doublerotation −0.57 → −0.46 (unchanged, as a
zero predictor must be — it has no appearance signal to lose).

**The claim "no published method beats zero-flow" is dead.** Three of five rows
beat zero on both active thresholds, two beat it globally, and the best of them
does so at **+7.08% global / +15.22% act₀.₂₅ / +12.47% at the poles**. The
−220.97% that anchored the strongest sentence in the thesis was our own bug, in
its entirety. This must be corrected everywhere it appears, and the correction is
not cosmetic: it changes what the work is allowed to claim.

**The best row is the plainest one.** `raft.pt` is RAFT-large trained from scratch
on FLOW360's own training split — no rotation equivariance, no spherical
machinery, no distortion handling. It beats every SLOF variant and every other
row in this table, including OSLO.

**And the ranking inverts between the two metrics.** On the same checkpoints, same
dataset:

| | ERP px, their metric (lower better) | geodesic, area-weighted (higher better) |
| --- | --- | --- |
| raft (scratch) | 2.058 — worst | **+7.08 — best** |
| raftfinetune | 1.624 | +0.63 |
| singlerotation | **1.568 — best** | **−0.72 — worst of the three** |

The paper's headline result is that rotation-equivariant training improves on
plain RAFT: 1.568 against 2.058, a 24% reduction. Under an area-weighted angular
metric on the same predictions that ordering **reverses**. This is a finding in
its own right, it does not depend on the zero baseline at all, and it is the
sharpest available argument for why the metric section of the thesis exists. It
also needs its own control before it is published — see §16.8.

**What this costs OSLO.** OSLO-RAFT is now beaten on flow360:test by three
published rows, one of which is a stock perspective architecture trained
in-domain. The sentence "OSLO is not an exception to the limit it measures" is
still true, but the limit is no longer universal, and OSLO is no longer merely
inside the failure set — it is in the worse half of it, on this benchmark.

**What survives, and is stronger than what fell:**

1. **No cited work reports a trivial baseline on its own test set.** Untouched, and
   now backed by a bit-exact reproduction of the table in question.
2. **Their `doublerotation` row is the trivial baseline**, to 0.06% in their metric
   and to 0.01 points in ours. Printed in Table I, read as their worst variant.
   Confirmed independently in both metrics.
3. **The regime contrast, restated quantitatively.** The binary is gone; the
   magnitude is not. The best method on flow360:test gains **+7%** over doing
   nothing; on flowscape:test the same class of method gains **+92.6%**. An
   order of magnitude separates the two regimes, and that statement is more robust
   than the binary it replaces, because no single row can falsify it.

### 16.8 The control the inversion needs

§16.7's ranking inversion compares two numbers that differ in four ways, not one:
the metric (ERP px versus area-weighted geodesic), the pair set (1289 forward
versus 2567 both directions), the refinement budget (12 versus 64), and the frame
path (raw PNG with PIL bicubic versus JPEG shards with bilinear+antialias). Only
the first is the claim. `run_slof_table1.py --directions both` now runs the ERP
metric on the geodesic run's exact pair set, so the other three can be pinned to
the geodesic side and the metric left as the only free variable:

```bash
for CK in raft raftfinetune singlerotation; do
  SHARDS_HOST=../sfprep/shards \
  docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
    python run_slof_table1.py \
      --source shards --shards /data/shards --dataset flow360 --mode test \
      --directions both --checkpoint /outputs/slof_weights/$CK.pt \
      --iters 64 --input-scale unit --batch-size 8 --device cuda \
      --output-dir /outputs/slof_erp_same_pairs_$CK
done
```

Pre-registered reading: if `singlerotation` still beats `raft` on ERP `s≥0` under
these settings while losing to it on geodesic global and actives, the inversion is
the metric's and is publishable as such. If the ordering matches the geodesic one
here, then it was the pair set or the iteration budget, and the ERP-versus-geodesic
inversion claim is withdrawn — leaving §16.4's reproduction and §16.7's correction,
which stand on their own.

A separate local run measures what our own pipeline costs: the same protocol,
`--source shards` (JPEG frames, float16 flow) against the raw-PNG run of §16.4,
same checkpoint. That number bounds how much of every row in this document is our
re-encoding rather than the method.

### 16.9 CONTROL RESULT (2026-08-03) — the inversion claim is WITHDRAWN

The three checkpoints re-scored in the ERP-pixel metric on the geodesic run's
exact conditions: 2567 pairs (both directions), iters 64, JPEG shards, `unit`.

| EPE px `s≥0` | published (1289 fwd, raw, iters 12) | matched (2567, shards, iters 64) | drift |
| --- | --- | --- | --- |
| raft (scratch) | 2.0576 | **2.0818** | +1.18% |
| raftfinetune | 1.6243 | 2.1822 | **+34.35%** |
| singlerotation | 1.5682 | 2.2270 | **+42.01%** |
| zero flow | 2.3375 | 2.2951 | −1.81% |

**Under matched conditions the ERP metric gives the geodesic ordering**: raft
2.082 < raftfinetune 2.182 < singlerotation 2.227, exactly the geodesic +7.08 >
+0.63 > −0.72. The pre-registered branch fires: **§16.7's ranking-inversion claim
is withdrawn.** It was not the metric. Two of the four differences were doing the
work, and they are on our side of the fence, not the metric's.

**What the control found instead is worse, and more useful.** The degradation is
not uniform — it is **inversely proportional to how good the checkpoint was**. The
best-published model loses 42%, the second loses 34%, and the one that was worst
published loses 1%. All three collapse into a narrow band (2.08–2.23) just under
the zero baseline (2.295), where published they spanned 1.57–2.06. That is the
exact signature of a nuisance that destroys fine-grained accuracy: it does not
penalise everyone equally, it deletes whatever margin came from resolving detail,
and it leaves the crudest predictor nearly untouched. This project has measured
that shape before, under its own name — §P0c/P0d, structured appearance
perturbation at sub-pixel displacement.

The pair set is excluded as the cause: it moves the *zero* baseline by −1.81%,
and zero does not read the frames at all. Two candidates remain, and both are
ours:

1. **JPEG.** `sfprep materialize` re-encoded FLOW360's PNGs as JPEG. Every
   geodesic row in this document — SLOF, PanoFlow, RAFT-large **and OSLO's own
   training and evaluation** — has been reading re-compressed frames.
2. **Iteration budget**, 12 against 64. RAFT is not guaranteed to be monotone in
   refinement steps for a checkpoint fine-tuned at a fixed budget.

Two isolations in flight, both local, both against the §16.4 raw run of the same
checkpoint: `--source shards --directions forward --iters 12` (changes JPEG and
nothing else) and a 128-pair paired sweep of iters 12 against 64 on raw frames
(changes the budget and nothing else). Until they land, no causal claim.

**This does not restore the universality claim.** §16.7's replay and §16.4's
reproduction are unaffected — both were run at their own protocols, and the
geodesic replay stands as the corrected version of §9.1 whatever explains the
drift. What is at stake now is whether *our whole evaluation substrate* has been
handicapping every method it measures, including ours.

### 16.10 JPEG IS EXONERATED (2026-08-03) — the drift is the iteration budget

flow360 test+val re-materialized with `--lossless` into a separate `shards_lossless`
(same manifest, `val_every=6`, splits verified 4555/791/2567), then the identical
protocol run on both shard sets: forward-only, iters 12, `unit`, `singlerotation`.

| source | EPE `s≥0` | vs published 1.568181 |
| --- | --- | --- |
| raw PNG/`.npy` files (§16.4) | 1.568184 | +0.0002% |
| **PNG shards** (`--lossless`) | 1.568143 | **−0.0025%** |
| **JPEG shards** (q95, what every row used) | 1.569069 | **+0.0566%** |

**Our JPEG costs 0.059%.** The hypothesis of §16.9 — that `sfprep`'s re-encoding
was deleting the sub-pixel signal and handicapping every method this project has
measured, including OSLO's own training — is **refuted, decisively and at full
split**. The shard pipeline is faithful to the raw files at the fourth decimal.
That is worth having on the record: it closes a question a referee would be right
to ask, and it removes any doubt about the substrate the P1 campaign trained on.

By elimination, §16.9's 34–42% drift is the **iteration budget**, 12 against 64.
The pair set was already excluded (it moves the *zero* row by −1.8%, and zero
never reads a frame), and JPEG is now excluded at 0.06%. Nothing else differs.

**This is our third protocol defect on the SLOF rows, and the largest.** Every
geodesic row in §5, §9.1 and even the corrected replay of §16.7 ran
`--iters 64`, which is `evaluate_raft.py`'s `__main__` default but not the setting
that produced the paper: `train.py:294` calls the evaluator with 12, and the
shipped CSVs are `iters_12`. Running RAFT for 64 refinement steps on a checkpoint
fine-tuned at 12, over displacements whose median is 0.13°, accumulates GRU drift
that a from-scratch checkpoint (`raft.pt`, +1.18%) happens to tolerate and the
fine-tuned ones do not.

So the definitive universality table is **`--princeton-input-scale unit --iters
12`**, their published setting on both axes, and it has not been run yet. §16.7's
numbers are a lower bound on those methods: they were measured under a handicap of
ours. The claim's death is unaffected — removing a handicap cannot make a method
that already beats zero stop beating it — but the magnitudes and the ordering in
that table are not final.

Two runs settle it, both minutes on the box:

```bash
# (a) confirm the attribution: shards, forward, unit, iters 64 (only iters differs
#     from the 1.5691 row above). Expect ~2.2 if the budget is the cause.
SHARDS_HOST=../sfprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_slof_table1.py --source shards --shards /data/shards \
    --dataset flow360 --mode test --directions forward \
    --checkpoint /outputs/slof_weights/singlerotation.pt \
    --iters 64 --input-scale unit --batch-size 8 --device cuda \
    --output-dir /outputs/slof_iterstest_64

# (b) the definitive geodesic table, their published protocol on both axes
SHARDS_HOST=../sfprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python rerun_from_json.py /outputs/universality_slof_*_test \
    --set geodesic_metric=haversine --set princeton_input_scale=unit --set iters=12 \
    --set motion_bands_deg=0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-suffix _hav_unit_i12 --run
```

The lossless shards are not needed for either — they were the control, and the
control passed. They stay on disk as the evidence.

### 16.11 THE DRIFT DECOMPOSED — it is the backward pairs, and they were never validated

`singlerotation`, EPE `s≥0`, one factor changed at a time from the published
setting:

| run | EPE `s≥0` | attributable |
| --- | --- | --- |
| published (raw files, forward, iters 12) | 1.568181 | — |
| JPEG shards, forward, iters 12 | 1.569069 | re-encoding **+0.06%** |
| JPEG shards, forward, **iters 64** | 1.638641 | refinement budget **+4.43%** |
| JPEG shards, **both directions**, iters 64 | 2.227017 | pair set **+35.9%** |

My §16.10 attribution to the iteration budget was wrong by an order of magnitude,
and my §16.9 dismissal of the pair set was wrong for the wrong reason: I argued
that it moves the *zero* row by only −1.8%, which says nothing about a model.
Splitting the `both` run by pixel count (forward 50.21%, backward 49.79%):

| backward pairs only, inferred | EPE `s≥0` |
| --- | --- |
| zero flow | 2.252 |
| singlerotation | 2.820 |

**The model is 25% WORSE than doing nothing on the backward half, while being 30%
better on the forward half.** A backward pair is a time-reversed pair; RAFT has no
difficulty with that, and no plausible model weakness reverses the *sign* of the
benefit. This is the signature of a ground truth whose convention is wrong.

There is structural reason to suspect exactly that. Their `exr2flow` builds
`flowf = (R, -G)` and `flowb = (B, -A)` — different sign patterns per channel.
sfprep pins flow360 to `identity` with `pin_convention = true`, precisely because
the motion is too small to diagnose photometrically, so the *backward* half was
adopted, never measured. And their own evaluator never reads it:
`validate_flow360` requests `fflow` only. **Nobody in this chain — the authors or
us — has ever validated the backward convention.**

If it is wrong it touches 3942 of flow360's 7913 pairs: half of test, half of the
val split where the `+4.5%` act₀.₅ lives, and half of the train split OSLO
consumed.

`run_slof_table1.py` gains `--directions backward` and `--gt-transform
{identity,negated,negate_x,negate_y}` for the diagnostic. The reading is
unambiguous because the zero row is invariant to a sign flip while the model is
not: whichever transform makes a real predictor beat zero on the backward half by
roughly the margin it achieves on the forward half is the correct convention.

Pre-registered: if `identity` stays ~−25% and one transform lands near +30%, the
sfprep pin is wrong and flow360 must be re-materialized with the corrected
backward convention — after which every flow360 number in this project, including
OSLO's training, is measured on half-corrupted data and must be redone. If no
transform helps, the backward half is genuinely harder and the finding is about
the dataset, not our pipeline.

### 16.12 THE BACKWARD CONVENTION IS INVERTED — half of flow360 has been wrong

`singlerotation`, flow360:test **backward pairs only** (1278), iters 12, `unit`,
zero baseline 2.2523:

| `--gt-transform` | EPE `s≥0` | vs zero |
| --- | --- | --- |
| `identity` — what sfprep ships | 2.8992 | **−28.72%** |
| **`negated`** | **2.0732** | **+7.95%** |
| `negate_x` | 2.2923 | −1.77% |
| `negate_y` | 2.7610 | −22.59% |

`negated` is the only transform under which a real predictor beats doing nothing,
and it does so under all four metric variants (+12.8 area, +13.4 sph_area) and in
every latitude band. The reading is clean because the zero row is invariant under
a sign flip while the model is not. **flow360's backward flows are stored negated
with respect to the frame1 → frame2 contract, and sfprep emitted them as
`identity`.**

Fixed in `sfprep/adapters/flow360.py`: the convention is now assigned per
direction, `identity` forward and `negated` backward, verified on a 400-record
discovery (202 forward/identity, 198 backward/negated). `pin_convention` did not
prevent this — it only stops `diagnose` from overriding the default, and diagnose
was never able to see the backward half anyway.

**Why nobody caught it.** Two independent blindfolds. The zero-flow baseline is
invariant under a sign flip, so no statistic computed from the GT alone can
detect it — the magnitudes, the active fractions, the band histogram are all
identical either way. And the dataset's own published evaluator never reads
`bflow`: `validate_flow360` requests `fflow` only, so the authors never exercised
it either. It takes a *model* to see it, and the model has to be run on the
backward half in isolation, which nothing in this project did until today.

**Blast radius.** 3942 of flow360's 7913 pairs: 1278 of test's 2567, roughly half
of val's 791, and roughly half of train's 4555.

1. **Every flow360 evaluation in this document** — §5, §9.1, §16.7 — averaged a
   correct forward half with a sign-inverted backward half. The corrected
   singlerotation row on the full test set at iters 12 is roughly **+20.7%**
   (0.502·1.5691 + 0.498·2.0732 = 1.820 against zero 2.295), against the −0.72%
   §16.7 reported.
2. **OSLO was trained on it.** `flow360:train` is in the P1-proper mix, so for
   half of those pairs the supervision pointed the wrong way. The campaign was
   fitting a sub-pixel matcher against a target that contradicted itself
   pair-to-pair.
3. **The `+4.5%` act₀.₅ headline** was measured on flow360:val, half of which
   carries inverted targets.

A residual remains and is not explained: even negated, the backward half reaches
only +7.95% where the forward half reaches +32.9%. A sign flip is unambiguously
wrong and unambiguously fixed, but the gap says something else is also off —
most likely which frame's grid `bflows/N.npy` is defined on, i.e. an off-by-one
in the pairing. That is the next thing to isolate, and it does not block the
re-materialization, since no ordering of frames makes an inverted sign correct.

### 16.13 Fix verified

`shards_fixed` (flow360 re-materialized with the per-direction convention),
backward pairs, `--gt-transform identity`: EPE `s≥0` = **2.0732028186847393**,
digit-for-digit the value the old shards produced under `--gt-transform negated`.
The materialization applied exactly the diagnosed transform and nothing else. The
corrected shards are the ones to read from here on.

### 16.14 THE CORRECTED TABLE (2026-08-03) — the universality claim is dead by a wide margin

flow360:test, 2567 pairs, `shards_fixed`, haversine, `--princeton-input-scale unit
--iters 12` (their published protocol on both axes). Zero baseline global
0.43670° (0.43145 on the corrupted shards — the 1.2% shift is the ERP endpoint
construction's nonlinearity under a sign flip, not a metric change).

| improvement % | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles | equator |
| --- | --- | --- | --- | --- | --- | --- |
| SLOF switchrotation | **+30.28** | **+35.03** | **+31.36** | **+19.92** | **+22.88** | +33.14 |
| SLOF singlerotation | **+26.88** | **+30.85** | **+27.16** | **+17.38** | **+21.13** | +29.56 |
| SLOF raftfinetune | **+25.40** | **+27.83** | **+24.76** | **+16.84** | **+24.95** | +25.43 |
| SLOF doublerotation | −0.21 | −0.12 | −0.14 | −0.16 | −0.11 | −0.25 |
| SLOF raft (scratch) | −11.96 | −2.41 | −1.21 | +0.03 | −9.97 | −13.52 |

Against §9.1, which is what the article prints: switchrotation **−21.66 → +30.28**,
singlerotation **−9.46 → +26.88**, raftfinetune **−10.38 → +25.40**. Three
published checkpoints beat the trivial baseline by a quarter to a third of its
error, on every region and every active threshold. **"No published method beats
zero-flow" is not merely false — it is off by fifty points on three rows.**

**Two of my own conclusions from this session are refuted by this table**, and
both were driven by the corrupted backward half:

- §16.7 concluded "the best row is the plainest one", `raft.pt` at +7.08% global.
  On correct targets `raft.pt` is the *worst* trained row at **−11.96%**. It
  looked good only because it was the least accurate predictor and therefore the
  least damaged by a sign-inverted target; the accurate models were the ones the
  corruption punished.
- §16.9's ordering under "matched conditions" carried the same defect and says
  nothing.

**What survives every correction.** `doublerotation` remains numerically
indistinguishable from doing nothing: −0.21% global here, −0.57% on the corrupted
shards, +0.03% under `acos`, and 2.3388 against zero's 2.3375 in their own ERP
metric. Four independent measurements, three different targets, two metrics. It is
the trivial baseline, printed unlabelled in their Table I as their worst variant,
and that finding is now the most robust thing in this document.

**The static-calibration result also survives, and is universal.** Every winning
row is catastrophic on the static majority — `band_0_0625` improvement −275%
(singlerotation), −340% (switchrotation), −173% (raftfinetune) over 32.8% of the
sphere — and wins anyway. The decision-gate proposal in `docs/plans/DECISION_GATE.md`
now has a *better* motivation than the one it was parked with: it would compound
with methods that already beat zero, instead of trying to rescue ones that do not.

**Pending, and now the only number that matters for the thesis**: OSLO's final
model re-read on `shards_fixed`. Until it lands nothing can be said about where
OSLO sits, except that the bar it has to clear is no longer zero — it is +25% to
+30%.

### 16.15 RETRACTION — the convention is inverted on the FORWARD half, and §16.12/§16.14 are void

The corrected-table replay returned frozen RAFT-large at **−66.60% global** on
`shards_fixed`, against −15.98% on the original shards. A model cannot get four
times worse from a fix. Two RAFT-family predictors were disagreeing about the
sign of the same targets, which is only possible if one of them predicts reversed
motion — so I stopped inferring and measured the 2×2 directly, with the one
arbiter that is independent of FLOW360: TorchVision RAFT-large, trained on
Sintel/KITTI/FlyingThings, unambiguous convention. flow360:test, geodesic global
improvement over zero:

| | `identity` | `negated` |
| --- | --- | --- |
| **forward** | −82.07 | **+39.46** |
| **backward** | **+49.31** | −75.50 |

Margins of 120 points. **The physical convention is `negated` forward and
`identity` backward** — the exact inverse of what §16.12 concluded and what
`shards_fixed` was built with. The original shards had the forward half wrong and
the backward half right; `shards_fixed` has both wrong, which is precisely why
RAFT-large collapsed on it.

**Why §16.12 got it backwards.** It used SLOF `singlerotation` as the arbiter.
SLOF's checkpoints were fine-tuned against this same inverted target, so they
predict physically reversed motion, and every convention question answered with
them comes back inverted. I used a reversed ruler to measure the ruler. The
bit-exact reproduction of §16.4 is not a defence: **a model trained against a
reversed target reproduces its own published table exactly.** That reproduction
validates our pipeline against theirs and says nothing about physical sign — a
hole I named two sections ago and then failed to close before drawing
conclusions from SLOF.

**What is now void.**

- §16.12's fix and its blast-radius reasoning: right that the two halves differ,
  wrong about which one and in which direction.
- §16.13's verification: it confirmed the materializer applied the transform I
  asked for, which it did. The transform was wrong.
- **§16.14's corrected table in its entirety.** Those +30% SLOF rows were measured
  on `shards_fixed`, i.e. against targets wrong in *both* halves. They mean
  nothing. The universality question is, as of now, **unmeasured** — not answered
  either way.
- §16.7 and §16.9 were already void for a different reason.

**What survives.** §16.4's reproduction, as a pipeline check. The `doublerotation`
≈ zero identification, since a zero predictor is invariant to every sign question
raised here. And the observation that the zero baseline's sign-invariance is
exactly what let this defect survive undetected in a published dataset, in its
authors' code, and in ours.

**Fixed in the adapter** (`negated` forward, `identity` backward, verified 202/198
on a 400-record discovery), but **do not re-materialize yet**: the 2×2 above is
six pairs at resolution 5. It must be repeated on the full split before any data
is rebuilt on it. Commands in §16.16.

### 16.16 Confirmation before rebuilding

Nothing gets re-materialized until the 2×2 is repeated at full scale, with a
second arbiter that is not a model at all.

**(a) Full split, frozen RAFT-large, four runs.** Same tool, same protocol as the
existing `universality_raftlarge_flow360_test_hav` row, on the ORIGINAL shards
(so both halves are raw and the transform is the only variable):

```bash
for D in forward backward; do for T in identity negated; do
  SHARDS_HOST=../sfprep/shards \
  docker compose -f docker-compose.oslo_raft.yml run --rm -e TORCH_HOME=/outputs/torch_home \
    oslo-raft python run_raft_shard_baseline.py \
      --shards /data/shards --sources flow360:test --resolution 6 \
      --predictor raft --directions $D --gt-transform $T \
      --geodesic-metric haversine --device cuda \
      --output-dir /outputs/conv_${D}_${T}
done; done
```

Expected if §16.15 holds: forward strongly positive under `negated` only,
backward strongly positive under `identity` only, both by tens of points.

**(b) A model-free arbiter.** The claim now implicates a published dataset, so it
should not rest on any network. Warp frame1 by each candidate flow and compare
photometrically against frame2: the correct convention minimises the residual,
and no training is involved. `sfprep diagnose` already implements this and was
switched off for flow360 by `pin_convention = true` on the grounds that the
motion is too small to see photometrically — a judgement that now looks like the
third blindfold in this chain, alongside the sign-invariant baseline and the
authors' evaluator never reading `bflow`. It needs a per-direction run and, if
the global motion really is too small, restriction to the high-magnitude tail.

Only with (a) and (b) agreeing does flow360 get rebuilt, and only then does any
number in this document — including OSLO's — mean anything again.

### 16.17 CONFIRMED AT FULL SCALE — and the wall was a sign error

flow360:test, resolution 6, haversine, frozen TorchVision RAFT-large, original
shards, 1289 forward + 1278 backward pairs:

| | `identity` | `negated` |
| --- | --- | --- |
| forward, global | −65.81 | **+40.73** |
| forward, act₀.₅ | −53.15 | **+45.17** |
| backward, global | **+37.82** | −67.44 |
| backward, act₀.₅ | **+42.75** | −54.77 |

~105-point margins, matching the 6-pair probe. **`negated` forward, `identity`
backward, settled.**

**The consequence is the whole thesis.** Combining the two halves under their
correct conventions, frozen RAFT-large scores global error 0.2601° against a zero
baseline of 0.4286° — **+39% global, +45% on the active nodes**, zero-shot, a
stock perspective architecture with no fine-tuning and no spherical machinery.

The row the article prints for that same network is **−15.98%**. It is the average
of a forward half whose targets were inverted (−65.8) and a backward half whose
targets were fine (+37.8). Every "nobody beats zero on sub-pixel motion" statement
in this project is that average.

**So the sub-pixel wall, as measured on flow360, does not exist.** It was our own
preparation pipeline. `pin_convention = true` fixed `identity` for the whole
dataset on the reasoning that the motion was too small to diagnose
photometrically — and that single unvalidated assumption produced: a wall, a
universality claim, a regime-contrast argument, a training set half of whose
supervision pointed backwards, and roughly a year of campaign built on top.

**What must now be re-measured, in order:**

1. Rebuild flow360 (adapter fixed, `negated`/`identity`), all three splits.
2. The universality table, every row. The question is now open in the *opposite*
   direction: not whether anyone beats zero, but by how much, and where OSLO sits
   among methods that clear +39%.
3. **OSLO's P1 campaign.** `flow360:train` is in the P1-proper mix, so half its
   flow360 supervision was reversed. Gate R2, the `+4.5%` act₀.₅, the EMA basin,
   the static-calibration diagnosis — all were fitted and read against that.
4. The same 2×2 on **every other dataset**. replica360, flowscape, chairs360 and
   mpf were never subjected to this test either. chairs360 and flowscape have
   warp-validated conventions on record; replica360 and mpf do not.

**What survives.** `doublerotation` ≈ zero, which no sign question can touch. The
§16.4 pipeline reproduction. And one methodological result that is now worth more
than the claim it destroyed: **a trivial-baseline comparison cannot detect an
inverted target, because the baseline is sign-invariant — and neither can
reproducing a published table, because a model trained against a reversed target
reproduces its own numbers exactly.** Three independent parties missed this on
the same dataset: its authors, whose evaluator never reads `bflow`; SLOF's
fine-tuning, which learned the reversal; and this project, which pinned the
convention and then measured it with SLOF.

### 16.18 Convention audit across every dataset — flow360 is the only casualty

Same 2×2, frozen RAFT-large, geodesic global improvement over zero. Six pairs at
resolution 5 for the local sets (margins are 60–180 points, so the sample size is
not the question); flow360's row is the full-split result of §16.17.

| dataset | directions | `identity` | `negated` | verdict |
| --- | --- | --- | --- | --- |
| flow360 forward | 3971 | −65.81 | **+40.73** | **BROKEN** |
| flow360 backward | 3942 | **+37.82** | −67.44 | ok as shipped |
| replica360 forward | 954 | **+88.81** | −89.67 | clean |
| replica360 backward | 954 | **+87.59** | −87.17 | clean |
| mpf forward | 4188 | **−0.33** | −66.09 | clean (66-pt margin) |
| flowscape forward | 6336 | **+73.09** | −91.64 | clean |
| chairs360 forward | 23512 | **+55.64** | −75.49 | clean |

**replica360 is clean in both directions**, which is the single most valuable
result here: Stage A, Gate R1, the OSLO-vs-RAFT head-to-head, A1 round 1 and the
retention stamp all rest on it and all survive. It carries the same two-direction
structure as flow360 and the same per-dataset (not per-direction) `diagnose`, so
it was exposed to the identical failure mode and simply did not have it.

mpf's 66-point asymmetry settles its sign. That RAFT-large only ties zero there
(−0.33 global) is a separate question and mpf carries no article result.

flowscape and chairs360 are forward-only, so no per-direction divergence is
possible, and both have warp-validated conventions on record — but a record is
not a measurement, so both were measured. Their shards did not exist locally;
their raws did, so a 64-pair slice of each was materialized on the spot
(`sfprep materialize --limit 64`) and scored. Margins of 131 and 165 points,
both favouring `identity`. The command below is what the box would have run:

```bash
for S in flowscape:test chairs360:val; do for T in identity negated; do
  SHARDS_HOST=../sfprep/shards \
  docker compose -f docker-compose.oslo_raft.yml run --rm -e TORCH_HOME=/outputs/torch_home \
    oslo-raft python run_raft_shard_baseline.py \
      --shards /data/shards --sources $S --resolution 5 \
      --predictor raft --directions forward --gt-transform $T \
      --geodesic-metric haversine --max-pairs 64 --device cuda \
      --output-dir /outputs/conv_${S%%:*}_${T}
done; done
```

**Scope, if those two come back clean:** the defect is confined to flow360, one
direction of it, and the rebuild is one dataset. Everything measured on
replica360 or flowscape stands as printed.

**Audit closed.** flow360 forward is the only broken convention in the project.
Every other dataset, in every direction it carries, prefers the shipped
`identity` by 131–178 points. The rebuild is one dataset.

### 16.19 The corrected table, first rows (2026-08-03) — the regime survives, the wall does not

flow360:test, `shards_v2`, 2567 pairs, haversine. Zero global 0.42865°.

| row | global | act₀.₂₅ | act₀.₅ | act₁.₀ | poles | seam |
| --- | --- | --- | --- | --- | --- | --- |
| **frozen RAFT-large** (zero-shot, native) | **+39.32** | **+46.27** | **+44.01** | **+37.82** | **+25.10** | +23.18 |
| SLOF switchrotation (unit, iters 12) | −56.55 | −47.28 | −40.32 | −26.90 | −55.26 | −28.48 |

The RAFT-large figure lands on the +39.3% predicted in §16.17 from combining the
two halves, which is the internal consistency check this whole chain needed.

**The sub-pixel wall does not exist on FLOW360.** A stock perspective
architecture, frozen, zero-shot, no spherical machinery, beats the trivial
baseline by 39% globally and 44% on the active nodes. The −15.98% this project
published for that same network was the average of an inverted forward half and a
correct backward half.

**SLOF's released checkpoints predict reversed motion.** Against physically
correct targets they are far *worse* than doing nothing: −56.55% global. A model
predicting exactly −g would read −100%; −56.55% is an attenuated reversal. Two
readings, and they converge: either FLOW360's `fflows` are mislabelled and SLOF
trained consistently with the dataset's own label, or the convention was simply
never checked by anyone. **Either way, whoever downloads the published weights
gets reversed flow**, and no published evaluation could have caught it — the
trivial baseline is sign-invariant, and reproducing the authors' table does not
test convention, because a model trained against a reversed target reproduces its
own numbers exactly (§16.4 does, to six figures).

**What replaces the regime argument.** The same frozen RAFT-large scores +74.4%
on flowscape:test (large motion, unaffected by any of this) against +39.3% on
flow360:test (sub-pixel). A 35-point gap is a real regime effect and it is
defensible; the sign flip that the thesis built its story on was an artefact of
our own data. The honest claim shrinks from "sub-pixel motion defeats every
method" to "sub-pixel motion halves what the same method achieves at large
motion", which is smaller, true, and measured.

Remaining: the other four SLOF rows, PanoFlow, and — the one that decides where
OSLO stands — the retrained model.

### 16.20 Shard set swapped and verified (2026-08-03)

`shards_v2` (corrected flow360) received the four unchanged datasets by hard link
from the old set, indexes merged, and was promoted to the canonical `shards`
path. The previous set is kept as `shards_v1_broken` — it is the evidence that
the defect existed, and every before/after number in §16 depends on being able to
show it.

Verification at the canonical path, frozen RAFT-large, 64 pairs, resolution 5:

| source | expected | measured |
| --- | --- | --- |
| flow360:test forward, `identity` | ≈+40% (was −65.81) | **+43.69** |
| flowscape:test, `identity` | ≈+73% | **+73.25** |

Both pass. Note for bookkeeping: after the swap, `args.shards` records
`/data/shards` for old and new runs alike, so **the output directory name is the
only marker of which vintage a run used**. Keep the `_v2` suffix on everything
produced from here on.

### 16.21 THE RETRAIN (2026-08-04) — Gate R2 passes by an order of magnitude

`P1proper_mix20k_v2`, 20k steps, 22.7 h, identical recipe to the original run
(same seed 7, same warm start from Stage A, same `--real-resample-prob 0.3`,
same edge-corruption 3.1, same OneCycle) — **only the flow360 shards changed**.
Validation `flow360:val`, acos, area-weighted node means, no rotation on val.

| metric | `P1proper_mix20k` (v1, broken) | `P1proper_mix20k_v2` | zero (both) |
| --- | --- | --- | --- |
| act₀.₂₅ | — | **+58.83** | 0.6152° |
| act₀.₅ | +4.0 | **+60.38** | 0.8559° |
| act₁.₀ | — | **+35.31** | 1.8140° |
| global | −31.1 | **+10.52** | 0.2106° |
| equator | — | +21.36 | 0.2170° |
| seam | — | +7.30 | 0.3077° |
| poles | — | **−43.06** | 0.1633° |

**The two runs are on the same ruler to within 0.04%, measured.**
`active_0_5_zero_geo_deg` reads 0.8556 before and 0.8559 after, and the active
set is selected on |g| in both. The near-invariance is expected — flipping the
sign of the flow cannot change how far the trivial baseline is from the truth —
but it is *near*, not exact, and §16.25 measures why. What changed is the
numerator: the model's own error on the active set fell from 0.8214° to
**0.3391°**, a factor of 2.42. A 0.04% shift in the denominator does not produce
a 2.42× change in the ratio; this is a real reduction in a directly comparable
quantity, not a rescoring artefact.

**Gate R2 (+5.2% act₀.₅) passes at 11.6×.** The gate was chased for the entire P1
campaign — approached three times, never consolidated, and closed at 86% with the
diagnosis "blocker is the level, paths forward are data scale and static
calibration" (memory `oslo-raft-p1*`). The blocker was neither. It was that half
of the flow360 training pairs carried the negated target, so the network was
being asked to predict +g and −g for the same appearance change, and the only
loss-minimising answer to contradictory supervision is to predict nothing. The
+4.5 ± 0.9 that took a 9× variance reduction to stabilise was the residue of that
compromise.

**Global goes positive for the first time on real flow360 pairs**: −31.1 → +10.52.
The static-calibration problem does not vanish — the model still spends error on
the static majority — but it is no longer large enough to swamp the leg.

**Poles are the one regression, and they are now the open problem.** −43.06%,
against a zero baseline of 0.1633° — the *lowest* zero error of any region, i.e.
the poles of this dataset are where the field is most nearly static. The model
commits motion there and pays for it. This is the static-calibration item in its
purest form and it is now isolated: every other region is positive.

**What this does not yet establish.** This is `flow360:val`, sequences held out
from our own split of the official *train* half. The frozen RAFT-large reference
(+39.32 global / +44.01 act₀.₅, §16.19) is on `flow360:test`, and the two are not
comparable. The number that goes in the table is OSLO on test, under haversine,
and it is a minutes-long eval. Until it exists, the claim is "the wall was
contradictory supervision", not "OSLO beats RAFT-large".

### 16.22 OSLO on corrected flow360:test — positive, genuine, and second

`P1final_test_flow360_v2`, the 20k raw checkpoint, `flow360:test`, 2567 pairs,
haversine, area-weighted, 8 min. Zero global 0.42865°, active fracs
34.6/18.0/5.8 — bit-identical to every other row in this table, so the
comparison is byte-level.

| row | global | act₀.₂₅ | act₀.₅ | act₁.₀ | equator | poles | seam |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen RAFT-large (zero-shot) | **+39.32** | **+46.27** | **+44.01** | **+37.82** | — | **+25.10** | **+23.18** |
| **OSLO 20k `_v2`** | +19.17 | +37.87 | +37.66 | +31.48 | +26.89 | **+0.10** | +16.48 |
| OSLO EMA, v1 broken shards | −14.0 | −4.9 | −4.0 | — | — | −24.8 | — |
| SLOF switchrotation | −56.55 | −47.28 | −40.32 | −26.90 | — | −55.26 | −28.48 |

**The correlation is entirely load-bearing.** The pre-registered control on the
same checkpoint and the same split: `--ablate-corr` takes global from 0.3465° to
**69.02°** — a 199× degradation, and the strongest collapse ever recorded in this
project (the previous best was 54° on the v1 model, §P1). +37.66% on the active
nodes is genuine correspondence, not an appearance prior.

**Same-pool → cross-pool, second measurement.** act₀.₅ goes +60.38 (val) →
+37.66 (test). The v1 model made the same crossing as +4.5 → −4.0. The drop is
real and has the same causes as before (all decisions were made on val; test is a
disjoint, 2× harder pool), but it no longer changes the sign — it costs 23 points
out of 60 instead of all of them.

**OSLO is positive and OSLO is second.** A frozen, zero-shot, perspective
RAFT-large beats it on every axis of this table. The honest statement is that
OSLO clears the trivial baseline by a wide margin on real sub-pixel video, which
nothing in this project had ever done, and that it does not beat a stock
perspective architecture on this dataset.

**⚠ THE POLAR ADVANTAGE DOES NOT REPLICATE HERE, and this is the third dataset.**
OSLO reads +0.10% at the poles — zero-parity, it does nothing there — against
RAFT-large's +25.10%. On replica360 and flowscape the split runs the other way
(OSLO wins poles, 2.3× and 2.4× flatter). Those are both large-motion; flow360 is
the sub-pixel regime. So the surviving claim has to carry the regime with it:
**native spherical geometry buys polar accuracy and sphere uniformity relative to
a perspective architecture in the large-motion regime**, replicated twice, and it
does not carry over to the sub-pixel regime, where the same comparison inverts.
Note also that the poles are not unusually static on this split (zero poles
0.4454° vs global 0.4286°), so "there was nothing to find" does not explain it.

### 16.23 The inventory (2026-08-04) — `shards_fixed` was accidentally the perfect control, and the table is already complete

Auditing every `/outputs/universality_*` JSON gave a vintage discriminator that
needs no bookkeeping: the zero-flow global. `0.4368` = v1 shards, acos. `0.4314`
= v1 shards, haversine. `0.4367` = `shards_fixed`, haversine. **`0.4286` =
`shards_v2`, haversine.** Every corrected row carries 0.4286 and is therefore
mutually comparable byte-for-byte.

**`shards_fixed` — the set that §16.15 retracted for being wrong in *both* halves
— is the exact inverse of the correct convention, which makes it the inverted arm
of a 2×2 nobody had to pay for.** flow360:test, global / act₀.₅:

| model | `shards_fixed` (inverted) | `shards_v2` (correct) | swing |
| --- | --- | --- | --- |
| frozen RAFT-large | −66.60 / −53.94 | **+39.32 / +44.01** | 106 / 98 |
| PanoFlow(CSFlow)+CFE | −49.11 / −39.86 | **+29.81 / +34.48** | 79 / 74 |
| SLOF raftfinetune | **+25.40 / +24.76** | −37.64 / −29.29 | −63 / −54 |
| SLOF singlerotation | **+26.88 / +27.16** | −48.68 / −37.06 | −76 / −64 |
| SLOF switchrotation | **+30.28 / +31.36** | −56.55 / −40.32 | −87 / −72 |
| SLOF doublerotation | −0.21 / −0.14 | −0.01 / +0.13 | ~0 / ~0 |
| SLOF raft (from scratch) | −11.96 / −1.21 | −13.26 / −2.22 | ~0 / ~0 |

The structure is exactly what the diagnosis predicts and could not have been
faked. **The two externally-trained models flip strongly negative → strongly
positive. The three SLOF checkpoints that actually learned something flip the
opposite way, positive → negative.** The two SLOF rows that flip nothing are the
two that predict nothing: `doublerotation` is the trivial zero-predictor (it sits
at ±0.2% under either convention, as a zero-predictor must, since the baseline is
sign-invariant) and `raft` is the from-scratch run that never converged (−465% on
val, §round 1).

That is six checkpoints from three labs, two disjoint training conventions, and
one sign axis — separated by 60 to 106 points in the direction predicted, with the
two null models correctly reading null. **SLOF's released weights predict motion
in FLOW360's own inverted forward convention.** No further argument is needed;
this replaces the single-row evidence of §16.19.

**The corrected universality table is complete and cost no additional GPU time**
— every row already existed under a `_v2` name. flow360:test, `shards_v2`, 2567
pairs, zero global 0.42865°, haversine, area-weighted:

| row | global | act₀.₅ |
| --- | --- | --- |
| frozen RAFT-large (zero-shot, perspective) | **+39.32** | **+44.01** |
| PanoFlow(CSFlow)+CFE (zero-shot, 360°-native) | +29.81 | +34.48 |
| **OSLO 20k `_v2`** (in-domain) | +19.17 | +37.66 |
| SLOF doublerotation (trivial zero-predictor) | −0.01 | +0.13 |
| SLOF raft (from scratch) | −13.26 | −2.22 |
| SLOF raftfinetune | −37.64 | −29.29 |
| SLOF singlerotation | −48.68 | −37.06 |
| SLOF switchrotation | −56.55 | −40.32 |

**Three independent architectures beat the trivial baseline comfortably.** The
universality claim is dead, and it is dead by a margin that leaves no room for
rescue. What remains true, and is what the thesis now argues: the same frozen
RAFT-large scores +74.4% at large motion and +39.3% here, so the sub-pixel regime
*halves* a method's margin. The five negative rows are all SLOF's, and they are
negative because their targets were inverted, not because the regime defeated
them.

**One finding worth keeping**: OSLO's active-node score (+37.66) sits **above
PanoFlow's** (+34.48) while its global sits well below (+19.17 vs +29.81). That
is the actives-versus-calibration trade-off this project has documented since P1,
now visible across labs rather than inside SLOF's variants: OSLO commits on the
movers and pays on the static majority. It is the strongest surviving argument
for the decision-gate proposal (`docs/plans/DECISION_GATE.md`).

**Open bookkeeping item.** The zero-flow global differs between vintages at fixed
metric: 0.4314 (v1, haversine) vs 0.4286 (v2, haversine), 0.65%. A sign flip
cannot do this — the baseline is sign-invariant — so the GT itself changed
slightly in the re-materialisation, beyond the frames going lossless. It does not
touch any conclusion here (all corrected rows share 0.4286 exactly), but any
sentence of the form "was −14.81, now +39.32" crosses both a metric change and a
GT change and must not be written as a single-cause statement. Worth pinning
before the article: diff `target_geo_deg_p50/p90` and the pair count between a v1
and a v2 run.

### 16.24 The root cause, in our own config — the check existed and was overridden

Tracing sfprep's git history closes the origin question. `sfprep/adapters/flow360.py`
carried `flow_convention=self.default_convention` for both directions from the
first commit (397c465), and `datasets.toml` supplies that default:

```toml
default_convention = "identity"   # validated convention used by the model repo
pin_convention = true             # lock it: FLOW360 motion is too small to diagnose photometrically
```

So the three shard vintages are exactly: **v1** = `identity`/`identity` (forward
wrong, backward right), **`shards_fixed`** = commit 6adf546, `identity` forward /
`negated` backward (both wrong — the inverted control of §16.23), **`shards_v2`**
= commit f4d37a6, `negated` forward / `identity` backward (both right).

**sfprep already contains an automated convention diagnoser, it abstained, and
the abstention was overridden by hand.** `sfprep/diagnose.py` warps frame 2 under
every candidate convention, measures photometric error against zero-flow, and
adopts the winner *only when the margin clears `--min-improvement`*; otherwise it
keeps the config default. Its own docstring names the failure mode: "Small-motion
datasets (e.g. FLOW360) are often inconclusive and should rely on their validated
default." The tool was right to say it could not tell. What went wrong is the
word **validated** in that config comment, next to a value that was never
validated — it was inherited from the dataclass default (`config.py:17`) — and
then locked with `pin_convention = true`.

This is the fourth and best blindfold, and unlike the other three it is ours. The
other three are properties of the problem (a sign-invariant baseline; a published
table that reproduces under either convention; SLOF never reading `bflow`). This
one is a process failure with a name: **an abstention was converted into a
finding by annotation.**

Two structural defects would have kept the diagnoser from catching this even
unpinned, and both must be fixed in sfprep before the next dataset is onboarded:

1. **No motion or gradient gating.** It averages the photometric error over every
   pixel. On flow360 the median displacement is ~0.23 ERP px and the real
   inter-frame appearance change is 3.1/255 (§P0), so the discriminating signal
   is a rounding error on the aggregate. The gate is not optional here; it is the
   only thing that makes the measurement possible.
2. **It diagnoses per dataset, not per direction.** flow360's two directions have
   *opposite* conventions. A single verdict for the dataset cannot be right, at
   any SNR. This alone invalidates the tool for any dataset shipping both
   directions.

`run_constancy_arbiter.py` (this repo) implements the gated, per-direction,
α-sweep version as the shard-side certification: it sweeps
`R(α) = mean |frame2[x + α·g(x)] − frame1[x]|` and reports where the minimum
sits, per direction, with a per-pair sign vote. Its `--self-test` synthesises a
pair with a known convention and asserts the arbiter recovers `+1`, then feeds it
`−g` and asserts `−1` — the guard this project earned in §16.15 by reasoning
about a sign instead of measuring it.

### 16.25 The zero-baseline discrepancy resolved, and the model-free arbiter delivers

**The open item of §16.23 is closed: the pool did not change, the parameterisation
did.** All three vintages, frozen RAFT-large, haversine, flow360:test:

| statistic | v1 (1 half inverted) | `shards_fixed` (2 halves) | `shards_v2` (0 halves) | spread |
| --- | --- | --- | --- | --- |
| `target_geo_deg_p50` | 0.132098 | 0.132132 | 0.132061 | 0.05% |
| `target_geo_deg_p90` | 0.756320 | 0.756850 | 0.755820 | 0.14% |
| `quantile_samples` | 126154189 | 126173100 | 126140285 | 0.03% |
| `active_0_5_frac` | 0.179705 | 0.179802 | 0.179582 | 0.12% |
| `equator_zero` | 0.409485 | 0.410274 | 0.408936 | 0.33% |
| `global_zero` | 0.431449 | 0.436698 | 0.428648 | 1.88% |
| `poles_zero` | 0.455637 | 0.478522 | 0.445366 | **7.45%** |

The pair pool, the validity mask and the GT *distribution* are the same to three
or four significant figures. Only the *means* move, they move most at the poles
and least at the equator, and the median barely moves at all — so this is a tail
effect concentrated at high latitude, carried by a small number of nodes.

**Mechanism.** |g| would be exactly sign-invariant if the target lived in the
tangent plane, but it does not: the target is the geodesic displacement between
the ERP source pixel and the ERP destination pixel `x + g`, both mapped to the
sphere, and that map is nonlinear. The arc from x to `x + g` is not the same
length as the arc from x to `x − g`, because the longitudinal scale `cos φ`
differs at the two destinations. The asymmetry is negligible where `cos φ` is
flat (equator, 0.33%) and large where it is steep (poles, 7.45%).

**This ordering is itself a check, and it passes.** At the poles: `shards_v2`
0.4454 < v1 0.4556 < `shards_fixed` 0.4785, i.e. **monotone in the number of
inverted halves** — 0, then 1, then 2. Globally the same order holds. The vintage
assignment of §16.23 was derived from model behaviour; this reproduces it from
the ground truth alone, with no model involved.

Two consequences. First, §16.21's "same ruler" claim is amended to what was
measured: 0.04% on the active-set denominator, which cannot manufacture a 2.42×
change. Second, a caveat worth carrying into the article: **any geodesic metric
derived from an ERP pixel-space flow inherits a parameterisation asymmetry at
high latitude.** It is small (0.3% at the equator) but it is not zero at the
poles, and polar numbers at the few-pixel scale should be read with that in mind.

**THE MODEL-FREE ARBITER (2026-08-04) — the defect is confirmed by photometry
alone.** `run_constancy_arbiter.py`, validated in Docker: the self-test recovers
`+1` from a synthesised pair with a known convention and `−1` when fed `−g`,
with R = 0.00 at the minimum. Then, run on the **v1 shards** (the broken vintage,
still present on the laptop, JPEG frames, 14 Jun) — 24 pairs per direction,
`--min-motion-px 2.0 --edge-quantile 0.9`, 1.2 s:

| direction | R(−1) | R(0) | R(+1) | grid argmin | parabolic argmin | pairs for +1 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | **18.52** | 28.10 | 36.91 | **−1.0** | −0.936 | 1 / 23 | negated |
| backward | 30.03 | 23.27 | **14.67** | **+1.0** | +0.975 | 21 / 23 | identity |

Every pre-registered prediction fired. The two directions of one dataset return
**opposite** conventions; the sub-grid minima land within 4% and 2.5% of exactly
±1, which is itself a consistency check (a mis-scaled or mis-registered flow
would not minimise at unit alpha); the correct sign explains 34% and 37% of the
photometric residual while the wrong sign is *worse than not warping at all*
(36.91 vs 28.10 unwarped, forward).

**The claim no longer depends on any neural network.** The evidence is now three
independent layers: photometric constancy with no model (this section), the
2×2 across six checkpoints from three labs (§16.23), and the corrected retrain
(§16.21-16.22). The gate that made this measurable is the one `sfprep/diagnose.py`
lacks — restricting to pixels that actually move. Ungated, on this dataset, the
signal is a rounding error; gated, it is a 34-point margin with a 22-to-1 vote.

### 16.26 Full-split arbiter, positive control, and the root-cause fix in sfprep

**Positive control first.** flowscape:test, 300 pairs, large motion, convention
known correct: argmin **+1.0**, parabolic 0.9918, **300 of 300 pairs** favour +1,
z = 17.26, the correct sign explains 43.0% of the residual. The instrument works
when the signal is there.

**flow360:test on `shards_v2`, full split**, 2567 pairs seen, 102 skipped for
carrying fewer than 64 gated pixels, 85 s:

| direction | pairs | R(−1) | R(0) | R(+1) | grid | parabolic | vote for +1 | z | explained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 1239 | 31.85 | 24.88 | **12.12** | **+1.0** | 0.9654 | 1205/1239 = 97.3% | 33.2 | **51.3%** |
| backward | 1226 | 32.00 | 23.95 | **14.28** | **+1.0** | 0.9582 | 1181/1226 = 96.3% | 32.4 | 40.4% |

**Both directions now read `identity`** — the stored flow *is* the motion, which
is exactly what a correct shard set must produce. Combined with the v1 reading
(forward −1, backward +1, §16.25), the shards that produced the retrain and every
row of the corrected table are photometrically certified, on the full split, by a
procedure containing no learned parameters.

**A number worth quoting in the thesis.** On flow360 the gate keeps ~1.4% of
pixels, and on that 1.4% the GT warp explains **51%** of the photometric
residual. §P0 measured that the same warp explains *nothing* over all pixels
(3.10 unwarped vs 3.12 warped). Both are correct, and together they are the
sharpest statement of the regime this project has: **the correspondence signal in
the sub-pixel regime is strong and it is intact — on the small minority of pixels
that move. What destroys it in aggregate is the static majority.** That is a
measurement of the problem the decision gate proposes to solve, not an analogy
for it.

Minor note: the parabolic minima sit slightly below 1 (0.958-0.992). Expected —
occlusion and appearance change contribute residual that no alpha removes, which
biases the fitted minimum toward 0. It is not a scale error.

**ROOT-CAUSE FIX SHIPPED IN sfprep.** `sfprep/diagnose.py` rewritten,
`constancy_gate()` added to `flow_io.py`, three CLI flags added
(`--min-motion-px`, `--edge-quantile`, `--max-abs-lat-deg`, plus `--datasets`).
Four changes, each closing one of the failure's paths:

1. **Gated scoring.** The photometric error is measured only where displacement
   exceeds `--min-motion-px` and gradient is in the top decile. The gate is built
   from |flow|, which is invariant under all eight candidate conventions, so every
   candidate is scored on identical pixels and the comparison stays paired.
2. **Per-direction diagnosis.** Records group by `(dataset, direction)`. When
   directions disagree the dataset-level convention is written as `"mixed"` and a
   banner prints; a single per-dataset setting for such a dataset is announced as
   wrong rather than silently averaged.
3. **A pin can no longer overrule a measurement.** `pin_convention` is honoured
   only while the diagnosis is inconclusive. A pin that contradicts a conclusive
   diagnosis **aborts** with both values in the message.
4. **Pin and default now come from the live config**, not from the `datasets.json`
   written by an earlier `build`. The file you edit was not the file the tool
   obeyed — a smaller instance of the same defect class.

**Acceptance test, run in Docker on the raw FLOW360 dataset** (30 samples per
direction, ~1.45% of pixels gated):

* On the *original* config, `pin_convention = true` and `default = identity`, the
  tool **aborts**: `[flow360:forward] PIN CONTRADICTS A CONCLUSIVE DIAGNOSIS:
  pinned 'identity', measured 'negated' (+66.2% over zero)`. It refuses to
  reproduce the defect on the exact configuration that caused it.
* With the pin removed it recovers both conventions unaided:
  **forward `negated`** (err 6.18 vs zero 18.27, **+66.2%**) and **backward
  `identity`** (9.52 vs 18.08, **+47.3%**), then prints the disagreement banner.
  In both directions the runner-up candidate is ~3.3× worse than the winner *and
  worse than zero-flow* — this is not a marginal call.
* Regression on known-good datasets: flowscape:forward `identity` (+42.7%),
  replica360 forward and backward `identity` (+72.5% / +73.4%), directions agree,
  no banner. All match the §16.18 audit.

`datasets.toml` no longer pins flow360. The comment that read `# validated
convention used by the model repo` is gone; the value it labelled was never
validated.

### 16.27 EMA on corrected data (2026-08-05) — the trade-off reappears inside one run

`P1proper_ema6k_v2`: 6k steps continued from the 20k `_v2` checkpoint, constant
lr 3e-5, decay 0.999, 7.3 h. flow360:val, acos, area-weighted.

| metric | 20k `_v2` | +6k raw | EMA |
| --- | --- | --- | --- |
| act₀.₂₅ | +58.83 | **+61.11** | +58.88 |
| act₀.₅ | +60.38 | **+62.80** | +59.61 |
| act₁.₀ | +35.31 | **+37.78** | +33.69 |
| global | +10.52 | +11.96 | **+16.09** |
| equator | +21.36 | +22.99 | **+26.04** |
| seam | +7.30 | +6.28 | **+10.68** |
| poles | −43.06 | −45.79 | **−30.91** |

**EMA no longer dominates.** In the v1 campaign weight averaging beat the raw
walk on every metric — that was the whole reason it was adopted, and it is why
the final v1 model was the EMA point. On corrected data the two checkpoints
split cleanly: **raw wins all three active thresholds** (act₀.₅ +62.80 vs
+59.61), **EMA wins every calibration metric** (global +16.09 vs +11.96, poles
−30.91 vs −45.79, a 15-point gap).

That is the actives-versus-calibration trade-off again, and this is the most
controlled instance of it this project has produced. Previous sightings compared
different training variants (SLOF's five checkpoints, §round 1) or different
labs (OSLO's actives above PanoFlow's while its global sits below, §16.23). Here
it is **two checkpoints from one run** — same data, same recipe, same optimiser
trajectory, differing only in whether the weights are averaged. Weight averaging
buys calibration and pays for it in commitment on the movers. The trade-off is
therefore not an artefact of how anyone trained; it is a property of the
objective on this data.

Converged ladder on flow360:val, act₀.₅ / global / poles:

| stage | act₀.₅ | global | poles |
| --- | --- | --- | --- |
| v1 P1d | +1.1 | −53.6 | — |
| v1 20k | +4.0 | −31.1 | −100.6 |
| v1 EMA (old final model) | +4.5 | −16.8 | −56 |
| **v2 20k** | +60.4 | +10.5 | −43.1 |
| **v2 +6k raw** | **+62.8** | +12.0 | −45.8 |
| **v2 EMA** | +59.6 | **+16.1** | **−30.9** |

The continuation is worth its 7.3 h on both corners: raw actives went +60.4 →
+62.8 and EMA global went +10.5 → +16.1 while cutting the polar deficit by 12
points.

**Poles remain the one negative region here** — −30.9% at best, against the
*lowest* zero error of any region on this split (0.1633°). Every other region is
positive on both checkpoints. (Amended by §16.28: the polar deficit is specific
to this split. On flow360:test, where the poles are not near-static, both
checkpoints are positive there.)

**Which model goes in the table is not yet decided**, and it should not be
decided on val — that is the selection pressure §round 2 identified. Both
checkpoints need the flow360:test eval under haversine, against frozen
RAFT-large's +39.32 global / +44.01 act₀.₅.

### 16.28 THE FINAL TABLE (2026-08-05) — the trade-off transfers, and the polar deficit was a split property

Both checkpoints on flow360:test, haversine, area-weighted, 2567 pairs, 8 min
each. Zero global 0.428648° and active fracs 34.6/18.0/5.8, identical to every
other row — byte-comparable throughout.

| row | global | act₀.₂₅ | act₀.₅ | act₁.₀ | equator | poles | seam |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen RAFT-large (zero-shot, perspective) | **+39.32** | **+46.27** | **+44.01** | **+37.82** | — | **+25.10** | **+23.18** |
| PanoFlow(CSFlow)+CFE (zero-shot, 360°-native) | +29.81 | — | +34.48 | — | — | — | — |
| **OSLO +6k raw** | +20.37 | +39.84 | **+40.22** | **+34.51** | +27.65 | +1.62 | +17.09 |
| **OSLO EMA** | **+22.44** | +38.80 | +38.32 | +32.41 | **+29.47** | **+4.95** | **+18.34** |
| OSLO 20k `_v2` | +19.17 | +37.87 | +37.66 | +31.48 | +26.89 | +0.10 | +16.48 |
| SLOF doublerotation (trivial zero-predictor) | −0.01 | — | +0.13 | — | — | — | — |
| SLOF singlerotation | −48.68 | — | −37.06 | — | — | — | — |

**1. The trade-off transfers, so it is not selection pressure.** The val ordering
was raw ahead on every active threshold and EMA ahead on every region metric.
Test reproduces it exactly: raw +40.22 vs +38.32 on act₀.₅, EMA +22.44 vs +20.37
on global. Had this been an artefact of tuning on val it would have washed out on
a disjoint, 2× harder pool. It did not. §16.27's claim stands on cross-pool
evidence: **the actives-versus-calibration split is a property of the objective
on this data, not of how anyone trained.** The gap does narrow — 3.19 → 1.90
points on act₀.₅, 4.13 → 2.07 on global — so the two corners are closer to each
other out of domain than in it.

**2. The 6k continuation generalises.** act₀.₅ +37.66 → +40.22 on *test*, not
just on val. Those 7.3 h bought 2.6 points cross-pool.

**3. The val→test crossing is now stable and quantified**: 22.7, 22.6 and 21.3
points of act₀.₅ across the three checkpoints. The EMA loses least, which is weak
evidence that weight averaging regularises against pool shift, but 1.3 points is
not enough to claim it.

**4. THE POLAR DEFICIT WAS A SPLIT PROPERTY, and this is the cleanest reading of
the static-calibration problem yet.** On flow360:val the poles carry the *lowest*
zero error of any region (0.1633°, i.e. they are near-static) and both checkpoints
are strongly negative there (−45.8 and −30.9). On flow360:test the poles carry
*more* motion than the global average (zero 0.4454° vs 0.4286°) and both
checkpoints are **positive** (+1.62 and +4.95). Same weights, same metric, same
code — the deficit appears exactly where the field is static and vanishes where it
is not. **The polar problem is not a polar problem. It is the static-confidence
problem, observed at whichever region happens to be static.** That is a direct
measurement in support of `docs/plans/DECISION_GATE.md`, and it retires "poles are
OSLO's weak region" as a formulation.

**5. OSLO is second on actives and third on global, and it loses to a frozen
perspective network on every axis.** Against RAFT-large: closest on act₀.₅
(+40.22 vs +44.01, 3.8 points) and furthest at the poles (+4.95 vs +25.10, 5×).
Against PanoFlow it now wins the actives clearly (+40.22 vs +34.48) while still
losing globally (+22.44 vs +29.81) — the same trade-off, across labs.

**The polar-advantage claim does not survive on this dataset, with the better
checkpoint.** §16.22 flagged it at +0.10 vs +25.10; the EMA improves OSLO to
+4.95 and RAFT-large still wins by 5×. The claim scopes to the large-motion
regime — replica360 and flowscape, where it replicated twice — and must be stated
with that scope everywhere it appears.

**Model selection, stated rather than implied.** Gate R2 was pre-registered on
act₀.₅, so the primary model is **`P1proper_ema6k_v2/oslo_raft.pt`** (raw
continuation): act₀.₅ +40.22 test / +62.80 val, and it wins all three active
thresholds on both splits. The EMA point is reported alongside it as the
calibration corner, not as a footnote — a pair of checkpoints from one run that
each win one half of the objective is the empirical case for a decision gate, and
it is more informative than either number alone.

### 16.29 The displacement-response curve — the whole failure is one band

Motion bands were already recorded in the corrected flow360:test runs. Improvement
over zero-flow per band, haversine, area-weighted:

| band | RAFT-large | PanoFlow+CFE |
| --- | --- | --- |
| 0 – 0.0625° | **−445.8** | **−321.9** |
| 0.0625 – 0.125° | +28.2 | +15.6 |
| 0.125 – 0.25° | +45.4 | +28.7 |
| 0.25 – 0.5° | +58.4 | +41.2 |
| 0.5 – 1° | +60.9 | +48.8 |
| 1 – 2° | +64.6 | +58.6 |
| 2 – 4° | **+65.2** | **+58.9** |
| 4 – 8° | +55.9 | +43.5 |
| 8 – 16° | +36.1 | +19.3 |
| 16 – 32° | +10.0 | +1.8 |
| > 32° | +4.6 | +0.0 |

**Every band above 0.0625° is positive, for both architectures.** The curve is an
inverted U: it climbs to +65% / +59% at 1–4°, then decays past 8° as displacement
outruns the correlation range, and it collapses to −446% / −322% in the lowest
band. The two labs' methods trace the same shape.

**This localises the entire flow360 deficit to one band.** The global figures
(+39.3 and +29.8) are not "moderate performance everywhere" — they are strong
performance at every magnitude the methods can address, dragged down by a single
band that carries the largest share of nodes. Magnitude is not the problem
*within* this dataset: the static majority is.

It also supersedes the cross-dataset band-matched control as the argument's
backbone. That control asked whether a fixed displacement behaves differently
across datasets; this asks where the error actually lives, and answers it without
needing a second dataset. The cross-dataset control is still unrun on corrected
data (flowscape:test rows carry no band breakdown).

**This is the decision-gate case, measured.** A per-node static/motion gate would
act on exactly the band that reads −446%, and every band it does not touch is
already positive. It also explains §16.28's polar finding directly: the poles read
negative on val, where they are near-static, and positive on test, where they are
not — same mechanism, seen through a regional mask instead of a displacement one.

### 16.30 OSLO by displacement band — the dead zone is 4× wider, and the gate ceiling is +35%

`P1final_test_flow360_v2_cont6k_bands`. Improvement over zero per band, with the
node mass of each band (fracs partition the sphere and the decomposition
reconstructs the global to 0.04%):

| band | nodes | RAFT-large | PanoFlow | **OSLO** |
| --- | --- | --- | --- | --- |
| 0 – 0.0625° | **32.8%** | −445.8 | −321.9 | **−1018.1** |
| 0.0625 – 0.125° | 15.8% | +28.2 | +15.6 | **−51.4** |
| 0.125 – 0.25° | 16.9% | +45.4 | +28.7 | **−0.9** |
| 0.25 – 0.5° | 16.6% | +58.4 | +41.2 | +37.8 |
| 0.5 – 1° | 12.2% | +60.9 | +48.8 | +55.8 |
| 1 – 2° | 3.5% | +64.6 | +58.6 | +62.4 |
| 2 – 4° | 1.0% | +65.2 | +58.9 | +56.9 |
| 4 – 8° | 0.7% | +55.9 | +43.5 | **+51.2** |
| 8 – 16° | 0.3% | +36.1 | +19.3 | **+33.1** |
| 16 – 32° | 0.1% | +10.0 | +1.8 | +8.4 |
| > 32° | 0.1% | +4.6 | +0.0 | +3.0 |

**Three readings.**

**1. OSLO's dead zone is four times wider.** Both raster methods cross into
positive at 0.0625°; OSLO crosses at 0.25°. It is negative in the three smallest
bands, which together hold **65.4% of the nodes** — that single fact explains the
whole gap between its actives (+40.2) and its global (+20.4).

**2. It over-commits twice as hard where nothing moves.** In the 0–0.0625° band
the truth moves 0.0167° and OSLO asserts 0.1863°, a factor of 11. RAFT-large
asserts 0.0909°, a factor of 5.5. The static-confidence problem is not shared
equally: OSLO has it worse than either published method.

**3. Above 4° it beats PanoFlow.** +51.2 vs +43.5 at 4–8°, +33.1 vs +19.3 at
8–16°. The large-displacement end is not where OSLO loses on this dataset.

**THE GATE CEILING, COMPUTED.** The band decomposition is exact, so the payoff of
a per-node static gate can be projected rather than guessed. Substituting the
zero-flow prediction for the model's own in the smallest bands:

| gated bands | global | improvement over zero |
| --- | --- | --- |
| none (measured) | 0.3413° | **+20.4%** |
| smallest 1 | 0.2859° | **+33.3%** |
| smallest 2 | 0.2784° | **+35.1%** |
| smallest 3 | 0.2781° | +35.1% |

**A perfect gate on the two smallest bands takes OSLO from +20.4% to +35.1%
global, leaving every active-node number untouched.** That lands between PanoFlow
(+29.8) and frozen RAFT-large (+39.3), from a head that adds no correlation
capacity. It is an oracle upper bound, not a promise — a learned gate will not be
perfect — but it converts A4 from "a plausible idea" into a target with a measured
ceiling and a known failure mode. This is the strongest single argument that the
project has a path worth funding.

### 16.31 The polar deficit, diagnosed — it is 1.29×, not 15×, and the mechanism is input sampling

**FIRST, THE NUMBER IS NOT WHAT IT LOOKS LIKE.** flow360:test, haversine,
area-weighted. RAFT-large's regional errors reconstructed from its published
improvements against the shared zero:

| region | zero | OSLO | RAFT-large | OSLO % | RAFT % | gap (deg) |
| --- | --- | --- | --- | --- | --- | --- |
| global | 0.4286 | 0.3413 | 0.2602 | +20.4 | +39.3 | 0.0812 |
| poles | 0.4454 | 0.4381 | 0.3336 | **+1.6** | **+25.1** | 0.1046 |
| seam | 0.9109 | 0.7553 | 0.6998 | +17.1 | +23.2 | 0.0555 |

**Polar tax, defined as a model's own polar error divided by its own global
error: OSLO 1.2837, RAFT-large 1.2825.** They are the same to three decimals.
The truth's own polar tax is 1.0390, so both models degrade at the poles by the
same 23% beyond what the content demands.

In degrees, OSLO trails RAFT by 0.105° at the poles and 0.081° globally — a
**1.29×** polar surcharge. The improvement-over-zero view reports the same facts
as **15.5×** (+25.1 vs +1.6), because that metric is a ratio and the poles are
where the zero baseline sits closest to both models: dividing by a number that is
barely above the numerator amplifies a small absolute gap without bound.

**Consequence for where the effort goes.** "OSLO loses at the poles" is mostly
"OSLO loses everywhere, and the polar percentage magnifies it". The dominant
lever is the global deficit, which §16.30 already localised to the three smallest
displacement bands. A genuine but secondary polar surcharge of 29% remains, and
the rest of this section is about that.

**THE MECHANISM HYPOTHESIS — the retina, not the estimator.** The retina samples
the ERP frame at HEALPix node directions with `bilinear_sample_erp`, a four-tap
kernel. How much raster each node stands for is a function of latitude, because
ERP pixel density per solid angle goes as 1/cos φ while HEALPix node density is
uniform:

| latitude | 0° | 30° | 45° | 60° | 75° | 85° | 89° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERP pixels per r=7 node | 1.7 | 2.0 | 2.4 | 3.4 | 6.6 | 19.5 | 97.3 |

Area-weighted inside the masks: **1.78 at the equator versus 6.61 at the poles, a
3.7× more aggressive decimation** performed by a 4-tap kernel. The retina holds
196,608 nodes against 524,288 ERP pixels, so it discards 62% of the raster
globally and far more than that at high latitude. Content above the node Nyquist
rate does not vanish — it folds back, and the model ingests aliasing as signal.
**RAFT-large reads the full raster and pays none of this**, which is exactly the
asymmetry the table above measures.

**Why SO(3) augmentation does not protect against it.** Training runs at
`so3_prob 1.0`, so the sphere is randomly rotated every sample and the content
distribution is rotation-uniform. But the *raster* keeps its own poles: rotation
moves which sphere direction a node reads, never where the ERP's own
oversampling lives. Evaluation runs at `val_so3_prob 0.0`, where the HEALPix
polar cap aligns with the ERP polar cap — the worst-aliased region of the input —
and no rotation-augmented training can teach a compensation for damage that the
augmentation itself scrambles.

**THE DIAGNOSTIC LADDER** (cheapest first, each with its reading pre-registered):

1. **Rotated evaluation**, `--set val_so3_prob=1.0`, 8 min, no code. If the polar
   deficit vanishes and the equator degrades ⇒ ERP raster sampling. If it
   vanishes and the equator holds ⇒ scene content at the poles of these
   sequences. If it persists ⇒ HEALPix grid geometry or the estimator itself.
2. **Retina aliasing probe**, `analyze_retina_aliasing.py`, model-free, minutes.
   Samples each frame twice — raw, and low-passed to the node spacing at each
   latitude — and reports the folded energy per latitude band, with an
   equiangular grid of equal node count as the control. Directly tests the
   mechanism above and measures the headroom an area-averaging prefilter would
   recover.
3. **Region × band cross-tabulation**, small change in `metrics.py` (intersect the
   region masks with the band masks). §16.30 showed the global deficit lives in
   bands below 0.25°; this asks whether the poles carry more mass there. The
   polar zero is 1.039× the global one, so the *mean* says poles move slightly
   more than average — but a mean hides a bimodal split of static cap plus fast
   movers, and only the cross-tab separates composition from per-band skill.
4. **Known measurement confound**, already quantified in §16.25: a geodesic
   target built from an ERP pixel-space flow carries a parameterisation asymmetry
   worth 7.45% at the poles against 0.33% at the equator. Part of the 29%
   surcharge is the ruler, not the model.

If the ladder lands on the retina, the fix is an area-correct prefilter before
sampling — cheap, one-time, and outside the network — and that is a far better
outcome than a capacity argument.

### 16.32 Polar diagnosis CLOSED — the aliasing hypothesis is dead, and there is no OSLO-specific polar defect

**STEP 1, ROTATED EVALUATION** (`P1final_test_flow360_v2_cont6k_rot`,
`--val-so3-prob 1.0`, flow360:test, 8 min):

| region | model error, unrotated → rotated | zero, unrotated → rotated |
| --- | --- | --- |
| equator | 0.2959 → 0.3387 (**+14.5%**) | 0.4089 → 0.4264 |
| global | 0.3413 → 0.3477 (+1.9%) | 0.4286 → 0.4304 |
| poles | 0.4381 → 0.3864 (**−11.8%**) | 0.4454 → 0.4379 |

Polar excess, defined as the model's poles/equator ratio divided by the truth's
own: **1.360 unrotated → 1.111 rotated. Rotation removes 69% of it**, while the
global error stays flat and the actives do not move (act₀.₅ +40.2 → +40.9). This
is redistribution, not repair: the deficit is attached to whatever sits at the
poles of these scenes.

**STEP 2, THE ALIASING PROBE — MY HYPOTHESIS WAS WRONG.**
`analyze_retina_aliasing.py`, model-free, 25 frames. Above-Nyquist energy the
retina folds in, per latitude band, in levels of 255:

| latitude | 0–15 | 15–30 | 30–45 | 45–60 | 60–75 | 75–90 |
| --- | --- | --- | --- | --- | --- | --- |
| HEALPix alias | 0.000 | 0.022 | **0.880** | 0.497 | 0.201 | 0.231 |
| equiangular alias (control) | 0.000 | 0.023 | 0.913 | 0.486 | 0.205 | 0.287 |
| band-limited contrast | 37.5 | 35.2 | 34.7 | 33.6 | **28.7** | **26.3** |

**Aliasing peaks at mid-latitude and *falls* toward the poles, never exceeding 1%
of the signal.** §16.31 was right that the retina decimates 3.7× harder inside the
polar mask, and wrong about the consequence: ERP polar rows are longitudinally
redundant by construction, so the content there is already smooth and there is
almost nothing above the node Nyquist rate to fold. The equiangular control tracks
HEALPix to three decimals, confirming the probe measures the raster and not the
grid. **The prefilter fix proposed in §16.31 would buy nothing and is withdrawn.**

Two process notes. The probe's first run reported 144/255 of "aliasing" with the
control identical — impossible on its face, and the tell that found the bug: the
box-blur summed a `2k`-wide window and divided by `k`, doubling the image. Fixed,
then gated on a constant image (must survive untouched) and on variance (must
fall) before any number was read. Second, the ladder's own pre-registration was
defective: rotating the sampling grid moves the content *and* the raster position
together, so steps 1's "equator degrades ⇒ raster" reading could never have
separated them. It took the model-free probe to do that.

**What the two measurements say together.** The deficit lives at the poles of the
scene; the raster contributes under 1%; and the band-limited contrast column shows
polar content carries **26–29 against 35–37 at the equator, roughly 25% less
texture**. Correlation has less to lock onto there, so the model leans on its
prior and commits — the same over-commitment §16.30 measured as a factor of 11 in
the static band.

**AND THE DECISIVE FRAMING, FROM §16.31: THERE IS NO OSLO-SPECIFIC POLAR DEFECT
ON THIS DATASET.** Polar tax, each model's polar error over its own global error:
OSLO **1.2837**, frozen RAFT-large **1.2825**, truth 1.0390. Both models degrade
at the poles by the same 23% beyond what the content demands. OSLO is not
disproportionately bad at the poles — it is uniformly less accurate, and the
improvement-over-zero ratio magnifies that at the one region where the trivial
baseline is closest to both.

**Diagnosis closed, and it collapses into the work already planned.** There is no
separate polar problem to solve: fix the global deficit — which §16.30 localised
to the bands below 0.25° and priced at +35.1% global for a perfect gate — and the
polar number follows. The A4 decision gate is the single lever for both. What
does *not* survive: any plan to attack the poles through the retina, the grid, or
sampling geometry.

### 16.33 THE ERP POLAR PENALTY IS PROPORTIONAL TO DISPLACEMENT — the uniformity claim is regime-bound, with a mechanism

The missing equator rows arrived, and they settle both questions at once.
flow360:test, haversine, area-weighted:

| | equator | poles | poles/equator | excess over the truth's own ratio |
| --- | --- | --- | --- | --- |
| zero (the truth) | 0.4089 | 0.4454 | 1.089 | — |
| **OSLO** | 0.2959 | 0.4381 | **1.481** | +36.0% |
| frozen RAFT-large | 0.2294 | 0.3336 | **1.454** | +33.5% |
| PanoFlow+CFE | 0.2661 | 0.3679 | **1.383** | +26.9% |

**1. OSLO's deficit has NO regional structure.** Dividing region by region,
OSLO / RAFT-large reads **1.290 at the equator, 1.312 globally, 1.313 at the
poles**. OSLO is uniformly about 30% less accurate everywhere. The dramatic
+1.6% versus +25.1% polar comparison is therefore an artefact of the
improvement-over-zero ratio in full: there is no polar-specific defect left to
explain, and §16.31's reframing is confirmed with the third region in hand.

**2. THE UNIFORMITY ADVANTAGE COLLAPSES, AND NOT BECAUSE OSLO GOT WORSE.**

| dataset | OSLO | RAFT-large | OSLO flatter by |
| --- | --- | --- | --- |
| flowscape:test | 4.33 | **10.40** | 2.40× |
| replica360 | 2.37 | **5.36** | 2.26× |
| **flow360:test** | 1.48 | **1.45** | **0.98× (tied, marginally behind)** |

RAFT-large's own polar ratio falls from 10.40 to 1.45 — a **7.2× collapse** —
while OSLO's falls from 4.33 to 1.48, a 2.9× collapse. Both converge on ~1.45.
The advantage did not evaporate because the spherical grid stopped working; it
evaporated because **the handicap it compensates for stopped existing**.

**3. THE MECHANISM: the ERP polar penalty scales with displacement magnitude.**
A displacement of $d$ degrees at latitude $\varphi$ occupies $d / (0.3516 \cos
\varphi)$ ERP pixels. At 85° on a 1024×512 raster:

| | displacement | ERP px at 85° | ERP px at equator |
| --- | --- | --- | --- |
| flowscape p50 | 2.467° | **80.5** | 7.0 |
| flowscape p90 | 7.290° | **237.9** | 20.7 |
| flow360 p50 | 0.132° | **4.3** | 0.4 |
| flow360 p90 | 0.756° | 24.7 | 2.2 |

At large motion, near-polar displacements reach **80 to 238 ERP pixels**. RAFT
correlates at 1/8 resolution with a lookup radius of 4, so 238 px is 30 cells —
far outside the finest level, forcing the match onto coarse levels that cannot
place it precisely. That is the polar failure the equal-area grid removes: the
same 7.29° is exactly 2.0 node spacings at r=4, comfortably inside the lookup
rings, at every latitude.

At sub-pixel motion the same arithmetic runs the other way. 0.132° is 4.3 px at
85° against 0.4 px at the equator, so **the ERP's polar oversampling magnifies an
otherwise unresolvable displacement into a measurable one**. RAFT is never pushed
outside its search range, the stretch cannot destroy an appearance match over 4
pixels, and its polar handicap simply does not fire. OSLO gets no such help: its
estimation grid is 3.665° at every latitude, so 0.132° is 0.036 of a node
spacing **everywhere on the sphere**.

**The equal-area grid is uniformly resolution-limited; the ERP raster is
non-uniformly resolution-limited, and near the poles it is over-resolved — which
is a liability at large motion and an asset at sub-pixel motion.**

**4. All three architectures degrade at the poles by 27–36% beyond what the
content demands**, and they sit within nine points of each other — one
sphere-native, two raster-native. In this regime the polar excess is a property
of the data, not of the representation, which is the same conclusion §16.32
reached from the texture measurement (polar contrast 26–29 against 35–37).

**Consequence for the thesis.** "Native spherical geometry buys polar accuracy
and sphere uniformity" survives, replicated twice, but it must carry its scope in
the sentence: **it holds where displacement is large enough that the ERP polar
stretch exceeds a raster method's search range.** That is now a mechanism with an
arithmetic threshold, not a caveat — a stronger claim than the unscoped version,
because it predicts where the advantage appears and where it will not. The
counterpart is that OSLO's uniformity advantage on flow360 is nil, and the
article must say so.

### 16.34 Orientation robustness — instrument built, experiment pre-registered

A 360° camera on a drone or a head tilts constantly, and the two representations
answer that differently: the ERP has a privileged axis, a sphere grid does not.
No paper in the §16 review reports this. Half the measurement already exists —
under full random SO(3) rotation OSLO's global error moves 0.3413° → 0.3477°,
**+1.9%** (§16.32). The other half needs the raster methods rotated the same way.

**Instrument.** `run_raft_shard_baseline.py` gains `--val-so3-prob` (plus
`--val-so3-max-angle-deg`, `--val-so3-uniform`, `--val-so3-seed`). The predictor
receives an ERP **re-rendered** under the rotation: for each output pixel
direction `d`, the rotated raster takes the real raster at `d @ R`, matching
`so3_augment_pair`'s convention exactly, so reading the rotated raster at an
unrotated node returns what the node-sampling path returns and the target that
`so3_augment_pair` produces scores both with no further bookkeeping.

**The fairness point that makes the comparison mean something**: each side pays
exactly one bilinear resampling — OSLO samples the real ERP at rotated node
directions, the raster method reads an ERP built by one bilinear. Neither carries
an interpolation advantage. `--predictor oracle` is refused under rotation, since
it would score the unrotated GT raster.

**Gates, Docker-validated:** identity rotation is a no-op (max 1.2e-5); rotated
output stays in range; and the two invariants that actually prove the transform —
rotating the sphere cannot change the distribution of |displacement|, and it does
not (`target_geo_deg_p50` 0.056114 → 0.056106, 0.01%; `active_0_5_frac` 0.10243 →
0.10269). Regional zero baselines converge as content is redistributed
(poles/equator 0.755 → 1.082 on an 8-pair probe), which is the expected
signature. With `--val-so3-prob 0` the original code path is taken unchanged.

**PRE-REGISTERED READING.** OSLO degrades +1.9%. If RAFT-large and PanoFlow
degrade by ≥10%, the claim is **sphere-native estimation is robust to camera
orientation and raster methods are not** — a property with a practical
motivation, an architectural cause, and no prior report in this literature. If
they degrade by a comparable ~2%, the claim dies and the cost was one afternoon;
that outcome is worth knowing too, because it would mean ERP methods tolerate
orientation better than the privileged-axis argument predicts.

**Caveat to carry:** OSLO's rotated run drew its rotations from `run_oslo_raft`'s
generator and these draw from `--val-so3-seed`, so the sequences differ. Across
2567 pairs the comparison is distributional, not paired.

### 16.35 RESULT — orientation robustness REFUTED, and an unplanned finding that survives it

flow360:test, 2567 pairs, `--val-so3-prob 1.0`, same seed for both raster runs.

| model | global, unrotated → rotated | degradation | improvement, unrot → rot |
| --- | --- | --- | --- |
| OSLO-RAFT | 0.34132 → 0.34771 | **+1.87%** | +20.37 → +19.23 |
| frozen RAFT-large | 0.26010 → 0.26436 | **+1.64%** | +39.32 → +38.59 |
| PanoFlow+CFE | 0.30087 → 0.30554 | **+1.55%** | +29.81 → +29.02 |

**The pre-registered hypothesis is refuted, and not narrowly: all three degrade by
1.5–1.9%, and OSLO degrades the most.** The residual is consistent with the one
bilinear resampling every arm pays. Sphere-native estimation buys no measurable
robustness to camera orientation on this data, and §16.34's claim is withdrawn.

**Why it failed is not a mystery, and §16.33 already predicted it.** The
orientation argument is a corollary of the polar-penalty argument: rotating the
scene hurts a raster method by moving content into the polar stretch. But §16.33
measured that the ERP polar penalty is proportional to displacement, and at
flow360's 0.132° median it has already collapsed to nothing — RAFT's own
poles/equator ratio is 1.45 here against 10.40 on flowscape. There is no penalty
left for rotation to trigger. **The correct prediction, which follows from the
same arithmetic rather than from a new guess, is that orientation robustness can
only appear in the large-motion regime**, and flowscape:test is the one-command
test of it.

**THE UNPLANNED FINDING.** Rotation uniformises content across regions, which is
exactly the control the raw regional comparison lacks. Under that control:

| model | equator | poles | poles/equator | excess over the truth |
| --- | --- | --- | --- | --- |
| zero (truth) | 0.42628 | 0.43673 | 1.025 | — |
| **OSLO-RAFT** | 0.33871 | 0.38644 | **1.141** | **+11.4%** |
| PanoFlow+CFE | 0.29177 | 0.34486 | 1.182 | +15.4% |
| frozen RAFT-large | 0.24904 | 0.31210 | 1.253 | +22.3% |

**With content controlled, OSLO is the flattest of the three, and RAFT-large's
polar excess is 2× OSLO's.** The unrotated ordering (OSLO 1.481, RAFT 1.454,
PanoFlow 1.383) inverts. The raw comparison was confounded: it scored each region
on whatever that dataset happens to put there, and flow360's poles carry 25% less
texture (§16.32), which penalises the *region* rather than the representation.

This partially rescues the uniformity claim, and it changes its form. It is not
"OSLO's polar error is lower" — on flow360 it is not. It is **"per unit of
content, OSLO's error varies least with latitude"**, and that holds in the
sub-pixel regime where the raw comparison said it did not.

**Discipline, because this was not pre-registered.** §16.34 registered a
prediction about orientation and that prediction failed; the regional table above
is exploratory, found while reading a null result. It needs its own replication
before it enters the article as a claim — on flowscape:test and replica360 under
the same rotation, where a content-controlled comparison also removes the
in-domain confounds. Until then it is a hypothesis with one supporting
measurement, which is exactly what the aliasing hypothesis was before its probe
killed it.

### 16.36 The regime prediction CONFIRMED on the first arm — PanoFlow degrades 286% under rotation at large motion

§16.35 closed with a prediction derived from §16.33's arithmetic rather than from
a new guess: orientation sensitivity is a *large-motion* phenomenon, because the
ERP polar penalty it depends on is proportional to displacement. flowscape:test,
1386 pairs, `--val-so3-prob 1.0`, same harness and seed as the flow360 leg.

| PanoFlow(CSFlow)+CFE | global | poles | equator | poles/equator |
| --- | --- | --- | --- | --- |
| unrotated | **0.251°** (+92.6) | 0.379° (+95.3) | 0.262° (+85.5) | 1.449 |
| rotated | **0.968°** (+71.42) | 1.742° (+58.33) | 0.801° (+73.40) | 2.173 |
| | **+285.6% (3.86×)** | | | |

**Same weights, same rotation protocol, two regimes: +1.55% degradation at
flow360's 0.132° median, +285.6% at flowscape's 2.47° median — a factor of 184.**
The prediction was quantitative and it held.

**The task did not get harder — only its placement on the raster changed.**
Rotation is an isometry of the sphere, so the motion statistics must be
invariant, and they are: target p50 2.4686 vs 2.467, p90 7.2945 vs 7.290, active
fracs 92.25/85.01/73.67 vs 92.3/85.0/73.7, global zero 3.3869 vs 3.402 (−0.4%).
Every number a "rotation just makes it harder" explanation would need to move
stayed put. What moved is *where the content sits*: the zero baseline's own
poles/equator ratio collapses 4.47 → 1.39, i.e. flowscape's large motion is
concentrated near the poles unrotated and spread evenly once rotated.

**What this does NOT yet establish.** One arm is not a comparison. The claim
under test is *relative* — that a sphere-native estimator is less orientation-
sensitive than a raster one — and it needs OSLO and RAFT-large on the same run.
Two readings remain open until then:

1. *The claim is alive.* OSLO degrades little, and PanoFlow's published 4.6× lead
   over OSLO on this benchmark (§7) turns out to be contingent on scene
   orientation.
2. *The claim is dead.* OSLO degrades comparably, and rotation is simply a harder
   placement for every method, sphere-native or not.

**Pre-registered reading, before the two controls run.** OSLO unrotated is
1.158°. If OSLO's degradation is ≤ 20% (≤ 1.39°) while PanoFlow's is 286%, the
gap closes from 4.6× to ≤ 1.4× and the orientation claim is alive with a
mechanism. If OSLO degrades ≥ 100%, the claim dies and this section joins §16.32
and §16.34 on the pile of refuted hypotheses. Anything between is a partial
result to be reported as such, not rounded toward the hypothesis.

**The one confound that survives, and its control.** Rotating a raster costs the
raster arms one extra bilinear resample that OSLO does not pay — OSLO samples
nodes once from the original frame at rotated directions, while `rotate_erp`
builds a resampled ERP for the raster arms. The flow360 leg bounds that cost at
1.55%, but blur cost need not be regime-invariant. The clean control is a
**small-angle rotation**: `--val-so3-max-angle-deg 15` pays the identical
resampling while barely moving content. If PanoFlow stays near 0.251° there, the
resampling is exonerated and the 3.86× is orientation. Cost: eight minutes.

### 16.37 THE CONTROLS LAND — the orientation hypothesis dies a second time, and a much better finding replaces it

flowscape:test, 1386 pairs, `--val-so3-prob 1.0`, all three arms through the same
harness. The zero baseline, p50, p90 and active fracs match across all rows to
four figures, so the task is bit-comparable.

| model | global unrot → rot | degradation | act₀.₅ unrot → rot | degradation |
| --- | --- | --- | --- | --- |
| PanoFlow(CSFlow)+CFE | 0.251 → **0.968** | **+285.6%** | 0.277 → 1.102 | +297.6% |
| OSLO-RAFT | 1.158 → **1.415** | **+22.2%** | 1.306 → 1.613 | +23.5% |
| frozen RAFT-large | 0.872 → **0.918** | **+5.2%** | 1.005 → 1.058 | +5.2% |

**The pre-registered prediction failed, and it failed in the direction that kills
the hypothesis outright.** §16.36 registered "OSLO ≤ 20% while PanoFlow is at
286% ⇒ the claim is alive". OSLO came in at 22.2% — but the number that settles
it is not OSLO's, it is RAFT-large's **+5.2%**. The *perspective raster* model is
four times more orientation-robust than the sphere-native one. Sphere-native
estimation buys no robustness to camera orientation in either regime, and OSLO
carries `so3_prob 1.0` in training, so it cannot even be excused as unaugmented.
**The orientation line (§16.34, §16.36) is closed as refuted.**

**But the run found something better than what it was looking for.** Two raster
models, identical input pipeline, identical `rotate_erp` resampling, same seed
and split: one moves 5.2% and the other 286%. The difference between them is not
architecture — it is that **PanoFlow was trained on this benchmark and RAFT-large
never saw it**.

**PanoFlow's near-saturation of its own benchmark is orientation-contingent.**
Rotation is an isometry of the sphere: p50 2.4686 vs 2.467, p90 7.2945 vs 7.290,
active fracs 92.25/85.01/73.67 vs 92.3/85.0/73.7, global zero 3.3869 vs 3.402.
Every motion statistic is preserved. Under that null transform PanoFlow loses
**78% of its margin over zero-flow** and ends up **behind out-of-domain frozen
RAFT-large** (0.968 vs 0.918). Its published lead over OSLO collapses from
**4.61× to 1.46×**.

**The resampling confound is closed, and by a better control than the one I
planned.** §16.36 proposed a small-angle probe; it ran (PanoFlow, 15°) and is
ambiguous on its own — global +80.1%, but decomposed it is **equator +9.4% and
poles +363%**, with polar motion magnitude unchanged (`poles_zero` 7.981 vs
8.065). So 15° is not a small perturbation in raster terms at the poles, and the
probe bounds interpolation cost at the equator only. The decisive control is
**RAFT-large**: it pays the identical resample, including the aliasing incurred
when polar content is mapped equatorward, and it moves 5.2%. Whatever the
resampling costs, it does not cost 286%.

**Why this matters more than the hypothesis it replaced.** It is the same failure
family as the two the thesis already documents. A benchmark with a canonical
camera orientation rewards a model for learning where things usually are, that
component is indistinguishable from correspondence competence in the reported
number, and **no published protocol measures it** — exactly as no protocol
measured the flow convention (§16.25) and none reported a zero-flow baseline.
Random SO(3) at evaluation is a one-line control that separates the two, and it
is free.

**And §16.35's unplanned finding does NOT replicate — withdrawn.** It was
flagged as exploratory and requiring exactly this run. Content-controlled
poles/equator on flowscape, each arm normalised by its own rotated zero ratio:

| model | rot poles/equator | own zero | normalised | equator − poles (pts) |
| --- | --- | --- | --- | --- |
| frozen RAFT-large | 1.925 | 1.387 | **1.388** | **9.8** |
| OSLO-RAFT | 1.974 | 1.414 | 1.396 | 15.3 |
| PanoFlow+CFE | 2.173 | 1.387 | 1.567 | 15.1 |

On flow360 the ordering was OSLO 1.141 < PanoFlow 1.182 < RAFT 1.253. Here
RAFT-large is flattest, OSLO ties PanoFlow, and the second normalisation
(improvement-point gap) puts OSLO last. **"Per unit of content, OSLO's error
varies least with latitude" does not survive its own replication and must not
enter the article.** One dataset supported it, one refuted it, and it was
registered as needing the second before it counted.

**Standing tally of this line**: aliasing refuted (§16.32), orientation
robustness refuted twice (§16.35, here), content-controlled uniformity refuted
(here). What the runs produced instead is a control that indicts a published
result, which is worth more to the thesis than any of the three would have been.

### 16.38 CONSEQUENCE — the same control threatens the thesis' central positive claim

The article's headline positive result is that spherical geometry buys polar
accuracy and uniformity against a perspective architecture, replicated on
replica360 and flowscape. The rotated flowscape run is a content control on
exactly that claim, and it does not pass. Polar tax = a model's polar error
divided by its own global error (§16.31's instrument):

| comparison | OSLO | frozen RAFT-large | verdict |
| --- | --- | --- | --- |
| replica360, raw | 1.839 | 3.151 | OSLO much flatter |
| flowscape, raw | 2.718 | 4.186 | OSLO much flatter |
| **flowscape, rotated** | **1.608** | **1.596** | **dead tie** |
| flow360, raw (§16.31) | 1.284 | 1.283 | **dead tie** |

**Both controls we have applied erase the advantage.** flowscape rotated
uniformises content by force; flow360 needs no rotation because its motion is
already near-uniform across latitude. The advantage survives only in the two raw
large-motion comparisons — and flowscape's raw zero baseline has a poles/equator
ratio of **4.47**, i.e. that dataset puts its large motion at the poles. Rotate
it to 1.39 and the advantage is gone.

Under rotation RAFT-large also beats OSLO at the poles in absolute terms (1.465°
vs 2.276°), reversing the raw ordering (3.650° vs 3.147°).

**This is §14.8's lesson a second time: the measurement stands, the attribution
falls.** OSLO does have lower polar error on both raw benchmarks, and that is
what a user sees on data with natural content placement. What is no longer
supported is *why* — "the equal-area grid removes the polar handicap" predicts an
advantage that persists under content control, and it does not.

**The decider is replica360 under rotation.** It is the independent replication
the claim rests on, and it is one command. Pre-registered reading, written before
it runs:
- OSLO's polar tax stays below RAFT-large's ⇒ the claim survives, scoped to
  "holds on natural content placement, absent under forced uniformity", and the
  flowscape result becomes a boundary rather than a refutation.
- The taxes tie ⇒ the abstract, §4.4 and the contributions list all overstate,
  and the honest headline becomes the regime-contrast result plus the
  orientation-contingency finding of §16.37, which is the stronger pair anyway.

Do not edit the article's claim until this runs. Do not soften it in advance
either — three refutations in one afternoon is a reason for care, not for
pre-emptive retreat.

### 16.39 THE DECIDER — the claim SURVIVES on replica360, and strengthens under the control

replica360:val, 162 pairs, `--val-so3-prob 1.0`, both arms rerun today on build
5dcac0dc. Zero baselines match to five figures (13.4075 both), so the two are
scored on the same task.

| | OSLO unrot | **OSLO rot** | RAFT unrot | **RAFT rot** |
| --- | --- | --- | --- | --- |
| global | 1.5640 | **1.3804** | 1.1582 | **1.0714** |
| poles | 2.8760 | **2.5036** | 3.6493 | **3.8333** |
| equator | 1.2116 | **1.0822** | 0.6808 | **0.4879** |
| **polar tax** | 1.8389 | **1.8137** (−1.4%) | 3.1508 | **3.5777** (+13.5%) |
| poles/equator | 2.374 | **2.313** | 5.360 | **7.857** |

**§16.38's branch 1 fires.** OSLO's polar tax is unmoved by forced content
uniformity; RAFT-large's gets 13.5% worse. OSLO's uniformity advantage grows from
**2.26× to 3.40×**, and it wins the poles in absolute terms by a wider margin
under the control than without it (2.504° vs 3.833°, against 2.876° vs 3.649°).

**The decisive comparison needs no cross-vintage arithmetic.** OSLO 1.8137 vs
RAFT 3.5777 is within the rotated condition, both arms run today on the same
build, same seed, same split. That single row is the content-controlled claim,
and it is a 1.97× gap in OSLO's favour.

**The two datasets disagree, and the mechanism says why — §16.33 again.**

| dataset | zero poles/equator | OSLO tax rot | RAFT tax rot | verdict |
| --- | --- | --- | --- | --- |
| replica360 | **1.024** (uniform) | 1.814 | 3.578 | advantage holds |
| flowscape | 1.387 (partly controlled) | 1.608 | 1.596 | advantage gone |

flowscape *unrotated* has a zero poles/equator of **4.47**: that dataset puts its
large motion at the poles, which is exactly where the ERP penalty
$d/(0.3516\cos\varphi)$ is catastrophic for a raster method. Rotation **relieves**
RAFT of that placement and its tax collapses 4.19 → 1.60. replica360's motion is
already latitude-uniform, so there is nothing to relieve and nothing collapses.

So the flowscape result is not a refutation of the claim, it is a measurement of
how much of the *raw* flowscape margin was placement rather than geometry — and
the answer is all of it. The claim's correct form:

> Against a perspective architecture on ERP, the equal-area grid reduces the
> polar tax by roughly 2×, and this survives forcing content to be uniform across
> latitude (replica360). Where a benchmark additionally concentrates large motion
> at the poles (flowscape), the raw margin overstates the geometric effect,
> because that placement penalises the raster method on its own terms.

Both halves are measured, and the second is a boundary the thesis states itself
rather than one a committee finds.

**Two loose ends, one command each, neither touching the decisive row.**

1. *Both globals improved under rotation* (OSLO −11.7%, RAFT −7.5%). That
   direction is unexplained and the unrotated runs are an older vintage that
   recorded no `geodesic_metric` key. Rerun both unrotated on the current build
   before quoting any unrot→rot delta. The rotated-condition comparison is
   unaffected.
2. *The flowscape rotation did not fully uniformise* — zero poles/equator went
   4.47 → 1.387, not → 1.0, because `--val-so3-uniform` was never set and the
   default angle distribution is not Haar. §16.37's flowscape control is
   therefore **partial**, and its numbers understate how far the placement
   confound reaches. replica360 landed at 1.024 only because its content was
   already uniform.

### 16.40 Both loose ends pulled — one closes, the other reopens §16.37

**Loose end 1 — build drift: ruled out.** `raft_erp_replica360_val` rerun on
build 5dcac0dc returns 1.1582 / 3.6493 / 0.6808, **bit-identical** to the old
vintage. No code has moved under that row. (OSLO's `_cur` leg is still pending.)

**And it delivered something better than the check it was run for.** The rerun
records the zero baseline the old vintage never wrote down:

| replica360:val, zero-flow | poles | equator | poles/equator |
| --- | --- | --- | --- |
| unrotated | 13.9596 | 13.2751 | **1.0516** |
| rotated | 13.6399 | 13.3171 | 1.0242 |

**replica360's motion is already latitude-uniform, unrotated.** The dataset does
not concentrate motion at the poles the way flowscape does (4.47). So the raw
replica360 head-to-head in the article — OSLO tax 1.8389 vs RAFT 3.1508 — **was
already a content-controlled comparison**, and the rotation run confirms rather
than rescues it. The claim never depended on the control it was being asked to
survive. This is the strongest footing the positive result has had.

**Loose end 2 — the flowscape rotation was a bad control, and by a lot.** With
`--val-so3-uniform` (Haar), same model, same split, same seed:

| RAFT-large on flowscape:test | global | polar tax | poles/equator | **zero p/e** |
| --- | --- | --- | --- | --- |
| unrotated | 0.8720 | 4.186 | 10.399 | **4.471** |
| rotated, non-Haar (§16.37) | 0.9177 | 1.596 | 1.925 | **1.387** |
| **rotated, Haar** | 0.9268 | **1.099** | **1.108** | **0.988** |

Only the Haar leg actually uniformises the content (zero p/e 0.988). The default
sampler left a 39% residual concentration, and RAFT's measured polar tax differs
by **45%** between the two rotations. **§16.37 and §16.38's flowscape rows are
computed on a partial control and must be re-read on the Haar leg before any of
them is quoted.** The direction of the error is known: the partial control
understated how much of the raw margin was placement.

Note what did *not* move: RAFT's global goes 0.872 → 0.927 (+6.3%) under Haar,
against +5.2% non-Haar. The model is genuinely orientation-robust; it is the
*regional* split that the sampler was mismeasuring.

**A clean two-point confirmation of §16.33, with content held uniform.** RAFT's
polar tax, measured only where the zero baseline is flat across latitude:

| condition | median displacement | RAFT polar tax |
| --- | --- | --- |
| flowscape, Haar-rotated (zero p/e 0.988) | 2.47° | **1.099** |
| replica360, native (zero p/e 1.052) | 11.88° | **3.151** |

The ERP polar penalty scales with displacement, and this is the first
measurement of it that owes nothing to where a dataset happens to put its motion.
It also predicts the shape of the OSLO comparison: the geometric advantage should
be small on flowscape and large on replica360 — which is what the raw numbers
said before any of this, for a reason that is now measured rather than asserted.

**Still outstanding**: OSLO on replica360 `_cur`, OSLO on flowscape `_rotu`, and
PanoFlow on flowscape `_rotu` — the last because §16.37's headline +285.6% is
also a non-Haar number and must be restated on the same footing as everything
else.

### 16.41 Haar controls land — the PanoFlow finding strengthens, and the polar penalty is a THRESHOLD, not a proportionality

**Build drift ruled out on both arms.** OSLO's replica360 rerun on 5dcac0dc
returns 1.5640 / 2.8760 / 1.2116, bit-identical to the old vintage, as RAFT's
did. The article's `tab:h2h` replica360 column needs no restating.

**§16.37's headline survives the proper control and gets bigger.** PanoFlow on
flowscape:test, three conditions, same split and seed:

| PanoFlow(CSFlow)+CFE | global | degradation | polar tax | poles/equator |
| --- | --- | --- | --- | --- |
| unrotated | 0.251 | — | 1.510 | 1.447 |
| rotated, non-Haar | 0.968 | +285.6% | 1.799 | 2.173 |
| **rotated, Haar** | **1.039** | **+313.9%** | **1.377** | **1.474** |

Under a genuine Haar rotation PanoFlow loses **314%** while frozen RAFT-large
loses 6.3%, and PanoFlow now sits clearly behind it (1.039 vs 0.927). The
orientation-contingency of a published in-domain result is confirmed on the
control it should have had from the start.

**A second split inside that result, which the Haar leg separates cleanly:**

| | global under rotation | polar tax under rotation |
| --- | --- | --- |
| PanoFlow | collapses (0.251 → 1.039) | **stable** (1.510 → 1.377) |
| RAFT-large | **stable** (0.872 → 0.927) | collapses (4.186 → 1.099) |

PanoFlow's cross-sphere uniformity is real and content-independent; its *global
competence* is what depends on orientation. RAFT-large is the mirror image: it is
genuinely orientation-robust, and its raw flowscape polar disaster (p/e 10.4) was
almost entirely **placement**, not architecture.

**The refinement: §16.33 said the ERP polar penalty is proportional to
displacement. It is better described as a threshold.** Area-weighted mean
$\sec\varphi$ is 3.908 over $|lat|>60$ and 1.047 over $|lat|<30$, so mean polar
ERP displacement is $11.11 \times d$ pixels at 1024 columns. RAFT correlates at
1/8 resolution with lookup radius 4, i.e. a **32 px reach** at full resolution:

| dataset | p50 | polar ERP px | vs 32 px reach | RAFT polar tax |
| --- | --- | --- | --- | --- |
| flow360 | 0.132° | **1.5** | far inside | 1.28 (§16.31) |
| flowscape, Haar | 2.469° | **27.4** | just inside | **1.099** |
| replica360, native | 11.878° | **132.0** | **4.1× beyond** | **3.151** |

Flat, flat, jump — not a slope. The penalty is nil while polar displacement fits
the correlation reach and bites hard once it does not. flow360's 1.28 sits above
flowscape's 1.099 despite 18× less motion because that row is *not*
content-controlled: flow360's poles carry 25% less texture (§16.32).

**This is what makes the two datasets agree instead of contradict.** The
equal-area grid can only help where the raster method is actually failing, and
that is a measurable threshold rather than a claim about spheres. replica360
crosses it by 4×, flowscape does not cross it at all — so the geometric advantage
must appear on the first and vanish on the second, which is exactly the pattern.
The equatorial numbers corroborate: replica360's equator sits at 35.4 px, right
at the reach boundary, and RAFT's equator error there (0.681) is double its
flowscape equator (0.351) at 7.4 px.

**Outstanding: OSLO on flowscape `_rotu`**, the last cell of the table.
Pre-registered: the threshold argument predicts OSLO's Haar polar tax lands near
RAFT's 1.099 rather than below it, because at 27.4 px there is no raster failure
left to fix. A value clearly below 1.099 would mean the grid buys something the
threshold model does not account for, and that would need its own explanation.

### 16.42 TABLE CLOSED — the prediction held, the two-dataset replication did not

flowscape:test under Haar rotation, all three arms, content uniform across
latitude:

| model | global | +% over zero | poles | equator | polar tax | poles/equator | normalised |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen RAFT-large | **0.927** | **+72.6** | **1.019** | **0.919** | **1.099** | **1.108** | **1.122** |
| PanoFlow+CFE | 1.039 | +69.3 | 1.431 | 0.970 | 1.377 | 1.474 | 1.492 |
| OSLO-RAFT | 1.498 | +55.8 | 2.030 | 1.320 | 1.355 | 1.538 | 1.428 |

(Normalised = poles/equator divided by that run's own zero poles/equator; the
OSLO leg draws rotations from `seed 7` while the raster legs use
`--val-so3-seed 1234`, so its zero band ratio is 1.077 against their 0.988 and
the normalisation is required rather than cosmetic.)

**§16.41's pre-registration held: OSLO's Haar polar tax is 1.355, near RAFT's
1.099 and not below it.** The threshold model predicted the sign correctly. It
did not predict the size — OSLO is 23% *worse*, so on flowscape under content
control the polar advantage is not merely absent, it is reversed.

**The consequence for the thesis is concrete and it is a demotion.** The article
currently claims the uniformity result is *replicated on two independent
datasets* (replica360 2.37 vs 5.36, flowscape 4.33 vs 10.4). The flowscape half
of that replication is now measured to be **placement, not geometry**: that
dataset concentrates its large motion at the poles (zero poles/equator 4.47),
which is precisely where a raster method is penalised on its own terms. Remove
the concentration and RAFT's polar tax falls 4.19 → 1.10 while OSLO's falls only
2.72 → 1.36.

**What is left standing, stated exactly:**

| dataset, content-controlled | polar ERP displacement | OSLO tax | RAFT tax | winner |
| --- | --- | --- | --- | --- |
| replica360 (natively uniform) | **132 px** (4.1× reach) | **1.839** | 3.151 | OSLO, 1.71× |
| flowscape (Haar-uniformised) | **27.4 px** (inside reach) | 1.355 | **1.099** | RAFT, 1.23× |

One dataset, not two — plus a measured mechanism that says why the second does
not show it, and a threshold that predicts where it would. The claim's honest
form:

> The equal-area grid reduces the polar error tax against a perspective
> architecture **where polar ERP displacement exceeds that architecture's
> correlation reach** (~32 px for RAFT-large at 1/8 resolution, radius 4).
> Measured at 132 px: 1.84 vs 3.15. Measured at 27.4 px: no advantage, RAFT leads
> 1.10 vs 1.36. The advantage is conditional on a regime, and the condition is
> arithmetic.

That is narrower than "spherical geometry buys polar accuracy and uniformity" and
it is the version that survives its own controls. It is also more useful: it
tells a reader when to reach for this architecture and when not to.

**Also settled: PanoFlow degrades 313.9% under Haar, OSLO 29.3%, RAFT-large
6.3%.** The orientation-contingency finding (§16.37) stands at full strength on
the proper control, and it is now the strongest *new* claim this line produced.

**The one test that would upgrade the threshold from two dataset-level points to
a within-dataset crossover**: region × displacement-band cross-tabulation. The
harness computes region masks and band masks as separate selections
(`spherical_flow/metrics.py:166,177`), so "polar error restricted to nodes moving
more than 2.9°" cannot be read from any existing JSON. It is a contained change —
intersect the two mask families — and it would test the threshold *inside*
flowscape, where every dataset-level confound is held fixed by construction.
Pre-registered: OSLO's polar tax should cross below RAFT's in the bands above
~2.9° (32 px / 11.11 px per degree) and sit above it below that.

### 16.43 Literature check (2026-08-14) — four things we were treating as ours are already published

Run before committing to a from-scratch RAFT. Full register in
`docs/plans/LITERATURE_SCOPE.md`; partners and split comparability in
`docs/plans/COMPARISON_PARTNERS.md`. What it changed:

| we had been treating as ours | actually | consequence |
| --- | --- | --- |
| geodesic angular metric | **SEPE**, "the geodesic distance between endpoints on the unit sphere", is standard in 360-flow and PriOr-Flow reports it for every baseline | keep the metric, drop any novelty framing; ours is the area weighting + zero denominator |
| polar/equatorial stratification | PriOr-Flow Table 6 publishes exactly this on FlowScape | claim nothing |
| evaluating under random global rotation | established in panoramic vision — Sphere-Depth (2026, depth), Spherical-GOF (2026, reconstruction), SO3UFormer (2026, segmentation) | ours is the **application to flow** and the **in-domain/out-of-domain discriminator**, which Sphere-Depth explicitly does not analyse |
| matched-backbone comparison across panoramic representations | PriOr-Flow Table 5 — SphereNet / TanImg / MPF-net / SLOF / PanoFlow / PriOr, all RAFT, all from the same pre-training, baselines re-run by the authors | **do not build it ourselves**; the missing cell is an OSLO row |

**Sphere-Depth's headline is our §16.37 in a different task**, verbatim: "even
models explicitly designed to process spherical images exhibit substantial
performance degradation when variations in the camera pose are observed." Our
PanoFlow result must be written as a replication in a new task plus a new
discriminator, not as a discovery.

**And one row of prior art argues against the thesis' premise.** PriOr-Flow's
SphereNet+RAFT — the distortion-aware spherical-convolution family, the closest
published relative of what we build — is **the worst row in their table** (13.2
EPE on EFT vs PriOr's 3.30) under matched backbone and matched pre-training. It
concerns a different mechanism than ours (adapted kernels, not estimation grid)
and a different displacement regime, but related work must engage it.

**Documentation action taken the same day.** Sixteen documents carried
inverted-target FLOW360 conclusions with no void marker, including both
thesis-facing notes and the reader-facing `EXPLICACAO_TECNICA.md`. All sixteen
now carry a status banner pointing at `LITERATURE_SCOPE.md`, and the refuted
"ERP methods structurally cannot win the SO(3) protocol" sentence in
`THESIS_REGIME_ARGUMENT.md` §5 is struck through in place with the measurement
that killed it.

### 16.44 THE CROSSOVER IS MEASURED — within one dataset, one run, one checkpoint

`--motion-band-regions` (§16.42's outstanding item) implemented, validated in
Docker (`run_band_region_check.py`: synthetic closed-form, plus the streaming
`accumulate_maps` proved to agree with the one-shot `summarize_maps` to 1.2e-06
across 53 keys), and run on flowscape:test. Sanity: global band fracs sum to
1.00000 and every region-band cell nests inside its global band.

**UNROTATED flowscape:test, POLES, geodesic degrees.** ERP px = the band's mean
displacement times the area-weighted mean $\sec\varphi$ of 3.908, over 0.3516°/px.

| band | ERP px | zero | OSLO | RAFT-large | OSLO/RAFT | % nodes |
| --- | --- | --- | --- | --- | --- | --- |
| 0–0.5 | 4 | 0.322 | 0.287 | **0.106** | 2.70 | 0.04% |
| 0.5–1 | 7 | 0.616 | 0.545 | **0.122** | 4.45 | 0.03% |
| 1–2 | 21 | 1.849 | 0.884 | **0.282** | 3.14 | 0.02% |
| 2–4 | 35 | 3.191 | 0.961 | **0.791** | 1.22 | 0.37% |
| **4–8** | **74** | 6.689 | **2.028** | 2.264 | **0.90** | 5.55% |
| **8–16** | **109** | 9.799 | **4.007** | 4.874 | **0.82** | 3.62% |
| **16+** | **237** | 21.341 | **17.503** | 19.736 | **0.89** | 0.29% |

**THE EQUATOR CONTROL, same run, same checkpoint, same pairs:**

| band | ERP px | OSLO | RAFT-large | OSLO/RAFT |
| --- | --- | --- | --- | --- |
| 0–0.5 | 1 | 0.324 | **0.119** | 2.72 |
| 0.5–1 | 2 | 0.466 | **0.159** | 2.93 |
| 1–2 | 4 | 0.639 | **0.189** | 3.37 |
| 2–4 | 8 | 0.713 | **0.272** | 2.62 |
| 4–8 | 15 | 0.998 | **0.521** | 1.92 |
| 8–16 | 31 | 3.814 | **1.827** | 2.09 |
| 16+ | 365 | 121.353 | **121.187** | 1.00 |

**The crossover exists, and it is polar-specific.** At the poles the ratio runs
2.70 → 4.45 → 3.14 → 1.22 → **0.90 → 0.82 → 0.89**: RAFT-large leads below 4°,
OSLO leads above it, and the flip is monotone in displacement. **At the equator
OSLO never leads at any displacement** — the ratio narrows from 2.72 to 1.92 but
never crosses 1. Same run, same weights, same 1386 pairs: the only variables are
displacement and latitude.

This is what §16.38–16.42 could not do. Those compared replica360 against
flowscape, which changed dataset, content, domain **and OSLO checkpoint** at
once. Here the checkpoint is one file, the data is one split, and the contrast is
internal.

**The pre-registration was one band off, and that is the honest number.** §16.42
predicted the crossing inside the 2–4 band, from a 32 px reach. The 2–4 band
sits at 35 px and RAFT still leads there by 1.22×; the flip happens by 74 px.
So the threshold model predicts the crossover's *location* to within a factor of
about two, not exactly. It also does not fully explain latitude: the equator's
8–16 band is 31 px with a ratio of 2.09, while the poles' 2–4 band is 35 px with
a ratio of 1.22 — comparable pixels, very different outcomes. **ERP displacement
sets the scale; the anisotropic polar stretch is a second effect that pixel count
alone does not capture.** A mechanism with a measured caveat, not a law.

**Under Haar rotation the crossover disappears** — RAFT-large leads every polar
band (ratios 2.17 / 2.75 / 3.07 / 2.80 / 2.01 / 1.38 / 1.04). The cause is not
placement: it is that **OSLO pays 29.3% for the rotation and RAFT-large 6.3%**.
§16.42's attribution — "all of the raw flowscape margin was placement" — is
therefore **too strong and is corrected here**. Within displacement bands, where
placement is controlled by construction, a genuine polar advantage is present in
the unrotated data. What rotation removes is not a placement artefact but OSLO's
own robustness deficit.

**Scope note found by the sanity check.** Unrotated, the polar cap carries only
74% of its geometric node mass after validity masking (0.0993 vs 0.1335) — CARLA
puts the ego vehicle at the bottom pole. Rotated, it recovers to 99% (0.1315).
So the unrotated polar rows describe the valid three-quarters of the polar cap,
and the rotation independently checks out as doing what it should.

**What this licenses in the article**, stated at exactly its width: on the
official FlowScape test split, in its native orientation, OSLO-RAFT has lower
polar error than frozen RAFT-large for displacements above roughly 4° (about 74
ERP px at the poles), covering 9.5% of the sphere's valid nodes, while losing at
every displacement at the equator and at every displacement once the scene is
randomly rotated.
