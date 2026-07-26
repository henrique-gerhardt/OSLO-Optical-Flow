# Universality table: published 360° methods vs zero-flow on the real-video leg

Workstream opened 2026-07-24 (see `docs/THESIS_REGIME_ARGUMENT.md` §5 for why this
table is load-bearing: the claim "the real-video regime is unsolved by anyone"
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
nothing on the real-video actives. Three architectures, three labs (Princeton-
lineage SLOF, cross-strip-correlation PanoFlow, native-spherical OSLO): all
negative.

**Domain status of the PanoFlow row — PINNED 2026-07-25 (naming collision
resolved).** PanoFlow's README calls *their own CARLA-rendered dataset*
"**FlowScape (Flow360)**" (8 city maps × 4 weathers, 1024×512, 6400 frames);
their `--validation Flow360` flag points at **that**, not at SLOF's real-video
FLOW360. Two different datasets, confusingly similar names:
- sfprep `flowscape` = PanoFlow's FlowScape/Flow360 (CARLA, large motion)
- sfprep `flow360`  = SLOF's FLOW360 (real video, sub-pixel)

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
on val), not OSLO.** The "real-video regime is unsolved by anyone" claim holds
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
  actually resolves, unlike the sub-pixel real-video leg.

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

| checkpoint | flowscape:test (large motion) | flow360:test (real video) | swing |
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
