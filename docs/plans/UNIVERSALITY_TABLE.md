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
| **OSLO EMA final** | −14.0 | **−4.9** | **−4.0** | **−5.7** | −24.8 |

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

1. **The wall is universal, not an OSLO artifact.** SLOF's best clean actives
   (singlerotation −3.2) sit right next to OSLO's (−4.0). Two independent
   architectures + training recipes land in the same negative band.
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

**Box protocol** (weights staged to `outputs/panoflow_weights/`): 2-pair smoke
(`--max-pairs 2`) to confirm the real ckpt strict-loads + CFE yields finite flow
and the printed zero-baseline matches the recorded test row (zero global
0.4368°, active fracs 34.6/18.0/5.8% ⇒ same split+resolution), then the full
flow360:test run → `universality_panoflow_csflow_test`. Row goes into §5.
