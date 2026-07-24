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
