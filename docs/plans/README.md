# Phase 2: from bounded thesis result to benchmark-grade paper

**Status: planning (2026-07-09).** Phase 1 (thesis campaign) concluded 2026-07-07 with
DECISION = WRITE UP: Gate R1 passed (+88.4% real replica360; three-seed 1.74° ± 0.25),
Gate R2 failed with the wall precisely bounded (real inter-frame appearance change at
sub-pixel motion — the decisive triangle, `THESIS_CH4_DRAFT.md` §7). Phase 2 is the
follow-up campaign whose goal is a *paper-grade positive result*: OSLO-RAFT measured in
the units the field accepts (ERP-pixel EPE), trained with the recipe that made RAFT
robust, on data that gives spherical matching the same bootstrap Chairs/Things gave
planar matching.

## The three plans, in dependency order

| plan | deliverable | unblocks | est. cost |
| --- | --- | --- | --- |
| [P2A — ERP EPE readout](P2A_EPE_READOUT.md) | HEALPix→ERP flow readout + EPE metric + grid-floor measurement | benchmark-unit reporting for *everything downstream*; EPE column in thesis §5.1 | ~1 session impl + 30-s evals |
| [P2B — FlyingChairs-360 generator](P2B_FLYINGCHAIRS360.md) | `chairs360:{train,val}` shards (~22k pairs, exact GT, layered occlusion) | the matching bootstrap Stage P1 trains on | ~1 week impl; generation is CPU-bound, hours |
| [P2C — RAFT-style training campaign](P2C_TRAINING_CAMPAIGN.md) | photometric/eraser augmentation + staged 100k-step schedule + gates | the paper's headline numbers | ~1 session impl + ~2–4 GPU-days |

P2A has no dependencies and its output (the grid floor) is itself a design input for
P2C (decides whether supervision moves to r7 or a learned upsampler is needed). P2B
and P2C-implementation can proceed in parallel with P2A; P2C-training needs both.

## Current plan (supersedes the table above for scheduling)

| plan | deliverable | est. cost |
| --- | --- | --- |
| [Universality table](UNIVERSALITY_TABLE.md) | published 360° methods vs zero-flow, one harness — **COMPLETE** | done |
| [Roadmap — seminário de andamento](ROADMAP_SEMINARIO.md) | what to run *before writing* (A1–A4) vs what to *propose* (B1–B4) | ~8–11 h GPU for the blocking set |

The P2A/P2B/P2C framing above is historical: P2A concluded, P2B shipped chairs360,
P2C's P1 campaign closed 2026-07-24 with Gate R2 approached-not-met. The grid
hypothesis it was built around was later **refuted by measurement** (OSLO sits
2.2–4.9× above its own r4 floor), so resolution is not the lever. Read
`ROADMAP_SEMINARIO.md` for what is actually next.

## Why this is the right attack (one paragraph)

The triangle isolated the wall exactly: the *same* checkpoint scores +80.6% (0.046°
residual) when frame 2 is photometrically identical and −32% when it is the real next
frame. RAFT's robustness to that gap is not architectural — it was *manufactured* by
asymmetric photometric augmentation + occlusion erasing over ~350k iterations on
Chairs/Things. Phase 1 never trained with any of it. P2B recreates the data, P2C
recreates the recipe, and P2A makes the result legible to the field. The measured
throughput (~1 step/s on visco3, Stage A) prices a 100k-step stage at ~28 h — the full
RAFT-scale schedule is days, not weeks.

## Standing conventions (inherited from Phase 1)

- All local validation in Docker (`oslo-raft:cuda` via plain `docker run` on the Mac;
  compose on the box). No local pip installs. CPU gates before any GPU spend.
- Metrics through `spherical_flow.metrics` with the fp-robust region masks.
- `--ablate-corr` / `--ablate-context` must keep working on every new variant.
- Seeds 7/11/19 for anything that becomes a headline number.
- Shards are the only data interface: new data lands as a sphereflow-dataprep adapter,
  never as a bespoke loader in this repo.
