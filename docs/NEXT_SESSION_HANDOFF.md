# Next Session Handoff

## Current Goal

The project is no longer trying to make the standalone OSLO MVP beat zero-flow. The current research direction is:

```text
strong ERP RAFT baseline
  -> convert/cache RAFT flow as HEALPix tangent flow
  -> train a small OSLO/HEALPix residual corrector
  -> evaluate final flow = RAFT + residual with spherical metrics
```

The current thesis-level question is whether OSLO-style spherical convolutions add useful 360-degree correction on top of a strong planar optical-flow model.

## Current Main Result

The best residual configuration is:

```text
RESOLUTION=6
RESIDUAL_MAX_RAD=0.01
RESIDUAL_REG_WEIGHT=0.05
POLE_RESIDUAL_REG_WEIGHT=0.05
hidden_channels=48
steps=3000
batch_size=1
```

Correct RAFT cache transforms:

```text
forward cache:  RAFT_FLOW_TRANSFORM=negated
backward cache: RAFT_FLOW_TRANSFORM=identity
```

Forward-only validation passed over seeds `7`, `11`, and `19`:

| metric | RAFT | residual mean | std | mean improvement |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.2699 | 0.2669 | 0.0002 | +1.11% |
| poles_geo_deg | 0.3420 | 0.3396 | 0.0007 | +0.72% |
| equator_geo_deg | 0.2380 | 0.2364 | 0.0001 | +0.70% |
| seam_geo_deg | 0.8488 | 0.8466 | 0.0001 | +0.26% |
| active_0_25_geo_deg | 0.6125 | 0.6071 | 0.0003 | +0.89% |
| active_0_5_geo_deg | 1.0404 | 1.0368 | 0.0009 | +0.35% |
| active_1_0_geo_deg | 2.6365 | 2.6342 | 0.0009 | +0.08% |

Mixed-transform `direction=both` validation also passed over seeds `7`, `11`, and `19`:

| metric | RAFT | residual mean | std | mean improvement |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.2676 | 0.2646 | 0.0002 | +1.14% |
| poles_geo_deg | 0.3393 | 0.3356 | 0.0009 | +1.08% |
| equator_geo_deg | 0.2363 | 0.2348 | 0.0002 | +0.64% |
| seam_geo_deg | 0.7104 | 0.7084 | 0.0002 | +0.28% |
| active_0_25_geo_deg | 0.6010 | 0.5969 | 0.0007 | +0.69% |
| active_0_5_geo_deg | 1.0160 | 1.0133 | 0.0006 | +0.26% |
| active_1_0_geo_deg | 2.5512 | 2.5495 | 0.0004 | +0.07% |

Interpretation: gains are modest but consistent. This is a defensible positive result because it improves a strong corrected RAFT baseline, not just zero-flow.

## Important Files

```text
README.md                         Short project status
docs/CONTEXT_AND_STATUS.md        Full chronological status
docs/RAFT_BASELINE.md             RAFT baseline convention and commands
docs/RAFT_RESIDUAL_EXPERIMENT.md  Residual experiment protocol and results
run_erp_raft_baseline.py          RAFT ERP baseline runner
run_flow360_cache_raft.py         RAFT-to-HEALPix cache generator
run_flow360_raft_residual.py      RAFT-conditioned residual trainer/evaluator
spherical_flow/metrics.py         Shared spherical metrics
spherical_flow/raft_adapter.py    RAFT transform/cache utilities
spherical_flow/models.py          RaftResidualCorrector
```

## Reproducible Cache Commands

Forward cache:

```bash
SPLIT=train DIRECTION=forward RESOLUTION=6 RAFT_FLOW_TRANSFORM=negated \
  bash scripts/flow360_cache_raft_r6.sh

SPLIT=test DIRECTION=forward RESOLUTION=6 RAFT_FLOW_TRANSFORM=negated \
  bash scripts/flow360_cache_raft_r6.sh
```

Backward cache:

```bash
SPLIT=train DIRECTION=backward RESOLUTION=6 RAFT_FLOW_TRANSFORM=identity \
  bash scripts/flow360_cache_raft_r6.sh

SPLIT=test DIRECTION=backward RESOLUTION=6 RAFT_FLOW_TRANSFORM=identity \
  bash scripts/flow360_cache_raft_r6.sh
```

Use `OVERWRITE=1` only when intentionally replacing old cache files.

## Reproducible Residual Commands

Forward-only:

```bash
DIRECTION=forward RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=7 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_seed7 \
  bash scripts/flow360_raft_residual_r6.sh
```

Bidirectional:

```bash
DIRECTION=both RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=7 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_both_mixed_transforms_seed7 \
  bash scripts/flow360_raft_residual_r6.sh
```

For robustness, repeat with `SEED=11` and `SEED=19`.

## Recommended Next Steps

1. Add a result aggregation script.
   - Input: one or more `raft_residual_metrics.json` files.
   - Output: CSV/Markdown table with mean, std, and improvement vs RAFT.
   - Reason: current result summaries were hand-calculated; this should be automated before more experiments.

2. Add direction-split evaluation for `direction=both`.
   - Current `both` metrics are aggregate over forward and backward pairs.
   - Add separate `forward_*` and `backward_*` metric groups or an evaluator option.
   - Reason: confirms the residual is not improving one direction while hiding damage in the other.

3. Add qualitative visualization/export.
   - Error maps for RAFT vs residual.
   - Seam/pole-focused plots.
   - Optional saved HEALPix residual magnitude and direction maps.
   - Reason: small numeric gains need visual explanation for a report or paper.

4. Try a RAFT-error-aware residual objective.
   - Weight loss more where cached RAFT is wrong, while preserving pole/global regularization.
   - Candidate inputs: RAFT tangent error proxy, endpoint consistency, warped color residual, or active-motion weighting.
   - Reason: current residual is intentionally conservative; this is the next plausible path to larger gains.

5. Only after the above, consider larger architecture changes.
   - Full spherical recurrent RAFT update.
   - Multi-scale/coarse-to-fine HEALPix features.
   - RAFT feature distillation into spherical features.

## Prompt For A New Codex Session

Use this prompt at the start of a new session:

```text
We are working in C:\Users\henrique.gerhardt\Developer\Mestrado\OSLO\OSLO-Optical-Flow.

Please read docs/NEXT_SESSION_HANDOFF.md, docs/RAFT_RESIDUAL_EXPERIMENT.md, docs/CONTEXT_AND_STATUS.md, and README.md first. Continue from the current OSLO optical-flow project state.

Current validated direction:
- Frozen TorchVision RAFT ERP baseline cached as HEALPix tangent flow.
- FLOW360 forward cache uses RAFT_FLOW_TRANSFORM=negated.
- FLOW360 backward cache uses RAFT_FLOW_TRANSFORM=identity.
- Best residual config is RESIDUAL_MAX_RAD=0.01, RESIDUAL_REG_WEIGHT=0.05, POLE_RESIDUAL_REG_WEIGHT=0.05.
- Forward-only and direction=both residual results are robust across seeds 7, 11, 19.

Do not restart the project or go back to standalone OSLO tuning. The next useful task is to automate result aggregation, add direction-split evaluation for direction=both, or add visualization/error-map tooling. Before editing, inspect the existing code and preserve the current experiment conventions.
```
