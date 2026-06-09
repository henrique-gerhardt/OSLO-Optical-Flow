# RAFT-Conditioned HEALPix Residual Experiment

## Motivation

The corrected ERP RAFT baseline is much stronger than the standalone OSLO MVP on FLOW360. The next experiment therefore tests a narrower and more useful question:

Can OSLO/HEALPix spherical processing improve a frozen RAFT prediction, especially around seam and polar regions?

The residual model starts exactly at RAFT:

```text
RAFT ERP flow, negated
  -> sample and convert to HEALPix tangent flow
  -> OSLO residual head predicts delta flow
  -> final flow = RAFT flow + delta
```

The residual head is zero-initialized and residuals are clamped by `residual_max_rad`, so the initial prediction is the RAFT baseline.

## Cache RAFT

Cache RAFT predictions as HEALPix tangent flow. Cache both train and test splits before training:

```bash
SPLIT=train RESOLUTION=6 DIRECTION=forward bash scripts/flow360_cache_raft_r6.sh
SPLIT=test RESOLUTION=6 DIRECTION=forward bash scripts/flow360_cache_raft_r6.sh
```

Default cache root:

```text
/outputs/raft_cache/<split>/<sequence>/<direction>/<frame>_r6.npz
```

Each cache file stores:

```text
flow_tangent: [N, 2] float32
model, weights, flow_transform, resolution, image size, sequence, direction, frame
```

FLOW360 forward flow uses `RAFT_FLOW_TRANSFORM=negated` by default.

## Train Residual

Smoke test:

```bash
RESOLUTION=4 MAX_TRAIN_PAIRS=2 MAX_VAL_PAIRS=1 STEPS=2 \
  bash scripts/flow360_raft_residual_r6.sh
```

Full first run:

```bash
RESOLUTION=6 DIRECTION=forward OUTPUT_DIR=/outputs/raft_residual_r6_forward \
  bash scripts/flow360_raft_residual_r6.sh
```

The runner writes:

```text
/outputs/raft_residual_r6_forward/raft_residual.pt
/outputs/raft_residual_r6_forward/raft_residual_metrics.json
```

The metrics JSON contains:

```text
raft_metrics
residual_metrics
vs_raft
```

## Acceptance Criteria

The current corrected RAFT baseline is:

```text
global_geo_deg:       0.2698
poles_geo_deg:        0.3420
equator_geo_deg:      0.2379
seam_geo_deg:         0.8537
active_0_5_geo_deg:   1.0404
active_1_0_geo_deg:   2.6364
```

Primary success:

```text
seam_geo_deg < 0.8537
global_geo_deg <= 0.2698
```

Secondary success:

```text
poles_geo_deg < 0.3420
or active_0_5_geo_deg < 1.0404
while global_geo_deg is no worse than 1% relative to RAFT
```

If the residual model does not beat RAFT, do not return to standalone OSLO MVP tuning. The next step should be a stronger RAFT-like HEALPix update mechanism or a teacher-student setup that uses RAFT features/cost volumes more directly.

## First Result

First full run:

```text
resolution:          6
steps:              3000
hidden_channels:    48
residual_max_rad:   0.05
residual_reg_weight: 0.01
seed:               7
elapsed:            127.0 s
```

The cached RAFT baseline inside this run was:

```text
global_geo_deg:       0.2699
poles_geo_deg:        0.3420
equator_geo_deg:      0.2380
seam_geo_deg:         0.8488
active_0_25_geo_deg:  0.6125
active_0_5_geo_deg:   1.0404
active_1_0_geo_deg:   2.6365
```

Residual result:

```text
global_geo_deg:       0.2705  (-0.20% vs RAFT)
poles_geo_deg:        0.3605  (-5.40% vs RAFT)
equator_geo_deg:      0.2367  (+0.54% vs RAFT)
seam_geo_deg:         0.8466  (+0.25% vs RAFT)
active_0_25_geo_deg:  0.6049  (+1.23% vs RAFT)
active_0_5_geo_deg:   1.0362  (+0.41% vs RAFT)
active_1_0_geo_deg:   2.6413  (-0.18% vs RAFT)
```

Decision:

- This is a secondary success, not a primary success.
- The residual model found a real correction signal for seam, equator, and moderate active motion.
- It failed the primary criterion because global error worsened slightly and pole error worsened materially.
- The model is too free to modify already-good RAFT regions.

Recommended next sweep:

```bash
RESIDUAL_MAX_RAD=0.02 RESIDUAL_REG_WEIGHT=0.05 \
  OUTPUT_DIR=/outputs/raft_residual_r6_constrained_002_005 \
  bash scripts/flow360_raft_residual_r6.sh

RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 \
  OUTPUT_DIR=/outputs/raft_residual_r6_constrained_001_005 \
  bash scripts/flow360_raft_residual_r6.sh

RESIDUAL_MAX_RAD=0.02 RESIDUAL_REG_WEIGHT=0.10 \
  OUTPUT_DIR=/outputs/raft_residual_r6_constrained_002_010 \
  bash scripts/flow360_raft_residual_r6.sh
```

Continue with the best constrained setting only if it preserves `global_geo_deg <= raft_global_geo_deg` and does not worsen poles by more than 1%.

## Constrained Sweep Result

The constrained sweep was run with the same cached RAFT baseline:

```text
RAFT global_geo_deg:       0.2699
RAFT poles_geo_deg:        0.3420
RAFT equator_geo_deg:      0.2380
RAFT seam_geo_deg:         0.8488
RAFT active_0_25_geo_deg:  0.6125
RAFT active_0_5_geo_deg:   1.0404
RAFT active_1_0_geo_deg:   2.6365
```

Results:

| residual_max_rad | residual_reg_weight | global vs RAFT | seam vs RAFT | poles vs RAFT | active 0.25 | active 0.5 | active 1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.05 | +0.01% | +0.24% | -3.97% | +1.11% | +0.39% | -0.03% |
| 0.01 | 0.05 | +0.51% | +0.24% | -1.49% | +1.09% | +0.40% | +0.05% |
| 0.02 | 0.10 | +0.30% | +0.22% | -2.53% | +1.09% | +0.38% | +0.02% |

Best current setting:

```text
RESIDUAL_MAX_RAD=0.01
RESIDUAL_REG_WEIGHT=0.05

global_geo_deg:       0.2686  (+0.51% vs RAFT)
poles_geo_deg:        0.3471  (-1.49% vs RAFT)
equator_geo_deg:      0.2367  (+0.58% vs RAFT)
seam_geo_deg:         0.8467  (+0.24% vs RAFT)
active_0_25_geo_deg:  0.6058  (+1.09% vs RAFT)
active_0_5_geo_deg:   1.0363  (+0.40% vs RAFT)
active_1_0_geo_deg:   2.6351  (+0.05% vs RAFT)
```

Decision:

- The constrained residual is now a real improvement over RAFT globally, on seam, on equator, and on all active-motion subsets.
- The result still misses the clean acceptance bar because poles regress by about 1.5%.
- The next experiment should keep the best constrained setting and add pole-protected residual regularization, instead of increasing residual capacity.

Run the next sweep with:

```bash
RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005 \
  bash scripts/flow360_raft_residual_r6.sh

RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.10 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_010 \
  bash scripts/flow360_raft_residual_r6.sh
```

Acceptance for the next run:

```text
global_geo_deg < 0.2699
seam_geo_deg < 0.8488
poles_geo_deg <= 0.3455
active_0_5_geo_deg < 1.0404
```

## Pole-Protected Sweep Result

The pole-protected residual sweep fixed the previous pole regression. Both runs passed the acceptance gate against the same cached RAFT baseline.

Results:

| pole_residual_reg_weight | global vs RAFT | seam vs RAFT | poles vs RAFT | equator vs RAFT | active 0.25 | active 0.5 | active 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | +1.10% | +0.25% | +0.86% | +0.67% | +0.95% | +0.45% | +0.12% |
| 0.10 | +0.95% | +0.29% | +0.44% | +0.70% | +0.88% | +0.48% | +0.14% |

Best balanced candidate:

```text
RESIDUAL_MAX_RAD=0.01
RESIDUAL_REG_WEIGHT=0.05
POLE_RESIDUAL_REG_WEIGHT=0.05

global_geo_deg:       0.2670  (+1.10% vs RAFT)
poles_geo_deg:        0.3391  (+0.86% vs RAFT)
equator_geo_deg:      0.2364  (+0.67% vs RAFT)
seam_geo_deg:         0.8466  (+0.25% vs RAFT)
active_0_25_geo_deg:  0.6067  (+0.95% vs RAFT)
active_0_5_geo_deg:   1.0358  (+0.45% vs RAFT)
active_1_0_geo_deg:   2.6333  (+0.12% vs RAFT)
```

The `0.10` pole weight gives slightly better seam, equator, and higher-motion active subsets, but `0.05` is the better balanced result because it gives better global and pole metrics.

Decision:

- This is the first clean positive residual result over corrected RAFT.
- The current best candidate is `0.01 / 0.05 / 0.05`.
- The next step is not another architectural change. Validate robustness over seeds and keep the same acceptance gate.

Run robustness validation:

```bash
RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=11 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_seed11 \
  bash scripts/flow360_raft_residual_r6.sh

RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=19 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_seed19 \
  bash scripts/flow360_raft_residual_r6.sh
```

Robustness acceptance:

```text
mean global_geo_deg < 0.2699
mean seam_geo_deg < 0.8488
mean poles_geo_deg < 0.3420
at least 2 of 3 seeds improve global, seam, poles, and active_0_5 together
```

## Robustness Result

The robustness validation passed. Seeds `7`, `11`, and `19` all improve corrected RAFT on global, poles, equator, seam, and active-motion subsets.

Per-seed residual metrics:

| seed | global | poles | equator | seam | active 0.25 | active 0.5 | active 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.2670 | 0.3391 | 0.2364 | 0.8466 | 0.6067 | 1.0358 | 2.6333 |
| 11 | 0.2672 | 0.3405 | 0.2365 | 0.8467 | 0.6071 | 1.0379 | 2.6354 |
| 19 | 0.2667 | 0.3391 | 0.2362 | 0.8464 | 0.6075 | 1.0368 | 2.6340 |

Three-seed summary against corrected RAFT:

| metric | RAFT | residual mean | std | mean improvement |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.2699 | 0.2669 | 0.0002 | +1.11% |
| poles_geo_deg | 0.3420 | 0.3396 | 0.0007 | +0.72% |
| equator_geo_deg | 0.2380 | 0.2364 | 0.0001 | +0.70% |
| seam_geo_deg | 0.8488 | 0.8466 | 0.0001 | +0.26% |
| active_0_25_geo_deg | 0.6125 | 0.6071 | 0.0003 | +0.89% |
| active_0_5_geo_deg | 1.0404 | 1.0368 | 0.0009 | +0.35% |
| active_1_0_geo_deg | 2.6365 | 2.6342 | 0.0009 | +0.08% |

Decision:

- This is now the main residual result: it is positive, reproducible across seeds, and measured against the corrected RAFT baseline.
- The gains are modest but consistent, which is a better signal than the standalone OSLO MVP.
- The next experiment should either strengthen the residual target with RAFT-error-aware training or test whether the same correction holds for `direction=both`; do not tune more scalar regularizers until this result is written up.

## Direction Both Attempt

The first `direction=both` run used the forward-validated `RAFT_FLOW_TRANSFORM=negated` for both forward and backward cache files:

```text
RESIDUAL_MAX_RAD=0.01
RESIDUAL_REG_WEIGHT=0.05
POLE_RESIDUAL_REG_WEIGHT=0.05
DIRECTION=both
seed=7
```

Result:

| metric | zero-flow | RAFT both | residual both | residual vs RAFT |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.4368 | 0.4836 | 0.4762 | +1.52% |
| poles_geo_deg | 0.4644 | 0.5296 | 0.5198 | +1.86% |
| equator_geo_deg | 0.4131 | 0.4477 | 0.4436 | +0.92% |
| seam_geo_deg | 0.9228 | 0.9872 | 0.9854 | +0.19% |
| active_0_25_geo_deg | 1.1006 | 1.1377 | 1.1148 | +2.01% |
| active_0_5_geo_deg | 1.7900 | 1.8354 | 1.8099 | +1.39% |
| active_1_0_geo_deg | 4.0783 | 4.1418 | 4.1128 | +0.70% |

Decision:

- The residual still improves RAFT on every tracked metric.
- This is not a valid `both` confirmation yet because the RAFT `both` baseline is worse than zero-flow.
- Since forward-only RAFT is strong, the likely issue is the backward FLOW360 convention or backward cache transform, not the residual model.
- Do not run more `direction=both` residual training until the backward RAFT transform is diagnosed.

Backward transform diagnostic:

```bash
SPLIT=test DIRECTION=backward RESOLUTION=6 MAX_PAIRS=64 SAVE_PREDICTIONS=1 \
  RAFT_FLOW_TRANSFORM=identity OUTPUT_DIR=/outputs/raft_r6_backward_raw_debug \
  bash scripts/flow360_raft_baseline.sh

SPLIT=test DIRECTION=backward MAX_PAIRS=64 \
  OUTPUT_DIR=/outputs/raft_r6_backward_raw_debug \
  bash scripts/flow360_raft_prediction_diagnostic.sh
```

Diagnostic result on 64 backward test pairs:

```text
zero_flow mean_epe_px: 4.4501
best_raw transform:    identity
best_raw mean_epe_px:  4.0139
best_raw improvement:  +9.80%
best_raw cosine:       0.4116
```

The correct backward transform is therefore `identity` for the current FLOW360 convention. Validate it on the full test split:

```bash
SPLIT=test DIRECTION=backward RESOLUTION=6 RAFT_FLOW_TRANSFORM=identity \
  OUTPUT_DIR=/outputs/raft_r6_backward_identity \
  bash scripts/flow360_raft_baseline.sh
```

If full backward RAFT still beats zero-flow, regenerate only the backward cache, preserving the existing forward `negated` cache:

Full backward identity validation passed:

```text
global_geo_deg:       0.2652 vs zero 0.4221 (+37.17%)
poles_geo_deg:        0.3364 vs zero 0.4455 (+24.48%)
equator_geo_deg:      0.2345 vs zero 0.3998 (+41.34%)
seam_geo_deg:         0.5733 vs zero 0.7742 (+25.94%)
active_0_25_geo_deg:  0.5894 vs zero 1.0584 (+44.32%)
active_0_5_geo_deg:   0.9912 vs zero 1.7088 (+42.00%)
active_1_0_geo_deg:   2.4639 vs zero 3.8477 (+35.96%)
```

This confirms the mixed-transform cache convention:

```text
forward cache:  RAFT_FLOW_TRANSFORM=negated
backward cache: RAFT_FLOW_TRANSFORM=identity
```

Regenerate only the backward cache, preserving the existing forward `negated` cache:

```bash
SPLIT=train DIRECTION=backward RAFT_FLOW_TRANSFORM=identity OVERWRITE=1 \
  bash scripts/flow360_cache_raft_r6.sh

SPLIT=test DIRECTION=backward RAFT_FLOW_TRANSFORM=identity OVERWRITE=1 \
  bash scripts/flow360_cache_raft_r6.sh
```

Then rerun `DIRECTION=both` residual training:

```bash
DIRECTION=both RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_both_mixed_transforms_seed7 \
  bash scripts/flow360_raft_residual_r6.sh
```

## Direction Both Mixed-Transform Result

The corrected mixed-transform cache makes `direction=both` valid:

```text
forward cache:  RAFT_FLOW_TRANSFORM=negated
backward cache: RAFT_FLOW_TRANSFORM=identity
```

Seed-7 result:

| metric | zero-flow | mixed RAFT both | residual both | residual vs RAFT |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.4368 | 0.2676 | 0.2643 | +1.23% |
| poles_geo_deg | 0.4644 | 0.3393 | 0.3351 | +1.23% |
| equator_geo_deg | 0.4131 | 0.2363 | 0.2347 | +0.68% |
| seam_geo_deg | 0.9228 | 0.7104 | 0.7086 | +0.25% |
| active_0_25_geo_deg | 1.1006 | 0.6010 | 0.5959 | +0.86% |
| active_0_5_geo_deg | 1.7900 | 1.0160 | 1.0128 | +0.32% |
| active_1_0_geo_deg | 4.0783 | 2.5512 | 2.5491 | +0.08% |

Decision:

- The mixed-transform RAFT baseline beats zero-flow strongly in `direction=both`.
- The residual again improves RAFT on every tracked metric.
- This validates that the residual correction is not forward-only, but it should still be repeated on seeds `11` and `19` before reporting `both` as robust.

Run robustness validation:

```bash
DIRECTION=both RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=11 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_both_mixed_transforms_seed11 \
  bash scripts/flow360_raft_residual_r6.sh

DIRECTION=both RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 POLE_RESIDUAL_REG_WEIGHT=0.05 SEED=19 \
  OUTPUT_DIR=/outputs/raft_residual_r6_pole_001_005_005_both_mixed_transforms_seed19 \
  bash scripts/flow360_raft_residual_r6.sh
```

Robustness validation passed. Seeds `7`, `11`, and `19` all improve mixed-transform RAFT on global, poles, equator, seam, and active-motion subsets.

Per-seed residual metrics:

| seed | global | poles | equator | seam | active 0.25 | active 0.5 | active 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.2643 | 0.3351 | 0.2347 | 0.7086 | 0.5959 | 1.0128 | 2.5491 |
| 11 | 0.2649 | 0.3369 | 0.2347 | 0.7082 | 0.5973 | 1.0131 | 2.5495 |
| 19 | 0.2646 | 0.3348 | 0.2351 | 0.7083 | 0.5975 | 1.0142 | 2.5500 |

Three-seed summary against mixed-transform RAFT:

| metric | RAFT | residual mean | std | mean improvement |
| --- | ---: | ---: | ---: | ---: |
| global_geo_deg | 0.2676 | 0.2646 | 0.0002 | +1.14% |
| poles_geo_deg | 0.3393 | 0.3356 | 0.0009 | +1.08% |
| equator_geo_deg | 0.2363 | 0.2348 | 0.0002 | +0.64% |
| seam_geo_deg | 0.7104 | 0.7084 | 0.0002 | +0.28% |
| active_0_25_geo_deg | 0.6010 | 0.5969 | 0.0007 | +0.69% |
| active_0_5_geo_deg | 1.0160 | 1.0133 | 0.0006 | +0.26% |
| active_1_0_geo_deg | 2.5512 | 2.5495 | 0.0004 | +0.07% |

Decision:

- `direction=both` is now validated with the same residual setup.
- The mixed-transform RAFT baseline is strong, and the residual improvement is consistent across seeds.
- This is a publishable ablation alongside the forward-only result.
