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
