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
