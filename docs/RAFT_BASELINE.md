# ERP RAFT Baseline

This baseline evaluates TorchVision RAFT directly on native FLOW360 equirectangular frames, then converts the predicted ERP pixel flow into the same spherical endpoint/tangent representation used by the OSLO-style runner.

It is evaluation-only. There is no RAFT fine-tuning in this step.

## Command

Build the CUDA image:

```bash
docker build -f Dockerfile.flow360 -t oslo-flow360:cuda .
```

Run RAFT Large with default TorchVision pretrained weights:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  -v "$MODEL_CACHE:/models" \
  oslo-flow360:cuda \
  bash scripts/flow360_raft_baseline.sh
```

The script defaults to:

```text
model:       raft_large
weights:     default
split:       test
direction:   forward
resolution:  6
batch size:  1
TORCH_HOME:  /models/torch
```

Use the cache mount because the first run downloads TorchVision RAFT weights. The runner writes:

```text
/outputs/raft_r6_forward/raft_metrics.json
```

By default, full ERP predictions are not saved. To save them:

```bash
SAVE_PREDICTIONS=1 bash scripts/flow360_raft_baseline.sh
```

Prediction files use:

```text
/outputs/raft_r6_forward/predictions/<split>/<sequence>/<direction>/<frame>.npy
```

Each prediction is `[H, W, 2]` ERP pixel displacement.

## Smoke Test

This avoids pretrained-weight download and limits evaluation to one pair:

```bash
RAFT_MODEL=raft_small RAFT_WEIGHTS=none RESOLUTION=4 MAX_PAIRS=1 \
  bash scripts/flow360_raft_baseline.sh
```

Expected outcome:

- the runner completes;
- `raft_metrics.json` is written;
- metric names match the OSLO runner, including global, poles, equator, seam, active-motion subsets, zero-flow baselines, improvements, and target percentiles.

## Comparison Table

Fill the RAFT column from `raft_metrics.json` after the first pretrained run.

| Metric | Zero-Flow R6 Forward | OSLO R6 Disp 1-Hop, 3 Seeds | RAFT ERP R6 Forward |
| --- | ---: | ---: | ---: |
| `global_geo_deg` | 0.4513 | mean 0.4423, range 0.4333-0.4508 | TBD |
| `poles_geo_deg` | 0.4831 | mean 0.4569, range 0.4514-0.4650 | TBD |
| `equator_geo_deg` | 0.4263 | mean 0.4253, range 0.4167-0.4298 | TBD |
| `seam_geo_deg` | 1.0735 | mean 1.0859, range 1.0750-1.0919 | TBD |
| `active_0_5_geo_deg` | 1.8704 | mean 1.7696, range 1.7457-1.7932 | TBD |
| `active_1_0_geo_deg` | 4.3036 | mean 4.1997, range 4.1811-4.2200 | TBD |

The decision point is not only global error. RAFT should be compared against OSLO on seam and pole behavior as well as active-motion subsets.
