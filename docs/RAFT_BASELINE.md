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

| Metric | Zero-Flow R6 Forward | OSLO R6 Disp 1-Hop, 3 Seeds | RAFT ERP R6 Forward |
| --- | ---: | ---: | ---: |
| `global_geo_deg` | 0.4513 | mean 0.4423, range 0.4333-0.4508 | 0.7361 |
| `poles_geo_deg` | 0.4831 | mean 0.4569, range 0.4514-0.4650 | 0.7901 |
| `equator_geo_deg` | 0.4263 | mean 0.4253, range 0.4167-0.4298 | 0.6892 |
| `seam_geo_deg` | 1.0735 | mean 1.0859, range 1.0750-1.0919 | 1.4348 |
| `active_0_5_geo_deg` | 1.8704 | mean 1.7696, range 1.7457-1.7932 | 2.8470 |
| `active_1_0_geo_deg` | 4.3036 | mean 4.1997, range 4.1811-4.2200 | 6.2544 |

The decision point is not only global error. RAFT should be compared against OSLO on seam and pole behavior as well as active-motion subsets.

## First RAFT Result

The first pretrained run used:

```text
model:       raft_large
weights:     C_T_SKHT_V2
torchvision: 0.20.1+cu124
split:       test
direction:   forward
resolution:  6
pairs:       1289
elapsed:     186.0 s
```

RAFT ERP underperformed zero-flow on every reported subset:

```text
global:        RAFT 0.7361 deg vs zero-flow 0.4513 deg (-63.12%)
poles:         RAFT 0.7901 deg vs zero-flow 0.4831 deg (-63.54%)
equator:       RAFT 0.6892 deg vs zero-flow 0.4263 deg (-61.65%)
seam:          RAFT 1.4348 deg vs zero-flow 1.0793 deg (-32.94%)
active >=0.25: RAFT 1.7804 deg vs zero-flow 1.1425 deg (-55.84%)
active >=0.5:  RAFT 2.8470 deg vs zero-flow 1.8704 deg (-52.21%)
active >=1.0:  RAFT 6.2544 deg vs zero-flow 4.3036 deg (-45.33%)
```

Interpretation:

- Direct ERP RAFT is not a competitive baseline on this FLOW360 setup.
- The failure is broad, not only a seam artifact.
- The pretrained RAFT output likely carries image-plane priors that do not match small-motion 360 ERP flow well enough without adaptation.
- Before concluding that RAFT is unusable for this dataset, run a small prediction diagnostic to verify flow direction, scale, and coordinate convention.

Recommended diagnostic:

```bash
MAX_PAIRS=8 SAVE_PREDICTIONS=1 OUTPUT_DIR=/outputs/raft_r6_forward_debug \
  bash scripts/flow360_raft_baseline.sh

MAX_PAIRS=8 OUTPUT_DIR=/outputs/raft_r6_forward_debug \
  bash scripts/flow360_raft_prediction_diagnostic.sh
```

The diagnostic writes:

```text
/outputs/raft_r6_forward_debug/raft_prediction_diagnostic.json
```

It compares the saved RAFT `.npy` files against the matching `fflows`/`bflows`, and reports:

- identity prediction error;
- sign-flipped and axis-swapped variants;
- best scalar fit for each variant;
- zero-flow pixel EPE for the same pairs.

If identity is the best raw variant and the best fitted scale is positive, the runner's sign and axis convention are likely correct. If a negated or swapped variant wins clearly, fix the runner before interpreting the RAFT baseline. If the best scaled variant improves dramatically, the issue is mostly magnitude calibration. If no variant beats zero-flow, the direct ERP RAFT baseline itself is the problem.
