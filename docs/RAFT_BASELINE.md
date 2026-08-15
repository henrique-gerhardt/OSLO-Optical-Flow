# ERP RAFT Baseline

<!-- SCOPE-BANNER -->
> **STATUS 2026-08-14 — read `docs/plans/LITERATURE_SCOPE.md` before quoting anything here.**
> Two independent corrections apply across this project's documents. (1) The FLOW360
> forward-flow convention was inverted, so every FLOW360 result recorded before
> 2026-08-04 is void. (2) A literature check on 2026-08-14 found that several things
> treated here as ours already exist in print: the geodesic (SEPE) metric, polar/equatorial
> stratification, rotation-robustness evaluation in panoramic vision, and matched-backbone
> comparison across panoramic representations. `LITERATURE_SCOPE.md` is the register of what
> we may and may not claim.
>
> **This file specifically:** Every FLOW360 number here predates the convention fix and is VOID.


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
transform:   negated
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

| Metric | Zero-Flow R6 Forward | OSLO R6 Disp 1-Hop, 3 Seeds | RAFT ERP R6 Forward, Negated |
| --- | ---: | ---: | ---: |
| `global_geo_deg` | 0.4513 | mean 0.4423, range 0.4333-0.4508 | 0.2698 |
| `poles_geo_deg` | 0.4831 | mean 0.4569, range 0.4514-0.4650 | 0.3420 |
| `equator_geo_deg` | 0.4263 | mean 0.4253, range 0.4167-0.4298 | 0.2379 |
| `seam_geo_deg` | 1.0793 | mean 1.0859, range 1.0750-1.0919 | 0.8537 |
| `active_0_5_geo_deg` | 1.8704 | mean 1.7696, range 1.7457-1.7932 | 1.0404 |
| `active_1_0_geo_deg` | 4.3036 | mean 4.1997, range 4.1811-4.2200 | 2.6364 |

The decision point is not only global error. RAFT should be compared against OSLO on seam and pole behavior as well as active-motion subsets.

## First RAFT Result, Identity Sign

The first pretrained run used the raw TorchVision sign:

```text
model:       raft_large
weights:     C_T_SKHT_V2
torchvision: 0.20.1+cu124
split:       test
direction:   forward
resolution:  6
pairs:       1289
transform:   identity
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

Interpretation: this was not a valid baseline because the prediction sign was opposite to the FLOW360 target convention.

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

If identity is the best raw variant and the best fitted scale is positive, the runner's sign and axis convention are likely correct. If a negated or swapped variant wins clearly, rerun the baseline with `RAFT_FLOW_TRANSFORM=<variant>` before interpreting the RAFT baseline. If the best scaled variant improves dramatically, the issue is mostly magnitude calibration. If no variant beats zero-flow, the direct ERP RAFT baseline itself is the problem.

The first 8-pair diagnostic found:

```text
zero-flow pixel EPE: 8.5314 px
identity:            9.3347 px (-9.42%)
negated:             7.9665 px (+6.62%)
identity scaled:     8.4738 px (+0.67%), scale=-2.9225
```

This means the first full RAFT run used the wrong sign relative to the FLOW360 targets. The next required run is the full spherical evaluation with negated RAFT flow:

```bash
OUTPUT_DIR=/outputs/raft_r6_forward_negated \
  bash scripts/flow360_raft_baseline.sh
```

Interpret that run, not the original identity run, as the direct ERP RAFT baseline.

## Corrected RAFT Result

The corrected full run used:

```text
model:       raft_large
weights:     C_T_SKHT_V2
torchvision: 0.20.1+cu124
split:       test
direction:   forward
resolution:  6
pairs:       1289
transform:   negated
elapsed:     184.6 s
```

Result:

```text
global:        RAFT 0.2698 deg vs zero-flow 0.4513 deg (+40.21%)
poles:         RAFT 0.3420 deg vs zero-flow 0.4831 deg (+29.22%)
equator:       RAFT 0.2379 deg vs zero-flow 0.4263 deg (+44.19%)
seam:          RAFT 0.8537 deg vs zero-flow 1.0793 deg (+20.90%)
active >=0.25: RAFT 0.6124 deg vs zero-flow 1.1425 deg (+46.40%)
active >=0.5:  RAFT 1.0404 deg vs zero-flow 1.8704 deg (+44.38%)
active >=1.0:  RAFT 2.6364 deg vs zero-flow 4.3036 deg (+38.74%)
```

Decision:

- Direct pretrained ERP RAFT is a strong baseline once its sign is matched to FLOW360.
- The current OSLO-style MVP is not competitive with RAFT on global, poles, equator, seam, or active-motion subsets.
- The next OSLO experiment should not be justified by beating zero-flow; it must close the gap against this corrected RAFT baseline.
- The FLOW360 script now defaults to `RAFT_FLOW_TRANSFORM=negated`. Override it only for diagnostics or for datasets with a different convention.

The next experiment is documented in [RAFT_RESIDUAL_EXPERIMENT.md](RAFT_RESIDUAL_EXPERIMENT.md). It freezes this corrected RAFT baseline, caches it as HEALPix tangent flow, and trains an OSLO residual correction head initialized to reproduce RAFT exactly.
