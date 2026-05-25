# First FLOW360 Experiment

This document contains the commands to run the first reproducible FLOW360 experiment in Docker on the NVIDIA host.

## Inputs

Set these paths on the Linux host:

```bash
export FLOW360_ROOT=/absolute/path/to/FLOW360
export OSLO_DATA_ROOT=/absolute/path/to/oslo_data
export OUTPUT_DIR=$PWD/outputs
```

Expected files:

```text
$FLOW360_ROOT/train/<sequence>/frames/*.png
$FLOW360_ROOT/train/<sequence>/fflows/*.npy
$FLOW360_ROOT/train/<sequence>/bflows/*.npy
$FLOW360_ROOT/test/<sequence>/frames/*.png
$FLOW360_ROOT/test/<sequence>/fflows/*.npy
$FLOW360_ROOT/test/<sequence>/bflows/*.npy
$OSLO_DATA_ROOT/neighbor_grids/healpix_grid_resolution_5.npz
```

## Build Image

Run from the repository root:

```bash
docker build -f Dockerfile.flow360 -t oslo-flow360:cuda .
```

## Verify GPU In Container

```bash
docker run --rm --gpus all oslo-flow360:cuda nvidia-smi
```

Expected: the RTX 3090 appears inside the container.

## Smoke Test

Run this before training:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_smoke.sh
```

The smoke test uses a small subset and should print:

```text
train_dataset=...
val_dataset=...
smoke device=cuda ...
smoke_metrics global_geo_deg=... global_zero_geo_deg=...
```

If it fails before the model forward pass, the likely issue is one of:

- wrong mount path;
- missing `neighbor_grids`;
- unexpected `.npy` flow shape;
- flow values are normalized and need `--flow-scale`.

## First Training Run

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_r5.sh
```

Default configuration:

```text
HEALPix resolution: 5
nodes:              12,288
batch size:         1
steps:              2,000
learning rate:      1e-4
hidden channels:    48
feature channels:   32
max flow radius:    1.2 rad
AMP:                enabled
```

Outputs:

```text
$OUTPUT_DIR/flow360_mvp.pt
$OUTPUT_DIR/flow360_metrics.json
```

## Useful Overrides

Shorter debug run:

```bash
STEPS=200 LOG_EVERY=20 bash scripts/flow360_train_r5.sh
```

Run both forward and backward flows:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  python run_flow360_mvp.py \
    --data-root /data/flow360 \
    --grid-dir /data/oslo_data/neighbor_grids \
    --output-dir /outputs \
    --device cuda \
    --amp \
    --resolution 5 \
    --batch-size 1 \
    --steps 2000 \
    --direction both
```

Try HEALPix `r=6` after `r=5` works:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash -lc 'RESOLUTION=6 BATCH_SIZE=1 STEPS=1000 bash scripts/flow360_train_r5.sh'
```

Use a flow scale if the `.npy` values are normalized:

```bash
python run_flow360_mvp.py \
  --data-root /data/flow360 \
  --grid-dir /data/oslo_data/neighbor_grids \
  --output-dir /outputs \
  --device cuda \
  --amp \
  --resolution 5 \
  --flow-scale 1024
```

Choose `--flow-scale` based on the dataset's stored unit convention.

## Compose Alternative

```bash
docker compose -f docker-compose.flow360.yml up --build
```

The compose file uses:

```text
FLOW360_ROOT
OSLO_DATA_ROOT
OUTPUT_DIR
```

from the environment.

## What To Record

Save these after the first run:

```text
docker image tag or commit hash
full command
FLOW360 path/version
HEALPix resolution
steps
metrics JSON
GPU memory peak if available
```

Primary metrics to inspect:

```text
global_geo_deg vs global_zero_geo_deg
poles_geo_deg vs poles_zero_geo_deg
equator_geo_deg vs equator_zero_geo_deg
seam_geo_deg vs seam_zero_geo_deg
```

The first run is useful if the model beats zero-flow, especially at poles and seam.

## First Run Result

The first completed run did not beat zero-flow:

```text
global:  model 0.4666 deg vs zero-flow 0.4309 deg
poles:   model 0.5251 deg vs zero-flow 0.4684 deg
equator: model 0.4286 deg vs zero-flow 0.4053 deg
seam:    model 0.8864 deg vs zero-flow 0.8337 deg
```

This means the direct MVP is not yet useful on FLOW360 as a global model. The likely reason is that the average target motion is small, so zero-flow is already a strong baseline.

The runner now reports active-motion subsets and initializes the final flow head to zero by default. Rebuild the image before the next run.

## Active-Motion Training Run

```bash
docker build -f Dockerfile.flow360 -t oslo-flow360:cuda .

docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_active_r5.sh
```

This run uses:

```text
steps:                3,000
loss-motion-weight:   4.0
loss-motion-ref-deg:  1.0
active thresholds:    0.25, 0.5, 1.0 deg
```

Primary metrics for the next decision:

```text
target_geo_deg_p50 / p90 / p95
active_0_5_geo_deg vs active_0_5_zero_geo_deg
active_1_0_geo_deg vs active_1_0_zero_geo_deg
seam_geo_deg vs seam_zero_geo_deg
poles_geo_deg vs poles_zero_geo_deg
```

If the active-motion metrics still lose to zero-flow, do not extend training blindly. Move to a multi-hop or coarse-to-fine cost volume before attempting a spherical RAFT update block.

## Zero-Init Result

The zero-initialized run with the same unweighted loss produced a weak but real improvement. A repeat run with explicit improvement metrics produced:

```text
global:        model 0.4209 deg vs zero-flow 0.4309 deg (+2.32%)
poles:         model 0.4408 deg vs zero-flow 0.4684 deg (+5.89%)
equator:       model 0.4039 deg vs zero-flow 0.4053 deg (+0.35%)
seam:          model 0.8408 deg vs zero-flow 0.8337 deg (-0.85%)
active >=0.25: model 0.9943 deg vs zero-flow 1.0840 deg (+8.27%)
active >=0.5:  model 1.6533 deg vs zero-flow 1.7596 deg (+6.04%)
active >=1.0:  model 3.8801 deg vs zero-flow 3.9832 deg (+2.59%)
```

The target motion distribution confirms why zero-flow is hard to beat:

```text
p50: 0.1312 deg
p90: 0.7554 deg
p95: 1.0690 deg
active >=0.25 deg: 34.62%
active >=0.5 deg:  17.95%
active >=1.0 deg:   5.78%
```

Decision:

- The MVP has enough signal to justify one more architectural step.
- The zero-init result is repeatable; the small differences between runs are not changing the conclusion.
- The seam regression must be fixed before claiming spherical advantage.
- This repeat was still unweighted (`loss-motion-weight=0.0`), so the next run should use `scripts/flow360_train_active_r5.sh`; if it improves active metrics but worsens seam, move to multi-hop/coarse-to-fine instead of just increasing training time.

The runner now writes improvement metrics such as:

```text
global_improvement_deg / global_improvement_pct
active_0_5_improvement_deg / active_0_5_improvement_pct
seam_improvement_deg / seam_improvement_pct
```

## Motion-Weighted Result

The `scripts/flow360_train_active_r5.sh` run used:

```text
steps:                3,000
loss-motion-weight:   4.0
loss-motion-ref-deg:  1.0
```

It improved active-motion nodes more strongly:

```text
active >=0.25: model 0.9679 deg vs zero-flow 1.0840 deg (+10.71%)
active >=0.5:  model 1.5992 deg vs zero-flow 1.7596 deg (+9.11%)
active >=1.0:  model 3.8099 deg vs zero-flow 3.9832 deg (+4.35%)
```

But it regressed global/equator/seam:

```text
global:  model 0.4353 deg vs zero-flow 0.4309 deg (-1.02%)
equator: model 0.4140 deg vs zero-flow 0.4053 deg (-2.13%)
seam:    model 0.8604 deg vs zero-flow 0.8337 deg (-3.21%)
poles:   model 0.4599 deg vs zero-flow 0.4684 deg (+1.82%)
```

Decision:

- Motion weighting confirms that the model can learn non-zero motion where motion exists.
- The stronger seam regression means loss weighting alone is not the right next lever.
- Do not spend the next iteration on longer weighted training.
- Implement multi-hop or coarse-to-fine local cost volume next, then compare against both current baselines:
  - zero-init unweighted: best global/seam tradeoff so far;
  - motion-weighted: best active-motion performance so far.

## Multi-Hop Cost Volume Run

The first architectural step is implemented as `--cost-num-hops`. It expands the matching support used by the local cost volume while keeping the OSLO `SDPAConv` feature encoder on the original 1-hop graph.

For `r=5`:

```text
cost-num-hops=1 -> center + 8 neighbors  -> cost_shape [B, 12288, 9]
cost-num-hops=2 -> center + 24 neighbors -> cost_shape [B, 12288, 25]
```

Run the unweighted multi-hop test first:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_multihop_r5.sh
```

Then run the motion-weighted variant:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash -lc 'LOSS_MOTION_WEIGHT=4.0 OUTPUT_DIR=/outputs/r5_costh2_active bash scripts/flow360_train_multihop_r5.sh'
```

Compare against the previous bests:

```text
Unweighted 1-hop:
global +2.32%, seam -0.85%, active>=0.5 +6.04%

Motion-weighted 1-hop:
global -1.02%, seam -3.21%, active>=0.5 +9.11%
```

The multi-hop run is interesting only if it preserves or improves the active metrics without making the seam worse. If seam remains negative, the next step should be coarse-to-fine matching rather than more loss weighting.

## Multi-Hop Result

Two `cost-num-hops=2` runs were completed at `r=5`.

Unweighted 2-hop:

```text
global:        model 0.4234 deg vs zero-flow 0.4309 deg (+1.75%)
poles:         model 0.4410 deg vs zero-flow 0.4684 deg (+5.84%)
equator:       model 0.4101 deg vs zero-flow 0.4053 deg (-1.17%)
seam:          model 0.8536 deg vs zero-flow 0.8337 deg (-2.40%)
active >=0.25: model 0.9998 deg vs zero-flow 1.0840 deg (+7.77%)
active >=0.5:  model 1.6643 deg vs zero-flow 1.7596 deg (+5.41%)
active >=1.0:  model 3.9102 deg vs zero-flow 3.9832 deg (+1.83%)
```

Motion-weighted 2-hop:

```text
global:        model 0.4358 deg vs zero-flow 0.4309 deg (-1.15%)
poles:         model 0.4440 deg vs zero-flow 0.4684 deg (+5.21%)
equator:       model 0.4261 deg vs zero-flow 0.4053 deg (-5.13%)
seam:          model 0.8696 deg vs zero-flow 0.8337 deg (-4.31%)
active >=0.25: model 0.9937 deg vs zero-flow 1.0840 deg (+8.32%)
active >=0.5:  model 1.6394 deg vs zero-flow 1.7596 deg (+6.83%)
active >=1.0:  model 3.8557 deg vs zero-flow 3.9832 deg (+3.20%)
```

Comparison against the strongest 1-hop runs:

```text
1-hop unweighted: global +2.32%, seam -0.85%, active>=0.5 +6.04%
2-hop unweighted: global +1.75%, seam -2.40%, active>=0.5 +5.41%

1-hop weighted: global -1.02%, seam -3.21%, active>=0.5 +9.11%
2-hop weighted: global -1.15%, seam -4.31%, active>=0.5 +6.83%
```

Decision:

- 2-hop local search does not justify continuing to `cost-num-hops=3`.
- Wider raw correlation support adds noise without giving the decoder enough structure to choose a reliable displacement.
- The seam problem is worse, not better.
- Keep the best 1-hop unweighted run as the current global/seam baseline.
- Keep the 1-hop motion-weighted run as the current active-motion baseline.
- The next implementation should be coarse-to-fine or displacement-aware residual matching, not more loss weighting or a wider flat cost volume.

## Displacement-Aware Residual Matching

The next implemented variant keeps the 2-hop candidate set but makes the candidate geometry explicit.

For every node and every cost-volume candidate, the runner now computes the tangent offset from the source node to the candidate node. The model converts cost scores into a softmax probability distribution, builds a soft flow prior, then predicts:

```text
final_flow = residual + gate * flow_prior
```

The final head is still zero-initialized. In displacement-aware mode, the gate bias starts near zero contribution, so step 1 remains effectively close to zero-flow instead of immediately trusting the untrained cost prior.

Run the unweighted displacement-aware test:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_displacement_r5.sh
```

Expected smoke shape at `r=5`:

```text
cost_num_hops=2 displacement_prior=True cost_shape=(1, 12288, 25)
```

If the unweighted run improves seam or active-motion performance, run the weighted version:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash -lc 'LOSS_MOTION_WEIGHT=4.0 OUTPUT_DIR=/outputs/r5_disp_costh2_active bash scripts/flow360_train_displacement_r5.sh'
```

Decision rule:

- Continue if it beats the 1-hop unweighted seam result, `seam -0.85%`, while preserving positive active metrics.
- Prefer it over the 1-hop weighted run only if `active>=0.5` approaches or exceeds `+9.11%` without worsening seam.
- If it fails both criteria, move to coarse-to-fine matching.
