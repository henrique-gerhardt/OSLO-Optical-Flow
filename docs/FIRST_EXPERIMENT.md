# First FLOW360 Experiment

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

## Displacement-Aware Result

Two `cost-num-hops=2`, displacement-aware runs were completed.

Unweighted displacement-aware 2-hop:

```text
global:        model 0.4310 deg vs zero-flow 0.4309 deg (-0.02%)
poles:         model 0.4413 deg vs zero-flow 0.4684 deg (+5.78%)
equator:       model 0.4198 deg vs zero-flow 0.4053 deg (-3.57%)
seam:          model 0.8663 deg vs zero-flow 0.8337 deg (-3.91%)
active >=0.25: model 1.0031 deg vs zero-flow 1.0840 deg (+7.46%)
active >=0.5:  model 1.6612 deg vs zero-flow 1.7596 deg (+5.59%)
active >=1.0:  model 3.8903 deg vs zero-flow 3.9832 deg (+2.33%)
```

Motion-weighted displacement-aware 2-hop:

```text
global:        model 0.4328 deg vs zero-flow 0.4309 deg (-0.45%)
poles:         model 0.4483 deg vs zero-flow 0.4684 deg (+4.28%)
equator:       model 0.4163 deg vs zero-flow 0.4053 deg (-2.71%)
seam:          model 0.8755 deg vs zero-flow 0.8337 deg (-5.02%)
active >=0.25: model 0.9521 deg vs zero-flow 1.0840 deg (+12.16%)
active >=0.5:  model 1.5640 deg vs zero-flow 1.7596 deg (+11.11%)
active >=1.0:  model 3.7398 deg vs zero-flow 3.9832 deg (+6.11%)
```

Comparison against the strongest prior runs:

```text
1-hop unweighted:        global +2.32%, seam -0.85%, active>=0.5 +6.04%
2-hop displacement:      global -0.02%, seam -3.91%, active>=0.5 +5.59%

1-hop weighted:          global -1.02%, seam -3.21%, active>=0.5 +9.11%
2-hop displacement + wt: global -0.45%, seam -5.02%, active>=0.5 +11.11%
```

Decision:

- The displacement prior is useful for active-motion nodes. The weighted run is the best active-motion result so far.
- The same mechanism worsens seam behavior more than the flat 2-hop run.
- The current 2-hop displacement-aware model is not a replacement for the 1-hop baseline because seam regression is too large.
- The next diagnostic is not a larger model yet. Run the displacement-aware prior with `cost-num-hops=1` to isolate whether the 2-hop candidate radius is the source of seam noise.

Recommended next runs:

```bash
COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r5_disp_costh1 bash scripts/flow360_train_displacement_r5.sh
LOSS_MOTION_WEIGHT=4.0 COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r5_disp_costh1_active bash scripts/flow360_train_displacement_r5.sh
```

If 1-hop displacement-aware still regresses seam, run a small temperature sweep before moving to coarse-to-fine:

```bash
COST_PRIOR_TEMPERATURE=0.10 OUTPUT_DIR=/outputs/r5_disp_costh2_temp010 bash scripts/flow360_train_displacement_r5.sh
COST_PRIOR_TEMPERATURE=0.20 OUTPUT_DIR=/outputs/r5_disp_costh2_temp020 bash scripts/flow360_train_displacement_r5.sh
```

Continue only if seam moves back toward the 1-hop unweighted baseline while active metrics remain above the plain 1-hop result.

## Displacement-Aware Isolation Result

The `cost-num-hops=1` displacement-aware diagnostic isolated the seam issue.

Unweighted displacement-aware 1-hop:

```text
global:        model 0.4165 deg vs zero-flow 0.4309 deg (+3.34%)
poles:         model 0.4397 deg vs zero-flow 0.4684 deg (+6.11%)
equator:       model 0.3992 deg vs zero-flow 0.4053 deg (+1.51%)
seam:          model 0.8353 deg vs zero-flow 0.8337 deg (-0.20%)
active >=0.25: model 1.0330 deg vs zero-flow 1.0840 deg (+4.71%)
active >=0.5:  model 1.7002 deg vs zero-flow 1.7596 deg (+3.37%)
active >=1.0:  model 3.9146 deg vs zero-flow 3.9832 deg (+1.72%)
```

Motion-weighted displacement-aware 1-hop:

```text
global:        model 0.4208 deg vs zero-flow 0.4309 deg (+2.35%)
poles:         model 0.4424 deg vs zero-flow 0.4684 deg (+5.55%)
equator:       model 0.4044 deg vs zero-flow 0.4053 deg (+0.22%)
seam:          model 0.8554 deg vs zero-flow 0.8337 deg (-2.61%)
active >=0.25: model 1.0049 deg vs zero-flow 1.0840 deg (+7.29%)
active >=0.5:  model 1.6608 deg vs zero-flow 1.7596 deg (+5.61%)
active >=1.0:  model 3.8729 deg vs zero-flow 3.9832 deg (+2.77%)
```

The 2-hop temperature sweep did not recover seam behavior:

```text
2-hop temp=0.10: global +0.71%, seam -3.30%, active>=0.5 +5.85%
2-hop temp=0.20: global -0.11%, seam -4.65%, active>=0.5 +5.31%
```

Updated comparison:

```text
1-hop unweighted baseline:        global +2.32%, seam -0.85%, active>=0.5 +6.04%
1-hop displacement unweighted:    global +3.34%, seam -0.20%, active>=0.5 +3.37%

1-hop weighted baseline:          global -1.02%, seam -3.21%, active>=0.5 +9.11%
2-hop displacement weighted:      global -0.45%, seam -5.02%, active>=0.5 +11.11%
```

Decision:

- The seam problem is primarily caused by the 2-hop candidate radius.
- The displacement-aware mechanism is useful, but only the 1-hop unweighted variant is currently balanced.
- `COST_NUM_HOPS=1`, `--use-displacement-prior`, and unweighted loss is the new best global/seam baseline.
- The weighted variants remain useful as active-motion diagnostics, but they are not acceptable as the main model while seam is negative.
- Do not continue the 2-hop temperature sweep.

Next experiment:

```bash
DIRECTION=both COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r5_disp_costh1_both bash scripts/flow360_train_displacement_r5.sh
```

Important: verify the saved JSON says `"direction": "both"`. If it still says `"direction": "forward"`, rebuild the Docker image or run the Python command directly with `--direction both`; the output directory name alone does not change the dataset direction.

The first run saved under `/outputs/r5_disp_costh1_both` still had `"direction": "forward"`. Its result was consistent with the balanced forward model, but it is not a valid bidirectional test:

```text
global +3.35%, poles +6.12%, equator +1.32%, seam -0.67%, active>=0.5 +3.51%
```

If that preserves seam behavior, run:

```bash
RESOLUTION=6 COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r6_disp_costh1 bash scripts/flow360_train_displacement_r5.sh
```

For `r=6`, the validation set has enough nodes that exact `torch.quantile` can fail with `RuntimeError: quantile() input tensor is too large`. The runner now uses `--target-quantile-max-samples` with a default of `2,000,000` samples for target-motion percentiles. This affects only `target_geo_deg_p50/p90/p95`; model metrics such as global, seam, poles, equator, and active-motion means still use all valid nodes.

## Bidirectional And R6 Results

The true bidirectional run used `"direction": "both"` and stayed positive:

```text
global:        model 0.4204 deg vs zero-flow 0.4239 deg (+0.83%)
poles:         model 0.4345 deg vs zero-flow 0.4551 deg (+4.52%)
equator:       model 0.4052 deg vs zero-flow 0.4024 deg (-0.68%)
seam:          model 0.7772 deg vs zero-flow 0.7718 deg (-0.71%)
active >=0.25: model 0.9611 deg vs zero-flow 1.0637 deg (+9.65%)
active >=0.5:  model 1.5874 deg vs zero-flow 1.7204 deg (+7.73%)
active >=1.0:  model 3.7473 deg vs zero-flow 3.8754 deg (+3.31%)
```

The `r=6` forward run also succeeded after bounded percentile sampling:

```text
global:        model 0.4333 deg vs zero-flow 0.4513 deg (+3.99%)
poles:         model 0.4514 deg vs zero-flow 0.4831 deg (+6.56%)
equator:       model 0.4167 deg vs zero-flow 0.4263 deg (+2.26%)
seam:          model 1.0750 deg vs zero-flow 1.0735 deg (-0.14%)
active >=0.25: model 1.0763 deg vs zero-flow 1.1425 deg (+5.79%)
active >=0.5:  model 1.7932 deg vs zero-flow 1.8704 deg (+4.13%)
active >=1.0:  model 4.2200 deg vs zero-flow 4.3036 deg (+1.94%)
```

Decision:

- The balanced displacement-aware 1-hop model is robust enough to continue.
- `r=6` is the best current main configuration because it improves global, poles, equator, and active-motion subsets while keeping seam almost neutral.
- The next blocker is no longer whether OSLO-style features can beat zero-flow. They can.
- The next blocker is whether this approach is competitive with a strong ERP RAFT/PWCNet baseline and stable across seeds.

Recommended next work:

```bash
SEED=11 RESOLUTION=6 COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r6_disp_costh1_seed11 bash scripts/flow360_train_displacement_r5.sh
SEED=19 RESOLUTION=6 COST_NUM_HOPS=1 OUTPUT_DIR=/outputs/r6_disp_costh1_seed19 bash scripts/flow360_train_displacement_r5.sh
```

The additional seeds produced:

```text
seed 7:
global +3.99%, poles +6.56%, equator +2.26%, seam -0.14%,
active>=0.5 +4.13%, active>=1.0 +1.94%

seed 11:
global +1.90%, poles +5.98%, equator -0.73%, seam -1.60%,
active>=0.5 +5.38%, active>=1.0 +2.46%

seed 19:
global +0.10%, poles +3.76%, equator -0.81%, seam -1.72%,
active>=0.5 +6.67%, active>=1.0 +2.85%
```

Summary:

```text
global:       mean +2.00%, range +0.10% to +3.99%
poles:        mean +5.43%, range +3.76% to +6.56%
equator:      mean +0.24%, range -0.81% to +2.26%
seam:         mean -1.15%, range -1.72% to -0.14%
active>=0.5: mean +5.39%, range +4.13% to +6.67%
active>=1.0: mean +2.42%, range +1.94% to +2.85%
```

Decision:

- Active-motion and pole improvements are stable enough to justify external comparison.
- Seam is not stable enough to claim a spherical advantage yet.
- Do not continue with a larger OSLO-only architecture before adding a strong ERP baseline.
- Implement or run an ERP RAFT/PWCNet baseline and evaluate it through the same spherical metrics.
