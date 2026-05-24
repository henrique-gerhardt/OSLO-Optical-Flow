# Context And Status

## Objective

We are evaluating whether OSLO/OSLO-IC spherical convolutions can be adapted from 360-degree image compression to 360-degree optical flow.

The long-term hypothesis is that directional, position-aware spherical convolutions on HEALPix can reduce the distortion problems created when models like RAFT or PWCNet are applied directly to equirectangular projections.

The short-term goal is narrower: run a controlled MVP on FLOW360 and decide whether the signal is strong enough to justify a larger spherical RAFT/PWCNet port.

## Why OSLO Is Relevant

OSLO contributes `SDPAConv`, a spherical directional and position-aware convolution over HEALPix nodes. Each node uses a center weight plus ordered directional neighbors. This is a better fit for spherical images than plain 2D convolution on ERP because local neighborhoods are defined on the sphere instead of on a distorted rectangular projection.

OSLO-IC extends OSLO's compression models with attention, residual blocks, transposed spherical convolution, hyperpriors, and autoregressive context. For optical flow, the most immediately reusable pieces are still the graph/neighborhood construction and `SDPAConv`; the compression entropy models are not directly useful.

## Current MVP

The implemented optical-flow MVP is intentionally small:

```text
frame1 ERP + frame2 ERP
  -> sample both frames at HEALPix nodes
  -> shared SDPAConv feature encoder
  -> local center + 8-neighbor cost volume
  -> SDPAConv motion decoder
  -> tangent flow [east, north] per HEALPix node
  -> spherical endpoint via exponential map
  -> geodesic endpoint loss
```

This tests the core spherical-feature idea before investing in:

- Spherical ConvGRU;
- recurrent RAFT update blocks;
- multi-scale correlation pyramids;
- differentiable spherical warping;
- learned upsampling of spherical flow.

## Implemented Files

```text
spherical_flow/geometry.py     HEALPix points, ERP/sphere conversion, tangent maps
spherical_flow/flow360.py      FLOW360/SLOF adapter
spherical_flow/models.py       SDPAConv feature encoder, cost volume, MVP model
spherical_flow/synthetic.py    Synthetic rotation-flow dataset
run_spherical_flow_mvp.py      Synthetic experiment runner
run_flow360_mvp.py             FLOW360 supervised experiment runner
Dockerfile.flow360             CUDA Docker image
scripts/flow360_smoke.sh       Smoke test
scripts/flow360_train_r5.sh    First training command
scripts/flow360_train_active_r5.sh Motion-weighted training command
```

## Current Validation

Completed locally on CPU:

- Python syntax validation for all new modules.
- Synthetic spherical-flow smoke test.
- FLOW360 runner smoke test with a generated mini dataset matching the expected layout.
- HEALPix topology loading from OSLO `neighbor_grids`.
- ERP pixel-flow conversion to spherical endpoint targets.

Completed synthetic HEALPix experiments:

```text
HEALPix r=3, max rotation 5 deg:  model 1.6012 deg vs zero-flow 1.9158 deg
HEALPix r=3, max rotation 10 deg: model 2.5357 deg vs zero-flow 3.8313 deg
HEALPix r=4, max rotation 5 deg:  model 1.9312 deg vs zero-flow 1.9158 deg
HEALPix r=4, max rotation 10 deg: model 3.7325 deg vs zero-flow 3.8312 deg
```

Interpretation:

- There is useful signal at low resolution.
- Scaling to higher resolution is not solved by the MVP alone.
- The next useful step is to train on FLOW360 with GPU and inspect regional metrics, not to port full RAFT immediately.

## Host And Container Assumptions

Target host already validated:

```text
GPU:            NVIDIA GeForce RTX 3090
VRAM:           24 GB
Driver:         580.95.05
Host CUDA:      13.0
Container CUDA: 12.4 via PyTorch 2.5.1 image
```

The NVIDIA driver can run older CUDA runtime containers, so the PyTorch CUDA 12.4 base image is expected to work on this host.

## FLOW360 Dataset Assumptions

Expected layout:

```text
FLOW360/
  train/<sequence>/frames/0001.png
  train/<sequence>/fflows/0001.npy
  train/<sequence>/bflows/0001.npy
  test/<sequence>/frames/0001.png
  test/<sequence>/fflows/0001.npy
  test/<sequence>/bflows/0001.npy
```

Current loader behavior:

- `direction=forward`: uses `fflows/<t>.npy` for `frame_t -> frame_t+1`.
- `direction=backward`: uses `bflows/<t>.npy` for `frame_t -> frame_t-1`.
- `direction=both`: includes both directions.
- Flow is interpreted as ERP pixel displacement.
- Horizontal ERP motion wraps around the seam.
- Vertical endpoints outside image bounds are masked invalid.

## FLOW360 Experiments

The first completed GPU run used HEALPix `r=5`:

```text
nodes:            12,288
batch size:       1
AMP:              enabled
steps:            2,000
hidden channels:  48
feature channels: 32
elapsed:          119.9 s
```

Validation result:

```text
global:  model 0.4666 deg vs zero-flow 0.4309 deg
poles:   model 0.5251 deg vs zero-flow 0.4684 deg
equator: model 0.4286 deg vs zero-flow 0.4053 deg
seam:    model 0.8864 deg vs zero-flow 0.8337 deg
```

Interpretation:

- The current direct MVP did not beat zero-flow on FLOW360.
- Zero-flow is a strong baseline because average target motion is small.
- This is not enough evidence to port full RAFT yet.
- The next run must inspect active-motion subsets, not only global mean error.

Implemented after this run:

- zero initialization for the final flow head, so the model starts exactly at zero-flow;
- target motion percentiles: `target_geo_deg_p50`, `target_geo_deg_p90`, `target_geo_deg_p95`;
- active-motion metrics above `0.25`, `0.5`, and `1.0` degrees;
- motion-weighted loss options: `--loss-motion-weight`, `--loss-motion-ref-deg`, `--loss-min-target-deg`;
- `scripts/flow360_train_active_r5.sh`.

Second completed run with zero-initialized flow head and unweighted loss:

```text
steps:              2,000
loss-motion-weight: 0.0
elapsed:            125.1 s
```

Validation result:

```text
global:        model 0.4215 deg vs zero-flow 0.4309 deg (+2.17%)
poles:         model 0.4419 deg vs zero-flow 0.4684 deg (+5.65%)
equator:       model 0.4045 deg vs zero-flow 0.4053 deg (+0.21%)
seam:          model 0.8419 deg vs zero-flow 0.8337 deg (-0.99%)
active >=0.25: model 0.9926 deg vs zero-flow 1.0840 deg (+8.43%)
active >=0.5:  model 1.6506 deg vs zero-flow 1.7596 deg (+6.19%)
active >=1.0:  model 3.8770 deg vs zero-flow 3.9832 deg (+2.67%)
```

Target motion distribution:

```text
p50: 0.1312 deg
p90: 0.7554 deg
p95: 1.0690 deg
active >=0.25 deg: 34.62%
active >=0.5 deg:  17.95%
active >=1.0 deg:   5.78%
```

Updated interpretation:

- The MVP now shows real positive signal on FLOW360.
- The global gain is small because most nodes have little motion.
- The gain is stronger on active-motion subsets and at poles.
- The ERP seam is still worse than zero-flow, so seam handling/cost-volume support is the clearest weakness.
- This is enough evidence to invest in one architectural step beyond the direct MVP, but not yet enough to port full RAFT.

Recommended next run, still before architecture changes:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_active_r5.sh
```

Success criteria for continuing:

- Model beats zero-flow on `active_0_5_*` and `active_1_0_*`.
- Model does not regress at the seam.
- Training loss decreases without numerical instability.
- Validation does not collapse after a few hundred steps.

If this fails:

- verify flow units with `--flow-scale`;
- inspect whether `.npy` layout is `[H, W, 2]` or `[2, H, W]`;
- lower `max-flow-rad` or raise it if predictions saturate;
- add multi-hop/coarse-to-fine before implementing RAFT recurrence.

## Next Engineering Steps

1. Rebuild the Docker image after the metric updates.
2. Run motion-weighted `r=5` training.
3. Inspect `outputs/flow360_metrics.json`, especially active-motion and seam metrics.
4. If active metrics improve further and seam does not regress, run:
   - `direction=both`;
   - `resolution=6`;
   - longer training.
5. If seam still loses, implement multi-hop or coarse-to-fine local cost volume before increasing model size.
6. Add an ERP RAFT/PWCNet baseline and evaluate with the same spherical metrics.
7. Only then consider a spherical RAFT update block.
