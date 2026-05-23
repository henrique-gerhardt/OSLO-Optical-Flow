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

## First Experiment

Start with HEALPix `r=5`:

```text
nodes:            12,288
batch size:       1
AMP:              enabled
steps:            2,000
hidden channels:  48
feature channels: 32
```

This is conservative for a 24 GB RTX 3090. If memory is comfortable, try `r=6` next.

Success criteria for continuing:

- Model beats zero-flow globally.
- Model beats zero-flow more clearly at poles and seam than at equator.
- Training loss decreases without numerical instability.
- Validation does not collapse after a few hundred steps.

If this fails:

- verify flow units with `--flow-scale`;
- inspect whether `.npy` layout is `[H, W, 2]` or `[2, H, W]`;
- lower `max-flow-rad` or raise it if predictions saturate;
- add multi-hop/coarse-to-fine before implementing RAFT recurrence.

## Next Engineering Steps

1. Run Docker smoke test on the RTX 3090 host.
2. Run `r=5` training for 2,000 steps.
3. Inspect `outputs/flow360_metrics.json`.
4. If metrics beat zero-flow, run:
   - `direction=both`;
   - `resolution=6`;
   - longer training.
5. Add an ERP RAFT/PWCNet baseline and evaluate with the same spherical metrics.
6. Implement multi-hop or coarse-to-fine local cost volume.
7. Only then consider a spherical RAFT update block.
