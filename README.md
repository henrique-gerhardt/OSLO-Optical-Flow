# OSLO Optical Flow

This repository is an experimental fork of the OSLO/OSLO-IC image-compression codebase. The goal is to test whether OSLO's spherical directional convolutions can become a useful backbone for 360-degree optical flow.

The first target is not a full RAFT port. The current project validates a smaller question first:

Can an OSLO-style spherical model, operating on HEALPix samples, learn optical-flow correspondences on 360-degree data better than trivial baselines and provide enough signal to justify a larger spherical RAFT/PWCNet implementation?

## Current Scope

Implemented:

- Synthetic spherical-flow MVP using known 3D rotations.
- FLOW360/SLOF dataset adapter for `frames`, `fflows`, and `bflows`.
- ERP pixel-flow to spherical endpoint conversion.
- HEALPix sampling using OSLO neighbor-grid `.npz` files.
- OSLO-style model with:
  - Siamese `SDPAConv` feature encoder;
  - configurable local cost volume over 1-hop or multi-hop HEALPix neighborhoods;
  - spherical decoder that predicts tangent flow `[east, north]`;
  - geodesic endpoint loss.
- Docker CUDA runtime for Linux/NVIDIA hosts.
- Metrics for global, poles, equator, ERP seam, target-motion percentiles, and active-motion subsets.

Not implemented yet:

- Full RAFT recurrent update block on the sphere.
- Coarse-to-fine cost volume.
- ERP RAFT/PWCNet baseline runner.
- Flow visualization/export utilities.
- Final benchmark table.

## Repository Map

```text
run_flow360_mvp.py             FLOW360 supervised train/eval runner
run_spherical_flow_mvp.py      Synthetic spherical-flow runner
spherical_flow/                Geometry, datasets, and MVP model modules
spherical_models/sdpa_conv.py  OSLO SDPAConv reused by the MVP
Dockerfile.flow360             CUDA Docker image for FLOW360 experiments
docker-compose.flow360.yml     Compose entrypoint for the first experiment
scripts/flow360_smoke.sh       Minimal CUDA smoke test
scripts/flow360_train_r5.sh    First training run for RTX 3090
scripts/flow360_train_active_r5.sh Motion-weighted follow-up run
scripts/flow360_train_multihop_r5.sh Multi-hop cost-volume run
scripts/flow360_train_displacement_r5.sh Displacement-aware residual-matching run
docs/                          Project context, status, and run commands
README_OSLO_ORIGINAL.md        Original OSLO README kept for reference
```

## Data Expectations

FLOW360/SLOF should be mounted with this layout:

```text
FLOW360/
  train/<sequence>/frames/*.png
  train/<sequence>/fflows/*.npy
  train/<sequence>/bflows/*.npy
  test/<sequence>/frames/*.png
  test/<sequence>/fflows/*.npy
  test/<sequence>/bflows/*.npy
```

The current loader assumes `.npy` flow files contain ERP pixel displacement `[du, dv]`. If the local dataset stores normalized displacement, use `--flow-scale` in `run_flow360_mvp.py`.

OSLO neighbor grids should be available as:

```text
oslo_data/neighbor_grids/healpix_grid_resolution_<r>.npz
```

## First Experiment

Read [docs/FIRST_EXPERIMENT.md](docs/FIRST_EXPERIMENT.md) for the exact Docker build, smoke test, and training commands.

Short version:

```bash
docker build -f Dockerfile.flow360 -t oslo-flow360:cuda .

docker run --rm --gpus all --shm-size 16g \
  -v /absolute/path/to/FLOW360:/data/flow360:ro \
  -v /absolute/path/to/oslo_data:/data/oslo_data:ro \
  -v "$PWD/outputs:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_active_r5.sh
```

## Status

The zero-initialized FLOW360 run beats zero-flow globally and on active-motion subsets, while the motion-weighted run improves active-motion nodes further but worsens global and seam metrics. A 2-hop cost-volume run was tested next, but it did not improve over the 1-hop baselines and made the ERP seam worse.

The next useful step is no longer wider local search by itself. The code now includes a displacement-aware residual-matching mode: it converts each local candidate into an explicit tangent offset, builds a soft flow prior from cost-volume probabilities, and lets the OSLO decoder predict residual flow plus a confidence gate.

See [docs/CONTEXT_AND_STATUS.md](docs/CONTEXT_AND_STATUS.md) for the plan, decisions already made, prior synthetic results, and next engineering steps.
