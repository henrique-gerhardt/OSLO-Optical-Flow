# FLOW360 Docker experiment

This setup runs the OSLO-style spherical optical-flow MVP on FLOW360 inside a CUDA Docker container.

## Host assumptions

Validated target hardware:

- NVIDIA GeForce RTX 3090
- 24 GB VRAM
- NVIDIA driver 580.95.05
- Host reports CUDA 13.0

The container uses a PyTorch CUDA 12.4 runtime. This is expected to run with the 580 driver because NVIDIA drivers are backward-compatible with older CUDA runtimes.

## Expected mounted data

FLOW360 should be mounted as:

```text
/data/flow360
  train/<sequence>/{frames,fflows,bflows}
  test/<sequence>/{frames,fflows,bflows}
```

OSLO data should be mounted as:

```text
/data/oslo_data/neighbor_grids/healpix_grid_resolution_<r>.npz
```

The scripts use the precomputed OSLO HEALPix grids and `astropy-healpix`, so `healpy` is not required.

## Build

From this directory:

```bash
docker build -f Dockerfile.flow360 -t oslo-flow360:cuda .
```

## Smoke test

```bash
docker run --rm --gpus all --shm-size 16g \
  -v /absolute/path/to/FLOW360:/data/flow360:ro \
  -v /absolute/path/to/oslo_data:/data/oslo_data:ro \
  -v "$PWD/outputs:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_smoke.sh
```

The smoke test runs at HEALPix `r=4` by default and limits the dataset to a few pairs.

## Initial training run

```bash
docker run --rm --gpus all --shm-size 16g \
  -v /absolute/path/to/FLOW360:/data/flow360:ro \
  -v /absolute/path/to/oslo_data:/data/oslo_data:ro \
  -v "$PWD/outputs:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_r5.sh
```

Default training settings:

```text
resolution:        5
nodes:             12,288
batch size:        1
steps:             2,000
hidden channels:   48
feature channels:  32
max flow radius:   1.2 rad
AMP:               enabled
```

## Motion-weighted training run

The first FLOW360 run showed that zero flow is a very strong baseline because the average target displacement is small. Use this run to give more weight to samples/nodes with measurable motion:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v /absolute/path/to/FLOW360:/data/flow360:ro \
  -v /absolute/path/to/oslo_data:/data/oslo_data:ro \
  -v "$PWD/outputs:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_active_r5.sh
```

This run keeps the same model but sets:

```text
loss-motion-weight:   4.0
loss-motion-ref-deg:  1.0
steps:                3,000
```

The model now zero-initializes its final flow head by default, so step 1 starts at the zero-flow baseline instead of emitting a large random flow field.

For the RTX 3090, start with `r=5`. If memory is comfortable, try `r=6` with `BATCH_SIZE=1`; if it OOMs, keep `r=5` and improve the model with multi-hop/coarse-to-fine before scaling resolution.

Example override:

```bash
RESOLUTION=6 BATCH_SIZE=1 STEPS=1000 bash scripts/flow360_train_r5.sh
```

## Docker Compose

Create environment variables:

```bash
export FLOW360_ROOT=/absolute/path/to/FLOW360
export OSLO_DATA_ROOT=/absolute/path/to/oslo_data
export OUTPUT_DIR=$PWD/outputs
```

Then run:

```bash
docker compose -f docker-compose.flow360.yml up --build
```

## Outputs

The runner writes:

```text
/outputs/flow360_mvp.pt
/outputs/flow360_metrics.json
```

Metrics include global, poles, equator, and seam geodesic error, each compared against the zero-flow baseline.
The runner also reports active-motion subsets for target displacements above 0.25, 0.5, and 1.0 degrees. These are important because a model can lose to zero flow globally while still improving where actual motion exists.

## Notes

- The current model is still the MVP: Siamese `SDPAConv` encoder, local center+8-neighbor cost volume, and direct tangent-flow regression.
- FLOW360 flow is assumed to be ERP pixel displacement in `.npy` files. Use `--flow-scale` if the dataset copy stores normalized flow instead of pixels.
- Forward flow uses `fflows/<frame>.npy` for `frame_t -> frame_t+1`; backward flow uses `bflows/<frame>.npy` for `frame_t -> frame_t-1`.
