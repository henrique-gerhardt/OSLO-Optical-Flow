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
