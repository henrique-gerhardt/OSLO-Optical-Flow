# OSLO-RAFT GPU container

Reproducible image for training `run_oslo_raft.py` on the GPU box with **real
HEALPix** (`--grid healpix`). The dependency-light CPU path (`--grid fibonacci`)
needs none of this; it is for pipeline validation on a laptop.

## What's in the image

- Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11, CUDA 12.4).
- Added deps (`requirements-oslo-raft.txt`): **astropy-healpix** (the nested
  HEALPix backend — `spherical_flow.geometry.healpix_unit_vectors` uses it
  automatically when healpy is absent; no cfitsio build needed), pillow,
  opencv-python-headless, tqdm.
- The OSLO repo is copied in; `torch`/`torchvision`/`numpy` come from the base.
- The **`sfprep` shard reader is baked in**: the Dockerfile clones
  `github.com/henrique-gerhardt/sfprep` at build time, pinned to `SFPREP_REF`
  (a commit SHA). It has no Python packaging, so it is cloned to
  `/workspace/sphereflow-dataprep` rather than `pip install`ed, and
  `shard_dataset.py` finds it via the `SPHEREFLOW_DATAPREP` env var. Update it with
  `--build-arg SFPREP_REF=<sha|branch>`.

## Mounts (only the large, regenerable data)

| host (override env) | container | why |
| --- | --- | --- |
| `SHARDS_HOST` (default `../sphereflow-dataprep/shards`) | `/data/shards` | normalized tar shards (`OSLO_SHARDS`) |
| `OUTPUT_DIR` (default `./outputs`) | `/outputs` | checkpoints + metrics JSON |

Only the shard *data* is mounted; the `sfprep` reader *code* is in the image. To
point at a different shard set, override `SHARDS_HOST` (or `OSLO_SHARDS` inside the
container).

## Build + smoke

```bash
SHARDS_HOST=../sphereflow-dataprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft
```

The default command is `scripts/container_smoke.sh`:

1. **Tier 1 (no data):** prints torch/CUDA/astropy-healpix versions, builds a real
   HEALPix level, and runs one OSLO-RAFT forward/backward on random frames —
   proves the image, the HEALPix path, SDPAConv, and CUDA all work. Always runs.
2. **Tier 2 (if shards mounted):** `run_oslo_raft.py --grid healpix --smoke-test`
   (one train step + one eval) — additionally exercises the sfprep data pipeline
   and the metrics. Skipped with a message if `/data/shards` is empty.

## Real training

Override the command (Stage T1, distillation pretraining — see
`docs/OSLO_RAFT_PLAN.md` §5):

```bash
... run --rm oslo-raft python run_oslo_raft.py \
    --grid healpix --resolution 4 \
    --shards /data/shards \
    --device cuda --amp --onecycle \
    --steps 100000 --batch-size 2 --num-workers 4 \
    --so3-prob 1.0 \
    --output-dir /outputs/oslo_raft_t1
```

`--resolution 4` = 3,072 nodes (the plan's debug/first grid); `--resolution 5` =
12,288 nodes is the T1 target once memory is confirmed on the card.

## Note on the model still being single-resolution

The image runs the current single-resolution model on real HEALPix. The
nested-HEALPix **multi-level** builder (encoder pyramid §4.1, correlation pyramid
§4.2, convex upsampler §4.5) is the next increment and lands inside this same
container, where it can be validated on real geometry + CUDA.
