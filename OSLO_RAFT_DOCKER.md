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
OSLO_GIT_SHA=$(git rev-parse HEAD) \
SHARDS_HOST=../sphereflow-dataprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft
```

Two habits this image needs:

- **Always `--build` (or `docker compose build` first) after editing source.**
  `docker compose run` alone reuses the existing image and silently runs stale code.
- **Pass `OSLO_GIT_SHA=$(git rev-parse HEAD)`** at build time. The image excludes
  `.git`, so this is how a run records its provenance hash (otherwise `unknown`);
  `git_hash()` reads the baked `OSLO_GIT_SHA` env var.

The default command is `scripts/container_smoke.sh`:

1. **Tier 1 (no data):** prints torch/CUDA/astropy-healpix versions, builds a real
   HEALPix level, and runs one OSLO-RAFT forward/backward on random frames —
   proves the image, the HEALPix path, SDPAConv, and CUDA all work. Always runs.
   Tier 1.5/1.6 cover the nested pyramid foundation and the multi-res model;
   **tier 1.65/1.7** cover OSLO-RAFT-R (the retina model): the CPU wiring suite,
   the r4 fast-graph parity check, the strict snap-constancy check on real
   HEALPix, the retina pyramid **build + disk cache round-trip** (lands in
   `/outputs/pyramid_cache`, so the real training run hits the cache), and one
   retina forward/backward with a <20 GB VRAM assert.
2. **Tier 2 (if shards mounted):** `run_oslo_raft.py --grid healpix --smoke-test`
   (one train step + one eval) — additionally exercises the sfprep data pipeline
   and the metrics. **Tier 2.7** adds the retina data integration: a loader
   throughput probe (pairs/s at the retina grid — plan §4.5) and a
   `--retina --smoke-test` end-to-end run. Skipped if `/data/shards` is empty.

Smoke geometry knobs: `OSLO_SMOKE_RETINA` (default 7), `OSLO_SMOKE_RETINA_SUP`
(default 6), `OSLO_SMOKE_RETINA_EST` (default 4), plus `OSLO_SMOKE_WORKERS`
(default 2) and `OSLO_SMOKE_ITERS` (default 8). On an emulated/memory-capped
host (e.g. Docker Desktop on a Mac, ~7.7 GiB VM), use `OSLO_SMOKE_WORKERS=0
OSLO_SMOKE_ITERS=4` and shrink the geometry (`OSLO_SMOKE_RETINA=5
OSLO_SMOKE_RETINA_SUP=4 OSLO_SMOKE_RETINA_EST=3`) — the r6+ retina tiers OOM
under qemu there; on the native GPU box the defaults apply.

## Real training

Override the command. **OSLO-RAFT-R Stage A** (the matching bootstrap — see
`docs/OSLO_RAFT_RETINA_PLAN.md` §8; run the default smoke once first so the
pyramid cache in `/outputs/pyramid_cache` is warm):

```bash
OSLO_GIT_SHA=$(git rev-parse HEAD) \
SHARDS_HOST=../sphereflow-dataprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --build --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina \
    --retina-resolution 7 --resolution 6 --estimation-resolution 4 \
    --pyramid-cache /outputs/pyramid_cache \
    --shards /data/shards \
    --device cuda --amp --onecycle \
    --steps 5000 --batch-size 2 --num-workers 6 \
    --so3-prob 1.0 \
    --train-sources replica360:train --val-sources replica360:val \
    --synth-rot-prob 0.5 --synth-rot-min-deg 1 --synth-rot-max-deg 15 \
    --val-synth-rot-prob 0.5 \
    --aux-match-weight 0.5 --aux-warmup-steps 500 \
    --eval-every 1000 \
    --output-dir /outputs/oslo_raft_retina_stageA
```

`--aux-match-weight`/`--aux-warmup-steps` drive the stencil matching loss — measured
necessary for correlation to bootstrap at all (retina plan §6/§9.2 notes): the warmup
phase trains matching alone, then the flow loss joins with the aux at 0.5x.

**Gate R1** afterwards = the same checkpoint evaluated twice (`--eval-only
--init-checkpoint /outputs/oslo_raft_retina_stageA/oslo_raft.pt`, once plain and
once with `--ablate-corr`): the corr-ablated eval must be dramatically worse on
the synth-rot val (an appearance prior cannot predict a random rotation).

The pre-retina recipes (single-res r4/r5, `--multi-res`, `--local-corr`) still
run unchanged; see `docs/OSLO_RAFT_PLAN.md` §5.
