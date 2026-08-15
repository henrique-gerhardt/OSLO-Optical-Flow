# Context And Status

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
> **This file specifically:** Every FLOW360 number here predates the convention fix and is VOID, including the frozen-RAFT-vs-zero comparisons.


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
elapsed:            126.1 s
```

Validation result:

```text
global:        model 0.4209 deg vs zero-flow 0.4309 deg (+2.32%)
poles:         model 0.4408 deg vs zero-flow 0.4684 deg (+5.89%)
equator:       model 0.4039 deg vs zero-flow 0.4053 deg (+0.35%)
seam:          model 0.8408 deg vs zero-flow 0.8337 deg (-0.85%)
active >=0.25: model 0.9943 deg vs zero-flow 1.0840 deg (+8.27%)
active >=0.5:  model 1.6533 deg vs zero-flow 1.7596 deg (+6.04%)
active >=1.0:  model 3.8801 deg vs zero-flow 3.9832 deg (+2.59%)
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
- The repeated zero-init run is stable: results are within noise of the previous run.
- The global gain is small because most nodes have little motion.
- The gain is stronger on active-motion subsets and at poles.
- The ERP seam is still worse than zero-flow, so seam handling/cost-volume support is the clearest weakness.
- This is enough evidence to invest in one architectural step beyond the direct MVP, but not yet enough to port full RAFT.

This second run was not motion-weighted yet (`loss-motion-weight=0.0`). Recommended next run, still before architecture changes:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v "$FLOW360_ROOT:/data/flow360:ro" \
  -v "$OSLO_DATA_ROOT:/data/oslo_data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  oslo-flow360:cuda \
  bash scripts/flow360_train_active_r5.sh
```

Third completed run with motion-weighted loss:

```text
steps:              3,000
loss-motion-weight: 4.0
loss-motion-ref:    1.0 deg
elapsed:            174.0 s
```

Validation result:

```text
global:        model 0.4353 deg vs zero-flow 0.4309 deg (-1.02%)
poles:         model 0.4599 deg vs zero-flow 0.4684 deg (+1.82%)
equator:       model 0.4140 deg vs zero-flow 0.4053 deg (-2.13%)
seam:          model 0.8604 deg vs zero-flow 0.8337 deg (-3.21%)
active >=0.25: model 0.9679 deg vs zero-flow 1.0840 deg (+10.71%)
active >=0.5:  model 1.5992 deg vs zero-flow 1.7596 deg (+9.11%)
active >=1.0:  model 3.8099 deg vs zero-flow 3.9832 deg (+4.35%)
```

Motion-weighted interpretation:

- Weighting by motion does what it is supposed to do: active-motion improvements become stronger.
- The tradeoff is clear: global, equator, and seam regress.
- The seam regression grows from about `-0.85%` to `-3.21%`.
- More loss weighting is unlikely to solve the core issue because it does not expand the model's correspondence search support.
- The next useful implementation should be architectural: multi-hop first, then coarse-to-fine if multi-hop is not enough.

Multi-hop cost volume implementation:

- `run_flow360_mvp.py` now exposes `--cost-num-hops`.
- `spherical_flow/models.py` accepts a larger cost graph separately from the 1-hop encoder graph.
- `cost-num-hops=1` keeps the original center + 8-neighbor cost volume.
- `cost-num-hops=2` uses center + 24 neighbors, so the smoke log should show `cost_shape=(1, 12288, 25)` at `r=5`.
- `scripts/flow360_train_multihop_r5.sh` runs the `r=5`, `cost-num-hops=2` experiment.

Fourth and fifth completed runs with 2-hop cost volume:

```text
Unweighted 2-hop:
global:        model 0.4234 deg vs zero-flow 0.4309 deg (+1.75%)
poles:         model 0.4410 deg vs zero-flow 0.4684 deg (+5.84%)
equator:       model 0.4101 deg vs zero-flow 0.4053 deg (-1.17%)
seam:          model 0.8536 deg vs zero-flow 0.8337 deg (-2.40%)
active >=0.25: model 0.9998 deg vs zero-flow 1.0840 deg (+7.77%)
active >=0.5:  model 1.6643 deg vs zero-flow 1.7596 deg (+5.41%)
active >=1.0:  model 3.9102 deg vs zero-flow 3.9832 deg (+1.83%)

Motion-weighted 2-hop:
global:        model 0.4358 deg vs zero-flow 0.4309 deg (-1.15%)
poles:         model 0.4440 deg vs zero-flow 0.4684 deg (+5.21%)
equator:       model 0.4261 deg vs zero-flow 0.4053 deg (-5.13%)
seam:          model 0.8696 deg vs zero-flow 0.8337 deg (-4.31%)
active >=0.25: model 0.9937 deg vs zero-flow 1.0840 deg (+8.32%)
active >=0.5:  model 1.6394 deg vs zero-flow 1.7596 deg (+6.83%)
active >=1.0:  model 3.8557 deg vs zero-flow 3.9832 deg (+3.20%)
```

2-hop interpretation:

- 2-hop still beats zero-flow in some regions, but it is worse than the best 1-hop runs.
- Unweighted 2-hop loses global, seam, and active-motion performance relative to unweighted 1-hop.
- Motion-weighted 2-hop loses active-motion and seam performance relative to motion-weighted 1-hop.
- The wider flat cost volume is adding noise; it is not enough for reliable correspondence selection.
- Do not continue to `cost-num-hops=3` before adding displacement-aware or coarse-to-fine structure.

Displacement-aware residual matching implementation:

- `run_flow360_mvp.py` now exposes `--use-displacement-prior` and `--cost-prior-temperature`.
- The runner builds tangent offsets for center + each local cost-volume candidate.
- `spherical_flow/models.py` builds a soft flow prior from cost probabilities and candidate offsets.
- The decoder predicts `residual + gate * flow_prior`.
- In this mode, the final head is initialized with residual zero and gate bias near closed, preserving the useful zero-flow starting point.
- `scripts/flow360_train_displacement_r5.sh` runs `r=5`, `cost-num-hops=2`, and `--use-displacement-prior`.

Sixth and seventh completed runs with displacement-aware 2-hop prior:

```text
Unweighted displacement-aware 2-hop:
global:        model 0.4310 deg vs zero-flow 0.4309 deg (-0.02%)
poles:         model 0.4413 deg vs zero-flow 0.4684 deg (+5.78%)
equator:       model 0.4198 deg vs zero-flow 0.4053 deg (-3.57%)
seam:          model 0.8663 deg vs zero-flow 0.8337 deg (-3.91%)
active >=0.25: model 1.0031 deg vs zero-flow 1.0840 deg (+7.46%)
active >=0.5:  model 1.6612 deg vs zero-flow 1.7596 deg (+5.59%)
active >=1.0:  model 3.8903 deg vs zero-flow 3.9832 deg (+2.33%)

Motion-weighted displacement-aware 2-hop:
global:        model 0.4328 deg vs zero-flow 0.4309 deg (-0.45%)
poles:         model 0.4483 deg vs zero-flow 0.4684 deg (+4.28%)
equator:       model 0.4163 deg vs zero-flow 0.4053 deg (-2.71%)
seam:          model 0.8755 deg vs zero-flow 0.8337 deg (-5.02%)
active >=0.25: model 0.9521 deg vs zero-flow 1.0840 deg (+12.16%)
active >=0.5:  model 1.5640 deg vs zero-flow 1.7596 deg (+11.11%)
active >=1.0:  model 3.7398 deg vs zero-flow 3.9832 deg (+6.11%)
```

Displacement-aware interpretation:

- The geometric prior helps active-motion nodes; the weighted displacement-aware run is the strongest active-motion result so far.
- It still fails the seam criterion. Seam regression reaches `-5.02%` in the weighted run.
- Because unweighted displacement-aware also regresses global and seam, the issue is not only motion weighting.
- The likely problem is over-trusting ambiguous wider candidates, especially around ERP seam regions and low-motion areas.
- Next diagnostic: run displacement-aware matching with `cost-num-hops=1`; if that does not recover seam behavior, run a softer prior-temperature sweep.

Eighth through eleventh diagnostic runs:

```text
Unweighted displacement-aware 1-hop:
global:        model 0.4165 deg vs zero-flow 0.4309 deg (+3.34%)
poles:         model 0.4397 deg vs zero-flow 0.4684 deg (+6.11%)
equator:       model 0.3992 deg vs zero-flow 0.4053 deg (+1.51%)
seam:          model 0.8353 deg vs zero-flow 0.8337 deg (-0.20%)
active >=0.25: model 1.0330 deg vs zero-flow 1.0840 deg (+4.71%)
active >=0.5:  model 1.7002 deg vs zero-flow 1.7596 deg (+3.37%)
active >=1.0:  model 3.9146 deg vs zero-flow 3.9832 deg (+1.72%)

Motion-weighted displacement-aware 1-hop:
global:        model 0.4208 deg vs zero-flow 0.4309 deg (+2.35%)
poles:         model 0.4424 deg vs zero-flow 0.4684 deg (+5.55%)
equator:       model 0.4044 deg vs zero-flow 0.4053 deg (+0.22%)
seam:          model 0.8554 deg vs zero-flow 0.8337 deg (-2.61%)
active >=0.25: model 1.0049 deg vs zero-flow 1.0840 deg (+7.29%)
active >=0.5:  model 1.6608 deg vs zero-flow 1.7596 deg (+5.61%)
active >=1.0:  model 3.8729 deg vs zero-flow 3.9832 deg (+2.77%)

2-hop displacement prior, temp=0.10:
global +0.71%, seam -3.30%, active>=0.5 +5.85%

2-hop displacement prior, temp=0.20:
global -0.11%, seam -4.65%, active>=0.5 +5.31%
```

Isolation interpretation:

- The 1-hop displacement-aware unweighted run is the new best balanced model.
- It beats the previous global result, improves equator and poles, and reduces seam regression from `-0.85%` to `-0.20%`.
- The active-motion metrics are lower than the weighted models, so there is still a tradeoff.
- The 2-hop temperature sweep confirms that softer probabilities do not fix the seam issue; the candidate radius is the primary source of noise.
- Mainline should move forward with `cost-num-hops=1`, `--use-displacement-prior`, and unweighted loss.

Operational note:

- A run stored under `/outputs/r5_disp_costh1_both` still reported `"direction": "forward"`, so it was not actually a bidirectional dataset run. It was nevertheless consistent with the current balanced model: global `+3.35%`, poles `+6.12%`, equator `+1.32%`, seam `-0.67%`, active>=0.5 `+3.51%`. Rebuild the Docker image after script changes and verify the JSON says `"direction": "both"`.
- `r=6` validation can exceed `torch.quantile`'s practical input size when collecting target-motion percentiles. The runner now limits percentile samples with `--target-quantile-max-samples` while keeping all primary model metrics exact over all valid nodes.

Twelfth and thirteenth validation runs:

```text
True bidirectional r=5, displacement-aware 1-hop, unweighted:
global:        model 0.4204 deg vs zero-flow 0.4239 deg (+0.83%)
poles:         model 0.4345 deg vs zero-flow 0.4551 deg (+4.52%)
equator:       model 0.4052 deg vs zero-flow 0.4024 deg (-0.68%)
seam:          model 0.7772 deg vs zero-flow 0.7718 deg (-0.71%)
active >=0.25: model 0.9611 deg vs zero-flow 1.0637 deg (+9.65%)
active >=0.5:  model 1.5874 deg vs zero-flow 1.7204 deg (+7.73%)
active >=1.0:  model 3.7473 deg vs zero-flow 3.8754 deg (+3.31%)

r=6 forward, displacement-aware 1-hop, unweighted:
global:        model 0.4333 deg vs zero-flow 0.4513 deg (+3.99%)
poles:         model 0.4514 deg vs zero-flow 0.4831 deg (+6.56%)
equator:       model 0.4167 deg vs zero-flow 0.4263 deg (+2.26%)
seam:          model 1.0750 deg vs zero-flow 1.0735 deg (-0.14%)
active >=0.25: model 1.0763 deg vs zero-flow 1.1425 deg (+5.79%)
active >=0.5:  model 1.7932 deg vs zero-flow 1.8704 deg (+4.13%)
active >=1.0:  model 4.2200 deg vs zero-flow 4.3036 deg (+1.94%)
```

Validation interpretation:

- The true `direction=both` run is positive on global, poles, and active-motion subsets, with seam still near neutral. This confirms the result is not only a forward-flow artifact.
- The `r=6` forward run strengthens the balanced model: global, poles, equator, and active-motion metrics improve over zero-flow, while seam is almost neutral.
- Scaling resolution did not break the method; the quantile sampling fix worked and reported about 2M percentile samples.
- The main open weakness is the tradeoff between active-motion strength and seam stability. The best balanced model is not the best active-motion model.

R6 seed stability:

```text
seed 7:
global +3.99%, poles +6.56%, equator +2.26%, seam -0.14%,
active>=0.25 +5.79%, active>=0.5 +4.13%, active>=1.0 +1.94%

seed 11:
global +1.90%, poles +5.98%, equator -0.73%, seam -1.60%,
active>=0.25 +7.28%, active>=0.5 +5.38%, active>=1.0 +2.46%

seed 19:
global +0.10%, poles +3.76%, equator -0.81%, seam -1.72%,
active>=0.25 +8.84%, active>=0.5 +6.67%, active>=1.0 +2.85%
```

Three-seed summary:

```text
global improvement:       mean +2.00%, range +0.10% to +3.99%
poles improvement:        mean +5.43%, range +3.76% to +6.56%
equator improvement:      mean +0.24%, range -0.81% to +2.26%
seam improvement:         mean -1.15%, range -1.72% to -0.14%
active>=0.5 improvement:  mean +5.39%, range +4.13% to +6.67%
active>=1.0 improvement:  mean +2.42%, range +1.94% to +2.85%
```

Seed-stability interpretation:

- Active-motion gains are stable and positive across seeds.
- Pole gains are stable and positive across seeds.
- Global improvement is positive on average but high variance.
- Seam is consistently slightly worse than zero-flow at `r=6`; seed 7 is nearly neutral, but seeds 11 and 19 regress around `-1.6%` to `-1.7%`.
- The method is promising enough to compare against ERP RAFT/PWCNet, but not stable enough to justify a large spherical RAFT port before that comparison.

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

1. Treat `r=6`, displacement-aware 1-hop, unweighted as the current main OSLO-style configuration.
2. Run the implemented TorchVision ERP RAFT baseline and evaluate it with the same spherical metrics.
3. Compare RAFT against zero-flow and the OSLO-style model on global, seam, poles, equator, and active-motion subsets.
4. If OSLO remains competitive at seam/poles but weak on active motion, investigate a gated motion loss or region-aware loss that preserves seam stability.
5. Only then consider a spherical RAFT update block.

The RAFT runner is `run_erp_raft_baseline.py`; the Docker entrypoint is:

```bash
RESOLUTION=6 DIRECTION=forward OUTPUT_DIR=/outputs/raft_r6_forward bash scripts/flow360_raft_baseline.sh
```

The first pretrained RAFT Large ERP run completed and was worse than zero-flow across all metrics:

```text
global:        RAFT 0.7361 deg vs zero-flow 0.4513 deg (-63.12%)
poles:         RAFT 0.7901 deg vs zero-flow 0.4831 deg (-63.54%)
equator:       RAFT 0.6892 deg vs zero-flow 0.4263 deg (-61.65%)
seam:          RAFT 1.4348 deg vs zero-flow 1.0793 deg (-32.94%)
active >=0.5:  RAFT 2.8470 deg vs zero-flow 1.8704 deg (-52.21%)
active >=1.0:  RAFT 6.2544 deg vs zero-flow 4.3036 deg (-45.33%)
```

Updated decision:

- Direct pretrained ERP RAFT is not competitive on FLOW360.
- The OSLO-style `r=6`, displacement-aware 1-hop model is much stronger than direct ERP RAFT on this spherical evaluation.
- Before investing in a spherical RAFT port, run a small RAFT prediction diagnostic to verify flow sign, scale, and coordinate convention:

```bash
MAX_PAIRS=8 SAVE_PREDICTIONS=1 OUTPUT_DIR=/outputs/raft_r6_forward_debug bash scripts/flow360_raft_baseline.sh
MAX_PAIRS=8 OUTPUT_DIR=/outputs/raft_r6_forward_debug bash scripts/flow360_raft_prediction_diagnostic.sh
```

The diagnostic compares saved RAFT `.npy` predictions against FLOW360 pixel-flow targets, including sign-flipped, axis-swapped, and scalar-fitted variants. The first 8-pair diagnostic found:

```text
zero-flow pixel EPE: 8.5314 px
identity:            9.3347 px (-9.42%)
negated:             7.9665 px (+6.62%)
identity scaled:     8.4738 px (+0.67%), scale=-2.9225
```

This means the original RAFT spherical run used the wrong sign relative to the FLOW360 targets. The runner now supports `--flow-transform`. The full corrected baseline is:

```bash
OUTPUT_DIR=/outputs/raft_r6_forward_negated bash scripts/flow360_raft_baseline.sh
```

The negated full run produced:

```text
global:        RAFT 0.2698 deg vs zero-flow 0.4513 deg (+40.21%)
poles:         RAFT 0.3420 deg vs zero-flow 0.4831 deg (+29.22%)
equator:       RAFT 0.2379 deg vs zero-flow 0.4263 deg (+44.19%)
seam:          RAFT 0.8537 deg vs zero-flow 1.0793 deg (+20.90%)
active >=0.25: RAFT 0.6124 deg vs zero-flow 1.1425 deg (+46.40%)
active >=0.5:  RAFT 1.0404 deg vs zero-flow 1.8704 deg (+44.38%)
active >=1.0:  RAFT 2.6364 deg vs zero-flow 4.3036 deg (+38.74%)
```

Updated decision:

- Interpret the negated run, not the original identity run, as the direct ERP RAFT baseline.
- Corrected direct ERP RAFT is much stronger than the current OSLO-style MVP across every reported subset.
- The OSLO MVP remains useful as a proof that spherical HEALPix features can learn motion, but it is not competitive with pretrained RAFT.
- Future OSLO work must target the RAFT gap directly, not just beat zero-flow. The first implemented step is a frozen-RAFT HEALPix residual corrector that starts exactly at RAFT and learns only a small OSLO/HEALPix tangent-flow correction.
- The FLOW360 RAFT script now defaults to `RAFT_FLOW_TRANSFORM=negated`; override it only for diagnostics or for datasets with a different convention.

## RAFT-Conditioned Residual Plan

The next project is implemented around this data flow:

```text
frozen RAFT ERP flow, negated
  -> cache as HEALPix tangent flow
  -> train OSLO/HEALPix residual head
  -> final prediction = RAFT tangent flow + residual
```

Implementation entry points:

```text
run_flow360_cache_raft.py
run_flow360_raft_residual.py
scripts/flow360_cache_raft_r6.sh
scripts/flow360_raft_residual_r6.sh
```

Run cache generation:

```bash
SPLIT=train RESOLUTION=6 DIRECTION=forward bash scripts/flow360_cache_raft_r6.sh
SPLIT=test RESOLUTION=6 DIRECTION=forward bash scripts/flow360_cache_raft_r6.sh
```

Run the residual experiment:

```bash
RESOLUTION=6 DIRECTION=forward OUTPUT_DIR=/outputs/raft_residual_r6_forward bash scripts/flow360_raft_residual_r6.sh
```

Success is measured against corrected RAFT, not against zero-flow:

```text
primary:   seam_geo_deg < 0.8537 and global_geo_deg <= 0.2698
secondary: improve poles or active-motion while keeping global within 1% of RAFT
```

First residual run:

```text
residual_max_rad:    0.05
residual_reg_weight: 0.01
steps:               3000
seed:                7

global:       residual 0.2705 vs RAFT 0.2699 (-0.20%)
poles:        residual 0.3605 vs RAFT 0.3420 (-5.40%)
equator:      residual 0.2367 vs RAFT 0.2380 (+0.54%)
seam:         residual 0.8466 vs RAFT 0.8488 (+0.25%)
active>=0.25: residual 0.6049 vs RAFT 0.6125 (+1.23%)
active>=0.5:  residual 1.0362 vs RAFT 1.0404 (+0.41%)
active>=1.0:  residual 2.6413 vs RAFT 2.6365 (-0.18%)
```

Decision:

- The first residual run is a secondary success, not a primary success.
- It found a small useful correction for seam, equator, and moderate active motion.
- It is not acceptable as the main result because global worsened slightly and poles worsened materially.
- The next step is not a larger residual model. The residual must be more conservative.

Recommended next sweep:

```bash
RESIDUAL_MAX_RAD=0.02 RESIDUAL_REG_WEIGHT=0.05 OUTPUT_DIR=/outputs/raft_residual_r6_constrained_002_005 bash scripts/flow360_raft_residual_r6.sh
RESIDUAL_MAX_RAD=0.01 RESIDUAL_REG_WEIGHT=0.05 OUTPUT_DIR=/outputs/raft_residual_r6_constrained_001_005 bash scripts/flow360_raft_residual_r6.sh
RESIDUAL_MAX_RAD=0.02 RESIDUAL_REG_WEIGHT=0.10 OUTPUT_DIR=/outputs/raft_residual_r6_constrained_002_010 bash scripts/flow360_raft_residual_r6.sh
```

Constrained sweep result:

```text
0.02 / 0.05:
global +0.01%, seam +0.24%, poles -3.97%, active>=0.25 +1.11%, active>=0.5 +0.39%, active>=1.0 -0.03%

0.01 / 0.05:
global +0.51%, seam +0.24%, poles -1.49%, active>=0.25 +1.09%, active>=0.5 +0.40%, active>=1.0 +0.05%

0.02 / 0.10:
global +0.30%, seam +0.22%, poles -2.53%, active>=0.25 +1.09%, active>=0.5 +0.38%, active>=1.0 +0.02%
```

Current decision after the constrained sweep:

- The best setting is `RESIDUAL_MAX_RAD=0.01` and `RESIDUAL_REG_WEIGHT=0.05`.
- This setting improves global, seam, equator, and active-motion metrics against corrected RAFT.
- The remaining blocker is pole regression: `poles_geo_deg` rises from `0.3420` to `0.3471`, about `-1.49%` vs RAFT.
- The next code path is implemented as `POLE_RESIDUAL_REG_WEIGHT`, an extra residual penalty applied only on the polar region.

Pole-protected result:

```text
0.01 / 0.05 / pole 0.05:
global +1.10%, seam +0.25%, poles +0.86%, equator +0.67%, active>=0.25 +0.95%, active>=0.5 +0.45%, active>=1.0 +0.12%

0.01 / 0.05 / pole 0.10:
global +0.95%, seam +0.29%, poles +0.44%, equator +0.70%, active>=0.25 +0.88%, active>=0.5 +0.48%, active>=1.0 +0.14%
```

Current best candidate:

```text
RESIDUAL_MAX_RAD=0.01
RESIDUAL_REG_WEIGHT=0.05
POLE_RESIDUAL_REG_WEIGHT=0.05

global:       residual 0.2670 vs RAFT 0.2699 (+1.10%)
poles:        residual 0.3391 vs RAFT 0.3420 (+0.86%)
equator:      residual 0.2364 vs RAFT 0.2380 (+0.67%)
seam:         residual 0.8466 vs RAFT 0.8488 (+0.25%)
active>=0.25: residual 0.6067 vs RAFT 0.6125 (+0.95%)
active>=0.5:  residual 1.0358 vs RAFT 1.0404 (+0.45%)
active>=1.0:  residual 2.6333 vs RAFT 2.6365 (+0.12%)
```

Robustness validation passed:

```text
seeds: 7, 11, 19

global mean:       0.2669 vs RAFT 0.2699 (+1.11%)
poles mean:        0.3396 vs RAFT 0.3420 (+0.72%)
equator mean:      0.2364 vs RAFT 0.2380 (+0.70%)
seam mean:         0.8466 vs RAFT 0.8488 (+0.26%)
active>=0.25 mean: 0.6071 vs RAFT 0.6125 (+0.89%)
active>=0.5 mean:  1.0368 vs RAFT 1.0404 (+0.35%)
active>=1.0 mean:  2.6342 vs RAFT 2.6365 (+0.08%)
```

This is the current main result. It is positive across all tracked metrics for all three seeds. More scalar regularizer tuning is lower priority.

The first `direction=both` validation is diagnostic only, not a clean confirmation:

```text
direction=both, seed=7:
zero-flow global: 0.4368
RAFT global:      0.4836
residual global:  0.4762

zero-flow poles:  0.4644
RAFT poles:       0.5296
residual poles:   0.5198
```

The residual improves RAFT on every tracked `both` metric, but RAFT itself is worse than zero-flow. Since forward-only RAFT is strong, the likely issue is the backward FLOW360 convention or the backward cache transform.

Backward transform diagnostic on 64 test pairs:

```text
zero-flow mean_epe_px: 4.4501
best_raw:              identity
best_raw mean_epe_px:  4.0139
best_raw improvement:  +9.80%
```

Full backward-only RAFT with `identity` passed:

```text
global:       0.2652 vs zero 0.4221 (+37.17%)
poles:        0.3364 vs zero 0.4455 (+24.48%)
equator:      0.2345 vs zero 0.3998 (+41.34%)
seam:         0.5733 vs zero 0.7742 (+25.94%)
active>=0.5:  0.9912 vs zero 1.7088 (+42.00%)
```

Use `RAFT_FLOW_TRANSFORM=identity` for backward cache and keep `RAFT_FLOW_TRANSFORM=negated` for forward cache.

Mixed-transform `direction=both` result, seed 7:

```text
global:       residual 0.2643 vs RAFT 0.2676 (+1.23%)
poles:        residual 0.3351 vs RAFT 0.3393 (+1.23%)
equator:      residual 0.2347 vs RAFT 0.2363 (+0.68%)
seam:         residual 0.7086 vs RAFT 0.7104 (+0.25%)
active>=0.25: residual 0.5959 vs RAFT 0.6010 (+0.86%)
active>=0.5:  residual 1.0128 vs RAFT 1.0160 (+0.32%)
active>=1.0:  residual 2.5491 vs RAFT 2.5512 (+0.08%)
```

Mixed-transform `direction=both` robustness passed across seeds `7`, `11`, and `19`:

```text
global mean:       residual 0.2646 vs RAFT 0.2676 (+1.14%)
poles mean:        residual 0.3356 vs RAFT 0.3393 (+1.08%)
equator mean:      residual 0.2348 vs RAFT 0.2363 (+0.64%)
seam mean:         residual 0.7084 vs RAFT 0.7104 (+0.28%)
active>=0.25 mean: residual 0.5969 vs RAFT 0.6010 (+0.69%)
active>=0.5 mean:  residual 1.0133 vs RAFT 1.0160 (+0.26%)
active>=1.0 mean:  residual 2.5495 vs RAFT 2.5512 (+0.07%)
```

This confirms the correction is not forward-only. The current result set now has robust forward-only and bidirectional validations.

Next priority:

```text
1. automate result aggregation from metrics JSON files;
2. add direction-split metrics for direction=both;
3. add qualitative RAFT-vs-residual error visualizations;
4. then try a RAFT-error-aware residual objective.
```

See `docs/NEXT_SESSION_HANDOFF.md` for the current handoff and next-session prompt, `docs/RAFT_BASELINE.md` for the corrected RAFT baseline, and `docs/RAFT_RESIDUAL_EXPERIMENT.md` for the residual experiment protocol and result log.
