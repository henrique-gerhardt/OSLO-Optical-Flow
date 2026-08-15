# Comparison partners — literature check before committing to a from-scratch RAFT

Written 2026-08-14, before any code was committed to the matched-budget plan.
The question this answers: *do we need to train a raster model ourselves to get a
fair architectural comparison, or does the literature already provide one?*

Short answer: **the comparison largely exists, it is on a dataset whose test
split we already hold, and one of its rows is evidence against the naive form of
our claim.** Training a RAFT ourselves is the expensive way to buy something we
can mostly read off a published table.

## 1. The training-budget gap, verified

The estimate quoted earlier in the session (~60×) was from memory of published
recipes. The torchvision reference recipe for the weights we actually run
(`raft_large`, `C_T_SKHT_V2`) is:

| stage | dataset | epochs | batch | GPUs | effective batch |
| --- | --- | --- | --- | --- | --- |
| 1 | FlyingChairs | 72 | 2 | 8×A100 | 16 |
| 2 | FlyingThings3D | 20 | 2 | 8×A100 | 16 |

torchvision states these are the translation of the original RAFT paper's
"100000 updates on each dataset". Chairs alone is therefore
72 × 22,232 = **1.60M pair presentations**; Things adds roughly 0.8M; the SKHT
fine-tune adds several hundred thousand more.

**Total ≈ 2.4–3.0M pair presentations against P1-proper's 20k steps × batch 2 =
40k. The gap is 60–75×, not 30×.** Any sentence about this in the article must
use the torchvision recipe, not the RAFT paper's, because the weights we load are
torchvision's.

## 2. The comparison already exists — PriOr-Flow, ICCV 2025 Highlight

`PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View`
(arXiv 2506.23897, code at github.com/longliangLiu/PriOr-Flow) publishes a
**matched-backbone comparison across panoramic representations**. Every row is
RAFT, every row is initialised from the same FlyingThings pre-training, and the
authors re-ran the baselines themselves rather than copying numbers — they state
they swapped DIS for RAFT in the tangent-image method and applied the SphereNet
weight transformation to RAFT, explicitly "to ensure fair comparison".

MPFDataset (EPE / SEPE, lower is better):

| method | backbone | EFT EPE | EFT SEPE | City EPE | City SEPE |
| --- | --- | --- | --- | --- | --- |
| **SphereNet** | RAFT | **13.2** | **15.7** | **8.28** | **7.44** |
| TanImg | DIS | 8.04 | 19.3 | 3.74 | 6.48 |
| TanImg | RAFT | 4.38 | 9.52 | 3.13 | 5.06 |
| MPF-net | PWC-net | 5.06 | 10.49 | 1.78 | 3.24 |
| SLOF | RAFT | 4.98 | 8.20 | 1.35 | 2.06 |
| PriOr-Flow | RAFT | **3.30** | **6.23** | **1.13** | **1.88** |

FlowScape, all weathers:

| method | backbone | EPE | SEPE |
| --- | --- | --- | --- |
| TanImg | RAFT | 18.3 | 25.3 |
| SLOF | RAFT | 7.59 | 5.79 |
| PanoFlow | RAFT | 3.38 | 4.78 |
| PriOr-Flow | RAFT | **2.33** | **3.49** |

**The row that matters most to us is SphereNet+RAFT, and it is the worst in the
table by a factor of four.** SphereNet is the distortion-aware sphere-adapted
convolution — the closest published relative of our thesis' premise — and under a
matched backbone and matched pre-training it loses decisively to methods that
stay on the raster and fix the problem elsewhere (orthogonal view, cyclic
estimation, tangent planes). This is prior evidence against "sphere-aware
geometry is the way to handle the poles", from a lab with no stake in our result.

It does not refute our replica360 finding, which is about a *different*
mechanism (estimation grid, not adapted convolution kernels) and a different
displacement regime. But it is the first thing a committee will find, and the
article must engage it rather than omit it.

## 3. Comparability audit against our shards

| | ours | PriOr-Flow's | verdict |
| --- | --- | --- | --- |
| FlowScape test | **1386 pairs, 14 seqs** | 1400 pairs | **same official split**, 14 pairs dropped for missing next-frame/validity |
| FlowScape train | 4455 + 495 val = 4950 | 5000 | same official split, MAP-grouped val carve is ours |
| MPFDataset | 1977 train / 2211 val | City 2000/138, EFT 2211/99 | **different splits — do not compare** |

**FlowScape is directly comparable and MPFDataset is not.** Our `mpf:val` is the
single sequence `EFTs_Car2000` (2211 pairs), which is exactly the size of
PriOr-Flow's EFT *training* set — our validation set is plausibly their training
data, and their 138/99 test pairs are not identified in our manifest at all.
`mpf:train` (City_100_r, City_2000_r, City_200_r, EFTs_Car100, EFTs_Car200) is in
OSLO's training mix, so **any MPFDataset comparison needs a leakage audit first**.

## 4. Two corrections to positions taken earlier in this session

**The field already has a geodesic metric.** PriOr-Flow reports SEPE, defined as
"the geodesic distance on the unit sphere", alongside ERP-pixel EPE. We are not
first, and no draft may claim to be. What is still ours is the *area-weighted*
node-mean over an equal-area grid plus the zero-flow denominator.

**The rotation control is not ours either.** Evaluating under random global
rotation is established in panoramic vision: Sphere-Depth (2026) benchmarks
monocular 360 depth under pitch/roll perturbation, Spherical-GOF (2026) trains
canonical and tests random global rotations on OmniBlender, and SO3UFormer
(2026) builds rotation-robust panoramic segmentation. Sphere-Depth's central
conclusion is ours in a different task, word for word: *"even models explicitly
designed to process spherical images exhibit substantial performance degradation
when variations in the camera pose are observed."* What survives as ours is the
application to **optical flow**, where we found no instance of it, and the
**in-domain versus out-of-domain discriminator** — Sphere-Depth states it does
not analyse whether its out-of-domain arm degrades differently, and that
contrast is exactly what makes our PanoFlow number mean something.

**But the EPE/SEPE gap is itself a measurable finding.** In PriOr-Flow's own
regional table on FlowScape:

| model | equator EPE | poles EPE | poles/equator | equator SEPE | poles SEPE | poles/equator |
| --- | --- | --- | --- | --- | --- | --- |
| PanoFlow | 0.52 | 6.25 | **12.0** | 2.87 | 6.68 | **2.33** |
| PriOr-RAFT | 0.53 | 4.13 | **7.8** | 2.94 | 4.03 | **1.37** |

**The same predictions, the same data, and the polar problem is 5× larger in
pixels than in geodesic units.** That is §16.33's mechanism showing up inside a
published table: ERP-pixel error near the pole is the angular error multiplied by
$\sec\varphi$. Our own measurement of PanoFlow on flowscape:test gives a geodesic
poles/equator of 1.45, in the same direction as their 2.33 and nowhere near 12.

## 5. What a 2026 survey says is missing

`Panoramic Scene Understanding: A Survey from Distortion-Aware Engineering to
Sphere-Native Modeling` (arXiv 2606.27745) names five evaluation gaps:
spherical-area-weighted metrics, seam-consistency tests, polar-robustness
stratification, cross-projection generalization, standardized open-world
protocols.

Two of those are things our harness already does (area-weighted metric, polar
stratification, and we also report a seam region), so the survey is a citable
authority that those gaps are real and that we fill them.

**None of our three findings appears on that list either** — but "not on the
survey's list" is not the same as "unclaimed", and §4 shows why: rotation
robustness is already established elsewhere in panoramic vision, the survey
simply does not track it. The safe reading is narrower: flow-convention
verification and the trivial-baseline audit are unclaimed as far as we have
looked; evaluation-time rotation is claimed for depth, segmentation and
reconstruction, and ours is its first application to flow plus the in-domain
discriminator.

## 6. Recommendation

**Do not train a RAFT from scratch.** The RAFT-backbone rows across four
representations already exist, on a benchmark whose test split we hold, run by
authors who took care to match the backbone. Rebuilding that costs a raster
training harness we do not have (`run_raft_shard_baseline.py` is eval-only; no
script in the repo runs `backward()` over ERP) and buys a row that is already
published.

The cheap, high-value moves, in order:

1. **Evaluate PriOr-RAFT's released FlowScape checkpoint under our protocol** —
   same thing we did for PanoFlow. Current ICCV SOTA, our geodesic area-weighted
   metric, a zero-flow denominator it has never been scored against, and the Haar
   rotation control. One adapter, no training. It also tests whether the
   orientation-contingency finding (§16.37) generalises to a third in-domain
   method or is specific to PanoFlow.
2. **Quantify the EPE/SEPE polar inflation** on our own runs. We have both
   metrics; the field publishes both; nobody has written down that the choice
   moves the polar gap by 5×.
3. **If a matched-budget OSLO row is wanted, train OSLO to *their* recipe** —
   FlowScape train, 100k steps, batch 6 ≈ 600k presentations, 15× our current
   budget — rather than training a raster model to ours. The missing cell in the
   published table is an OSLO row, not another RAFT row.

Item 3 is the only one that costs real compute, and it is the one that would let
us say something about architecture at matched budget on a third-party benchmark.
Items 1 and 2 are afternoons.

---

## 7. Item 3 costed — their recipe is 14 days on one GPU

Their FlowScape recipe is 100k steps at batch 6 = **600k pair presentations**.
Our P1-proper was 20k × 2 = 40k, so the gap to *them* is 15× (the 60–75× figure
in §1 is against torchvision's RAFT weights, a different and larger budget).

Measured throughput: `P1proper_mix20k_v2` took **22.7 h for 20k steps at batch 2**
= 4.09 s/step. Step time is dominated by the r7 retina forward/backward, not by
dataset size, so training on flowscape:train alone does not make it cheaper.

| option | steps | eff. batch | presentations | wall clock |
| --- | --- | --- | --- | --- |
| their recipe as written | 100k | 6 | 600k | **340 h ≈ 14.2 days** |
| half their steps | 50k | 6 | 300k | 170 h ≈ 7.1 days |
| their steps, our batch | 100k | 2 | 200k | 113 h ≈ 4.7 days |
| 2.5× our current budget | 50k | 2 | 100k | **57 h ≈ 2.4 days** |

**Their recipe as written is not affordable on visco3.** Anything we run will be
a stated fraction of it, which is fine — a budget gap that is *measured and
declared* is a different object from one that is unmeasured and implicit, and it
is what we have been criticising others for leaving out.

### Before committing: a throughput probe

The 4.09 s/step anchor comes from the mix with augmentation, not from
flowscape-only at effective batch 6. Measure the real number first — 200 steps,
about 40 minutes, and it decides the schedule:

```bash
cd ~/Developer/OSLO-Optical-Flow
SHARDS_HOST=../sfprep/shards \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py --shards /data/shards \
    --train-sources flowscape:train --val-sources flowscape:test \
    --output-dir /outputs/prior_recipe_probe \
    --grid healpix --resolution 6 --retina --pyramid-cache /outputs/pyramid_cache \
    --steps 200 --batch-size 2 --grad-accum 3 \
    --lr 1e-4 --onecycle --so3-prob 1.0 --so3-uniform \
    --eval-every 0 --log-every 20 --num-workers 6 --amp
```

`--grad-accum 3` gives their effective batch 6 without risking an OOM at r7.
Read `s/step` from the log, multiply, and pick the row from the table above.

### Design decisions to settle before the real run

1. **From scratch or warm-started?** They initialise from RAFT pre-trained on
   FlyingThings, which we cannot match architecturally. The clean analogue is
   warm-starting OSLO from chairs360 — our synthetic pre-training — and the clean
   *control* is from scratch. Running both doubles the cost; running only the
   warm start invites "you had extra data", running only from scratch discards
   the closest analogue of their protocol.
2. **Leakage check, mandatory.** `P1proper_ema6k`'s mix included flowscape. Its
   train sources must be confirmed to be `flowscape:train` and not `:test` before
   that checkpoint is used as an initialisation for a flowscape:test evaluation.
   The verification note at `UNIVERSALITY_TABLE.md` §7 flagged this and it was
   never closed.
3. **Report in both metric families.** Ours (area-weighted geodesic degrees over
   an equal-area grid, against a zero-flow denominator) *and* theirs (ERP-pixel
   EPE and SEPE), so the row is readable inside their table. We have the EPE
   readout from P2A; SEPE is our geodesic distance in different units.
4. **Pre-register the reading before launching.** Three refuted predictions in
   one afternoon on 2026-08-14 is the argument for this, not against it.

---

## 8. PriOr-RAFT vendored (2026-08-15) — built and validated on CPU

`spherical_flow/prior_vendor/` (7 files, from longliangLiu/PriOr-Flow) plus
`spherical_flow/prior_adapter.py`, wired into `run_raft_shard_baseline.py` as
`--prior-checkpoint` / `--prior-eval-iters`.

**Three edits to the vendored source, all recorded in each file's header:**
package-relative imports; removal of the unused `timm` import in `extractor.py`
(dead — it is imported and never referenced, so vendoring drops a dependency);
and a lazy `scipy` import inside `forward_interpolate`, a KITTI warm-start helper
this evaluation never calls. Upstream also hardcodes `.cuda()` at 14 sites in the
projection and sampling helpers, which makes the module unimportable without a
GPU. Those became an explicit `_DEVICE` that the adapter pins via `set_device`, so
box placement is identical to upstream and CPU validation became possible.

**Licence: the upstream repository declares none.** The copy is headed with its
provenance and a note that it must not be redistributed without asking the
authors. Flagging rather than assuming.

**Validated in Docker, no GPU:** the network builds at 8,337,646 parameters, and
a full run through the harness on replica360:val loads a 217-key checkpoint saved
the way upstream saves it (`DataParallel`, hence the `module.` prefix), strips the
prefix, produces finite ERP flow, and scores through the same geodesic stack every
other row uses.

**One trap found and guarded.** At 64x128 the network returns **100% NaN** without
raising, because the 1/8 feature map cannot host a 4-level correlation pyramid. At
256x512 it is clean. The adapter now refuses inputs below 128 px on either side
and raises on any non-finite output, so a bad crop can never look like a result.

**Before trusting the row: reproduce a published number.** A divergence from
PriOr-Flow's table would otherwise be ambiguous between "the method is like that"
and "we vendored it wrong" — the same ambiguity the SLOF reproduction resolved
when the convention defect surfaced. The cheap version of that check is ERP-pixel
EPE on flowscape:test against their published 2.33 px "All". Our split holds 1386
of their 1400 pairs and our readout is at r6 nodes, so a few per cent of
disagreement is expected and a large gap is not.
