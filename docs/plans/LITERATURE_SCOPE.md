# What is ours and what already exists — check here before writing any claim

Opened 2026-08-14 after a literature check found that several things this project
had been treating as its own were already published. **Every novelty sentence in
the article, the thesis, or a slide must be checked against this file first.**
If a claim is not in the "ours" column here, it does not get written as ours.

The rule this file enforces: *we may claim a measurement we made; we may not
claim a method, metric, or protocol that someone else published first.*

---

## 1. The register

| claim | status | who has it | what we may still say |
| --- | --- | --- | --- |
| Geodesic (angular) error metric for 360° flow | **EXISTS** | SEPE — "the geodesic distance between endpoints on the unit sphere" — is standard in the 360-flow literature and PriOr-Flow reports it for every baseline row | We use an angular metric *because the field already does*. Ours is the **area-weighted node mean on an equal-area grid**, plus a **zero-flow denominator**, not the metric itself |
| Polar vs equatorial regional breakdown | **EXISTS** | PriOr-Flow, Table 6 (FlowScape: equator/poles/all, in both EPE and SEPE) | Nothing. Do not present regional stratification as our idea |
| Area-weighted spherical metric | **OPEN, weakly** | the 2026 panoramic survey lists "spherical-area-weighted metrics" as a *missing* evaluation component | "We implement what [survey] identifies as missing." Not "we invented" |
| Evaluating under random global rotation | **EXISTS in panoramic vision, NOT in flow** | Sphere-Depth (2026) for monocular 360 depth; Spherical-GOF (2026) trains canonical and tests random global rotations on OmniBlender; SO3UFormer (2026) for rotation-robust panoramic segmentation | The **application to 360 optical flow**, and the **in-domain vs out-of-domain discriminator** — Sphere-Depth explicitly does *not* analyse whether in-domain models degrade more |
| "Spherical-aware models degrade under camera-pose change" | **EXISTS** | Sphere-Depth's central conclusion, verbatim: "even models explicitly designed to process spherical images exhibit substantial performance degradation when variations in the camera pose are observed" | Our numbers replicate this in a new task. **Frame as replication, not discovery** |
| No 360°-flow paper reports a trivial baseline | **OURS** (reporting-practice claim) | six papers read in full text, none reports a trivial predictor | Safe as stated. See §2 for the *derived* claim that is void |
| Flow-convention verification (sign/direction audit) | **OURS** | no prior art found | Safe. The methodological point — a sign-invariant baseline plus an exactly reproduced published table cannot detect it — is the contribution |
| Matched-backbone comparison across panoramic representations | **EXISTS** | PriOr-Flow Table 5: SphereNet / TanImg / MPF-net / SLOF / PanoFlow / PriOr, all on RAFT, all from the same FlyingThings pre-training, baselines re-run by the authors | Do not propose this as a novel experimental design. An **OSLO row** in that design is what is missing |

---

## 2. Claims of ours that are dead, and must never be revived

| dead claim | why | replacement |
| --- | --- | --- |
| "No published 360° method beats zero-flow" | the FLOW360 forward-flow convention was inverted; on corrected data frozen RAFT-large scores **+39.3%**, PanoFlow **+29.8%**, OSLO **+20.4%** | *No published 360° flow paper **reports** a trivial baseline.* A statement about practice, not about performance |
| "Gate R2 approached but not met (+4.5 vs +5.2, 86%)" | same defect; the corrected retrain reaches act₀.₅ **+60.4**, passing by 11.6× | The gate passes. The P1 campaign's *conclusions* are void, not its instruments |
| "The sub-pixel regime is unsolved by anyone" | rests entirely on the void table | Scope to what the corrected table shows, which is much weaker |
| "The SO(3) rotation protocol, which ERP methods structurally cannot win" (`THESIS_REGIME_ARGUMENT.md` §5) | **refuted by our own measurement**: under Haar rotation on flowscape:test, frozen RAFT-large degrades **6.3%**, OSLO **29.3%**, PanoFlow **313.9%**. The ERP method is the *most* robust of the three | Orientation sensitivity tracks **in-domain training**, not representation |
| "Spherical geometry buys polar accuracy and uniformity, replicated on two independent datasets" | the flowscape half is placement, not geometry: that dataset's zero baseline has poles/equator **4.47**; under Haar rotation RAFT's polar tax falls 4.19 → 1.10 and OSLO's only 2.72 → 1.36 | One dataset (replica360, natively uniform at 1.05), plus a **displacement threshold**: the advantage needs polar ERP displacement above the raster method's correlation reach (~32 px) |
| "The decisive triangle" | retired 2026-07-29 as confounded (field and appearance swapped together), and its numbers are void anyway | the P0d six-cell decomposition, itself void on flow360 |

---

## 3. The one piece of prior art that argues against us

PriOr-Flow's Table 5 includes **SphereNet weight transformation applied to RAFT**
— the distortion-aware spherical-convolution family, the closest published
relative of this thesis' premise. Under a matched backbone and matched
pre-training it is **the worst row in the table**: 13.2 EPE on MPFDataset-EFT and
8.28 on City, against PriOr-RAFT's 3.30 and 1.13.

This does not refute our replica360 result, which concerns a different mechanism
(estimation grid, not adapted convolution kernels) and a displacement regime
four times past the raster method's search reach. **But it must be cited and
engaged in the related-work section.** A committee will find it, and finding it
first is the difference between a scoped claim and an embarrassment.

---

## 4. Findings that survive the check

Stated at the width the evidence supports, no wider:

1. **The FLOW360 convention defect**, and the methodological point that neither a
   sign-invariant baseline nor an exactly reproduced published table detects it.
2. **360°-flow papers do not report trivial baselines** — six-paper full-text sweep.
3. **In-domain benchmark performance is orientation-contingent**: PanoFlow loses
   314% of its margin under an isometry that preserves every motion statistic and
   falls behind an out-of-domain frozen RAFT-large that loses 6.3%. The control
   exists in panoramic depth and segmentation; **applying it to flow, and using
   in-domain vs out-of-domain as the discriminator, is ours**.
4. **The polar penalty is a threshold, not a slope** — nil while polar ERP
   displacement fits the correlation reach (~32 px), severe past it. Measured at
   1.5 px, 27.4 px and 132 px.
5. **The equal-area grid's polar advantage, scoped to that threshold**, on one
   content-controlled dataset.
6. **EPE-versus-SEPE polar inflation**: in PriOr-Flow's own table the polar gap is
   5× larger in pixels than in geodesic units (PanoFlow 12.0 vs 2.33). Both
   metrics are published; the comparison between them is not.

---

## 5. Maintenance

Add a row whenever a literature check resolves a claim, in either direction.
Related: `COMPARISON_PARTNERS.md` (who we can compare against and on which
splits), `UNIVERSALITY_TABLE.md` §16.25 onward (the convention defect and every
control run since).
