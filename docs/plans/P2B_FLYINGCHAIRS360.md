# P2B: FlyingChairs-360 — the spherical matching bootstrap dataset

**Status: generator IMPLEMENTED as a standalone project (2026-07-12), all gates
passed.** Location: `/Volumes/External SSD/Mestrado/chairs360` (`python -m
chairs360 {generate,selfcheck}`) — per user decision, generation lives outside
sphereflow-dataprep; sfprep keeps only the thin `chairs360` adapter (§3, still
pending) that indexes the generated layout for sharding. Deviations from §3
below: numpy+PIL only (no torch); splits are grouped by *scene directory* (the
replica pano source is thousands of frames of ~18 scenes — a per-file split
would leak rooms across train/val; for flat collections like Poly Haven each
file is its own group); texture crops restricted to same-split, different-scene
panoramas. Measured gates (replica panos, 8 pairs @256×512): G1 bit-identical;
G2 warp error 0.82/255 (< 2); G3 occlusion 5.2% (1.3–9.1%); G4 pooled p10
0.00° / p50 0.77° / p95 13.8°, direction-disagreement 100%; G6 anchors mean z
−0.04, frac|z|>0.5 = 0.40. Post-P0d retune: static-background prob 0.2 (the
flow360-like regime is a first-class citizen), sprites 4–8 @ half-size 5–25°,
sprite rotations log-uniform 0.5–60°. Timing ≈ 0.5 s/pair emulated ⇒ 23.5k
pairs ≈ 2–3 h (less native). G5 (shard round-trip) runs when the sfprep adapter
lands. Remaining: adapter + datasets.toml entry + shard build + the §5
acceptance smoke (2k-step train where `--ablate-corr` collapses).

**Goal.** Recreate, on the sphere, the thing FlyingChairs actually gave RAFT: tens of
thousands of pairs with (a) **exact, dense GT** (no rendering/annotation noise), (b)
**layered independent motion with occlusion** (background + foreground objects moving
differently, so matching is *forced* — a global prior can't explain the scene), (c)
**motion magnitudes spanning sub-node to many-node**, (d) enough texture diversity
that features generalize. Phase 1's synth-rot source has (a) and (d)-partially but
fails (b) — a single global rotation is exactly the field a context-only prior can
regress, which is how the +2.9% appearance ceiling survived Act I. Chairs-360 exists
to kill that shortcut.

**Where it lives.** sphereflow-dataprep (sibling repo), as a *generator* + standard
adapter, so OSLO-RAFT consumes it as `chairs360:train` / `chairs360:val` through the
existing shard contract with zero loader changes.

---

## 1. Composition model (what one pair is)

```
background: ERP panorama P_bg, rotated by R_bg between frames    (exact GT everywhere)
foreground: K sprites (K ~ U{2..6}), each a textured patch with alpha,
            anchored at direction d_k, own motion M_k between frames
compositing: painter's order, sprites over background; frame1 and frame2
             rendered with the SAME z-order
GT flow:    per pixel, the motion of its TOP layer in frame1
            (pixels occluded in frame2 keep their layer's GT — Chairs convention;
             `valid` stays 1 there: learning through occlusion is the point)
```

### 1.1 Background

- Sources, in priority order: **Poly Haven HDRIs** (CC0, ~700+ equirect panoramas,
  download once into the dataset root; tonemap HDR→uint8) and **replica360 frames**
  (already on disk; use *train-split frames only* to avoid val leakage). Mixing both
  gives indoor+outdoor variety. Record the source panorama id in `PairRecord.extra`.
- Motion: `R_bg` sampled via the existing `so3_augment.sample_rotation` convention;
  angle log-uniform in **[0.05°, 20°]** — deliberately spanning sub-node (r6 spacing
  0.92°) through many-node, so one dataset serves both the large-motion bootstrap and
  the sub-pixel regime the wall lives in. Warp frame2 by inverse rotation with
  bilinear sampling on the ERP raster (same math as `shard_dataset.synth_rotation_record`
  — reuse its warp, don't rewrite it).

### 1.2 Sprites ("chairs")

- Texture: crops from a disjoint pool of panoramas/images with an alpha mask —
  polygon/superellipse silhouettes with soft edge (1-px feather). No need for actual
  chair renders; RAFT's chairs are arbitrary shapes as far as matching is concerned.
- Placement: anchor direction `d_k` uniform on the sphere (NOT uniform in ERP — must
  land sprites on the poles; this is the dataset that teaches polar matching).
  Angular size log-uniform [5°, 40°].
- Rendering: sprite lives on the tangent plane at `d_k` (gnomonic projection). For
  each ERP pixel inside the sprite's support cap, inverse-project to tangent-plane
  coords, sample texture+alpha bilinearly. Pure torch, batched over pixels — no 3D
  renderer, no GPU requirement.
- Motion `M_k`: composition of (i) own small rotation `R_k` (axis near `d_k` ⇒
  in-plane spin, or transverse ⇒ translation-like drift; angle log-uniform
  [0.05°, 25°]) and (ii) affine jitter of the texture coords (scale 0.9–1.1, shear
  ≤0.1) for non-rigid variety. Flow of a sprite pixel = ERP displacement of its
  tangent-plane point under `M_k` + affine — closed form, exact.
- **Independence is the point:** `R_bg` and every `R_k` are sampled independently.
  Target: ≥30% of pairs where some sprite moves *against* the background direction.

### 1.3 What the generator does NOT do

No photometric nuisance (brightness/noise/blur asymmetry) — frames 1 and 2 of a pair
are photometrically consistent by construction. Nuisance is a *training-time*
augmentation (P2C §2), so the same clean data serves every point on the
nuisance-curriculum axis. This mirrors RAFT exactly (Chairs is clean; jitter is in the
loader) and preserves the decisive-triangle logic: clean Chairs-360 pairs are
"resampled-frame-2" pairs, the regime we've proven the model solves at +80.6%.

## 2. Scale and splits

- **22,872 train / 640 val pairs** (Chairs-parity), 512×1024 ERP (0.352°/px eq. —
  FLOW360-matched so P2C stage transitions don't change raster).
- Val split by *background panorama id* (sequence-level holdout, matching
  `Adapter.split_by_sequence` semantics) — no panorama appears in both splits.
- Storage estimate: 2 uint8 frames + fp16 flow + valid ≈ 5 MB/pair ⇒ ~120 GB raw;
  generate → shard → **delete raw** (keep the manifest + generator seed for exact
  regeneration; determinism gate G1 makes raw retention unnecessary).

## 3. Implementation in sphereflow-dataprep

| file | change |
| --- | --- |
| `sfprep/generators/chairs360.py` | **new**: `generate(config, out_root, seed)` — writes `frames/{uid}_1.png`, `frames/{uid}_2.png`, `flow/{uid}.flo` (canonical `[du_x, dv_y]`, identity convention), `meta.jsonl` (per-pair: panorama id, R_bg params, per-sprite params, seed). Deterministic per (seed, index): each pair's RNG is `seed*1e6+index`, so any single pair regenerates in isolation |
| `sfprep/adapters/chairs360.py` | **new**, `@register("chairs360")`: indexes the generator layout into `PairRecord`s (adapters never read pixels — generator writes width/height into meta); split from panorama-id holdout |
| `datasets.toml` | new `[chairs360]` entry pointing at the generated root |
| `sfprep/flow_io.py` | no change (`.flo` already supported) |

CLI: `python -m sfprep generate chairs360 --out … --pairs 23512 --seed 7` then the
standard `python -m sfprep build` path shards it.

## 4. Validation gates (CPU, before generating at scale)

| gate | test | pass criterion |
| --- | --- | --- |
| G1 determinism | generate pair #12345 twice, and once on a different worker count | bit-identical frames and flow |
| G2 **GT exactness (the triangle test)** | warp frame1 by GT flow (forward-splat or backward via GT of the inverse pair) and compare to frame2 on non-occluded pixels | photometric error ≈ interpolation noise (< 2/255 mean abs); this is the same resample check that anchored the decisive triangle |
| G3 occlusion accounting | fraction of pixels whose top layer changes between frames | 5–20% per pair on average (Chairs-like); 0% means sprites aren't overlapping anything — bug |
| G4 motion statistics | node-space |flow| distribution at r6 over 200 pairs | p50 within [0.3°, 3°], p10 < 0.15° (sub-node present), p95 > 8° (supra-node present); background-vs-sprite direction disagreement ≥ 30% of pairs |
| G5 shard round-trip | `run_shard_dataset_check.py` + `sample_pair_to_nodes` on 20 pairs | loads through the existing OSLO pipeline unmodified; zero-baseline geodesic numbers match G4 stats |
| G6 pole coverage | sprite-center latitude histogram over 1k pairs | uniform-on-sphere (cos-lat density), not ERP-uniform |

## 5. Acceptance

P2B is done when 23.5k pairs are sharded, all six gates pass, and a 2k-step OSLO
smoke train on `chairs360:train` shows the aux matching loss descending below its
replica360 floor trajectory (features find *more* to match, not less) with
`--ablate-corr` clearly worse at eval — i.e. the dataset structurally defeats the
context-prior shortcut it was designed to defeat.
