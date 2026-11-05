# 360° optical flow methods and spherical operators for flow on HEALPix

Scope note: Facts carry inline links. When a link comes from my prior knowledge (an arXiv ID or venue page that I recognise but did not re-open in this session), it is tagged **[link from memory, not re-verified]**. Every "Inferences" bullet is my own assessment. Session date: 2026-11-04. I ran about 15 search/fetch calls, so coverage of 2026 preprints is thin. Gaps are listed explicitly.

---

## Q1. Published 360° flow methods: representation, backbone, key trick, numbers, code

### Takeaway
Every competitive published 360° flow method is a **planar RAFT-family network running on ERP, plus a cheap geometric patch**: PanoFlow adds cyclic half-width shifting, ERP-distortion augmentation and deformable convolutions; PriOr-Flow adds a second branch on a 90°-rotated ERP, with the cost volumes fused. None of them runs natively on a sphere grid. On FlowScape the ranking is PriOr-RAFT 2.33 < PanoFlow-CSFlow 3.31 / PanoFlow-RAFT 3.38 ≪ SLOF 7.59 < SphereNet+RAFT 12.9 < TanImg+RAFT 18.3 EPE. The truly "spherical" baselines (SphereNet kernels, tangent images) are the worst rows.

### Cited Findings

**PriOr-Flow (Liu et al., ICCV 2025)**
- Representation is ERP. A second "orthogonal view" is made by rotating the sphere 90° about the x-axis and re-projecting to ERP with bilinear interpolation, which gives that branch low distortion at the original poles — [arXiv 2506.23897 HTML](https://arxiv.org/html/2506.23897v3); [ICCV open access](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PriOr-Flow_Enhancing_Primitive_Panoramic_Optical_Flow_with_Orthogonal_View_ICCV_2025_paper.html)
- Backbone is RAFT: two branches, each with its own feature encoder, 4-level correlation pyramid and ConvGRU. **DCCL** (Dual-Cost Collaborative Lookup) uses the current flow to sample correlations from both pyramids, mapping the orthogonal samples back to primitive coordinates. **ODDC** (Ortho-Driven Distortion Compensation) converts the orthogonal flow to primitive format, computes group-wise correlation confidences for both flows and fuses them in the motion encoder before the primitive GRU predicts the residual. It also works on GMA and SKFlow backbones — [arXiv HTML](https://arxiv.org/html/2506.23897v3)
- Training: initialised from RAFT trained on FlyingThings; AdamW with one-cycle LR at 1e-4; FlowScape batch 6 for 100k steps (following PanoFlow); 12 iterations; **spherical-area-weighted L1** with γ = 0.8 on both branches; RTX 3090 — [arXiv HTML](https://arxiv.org/html/2506.23897v3)
- FlowScape test, EPE / SEPE (All): PriOr-RAFT **2.33 / 3.49**, PanoFlow-RAFT 3.38 / 4.78, PanoFlow-CSFlow 3.31 / 4.44, SLOF 7.59 / 5.79, SphereNet+RAFT 12.9 / 21.0, TanImg+RAFT 18.3 / 25.3, TanImg+DIS 18.7 / 29.9 — [arXiv HTML, Table 6](https://arxiv.org/html/2506.23897v3)
- Polar vs equatorial EPE on FlowScape: PanoFlow equator 0.52 / poles 6.25; PriOr-RAFT equator 0.53 / poles 4.13. **The whole gain is polar; the equator ties** — [arXiv HTML, Table 7](https://arxiv.org/html/2506.23897v3)
- Ablation on MPF-EFT: baseline RAFT 4.49 EPE (poles 7.90) → +ortho view and DCCL 4.32 (poles 7.56) → +ODDC 3.30 (poles 5.57). **ODDC, the feature-level fusion, gives most of the gain, not the extra view on its own** — [arXiv HTML, Table 1](https://arxiv.org/html/2506.23897v3)
- Runtime on EFT: RAFT 0.07 s vs PriOr-RAFT 0.20 s (0.10 s at 4 iterations) — [arXiv HTML, Table 3](https://arxiv.org/html/2506.23897v3)
- Code: [github.com/longliangLiu/PriOr-Flow](https://github.com/longliangLiu/PriOr-Flow) — [arXiv HTML](https://arxiv.org/html/2506.23897v3)

**PanoFlow (Shi et al., IEEE T-ITS; arXiv 2202.13388)**
- **CFE (Cyclic Flow Estimation):** encoder features of both frames are split down the vertical centreline into halves a and b and regrouped as {a⊕b} and {b⊕a}, i.e. shifted by half the width; context features are regrouped the same way. The decoder produces two full-width flows V and V′, and each half takes the *shorter* of its two complementary flows: V̂ = min(Va, V′a) ⊕ min(V′b, Vb) — [ar5iv 2202.13388](https://ar5iv.labs.arxiv.org/html/2202.13388)
- **FDA (Flow Distortion Augmentation):** FDA-R is radial-polynomial distortion; FDA-E maps to the sphere, rotates (θy ~ U(0, π), θz ~ U(0, 2π)) and re-projects to a perspective view with FoV ~ U(π/3, 2π/3). The flow GT is warped consistently — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13388)
- **DCN:** the first encoder layer and the context encoder are replaced by 7×7 modulated deformable convolutions — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13388)
- Ablation on FlowScape with RAFT trained on Chairs: baseline 17.43; FDA-E 15.75; CFE alone 15.62 (the largest single gain); DCN alone 17.08; full model 13.75. CFE variants after fine-tuning: baseline 4.50; **circular convolution 5.59 (worse than baseline)**; double estimation 3.58; half zero-padding 3.59; default CFE 3.36. Latency 0.13 s vs 0.10 s — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13388)
- FlowScape is built in CARLA from 6×90° pinhole cameras re-projected to ERP: 6,400 frames at 512×1024, 5,000 train / 1,400 test, 4 weather conditions. Headline result: CSFlow + FDA-E fine-tuned **3.25 px** (from 4.47). On the OmniFlowNet dataset: 3.17 px vs 7.12 for the prior best. The paper's intro says 3.34, which conflicts with its Table VII — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13388)
- Code: [github.com/MasterHow/PanoFlow](https://github.com/MasterHow/PanoFlow) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.13388)

**SLOF and the FLOW360 dataset (Bhandari et al., ECCV 2022)**
- FLOW360 is a synthetic 360° flow benchmark of 40 videos and 4,000 frames. SLOF is a Siamese framework that feeds consecutive frames and their rotated counterparts and trains with contrastive + flow losses, exploiting the rotation invariance of 360° video. The authors claim up to 40% gains over prior work. Code and data at siamlof.github.io — [ECCV 2022 page](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7090_ECCV_2022_paper.php); [arXiv 2208.03620](https://arxiv.org/abs/2208.03620v1)
- There is an extended journal version ("Optical Flow Estimation in 360° Videos") — [ar5iv 2301.11880](https://ar5iv.labs.arxiv.org/html/2301.11880)
- On FlowScape, SLOF (RAFT) scores 7.59 EPE but 5.79 SEPE, the only method whose SEPE is *below* its EPE — [PriOr-Flow Table 6](https://arxiv.org/html/2506.23897v3)

**Older and baseline methods (as characterised in PriOr-Flow's related work)**
- LiteFlowNet360: the first deep dense 360° flow, with kernel-transformer adaptation and self-supervision. OmniFlowNet: distortion-aware convolutions on perspective CNNs. TanImg: tangent-plane decomposition. MPF-Net: fusion of several projections (tri-cylinder, cube padding) on PWC-Net, MPF-All 3.42 EPE vs PriOr 2.22. SphereNet: spherical kernel warping, used as a RAFT baseline — [PriOr-Flow related work and Table 6](https://arxiv.org/html/2506.23897v3)
- TanImg paper — [arXiv 2112.14331](https://arxiv.org/pdf/2112.14331); OmniFlow (synthetic human omnidirectional flow dataset) — [arXiv 2104.07960](https://arxiv.org/pdf/2104.07960); [IEEE](https://ieeexplore.ieee.org/document/9523053/)
- LiteFlowNet360 — arXiv 2010.08045 **[link from memory, not re-verified]**; MPF-Net / MPFDataset (Li et al., ECCV 2022 "Deep 360° optical flow estimation based on multi-projection fusion") — arXiv 2208.00776 **[link from memory, not re-verified]**

**2025–2026 newcomers found**
- **SVI360: Spherical Video Interpolation** (Nguyen, Martins, Vasseur, Demonceaux; ECCV 2026; arXiv July 2026). A dual-branch design with the original and a rotated orthogonal view that strengthens equivariance of flow displacements between the two views, used for frame interpolation. Builds on the PriOr-Flow idea. Code at icb-vision-ai.github.io/video360_interpolation. The abstract page gives no numbers — [arXiv 2607.11710](https://arxiv.org/abs/2607.11710)
- PanFlow (arXiv 2512.00832) is *generative* (panoramic video generation with flow decomposed into rotation + residual), not an estimator — [arXiv 2512.00832](https://arxiv.org/html/2512.00832)
- A 2026 Frontiers paper on event-camera "omnidirectional" flow decomposes motion into forward, backward, scaling and rotation components (MVSEC 0.67, DSEC 0.63 EPE). It is not 360°-panoramic in the ERP sense — [Frontiers Neurorobotics 2026](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1884626/xml)
- A phase-based omnidirectional flow method using spherical wavelets (classical, not deep) — [academia.edu](https://www.academia.edu/121203562/A_phase_based_framework_for_optical_flow_estimation_on_omnidirectional_images)
- A 2024 survey notes that comparisons between spherical flow estimators "are still scarce" — [arXiv 2401.09252](https://arxiv.org/pdf/2401.09252)

### Inferences
- **What makes them win:** (a) they keep the full-resolution planar RAFT machinery: 1/8-resolution correlation at 512×1024 ERP, about 0.35°/px at the equator, convex upsampling, and FlyingThings pretraining. (b) They spend their geometric effort on **two specific failure modes, the seam and the poles**. CFE fixes the seam; the orthogonal view fixes the poles. Neither tries to make the network globally equivariant.
- PanoFlow's own Table VI shows circular convolution is *worse* than baseline (5.59 vs 4.50), while half-shift double estimation helps (3.58). My reading: decoding twice with different seam placements and picking the confident answer beats making the operator seam-free. That is an ensembling/selection effect, which fits the OSLO-RAFT finding that "sphere-native operators" are not the lever.
- PriOr-Flow's ablation (view+DCCL gives only −0.17; ODDC gives −1.0) suggests the gain comes from **fusing cost/motion features from two charts with confidence**, not merely from having a low-distortion chart. On HEALPix the analogue would be fusing two *rotated HEALPix samplings*. HEALPix has no "bad pole", but OSLO's error is still polar (an orientation/frame issue, not area distortion).
- The matched comparison against OSLO-RAFT should be made on SEPE, not only ERP-px EPE. SLOF shows the two metrics can reorder methods.
- PriOr-Flow and SVI360 both use a "90°-rotated companion view". This is a cheap, architecture-agnostic trick that OSLO could adopt directly: two HEALPix samplings related by a 90° rotation, with the cost volumes fused.

### Gaps
- I did not verify per-method FLOW360 numbers from the SLOF paper, or OmniFlowNet/LiteFlowNet360 architecture details, from primary sources in this session.
- No 2026 panoramic flow *estimator* with FlowScape numbers turned up beyond PriOr-Flow. SVI360's flow numbers are unknown.
- I found no "omnidirectional flow transformer" with FlowScape results, but the search was shallow.

---

## Q2. Spherical operators that could replace OSLO's SDPA convolutions

### Takeaway
Operators fall into three families. **(i) Isotropic graph/Chebyshev filters** (DeepSphere): cheap and rotation-equivariant, but they cannot represent oriented filters, which flow needs. **(ii) Continuous-filter local convolutions** (DISCO): approximately SO(3)-equivariant, linear cost, mature GPU code in NVIDIA torch-harmonics, but they need a fixed tangent frame. **(iii) Gauge-equivariant / spin-weighted operators**: the only family that handles tangent-vector (flow) fields correctly, but they are expensive or restricted to icosahedra or spectral grids. Separately, **the DLWP-HPX face-padding approach** gives the cheapest path to standard 2D convolutions on HEALPix and fits OSLO's NESTED quadtree exactly.

### Cited Findings
- **DISCO** (Ocampo, Price, McEwen, ICLR 2023) is a hybrid discrete-continuous group convolution: SO(3)-equivariant, with an asymptotic SO(3)/SO(2) property. A sparse-tensor implementation scales linearly with pixel count. The authors report savings of 10⁹ in compute and 10⁴ in memory at 4k vs the most efficient equivariant alternative, and state-of-the-art results on spherical segmentation and depth — [arXiv 2209.13603](https://arxiv.org/abs/2209.13603); [ICLR poster](https://iclr.cc/virtual/2023/poster/11399)
- NVIDIA **torch-harmonics** ships `DiscreteContinuousConvS2` (local DISCO). The continuous filter makes it approximately SO(3)-equivariant, and quadrature makes it scale. It is used in local neural operators (Liu-Schiaffini, Berner, Bonev et al., ICML 2024) on structured and unstructured spherical meshes. The library also underlies SFNO (Bonev et al., ICML 2023) and ships SHT, DISCO and spherical-attention benchmarks — [torch-harmonics docs](https://nvidia.github.io/torch-harmonics/); [DISCO guide](https://nvidia.github.io/torch-harmonics/guide/disco_convolutions.html)
- **Gauge Equivariant CNNs and the Icosahedral CNN** (Cohen et al., ICML 2019) extends equivariance to local gauge (frame) changes and implements it on the icosahedron as a sphere approximation — [PMLR v97](https://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf)
- **Gauge Equivariant Mesh CNNs** (de Haan et al., ICLR 2021) give anisotropic, gauge-equivariant kernels on general meshes, with parallel transport of features between neighbours — [arXiv 2003.05425](https://arxiv.org/pdf/2003.05425)
- **Spin-Weighted Spherical CNNs** (Esteves et al., NeurIPS 2020) handle spin-weighted functions, so they can map scalar fields to vector fields and back on the sphere; tangent vectors are spin ±1 components. This is the closest published match to "predict a tangent field on the sphere" — [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/6217b2f7e4634fa665d31d3b4df81b56-Paper.pdf)
- **Coordinate-independent CNNs** (Weiler et al.) is the general theory of isometry- and gauge-equivariant convolutions on Riemannian manifolds — [arXiv 2106.06020](https://arxiv.org/pdf/2106.06020)
- Gauge-equivariant CNNs have been applied to diffusion MRI, i.e. orientation-valued data on the sphere — [PMC 11926199](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11926199/)
- **OSLO** (Bidgoli et al., IEEE TIP 2022) redefines convolution, stride and pixel shuffle on HEALPix for 360° compression — [DOI 10.1109/tip.2022.3202357](https://doi.org/10.1109/tip.2022.3202357)
- DeepSphere is a HEALPix graph CNN; Krachmalnicoff & Tomasi apply CNNs directly to HEALPix maps (cited in) — [core.ac.uk record](https://core.ac.uk/works/54166887)
- The DeepSphere papers (arXiv 1810.12186; ICLR 2020 "DeepSphere: a graph-based spherical CNN", arXiv 2012.15000), Cohen et al. "Spherical CNNs" (ICLR 2018, arXiv 1801.10130), SphereNet (Coors et al., ECCV 2018), UGSCNN (Jiang et al., ICLR 2019, arXiv 1901.02039), SpherePHD (Lee et al., CVPR 2019, arXiv 1811.08196), HexRUNet / orientation-aware icosahedral segmentation (Zhang et al., ICCV 2019, arXiv 1907.12849), Clebsch-Gordan nets (Kondor et al., NeurIPS 2018, arXiv 1806.09231) and SFNO (Bonev et al., ICML 2023, arXiv 2306.03838) — **[links from memory, not re-verified]**
- Of these, only SphereNet has been tested inside a flow network on FlowScape: SphereNet+RAFT 12.9 EPE vs plain-ERP PanoFlow-RAFT 3.38 — [PriOr-Flow Table 6](https://arxiv.org/html/2506.23897v3)

### Inferences (operator-by-operator for flow, my assessment)
| Operator | Rot. equivariance | Tangent-vector correctness | Cost at nside 64–256 | Fit for OSLO flow |
|---|---|---|---|---|
| DeepSphere Chebyshev (isotropic) | yes (isotropic) | no: cannot orient | very cheap | poor; flow needs oriented filters (edges, aperture) |
| OSLO SDPA (8-nbr sorted by angle) | approx. (local frame) | frame not transported (measured holonomy ≤ 14°) | cheap | current baseline; slot-orientation bug found |
| DISCO (torch-harmonics) | approx. SO(3) | uses a fixed east/north frame, so vectors are still gauge-dependent | linear, GPU kernels exist | **best drop-in candidate**; supports arbitrary grids including HEALPix through quadrature (worth verifying) |
| Gauge-equivariant mesh CNN | gauge + approx. SO(3) | **yes** (parallel transport) | moderate (irreps, transporters) | principled for flow; no flow evidence |
| Spin-weighted spherical CNN | exact SO(3) | **yes** (spin-1 output) | spectral, so heavy at high bandwidth | good for the flow *head* only, at low res |
| S2CNN / CG-nets / SFNO global | exact | spectral only | O(L³) or worse; global, not local | poor for local correspondence; SFNO suits smooth global fields |
| DLWP-HPX face-padded 2D conv | none (fixed per-face frame) | no, but east is consistent across faces | **cheapest**; cuDNN | best engineering lever (see Q3) |

- The memory notes say OSLO's measured holonomy damage is about 3% of field magnitude and not load-bearing. So **gauge-equivariant operators are unlikely to close the 6.3 vs 8.6 px gap with planar RAFT-small**. Their value would be a principled formulation the advisor might like, not an accuracy win. Expect parity at best.
- Rotation equivariance is already bought by SO(3) augmentation (the project's main lever). Exactly equivariant operators would mostly save augmentation epochs.

### Gaps
- No published work applies gauge-equivariant or spin-weighted networks to optical flow or image motion. The search found none, which matches the absence of such rows in PriOr-Flow's tables.
- I did not verify whether torch-harmonics DISCO accepts a HEALPix grid natively (its documented grids are equiangular/Gauss-Legendre; unstructured-mesh support via local neural operators is mentioned). Check this before planning.
- Unverified: the exact cost at nside 64–256 for each operator.

---

## Q3. DLWP-HPX: 12 HEALPix faces as images with neighbour padding — could an off-the-shelf planar flow network run natively on HEALPix?

### Takeaway
This is published, mature for weather, and comes with code including a CUDA padding kernel. It treats the 12 base faces as nside×nside images, pads each face from its neighbours (with rotations at the polar faces and averaging for the 2 missing corners of the equatorial faces), and then uses ordinary Conv2d. I found **no vision or optical-flow work that does this**, so running RAFT/SEA-RAFT on HEALPix faces would be novel. The conv and pooling parts port directly; the correlation lookup is the part that needs a sphere-aware rewrite.

### Cited Findings
- Karlbauer et al. (JAMES 2024; arXiv 2311.06253) "treat the twelve faces as distinct images" and pad each face's boundaries with neighbour-face data so standard 2D convolutions run per face. Faces are grouped as 4 northern, 4 equatorial and 4 southern; northern and southern neighbours sometimes need rotation. Example: equatorial face E3 gets N2 on the left and S3 on the right — [ar5iv 2311.06253](https://ar5iv.labs.arxiv.org/html/2311.06253)
- The tropical faces have north and south corners with no natural neighbour. The corner value is the average of the two adjacent N (or S) face values. Padding must not exceed one neighbouring face, which bounds kernel size × dilation by nside — [ar5iv 2311.06253](https://ar5iv.labs.arxiv.org/html/2311.06253); [arXiv PDF](https://arxiv.org/pdf/2311.06253)
- There is a custom CUDA padding kernel. The final model runs at HPX64 (≈110 km). Code: [CognitiveModeling/dlwp-hpx](https://github.com/CognitiveModeling/dlwp-hpx) and [NVIDIA modulus dlwp_healpix](https://github.com/NVIDIA/modulus/tree/main/examples/weather/dlwp_healpix) — [ar5iv 2311.06253](https://ar5iv.labs.arxiv.org/html/2311.06253)
- The claimed advantage over the cubed sphere is "east to the right", a consistent orientation across all cells, so one set of location-invariant kernels serves the globe. The north–south consistency holds only equatorward of the equatorial faces' maximum extent. On the cubed sphere, eastward maps to any of 4 stencil directions on polar faces. The authors offer this as "a likely explanation", not a proof — [ar5iv 2311.06253](https://ar5iv.labs.arxiv.org/html/2311.06253)
- Quantitatively, cubed sphere 64² per face → HPX32 (half the points) delays the Z500 RMSE = 50 m crossing by about 0.5 day. The Table 3 comparison with Weyn 2021 is confounded by other changes — [ar5iv 2311.06253](https://ar5iv.labs.arxiv.org/html/2311.06253)
- The approach is in NVIDIA PhysicsNeMo as `HEALPixRecUNet` — [PhysicsNeMo source](https://docs.nvidia.com/physicsnemo/latest/_modules/physicsnemo/models/dlwp_healpix/HEALPixRecUNet.html); [weather models API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/weather.html)
- It was reused in 2026: HealDA exchanges halo cells between faces by HEALPix connectivity — [arXiv 2601.17636](https://arxiv.org/pdf/2601.17636). Aircast-Mars has its own HEALPixPadding layer that borrows strips by the adjacency graph with zone-specific rotations, and argues this avoids filters seeing zero- or reflect-padded values — [arXiv 2607.19370](https://arxiv.org/pdf/2607.19370)
- In 360° vision the related, published practice is **cube padding** on cubemap faces (e.g. SegFuse/Depth360) and **circular horizontal padding** on ERP (Helvipad stereo depth). No HEALPix-face vision paper was found — [Depth360 arXiv 2202.08010](https://arxiv.org/pdf/2202.08010); [Helvipad arXiv 2411.18335](https://arxiv.org/html/2411.18335)
- Related evidence from flow: cube padding is a component of MPF-Net, at 3.42 EPE on MPF-All vs 2.22 for PriOr-Flow — [PriOr-Flow](https://arxiv.org/html/2506.23897v3)

### Inferences (my assessment of porting RAFT/SEA-RAFT to HEALPix faces)
- **Fit with OSLO:** in NESTED ordering each base face is a Z-order quadtree, so de-interleaving the index bits gives an nside×nside face image. OSLO's exact NESTED quad-tree pooling then becomes a standard `AvgPool2d(2)` on the face image. Pooling, the encoders, ConvGRU and convex upsampling all port unchanged. Batch layout would be [B·12, C, nside, nside] plus a pad op before every k > 1 conv.
- **Orientation:** face-image axes run along the diamond edges, i.e. about ±45° from east/north (my understanding of HEALPix geometry, not verified in a source this session). "East to the right" in DLWP means *consistent* orientation across faces, not that it lies along the axes. Flow output therefore needs a fixed per-face 45° rotation (plus the polar-face rotations) into east/north. This also hints at a possible root cause for the SDPAConv 45° hemisphere slot flip already found in OSLO.
- **Correlation:** all-pairs correlation is global, so it is unaffected (dot products over all 12·nside² features). **The lookup is the problem**: RAFT's `grid_sample` of a (2r+1)² window around the current target lives in one face's coordinates and breaks when a window crosses a face edge or a polar corner. Options: (a) look up on padded face pyramids (works when the window radius ≤ pad width; corners are approximate); (b) keep OSLO's existing spherical lookup, which turns the hybrid into "planar convs + spherical correlation"; (c) SEA-RAFT-style direct regression avoids little here, since lookup is still needed.
- **Pretraining transfer:** because face pixels are roughly square and near-isometric, **FlyingThings-pretrained RAFT weights could load directly** into the face network. PriOr-Flow and PanoFlow both rely on Things init, while OSLO trains from scratch. This may be the single biggest practical lever the face formulation unlocks, and OSLO's graph convolutions cannot use it.
- **Resolution:** a FlowScape ERP 512×1024 has 524k px. HEALPix nside 256 has 12·65,536 = 786k px. At RAFT's 1/8 feature stride that gives a 32×32 face grid (≈1.8°/node), vs OSLO's r4 estimation grid at 3.66°. The face formulation makes it cheap to run correlation at nside 32–64 instead of r4 = nside 16.
- **Kernel-size limit:** padding must stay within one face. With nside 32 at the 1/8 level, 7×7 convolutions and dilations are fine; at nside 8 (coarse pyramid levels) a radius-4 lookup approaches the limit.

### Gaps
- No vision or flow work using HEALPix-face padding was found. This is a novelty opportunity, but also means there is no evidence that it works.
- Unverified: whether the dlwp-hpx CUDA pad kernel supports autograd at the channel counts RAFT needs, and how fast it is at nside 256.

---

## Q4. Evidence that tangent-vector-field-aware (gauge-equivariant) networks improve flow or motion estimation

### Takeaway
**I found no published evidence** that gauge-equivariant or spin-weighted networks improve optical flow or image motion estimation. The theory and tooling exist (Cohen 2019, de Haan 2021, Esteves 2020, Weiler 2021), and they have been applied to diffusion MRI and to synthetic vector-field prediction tasks. The project's own measurement (holonomy damage about 3% of field magnitude) predicts little gain.

### Cited Findings
- Spin-weighted spherical CNNs explicitly support scalar↔vector field prediction on the sphere (spin-1 tangent components). Their experiments are not optical flow — [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/6217b2f7e4634fa665d31d3b4df81b56-Paper.pdf)
- Gauge-equivariant mesh CNNs provide anisotropic kernels with parallel transport between neighbours — [arXiv 2003.05425](https://arxiv.org/pdf/2003.05425)
- Gauge-equivariant CNNs applied to orientation data on the sphere (dMRI) — [PMC 11926199](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11926199/)
- A targeted search for gauge/vector-field spherical networks for flow estimation returned no flow application — [search over PMLR/ICLR/NeurIPS sources above](https://proceedings.mlr.press/v97/cohen19d/cohen19d.pdf)
- The best-performing 360° flow methods have no gauge treatment at all. They predict (u, v) in ERP pixel coordinates and get the polar gain from a rotated view (PriOr-Flow) — [arXiv 2506.23897](https://arxiv.org/html/2506.23897v3)
- A learned model does regress spherical camera rotation from ERP flow patterns (E-CNN), but on ERP, not gauge-equivariant — [U-Tokyo page](https://www.robot.t.u-tokyo.ac.jp/~pathak/research)

### Inferences
- In practice, the evidence says to **treat frame-correctness as a hygiene fix rather than a performance lever**. Fix the SDPA slot flip and transport the frame in the flow-feature path (a cheap rotation of the 2-vector channels by the connection angle), but do not expect it to close the gap with RAFT.
- A clean, formulation-level contribution the advisor might value is to **predict flow as a spin-1 field / rotation-vector field on the sphere**. On S², rigid camera rotation is a Killing field (ω × x), so a head could output ω (global, 3 DoF) plus a residual tangent field, much like PanFlow's rotation + residual decomposition ([arXiv 2512.00832](https://arxiv.org/html/2512.00832)). That would be gauge-free by construction. This is my proposal, not published flow work.

### Gaps
- No ablation anywhere isolates "vector-field-aware operator" vs "scalar-channel operator" for flow. That remains open and would be new.

---

## Q5. Minimal experiments, with success criteria (my proposals)

All experiments compare against the current matched pair on FlowScape (OSLO about 8.6 vs RAFT-small about 6.3 ERP px, same data and recipe, 2 seeds), using ERP-px EPE, SEPE and polar/equatorial strata (PriOr-Flow Table 7 convention). Where possible, **pre-register a ±0.3 px noise band** from the 2-seed spread.

1. **E1. HPX-face RAFT-small (DLWP padding + OSLO spherical lookup), from scratch.**
   - Replace SDPA convs with Conv2d on 12 padded faces. Keep OSLO's spherical correlation lookup and output head; rotate the flow to east/north per face.
   - Success: within 0.3 px of planar RAFT-small (≤ 6.6 px) at the same steps. If so, the operator (not sphere-nativeness) was OSLO's handicap.
   - Failure: still ≈ 8.6 px, which implicates the lookup or estimation grid instead.
   - Cost: pad op + reshaping; reuses dlwp-hpx code.
2. **E2. E1 + FlyingThings-initialised RAFT weights loaded into the face convs.**
   - Success: beats E1 by more than 0.5 px and approaches PanoFlow-RAFT (3.38 px reported with Things init and 100k steps). This tests whether *pretraining transfer*, which only the face formulation enables, explains much of the gap to the published methods.
3. **E3. Correlation resolution: the face network at nside 32 (≈1.8°) vs nside 16 (r4, 3.66°).**
   - Success: the small-motion stratum (<15°) and poles improve by more than 10% at nside 32. This addresses "OSLO's error is in small motions".
   - Memory notes say est r5 did not help OSLO (10.71 vs 10.31). A result that differs under the face network would point to the operator × resolution interaction.
4. **E4. Orthogonal-HEALPix dual branch (PriOr-style) on whichever network wins E1/E2.**
   - Use a second HEALPix sampling rotated 90° about x, then do DCCL-style dual lookup and confidence fusion.
   - Success: polar EPE drops by at least 20% with the equatorial stratum within noise (the PriOr pattern: poles 6.25 → 4.13, equator flat).
   - If HEALPix + SO(3) augmentation already make the poles uniform, the null result is itself informative: HEALPix removes PriOr's motivation.
5. **E5. DISCO (torch-harmonics) convs in place of SDPA, same RAFT-small layout.**
   - First check HEALPix grid support.
   - Success: beats SDPA by at least 0.5 px *and* rotated-vs-upright EPE gap ≤ SDPA's without SO(3) augmentation. This tests whether a principled, approximately equivariant local operator buys accuracy or only equivariance.
6. **E6 (formulation, optional). Rotation + residual head.**
   - Predict a global ω ∈ so(3) (Killing field ω × x) plus a residual spin-1 field, with GRU updates on the residual.
   - Success: lower EPE in rotation-dominated pairs and no loss on parallax-dominated pairs. This gives the advisor a "not tied to RAFT" spherical formulation with a clear physical prior.
- **Decision rule:** if E1/E2 close the gap, the thesis story becomes "HEALPix is fine; the graph operator and from-scratch training were the problem". If neither does, the remaining difference lies in lookup/upsampling resolution (E3), not the convolution family.

### Inferences
- Order by cost to information: E1 → E2 → E3 → E4. E5 and E6 are formulation explorations for the advisor and are lower-priority on expected accuracy.

### Gaps
- PanoFlow's and PriOr-Flow's step budgets (100k, batch 6, Things init) differ from OSLO's recipe, so E2's comparison to published numbers is still not compute-matched. The memory notes record a 60–75× budget gap.
