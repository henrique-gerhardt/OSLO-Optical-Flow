# Transformer / global-matching optical flow, dense matchers + interpolation, and their mapping onto HEALPix

Scope note: this session verified sources with ~15 web calls. Claims marked **[verified]** were checked against the source page during this session. Claims marked **[recall]** come from my prior knowledge of the paper (arXiv ID given). The writer should treat [recall] numbers as needing a spot-check before they appear in the thesis. Every "Inferences" bullet is my own assessment, not a published result.

## Q1. Global-matching and transformer optical flow (GMFlow → 2026 leaders): mechanism, results, params, code

### Takeaway
The field has split into three lines. (a) Pure global matching: GMFlow/UniMatch use all-pairs correlation, then softmax, then an expected coordinate, with one or zero refinements. (b) RAFT-style iterative methods that kept the recurrence but simplified everything else: SEA-RAFT, and WAFT, which drops the cost volume entirely in favour of warping plus a ViT update. (c) Frozen-foundation-feature methods: FlowSeek, a 2026 GMFlow-on-DINOv2 variant, UFM. The 2025–2026 benchmark leaders are line (b) or multi-frame: WAFT, MEMFOF, DPFlow, SEA-RAFT. Pure single-shot global matching is still worse at fine accuracy (KITTI, Sintel clean) unless a local refinement is added. That matters directly for OSLO's small-motion problem.

### Cited Findings
**Pure global matching**
- GMFlow recasts flow as global matching. A transformer enhances the features (self- plus cross-attention). Matching is a correlation followed by a softmax, and flow is propagated by self-attention. — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680) [verified via search snippet]; code [github.com/haofeixu/gmflow] [recall]
- GMFlow is about 4.7M params and runs 6 transformer blocks at 1/8 resolution. An optional refinement at 1/4 uses local-window (not global) correlation. Sintel test (clean/final) is 1.74/2.90 with refinement. — [arXiv 2111.13680](https://arxiv.org/abs/2111.13680) [recall]
- UniMatch (GMFlow+) unifies flow, stereo and depth as one dense-correspondence matching problem. It "outperform[s] RAFT with our unified model on the challenging Sintel dataset". With "a few task-specific refinement steps" it matches or beats SOTA on 10 flow/stereo/depth datasets. Code: github.com/autonomousvision/unimatch. — [UniMatch, arXiv 2211.05783](https://arxiv.org/abs/2211.05783) [verified]
- GMFlow+ Sintel test is about 1.03/2.37 and KITTI F1-all about 4.49%, with 6 refinement iterations. — [arXiv 2211.05783](https://arxiv.org/abs/2211.05783) [recall]
- Under matched Chairs→Things training (EPE Things val / Sintel clean / Sintel final), a 2026 paper reports: RAFT (32 iterations) 4.25/1.43/2.71; GMFlow with 0 refinements 3.48/1.50/2.96; GMFlow with 1 refinement 2.80/1.08/2.48; SEA-RAFT(S) –/1.27/4.32; FlowSeek(T) 3.94/1.16/2.48. — [Rethinking Dense Optical Flow without Test-Time Scaling, arXiv 2605.08000](https://arxiv.org/html/2605.08000) [verified]
- After KITTI fine-tuning (KITTI train EPE / F1-all), the same paper reports: RAFT 0.63/1.47; GMFlow with 0 refinements 2.06/7.57; GMFlow with 1 refinement 1.36/5.17; SEA-RAFT(S) 0.93/2.65. — [arXiv 2605.08000](https://arxiv.org/html/2605.08000) [verified]
- That 2026 paper's own model uses frozen DINOv2-S features fused with frozen Depth-Anything-V2-B features, a transformer, GMFlow-style softmax global matching, self-similarity propagation, and no recurrent refinement. It scores 3.02/1.46/2.81 (Chairs+Things) and KITTI 1.99/7.40. It reports no code and no parameter count. The authors concede that SEA-RAFT and FlowSeek lead on KITTI and Sintel clean. — [arXiv 2605.08000](https://arxiv.org/html/2605.08000) [verified]
- GMFlowNet does global matching with Patch-based Overlapping Attention (POLA), then RAFT-style refinement. — [arXiv 2203.11335](https://arxiv.org/abs/2203.11335) [recall]

**Transformer cost-volume methods**
- FlowFormer tokenizes the 4D cost volume into latent cost tokens (an alternate-group transformer encoder) and decodes recurrently with cost-memory cross-attention. It has about 16M params. Sintel test is 1.16 clean / 2.09 final. — [arXiv 2203.16194](https://arxiv.org/abs/2203.16194) [recall]
- FlowFormer++ adds Masked Cost Volume Autoencoding pretraining and reaches Sintel test about 1.07/1.94. — [arXiv 2303.01237](https://arxiv.org/abs/2303.01237) [recall]
- MatchFlow pretrains the feature encoder on geometric image matching (MegaDepth) through a QuadTree-attention "Feature Matching Extractor", and then runs RAFT. — [arXiv 2303.08384](https://arxiv.org/abs/2303.08384) [recall]
- CRAFT adds a Semantic Smoothing Transformer and a "cross-frame attention" correlation layer to RAFT, which makes it robust to large motion. — [arXiv 2203.16896](https://arxiv.org/abs/2203.16896) [recall]
- KPA-Flow (Kernel Patch Attention, CVPR 2022) does local, patch-kernel attention for context aggregation. — [CVPR 2022 open access] [recall; arXiv ID not verified]

**Simplified RAFT descendants and 2025–2026 leaders**
- SEA-RAFT makes three changes: a mixture-of-Laplace loss, direct regression of the initial flow (for faster convergence), and rigid-flow pretraining. On Spring the abstract reports "3.69 EPE / 0.36 1px", a 22.9% / 17.8% reduction over the best published result. It is ≥2.3× faster at competitive accuracy, has the best cross-dataset generalization on KITTI and Spring, and its code is github.com/princeton-vl/SEA-RAFT. — [arXiv 2405.14793](https://arxiv.org/abs/2405.14793) [verified]. Caveat [recall]: I believe the paper's table has these two metrics the other way round (1px = 3.69, EPE = 0.36), so check before quoting.
- WAFT (ICLR 2026 oral) keeps RAFT's iterative structure but removes the cost volume. It warps frame-2 features at the current estimate, uses a modified DPT-Small ViT as the update module, and drops the context encoder. The authors report 1st place on Spring and KITTI, the best zero-shot KITTI result, and up to 4.1× faster than methods of similar accuracy. Its memory efficiency lets it train at full 1080p on Spring. Code: github.com/princeton-vl/WAFT. — [arXiv 2506.21526](https://arxiv.org/abs/2506.21526); [ICLR 2026](https://iclr.cc/virtual/2026/oral/10010387) [verified via search snippets]
- MEMFOF is a multi-frame, memory-efficient method. It was 1st on Spring at submission (1px 3.289) with Sintel clean 0.963, aimed at 1080p. — [arXiv 2506.23151](https://arxiv.org/abs/2506.23151) [verified via search snippet]
- DPFlow (a dual-pyramid adaptive method) claims SOTA on Sintel, KITTI 2015 and Spring, beating SEA-RAFT and RPKNet on Spring by 6.5% and 10.8%. — [arXiv 2503.14880](https://arxiv.org/html/2503.14880v2) [verified via snippet]
- FlowSeek (ICCV 2025) combines a flow network with the Depth Anything v2 foundation model and low-dimensional motion bases. It trains on one consumer GPU (about 8× less hardware) and reports +10% / +15% cross-dataset improvement over SEA-RAFT on Sintel final and KITTI. — [arXiv 2509.05297](https://arxiv.org/abs/2509.05297) [verified]
- Flow-Anything is a data-centric approach that trains on large-scale flow synthesized from single-view images. — [arXiv 2506.07740](https://arxiv.org/html/2506.07740v1) [verified via snippet]
- MemFlow (memory bank, real-time streaming), StreamFlow (multi-frame in a single pass) and VideoFlow (TROF/MOP multi-frame) are the 2023–2024 multi-frame leaders before MEMFOF. — [MemFlow arXiv 2404.04808](https://arxiv.org/abs/2404.04808), [StreamFlow arXiv 2311.17099](https://arxiv.org/abs/2311.17099), [VideoFlow arXiv 2303.08340](https://arxiv.org/abs/2303.08340) [recall]

### Inferences
- Pure global softmax matching (GMFlow with 0 refinements) is clearly worse than RAFT at fine accuracy. On KITTI F1 it scores 7.57 vs 1.47 for RAFT, and it needs a local refinement step to compete. The softmax-expectation readout is good for large motion and bad for sub-pixel precision. OSLO's error lives in small motions (0–5°) and at the poles, so global matching on its own is unlikely to fix OSLO. It is a fix for the problem OSLO does not have.
- The 2025–2026 consensus points the other way from global matching. The winners (WAFT, SEA-RAFT) keep an iterative local update but make it simpler, with no cost volume and direct init. WAFT is the most HEALPix-friendly leader: warping is a feature gather at the current flow estimate, which OSLO's spherical lookup already implements. Its ViT update could become HEALPix window attention, so no 4D cost volume and no lookup pyramid would be needed.
- The 60–75× budget gap to PriOr-Flow and the ERP RAFT-small win at matched budget both suggest that pretrained features (DINOv2, Depth Anything), the FlowSeek/UFM route, may matter more than the matching formulation. DINOv2 features are planar, though, so on HEALPix they would need tangent-patch or ERP extraction followed by resampling.

### Gaps
- I did not retrieve parameter counts for WAFT, MEMFOF or DPFlow, or their Spring EPE tables. The official Spring and Sintel leaderboards were not fetched, so the "rank 1" claims are self-reported.
- I did not verify the KPA-Flow arXiv ID.

## Q2. Dense matchers and matching + interpolation pipelines (incl. geodesic interpolation on the sphere)

### Takeaway
Dense matchers have moved to frozen foundation backbones (DINOv2/v3) with a coarse global match followed by local refinement: DKM → RoMa → RoMa v2, and UFM unifies flow and wide-baseline matching. A 360° version exists. EDM (CVPR 2025) adapts DKM to ERP with spherical positional embeddings and geodesic refinement. The classical sparse-match-then-interpolate pipeline (EpicFlow, RicFlow, InterpoNet) was superseded by end-to-end networks. Its edge-aware geodesic interpolation does map naturally onto a HEALPix graph, but nothing suggests it would improve sub-pixel accuracy.

### Cited Findings
- LoFTR does detector-free coarse matching with linear-attention self/cross transformers at 1/8, then fine refinement at 1/2. — [arXiv 2104.00680](https://arxiv.org/abs/2104.00680) [recall]
- DKM (Dense Kernelized Matching) does a global match with a Gaussian-process regression at the coarse level, then warp refinement with certainty. — [arXiv 2202.00667](https://arxiv.org/abs/2202.00667) [recall]
- RoMa uses frozen DINOv2 coarse features, a transformer match decoder that predicts anchor probabilities (classification, not regression), fine ConvNet features, and a robust regression loss. — [arXiv 2305.15404](https://arxiv.org/abs/2305.15404) [recall]
- RoMa v2 (Nov 2025) uses a frozen DINOv3 backbone, decoupled matching and refinement stages, and a custom CUDA kernel to cut refinement memory. It aims to combine RoMa's robustness with UFM's speed and claims SOTA across pose and dense-matching benchmarks. — [arXiv 2511.15706](https://arxiv.org/abs/2511.15706) [verified via snippet]
- UFM (NeurIPS 2025) runs DINOv2 encoding, then self-attention over both images, then DPT heads that directly regress flow and covisibility. It uses no coarse-to-fine cost volume. Authors report 28% more accurate than UniMatch on flow, and 62% lower error and 6.7× faster than RoMa. It introduces the TA-WB benchmark. The training set is "12" datasets on the project page vs "1212" in arXiv v2, a discrepancy I could not resolve. — [arXiv 2506.09278](https://arxiv.org/html/2506.09278v2) [verified via snippet]
- EDM (CVPR 2025) is DKM adapted to ERP. It uses the spherical camera model, spherical positional embeddings from 3D Cartesian coordinates, and geodesic flow refinement that converts between spherical and Cartesian coordinates. It reports +26.72 and +42.62 AUC@5° on Matterport3D and Stanford2D3D. Runtime is ≈ DKM's, about 0.24 s/pair on an RTX 3090, with no extra learnable params. — [arXiv 2502.20685](https://arxiv.org/abs/2502.20685); [CVPR 2025 supp.](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Jung_EDM_Equirectangular_Projection-Oriented_CVPR_2025_supplemental.pdf) [verified via snippet]
- EpicFlow takes sparse DeepMatching matches and interpolates them with edge-aware geodesic distance on an SED edge map (locally weighted affine), followed by variational energy refinement. — [arXiv 1501.02565](https://arxiv.org/abs/1501.02565) [recall]
- RicFlow (Hu et al., CVPR 2017) does robust interpolation of correspondences using superpixel graphs and RANSAC-fitted local affine models. — [CVPR 2017 open access] [recall; no arXiv]
- InterpoNet learns the sparse-to-dense interpolation step with a CNN. — [arXiv 1611.09803](https://arxiv.org/abs/1611.09803) [recall]
- DIP (Deep Inverse PatchMatch, CVPR 2022) runs PatchMatch-style propagation plus local search, inside a learned iterative framework. — [CVPR 2022] [recall; arXiv ID not verified]

### Inferences
- EpicFlow's interpolation is Dijkstra over an edge-weighted pixel graph. On HEALPix the graph is already built (the OSLO neighbour lists), so geodesic interpolation on the sphere is a direct port: shortest paths over the 8-neighbour HEALPix graph with edge costs taken from image gradients. It is seamless and pole-safe by construction. But its accuracy comes from the sparse matches, so it moves the problem rather than fixing small motions.
- EDM shows the cheap "make the matcher sphere-aware" recipe: positional embedding in xyz, and refinement done geodesically. Its attention still runs on the ERP grid, which supports the project's finding that sampling is not what matters.
- RoMa/UFM-style matchers are built for wide baselines and large viewpoint change. For 0–5° flow, their coarse stage adds nothing beyond what refinement already does.

### Gaps
- I found no HEALPix-native dense matcher and no published geodesic sparse-to-dense interpolation on spherical grids (searched indirectly; none surfaced).

## Q3. Spherical / HEALPix attention (compression, weather, 360 vision): window definitions, cost, code

### Takeaway
HEALPix window attention is established practice. HEAL-SWIN (CVPR 2024) defines windows as runs of consecutive NESTED indices (4^k pixels, i.e. sub-quads of a base pixel), with shifting done by a nested→ring→roll→nested permutation. HEAL-ViT does the same for weather, and the HEALPix compression transformers (Hu et al. TIP 2025, Displays 2025) use hierarchy-defined windows plus spherical convolution to connect neighbouring windows. Cost is the same as planar Swin per pixel. These are drop-in modules for OSLO's NESTED encoder.

### Cited Findings
- HEAL-SWIN works on a 1D nested-order list. A patch is n_patch consecutive pixels (a power of 4, default 4), and a window is n_win consecutive patches (16 or 64). Thanks to nested order these are equal-area quadrilaterals inside a base pixel. Patch merging and expansion group 4^k consecutive patches. — [HEAL-SWIN arXiv 2307.07313](https://arxiv.org/html/2307.07313v2) [verified]
- HEAL-SWIN offers two shift schemes. Grid shift moves windows half a window along base-quad axes and has boundary problems at clashing base-pixel edges, handled by reshuffling and masking. Spiral shift converts nested to ring, rolls by n_shift = √n_win/2, and converts back. Spiral was slightly better (Stanford mIoU 43.2 vs 42.3). Both are precomputed index permutations. — [arXiv 2307.07313](https://arxiv.org/html/2307.07313v2) [verified]
- HEAL-SWIN uses a relative position bias table of size (2√n_win−1)², indexed by Cartesian differences within the window and shared across all windows and base quads. An absolute position embedding gave no benefit. — [arXiv 2307.07313](https://arxiv.org/html/2307.07313v2) [verified]
- HEAL-SWIN results:
  - Per-pixel inference cost is ≈ Swin's, 297 vs 296 ns/pixel, with the same ≈41M params.
  - On SynWoodScape spherical mIoU it scores 0.947 vs 0.918 for Swin. On flat-projected mIoU Swin wins (0.930 vs 0.899), so the advantage depends on the evaluation domain.
  - Stanford 2D-3D-S mIoU is 44.3. The advantage grows with dataset size.
  - Code: github.com/JanEGerken/HEAL-SWIN.
  — [arXiv 2307.07313](https://arxiv.org/abs/2307.07313) [verified]
- HEAL-ViT (weather) uses "pixels from a coarse refinement level … as non-overlapping windows within which self-attention can be performed on fine pixels". — [arXiv 2403.17016](https://arxiv.org/abs/2403.17016) [verified via snippet]
- The feature-enhanced spherical Transformer (Displays 2025) uses HEALPix's hierarchical structure to split the sphere into windows and computes attention within each window, which cuts attention cost. It builds on OSLO. — [ScienceDirect S0141938225000393](https://www.sciencedirect.com/science/article/abs/pii/S0141938225000393) [verified via snippet]
- A BJUT repository entry says that because windows do not communicate, spherical convolution is added to aggregate across neighbouring windows. This is plausibly the Hu et al. TIP 2025 paper ("Learned Spherical Image Compression With Spherical Convolution-Self-Attention and Transformer Context Model", TIP 34:4736–4750), but I could not confirm the identity. — [researchr record](https://researchr.org/publication/HuSWLLY25/bibtex); [BJUT IR](https://ir.bjut.edu.cn/item/423201) [verified as metadata only]
- OSLO-IC adds attention modules built only from 0- and 1-hop convolutions plus element-wise ops, and a spherical masked-conv context model in nested indexing. — [arXiv 2503.13119](https://arxiv.org/pdf/2503.13119) [verified via snippet]
- A HEALPix INR codec reports that the transformer HEALPix codecs depend on windowed spherical self-attention and a transformer context model. — [arXiv 2610.32561](https://arxiv.org/html/2610.32561) [verified via snippet]
- A mask-aware HEALPix transformer has been used for weak-lensing mass mapping, which shows the pattern spreading beyond vision. — [arXiv 2603.25471](https://arxiv.org/pdf/2603.25471) [verified via snippet]
- Other spherical-attention lines:
  - DLWP-HPX (HEALPix CNN/ConvGRU weather model, not attention). — [arXiv 2311.06253](https://arxiv.org/abs/2311.06253) [recall]
  - Spherical Fourier Neural Operators (global mixing via the spherical harmonic transform). — [arXiv 2306.03838](https://arxiv.org/abs/2306.03838) [recall]
  - ClimaX (planar-grid ViT on lat-lon tokens). — [arXiv 2301.10343](https://arxiv.org/abs/2301.10343) [recall]
  - PanoFormer (tangent-patch "spherical token locating" attention on ERP for 360 depth). — [arXiv 2203.09283](https://arxiv.org/abs/2203.09283) [recall]
  - Spherical Transformer on an icosahedral grid (Cho et al.). — [arXiv 2202.04942](https://arxiv.org/abs/2202.04942) [recall]
  - SGFormer (360 depth). — [arXiv 2404.14979](https://arxiv.org/html/2404.14979v3) [verified via snippet]
  - SGAT4PASS (panoramic segmentation). — [arXiv 2306.03403](https://arxiv.org/pdf/2306.03403) [verified via snippet]
  - "SphereFormer" (CVPR 2023) is LiDAR radial-window attention, not 360 images, so the name is a likely confusion. — [arXiv 2303.12766](https://arxiv.org/abs/2303.12766) [recall]

### Inferences
- In OSLO's NESTED layout a HEALPix window is a contiguous slice: `x.view(B, 12*nside²/n_win, n_win, C)`. Window attention therefore costs a reshape, with no gather and no padding, and base-pixel boundaries are the only seams. At r4 (nside 16, 3072 nodes) even *full* global attention is a 3072×3072 matrix per head, about 9.4M entries. That is cheap, so at the estimation grid windowing is only needed for the r5/r6 refinement levels.
- Spiral shift (a ring-order roll) gives cross-window mixing near the poles for free. Pole-ward distortion of the ring roll is a known caveat, and it matters here because OSLO's error is polar.
- The HEAL-SWIN flat-vs-sphere mIoU reversal is a cautionary parallel to the RAFT-small-on-ERP result. Sphere-native architectures win when you score on the sphere, and planar ones win on planar metrics. The EPE-in-ERP-px vs SEPE disagreement already seen in the project fits that pattern.

### Gaps
- I could not access the full text of Hu et al. TIP 2025 or the Displays 2025 paper, so their window size, shift scheme, results and code availability are unknown.

## Q4. Has global matching been applied to 360° flow?

### Takeaway
I found no published GMFlow/UniMatch-backbone 360° flow method and no HEALPix-native global matching. Panoramic flow is dominated by RAFT-family backbones: PanoFlow, then PriOr-Flow with RAFT, GMA and SKFlow. The closest attention-based work uses tangent images plus a transformer. Global softmax matching on a HEALPix graph appears to be an unoccupied niche. Whether it is a *useful* niche for small motion is doubtful (see Q1).

### Cited Findings
- PriOr-Flow uses RAFT as its main backbone. Its universality study applies it to GMA and SKFlow, and GMFlow/UniMatch appear only in related work. — [arXiv 2506.23897](https://arxiv.org/html/2506.23897) [verified]
- PriOr-Flow results (EPE/SEPE):
  - EFT scene: RAFT 4.49/7.43 → PriOr-RAFT 3.30/6.23; GMA 4.26/7.07 → 3.25/6.17; SKFlow 3.79/6.55 → 3.19/6.13.
  - FlowScape all: PriOr-RAFT 2.33/3.49, vs PanoFlow(RAFT) 3.38/4.78 and PanoFlow(CSFlow) 3.31/4.44.
  - Poles: PanoFlow 6.25/6.68 vs PriOr-RAFT 4.13/4.03. Equator: 0.52/2.87 vs 0.53/2.94.
  - Code: github.com/longliangLiu/PriOr-Flow.
  — [arXiv 2506.23897](https://arxiv.org/html/2506.23897) [verified]
- PanoFlow (360° flow, RAFT/CSFlow backbones). — [arXiv 2202.13388](https://arxiv.org/abs/2202.13388) [verified]. One search tool summary attributed PanoFlow to arXiv 2010.02502, which conflicts with this, so 2202.13388 is the correct ID.
- "Learning 360° Optical Flow Using Tangent Images and Transformer" (Springer, 2024): tangent images plus stereo embeddings plus a transformer that aggregates encoder features globally for large displacements. It cites GMFlow, but the snippet does not confirm it is GMFlow-based. — [Springer chapter](https://link.springer.com/chapter/10.1007/978-981-97-8502-5_11) [verified via snippet]
- EDM is the closest "global/dense matching on the sphere" work, but it targets 360 *matching/pose*, not flow. — [arXiv 2502.20685](https://arxiv.org/abs/2502.20685) [verified]
- PanFlow (2025) uses spherical flow decomposed into rotation plus residual, but for video *generation*, not estimation. — [arXiv 2512.00832](https://arxiv.org/html/2512.00832) [verified via snippet]

### Inferences
- PriOr-Flow's pole/equator split shows that even SOTA panoramic flow has about 8× higher EPE at the poles than at the equator (4.13 vs 0.53). The polar problem is universal and not specific to OSLO.
- Global matching's selling points on the sphere are no seam, rotation-friendliness and unbounded search range. OSLO's HEALPix lookup already has the first two, and FlowScape at p50 ≈ 2.5° doesn't need the third (project memory: grid r4 is sub-node even for "large" motion).

### Gaps
- My search coverage was limited (2 queries). A GMFlow-on-ERP 360 baseline may exist in a less-indexed venue.

## Q5. Minimal HEALPix experiment proposal and risks (own assessment, grounded in the findings above)

### Takeaway
The cheapest informative experiment is to keep the OSLO encoder and replace RAFT's correlation-lookup-GRU with GMFlow-style global matching at r4. There are two arms, (A) a global-softmax readout and (B) WAFT-style warping with HEALPix-window attention updates, and both are scored on FlowScape *by motion bin and latitude*. The prediction from the literature is that global matching helps large motion and leaves small-motion and polar error unchanged, unless a local refinement is added.

### Cited Findings
- Pure global matching needs a local refinement to approach RAFT at fine accuracy: KITTI F1 is 7.57 with 0 refinements, 5.17 with 1, and 1.47 for RAFT. — [arXiv 2605.08000](https://arxiv.org/html/2605.08000) [verified]
- Warping-only iterative updates with a ViT updater are a current SOTA formulation, and they need no cost volume. — [WAFT arXiv 2506.21526](https://arxiv.org/abs/2506.21526) [verified via snippet]
- HEALPix NESTED windows at ≈ Swin cost exist, with code. — [HEAL-SWIN arXiv 2307.07313](https://arxiv.org/html/2307.07313v2) [verified]
- SEA-RAFT's direct-regression initialisation and mixture-of-Laplace loss are formulation-independent and cheap to adopt. — [arXiv 2405.14793](https://arxiv.org/abs/2405.14793) [verified]

### Inferences (proposal)
1. **Arm A, HEALPix-GMFlow at r4.**
   - Add 3–6 HEALPix window-attention blocks (n_win = 64, spiral shift), with cross-attention frame1↔frame2. Use global or windowed cross-attention, since global is affordable at 3072 nodes.
   - Matching: full 3072×3072 correlation, then softmax over frame-2 nodes, giving an expected position as a softmax-weighted sum of unit vectors **in R³**, renormalised. The tangent displacement at node i is log_map(x_i, x̂). This is seamless and SO(3)-equivariant up to the feature extractor.
   - Propagate with self-attention, then convex-upsample to r6 (the existing code).
   - Cost: for r4 ≈ 9.4M entries per batch item. Trivial.
2. **Arm B, HEALPix-WAFT.** Warp frame-2 features via the existing spherical lookup at the current flow. Replace the ConvGRU with HEALPix window-attention blocks, using direct-regression init (from SEA-RAFT) or Arm A's output as init. Iterate at r5 so that the update operates above the r4 sub-node limit.
3. **Hybrid A+B** is the GMFlow+1-refinement analogue. It is the configuration the literature says is needed for fine accuracy.
4. **Readout to compare.** Stratified EPE in ERP px and SEPE, by motion bin (0–1°, 1–5°, 5–15°, >15°) and by latitude band, against the existing OSLO and RAFT-small at matched steps and budget.
5. **Risks.**
   - (i) The softmax expectation is biased toward the centre of the match distribution. At 3.66° node spacing, sub-node accuracy depends on how soft the softmax is (the GMFlow 0-refinement precision issue). Arm A alone will likely *not* fix 0–5° error.
   - (ii) Expectation in R³ shrinks toward the origin for broad distributions. Renormalise, or use a local-window soft-argmax around the argmax.
   - (iii) Spiral shift distorts near the poles, which is exactly where OSLO's error sits. Prefer neighbour-window attention or the extra spherical conv used in the compression transformers.
   - (iv) Attention needs more data and steps than convs; HEAL-SWIN's gain grew with dataset size. FlowScape (≈6k pairs) may be too small, so pretrain on chairs360.
   - (v) Given the RAFT-small-ERP result and the finding that the SO(3) augmentation is the main lever, a formulation change may again show "no global effect". Pre-register the motion-bin and polar metrics as the decision criterion.
6. **Higher-prior alternative.** Frozen foundation features (DINOv2 or Depth Anything, resampled ERP→HEALPix), as in FlowSeek and UFM. The 2025–2026 evidence favours better features plus a simple head over new matching formulations.

### Gaps
- I found no published evidence on softmax global matching accuracy for sub-node motion on a graph or irregular grid. Risk (i) is extrapolated from planar KITTI numbers.
