# Pyramidal / coarse-to-fine optical flow (warping, local cost volumes, hierarchical probabilistic flow) and its mapping onto a HEALPix NESTED quad-tree

Scope note: these notes cover (1) the planar coarse-to-fine (C2F) family and its recent revival, (2) evidence for pyramidal vs single-scale estimation on small motions, (3) prior C2F/warping on spherical grids, (4) HD3 as a quad-tree-native probabilistic formulation, (5) a minimal HEALPix experiment. "Assessment" items are my own reasoning and are kept in the Inferences subsections. HEALPix level spacings below are computed as sqrt(4π/N), N = 12·4^r: r2 14.66°, r3 7.33°, r4 3.66°, r5 1.83°, r6 0.92°, r7 0.46° (my own calculation, consistent with the project's r4 ≈ 3.66° figure).

## Q1. The C2F method family: mechanism, accuracy vs RAFT, params, code

### Takeaway
Classic 2018–2020 pyramidal nets (SPyNet, PWC-Net, LiteFlowNet, IRR, VCN, MaskFlowNet, HD3, ScopeFlow) all lose to RAFT on Sintel. The 2022–2025 *recurrent* pyramidal methods (MS-RAFT, MS-RAFT+, CCMR, DPFlow) beat RAFT clearly, by roughly 15–35% EPE. They do it by putting RAFT-style iterative refinement at every level of a quad-tree-like pyramid, with flow passed from each level to the next as initialisation. So the evidence in 2026 says "pyramidal returns, but recurrent": you keep RAFT's update operator and run it once per level.

### Cited findings

**RAFT's stated case against C2F (the "RAFT is not pyramidal" position)**
- RAFT "maintains and updates a single fixed flow field at high resolution". It claims to overcome "several limitations of a coarse-to-fine cascade", and says C2F "may miss small fast-moving objects and have difficulty recovering from early mistakes". It also notes that a single resolution allows warm-start from previous frames, which "cannot be easily done using the coarse-to-fine model". — [RAFT, arXiv 2003.12039](https://ar5iv.labs.arxiv.org/html/2003.12039)
- RAFT is 4.8M params (5.3M with upsampler), and RAFT-S is 1.0M. RAFT says RAFT-S "outperforms PWC-Net and VCN which are more than 6x larger". — [RAFT, arXiv 2003.12039](https://ar5iv.labs.arxiv.org/html/2003.12039)

**Reference numbers (Sintel test Clean/Final EPE, KITTI-15 test Fl-all), from RAFT Table 1 and Table 4** — [RAFT, arXiv 2003.12039](https://ar5iv.labs.arxiv.org/html/2003.12039)
- FlowNet2: 4.16 / 5.74 / 11.48 (C+T+S/K); 162M params.
- HD3: 4.79 / 4.67 / 6.55.
- IRR-PWC: 3.84 / 4.58 / 7.65; 6.4M params.
- ScopeFlow: 3.59 / 4.10 / 6.82.
- LiteFlowNet2: 3.48 / 4.69 / 7.74. LiteFlowNet is 5.4M params; LiteFlowNetX is 0.9M.
- PWC-Net+: 3.45 / 4.60 / 7.72; 9.4M params.
- VCN: 2.81 / 4.40 / 6.30; 6.2M params.
- MaskFlowNet: 2.52 / 4.17 / 6.10.
- RAFT (2-view, C+T+S+K+H): 1.94 / 3.18 / 5.10. RAFT warm-start: 1.61 / 2.86.

**Older entries** (HD3 paper Table 2/3) — [HD3, arXiv 1812.06264](https://arxiv.org/abs/1812.06264)
- SPyNet-ft: 6.64 / 8.36 Sintel test, KITTI-15 35.07% Fl-all, 0.16 s.
- PWC-Net-ft: 4.39 / 5.04, KITTI-15 9.60%, 0.03 s.
- LiteFlowNet-ft: 4.86 / 6.09, KITTI-15 10.24%.
- FlowNet2-ft: 4.16 / 5.74, KITTI-15 10.41%.

**Other entries** (MS-RAFT Table 1) — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)
- Flow1D: 2.24 / 3.81 / 6.27.
- DICL-Flow+: 1.86 / 3.32 / 6.31.

**Mechanisms of the classic pyramidal family**
- PWC-Net builds "a learnable feature pyramid" and "uses the current optical flow estimate to warp the CNN features of the second image" at each level before a cost volume. The C2F idea descends from variational methods, which the paper calls the most popular framework for optical flow. — [PWC-Net, CVPR 2018 (arXiv 1709.02371)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_PWC-Net_CNNs_for_CVPR_2018_paper.pdf)
- SPyNet keeps each level's displacement small, but is "known to have problems when small objects move quickly". — [SPyNet, arXiv 1611.00850](https://ar5iv.arxiv.org/html/1611.00850) (as summarised in the search results; not verified against the full text)
- IRR shares the estimation network across all scales. — [MS-RAFT related-work text, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)

**The recurrent-pyramid revival**
- **MS-RAFT** (ICIP 2022) combines four ideas:
  - a "partially shared coarse-to-fine architecture": the matching block and convex-upsampling mask are shared across scales, and flow is ×2 convex-upsampled to initialise the next scale;
  - U-Net-style multi-scale features;
  - a hierarchical cost volume (RAFT correlation pyramid with fewer lookup levels at coarse scales);
  - a multi-scale multi-iteration loss.

  The final model is 3 scales, finest at 1/4 resolution (2× RAFT), with 2 lookup levels and about 13.5M params. Sintel test is 1.37 / 2.67, KITTI 4.88, which it reports as +14.6% Clean / +6.6% Final / +4.3% KITTI over RAFT. Matched (non-occluded) Clean is +23.1% over RAFT. — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)
- **MS-RAFT ablation (most relevant to this project; Sintel train after C+T):**
  - Single-scale RAFT at 1/8 res, 12 iterations: 1.40 / 2.67.
  - Single-scale RAFT at 1/8 res, 18 iterations: 1.45 / 2.70.
  - Single-scale RAFT at a finer 1/4 res: 1.58 / 3.10 (12 iterations) and 1.52 / 3.08 (18 iterations), so *worse* than at 1/8.
  - 3-scale C2F at a finest scale of 1/4: 1.13 / 2.60.
  - The authors: "simply considering single-scale RAFT on a finer resolution … does not improve results … without a proper initialization and without the use of multi-scale features it seems difficult to obtain improvements."
  - Multi-scale loss beats a finest-scale-only loss.

  — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)
- **MS-RAFT+** (IJCV 2024; code at cv-stuttgart/MS_RAFT_plus) has 4 scales, finest at 1/2, and 16M params. Sintel test is 1.23 / 2.68 and KITTI 4.15. It gets these results "with a single model and a single parameter setting" across KITTI, Sintel, Middlebury and VIPER. — [MS-RAFT+ IJCV](https://link.springer.com/article/10.1007/s11263-023-01930-7); numbers from [CCMR Table 1, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- **CCMR / CCMR+** (WACV 2024):
  - C2F with a shared GRU, based on MS-RAFT+. It starts at 1/16 with zero flow; after T_s GRU iterations the flow is ×2 convex-upsampled to initialise the next scale, down to 1/2 for CCMR+. Training uses 4/5/5/6 iterations from coarse to fine.
  - The all-pairs cost volume is *not* precomputed: local costs around the current estimate are computed on the fly "due to the large memory consumption for finer scales".
  - It adds XCiT global-context-guided motion grouping. A GMA-style token attention at 1/2 scale would need about 80 GB for a Sintel frame, against less than 5 GB for CCMR.
  - Params: CCMR 10.8M (3 scales), CCMR+ 11.5M (4 scales).
  - Sintel test: CCMR 1.19 / 2.14, CCMR+ 1.07 / 2.10. KITTI: CCMR 4.04, CCMR+ 3.86 (state of the art at the time). Matched Clean for CCMR+ is 0.31.
  - In matched-protocol KITTI-val: MS-RAFT 5.21, CCMR 4.75, MS-RAFT+ 4.71, CCMR+ 4.57, against single-scale GMA at 5.01.

  — [CCMR, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- **DPFlow** (CVPR 2025):
  - A "dual pyramid": an image pyramid, plus a feature pyramid propagated by forward and backward ConvGRUs across levels.
  - The level count is set adaptively from the input diagonal (base 3); training uses 3 scales × 4 refinement iterations. 10.0M params.
  - Sintel test 1.04 / 1.97 (RAFT 1.60 / 2.85, SEA-RAFT-L 1.31 / 2.60). KITTI-15 3.56 (CCMR+ 3.86, RAFT 5.10). Spring 1px 3.06.
  - It generalises to 8K inputs when trained at low resolution; on Kubric-NK 8K, RAFT scores 82.7 EPE against DPFlow's 4.11.
  - Code is in ptlflow.

  — [DPFlow, arXiv 2503.14880](https://arxiv.org/html/2503.14880v2)
- **DIP** (CVPR 2022, PatchMatch-based): Sintel test 1.44 / 2.83, KITTI 4.21. Pre-training generalisation (Sintel train 1.30 / 2.82; RAFT 1.43 / 2.71). CCMR describes DIP as relying on additional refinement steps on a second scale. — [CCMR Tables 1–2, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- **Hofinger et al.** ("Improving Optical Flow on a Pyramid Level", ECCV 2020) reported state of the art on Sintel and KITTI 2012 test at the time with a non-recurrent PWC-style pyramid, using per-level training fixes (loss max-pooling). — [ECVA page](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6440_ECCV_2020_paper.php); arXiv 1912.10739 appeared in the search results

### Inferences
- The MS-RAFT ablation reproduces the project's r5 result almost exactly. Making a single-scale RAFT finer (1/8 → 1/4) *hurt* (1.40 → 1.58 Clean), while the same finest resolution reached through a C2F cascade *helped* (1.13). So the r5-alone failure is not evidence against finer estimation. It is evidence that finer estimation needs an initialisation from the coarser level and multi-scale features. This is the strongest single argument for a HEALPix C2F experiment.
- "Pyramidal" in 2026 means *recurrent per level, with warm-start from the coarser level*. A PWC-style one-shot decoder per level is the weaker variant: PWC-Net+, IRR and ScopeFlow are all ≥ 3.4 Sintel Clean. A HEALPix design should copy MS-RAFT/CCMR (shared GRU across levels, local on-the-fly cost, convex ×2 upsampling, multi-scale multi-iteration loss) rather than PWC-Net. That keeps the existing OSLO-RAFT components (SDPA encoders, ConvGRU, convex upsampler) and changes only the scheduling, so it is a modest departure from RAFT. If the advisor wants a mechanism genuinely "not RAFT", the HD3-style discrete-density decoder (Q4) is the clean break.
- Parameter budgets are comparable: MS-RAFT 13.5M, CCMR+ 11.5M, DPFlow 10M, RAFT 5.3M. The cascade costs about 2–3× RAFT params, mostly in the feature extractor (MS-RAFT says the overhead is in the extractor).

### Gaps
- I did not verify LiteFlowNet3 numbers or the arXiv IDs of LiteFlowNet2/3, MaskFlowNet, VCN, IRR, ScopeFlow, DICL, Flow1D or DIP directly. Their numbers above come from the RAFT, MS-RAFT and CCMR tables. Recalled IDs, unverified: PWC-Net 1709.02371, FlowNet2 1612.01925, LiteFlowNet 1805.07036, IRR 1904.05290, MaskFlowNet 2003.10955, VCN 1911.04031, ScopeFlow 2002.10770, DICL 2010.14851, Flow1D 2104.13918, MS-RAFT+ 2210.16900.
- I found no 2026 pyramidal paper newer than DPFlow, though searches were limited. A Papers-with-Code/Sintel leaderboard snapshot for Nov 2026 was not retrieved.
- VCN's 4D volumetric cost (separable 4D conv) was not read in full. Only its table numbers are cited.

## Q2. Pyramidal vs single-scale for SMALL motions and sub-pixel accuracy

### Takeaway
No paper I found isolates "small displacements" (as opposed to small *objects*) as a pyramid-vs-single-scale variable. The closest proxies are matched/non-occluded-region EPE and fine-detail claims, and those consistently favour recurrent C2F. The documented C2F failure modes concern small *fast* objects and error propagation, not small slow motions.

### Cited findings
- MS-RAFT is "particularly accurate in non-occluded regions": rank 1 on matched Sintel Clean and Final, +23.1% over RAFT on Clean-matched. Its flow fields show "comparably more structural details". — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)
- C2F schemes "not only improve robustness regarding ambiguous matches, but also produce more detailed flow fields … valuable initializations from coarser scales allow them to operate more effectively on higher resolutions". CCMR+ reaches an "unprecedented" matched Sintel Clean of 0.31 (RAFT 0.62). — [CCMR, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- Historically, C2F "primarily helped to improve the estimation of large displacements". — [CCMR, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- Small/fast-object failures:
  - With Gaussian pyramids, "the object may completely disappear at coarse (high) levels of the pyramid". — [Sevilla-Lara et al., ECCV 2014](https://people.cs.umass.edu/~lsevilla/SevillaECCV2014.pdf)
  - "Current coarse-to-fine optical flow networks fail to capture small and fast-moving objects which has disappeared at coarse resolution levels". The cause is the "contradiction between limited search range of warping based local matching and large displacement". — [HMFlow, arXiv 2011.09654](https://arxiv.org/pdf/2011.09654)
  - Fine structures visible at one level "propagate contradicting gradients towards the coarser levels when they move in a different direction compared to their background". The proposed fix is per-level loss max-pooling. — [Hofinger et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6440_ECCV_2020_paper.php) (via search summary; full text not read)
- HD3 pretrained (no fine-tuning) has high Sintel EPE (3.84 / 8.77 train) but the lowest KITTI outlier rate. The authors say "the metric of EPE emphasizes large motion error". After fine-tuning HD3 has the best Sintel Final (4.67) of its generation, and they note advantages "in estimating many thin structures". — [HD3, arXiv 1812.06264](https://arxiv.org/abs/1812.06264)

### Inferences
- In this project's regime (77% of pixels at 0–5°, error concentrated in small motions), the documented C2F failure mode (small *fast* objects lost at coarse levels) is mostly the wrong regime. Small slow motions do not need a coarse level to find them. They need a *fine* level whose search window is centred correctly and whose features resolve sub-node offsets. C2F helps here mainly by making a finer estimation grid (r5/r6) affordable and well-initialised, not by extending range.
- The real risk for this project is the reverse: at r2/r3 (14.7°/7.3° nodes) most of the field is sub-node. If coarse levels output confident non-zero junk, the error is propagated into fine levels through warping. The literature names "difficulty recovering from early mistakes" as RAFT's critique. The coarse levels should be cheap, short (few iterations), and initialised and supervised so that "zero residual" is easy to output.
- The polar error is plausibly a fine-resolution problem (feature and readout resolution near the poles), which C2F addresses only indirectly. The RAFT-parity memory notes that SO(3) augmentation, not the architecture, was the lever. So C2F should be judged on the matched recipe.

### Gaps
- No source quantifies EPE binned by displacement magnitude (e.g. < 1 px) for pyramidal vs single-scale networks under matched training. This would have to be measured in the project.
- Hofinger et al. and SPyNet full texts were not read; their statements come from search-result excerpts.

## Q3. Prior coarse-to-fine / warping on spherical, icosahedral or HEALPix grids

### Takeaway
I found **no** published optical-flow method that runs a coarse-to-fine pyramid with warping natively on a HEALPix or icosahedral hierarchy. 360° flow work either uses ERP with planar networks (PWC-Net or RAFT), tangent/cubemap/icosahedral *projections* of planar networks, or ERP-adapted convolutions. A HEALPix C2F with rotation (exponential-map) warping appears to be open ground; this is subject to the literature-scope register check.

### Cited findings
- A multi-projection-fusion study of 360° flow (ECCV 2022) found that among the backbones it tested, "PWC achieved the highest performance", and that "the equirectangular representation is the best single projection". Fine-tuning PWC with spherical convolutions "cannot achieve a good performance due to kernel shape changes from the original pre-trained PWC model". — [MPF-Net, arXiv 2208.00776](https://arxiv.org/pdf/2208.00776) (via search summary)
- 360° flow with tangent images uses gnomonic projection onto cubemap and icosahedron-vertex tangent planes to "incrementally refine the estimated 360° flow fields", including under large rotations. The abstract does not say whether this is a scale pyramid. — [arXiv 2112.14331](https://arxiv.org/abs/2112.14331)
- OmniFlowNet (ICPR 2021) adapts a perspective flow CNN's convolutions to be consistent with ERP, with no new training data. — [OmniFlowNet](https://i-cirena.ubfc.fr/node/29959)
- SphereNet adapts filter sampling locations to undo ERP distortion and wraps filters around the sphere. It was evaluated on detection and classification, not flow. — [SphereNet, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.html)
- Spherical CNN grids:
  - HEALPix level r has 12·4^r pixels, and "most spherical points on HEALPix grid have 8 neighbor points"; DeepSphere builds a HEALPix graph. — [survey, arXiv 2502.10444](https://arxiv.org/html/2502.10444)
  - Icosahedral alternatives (GICOPix) give more regular graphs than HEALPix. — [Yang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Rotation_Equivariant_Graph_Convolutional_Network_for_Spherical_Image_Classification_CVPR_2020_paper.pdf)
- Project-supplied context (not independently fetched): the HEALPix overfitted image codec arXiv 2610.32561 attributes about 45% of its gain to multiscale hierarchy and about 4% to equal-area sampling.

### Inferences (assessment of fitness, my own)
- **Pyramid ↔ NESTED quad-tree: exact fit.**
  - In NESTED ordering, parent = child >> 2 and the 4 children of p are 4p…4p+3. Pooling and unpooling are reshapes, with no resampling and no seam, and OSLO already does this.
  - A planar ×2 pyramid on ERP has neither property: rows near the poles are over-sampled at every level.
  - So the hierarchy (the part arXiv 2610.32561 says matters about 45% in a related HEALPix task) maps natively.
  - Caveat: HEALPix child pixels are not congruent to the parent near the 8 base-pixel vertices, so the "×2 convex upsampler" must learn per-node weights over a parent k-ring, not a fixed 3×3 mask. OSLO's convex upsampling to r6 already does a version of this.
- **Warping on the sphere:**
  - Represent the flow at node x as a tangent vector v (east/north, in radians). The warped location is exp_x(v) = cos|v|·x + sin|v|·v̂ (great-circle step).
  - Sample frame-2 features at that point by interpolation over the target's HEALPix neighbours (bilinear-on-sphere or barycentric). The project already has a "spherical lookup" for correlation.
  - There is no seam because the target is a 3-D unit vector. There is no polar singularity in the *warp*, because exp_x is defined everywhere.
  - The east/north *frame* is singular exactly at the poles. The gauge-diagnostic memory shows the frame is inconsistent but not load-bearing. A frame-free alternative is to carry flow as a 3-D tangent vector (x × axis or v ∈ T_xS²) and upsample by parallel transport, or to parameterise each node's flow as a small rotation (axis-angle ω ∈ R³, with x' = R(ω)x).
  - Upsampling a parent's flow to its 4 children needs parallel transport of the tangent vector from the parent centre to the child centre. This is negligible at fine levels but not at r2/r3 near the poles.
- **Local cost volume over k-rings:**
  - PWC/CCMR-style local costs over a (2d+1)² window map to a k-ring of radius d around the *warped* point.
  - A k-ring of radius d has 1 + 4d(d+1) nodes generically (9 at d = 1, 25 at d = 2), fewer at the 8 vertex pixels (7 neighbours at d = 1). The project memory reports a "fixed pattern at 24 r7 7-neighbour pixels" artefact.
  - Because the warped point is off-grid, correlation needs feature interpolation at off-grid positions. This can be done either by sampling frame-2 features at exp_x(v + δ_k) for a fixed set of tangent offsets δ_k (cleaner, regular in the tangent plane, gauge-dependent), or by taking the k-ring of the nearest node (irregular, gauge-free).
  - Costs are cheap: r6 has 49 152 nodes; 49 offsets × 49 152 ≈ 2.4M dot products per iteration.
- **Residual flow per level:** f_l = up(f_{l−1}) + g_l, with g_l bounded by roughly 1–2 node spacings of level l (r4 3.66°, r5 1.83°, r6 0.92°). The project's 0–5° motion bulk is about 1.4 r4 nodes, 2.7 r5 nodes and 5.4 r6 nodes. A 3-level cascade r4 → r5 → r6 with radius-2 to radius-3 local windows would cover the bulk at each level, given a correct parent estimate.
- **Failure modes on HEALPix specifically:**
  1. Coarse levels r2/r3 cannot represent the dominant sub-node field, so they should be skipped or kept very short. Starting the cascade at r4 (the current estimation grid) avoids most of the "small object lost at coarse level" risk because the motion is already mostly sub-node there.
  2. Polar parallel-transport and frame errors accumulate across levels if the flow is upsampled in east/north components without transport.
  3. Irregular neighbourhoods at the 8 vertex pixels per level: there are 8 per level independent of resolution, so they matter more at coarse levels.
  4. Large global rotations (the SO(3)-augmented regime) exceed any local window at fine levels and must be captured at r4 or by an explicit global-rotation pre-stage (derotation), as in the tangent-image work.

### Gaps
- MPF-Net (2208.00776) and 2112.14331 were not read in full. Whether 2112.14331's "incremental refinement" is a scale pyramid with derotation is unconfirmed.
- I found no publication of spherical PWC, icosahedral flow pyramids or HEALPix flow. Absence of evidence after two targeted searches only, so it needs a check against the literature-scope register.

## Q4. HD3 (hierarchical discrete distribution decomposition): does a per-level discrete distribution match the quad-tree?

### Takeaway
HD3 is the most quad-tree-shaped formulation in the literature. It models the full flow as a composition of per-level *residual* discrete distributions, each conditioned on coarser levels via warping, and trains them with a KL loss. That yields calibrated uncertainty at no extra cost. Its residual support is a 2×2 *displacement* window per level, and that is **not** the same thing as the 4 spatial children of a NESTED parent. The match to HEALPix is in the spatial pyramid, not in the displacement support.

### Cited findings (all from [HD3, arXiv 1812.06264](https://arxiv.org/abs/1812.06264), CVPR 2019; code github.com/ucbdrive/hd3)
- **Motivation.** A full match density is intractable: a 1000×1000 image with a ±50 range has 10^6 pixels × 10^4 support, about 10^10 cells.
- **Decomposition.** Multi-scale fields f^l at half resolution per level, with residuals g^l = f^l − φ(f^{l−1}), where φ is ×2 upsampling. The full field is f = Σ_l φ^{L−l}(g^l), and the density factorises as p(f) = Σ Π_l p(g^l | G^{l−1}). The decomposed densities "have quite low variance", so each is approximated on a small support.
- **Support.**
  - A real-valued motion vector falls in a unique 2×2 grid window. Ground truth is converted to a density by splatting bilinear weights onto that window (V2D).
  - At inference HD3 takes the 2×2 window with maximal mass and its local expectation (D2V). It always uses the max-posterior per level ("we always take the max posterior of g^l"), with no quad-tree sampling, to avoid training/test sampling cost.
  - The paper explicitly says "Ideally, we can sample g^s ∈ G^{l−1} in a quadtree fashion", and calls the hierarchy "levels of local distributions similar to quadtrees".
- **Architecture.**
  - DLA-34-Up feature pyramid, coarsest level ×64 downsampled.
  - Each level warps F2 by φ(E[f^{l−1}]), computes a correlation of range 4 ("consistent with the size of match density support"), and concatenates it with F1, the upsampled flow and an upsampled *density embedding* (a "density bypass").
  - A decoder (2 residual blocks + aggregation node, dilated context net at the last level) outputs the per-level match density through a classifier.
  - 5 levels for flow. Loss = Σ_l KL(p_gt^l ‖ p_res^l).
- **Results.**
  - Sintel test 4.79 Clean / **4.67 Final** (best Final in its table, vs PWC-Net-ft 5.04). KITTI-15 Fl-all **6.55%**, the best two-frame result at the time and better than some scene-flow methods. 0.08 s.
  - Pretrained HD3 (no fine-tuning) has poor Sintel EPE (3.84 / 8.77 train), which the authors attribute to EPE emphasising large motions.
  - Uncertainty: best log-likelihood vs FlowNet-Dropout/ProbOut/ADF (Sintel Clean −1.49 vs −3.88 for the best baseline). Outlier detection mean IoU 68.0 vs 49.5 for forward-backward consistency on KITTI.
- **Limitations stated by the authors.** Errors concentrate at boundaries and occlusions. "We do not consider relationships between match densities of adjacent pixels."
- **Later context.** HD3 (4.79 / 4.67) is far behind RAFT (1.61 / 2.86) on Sintel test. — [RAFT Table 1, arXiv 2003.12039](https://ar5iv.labs.arxiv.org/html/2003.12039)

### Inferences (assessment, my own)
- **What maps cleanly onto NESTED HEALPix:**
  - the spatial pyramid (exact);
  - per-level residual g^l with warp-conditioning (via expmap);
  - the KL loss on per-level densities;
  - the per-node confidence from the density.

  The last is directly useful for the project's "static-majority / decision-gate" thread: the confidence map is a learned abstention signal, and HD3 showed it beats forward-backward consistency.
- **What does not map one-to-one:**
  - HD3's displacement support is a 2×2 window of *grid offsets* at level l (the bilinear cell containing the residual), not the 4 children.
  - On HEALPix the natural support is "the target node's cell in the level-l tessellation plus its k-ring". Bilinear splatting of a tangent-plane residual onto the 3–4 surrounding HEALPix centres is well-defined with barycentric weights, but the cell shapes vary, especially near the base-pixel vertices and the poles.
  - A cute alternative that *does* use the quad-tree: at level l+1, express the refinement of a level-l displacement as a distribution over the 4 children of the *target* pixel (plus a neighbour ring for robustness). That makes the displacement hierarchy itself a NESTED descent: coarse target cell → child → grandchild, so the composed density is a distribution over target pixels at the finest level, factored along the NESTED tree. This is a genuinely non-RAFT, discrete, tree-structured matching formulation.
  - Its known weakness is HD3's own: argmax/hard window choice at a coarse level is irrecoverable when the true target lies across a parent boundary. The mitigation is overlapping support (children of the target parent *and* of its neighbours), i.e. a k-ring, which brings back the irregular-neighbourhood issue.
- **Fit for this project's regime:** HD3's discrete per-level classification with local expectation gives sub-cell estimates via the expectation. Whether that resolves 0.1–1° motions at r6 (0.92° cells) is the open question. The project's history (correlation-argmax "can't resolve sub-node motion"; matching inert at sub-node; the 2026-08-04 defect voiding the earlier flow360 evidence) suggests testing on replica360/FlowScape first.
- **Accuracy prior:** HD3-class decoders are about 3× worse than RAFT on Sintel, so a pure HD3 port is unlikely to beat the matched RAFT-small baseline. The safer hybrid: CCMR-style recurrent C2F as the estimator, with an HD3-style discrete density head (KL loss) at each level for uncertainty and residual classification.

### Gaps
- HD3 parameter count is not stated in the text I extracted, and HD3 was not in RAFT's parameter table.
- No follow-up work found that implements HD3 on irregular grids or graphs.

## Q5. What a minimal HEALPix PWC/HD3-style experiment would look like, and its risks

### Takeaway
My own design proposal, grounded in the MS-RAFT/CCMR findings above. Keep OSLO-RAFT's encoders, ConvGRU and convex upsampler. Replace the single r4 estimation with a 3-level NESTED cascade r4 → r5 → r6. Each level gets expmap warping of frame-2 features, a local tangent-offset cost volume, a shared GRU warm-started from the upsampled parent flow, and a multi-scale multi-iteration loss. Compare on replica360 and FlowScape against the matched RAFT-small baseline.

### Cited findings (design anchors)
- Use a shared update operator across scales, ×2 convex upsampling as initialisation, and a multi-scale multi-iteration loss with weights increasing over scales and iterations. — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163); [CCMR, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- Compute local costs on the fly at fine scales rather than all-pairs, for memory. — [CCMR, arXiv 2311.02661](https://arxiv.org/abs/2311.02661)
- At coarse scales, reduce lookup range so that "not too many looked-up candidates lie outside the image domain". On the sphere there is no outside, so this constraint disappears. — [MS-RAFT, arXiv 2207.12163](https://arxiv.org/abs/2207.12163)
- Iteration schedules used: MS-RAFT 4/6/8 for 3 scales; CCMR+ 4/5/5/6 for 4 scales. — [MS-RAFT](https://arxiv.org/abs/2207.12163), [CCMR](https://arxiv.org/abs/2311.02661)
- Pyramid failure-mode remedies: per-level loss max-pooling ([Hofinger et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6440_ECCV_2020_paper.php)) and global/hybrid matching features for small fast objects ([HMFlow, arXiv 2011.09654](https://arxiv.org/pdf/2011.09654)).

### Inferences (proposed experiment, my own)
- **Arms** (same recipe, SO(3) aug, `erp_px` loss, area-weighted metrics, 2 seeds; matched against the existing RAFT-small baseline at 6.3 px on FlowScape):
  - A0: current OSLO (r4 estimation → convex up to r6).
  - A1: **C2F-recurrent**, levels r4/r5/r6, shared GRU, iterations e.g. 4/5/6, local cost by sampling frame-2 features at exp_x(f + δ_k) for a fixed tangent stencil (radius 3 → 7×7 offsets at the level's spacing), parent → child flow by convex upsampling with parallel transport.
  - A2: A1 + HD3 head: per-level discrete residual density over the stencil, KL loss with bilinear-splat targets, local-expectation readout, confidence map.
  - Optional A3: A1 without warping (cost centred at the unwarped node). This isolates warping vs mere multi-resolution.
- **Pre-registered readouts:**
  - global and polar ERP-px EPE;
  - EPE binned by |flow| (0–1°, 1–5°, 5–15°, >15°), since this is where the error lives;
  - per-level EPE, to detect coarse-level error propagation;
  - for A2, calibration of the confidence (AUSE or IoU of outliers).
- **Decision rule suggestion:** A1 must beat A0 by more than the 2-seed spread, and close at least half of the 8.63 vs 6.32 px gap to RAFT-small on FlowScape, to count. MS-RAFT's ablation gives a prior that it can (finer C2F 1.13 vs coarse single-scale 1.40 vs finer single-scale 1.58).
- **Risks:**
  1. Coarse-level error propagation: hard to recover from early mistakes (the RAFT critique). Mitigate with short coarse schedules and multi-scale supervision.
  2. The tangent-frame and parallel-transport implementation at the poles is the exact region where OSLO already fails; a bug there would be confounded with the hypothesis. Unit-test that expmap warping reproduces a known rotation field exactly (cf. the project's "warp-validated" convention checks).
  3. Interpolation of frame-2 features at off-grid points on HEALPix (barycentric over irregular cells) adds a new fp/geometry surface. The project's earlier sanity suite found slot-orientation and fixed-pattern bugs at 7-neighbour pixels.
  4. Compute: r6 recurrent iterations over 49k nodes with off-grid sampling are about 16× the r4 node count. Fine for 2k-step pilots, likely 2–4× OSLO wall-clock.
  5. Benchmark realism: FlowScape's large motion (p90 70–93 px) favours C2F range extension. The small-motion benefit needs the magnitude-binned readout to show.
  6. Novelty: C2F, warping and local cost volumes are all published (PWC-Net, MS-RAFT, CCMR). What would be new is only the HEALPix-native rotation-warped cascade (and, for A2, a NESTED discrete density), subject to a literature-scope check.

### Gaps
- No published spherical baseline exists for this design, so expected gains are extrapolated from planar results only.
- Whether expmap warping vs rotation-matrix (per-node axis-angle) parameterisation matters empirically is unknown. No source was found.
