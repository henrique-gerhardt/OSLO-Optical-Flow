# Structured / non-dense flow representations that use spherical geometry (ego-motion + parallax, local SO(3) fields, piecewise-parametric flow, spherical bases)

Scope note: research done 2026-11-04 with about 16 search/fetch calls. Most primary sources were read at abstract or snippet level. Roberts et al. 2009 was read in full text (first pages). Anything marked **[own]** is my derivation or assessment, not a published fact.

---

## Q1. Spherical/omnidirectional ego-motion and flow decomposition (classic and learned). Does anyone parametrize 360° flow as rotation + translation·inverse-depth + residual?

### Takeaway
The decomposition is old and well established in classic omnidirectional vision: map flow to the sphere, then solve for ego-motion. Perspective deep learning has used rigid flow + residual flow since 2018 (GeoNet, EPC++, RigidMask, DytanVO). I found **no published learned 360° optical-flow network that outputs flow as rotation + translation·inverse-depth + residual on the sphere**. Learned 360° work estimates depth and pose (SfMLearner-style), not flow. This combination looks open, but my search was limited (see Gaps).

### Cited Findings
**Classic omnidirectional**
- Gluckman & Nayar, "Ego-Motion and Omnidirectional Cameras" (ICCV 1998) compute flow in the omnidirectional image, map it onto a sphere with the projection Jacobian, and run existing ego-motion algorithms on the spherical flow. They argue that projecting the *flow* avoids the artifacts of resampling the *images*. Motivation: a narrow FoV can leave the translation direction (FoE) out of view, which makes ego-motion noise-sensitive — [mlanthology](https://mlanthology.org/iccv/1998/gluckman1998iccv-ego); [typeset PDF](https://typeset.io/pdf/ego-motion-and-omnidirectional-cameras-55uvf80co4.pdf)
- Follow-ups mentioned in the same search results: Vassallo et al. derived a general Jacobian for the central panoramic model. Shakernia et al. showed that perspective ego-motion algorithms apply directly to "back-projection flow" — [summary via search results / IST OMNIVIS paper](https://isr.tecnico.ulisboa.pt/wp-content/uploads/2015/05/357_02-omnivis-raquel.pdf) (secondary, not verified in full text)
- Bagnato, Frossard & Vandergheynst, "A Variational Framework for Structure from Motion in Omnidirectional Image Sequences" (J. Math. Imaging Vis., 2011): images are mapped to S² (one model for single-viewpoint cameras). Dense depth is solved as graph-based TV-L1 on the sphere. Ego-motion comes from brightness derivatives *without computing optical flow*, alternating between depth and ego-motion — [EPFL infoscience](https://infoscience.epfl.ch/record/142481?ln=en); [PDF](https://doc.rero.ch/record/312555/files/10851_2011_Article_267.pdf); [tech report](https://www.epfl.ch/labs/lts4/wp-content/uploads/2018/10/SFMOmni-TR.pdf). Relevance: this is an existence proof of *direct* (flow-free) sphere-domain rigid estimation on a graph, the same kind of discretization as HEALPix.
- Roberts, Potthast & Dellaert, "Learning General Optical Flow Subspaces for Egomotion Estimation and Detection of Motion Anomalies" (CVPR 2009). Flow at each pixel of a *generalized* imaging system (catadioptric and multi-camera included) is approximately linear in the 6-DoF incremental platform motion: t_j(δ,Z) ≈ T_j δ, with T_j ∈ R^{2×6}. They replace per-pixel depth with the *expected inverse depth* μ_j, using the fact that for ground vehicles "depth at each pixel is often approximately constant" over time. They learn the subspace with a robust probabilistic-PCA extension from sparse flow. A per-pixel Gaussian-mixture outlier process absorbs moving objects and depth violations. They recover dense flow and ego-motion — [PDF](https://dellaert.github.io/files/Roberts09cvpr.pdf). **This is the closest published precedent for a "6-D ego-motion basis + outlier residual" prior on a vehicle-mounted wide-FoV camera.**

**Learned perspective rigid/residual decomposition**
- GeoNet (Yin & Shi, CVPR 2018) reasons "about static and dynamic scene parts separately": rigid flow comes from depth + pose, and a second stage adds residual flow. It uses an adaptive geometric consistency loss for outliers and non-Lambertian regions. Code: github.com/yzcjtr/GeoNet — [arXiv 1803.02276](https://arxiv.org/abs/1803.02276)
- EPC++ (Luo et al.) uses three networks (MotionNet, DepthNet, OptFlowNet) and a "holistic motion parser" that recovers per-pixel 3D motion for rigid background and moving objects. It reports that on non-occluded pixels learned optical flow is usually *more accurate* than rigid flow from depth+pose, and uses that adaptively. Rigid flow *helps in occluded regions* on KITTI 2015 — [arXiv 1810.06125](https://arxiv.org/abs/1810.06125v2)
- RigidMask (Yang & Ramanan, CVPR 2021) models the scene as background plus multiple independently moving rigid bodies. A two-stream network takes rigidity cost maps and rectified scene flow and outputs masks. It ranked 1st on the KITTI scene flow leaderboard at submission (SF error 4.89% vs 6.31%). Code and weights: github.com/gengshan-y/rigidmask — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Learning_To_Segment_Rigid_Motions_From_Two_Frames_CVPR_2021_paper.pdf); [project](https://gengshan-y.github.io/rigidmask); [code](https://github.com/gengshan-y/rigidmask)
- RAFT-3D ("Scene Flow using Rigid-Motion Embeddings", Teed & Deng): RAFT-style iterative updates over per-pixel SE(3) motion with rigid-motion embeddings that softly group pixels into rigid objects — [arXiv 2012.00726](https://arxiv.org/pdf/2012.00726). Needs RGB-D in the original paper.
- DytanVO (Shen et al., ICRA 2023): iteratively and jointly refines ego-motion and motion segmentation from two monocular frames. Reports 27.7% average ATE improvement on real dynamic sequences. Code listed at github.com/Geniussh/DytanVO — [arXiv 2209.08430](https://arxiv.org/abs/2209.08430v1); [CMU RI](https://ri.cmu.edu/publications/dytanvo-joint-refinement-of-visual-odometry-and-motion-segmentation-in-dynamic-environments)
- arXiv 2511.01502 ("Discriminately Treating Motion Components…", 2025) splits ego-motion into rotation (depth-independent flow), tangential translation (∝ inverse depth), and radial translation (depth plus perspective scaling). It strips the rotational flow first using PoseNet's rotation. Pinhole only — [arXiv 2511.01502](https://arxiv.org/html/2511.01502)
- Translational flow by subtracting a homography-based rotational flow is exact under pure rotation and only approximate otherwise in perspective — [arXiv 2606.09088](https://arxiv.org/pdf/2606.09088)

**Learned 360° depth + pose (no flow head found)**
- 360 SfMLearner (Wang et al., ACCV 2018) adapts SfMLearner to panoramas and introduces the PanoSUNCG dataset with GT depth and camera motion — [arXiv 1811.05304](https://arxiv.org/abs/1811.05304)
- BiFuse++: self-supervised 360° depth with DepthNet + PoseNet over adjacent panoramas — [arXiv 2209.02952](https://arxiv.org/pdf/2209.02952.pdf)
- PanoPose (CVPR 2024): self-supervised *scaled* relative pose between panoramas — [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tu_PanoPose_Self-supervised_Relative_Pose_Estimation_for_Panoramic_Images_CVPR_2024_paper.pdf)
- Hasegawa & Ikehata: distortion-aware self-supervised 360° depth directly on ERP — [arXiv 2204.01027](https://arxiv.org/pdf/2204.01027.pdf)
- An NTHU thesis converts ERP to cube faces and adds a cross-face camera-motion consistency loss — [NTHU](https://etd.lib.nthu.edu.tw/thesis/detail/6ca3b568f28d8ab8b2aa4903e9c28f93). Cylindrical-panorama unsupervised depth/ego-motion reports that a wider FoV improves ego-motion — [via search results](https://par.nsf.gov/biblio/10189876) (secondary)

### Inferences
- **[own] Exact spherical rigid-flow formula, usable as a closed-form network layer.** Let p ∈ S² be a ray, ρ(p) = 1/d(p) the inverse range, and (R, t) the camera motion. A static point X = p/ρ maps to p′ = normalize(R(p/ρ − t)), which is exact for finite motion. To first order the tangent flow is v(p) = −ω×p − ρ(p)·(t − (t·p)p). The rotation term carries 3 DoF, has no depth, and has no singularity anywhere on S². The translation term is a known direction field (along great circles away from the FoE +t and toward the FoC −t) times a scalar ρ(p) ≥ 0. **So static background has 1 unknown per node (ρ) plus 6 global numbers (5 up to scale), against 2 per node for dense flow.** In ERP the same field picks up the 1/cos(lat) stretching that hurts OSLO and RAFT near the poles. In the sphere-domain formula it does not.
- **[own] Why this targets OSLO's failure modes.** (i) Far background and sky (ρ≈0), which is most of the upper hemisphere in driving scenes and therefore the north polar cap, has flow that is *pure rotation*. A global 3-parameter estimate pooled over the whole sphere covers it with no per-node noise. (ii) A stopped or slow car (t≈0) gives a flow of exactly −ω×p, and ω≈0 gives *exactly zero*. The "static-calibration" problem in the project memory (a model that overcommits on the static majority) becomes a single global parameter going to zero. Small motions (0–5°, 77% of pixels) are mostly rigid background, which is exactly where a 6-DoF + ρ parametrization removes per-pixel variance.
- **[own] Counterpoint from EPC++:** where matching is reliable, dense learned flow beats rigid flow. The rigid prior should therefore be a *prior/initialization plus a gated residual*, not a hard replacement.
- **[own]** Roberts 2009 is the strongest analogue of the "vehicle + constant expected inverse depth" assumption, and FlowScape (CARLA car rig) satisfies it well.

### Gaps
- I could not confirm whether any 2023–2026 360° *optical-flow* network (PanoFlow, PriOr-Flow, SLOF, etc.) uses a rigid/ego-motion head. Those papers are covered by other notes. My searches for "omnidirectional/equirectangular rigid flow + residual" returned nothing.
- I did not read the Gluckman & Nayar full text (only abstracts and summaries).
- I did not check whether FlowScape ships GT depth and pose. This decides whether a supervised rigid head and the oracle experiments below are possible. Check sfprep's flowscape adapter.

---

## Q2. Local rotation fields: flow on S² as a field of local so(3) generators; piecewise-rigid motion and 3D motion segmentation in 360°

### Takeaway
On S² the infinitesimal rotations are exactly the 3-dimensional space of Killing vector fields (v(p)=ω×p), and this space is isomorphic to so(3). A global camera rotation is one such field. I found no published flow method on S² that represents flow as a spatially varying so(3) field. Piecewise-rigid / multi-body segmentation exists only in perspective (Vogel, Menze & Geiger, RigidMask, RAFT-3D).

### Cited Findings
- The Killing fields on S² form a 3-dimensional space, one per independent rotation in R³. They can be written K_v(q) = v × q, and they form a Lie algebra isomorphic to so(3). A Killing field's flow consists of isometries — [Wikipedia: Killing vector field](https://en.wikipedia.org/wiki/Killing_vector_field); [MDPI Mathematics 13(17) 2880](https://www.mdpi.com/2227-7390/13/17/2880); [PhysicsForums derivation](https://physicsforums.com/threads/killing-vector-fields-on-s2.1057618)
- RAFT-3D represents per-pixel SE(3) motions with embeddings that group pixels into rigid bodies — [arXiv 2012.00726](https://arxiv.org/pdf/2012.00726)
- RigidMask segments background plus an arbitrary number of rigid movers from two frames — [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Learning_To_Segment_Rigid_Motions_From_Two_Frames_CVPR_2021_paper.pdf)

### Inferences
- **[own] The key geometric fact for a head design.** On the sphere, a *locally rigid translation of a small patch* is itself a rotation about an axis perpendicular to the patch. So "piecewise-constant flow" (the planar default) is naturally **piecewise-so(3)**: each region c carries ω_c ∈ R³ and has flow v(p) = ω_c × p inside c. This has 3 parameters per cell. The component of ω_c along p_c (in-plane spin) can be dropped or regularized, leaving 2 effective DoF, matching the planar (u,v). The planar 6-parameter affine model maps to ω_c plus a local divergence/shear term. Note: ω_c × p is not "camera rotation" locally. It is just a seam-free, pole-free way to parametrize a locally constant tangent vector, avoiding the east/north frame that the gauge diagnostic in memory found to be inconsistent (holonomy up to 14° pole-ward).
- **[own]** A global camera rotation is the special case where all ω_c are equal. A Potts or TV penalty on ‖ω_c − ω_c′‖ across neighboring cells therefore has the rotational ego-motion as its zero-cost state. That is an exact, built-in "rigid background" prior.
- **[own]** Divergence: Killing fields are divergence-free (standard Riemannian geometry, ∇_aξ_b antisymmetric; I did not get a citable source in this search). Translation flow has nonzero divergence (expansion at the FoE). A piecewise-so(3) model therefore needs an extra per-cell divergence/scale term, or the ρ·(translation direction) term from Q1, to represent forward motion.

### Gaps
- I found no paper on "flow as a Lie-algebra field on S²" or on 360° 3D motion segmentation. This may be a search-coverage gap: I did not search 360° motion-segmentation datasets specifically.
- I have no citable textbook source for "Killing ⇒ divergence-free".

---

## Q3. Piecewise-parametric / segment / layered flow, and its relevance to HEALPix cells

### Takeaway
Piecewise-planar, slanted-plane and superpixel models (Vogel 2013 → Menze & Geiger 2015 object scene flow) led KITTI before deep learning. They work because driving scenes are mostly a few rigid bodies. Layered PCA flow (Wulff & Black) gives sharp boundaries. HEALPix NESTED cells are a ready-made, equal-area, multiresolution segmentation that could host per-cell parameters, but they are not image-aligned, so they need either image-adaptive merging or a residual.

### Cited Findings
- Piecewise-rigid scene flow (Vogel et al., ICCV 2013) approximates the scene with piecewise-planar superpixels ("slanted planes") and assigns each superpixel the best label from a set of moving-plane proposals. Lv et al. later replaced the discrete labeling with continuous optimization (3rd on KITTI SF, 3–30× faster) — [Janai et al. survey arXiv 1704.05519](https://arxiv.org/pdf/1704.05519); [ECCV paper PDF](https://link.springer.com/content/pdf/10.1007/978-3-319-46484-8_46.pdf) (via search results)
- Object Scene Flow (Menze & Geiger, CVPR 2015) assumes a finite number of rigidly moving objects. Each superpixel is a 3D plane plus an object index. The paper introduced KITTI 2015 (400 dynamic scenes, CAD-fitted vehicles; static background recovered by removing dynamic objects from laser scans) — [project](https://www.cvlibs.net/projects/objectsceneflow/); [abstract](https://www.cvlibs.net/publications/Menze2015CVPR_abstract.pdf); [poster](https://www.cvlibs.net/publications/Menze2015CVPR_poster.pdf)
- PCA-Layers (Wulff & Black, CVPR 2015) uses a sparse layered model where each layer is a PCA-Flow field, merged with an image-aware MRF. It is more accurate than PCA-Flow alone and reported SOTA in occluded regions on Sintel at the time. Runtime is reported as 3.2 s in one source and 3.6 s in another — [CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.html); [MPI](https://is.mpg.de/ps/publications/wulff-cvpr-2015)

### Inferences
- **[own] HEALPix fit.** At NESTED order k there are 12·4^k equal-area cells, and children of a cell are contiguous index ranges. That makes per-cell parameters and coarse-to-fine refinement nearly free (reshape + broadcast). Order 2 gives 192 cells, about 14.7° each, so 192×3 = 576 so(3) parameters. Order 3 gives 768 cells, about 7.3° each. The fixed tessellation cuts across object boundaries, so an image-adaptive soft assignment (RAFT-3D-style embeddings, or per-node softmax over K motion hypotheses) is better than hard cells. The quad-tree is still useful as the *hierarchy* for the hypotheses.
- **[own]** The more useful driving-scene parametrization is "layers = {background rigid (6-DoF + ρ), K movers each with their own SE(3), residual}", i.e. Menze & Geiger / RigidMask on S². Per-cell so(3) is the generic fallback for scenes that are not rigid.

### Gaps
- I did not find SegFlow, PPAC-HD3, or recent learned superpixel-flow numbers. No search budget was left.
- I found no 360°-specific piecewise-planar flow work.

---

## Q4. Spherical bases: VSH, Helmholtz–Hodge, needlets/wavelets on HEALPix, learned compact bases; evidence for predicting flow coefficients

### Takeaway
VSH-based optical flow on S² exists (Kirisits, Lang & Scherzer 2013/14), and it gives a Helmholtz decomposition as a by-product. It was applied to microscopy, not camera ego-motion. Mature differentiable spherical-transform software exists (S2FFT with HEALPix, S2WAV, S2LET, healpy, needlet tooling). Learned low-dimensional flow bases work in perspective: Black et al. 1997, PCA-Flow 2015, Roberts 2009 ego-motion subspace, and "Dimensions of Motion" 2021, where a network predicts a scene representation from which the flow basis is computed.

### Cited Findings
- Kirisits, Lang & Scherzer, "Decomposition of Optical Flow on the Sphere" (GEM Int. J. Geomath., 2014; arXiv 2013). Variational regularization for estimating and decomposing motion on S², with u+v and hierarchical decompositions. The optimization is projected onto finite-dimensional spaces of vector spherical harmonics, and a Helmholtz decomposition comes out "as a natural by-product". Tested on zebrafish time-lapse microscopy — [arXiv 1312.4354](https://arxiv.org/abs/1312.4354); [DOI 10.1007/s13137-013-0055-8](https://dx.doi.org/10.1007/s13137-013-0055-8); [PDF](https://csc.univie.ac.at/paper/KirLanSch14.pdf)
- S2FFT is a JAX library of differentiable, GPU-accelerated spherical harmonic and Wigner transforms. It supports equiangular and HEALPix sampling, and HEALPix trades exact transforms for equal-area pixels — [arXiv 2311.14670](https://arxiv.org/html/2311.14670v3)
- S2WAV, the sister package, provides differentiable scale-discretised wavelet transforms on the sphere and ball, built on S2FFT, in JAX, with partial PyTorch frontend support (v1.0.0) — [PyPI](https://pypi.org/project/s2wav/1.0.0); [arXiv 2402.01282](https://arxiv.org/html/2402.01282v2). HEALPix support for S2WAV's wavelet layer specifically was not confirmed.
- S2LET implements a scale-discretised wavelet transform that reduces to needlets in the axisymmetric case and supports HEALPix (not exact, but numerically accurate) — [ASCL listing via search](https://ascl.net/code/cs/McEwen%2C%20Jason)
- Needlets on HEALPix: Marinucci et al. use HEALPix pixel centres as cubature points and give an implementation recipe — [arXiv 0707.0844](https://ar5iv.arxiv.org/html/0707.0844). Mexican needlets (Scodeller et al.) have a parameter p that trades pixel-space against harmonic-space localization — [arXiv 1004.5576](https://web3.arxiv.org/pdf/1004.5576). The skytools package has cosine/Gaussian needlet bands with alm→needlet transforms — [skytools docs](https://skytools.readthedocs.io/docs/needlet.html). healpy handles map↔alm transforms and up/downgrading — [ASCL](https://ascl.net/code/cs/Gorski%2C%20Krzysztof)
- PCA-Flow (Wulff & Black, CVPR 2015) fits sparse matches to a learned basis of full-frame flow fields, obtained by PCA on flow from four Hollywood movies. Runtime is reported as <300 ms/frame on one CPU in one source and <200 ms in another. It beats LDOF and Classic+NL on accuracy and speed, but is "too smooth" alone, which motivated PCA-Layers — [CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.html); [MPI page](https://is.mpg.de/ps/publications/wulff-cvpr-2015); [PDF](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Wulff2015CVPR.pdf)
- Black et al. (1997, "Learning parameterized models of image motion") learn orthogonal basis flows by PCA and estimate the coefficients directly from intensity derivatives — [PDF](https://www-pequan.lip6.fr/~bereziat/VISION/papers/black97.pdf)
- "Dimensions of Motion: Monocular Prediction through Flow Subspaces" (arXiv 2112.01502) has a network predict a scene representation (disparity, object embeddings), turns it into a flow basis, and trains by flow-reconstruction error through a differentiable SVD. The basis comes from the representation, not directly from the network — [arXiv 2112.01502](https://arxiv.org/pdf/2112.01502). This is the closest learned analogue of "predict ρ, get a rigid-flow subspace".

### Inferences
- **[own] Exact VSH content of rigid motion.** In VSH terms, the rotational flow −ω×p is exactly the degree-ℓ=1 *toroidal* (curl-type, divergence-free) VSH triplet. The translational flow at *uniform* depth, −ρ₀(t − (t·p)p) = −ρ₀∇_S(t·p), is exactly the ℓ=1 *poloidal* (gradient) triplet. **So 6 VSH coefficients at ℓ=1 represent exact rigid ego-motion with constant depth.** Non-uniform depth ρ(p) moves energy to higher ℓ in *both* components, because ρ∇f is not a pure gradient when ∇ρ is not parallel to ∇f. The Helmholtz split therefore does *not* cleanly separate rotation from translation in real scenes. It separates them only at ℓ=1. Use VSH as a *low-ℓ global prior*, not a full flow model.
- **[own]** Truncated VSH or needlets are a poor fit for the residual. Flow discontinuities at object boundaries need high ℓ (Gibbs ringing), which is the same "too smooth" failure as PCA-Flow. Needlets/wavelets localize better than VSH, and a NESTED-HEALPix Haar-like quad-tree basis is the cheapest localized multiresolution option, implementable as reshapes without S2FFT.
- **[own] Tooling.** S2FFT/S2WAV are JAX-first and the OSLO stack is PyTorch, so VSH at small L (≤8) is easier to do as an explicit precomputed matrix on the r4/r6 node set (N×(2·L(L+2)) real basis). That is cheap at r4 (3072 nodes) and avoids the inexact HEALPix quadrature (least-squares fit instead of quadrature).

### Gaps
- I found no paper that predicts VSH coefficients of optical flow with a network.
- I did not find whether any 360° flow work uses spherical wavelets.
- I found no DCT-basis flow-regression network.

---

## Q5. How FlowScape's structure (vehicle camera, mostly static scene) favors an ego-motion + depth head; rigid-flow fraction in KITTI/CARLA

### Takeaway
I did not find a published per-pixel rigid (background) vs dynamic fraction for KITTI 2015 or for CARLA/FlowScape. KITTI evaluates bg/fg separately and only annotates moving vehicles. The case for an ego-motion head rests on Roberts 2009 (vehicle → near-constant expected inverse depth → a 6-D subspace explains most flow, with outliers absorbed) and on the project's own statistics (77% of pixels in 0–5°, much of it near-static background).

### Cited Findings
- KITTI 2015 reports flow error separately for background (bg), foreground (fg = moving objects) and all pixels. GT motion exists only for moving vehicles, not pedestrians or cyclists — [SceneFlowFields++ arXiv 1902.10099](https://arxiv.org/pdf/1902.10099); [KITTI SF benchmark](https://cvlibs.net/datasets/kitti/eval_scene_flow.php)
- KITTI 2015 has 200 train / 200 test scenes at 1242×375 — [DeGraF-Flow arXiv 1901.09971](https://arxiv.org/pdf/1901.09971)
- Roberts et al.: for ground vehicles "the depth at each pixel is often approximately constant", so the flow subspace stays constant over extended motion. Ground plane, buildings and walls stay at roughly constant depth — [PDF](https://dellaert.github.io/files/Roberts09cvpr.pdf)
- EPC++: rigid flow helps in occluded regions on KITTI 2015 — [arXiv 1810.06125](https://arxiv.org/pdf/1810.06125)

### Inferences
- **[own]** The rigid fraction for FlowScape can be *measured* instead of looked up, if GT depth and pose are available or can be fitted (see E0 below). This is a cheap, decisive first experiment.
- **[own]** A vehicle camera is the most favorable rig for a rigid head. Motion is mostly yaw plus forward translation, the FoE/FoC lie near the horizon (equator), and the poles are sky (ρ≈0, pure rotation) and road (large ρ, but a smooth ground plane). Both poles are therefore *low-complexity* under the rigid model, even though they are high-error under ERP-dense models. This is a direct, testable mechanism for OSLO's polar error.

### Gaps
- No published numbers on rigid/static pixel fraction for KITTI, CARLA, or FlowScape.
- Unknown whether FlowScape includes depth or ego-pose GT. If not, ρ and (R,t) can be fitted to GT flow by least squares, but that gives up the independent supervision.

---

## Q6. Concrete HEALPix heads/priors and minimal experiments with success criteria (all [own] unless cited)

### Takeaway
The cheapest high-information path: (E0) oracle-fit rigid + residual models to FlowScape GT to bound the gain. Then (E1) add a rigid head (global ω,t + per-node ρ + gated residual) as the *initialization/prior* of an existing estimator, testing on both OSLO and planar RAFT-small. Results go to the polar and 0–5° bins. Per-cell so(3) and VSH/needlet bases are secondary arms.

### Cited Findings
- GeoNet (rigid then residual) and EPC++ (adaptive rigid/flow consistency) are the precedents for rigid + residual composition — [arXiv 1803.02276](https://arxiv.org/abs/1803.02276); [arXiv 1810.06125](https://arxiv.org/abs/1810.06125v2)
- DytanVO jointly and iteratively refines ego-motion and the motion mask — [arXiv 2209.08430](https://arxiv.org/abs/2209.08430v1)
- Roberts 2009 shows that a per-pixel mixture outlier process over a 6-D motion subspace handles movers and depth violations — [PDF](https://dellaert.github.io/files/Roberts09cvpr.pdf)

### Inferences (proposed designs)
**H1 — Spherical rigid-flow head (main proposal).**
- From the r4 feature/correlation state: (a) global pooling → ω ∈ R³ (axis-angle), t ∈ R³. (b) Per-node ρ = softplus(·) ≥ 0 at r4, convex-upsampled to r6 like the current flow mask. (c) Per-node residual r(p) in the tangent plane (or as per-node ω_p, see Q2) plus a gate m(p) ∈ [0,1].
- Flow: v(p) = log_p( normalize(R(p − ρ(p)·t)) ) + m(p)·r(p). This is the exact finite-motion map with scale absorbed (p/ρ − t ∝ p − ρt), so it is seam-free and pole-free by construction. log_p is the sphere log map, giving geodesic displacement that matches the haversine/SEPE metric.
- Translation scale is unobservable, so normalize ‖t‖ = 1 and let ρ carry the scale (ρ = ‖t‖/d).
- Use inside the RAFT loop: replace the zero-flow initialization with the H1 rigid flow, and let the GRU update ρ, (ω,t) and r. This is a RAFT-3D-style parameter update, but with 6 global + 1 per-node + residual parameters.
- Losses: the current EPE/ERP-px loss on v. An optional sparsity/L1 on m·r keeps static nodes rigid. If GT pose exists, optionally supervise (ω,t).
- It is architecture-agnostic: the same head can sit on planar RAFT-small by computing on ERP pixel rays. Given the memory note that the architecture question is closed and RAFT-small wins, the head is the contribution, and it should be shown on both backbones.

**H2 — Piecewise-so(3) on NESTED cells.** Per-cell ω_c at order 2–3, plus a K-hypothesis soft assignment per node. TV on ω_c across cell adjacency. This is the general non-rigid alternative to H1 and avoids the east/north frame entirely.

**H3 — Low-ℓ VSH prior.** Predict the 6 ℓ=1 coefficients (exact uniform-depth rigid motion) plus optional ℓ≤L as the coarse global flow, with a dense residual. A cheaper, weaker H1 variant (no depth), useful as an ablation: "does depth-modulated parallax matter beyond global rotation + uniform translation?"

**H4 — Multiresolution NESTED Haar/needlet coefficients** as the output parametrization instead of per-node flow. Lowest priority. PCA-Flow's "too smooth" result suggests a basis-only head will not beat dense flow.

### Minimal experiments and success criteria
- **E0 (oracle ceiling, no training, a few hours).** On FlowScape val/test GT flow:
  - (a) fit global rotation only;
  - (b) fit (ω,t) + per-node ρ ≥ 0 by alternating least squares or Gauss–Newton (robust Huber);
  - (c) as (b), but ρ restricted to piecewise-constant on order-3 cells;
  - (d) ℓ=1 VSH (6 params);
  - (e) VSH ℓ ≤ 4, 8, 16.
  
  Report the fraction of pixels with residual < 0.5 / 1 ERP px, stratified by latitude band and by |flow| bin (0–5°, …), and EPE of fit vs GT.
  - **Go criterion for H1:** model (b) explains ≥ 85–90% of pixels to < 1 px, and its residual EPE in the polar caps and 0–5° bin is well below OSLO's current error there. Then the head has room to work.
  - **Kill criterion:** if (b) leaves large polar or small-motion residuals (e.g. GT seam/pole issues, or many non-rigid pixels), stop.
  - Also check the project's known replica360 seam-target defect for an analogue in FlowScape before trusting the fits.
- **E1 (rigid initialization only).** Feed the E0(b) *oracle* rigid flow as the init of the existing OSLO and RAFT-small (no new head), evaluated without retraining and after short fine-tuning. If the GRU cannot exploit even an oracle init, H1 will not help.
  - **Success:** ≥ 15% EPE reduction in the polar bands and 0–5° bin.
- **E2 (learned H1 head)**, same recipe and steps as the current matched OSLO/RAFT-small pair, 2 seeds (consistent with the project's stopping rule).
  - **Success:** OSLO+H1 beats its baseline by more than the 2-seed spread, and closes the 8.6 → 6.3 px gap to RAFT-small (≤ 6.3 px).
  - **Stronger claim:** RAFT-small+H1 also improves. That would make the head a contribution independent of the sphere backbone.
  - Diagnostics: the predicted ω and t against GT pose if available, and the gate-m map (it should light up on vehicles only).
- **E3 (ablations).**
  - H3 (no depth) vs H1, to measure how much depth-modulated parallax adds.
  - H1 with m ≡ 1 (residual everywhere) vs learned m, to test whether the sparsity/static prior is what fixes small motions.
  - H2 vs H1 on dynamic-object pixels.
- **E4 (robustness)**: evaluate with SO(3)-rotated inputs, since the memory notes that SO(3) augmentation was the hidden lever. H1 is equivariant by construction if ω and t are predicted from equivariant pooling, and it should show zero rotated-vs-upright gap.

### Gaps
- Feasibility depends on FlowScape GT depth/pose availability (unverified here).
- No published evidence that rigid heads improve *small-motion* EPE specifically. EPC++ only shows gains in occluded regions, and notes dense flow is better on non-occluded pixels. E0/E1 exist to settle this before any training investment.
