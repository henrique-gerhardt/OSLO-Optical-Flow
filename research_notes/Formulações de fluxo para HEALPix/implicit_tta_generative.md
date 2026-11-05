# Implicit/continuous, per-instance (test-time) and generative/probabilistic optical-flow formulations — fit to HEALPix

Scope note: research done 2026-11-04 with web search and fetch (about 23 calls). Project facts (OSLO numbers, the 2610.32561 ablation, the parked gate idea) come from the project's own memory files. They are cited as [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md) and are not externally verified. "Assessment" bullets are my own reasoning, not published facts.

## Q1. Test-time optimization / adaptation for flow: what gains over the feed-forward initialization?

### Takeaway
The large published per-instance gains come from per-video optimization for long-range tracking (OmniMotion: +21 to +35 AJ over chained RAFT), not from two-frame flow. For two-frame dense flow, naive self-supervised refinement of a supervised network (photometric/census + smoothness, or cycle-consistency) is reported to make it **worse**. The methods that gain use pseudo-label distillation with consistency filtering, or teacher–student consistency. The OSLO plan (census + smoothness + anchor) sits in the regime the literature flags as risky, so the anchor term and filtering are load-bearing.

### Cited Findings
- **Refining Pre-Trained Motion Models** (ICRA 2024). The authors find that "when the initialization is supervised weights, most existing self-supervision techniques actually make performance worse instead of better". They name warp-error (colour constancy) + smoothness methods and cycle-consistency methods as the culprits. Their fix has two stages: (1) keep only cycle-consistent estimates of the pretrained model as sparse pseudo-labels, then densify and re-balance them; (2) fine-tune on those labels with augmentation. They report "reliable gains over fully-supervised methods" on real videos for RAFT-style flow and PIPs-style tracking. The abstract gives no numbers. — [arXiv 2401.00850](https://arxiv.org/abs/2401.00850)
- **Triangular Consistency** (2026). Adapts a pretrained flow model with one epoch of self-supervised teacher–student training on Sintel's unlabeled test split and reports substantial gains on the labeled train split. I did not verify the numbers or venue. — [arXiv 2606.19938](https://arxiv.org/pdf/2606.19938)
- **TTSFlow**, as cited by a 2026 paper: Sintel clean/final EPE 0.847/1.140. On KITTI (EPE 1.99, Fl-all 7.40%) it is described as only "slightly improving on the corresponding GMFlow configuration". The baseline for the Sintel numbers was not visible. — [arXiv 2610.26690](https://arxiv.org/html/2610.26690)
- **Self-supervised motion magnification** fine-tunes at inference on the input video with crop/colour augmentation. The authors say this helps most out of domain. — [arXiv 2311.17056](https://arxiv.org/pdf/2311.17056)
- **"Rethinking Dense Optical Flow without Test-Time Scaling"** (2026) treats RAFT-style iterative refinement as test-time scaling. Examples it reports:
  - GMFlow, 0→1 refinement steps: Sintel Final 2.96→2.48 EPE (C+T setting) and 1.276→1.110 (TSKH).
  - Its own refinement-free single pass: 2.81 / 1.140.
  - It still loses to SEA-RAFT on Sintel and on KITTI (1.99 vs 0.93 EPE).
  - It claims foundation priors only "partially substitute" for test-time compute. — [arXiv 2605.08000](https://arxiv.org/html/2605.08000)
- **OmniMotion** (ICCV 2023) does per-video test-time optimization of a quasi-3D canonical volume with bijections. It is supervised by exhaustive pairwise RAFT flows filtered by cycle and appearance checks, plus a photometric loss. It trains for 200K Adam iterations; the paper gives no GPU-hours and calls it "computationally expensive". Gains of Ours(RAFT) over chained RAFT-C on TAP-Vid:
  - AJ: Kinetics 31.7→55.1, DAVIS 30.7→51.7, RGB-Stacking 42.0→77.5.
  - δavg: DAVIS 46.6→67.5.
  - Evaluation runs at 256×256. — [ar5iv 2306.05422](https://ar5iv.labs.arxiv.org/html/2306.05422)
- **DINO-Tracker** (ECCV 2024) does test-time training on a single video: a residual "Delta-DINO" on frozen DINO-ViT features, using optical flow as the motion signal plus a semantic-correspondence anchor. No runtime figure was found. — [arXiv 2403.14548](https://arxiv.org/html/2403.14548v2), [project page](https://dino-tracker.github.io)
- **Neural Scene Flow Prior** cost: NSFP (runtime coordinate-MLP optimization of scene flow per lidar pair) is "up to 100 times slower" than learned methods. **Fast NSFP** (ICCV 2023) reaches about 30× speedup by replacing the Chamfer loss with a distance transform. The paper finds "the dominant computational burden … is not the network architecture, but the loss function", and the usual coordinate-network speedups "have little effect". — [arXiv 2304.09121](https://arxiv.org/abs/2304.09121v1), [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Fast_Neural_Scene_Flow_ICCV_2023_paper.html)
- **SMURF** (CVPR 2021, unsupervised RAFT trained from scratch, not test-time) reports a 36–40% improvement over UFlow on all benchmarks. — [arXiv 2105.07014](https://arxiv.org/abs/2105.07014)
- **Project fact**: in OSLO's own refinement probe, per-node Adam failed, while a sum-over-HEALPix-quad-tree-levels residual recovered a known 1° rotation to 0.10°. — [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)

### Inferences
- Assessment: the published pattern matches OSLO's own finding. Per-element free parameters fail; structure (a multiscale basis, a coordinate network, a canonical volume) is what makes test-time optimization work. Fast NSFP adds that the loss evaluation is the bottleneck, not the parametrization. On HEALPix that points to the census/warp sampling on r6/r7 as the cost to profile.
- Assessment: 2401.00850 is a direct warning for the current OSLO probe. Census + smoothness on a supervised initialization is the documented failure family. Two cheap safeguards follow: (a) cycle/forward–backward filtering of where refinement may move the flow, and (b) reporting the rate at which refinement makes things worse, not only the mean.
- Assessment: OSLO's error is at the poles and in small motions (0–5°). A photometric or census objective is weakest in exactly those places: low texture near the poles in ERP-derived content, and sub-node motion where the warp residual is noise-dominated. An earlier project diagnostic found that LK flow was anti-correlated with motion at sub-node scale ([project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)). So expect gains on mid-size motions and textured regions, and expect small-motion polar error to resist.

### Gaps
- No primary-source, per-pair EPE gain numbers were found for two-frame TTA of RAFT on Sintel/KITTI. The 2401.00850 tables and the TTSFlow primary paper were not read.
- I found no paper titled "Test-time adaptation for optical flow (TTA-OF)" or "Meta-learning flow adaptation" with reliable details. The DCT-parametrized NSFP variant was not located. FlowFields and variational-on-CNN refinement numbers were not retrieved. ARFlow/UFlow numbers were not retrieved.

## Q2. Multi-resolution grid INRs on the sphere for a per-pair flow field

### Takeaway
HEALPix feature grids plus a tiny MLP already exist as a published INR (HNeR-S, ICML 2024). Spherical-harmonic + SIREN encoders exist for the globe (Rußwurm et al., ICLR 2024). The Cool-Chic family shows that hierarchical latent grids + a tiny decoder can be fit per instance at 300–2,300 MAC/pixel. No published work represents a flow field this way on the sphere. OSLO's quad-tree residual is already a HEALPix multi-level grid, so the natural next step is to add a tiny shared decoder.

### Cited Findings
- **HNeR-S** (Kim et al., ICML 2024) builds spherical feature grids, interpolates them at a (lat, lon) query and feeds a small MLP. It uses an equirectangular grid for weather/climate and a HEALPix grid for CMB (12·Nside² pixels). It reports beating five INR baselines on regression, super-resolution, temporal interpolation and compression. I did not confirm a code release. — [arXiv 2402.05965](https://arxiv.org/html/2402.05965v1), [PMLR](https://proceedings.mlr.press/v235/kim24i.html)
- **Spherical-harmonic positional encoding + SIREN** (Rußwurm et al., ICLR 2024). Double-Fourier-sphere sin/cos embeddings assume a rectangular domain and produce artifacts, "especially at the poles". The paper proposes SH embeddings + SIREN and releases code at github.com/marccoru/locationencoder. — [arXiv 2310.06743](https://arxiv.org/abs/2310.06743), [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/073c8584ef86bee26fe9d639ec648e28-Abstract-Conference.html)
- **Cool-Chic** decoder cost is about 2,300 multiplications per decoded pixel at VVC-level performance. A reduced variant runs at about 300 MAC/pixel and still beats HEVC. The autoregressive entropy model is sequential and slow despite its low MAC count. — [arXiv 2307.12706](https://arxiv.org/pdf/2307.12706), [arXiv 2403.11651](https://arxiv.org/html/2403.11651v1)
- **arXiv 2610.32561** (HEALPix overfitted omnidirectional codec), from its ablation as recorded in project notes:
  - A single HEALPix level costs +44.5% rate.
  - ERP with 8 levels costs only +4.2%, but uses 1.5× the samples.
  - Sphere-nearest context vs RING-order context: +7.1%.
  - It evaluates in 6 viewports because ERP-domain metrics are biased.
  - Takeaway: hierarchy is the big lever and equal-area placement is about a 4% effect. — [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/healpix-inr-codec-paper.md)
- **Hash-encoding anisotropy**: a 2026 analysis finds grid-induced anisotropy in multiresolution hash encodings and proposes rotating the input per level ("Rotated MHE"). It covers Euclidean grids only. — [arXiv 2602.10495](https://arxiv.org/abs/2602.10495)
- **Flow as a coordinate network** (planar precedents):
  - AnyFlow uses an implicit flow upsampler for arbitrary-scale output. — [ar5iv 2303.16493](https://ar5iv.labs.arxiv.org/html/2303.16493)
  - Burst-fusion neural image representations fit a per-scene MLP that maps coordinates to a deformation (homography or flow). — [arXiv 2108.01199](https://arxiv.org/pdf/2108.01199)
  - E-MoFlow uses a coordinate MLP (x,t)→(u,v) as an implicit regularizer for event-camera flow. — [arXiv 2510.12753](https://arxiv.org/pdf/2510.12753)
- **Project fact**: an equiangular twin beat HEALPix at 2k steps; at 20k the result was a partial replication with a sign flip on act₀.₅. Equal-area placement is not the accuracy mechanism. — [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)

### Inferences
- Assessment, the proposed design is a "flow HNeR-S":
  - Per-pair latent grids at HEALPix levels L∈{r3…r7}, stored NESTED so upsampling is a reshape: each parent has 4 children.
  - Sum or concatenate the upsampled latents and decode with a tiny MLP shared across levels. The MLP could also be meta-learned once over the training set.
  - Output a tangent-plane (east, north) flow vector, or better a 3-vector projected to the tangent plane. This avoids the pole singularity of the e/n frame, which project notes found to be non-parallel-transported.
  - Initialize at the network prediction, with the latents representing the residual.
- Assessment: this generalizes the current quad-tree residual. Plain sum-of-levels is the decoder-free special case. Adding the decoder buys nonlinear coupling, such as sharp motion boundaries from coarse latents, at the cost of more optimization difficulty. The codec ablation suggests most of the value is in the hierarchy itself. So a decoder should be added only if the plain multi-level version plateaus.
- Assessment: an SH or SIREN global encoder is a poor fit for per-pair flow refinement. Global bases converge slowly on sharp local motion boundaries and cost O(L²) per query. They may suit only a smooth global component, such as camera rotation.

### Gaps
- No published spherical Instant-NGP or HEALPix hash-grid paper was found, and none representing *vector/flow* fields on S² as an INR. HNeR-S per-instance fitting time and parameter counts were not retrieved. "Neural fields on manifolds" (generalized INRs) was not retrieved.

## Q3. Diffusion / generative flow and spherical/graph diffusion — cost vs a small budget

### Takeaway
Diffusion flow works at the top of the benchmarks but needs large pretraining and multi-sample inference:
- DDVM: 8 samples averaged, multi-dataset pretraining on TPUv4.
- FlowDiffuser: denoising added to a RAFT-like recurrent decoder.
- GenCast shows diffusion on an icosahedral mesh is viable, but at DeepMind scale.

On one RTX 3090 with OSLO's budget, a from-scratch diffusion flow model is not realistic. A cheap version would be a diffusion-style noise-to-flow *refiner* on the existing GRU.

### Cited Findings
- **DDVM** (Saxena et al., NeurIPS 2023):
  - Results: Sintel test EPE 1.75 clean / 2.48 final (fine-tuned). KITTI Fl-all 3.26%, claimed about 25% better than the best published method. Zero-shot: 1.24 / 2.00, and 7.58% Fl-all on KITTI train.
  - Pretraining: AutoFlow + FlyingThings3D + Kubric + TartanAir, plus Palette-style self-supervised pretraining. Efficient U-Net backbone.
  - Resolution: 320×448 pretraining, 320×768 fine-tuning.
  - Inference: averages 8 samples at coarse resolution, then refines coarse-to-fine.
  - Uncertainty: sample variance highlights reflective/transparent regions and boundaries.
  - Step-unrolled training adds about 15% training time on TPUv4. Total TPU count and steps were not retrieved. — [arXiv 2306.01923](https://arxiv.org/html/2306.01923), [abs](https://arxiv.org/abs/2306.01923)
- **FlowDiffuser** (Luo et al., CVPR 2024) uses a "noise-to-flow" strategy: it progressively denoises random flows conditioned on the pair. It adds a Conditional Recurrent Denoising Decoder and Hidden State Denoising, and can be added to existing flow networks. It claims SOTA on Sintel/KITTI. Code: github.com/LA30/FlowDiffuser. The exact numbers were not retrieved (PDF access returned 403). — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_FlowDiffuser_Advancing_Optical_Flow_Estimation_with_Diffusion_Models_CVPR_2024_paper.html), [CVPR poster](https://cvpr.thecvf.com/virtual/2024/poster/30407)
- **GenCast**:
  - Architecture: encoder onto a 6-times-refined icosahedral mesh, then a graph-transformer processor with k-hop attention.
  - Training: standard denoising objective on 40 years of ERA5.
  - Sampling: 20 second-order solver steps = 39 denoiser evaluations per timestep. A 15-day forecast takes about 8 min on one TPUv5. Training compute was not retrieved.
  - Released variants include 1° models on 4- or 5-times-refined meshes. — [arXiv 2312.15796](https://arxiv.org/html/2312.15796v2), [DeepMind repo README](https://cdn.jsdelivr.net/gh/google-deepmind/graphcast@main/README.md)

### Inferences
- Assessment: the useful idea from DDVM/FlowDiffuser at OSLO scale is the **uncertainty-from-samples** byproduct and the noise-injected recurrent decoder, not the generative pretraining. FlowDiffuser's "plug into existing networks" claim makes it the closest fit to OSLO's ConvGRU. Mapping it onto a HEALPix graph is mechanical, since the denoiser is the update operator.
- Assessment: cost risk. DDVM's 8-sample averaging means 8× inference. Diffusion's known strength is multimodality (occlusions, transparency), while OSLO's documented error is at the poles and in small motions, which is a precision problem. There is no evidence that diffusion helps small-motion precision. Priority: low.

### Gaps
- FlowDiffuser numbers, step count and runtime; FlowSeek details; DDVM training-compute totals; the DiffCast paper; and any graph-domain diffusion for dense vector fields were not retrieved.

## Q4. Probabilistic / uncertainty heads and static-vs-motion gating

### Takeaway
SEA-RAFT's mixture-of-Laplace loss is the low-cost, proven option. Its mixture has a fixed-variance L1 component plus a learned-variance component, which yields per-pixel uncertainty and better generalization. The parked per-node static/motion gate fits naturally as a mixture whose component is a delta at zero flow.

### Cited Findings
- **SEA-RAFT** (ECCV 2024 oral):
  - Replaces L1 with a mixture-of-Laplace loss. One component has constant variance (≡ L1, aligned with EPE); the other models ambiguity/occlusion. The authors say this reduces overfitting to ambiguous cases.
  - Also uses direct initial-flow regression and rigid-motion pretraining.
  - Spring: EPE 3.69, 1px outlier 0.36. Code is public. — [ar5iv 2405.14793](https://ar5iv.labs.arxiv.org/html/2405.14793), [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01065.pdf)
- **ProbFlow** (ICCV 2017) estimates flow and uncertainty jointly. It uses mean-field variational inference on an energy-based posterior, with uncertainty given by the marginal entropy. Flow is competitive with the base energy method, and the uncertainty beats post-hoc confidences (authors' claim). — [CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wannenwetsch_ProbFlow_Joint_Optical_ICCV_2017_paper.pdf), [arXiv 1708.06509](https://arxiv.org/abs/1708.06509v1)
- **DDVM** sample variance as uncertainty: see Q3. — [arXiv 2306.01923](https://arxiv.org/html/2306.01923)
- **Project fact (parked gate)**: error mass sits on the static majority. A per-node static/motion gate was estimated to flip global from −16.8% to about +3% on the *pre-correction* flow360 numbers. The flow360 premise was later voided by a convention fix (2026-08-04), so the size of the gain must be re-measured. — [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)

### Inferences
- Assessment: implement as a 3-component mixture per node: {δ(0) static, Laplace(b fixed), Laplace(b learned)}. The δ-component weight *is* the gate, so the gate comes out of the existing loss and needs no extra head-training recipe. It costs one extra head channel set on the r4/r6 graph.
- Assessment: uncertainty is also the tool that makes test-time refinement safe (Q1). Gate the refinement anchor weight by predicted variance, so refinement moves only the low-confidence nodes. This addresses the "self-supervision makes supervised init worse" failure mode.

### Gaps
- No published static/motion gate evaluation for 360° flow was found. The mixture-of-Laplace + zero-delta combination is my proposal, not a published method.

## Q5. Which is cheap on one RTX 3090 and gives fast evidence? Minimal experiments with success criteria

### Takeaway
Ranked by evidence per GPU-hour:
1. Multi-level HEALPix residual + census refinement, with safeguards (already in progress).
2. A mixture-of-Laplace / zero-gate head as a fine-tune.
3. A "flow HNeR-S" per-pair latent-grid + tiny-decoder INR.
4. FlowDiffuser-style noise-to-flow on the GRU.

Diffusion pretraining and OmniMotion-style per-video optimization are out of budget or out of scope (two-frame benchmark).

### Cited Findings
- Per-instance structured parametrization succeeds where per-element optimization fails. OSLO quad-tree: 1° rotation recovered to 0.10° ([project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)). OmniMotion relies on a structured canonical representation ([ar5iv](https://ar5iv.labs.arxiv.org/html/2306.05422)).
- Naive photometric/cycle self-supervision on a supervised init tends to hurt — [arXiv 2401.00850](https://arxiv.org/abs/2401.00850).
- Loss evaluation dominates runtime-optimization cost — [arXiv 2304.09121](https://arxiv.org/abs/2304.09121v1).
- Hierarchy is the dominant lever in HEALPix overfitting (+44.5% rate without it) — [project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/healpix-inr-codec-paper.md).
- The SEA-RAFT MoL loss is a drop-in loss change with public code — [ar5iv 2405.14793](https://ar5iv.labs.arxiv.org/html/2405.14793).

### Inferences
- **E1, test-time multi-level refinement, done properly (≤1 day).**
  - On FlowScape test, refine each pair from the OSLO prediction. Add forward–backward consistency masking and a variance-weighted anchor.
  - Report three things: median and mean ERP-px EPE, the *fraction of pairs made worse*, and a polar vs equatorial and 0–5° vs >5° breakdown.
  - Success: mean EPE improves ≥5% (e.g. 8.6→≤8.2 px) with ≤20% of pairs worse.
  - Stretch: close ≥25% of the gap to RAFT-small (8.6→≤8.0).
  - Kill: improvement <2%, or gains only on >15° motions.
  - Also profile the time split between the census/warp step and the parameter update.
- **E2, per-pair "flow HNeR-S" (1–2 days).**
  - Replace the sum-of-levels with latent grids r3–r7 (C=4–8 channels) and a shared 2-layer MLP (~32 hidden) outputting a 3-vector projected to the tangent plane. Same losses as E1.
  - Variant: meta-learn the MLP weights on 200 training pairs, then freeze them.
  - Success: beats E1 at equal iterations/time on the same pairs, or reaches equal error ≥2× faster.
  - Kill: no gain over the decoder-free sum. This matches the codec ablation, where hierarchy, not the decoder, carries most of the gain.
- **E3, MoL + zero-delta gate head (fine-tune 5–10k steps from the current checkpoint).**
  - Success:
    - Global EPE does not regress by more than 2%.
    - Gate AUROC for |GT|<0.25° vs moving is ≥0.85.
    - Gating outputs at gate>0.5 to zero improves the static-majority error by ≥10%.
    - The calibration (ECE) of the learned b is reported.
  - Re-measure the gate's value on corrected flow360, since the old estimate is void.
- **E4, optional: FlowDiffuser-style noise injection into the GRU update (fine-tune).**
  - Success: ≥3% EPE gain at ≤4 samples, *or* sample variance AUROC beats the E3 learned b.
  - Do this only after E1–E3.
- **Not recommended within budget:** DDVM-style diffusion pretraining (multi-dataset, TPU-scale, 8-sample inference); GenCast-style mesh diffusion (DeepMind scale); OmniMotion-style per-video optimization (200K iterations, built for multi-frame tracking, not two-frame EPE).

### Gaps
- No published numbers for any test-time or INR flow method *on the sphere/360°*, so all spherical expectations above are extrapolation. The success thresholds are my proposal, calibrated to the project's 8.6 vs 6.3 px FlowScape gap ([project memory](file:///Users/henriquegerhardt/.claude/projects/-Volumes-External-SSD-Mestrado-OSLO-Optical-Flow/memory/MEMORY.md)).
