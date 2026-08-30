# Video virtual try-on research record

> Freeze date: **2026-08-30 (Asia/Shanghai)**. This record supports
> `docs/tasks/video-virtual-try-on.md`. It is a primary-source-grounded scoping
> review, artifact audit and evaluation design. No VVT checkpoint was trained,
> downloaded or executed in this batch.

## 1. Review questions

1. Which input contracts count as video virtual try-on (VVT), and which belong
   to image VTON, pose-driven animation, generic video editing or 3D cloth
   simulation?
2. What must remain invariant when the garment changes: person identity, body
   shape, source motion, camera, background, non-target clothing and timeline?
3. How did the field move from explicit flow/warping to diffusion, native video
   DiT, geometry/detail injection, instruction control and rolling streaming?
4. Which milestones are first-public papers, formal proceedings, datasets,
   released artifacts or only author promises?
5. Which datasets truly test cross-garment transfer, back views, occlusion,
   re-entry, long horizons and in-the-wild scenes?
6. Which metrics measure garment fidelity, person/body preservation,
   background, temporal seams, interaction geometry and system latency without
   rewarding copying or static output?
7. Which headline claims fail under protocol inspection, table inspection or
   source-garment leakage tests?

## 2. Search and source protocol

### 2.1 Source order

1. CVF, ECVA, AAAI and ACM proceedings for formal publication status;
2. arXiv version pages and full HTML/PDF for first-public dates, methods,
   tables, ablations and limitations;
3. official author project pages, GitHub and Hugging Face for code, weights,
   data and benchmark status;
4. DOI/Crossref metadata only when a proceedings page did not expose complete
   bibliographic metadata;
5. no survey, leaderboard screenshot, vendor summary or secondary article was
   used as evidence for a mechanism, number or milestone.

### 2.2 Query families

- arXiv API exact phrase: `all:"video virtual try-on"`, sorted by submitted
  date, descending, 200 records;
- `video-based virtual clothing try-on`, `video virtual try on diffusion`,
  `VVTON`, `VVT garment temporal consistency`;
- `site:openaccess.thecvf.com video virtual try-on`;
- `site:ecva.net video virtual try-on`;
- `site:ojs.aaai.org video virtual try-on`;
- DOI and title searches for ACM Multimedia route-forming papers;
- backward and forward citation chaining from FW-GAN, ClothFormer, ViViD,
  CatV2TON, DPIDM, KeyTailor and TripVVT;
- official repository and dataset links found in papers or author pages.

### 2.3 Search snapshot and screening

The exact-phrase arXiv API snapshot returned **26 records** through
2026-08-27. Twenty-five directly concerned VVT; VideoAnydoor was retained only
as an adjacent object-insertion comparison and excluded from the VVT lineage.
The exact phrase query misses works whose titles use `VVTON`, `virtual try-on`,
`video-based`, `textured 3D`, `streaming` or a system name. Citation chaining
therefore added FW-GAN, GPD-VVTO, Tunnel Try-on, VITON-DiT, 3DV-TON, VFR,
FashionChameleon and other route-defining papers.

This is a reproducible **scoping review**, not a PRISMA systematic review and
not a claim that every workshop paper, product or image-only VTON variant was
exhaustively enumerated. Inclusion required a direct change to at least one of:

- the VVT input/output contract;
- garment correspondence, deformation, occlusion or detail preservation;
- source-person, body, background, motion or timeline preservation;
- long-video, interactive, camera-controlled or streaming inference;
- VVT data, benchmark, metric or release surface.

Image VTON was included only as an ancestor or two-stage component. Generic
editing, pose animation and 3D cloth simulation were kept as neighbors unless
the paper directly evaluated the VVT contract.

## 3. Frozen task boundary

The chapter accepts two related contracts.

### 3.1 Source-video VVT

> Given a complete person video and one or more target-garment references,
> optionally with masks, parsing, pose, DensePose, instructions or geometry,
> produce a video on the same source timeline in which the intended person wears
> the target garment while identity, body shape, source motion, camera,
> background and non-target content are preserved.

This is a strict V2V subtask because the source video defines the output
timeline and preservation region.

### 3.2 Pose-driven try-on animation

> Given a person image, target garment and pose/driving sequence, generate a new
> try-on video following the driving sequence.

This is VVT because it transfers a target garment over time, but it is not
strict V2V because no complete source RGB video defines the output timeline.
FW-GAN belongs to this historical branch.

Consequences:

- image-only VTON cannot establish temporal consistency;
- generic inpainting does not by itself establish garment identity, fit or
  dynamic occlusion correctness;
- pose animation can preserve a person's original clothing without learning a
  target-garment transfer contract;
- personalization owns subject identity acquisition; VVT owns the target
  garment as the new fidelity object;
- story/multishot generation owns when a character changes outfit across shots;
  VVT owns the per-shot realization;
- visual try-on does not prove physical size, comfort, pressure, true material
  behavior or return-rate reduction.

## 4. Route taxonomy used in the chapter

| Route | Core mechanism | What it adds | Principal failure |
|---|---|---|---|
| A. Flow/warp + GAN | optical flow, explicit garment warp, recurrent/memory refinement | direct correspondence and early temporal propagation | warp holes, topology failure, texture stretch, occlusion |
| B. Image VTON + temporal guidance | image diffusion or inpainting plus VideoMAE/feature/flow guidance | uses abundant still-image data with limited video training | framewise shortcut, seam and long-horizon drift |
| C. Dual-branch latent diffusion | garment encoder/UNet plus source-video UNet and temporal attention | higher realism and garment-detail conditioning | mask/pose pipeline errors, clip cost, dataset dependence |
| D. Native video DiT | temporal concatenation, full spatiotemporal attention or conditional inpainting | unified image/video or larger video prior | high compute, window boundaries, source leakage |
| E. Pose/geometry/detail injection | human–garment skeleton, textured 3D proxy, keyframe/detail memory | explicit fit, occlusion or high-frequency anchors | estimator failure, shortcut leakage, incomplete 3D artifacts |
| F. Long/streaming memory | overlap, anchors, autoregressive segments, KV cache, rolling diffusion | long video or bounded online cost | accumulation, finite context, look-ahead latency |
| G. Semantic/instruction control | MLLM task tokens, coarse/no mask, reward/post-training | target disambiguation and fewer inference priors | reward gaming, instruction–reference conflict, synthetic affinity |
| H. Interactive/expanded control | hand contact, multi-garment switching, anything try-on, camera trajectory | richer user and camera interaction | scope expansion weakens strict garment/body evidence |

These routes combine; they are not a single chronological replacement ladder.

## 5. Milestone ledger

`First public`, `formal venue` and `artifact` are independent fields.

| First public | Work | Route-defining contribution | Formal status at freeze | Artifact boundary |
|---:|---|---|---|---|
| 2019 | FW-GAN | explicit flow-navigated warping for pose-driven video try-on | ICCV 2019 | author-linked Drive asset; no official GitHub manifest |
| 2020 | ShineOn | design audit for practical video clothing try-on | WACV 2021 workshop | official code repository |
| 2021 | MV-TON | memory refinement over video try-on | ACM MM 2021 | no verified official implementation |
| 2022 | ClothFormer | anti-occlusion warping, flow tracking and dual-stream Transformer | CVPR 2022 | official demonstration/project repository, no method code at freeze |
| 2024 | WildVidFit | image-trained controlled diffusion with VideoMAE and adjacent-frame guidance | ECCV 2024 | project page; no verified code/weights |
| 2024 | GPD-VVTO | garment-aware latent diffusion and temporal attention | ACM MM 2024 | no verified public code/data |
| 2024 | ViViD | 9,700-pair dataset and video diffusion with garment/pose/temporal modules | arXiv preprint | inference code, weights and dataset released |
| 2024 | Fashion-VDM | split CFG and progressive 8-to-64-frame training | SIGGRAPH Asia 2024 | project/repository and sample benchmark; no model code/weights |
| 2024 | SwiftTry | ShiftCaching and TikTokDress for faster long-video inference | AAAI 2025 | training/inference/evaluation code, weights and dataset released |
| 2025 | CatV2TON | one DiT for image/video via temporal concatenation; overlap + AdaCN | arXiv; CVPR 2025 workshop label | inference/evaluation code and weights; no full training code |
| 2025 | DPIDM | explicit human–garment pose interaction in spatial and temporal attention | CVPR 2025 | official page repository without usable method code |
| 2025 | 3DV-TON | textured 3D guidance and HR-VVT benchmark | ACM MM 2025 | inference/weights/data; 3D guidance pipeline remains TODO |
| 2025 | MagicTryOn | DiT, fine/coarse garment features, garment-aware spatiotemporal RoPE | arXiv preprint | official inference ecosystem, weights and demo |
| 2025 | KeyTailor | keyframe-driven garment/background detail injection; ViT-HD | CVPR 2026 | ViT-HD released; paper still says method code will release |
| 2026 | TripVVT | 10,031 in-the-wild reverse triplets, benchmark and coarse-mask baseline | arXiv v1; project claims ECCV 2026 | data/benchmark released; no model code/weights |
| 2026 | FashionChameleon | interactive multi-garment streaming with KV-cache operations | arXiv preprint | author-reported real-time result; verify artifact separately |
| 2026 | iTryOn | hand–garment interaction and timestamped semantic control | arXiv preprint | frontier evidence only |
| 2026 | OmniTryOn | garments, bags, shoes and faces in one video editing interface | arXiv preprint | TryAny-Bench author protocol |
| 2026 | TryOnCrafter | camera-controllable VVT through a renderable 4D try-on proxy | arXiv preprint | author protocol; iterative 3D and diffusion cost remains |
| 2026 | UniVVT | no mask/pose/warp at inference; MLLM semantic bridge | arXiv v2 | no verified public implementation |
| 2026 | InstructVVT | source + garment + instruction, MLLM reward and DiffusionNFT | arXiv v1 | paper promises future release; no verified artifact |
| 2026 | LiveVVT | bounded-look-ahead rolling diffusion with temporal/global memories | arXiv v1 | placeholder repository; no code/weights/latency script |

## 6. Primary evidence ledger and paper review

### 6.1 Explicit correspondence: FW-GAN to ClothFormer

**FW-GAN.** The historical input is a person image, target garment and pose
sequence rather than a complete source-person video. Flow-navigated warping
propagates garment/person appearance and a GAN synthesizes frames. Its lasting
contribution is the correspondence-first decomposition and the VVT data lineage,
not evidence that a modern source-video editing contract was already solved.
Optical flow can align visible texture, but it cannot observe the back of a
single-view garment and fails when topology, occlusion order or large
deformation changes.

**MV-TON and ClothFormer.** MV-TON adds a memory-based refinement route.
ClothFormer separates anti-occlusion garment warping from synthesis, estimates
appearance-flow tracks with ridge regression plus optical-flow correction, and
uses a dual-stream Transformer. This makes temporal garment propagation an
explicit object. The current official ClothFormer repository exposes results
and a README but not a reproducible training/inference implementation; a paper
footnote saying code is available is therefore not sufficient artifact evidence.

### 6.2 Diffusion without a native video training dependency

**WildVidFit.** The task network is image-based; at sampling time a pretrained
VideoMAE reconstruction signal and adjacent-frame latent alignment guide a
sequence, followed by temporal co-denoising for overlap. This is useful because
it leverages still-image try-on data, but the safe claim is not “uses no video
data”: TikTok video frames enter joint training, and temporal quality depends on
a pretrained video model. On VVT the paper reports VFID 4.202 and 39.86% user
preference, slightly worse than ClothFormer's 4.192 and 46.44% under that table.
On its selected TikTok set it reports VFID 9.87 and 73.10% preference. The paper
also contains two internal reporting problems: 13,679 total minus 11,647 train
implies 2,032 VITON-HD test pairs, not the written 2,023; prose swaps the FID and
KID labels for 8.67 and 0.10. Small text/patterns and parsing failure remain.

### 6.3 Clip diffusion: ViViD, GPD-VVTO and Fashion-VDM

**ViViD.** A Stable-Diffusion-1.5-derived main UNet, garment encoder, attention
feature fusion, DensePose encoder and temporal modules are trained jointly on
still images and video. The associated dataset contains 9,700 clothing-video
pairs, 1,213,694 frames and three garment categories at 832x624, split
7,759/1,941. Training uses 24-frame samples at 512x384 on four A100 GPUs for
about 120 hours. On the legacy VVT protocol the paper reports SSIM 0.949,
LPIPS 0.068, VFID-I3D 3.405 and VFID-ResNeXt 5.074. Those values are not
directly comparable with later rescaled VFID tables. The paper calls ViViD
better overall, but ClothFormer's VFID-ResNeXt 5.048 is slightly lower than
ViViD's 5.074 in that same table.

**GPD-VVTO.** The image stage supplies DINOv2 semantic and dense garment
features; video fine-tuning adds garment-aware temporal attention. The paper
reports VVT SSIM 0.928, LPIPS 0.056 and VFID 1.28, and a route-specific change
from 1.79 to 1.28 after temporal modeling. Its self-collected 12,082-pair data
and protocol were not found as public artifacts, so this cannot anchor a public
leaderboard.

**Fashion-VDM.** Split classifier-free guidance separates person/garment
conditioning, and progressive temporal training grows from 8 to 16, 32 and 64
frames. A 64-frame 512x384 clip is generated in one pass in the authors'
protocol. The paper itself identifies body-shape loss from keypoints,
agnostic-boundary artifacts and hallucinated unseen garment views; 10 of 17
failed human-evaluation videos were attributed to agnostic-input errors. This
is stronger evidence than a generic “high fidelity” statement because it names
the pipeline failure surface.

### 6.4 Native video DiT and long clips

**SwiftTry.** Conditional video inpainting receives the garment as condition;
temporal attention improves coherence, while ShiftCaching avoids repeatedly
processing overlapping frames. TikTokDress expands to higher-resolution,
complex-background dance footage. “More than 60% faster” must remain bound to
the authors' resolution, clip, model and hardware protocol.

**CatV2TON.** Person and garment conditions are concatenated on the temporal
axis of one DiT; fewer than one fifth of backbone parameters plus the pose
encoder are trained. Long videos use overlapping clips, previous output frames
as prompts and Adaptive Clip Normalization (AdaCN). Training progresses through
256x192/72 frames, 512x384/48 frames and 832x624/32 frames on four A100 GPUs.
The ViViD-S training subset contains 6,064 videos and 513,896 frontal frames;
its test set is only 180 frontal 64-frame clips. It therefore does not validate
rear-view recovery. AdaCN is not Pareto-dominant in the paper's ablation: some
SSIM/LPIPS/VFID-R values get worse while selected VFID values improve. The
paper also lists insufficient 832x624 clarity and missing physical garment
motion as limitations.

**MagicTryOn.** A Wan2.1 14B DiT variant uses coarse semantic and fine local
garment paths, garment-aware spatiotemporal RoPE and mask-aware loss. The paper
reports, on its ViViD protocol, paired VFID-I 8.4030, VFID-R 0.2346, SSIM
0.9011 and LPIPS 0.0602. Its distilled Turbo variant reports 6.69 seconds for
64 frames at 624x832 on one H20, versus 345.271 seconds for the full model;
these are author measurements, not generic runtime guarantees. Dividing output
frames by wall time gives about 9.57 generated FPS, so the result must not be
rewritten as 24/30-FPS real time; preprocessing and I/O inclusion are unstated.

### 6.5 Pose, 3D and keyframe detail

**DPIDM.** A separately trained garment-pose estimator aligns garment
landmarks with human poses. Pose-aware spatial attention models within-frame
fit; temporal-shift and pose-aware temporal attention model adjacent and
longer motion; a regularizer links consecutive attention. VVT uses 661/130
clips, and the authors train 24-frame samples for 80,000 iterations on 16 A100
GPUs. The reported VVT result is SSIM 0.930, LPIPS 0.041, VFID-I 0.506 and
VFID-R 0.047. “60.5% improvement” means only the relative VFID-I reduction
from GPD-VVTO's 1.280, not every metric. ViViD's SSIM 0.949 is still higher,
and DPIDM removes back-facing ViViD test segments.

**3DV-TON.** A keyframe image try-on initializes a textured 3D clothed proxy,
which is reconstructed and animated to provide per-frame correspondence to a
diffusion generator. This attacks large viewpoint change more directly and
introduces the 130-video HR-VVT benchmark. The official repository releases
inference code, weights and data, but still marks the textured-3D guidance
pipeline and integrated image-try-on step as TODO. Reproducing only the final
diffusion stage is not a full-method reproduction.

**KeyTailor.** Qwen-based keyframe selection, garment dynamic-detail
enhancement (GDDE) and collaborative background detail optimization inject
high-frequency anchors into a Wan2.1-I2V-14B model. ViT-HD contains 15,070
videos at 810x1080, split 13,070/2,000; the dataset is public. The main red-team
risk is a shortcut: GDDE reads the garment region of input-video keyframes.
During paired reconstruction that region already contains the reference/target
garment, so strong paired detail metrics can partially reflect direct access to
target texture. During unpaired transfer the same branch can leak the source
garment. A necessary experiment hides, swaps and corrupts the source-garment
region. “Lightweight” refers to about 0.2057B trainable parameters; inference
still uses a roughly 14.6B model, and the paper reports 281.65 seconds for 64
frames without a stated timing GPU.

### 6.6 In-the-wild triplets and semantic control

**TripVVT.** TripVVT-10K contains 10,031 121-frame, 720x1280 triplets over 30
garment categories. The construction is reverse supervision: a real source
video is the target/ground truth; Nano Banana edits a first frame; Wan-Animate
produces the synthetic different-garment source video; a garment reference is
reconstructed. This gives cross-garment triplets and complex scenes, but not
real before/after clothing captures. TripVVT-Bench has 100 held-out cases from
the same pipeline. Training a Wan2.1-Fun-14B-control backbone uses three stages
of 25k, 25k and 5.5k steps, 49 frames, batch size 2 and eight H100 GPUs.

The paper's text says removing pose has a minor quantitative effect, but the
ablation changes VFID-I from 20.72 to 33.66, SSIM from 0.854 to 0.576 and LPIPS
from 0.105 to 0.274. The table, not the prose, governs the safe conclusion.
Across papers, the same benchmark also drifts: TripVVT reports its own CLIP-I/
CLIP-F as 0.7110/0.9606, while InstructVVT re-evaluates them as
0.9373/0.9876; ViViD LPIPS changes from a suspicious 0.8393 to 0.1343.
Cross-paper ranks are invalid without frozen outputs and evaluator versions.

**UniVVT.** At inference, a Qwen3-VL-2B scene-task perceiver consumes the
source, garment and task, and a semantic bridge conditions a Wan2.1-Fun-Control
generator without masks, pose or warping. Training nevertheless synthesizes
source conditions with separate DensePose-conditioned inpainters; “no geometric
prior at inference” is not “geometric-prior-free data construction.” On
ViViD-S the paper reports paired VFID-I/VFID-R 8.3623/0.1934, SSIM 0.8922 and
LPIPS 0.0456. Its 17.7–38.1x latency claim compares conditioning/preprocessing
only: semantic encoding grows from 2.06 to 2.92 seconds for 30–90 frames versus
36.36–111.39 seconds for explicit geometry. It is not an end-to-end generation
speedup, and the authors say iterative diffusion remains non-real-time.

**InstructVVT.** The inference surface is source video, garment image and
natural-language instruction. An MLLM emits edit tokens, garment tokens carry
fine appearance, source latents anchor structure, and a Qwen3-VL-32B reward
with DiffusionNFT post-trains the generator. Training includes TripVVT-10K,
10,000 MagicTryOn-generated ViViD triplets with no generated-target filtering,
and about 100,000 CatVTON-generated image triplets. This creates teacher and
benchmark affinity risks.

The reported 79.4% preference is first-place votes only in a four-method
shortlist—InstructVVT, MagicTryOn, UniVideo and Kling 1.6—over 50 examples and
20 participants. Reward–human agreement on 100 examples is 65.5% pairwise,
Spearman 0.368 and Kendall 0.310: useful training signal, not a human oracle.
The full RL variant slightly worsens VFID-R and CLIP-F relative to no RL while
improving other measures. The three supervised stages use eight H100 GPUs,
while RL uses 40 H100 GPUs; inference steps and end-to-end latency are not
reported. Text–garment conflicts, extreme length, occlusion and unusual
geometry remain explicit limitations.

### 6.7 Interaction, camera and streaming frontier

**FashionChameleon.** Streaming distillation and explicit KV-cache refresh,
withdrawal and disentanglement support switching among garments. The authors
report 23.8 FPS and 30–180x speedups. The chapter treats this as an
author-reported interactive protocol until hardware, preprocessing, cache state
and public implementation are matched.

**iTryOn.** A 3D hand prior plus global and timestamped language conditions
target hand–garment contact. This expands VVT from passive movement to
interaction, where finger order, grasp continuity and garment response require
special tests beyond generic CLIP similarity.

**OmniTryOn.** A First Frame Wearable Cache and spatiotemporal-consistent RoPE
generalize the interface from clothing to bags, shoes and faces. TryAny-Bench
contains 1,460 paired videos, split 1,243/217, with VQA-based evaluation. This
is a broader wearable/object editing task; evaluator calibration and per-object
geometry must not be hidden by an aggregate score.

**TryOnCrafter.** A 4D try-on proxy combines a clothed 3D Gaussian avatar,
SMPL-X and a background point cloud, then anchors a video DiT to user camera
trajectories. This is the clearest camera-control route. The authors explicitly
list extreme parallax/SMPL-X misalignment and iterative proxy/diffusion cost as
limitations; a 360-degree example is not proof of arbitrary unseen texture.

**LiveVVT.** A 1.3B rolling student maintains a four-chunk staggered-noise
window with three latent frames per chunk, a 21-entry temporal cache and a
persistent memory built from the garment plus a generated frontal try-on
reference. A 14B teacher supports three-stage distillation. The paper evaluates
the 60 longest clips in ViViD-S and ViT-HD (194–420 frames) and TikTokDress-L
(352–693 frames), so this is a length stress test, not a random population
estimate.

At 512x384 the authors report 1.56-second first-chunk latency and 22.39 FPS,
26.35x lower latency and 11.37x higher throughput than a similarly sized
MagicTryOn baseline. The safe interpretation is bounded-look-ahead streaming,
not zero-latency strict causality: each frame still needs a mask, DensePose and
garment-agnostic image; deployment also requires a frontal A-pose keyframe.
Only training hardware—eight A100 80GB GPUs—is disclosed in the visible
protocol, not the timing GPU or whether preprocessing is included. Full memory
reduces throughput from 26.74 to 22.39 FPS and worsens LPIPS from 0.088 to
0.099 while improving VFID/SSIM. Severe occlusion, extreme rotation, fast
motion and illumination shifts remain failure cases.

## 7. Dataset ledger

| Dataset | Scale and split | What it can support | What it cannot support |
|---|---|---|---|
| VVT | 791 clips, 256x192; commonly 661/130; total frames conflict across papers: 205,675 vs 190,101 | legacy paired protocol and method history | modern resolution, broad actions/backgrounds, reliable cross-garment proof, stable frame-count provenance |
| ViViD | 9,700 pairs, 1,213,694 frames, 832x624; 7,759/1,941 | three garment categories, higher resolution, image-video training | unfiltered benchmark comparability when papers remove back views |
| ViViD-S | 6,064 training videos/513,896 frontal frames; 180 test clips x 64 frontal frames | controlled frontal long-clip comparison | rear-view detail, full distribution, arbitrary occlusion/re-entry |
| TikTok subset in WildVidFit | 165 selected upper-body videos; 130/35, 34,933/9,816 frames | harder dance and occlusion than VVT | full TikTok distribution, all garment categories, unbiased user sample |
| TikTokDress | released high-resolution dynamic set | complex motion and speed studies | claims outside its capture/garment distribution |
| HR-VVT | 130 high-resolution videos | 3D-guidance test surface | large-scale population claims |
| ViT-HD | 15,070 e-commerce videos at 810x1080; 13,070/2,000 | high-resolution detail and longer display videos | severe occlusion after collection filtering; general street scenes |
| TripVVT-10K | 10,031 reverse triplets, 121 frames, 720x1280, 30 categories | cross-garment/in-the-wild training with target person preservation | real before/after capture; freedom from generator contamination |
| TripVVT-Bench | 100 held-out cases | fixed hard examples and multi-person ambiguity | broad statistical confidence; pipeline-independent test distribution |
| TryAny-Bench | 1,460 paired videos; 1,243/217 | multiple wearable/object categories | pure garment-only ranking without category-stratified analysis |

Every benchmark split should isolate person identity, capture session and
garment/SKU. Frame-level random splits leak nearly identical content.

## 8. Metric and protocol audit

| Claim | Minimum evidence | Common proxy failure |
|---|---|---|
| target garment preserved | crop retrieval plus calibrated DINO/CLIP, logo/text OCR, line/pattern/material annotation, human comparison | CLIP-I preserves category/color but misses logo, seam and weave |
| person/body preserved | face/identity, silhouette/body-shape, skin/hair/hands, pose and motion | SSIM dominated by background or rewards source copying |
| background/non-target preserved | outside-edit-region L1/DINO and region review | mask definition changes the support and can hide collateral edits |
| temporal coherence | optical-flow warp error, temporal LPIPS, seam, occlusion/re-entry and long-horizon tests | adjacent CLIP-F rewards static/over-smoothed output |
| realistic fit | neckline/sleeve/hem/contact/order annotations and human review | generic perceptual score cannot verify geometry or physical fit |
| distribution quality | VFID with exact backbone, sample count, resolution, frames and preprocessing | VFID values from different implementations are not comparable |
| MLLM quality | frozen model/version/prompt/frame sampler plus independent human calibration | reward leakage, model preference and evaluator drift |
| real-time system | end-to-end P50/P95 first-visible latency, sustained FPS, jitter, memory and preprocessing on stated hardware | FPS omits initial buffer, pose/mask extraction or look-ahead |

Paired SSIM/LPIPS require pixel-aligned ground truth. In an unpaired target-
garment transfer, there is no ground-truth pixel frame, so reporting paired
reconstruction values as transfer quality is invalid.

VFID-ResNeXt also has a roughly 100x scale drift: the same VVT/FW-GAN entry is
0.1215 in CatV2TON and 12.15 in DPIDM, while VVT/ClothFormer is 0.0505 versus
5.048. ViViD is not explained by a single scale factor: CatV2TON's rerun gives
VFID-I/VFID-R/SSIM/LPIPS 3.793/0.0348/0.822/0.107, whereas the original table
gives 3.405/5.074/0.949/0.068. CatV2TON calls its 180x64 frontal subset
ViViD-S in the prose but ViViD in the table caption. UniVVT and InstructVVT
then reuse nearly identical baselines under stated 512x384 and 180x61,
832x624 protocols without saying whether the rows were rerun. These values can
only support within-table comparisons, not a merged leaderboard.

## 9. Artifact status at the freeze date

| Work | Verified release surface |
|---|---|
| FW-GAN | author-linked Drive asset; no official GitHub metadata in registry |
| ClothFormer | official README/demo repository; no training/inference code, weights or data download |
| ViViD | inference code, model weights and dataset; no complete training entry documented |
| Fashion-VDM | project-page repository and benchmark sample; no model code/weights |
| SwiftTry | training, inference and evaluation code; weights and TikTokDress released |
| CatV2TON | image/video inference and evaluation, 256/512 weights and test resources; no training code |
| DPIDM | static official project-page repository; no usable code/weights/data |
| 3DV-TON | inference code, weights and HR-VVT; textured-3D guidance pipeline remains TODO |
| MagicTryOn | official repository, inference assets, checkpoints and demo under the stated non-commercial license |
| KeyTailor | ViT-HD dataset released; no verified official method code or weights |
| TripVVT | project/qualitative assets and TripVVT-10K/Bench; no training/inference code or checkpoints |
| UniVVT | no verified public method repository at freeze date |
| InstructVVT | paper promises release; no verified public implementation at freeze date |
| LiveVVT | official placeholder repository; code, weights and timing script not released |

`official-project-page`, `official-research-artifact` and `official-code` are
kept distinct in the central registry. A repository with only HTML/README is
not code release, and inference-only release is not training reproducibility.

## 10. Red-team claims and required counterfactuals

1. **Source-garment leakage:** use source and target garments with conflicting
   color/logo; hide, blur, swap and shuffle the source garment region.
2. **Rear-view identifiability:** separate “not visible in the reference” from
   “model failed”; require multi-view references or mark the target unknown.
3. **Occlusion recovery:** score before occlusion, during occlusion and after
   re-entry rather than averaging the whole clip.
4. **Body-shape rewrite:** compare silhouette, shoulders, waist, limbs and face
   outside the target clothing support.
5. **Static-metric gaming:** include motion amplitude and source-flow agreement
   so a frozen or blurred output cannot win temporal metrics.
6. **Window seams:** place turns, zooms and arm crossings exactly at clip/rolling
   boundaries and compare overlap, memory and no-memory variants.
7. **Reward leakage:** keep human gold videos, prompts and garments outside
   reward/post-training data; freeze the judge version and prompt.
8. **Timing completeness:** include parsing, DensePose, mask smoothing, garment
   encoding, keyframe synthesis, look-ahead buffer, decoding and I/O in P50/P95.
9. **Physical-fit boundary:** compare visual plausibility separately from real
   size/comfort/pressure/drape evidence; do not infer the latter.
10. **Rights and safety:** require consent, garment/logo license, provenance,
    deletion path, demographic/size fairness and a prohibition on involuntary
    clothing removal or exposure.

## 11. Teaching visual record

### 11.1 Final asset

- File: `assets/diagrams/video-virtual-try-on-conservation-contract.png`
- Built-in image generation tool followed by a targeted semantic correction.
- Pixel dimensions: **1672 x 941** (approximately 16:9)
- Color space: RGB, non-interlaced PNG
- File size: **1,992,891 bytes**
- SHA-256:
  `52f5f786280c19f3dcb17237cbb2fbf43723c8911fd26e7b5773365e75d66703`

### 11.2 Normalized generation/edit intent

The base intent requested an original, high-contrast, vector-like teaching
infographic: source person video, target garment and optional pose/mask enter
four operations—correspond, occlude, propagate and synthesize—then pass three
independent conservation ledgers (person/body, garment, scene) and five output
gates (fit, garment detail, person, background, temporal). The bottom strip
shows detail smear, body drift, background leak, seam flicker and occlusion
failure. No paper layout, logo, benchmark score or watermark was requested.

The targeted correction strengthened the visible distinction of `BODY DRIFT`
and `OCCLUSION FAIL` without changing the rest of the layout. This description
records the final semantic intent; it is not presented as a verbatim hidden
tool prompt.

### 11.3 Visual inspection

- [x] Original-size inspection passed; no clipped label or broken arrow found.
- [x] The input, operations, output and independent ledgers are visually
  distinct.
- [x] All five bottom failure families are readable and semantically different.
- [x] A grayscale rendering at 1200x675 remains interpretable; labels, borders,
  line structure and icons carry meaning without hue.
- [x] The chapter supplies a detailed caption and sequential prose alternative.
- [x] No paper figure, model logo, metric score or watermark appears.

The chapter also contains two editable Mermaid diagrams with `accTitle` and
`accDescr`. They explain task routing and the evaluation gate sequence rather
than duplicating the generated visual.

## 12. Proposed experiment and reproduction boundary

`TryOnLedger-1` is a proposed matched evaluation, not a completed benchmark
run. It freezes source videos, target garment references, instructions, masks,
preprocessing, random seeds, resolution, frame count, sampler, evaluator
versions and hardware. It compares one explicit correspondence baseline, one
latent-diffusion baseline, one video-DiT baseline and one long/streaming route.

Two tracks are required:

- **paired reconstruction:** the target garment is the source garment; useful
  for pixel metrics but vulnerable to source-copy shortcuts;
- **unpaired transfer:** the target garment differs in category, color, logo or
  structure; no pixel ground truth, so garment/person/background ledgers and
  human judgment replace SSIM/LPIPS.

Hard cases include crossed arms, hair and bag occlusion, hand tugging, sitting,
fast motion, zoom, rear view, 180/360-degree turn, out-of-frame re-entry,
multi-person ambiguity, 60-second duration and a challenge placed at every
window boundary. Each ledger has a hard gate; a high average cannot compensate
for exposed skin, wrong person, wrong logo or background rewrite.

No checkpoint was run, so this batch can claim a deep primary-source review,
artifact audit, teaching visual and executable evaluation contract. It cannot
claim independent confirmation of any paper's visual quality, physics,
generalization, speed, memory or safety.

## 13. Validation record

Integration checks are recorded only after they are actually run.

| Check | Current state |
|---|---|
| Markdown lint | passed for chapter and research log with markdownlint-cli2 0.20.0 / markdownlint 0.40.0; 0 errors |
| Local links and images | passed; every relative Markdown target in both files resolves |
| Reference anchors | passed; 37 cited references, 37 definitions, no missing or uncited entry |
| External primary links | 37 unique URLs checked; 0 hard 404/410 failures; three ACM DOI resolvers returned automation-only 403 responses |
| Mermaid render and grayscale | both Mermaid blocks rendered with mermaid-cli 11.12.0; generated visual passed original-size and grayscale inspection |
| Generated visual | local image link resolves and the reviewed diagram is integrated with alt text, caption and sequential fallback |
| Timeline image preservation | not exercised in this commit; timeline remains unchanged; future invariant is 75 HTML image source/alt pairs |
| Bibliography generation | not exercised in this commit; registry remains at 132 entries, A–N pending cross-page integration |
| Diff, credential and size hygiene | passed at the final staged snapshot: no whitespace error, credential pattern hit or file above 2 MB |
