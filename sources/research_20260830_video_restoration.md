# Video restoration research and integration log

> Freeze date: **2026-08-30 (Asia/Shanghai)**. Corresponding chapter: [`docs/tasks/video-restoration.md`](../docs/tasks/video-restoration.md).

## 1. Why this audit was necessary

The repository previously routed super-resolution, denoising, deblurring and spot removal from `video-to-video.md` to `video-inpainting.md`. That target chapter has a strict mask-completion contract, while degradation restoration is an inverse problem over observed but corrupted pixels. The old navigation therefore created a classification error, not merely a missing optional topic.

The corrected distinction is:

```text
degradation restoration: Y = D_phi(X) + N -> X_hat
inpainting/completion:    Y = M * X, M -> missing support
frame interpolation:     known temporal samples -> missing time coordinates
semantic editing:        source video + instruction -> allowed counterfactual change
```

Hybrid cases remain possible. Bitstream corruption can use metadata to estimate damaged support, then combine masked recovery and restoration; this does not erase the contracts above.

## 2. Review questions

1. Which degradation families belong in a Video Generation 101 restoration chapter?
2. Which technical routes are genuinely different from one another?
3. Which papers changed the task, evidence or deployment contract rather than only improving a score?
4. How should formal publication, preprint, code, weights and author-reported speed be labeled?
5. Which tests can falsify fidelity, temporal consistency, real-degradation robustness and hallucination claims?

## 3. Search and source policy

### 3.1 Query families

Primary-source searches included combinations of:

- `video restoration super-resolution deblurring denoising CVPR 2024 2025 2026`;
- `real-world video super-resolution diffusion temporal consistency`;
- `streaming one-step video super resolution CVPR 2026`;
- `blind bitstream corrupted video recovery metadata diffusion`;
- exact-title lookups for EDVR, BasicVSR, BasicVSR++, RealBasicVSR, RVRT, VRT, SATeCo, Upscale-A-Video, VideoGigaGAN, PatchVSR, DiffVSR, TurboVSR, SeedVR, SeedVR2, FlashVSR, DGAF-VSR, STCDiT and DTG-Restore.

### 3.2 Inclusion rules

A work was included when it satisfied at least one of these conditions:

- introduced a reusable restoration mechanism or task decomposition;
- changed the degradation or deployment contract;
- provided a formal milestone for a 2024–2026 route;
- supplied a diagnostic benchmark, hybrid boundary case or falsifier;
- exposed a release/evidence distinction that prevents a common misclassification.

Image-only restoration papers were excluded unless they were direct priors adapted by a cited video work. Application-specific satellite, medical or event-camera papers were not used as the main historical spine. Challenge reports were treated as protocol evidence, not as proof that a single method is generally best.

### 3.3 Evidence labels

- **A:** formal peer-reviewed proceedings/journal page plus traceable paper artifact;
- **B:** arXiv preprint plus author project or repository;
- **C:** author/organization demo without a matching public paper artifact;
- **D:** secondary discovery lead only; not used for a technical claim.

Formal venue and first-public-preprint year are recorded separately. VideoGigaGAN, for example, first appeared as a 2024 preprint and was formally published at CVPR 2025. SeedVR remained a preprint at the freeze date, while SeedVR2 had an ICLR 2026 proceedings page.

## 4. Correct task boundary

| Contract | Known evidence | Unknown part | Primary loss/risk |
|---|---|---|---|
| Super-resolution | all LR pixels | HR samples/high frequencies | incorrect invented texture |
| Deblurring | blurred exposure integral | sharp latent frames/motion kernel | ringing, wrong motion boundaries |
| Denoising | noisy pixels | clean signal/noise realization | detail removal or noise hallucination |
| Compression restoration | decoded frames + optional metadata | pre-quantization signal | block/ringing amplification, GOP drift |
| Weather/low-light removal | corrupted visible stream | clean radiance and degradation | unseen-weather failure, color/identity drift |
| Inpainting | valid pixels + missing mask | missing support | outside-mask leakage |
| Frame interpolation | endpoint/neighbor frames | unsampled times | wrong temporal position/occlusion order |

The chapter uses “restoration” for the first five rows and “completion” for the mask row. “Enhancement” is broader and may legitimately change appearance; it cannot be used as a synonym for faithful recovery in evidence-sensitive settings.

## 5. Primary-source milestone map

| First release / formal venue | Work | Route contribution | Evidence boundary |
|---|---|---|---|
| 2017 / CVPR 2017 | [Deep Video Deblurring](https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Deep_Video_Deblurring_CVPR_2017_paper.html) | end-to-end neighboring-frame aggregation and high-frame-rate blur synthesis | synthetic blur does not span all real shutter effects |
| 2019 / CVPRW 2019 | [EDVR](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.html) | pyramid/cascading deformable alignment and temporal-spatial attention fusion | formal workshop paper; not universal real-degradation proof |
| 2020 / CVPR 2020 | [FastDVDnet](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html) | fast multi-level video denoising without explicit flow estimation | a denoising milestone, not a general SR/deblur result |
| 2021 / CVPR 2021 | [BasicVSR](https://openaccess.thecvf.com/content/CVPR2021/html/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.html) | propagation, alignment, aggregation and upsampling as essential components | bidirectional propagation reads future frames |
| 2021 / ICCV 2021 | [Deep Blind VSR](https://openaccess.thecvf.com/content/ICCV2021/html/Pan_Deep_Blind_Video_Super-Resolution_ICCV_2021_paper.html) | explicitly blind degradation estimation/restoration | blind within the paper’s modeled distribution |
| 2021 / ICCV 2021 | [COMISR](https://openaccess.thecvf.com/content/ICCV2021/html/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.html) | compression-aware VSR | codec-aware evidence is configuration-bound |
| 2022 / CVPR 2022 | [BasicVSR++](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html) | second-order grid propagation and flow-guided deformable alignment | stronger propagation can still amplify bad evidence |
| 2021 preprint / CVPR 2022 | [RealBasicVSR](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.html) | pre-propagation cleaning, dynamic refinement, VideoLQ and training trade-offs | no-GT real video needs human and diagnostic evaluation |
| 2022 / NeurIPS 2022 | [RVRT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html) | local parallel clips plus global recurrence and guided deformable attention | offline/hybrid architecture, not a streaming claim |
| 2022 preprint / TIP 2024 | [VRT](https://arxiv.org/abs/2201.12288), [journal DOI](https://doi.org/10.1109/TIP.2024.3372454) | parallel video restoration Transformer across multiple degradations | first release and formal publication dates differ |
| 2024 / CVPR 2024 | [FMA-Net](https://openaccess.thecvf.com/content/CVPR2024/html/Youk_FMA-Net_Flow-Guided_Dynamic_Filtering_and_Iterative_Feature_Refinement_with_Multi-Attention_CVPR_2024_paper.html) | joint SR+deblur with flow-guided dynamic degradation/restoration filtering | joint task is still tied to the declared degradation generator |
| 2024 / CVPR 2024 | [Blur-aware sparse Transformer](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Blur-aware_Spatio-temporal_Sparse_Transformer_for_Video_Deblurring_CVPR_2024_paper.html) | blur-aware sparse spatiotemporal attention | direct deblur route, not generic restoration evidence |
| 2024 / CVPR 2024 | [SATeCo](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Learning_Spatial_Adaptation_and_Temporal_Coherence_in_Diffusion_Models_for_CVPR_2024_paper.html) | frozen image SR prior with spatial adaptation and temporal alignment | stochastic prior can hallucinate despite sharper output |
| 2023 preprint / CVPR 2024 | [Upscale-A-Video](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.html) | local temporal layers, global flow-guided latent propagation, adjustable fidelity-generation trade-off | user-controlled hallucination risk must be disclosed |
| 2023 preprint / ECCV 2024 | [MGLD-VSR](https://eccv.ecva.net/virtual/2024/poster/2534) | motion-guided latent diffusion for real-world VSR | author benchmark evidence |
| 2023 preprint / ECCV 2024 | [StableVSR](https://eccv.ecva.net/virtual/2024/poster/1051) | temporally consistent perceptual detail synthesis | perceptual detail is not hidden-ground-truth recovery |
| 2024 / ECCV 2024 | [VD-Diff](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6210_ECCV_2024_paper.php) | wavelet-aware dynamic Transformer plus compact diffusion prior for deblurring | generated high frequency needs fidelity checks |
| 2025 / WACV 2025 | [FLAIR](https://openaccess.thecvf.com/content/WACV2025/html/Zou_FLAIR_A_Conditional_Diffusion_Framework_with_Applications_to_Face_Video_WACV_2025_paper.html) | conditional diffusion specialized to face video restoration | face perceptual quality cannot substitute for identity fidelity |
| 2024 / CVPR 2024 | [Diff-TTA](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Genuine_Knowledge_from_Practice_Diffusion_Test-Time_Adaptation_for_Video_Adverse_CVPR_2024_paper.html) | diffusion test-time adaptation for seen/unseen adverse weather | online adaptation cost and drift must be reported |
| 2024 preprint / CVPR 2025 | [VideoGigaGAN](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_VideoGigaGAN_Towards_Detail-rich_Video_Super-Resolution_CVPR_2025_paper.html) | generative image upsampler extended to temporally stable 8x video SR | 8x is a protocol setting, not a universal guarantee |
| 2025 / CVPR 2025 | [PatchVSR](https://openaccess.thecvf.com/content/CVPR2025/html/Du_PatchVSR_Breaking_Video_Diffusion_Resolution_Limits_with_Patch-wise_Video_Super-Resolution_CVPR_2025_paper.html) | patch/global dual conditions and joint modulation for high-resolution output | patch seams and global-semantic errors are new risks |
| 2025 / ICCV 2025 | [DiffVSR](https://openaccess.thecvf.com/content/ICCV2025/html/Li_DiffVSR_Revealing_an_Effective_Recipe_for_Taming_Robust_Video_Super-Resolution_ICCV_2025_paper.html) | progressive learning for complex degradation plus interweaved latent transition | robust under tested distributions, not every real degradation |
| 2025 / ICCV 2025 | [TurboVSR](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_TurboVSR_Fantastic_Video_Upscalers_and_Where_to_Find_Them_ICCV_2025_paper.html) | high-compression VAE, factorized conditions and shortcut few-step sampling | speed is hardware, resolution, duration and pipeline bound |
| 2025 / CVPR 2025 | [Metadata-guided bitstream recovery](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Blind_Bitstream-corrupted_Video_Recovery_via_Metadata-guided_Diffusion_Model_CVPR_2025_paper.html) | estimates corruption support from metadata before masked recovery/refinement | hybrid boundary case, not ordinary full-frame VSR |
| 2025 / arXiv | [SeedVR](https://arxiv.org/abs/2501.01320) | generic arbitrary-length/resolution diffusion-Transformer restoration claim | preprint at freeze date |
| 2026 / ICLR 2026 | [SeedVR2](https://proceedings.iclr.cc/paper_files/paper/2026/hash/444d69470b24ded080183c907b711bbf-Abstract-Conference.html) | one-step diffusion adversarial post-training and adaptive window attention | one step does not remove hallucination or end-to-end cost |
| 2026 / CVPR 2026 | [FlashVSR](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_FlashVSR_Towards_Real-time_Diffusion-Based_Streaming_Video_Super_Resolution_CVPR_2026_paper.html) | three-stage distillation, sparse attention and tiny decoder for one-step streaming VSR | 17 FPS is author-reported for a named A100/output setting |
| 2026 / CVPR 2026 | [DGAF-VSR](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Rethinking_Diffusion_Model-Based_Video_Super-Resolution_Leveraging_Dense_Guidance_from_Aligned_CVPR_2026_paper.html) | dense guidance from aligned adjacent features | author comparisons; alignment still fails under occlusion |
| 2026 / CVPR 2026 | [STCDiT](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html) | motion-aware VAE segments and anchor-frame guidance | anchor/segment boundary must be stress-tested |
| 2026 / CVPR 2026 | [DTG-Restore](https://openaccess.thecvf.com/content/CVPR2026/html/Yesiltepe_DTG-Restore_Training-Free_Diffusion_Refinement_for_Generative_Video_Super-Resolution_CVPR_2026_paper.html) | time-decoupled unconditional guidance and GenWarp480 | training-free means no parameter update, not zero compute |

## 6. Technical synthesis

The papers do not form a single replacement ladder. They select from orthogonal axes:

| Axis | Options |
|---|---|
| Degradation | known / estimated / blind; single / compound; photometric / structural / missing support |
| Temporal access | single frame / local window / bidirectional full clip / recurrent / causal streaming |
| Alignment | no explicit alignment / optical flow / deformable conv / deformable attention / feature correlation |
| Prior | regression / adversarial / image diffusion / video diffusion / hybrid diffusion feature prior |
| Output scale | fixed / arbitrary scale / tiled or patch-wise / streaming resolution |
| Sampling | deterministic feed-forward / iterative diffusion / distilled few-step / one-step adversarial post-training |
| Evidence | paired GT / unseen synthetic degradation / paired capture / no-GT real video / AIGC artifact benchmark |

A method can be recurrent and diffusion-based, patch-wise and Transformer-based, or one-step and adversarially post-trained. “Transformer”, “diffusion”, “real-world” and “streaming” therefore do not identify mutually exclusive families.

## 7. Evaluation contract

### 7.1 Minimum test matrix

```text
content axes: text / faces / repeated texture / thin lines / fast motion / occlusion / scene cut
degradation axes: blur / resize / noise / compression / operator order / compound severity
distribution axes: matched / parameter-held-out / generator-held-out / codec-camera shift / real capture
system axes: short / long / tile boundary / cold-warm / offline-causal
```

### 7.2 Required evidence

- fidelity: paired full-reference metrics where GT exists, plus re-degradation consistency;
- temporal: flow/track-aligned error, flicker spectrum and long-horizon drift;
- perceptual: blinded pairwise human evaluation and calibrated no-reference metrics;
- hallucination: OCR, identity, object count, seed sensitivity and high-risk manual audit;
- systems: full pipeline p50/p95, NFE, memory, hardware, clip length, resolution and I/O/VAE inclusion.

### 7.3 Claim downgrade rules

- perceptual improvement with reduced OCR/identity fidelity -> “enhancement,” not faithful recovery;
- improvement only on the training degradation generator -> “matched synthetic restoration,” not real-world robustness;
- no future-frame disclosure -> no online/streaming claim;
- denoiser-only speed -> no end-to-end real-time claim;
- interface accepts arbitrary length but no drift/seam curve -> no long-term stability claim;
- paper/project only, no clean reproduction -> author-reported result.

## 8. Visual asset record

Final project asset: [`assets/diagrams/video-restoration-contract.png`](../assets/diagrams/video-restoration-contract.png).

### 8.1 Learning objective

Show in one scan that:

1. full-frame degradation restoration starts from observed but corrupted evidence;
2. alignment/fusion, propagation/attention and generative priors are alternative or composable routes;
3. fidelity, temporal stability and perceptual detail are independent gates;
4. missing support enters a separate inpainting contract.

### 8.2 Built-in image generation prompt

The project used the built-in image generation path with the `scientific-educational` use case. Required verbatim labels included `VIDEO RESTORATION CONTRACT`, `DEGRADATION CONTRACT`, `RESTORATION ROUTES`, `ACCEPTANCE GATES`, `BLUR`, `DOWNSAMPLE`, `NOISE`, `COMPRESSION`, `ALIGN + FUSE`, `PROPAGATE + ATTEND`, `GENERATIVE PRIOR`, `EVIDENCE FIDELITY`, `TEMPORAL STABILITY`, `PERCEPTUAL DETAIL`, `MISSING SUPPORT → INPAINTING`, `SEPARATE TASK CONTRACT`, and `RESTORE OBSERVED EVIDENCE; DO NOT INVENT A NEW SCENE.`

The first output contained transparent regions that rendered as black in dark contexts. A targeted image edit changed only the background/transparent regions to solid white while preserving all labels and layout. The first generated file was not retained as the project asset; the corrected opaque file was copied into the repository.

### 8.3 Asset checks

At integration time:

```text
pixel size: 1672 x 941
format: PNG
alpha: no
SHA-256: bd2842ad8150ff276d6eabc1af609fe8f928f117c5d8632dc8a9025ad42f45c0
```

The chapter also includes an accessible Mermaid alternative and a sequential text alternative. The raster is explanatory, not a benchmark plot; it contains no fabricated scores or rankings.

## 9. Integration checklist

- add the dedicated task chapter;
- split taxonomy rows for degradation restoration and masked completion;
- correct V2V routing and decision tree;
- add an explicit adjacent-task note to inpainting;
- link restoration metrics in evaluation;
- update README task map/tree and learner path;
- add the chapter to the coverage audit and visual queue;
- update timeline/reading route with only the milestones needed to expose the technical transition;
- run citation-anchor, relative-link, markdownlint, Mermaid and image checks.

### 9.1 Verification result

Checks were run after the chapter, task-boundary corrections, navigation, timeline and course-route integration:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 15 changed/new docs and logs, 0 issues; README retains two pre-existing warnings outside the changed lines |
| Reference closure | Restoration chapter: 30 reference anchors, 56 citation occurrences, 30 unique cited references; no missing, orphan, duplicate or numbering gap |
| Primary-source URLs | All 30 chapter reference URLs returned HTTP 200 during the final link audit |
| Local links and images | 306 relative targets checked across 16 changed/new Markdown files; 0 missing |
| Mermaid | 23 blocks across 13 changed documentation pages; all contain `accTitle` and `accDescr`, and all rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Mermaid visual check | The six-branch reading route, shared falsification loop, R1–R4 reasoning scale, two WM diagrams and both restoration diagrams were inspected individually or in a contact sheet; no clipped terminal node or broken branch was found |
| Generated PNG | 1672×941 opaque sRGB PNG; original and grayscale views inspected; all required labels are correct and legible; SHA-256 matches the value above |
| Timeline media preservation | All 75 pre-existing timeline images still have non-empty alt text; the current count matches `HEAD` |
| Patch hygiene | `git diff --check` returned no error; changed-file credential-pattern scan found no candidate secret |

Temporary Mermaid and grayscale audit artifacts remained outside the repository. These checks validate documentation structure, not the cited model checkpoints.

## 10. Evidence limit

This integration is a primary-source literature and documentation audit. It does not independently train or reproduce the cited large restoration models. Proceedings pages verify titles, venues and author-reported mechanisms; they do not convert speed, visual quality, robustness or hallucination claims into independent results. The chapter therefore labels author protocols, formal publication status and required falsifiers separately.
