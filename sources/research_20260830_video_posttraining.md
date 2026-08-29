# Research log: video-generation post-training, alignment, and few-step adaptation

> Search freeze: **2026-08-30 (Asia/Shanghai)**. This file records the evidence used by `docs/generative-models/video-post-training-alignment.md`; it is not an independent reproduction of any large-model result.

## 1. Review question and scope

The review asks: after a video generator has a usable pretrained checkpoint, which mechanisms change its capability or preferences, which reduce sampling cost, and which do both?

Included:

- continued pretraining and supervised fine-tuning when they are part of a post-training recipe;
- video reward data, reward models, and temporal/timestep/set-level reward shaping;
- offline and online direct preference optimization;
- reward-weighted regression and policy-gradient/GRPO-style optimization;
- verifier-guided or test-time search/adaptation;
- consistency, DMD, and other distillation routes when compared with preference alignment;
- formal 2023–2026 mechanism ancestors and the 2025–2026 video frontier.

Excluded as evidence of generator alignment:

- video-language reasoning RL that updates a VLM/reasoner rather than a generator;
- product pages without a disclosed training or inference mechanism;
- image-only mechanisms presented as if they were direct video results;
- a reward model score increase without a generator update or inference intervention;
- “online” used only as a label, without checking whether it means online pair collection, policy-gradient rollout, or test-time optimization.

## 2. Search protocol

### 2.1 Complementary primary-source families

1. **Formal proceedings:** CVF Open Access, NeurIPS, ICLR, PMLR, and ACL Anthology pages plus their linked papers/supplements.
2. **Preprint registry:** arXiv abstract/version pages for work not formally published by the freeze date.
3. **Official research artifacts:** author project pages and official code repositories, used for implementation/release-surface checks rather than upgrading a claim's publication status.

Secondary surveys, news articles, and search-result snippets were not used to support technical claims.

### 2.2 Exact query strings

Queries were issued in combinations of the following strings, then resolved to the primary paper page:

```text
site:openaccess.thecvf.com video preference optimization reward video generation
site:proceedings.neurips.cc video generation human feedback reward Flow-DPO
site:proceedings.iclr.cc video generation GRPO reward latent 2026
site:proceedings.mlr.press direct preference optimization IPO
site:aclanthology.org ORPO preference optimization VideoScore
site:arxiv.org video generation post-training GRPO reward 2026
site:arxiv.org video generation verifiable reward camera trajectory
site:github.com T2V-Turbo official
"DynamicsBoost" continuation preference optimization
"DPP-GRPO" diverse video generation
"OnlineVPO" video-centric preference
"Dual-IPO" video generation
"Consistent Noisy Latent Rewards" video
"BranchGRPO" video diffusion
```

Search terms were expanded by citation chaining from VideoAlign, VideoDPO, T2V-Turbo-v2, DynamicsBoost, BranchGRPO, and Dual-IPO. Venue metadata was checked on proceedings pages rather than inferred from an arXiv note.

### 2.3 Evidence levels

| Level | Meaning | Use in chapter |
|---|---|---|
| A | Formal peer-reviewed proceedings paper | Main mechanism and milestone evidence; quantitative outcomes remain author-reported |
| B | arXiv preprint as of 2026-08-30 | Frontier signal only; explicitly labeled preprint |
| C | Official code, weights, or project page | Artifact/release check; does not upgrade a paper's evidence level |
| D | Indirect or out-of-task evidence | Mechanism ancestor or exclusion boundary, never direct video-alignment proof |

## 3. Evidence matrix

### 3.1 Foundations and formal video evidence

| Source | Level | Primary contribution used | Boundary recorded |
|---|---:|---|---|
| [DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) | A/D | Offline direct preference objective without explicit RM/on-policy sampling | Language-model evidence, not video evidence |
| [IPO / general $\Psi$PO](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html) | A/D | Identity mapping and preference-learning theory | Language-only foundation; not Dual-IPO |
| [ORPO](https://aclanthology.org/2024.emnlp-main.626/) | A/D | Reference-free odds-ratio term within SFT | Language-only foundation |
| [DDPO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html) | A/D | Treats denoising as a multistep decision process for policy gradient | Image evidence; mechanism ancestor |
| [Diffusion-DPO](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html) | A/D | Adapts DPO-style preference training to diffusion | Image evidence; mechanism ancestor |
| [InstructVideo](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_InstructVideo_Instructing_Video_Diffusion_Models_with_Human_Feedback_CVPR_2024_paper.html) | A | Direct video reward fine-tuning, partial DDIM chain, temporally attenuated reward | Author-reported quality; reward is not a human oracle |
| [VideoScore](https://aclanthology.org/2024.emnlp-main.127/) | A | 37.6K videos, 11 models, fine-grained human scores and automatic evaluator | Reported correlations are dataset/protocol-specific |
| [VideoRM / VideoPrefer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html) | A | 135K MLLM preference annotations and direct-video RM | MLLM preference is not automatically human ground truth |
| [VideoDPO](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html) | A | Seven-dimensional OmniScore, multiple candidates, extreme pair and reweighting | Multiple judges add compute; metric weights encode values |
| [VideoAlign / VideoReward](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76227feb18ea0ee40bd15cf02c33e18e-Abstract-Conference.html) | A | 16K prompts, 108K videos, 182K triplets; Flow-DPO, Flow-RWR, Flow-NRG | Same-RM relabel/evaluation can be circular; human study remains author protocol |
| [DenseDPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html) | A | Same-source corrupted pairs and segment-level preference | Synthetic corruption need not match all generator failures |
| [OnlineVPO](https://openaccess.thecvf.com/content/WACV2026/html/Zhang_Align_Video_Diffusion_Model_with_Online_Video-Centric_Preference_Optimization_WACV_2026_paper.html) | A | Current-policy video sampling, VQA reward, curriculum reference update | Online pair collection plus DPO-style update, not policy gradient |
| [DynamicsBoost](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.html) | A | Continuation pairs; excludes shared prefix and normalizes generated length | Annotation-free ordering rests on a continuation-quality assumption |
| [Dual-IPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a0d3f77bb435817807d463c5dcef1ab-Abstract-Conference.html) | A | Iteratively updates both RM and video generator | RM-generator co-drift needs frozen human gold auditing |
| [Consistent Noisy Latent Rewards](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html) | A | Noise-compatible latent RM and preference consistency across timesteps | Multi-timestep feedback is approximate credit, not causal proof |
| [BranchGRPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/233d16f17f809981763db2f01b7f9603-Abstract-Conference.html) | A | Shared prefixes, depth-wise advantage/reward fusion, pruning; includes a WanX setting | Speedups are author-reported in a specific implementation |
| [TempFlow-GRPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d75f561eaaf2cb754bc8d7e36d8af362-Abstract-Conference.html) | A/D | Noise-aware temporal weights and seed grouping | Formal experiments are mainly image generation |
| [DPP-GRPO](https://openaccess.thecvf.com/content/CVPR2026/html/Kazimi_Diverse_Video_Generation_with_Determinantal_Point_Process-Guided_Policy_Optimization_CVPR_2026_paper.html) | A | DPP set reward for relevance and diversity | Optimizes an LLM prompt policy, not video backbone weights |

### 3.2 Few-step, test-time, and data routes

| Source | Level | Primary contribution used | Boundary recorded |
|---|---:|---|---|
| [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) | A/D | Learns direct flow-map-style consistency; standalone or distilled | Sampling-cost mechanism; not preference alignment by default |
| [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html) | A/D | One-step distribution matching student | Image evidence; teacher/distribution objective, not preference |
| [DMD2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | A/D | Stabilizes/improves distribution matching | Image evidence; not a video preference result |
| [T2V-Turbo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a57aa8e8b57e64a42e95f7dceb0adb9-Abstract-Conference.html) | A | Reward-guided consistency distillation and reported four-step video | Combines two contracts; fixed reward is not policy gradient |
| [T2V-Turbo-v2](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e68af7d8a44bc1964f6be4de464e38f9-Abstract-Conference.html) | A | Curated data, multiple rewards, conditional/motion guidance in distillation | Teacher, encoder, and RM context constraints remain |
| [DOLLAR](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.html) | A | Variational/consistency distillation plus latent reward | 1/4-step results are author-reported |
| [Free2Guide](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html) | A | Gradient-free, training-free black-box LVLM guidance | Adds inference-time reward calls; no persistent post-training |
| [TTOM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/727855c31df8821fd18d41c23daebf10-Abstract-Conference.html) | A | Test-time new parameters/layout attention and parametric memory | Base model may be frozen, but per-request optimization costs time |
| [VideoUFO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1e6057620ed314b0020b3a30284b0f83-Abstract-Datasets_and_Benchmarks_Track.html) | A | 1.09M clips over 1291 user-focused topics | CPT/SFT data, not preference-pair data |
| [SkyReels-V2](https://arxiv.org/abs/2504.13074) | B | Separates multistage pretraining, concept-balanced SFT, motion RL, final SFT | Technical report/preprint; reported metrics are not independently reproduced |

### 3.3 Preprint frontier and explicit exclusions

| Source | Level | Why retained | Boundary recorded |
|---|---:|---|---|
| [A Systematic Post-Train Framework](https://arxiv.org/abs/2604.25427) | B | Combines SFT, GRPO-RLHF, prompt enhancement, inference optimization | Still a preprint at freeze date |
| [RewardDance](https://arxiv.org/abs/2509.08826) | B | Scaling/context/variance claims for generative RM | Hacking/collapse mitigation is author-reported |
| [Video Generation Models Are Good Latent Reward Models](https://arxiv.org/abs/2511.21541) | B | VGM-derived noisy latent reward and decode-saving claim | Preprint; efficiency is author-reported |
| [Verifiable Geometry Reward](https://arxiv.org/abs/2512.02870) | B | Segment relative-pose reward for camera control | Pose is estimated, not ground-truth verification |
| [World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](https://arxiv.org/abs/2604.24764) | B | Flow-GRPO with 3D/VLM rewards and periodic updates | Pretrained estimator/judge remains fallible; arXiv lists ICML 2026, but this review conservatively keeps preprint status |
| [Reward-Forcing: Autoregressive Video Generation with Reward Feedback](https://arxiv.org/abs/2601.16933) | B | Direct reward route for autoregressive few-step video | Reduced teacher-dependence claim is author-reported |
| [Wan-R1: Verifiable-Reinforcement Learning for Video Reasoning](https://arxiv.org/abs/2603.27866) | B/D | Boundary example for video reasoning RL | Reasoning-task evidence is not general preference-alignment evidence |
| [VLMs are Good Teachers for Video Reasoning via Adaptive Test-Time Optimization](https://arxiv.org/abs/2606.02564) | B/D | Boundary example for VLM reasoning teachers | Does not establish general generated-video preference improvement |

### 3.4 Official artifact checks

| Artifact | Level | Check performed | Claim boundary |
|---|---:|---|---|
| [T2V-Turbo code](https://github.com/Ji4chenLi/t2v-turbo) | C | Official implementation/release surface identified | Repository existence does not reproduce quality/speed |
| [VideoDPO project](https://videodpo.github.io/) | C | Project examples and paper linkage identified | Curated examples are not a blind evaluation |
| [TTOM code](https://github.com/LgQu/TTOM) | C | Official code/release surface identified | Availability does not validate every benchmark claim |

## 4. Synthesis decisions

### 4.1 Route classification

- CPT/SFT changes data coverage or interface and does not reduce base NFE by itself.
- RM creates a measurement model; reward shaping determines how its score is assigned.
- DPO/IPO/ORPO are direct preference families; direct video evidence here is mainly DPO-derived, while foundational IPO/ORPO evidence is language-only.
- OnlineVPO is online sampling with a DPO-style update, not policy gradient.
- DDPO/GRPO-style methods use rollout rewards and have explicit trajectory credit/cost.
- Test-time guidance/search may change only the current output and usually adds inference cost.
- Consistency/DMD/distillation reduces student sampling cost; preference changes only when preference/reward supervision is explicitly added.

### 4.2 Pair-construction taxonomy

The chapter distinguishes human A/B/tie, MLLM labels, composite-metric extremes, same-source corruption, current-policy online pairs, continuation-derived order, and alternating RM-generator feedback. These were kept separate because they encode different counterfactuals and biases.

### 4.3 Credit-assignment taxonomy

Two temporal axes were retained:

1. video time: frames and semantic/motion segments;
2. diffusion/flow time: noisy latent or denoising/transport steps.

A terminal clip score broadcast to all steps is a high-variance surrogate. DenseDPO supplies segment pairs; Consistent Noisy Latent Rewards evaluates noisy timesteps; BranchGRPO assigns depth-wise branch advantages. None was described as recovering ground-truth causal credit.

### 4.4 Frontier summary

The 2025–2026 frontier was summarized as:

- offline fixed pairs to online current-policy feedback;
- scalar terminal reward to multidimensional, segment, noisy-latent, branch-depth, and set-level signals;
- persistent generator-weight updates to test-time guidance/adaptation;
- preference-only training and few-step compression increasingly combined but still separately evaluated.

## 5. Educational figure audit

### 5.1 Accepted generation prompt

```text
Use case: scientific-educational
Asset type: 16:9 landscape teaching figure for an advanced Chinese-language textbook chapter on video-generation post-training
Primary request: Create a clean, scientifically accurate parallel-route decision map, not a sequential pipeline. On the left, one compact icon labeled exactly "PRETRAINED VIDEO GENERATOR". From it, five separate horizontal lanes fan out to the right. Lane labels must be exactly: "DATA / SFT", "PAIRWISE", "REWARD / RL", "TEST-TIME", "DISTILL". Use simple visual metaphors: curated film-strip data for DATA / SFT; two compared clips with a preference checkmark for PAIRWISE; grouped rollouts feeding a reward gauge and feedback loop for REWARD / RL; a frozen generator plus search/guidance branches for TEST-TIME; a large teacher network compressing into a small few-step student for DISTILL. At the far right of each lane, use only tiny effect chips with these exact texts: DATA / SFT: "CAPABILITY UP  STEPS SAME"; PAIRWISE: "PREFERENCE UP  STEPS SAME"; REWARD / RL: "PREFERENCE UP  TRAIN COST UP"; TEST-TIME: "PREFERENCE UP  INFER COST UP"; DISTILL: "STEPS DOWN  TEACHER DEP.". Add one small warning triangle beside DISTILL with the exact caption "PREFERENCE ONLY IF REWARD IS ADDED". Keep all five lanes visually independent and aligned; do not connect one lane into another.
Style/medium: flat vector-like scientific infographic, textbook-quality, white background, crisp dark sans-serif typography, minimal icons, restrained Okabe-Ito colorblind-safe palette, redundant lane numbering 1-5 and distinct icons so the diagram works in grayscale
Composition/framing: exact 16:9 landscape; generous margins; strong left-to-right hierarchy; equal lane spacing; no title banner; no footer
Constraints: render only the specified text, verbatim; no abbreviations beyond the specified labels; no decorative gradients; no human faces; no logos; no trademarks; no watermark; no extra arrows between lanes; no claim that all methods are sequential; high contrast; all text large and readable at half-page width
Avoid: dense paragraphs, tiny labels, crossed arrows, 3D rendering, photorealism, neon colors, red-green-only encoding, clutter
```

### 5.2 Accepted asset

- Repository path: `assets/diagrams/video-posttraining-evidence-map.png`
- Format: PNG, RGB, non-interlaced
- Dimensions: **1672 × 941 px**; aspect ratio 1.7768, within rounding of 16:9
- SHA-256: `b774db2ab80b9963934abedd31ee1f2cfb785665ab9d0b9ddcf52efccf7d0c50`
- Original visual inspection: all requested labels are legible and spelled as specified; five numbered routes remain independent; no clipped text, overlap, watermark, logo, or unintended inter-route arrows was observed.
- Grayscale inspection: a temporary grayscale rendering remained distinguishable through redundant numbering, icons, shape, and text. Image statistics were `min=0`, `max=65535`, `mean=58463.3`, `stddev=16370.9`; no workspace grayscale derivative was retained.

### 5.3 Rejected attempt

The first image-tool invocation was rejected by the JavaScript wrapper parser before generation. It produced no image artifact and was not copied into the repository. The second invocation used the unchanged prompt above and produced the accepted asset.

### 5.4 Scientific boundary

The figure is a parallel decision map, not a temporal pipeline. `STEPS SAME` means that the route alone does not reduce the base sampler's neural function evaluations. `PREFERENCE UP` names the intended objective rather than a universal outcome. `INFER COST UP` accounts for search, guidance, or reward calls. Teacher dependence is common in distillation rather than logically mandatory, and distillation changes preferences only when reward/preference supervision is included.

### 5.5 Text-equivalence contract

The chapter contains an editable Mermaid diagram with `accTitle` and `accDescr`, followed by a six-step sequential text alternative. Both encode the same five independent routes and effect boundaries as the PNG; visual styling is not treated as evidence.

## 6. Verification record

The following checks are required after the chapter and this log are finalized:

```text
markdownlint: passed with markdownlint-cli2 0.18.1 / markdownlint 0.38.0, 0 errors
local Markdown links: passed, 7/7 targets exist
external HTTP links: passed, 40/40 unique URLs returned status below 400 after redirects
reference-anchor closure: passed, 38/38 numbered references cited and defined; 0 dangling anchors
Mermaid real render: passed with mermaid-cli 11.12.0 and local Google Chrome; SVG output 31,011 bytes
PNG dimensions/SHA-256: passed (1672x941; b774db2...d0c50)
original and grayscale visual inspection: passed
git diff --check on owned files: passed
```

These checks validate document integrity, not the papers' performance claims.
