# Text-to-video research and evidence log — 2026-08-30

This file records the evidence trail for the rewrite of `docs/tasks/text-to-video.md`. It is not a second tutorial. Its purpose is to make search scope, inclusion decisions, evidence grades, negative classifications, visual provenance and validation reproducible.

## 1. Rewrite trigger and frozen snapshot

Pre-rewrite snapshot:

- chapter length: 207 lines;
- numbered references: 12;
- Mermaid blocks: 1;
- embedded project PNGs: 0;
- audit status: depth 2.5, P0;
- explicit gaps: 2025–2026 claims without direct citations; no dedicated native audio-video, post-training/RL, long-video or release-surface analysis.

Rewrite questions:

1. What exactly makes a task pure T2V rather than I2V/TI2V, V2V, prediction or a closed-loop world model?
2. Which routes are genuinely alternative factorizations/mechanisms, and which labels mix representation, backbone and objective?
3. Which papers changed a testable capability rather than merely becoming popular?
4. How did post-training, prompt optimization and distillation alter the T2V stack after 2024?
5. What does “long video” prove, and what does it not prove?
6. Which systems jointly generate audio and video, which generate audio after video, and which only have product claims?
7. How should paper, demo, API/product, code and weights be separated?
8. What experiment would let another researcher falsify the chapter's claims?

Review date: **2026-08-30, Asia/Shanghai**.

## 2. Inclusion and exclusion protocol

Included sources had to satisfy at least one condition:

- direct T2V method paper that introduced a new condition interface, representation, generation factorization, training objective, length mechanism or audio-video mechanism;
- formal benchmark paper with a stated prompt set or diagnostic protocol;
- formal proceedings page used to verify title and venue;
- arXiv record used for frontier work that did not yet have a verified proceedings record;
- official model card, system card or product page used only for the producer's own product/availability claim;
- official repository used to determine code/weight/config release surface.

Excluded or demoted evidence:

- secondary “best model” lists, press summaries and SEO pages;
- montage quality without a fixed prompt, seed or evaluation protocol;
- cross-paper leaderboard numbers with different duration, resolution, prompt or evaluator versions;
- a demo as proof of public weights or a product page as proof of architecture;
- generic video generation ancestors presented as direct pure-T2V milestones;
- long output duration presented as proof of state or narrative consistency;
- author-reported performance presented as independent confirmation.

## 3. Search log

Searches were run on 2026-08-30. Counts are discovery diagnostics, not the population size of the field.

| Source | Exact query | Returned count / result | Use and caveat |
|---|---|---:|---|
| arXiv API | `ti:"text-to-video"` | 323 | High-precision title discovery; misses papers without the exact term |
| arXiv API | `all:"text-to-video generation"` | 348 | Phrase discovery across metadata/full indexed fields |
| arXiv API | `all:"text-to-video"` | 1,424 | Broad recall set; includes surveys, evaluation and adjacent tasks |
| OpenAlex API | `filter=title.search:"text-to-video"` | 1,075 | Broad tokenized title search; count is not directly comparable with arXiv |
| Semantic Scholar API | `text-to-video generation` | HTTP 429 | Failed/rate-limited; no metadata or count from this request was used as evidence |
| CVF Open Access | site search + exact-title lookup | n/a | Primary venue pages for CVPR/ICCV/WACV papers |
| NeurIPS Proceedings | exact-title lookup | n/a | Primary venue pages for Video Diffusion Models, T2V-Turbo and MLLM feedback |
| OpenReview | exact-title lookup | n/a | Verified Phenaki ICLR record `vOEXS39nOF` |
| Official model sites | exact model/card lookup | n/a | Product/model-card claims only; not independent validation |

Crossref was consulted only as a DOI/metadata spot-check path. A broad Crossref keyword count was rejected because it was dominated by tokenized, secondary and unrelated records.

### Targeted query families

- `text-to-video post-training preference DPO RLHF GRPO 2024 2025 2026`
- `text-to-video long video autoregressive chunk diffusion forcing`
- `joint audio video generation text native audio video`
- `text-to-video compositional benchmark prompt following temporal relation`
- exact titles and arXiv identifiers for all candidate milestones;
- official proceedings title lookups for papers claimed as CVPR, ICCV, WACV, NeurIPS or ICLR;
- official model-card/product pages for Sora 2 and Veo 3.1 Lite.

## 4. Evidence grades

| Grade | Evidence | Allowed wording |
|---|---|---|
| A | Formal proceedings or peer-reviewed venue page, plus paper | “The paper introduces/reports”; venue and year may be asserted |
| B | arXiv preprint or technical report, optionally with official artifact | “The authors report/propose”; no independent-SOTA wording |
| C | Official product page, system card or model card | “The provider states/records”; no public-checkpoint or architecture inference |
| S | Repository synthesis or experimental recommendation | Clearly labelled as this review's framework/protocol, not a paper result |

“Code” and “weights” are not grades. They are release-surface fields that must be checked for the exact version. A grade-B paper can have excellent artifacts; a grade-A paper can have none.

## 5. Candidate and milestone evidence matrix

| Work | Identifier / primary record | Grade | Direct pure T2V? | Included role | Evidence boundary |
|---|---|---:|---:|---|---|
| Video Generation From Text | AAAI article 12233 | A | yes | Early direct language-conditioned generation | Historical interface milestone, not modern quality evidence |
| To Create What You Tell | ACM MM 2017, DOI `10.1145/3123266.3127905`; arXiv:1804.08264 | A | yes | TGANs-C and frame/video/motion discriminator signals | Formal ACM Multimedia record establishes the 2017 venue; the 2018 arXiv version is linked as an accessible manuscript |
| CogVideo | arXiv:2205.15868; ICLR 2023 | A/B | yes | Discrete-token autoregressive T2V | Scaling/architecture result, not current product claim |
| Video Diffusion Models | NeurIPS 2022 proceedings | A | partly/general | Video diffusion and temporal super-resolution ancestor/direct experiments | Not used as proof of every modern T2V system |
| Make-A-Video | arXiv:2209.14792; ICLR 2023 | A/B | yes | Pixel/cascade diffusion using image-text semantics and unpaired video motion | Paired-data reduction, not complete caption alignment; it is not classified as latent video diffusion |
| Imagen Video | arXiv:2210.02303 | B | yes | Cascaded spatial/temporal diffusion | Closed reproduction surface; author evidence |
| Phenaki | OpenReview `vOEXS39nOF` | A | yes | Variable length, continuous prompts and MaskGIT-style masked video-token generation | “Variable length” not “no drift”; it is masked-token rather than autoregressive generation |
| MAGVIT | arXiv:2212.05199; CVPR 2023 | A/B | general/conditional, not direct text-conditioned generation | Masked video-token mechanism using known frames/partial video/class conditions | Technical ancestor only; it must not be presented as a pure-T2V or text-conditioned milestone |
| VideoPoet | PMLR 235 / arXiv:2312.14125 | A | multimodal | Unified autoregressive modality tokens | Multi-task model, not a T2V-only checkpoint |
| Align your Latents | CVPR 2023 proceedings | A | yes | Latent video diffusion practical template | Image prior does not prove motion/physics |
| Lumiere | arXiv:2401.12945 | B | yes | Full-duration space-time generation | No public checkpoint inference |
| CogVideoX | arXiv:2408.06072; ICLR 2025 | B/A | yes | Expert Transformer + 3D causal VAE with a v-prediction diffusion objective | Its objective must not be relabelled as flow matching; artifact state remains version-specific |
| Movie Gen | arXiv:2410.13720 | B | model family | Large media foundation system; staged video/audio distinction | Video and Audio are not one native joint AV checkpoint |
| HunyuanVideo | arXiv:2412.03603 | B | yes | Large DiT/flow system and open ecosystem | Self-reported benchmark not independent evidence |
| Wan | arXiv:2503.20314 | B | family | Open large-scale T2V/I2V family | Checkpoint sizes/tasks are not interchangeable |
| Step-Video-T2V | arXiv:2502.10248 | B | yes | 30B system, VAE, DiT, flow, Video-DPO | 204 frames is reported limit, not arbitrary horizon |
| Open-Sora 2.0 | arXiv:2503.09642 | B | system/family | Training-cost and open-system case study | “Commercial-level” and \$200k are author claims |
| SkyReels-V2 | arXiv:2504.13074 | B | T2V/I2V family | Diffusion Forcing, motion RL, continuation | “Infinite-length” is architectural/author wording |
| MAGI-1 | arXiv:2505.13211 | B | no, mainly TI2V/I2V | Chunk-autoregressive denoising adjacency | Explicitly excluded from pure-T2V milestones |
| InstructVideo | CVPR 2024 proceedings | A | yes | Human-feedback instruction tuning | Reward coverage is limited |
| MLLMs Feedback / VideoPrefer | NeurIPS 2024 proceedings | A | yes | VideoRM and large MLLM preference set | 135k and gains are author-reported |
| T2V-Turbo | NeurIPS 2024 proceedings | A | yes | Mixed-reward consistency distillation | Speed requires matched output/hardware protocol |
| VideoDPO | CVPR 2025 proceedings | A | yes | Omni-preference diffusion-DPO | Label/evaluator coverage remains a boundary |
| VPO | ICCV 2025 proceedings | A | yes/input-side | Prompt optimization | Benefit cannot all be assigned to backbone |
| Prompt-A-Video | ICCV 2025 proceedings | A | yes/input-side | Preference-aligned LLM prompt compiler | Must test semantic preservation |
| Systematic Post-Train Framework | arXiv:2604.25427 | B | yes | SFT→video RL→prompt→inference stack | Preprint/technical report, not consensus |
| DynamicsBoost | CVPR 2026 proceedings | A | yes | Continuation-derived dynamic preference pairs | Results remain author protocol results |
| Ovi | arXiv:2510.01284 | B | joint AV | Twin-backbone cross-modal generation | Preprint; exact artifact surface versioned separately |
| LTX-2 | arXiv:2601.03233 + official `Lightricks/LTX-2` repository | B + artifact surface | joint AV | Asymmetric dual-stream audio-video model | Freeze-day repository recommends LTX-2.5 and marks LTX-2.3 legacy; do not assume the current checkpoint is the paper checkpoint or treat author evaluation as independent certification |
| Sora 2 | official product page + system card | C | product | Synchronized dialogue/SFX and multi-shot product claim | No public checkpoint/complete training recipe; availability can change |
| Veo 3.1 / Lite | official page + model card | C | product | Native-audio T2V/I2V product claim | Internal/product evidence, not open reproduction |
| T2V-CompBench | CVPR 2025 proceedings | A | benchmark | 1,400 prompts, seven compositional categories | Benchmark score is protocol-specific |
| VBench | CVPR 2024 proceedings | A | benchmark | Multi-dimensional evaluation suite | Evaluator calibration and version still required |
| FETV | NeurIPS 2023 Datasets and Benchmarks | A | benchmark | Fine-grained open-domain evaluation | Not sufficient for physical/closed-loop claims |
| EvalCrafter | CVPR 2024 proceedings | A | benchmark | Comprehensive metric/human evaluation study | Total score cannot localize all failures |
| Synthesizing Compositional Videos | WACV 2026 proceedings | A | method | Recent compositional branch | Included as frontier reference, not a universal winner |

## 6. Negative classifications and corrected statements

These exclusions are part of the result, not omissions:

1. **MoCoGAN** is a foundational content-motion decomposition method, not direct open-language T2V evidence.
2. **VideoGPT** is a video-token autoregressive ancestor, not a direct T2V milestone.
3. **Stable Video Diffusion** is primarily an image-to-video model; it must not stand in for pure T2V.
4. **AnimateDiff** contributes a motion-module route on text-to-image backbones, but its task surface and personalization setup differ from a native large T2V system.
5. **MAGI-1** uses text-conditioned image/video continuation; it is cited for chunk autoregression, not promoted to pure T2V.
6. **Movie Gen Audio** conditions on generated/existing video; Movie Gen therefore does not prove one jointly trained native AV checkpoint.
7. **Sora 2/Veo 3.1** official claims prove what the provider documents about a product/system, not public weights, architecture completeness or independent benchmark superiority.
8. **Infinite-length** means a process can continue sampling under its window/state design. It does not mean identity, geometry, state or story has a zero-drift guarantee.
9. **Prompt optimization** changes the input distribution. Its gain must not be attributed solely to the video backbone.
10. **Language conditioning** is an interface, not a generation objective; it can coexist with GAN, AR, masked, diffusion or flow routes.
11. **Phenaki and MAGVIT are not interchangeable evidence.** Phenaki is a text-conditioned masked-token generator; MAGVIT is a masked video-token ancestor whose paper uses known-frame, partial-video and class conditions rather than open text.
12. **Make-A-Video and CogVideoX use different objectives.** The former belongs in the pixel/cascade diffusion route, while CogVideoX uses v-prediction rather than a flow-matching target.

## 7. Primary metadata checks

The following arXiv identifiers and titles were fetched as a batch and compared against the chapter before citation:

- `1710.00421` — Video Generation From Text
- `1804.08264` — To Create What You Tell: Generating Videos from Captions
- `2204.03458` — Video Diffusion Models
- `2205.15868` — CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers
- `2209.14792` — Make-A-Video: Text-to-Video Generation without Text-Video Data
- `2210.02303` — Imagen Video: High Definition Video Generation with Diffusion Models
- `2210.02399` — Phenaki: Variable Length Video Generation From Open Domain Textual Description
- `2304.08818` — Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models
- `2312.14125` — VideoPoet: A Large Language Model for Zero-Shot Video Generation
- `2401.12945` — Lumiere: A Space-Time Diffusion Model for Video Generation
- `2408.06072` — CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
- `2410.13720` — Movie Gen: A Cast of Media Foundation Models
- `2412.03603` — HunyuanVideo: A Systematic Framework For Large Video Generative Models
- `2412.14167` — VideoDPO: Omni-Preference Alignment for Video Diffusion Generation
- `2502.10248` — Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model
- `2503.09642` — Open-Sora 2.0: Training a Commercial-Level Video Generation Model in \$200k
- `2503.20314` — Wan: Open and Advanced Large-Scale Video Generative Models
- `2504.13074` — SkyReels-V2: Infinite-length Film Generative Model
- `2505.13211` — MAGI-1: Autoregressive Video Generation at Scale
- `2510.01284` — Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation
- `2601.03233` — LTX-2: Efficient Joint Audio-Visual Foundation Model
- `2604.25427` — A Systematic Post-Train Framework for Video Generation

Formal records additionally checked:

- Video Generation From Text — AAAI article `12233`;
- To Create What You Tell — ACM Multimedia 2017, DOI `10.1145/3123266.3127905` (the accessible arXiv manuscript appeared in 2018);
- Phenaki — OpenReview `vOEXS39nOF`;
- Video Diffusion Models, T2V-Turbo, MLLM Feedback and FETV — NeurIPS proceedings pages;
- VideoPoet — PMLR/ICML 2024 proceedings page;
- InstructVideo, VideoDPO, T2V-CompBench, VBench, EvalCrafter, DynamicsBoost — CVF proceedings;
- VPO and Prompt-A-Video — ICCV 2025 CVF proceedings;
- Synthesizing Compositional Videos from Text Description — WACV 2026 CVF proceedings.

## 8. Visual design and audit

Learning objective: teach a reader to trace T2V from intent to acceptance while keeping three task boundaries independent: pure text input, source-pixel hybrid generation and action/observation feedback.

### Generation prompt constraints

- clean 16:9 scientific infographic;
- five left-to-right stages: user intent, prompt contract, condition/generate, decode/rollout, acceptance gates;
- show diffusion/flow and video-token as alternative branches, not a chronology;
- show optional audio as a separate output;
- show three independent boundary cards: pure T2V, hybrid/source pixels, closed-loop/actions and observations;
- only the closed-loop card may point to world model;
- no model names, rankings, benchmark scores, logos, decorative screenshots or watermarks;
- short English labels to reduce typography risk; arrows must encode only scientifically valid relations.

### Iteration record

| Version | Generated artifact | Decision | Reason |
|---|---|---|---|
| v1 | superseded draft; artifact reference removed | rejected | An arrow from PURE T2V to HYBRID implied a progression; “2–5 s” also introduced an unnecessary duration assumption |
| v2 | superseded draft; artifact reference removed | rejected after evidence audit | Boundary cards were fixed, but `VIDEO LATENT` incorrectly implied that optional audio must be decoded from a video-only latent; the acceptance row also lacked explicit visual-continuity and safety/provenance gates |
| v3 | `generated_images/<thread>/exec-15b15c66-c025-4108-aecc-af137a57d4db.png` | accepted | The system abstraction is now `MEDIA LATENTS / TOKENS`; optional audio can represent joint AV or a separately labelled staged path, and the acceptance row explicitly includes visual continuity plus safety/provenance |

Project asset:

- path: `assets/diagrams/text-to-video-contract-evidence.png`;
- dimensions: `1663 × 946`;
- format: PNG, no alpha channel;
- size at audit: `1,317,131` bytes;
- SHA-256: `0afbcf9ad417d97a5294de8bb1dd1534a2e5144af1cfce452118c3fe25c311a0`.

Original-resolution audit:

- all five stage headings legible;
- PROMPT CONTRACT contains entity/relation/action/time/camera/sound;
- diffusion/flow and video-token branches are parallel;
- `MEDIA LATENTS / TOKENS` is explicitly a system-level abstraction, not a claim that every architecture uses one joint latent;
- OPTIONAL AUDIO is visibly separate and is not shown as necessarily decoded from a video-only latent;
- acceptance gates cover semantics, temporal, visual, physics, AV sync, safety/provenance;
- task cards are not connected as a false maturity ladder;
- no model name, fabricated number, logo, watermark or accidental text remains.

Grayscale audit:

- audit copy: `/tmp/text-to-video-contract-evidence-gray.png` (temporary, not committed);
- inspected at original resolution;
- labels, icons, borders and arrows remain distinguishable without color;
- chapter includes full alt text, caption and a sequential prose alternative.

The three Mermaid figures are editable/accessibility complements, not duplicates of the generated image: task boundary, data/training contract and post-training evidence risks. The boundary diagram asks about continuing action/observation feedback first, so a closed-loop system with pixel observations cannot be misrouted into I2V/V2V. The training diagram separates base diffusion/flow or AR/masked objectives from optional preference/reward post-training, includes a direct freeze path, and shows version-matched runtime prompt tokens entering inference. The post-training diagram further separates generator-updating SFT/DPO/RL/distillation paths from a prompt optimizer that produces an optimized prompt for a frozen generator, so the latter cannot be misread as changing model weights.

## 9. Claim-to-evidence rules applied in prose

- Numbers such as 30B, 204 frames, 135k annotations, 1,400 prompts or \$200k are explicitly attributed to the paper/author protocol.
- “Formal paper” means a verified proceedings page; otherwise the source is called a preprint/technical report.
- Product/model cards are labelled official provider evidence.
- Release surface is described with version-specific caution instead of a single `open` adjective.
- Recent frontier sources do not create a total ordering across incompatible protocols.
- Benchmark descriptions identify diagnostic scope and protocol dependence.
- Repository-authored evaluation and failure-localization advice is synthesis (grade S), not falsely cited as a paper result.

## 10. Machine validation record

Validation snapshot: **2026-08-30, Asia/Shanghai**. Commands and exact results are recorded rather than inferred from source syntax.

- [x] `markdownlint-cli2` checked the chapter and this log: 2 files, 0 issues.
- [x] Anchor audit found 36 numbered references, 36 unique anchors and no missing, duplicate or uncited anchors.
- [x] Every relative Markdown link and the project image path resolves from its containing file.
- [x] All 40 unique external reference URLs were checked: 36 original URLs returned HTTP 200; the ACM DOI and two official OpenAI pages returned HTTP 403 anti-bot responses rather than a missing-page status; the newly added official LTX-2 repository was confirmed through its GitHub page/API after a transient direct TLS failure; 0 URL was identified as missing.
- [x] All 3 Mermaid blocks were extracted and rendered with Mermaid CLI 11.16.0 using the system Google Chrome; every render exited 0. The final 1600-width audit outputs measured `1177 × 876`, `794 × 1748` and `1255 × 910`, were non-empty, and were inspected for legibility and correct branch semantics.
- [x] The project PNG was re-opened as RGB, measured at `1663 × 946`, checked at `1,317,131` bytes and hashed to `0afbcf9ad417d97a5294de8bb1dd1534a2e5144af1cfce452118c3fe25c311a0`.
- [x] An original-resolution grayscale audit copy was generated under `/tmp` and inspected; headings, short labels, arrows, boundary cards and all seven acceptance gates remain distinguishable without color.
- [x] No absolute personal path, secret or token appears in either file; generated-image provenance in this log uses the repository-safe `generated_images/<thread>/<artifact>.png` form.
- [x] `git diff --check` passes for the chapter and this log.
- [x] Independent evidence/readability audit findings were incorporated: masked-token classification, pixel/cascade placement, v-prediction naming, feedback-first task classification, optional base-vs-post-training flow, runtime prompt conditioning, prompt-optimizer attribution, formal TGANs-C venue and the final visual contract.

## 11. Residual uncertainty

- Product availability and API behavior can change after the snapshot date.
- Repository tags, licenses and model cards can change independently of the paper; the chapter therefore points detailed operational status to the resource/model registry.
- Some 2026 papers are recent even when a formal CVF page exists; their reported improvements have not thereby become independent reproductions.
- The review did not run all candidate checkpoints under a common compute/data protocol. It therefore extracts routes and evidence boundaries, not a universal performance ranking.
