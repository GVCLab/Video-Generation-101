# Open-set personalized video generation research record

> Freeze date: **2026-08-30 (Asia/Shanghai)**. This record supports
> `docs/tasks/personalized-video-generation.md`. It distinguishes primary-source
> review, artifact inspection, proposed tests and actual reproduction. No model
> checkpoint was trained or run in this batch.

## 1. Review questions

1. When is a reference image a subject definition rather than an I2V time anchor?
2. What does `open-set` mean when the foundation model's pretraining set is not
   fully auditable?
3. Which routes adapt one subject at test time, which amortize personalization
   into an encoder or adapter, and which inject reference tokens without tuning?
4. How do methods bind several references to several entities without identity
   blending, subject drop or role/action exchange?
5. Which measurements separate identity fidelity, prompt/action following,
   motion, temporal drift, reference leakage and adaptation cost?
6. What changed from the image-personalization ancestors of 2022–2023 through
   the formal 2024–2026 video milestones?
7. Which code, weights, data and evaluation artifacts were actually public at
   the freeze date?

## 2. Search and source protocol

### 2.1 Source order

1. CVF, NeurIPS, AAAI, ACM, Springer and OpenReview proceedings;
2. arXiv version pages and paper HTML/PDF for first-public dates and preprints;
3. author project pages, GitHub and Hugging Face for artifact status;
4. no survey, leaderboard screenshot, vendor blog or secondary article was used
   as evidence for a mechanism or publication status.

### 2.2 Query families

- `subject-to-video open-set personalization multi subject`
- `video identity customization motion preservation`
- `tuning-free multi-concept video personalization anchored prompt`
- `identity preserving video generation DiT frequency decomposition`
- `video personalization preference optimization reward`
- `multi-reference video generation temporal interval identity`
- `subject video dataset benchmark identity leakage binding`
- `CVPR ICCV NeurIPS 2025 2026 personalized video`

### 2.3 Inclusion rules

A work enters the main lineage when it changes at least one of the following:

- subject acquisition or adaptation surface;
- text–reference or reference–slot binding;
- identity–motion conflict handling;
- multi-subject composition or role/relation control;
- subject-to-video data, evaluation or temporal-presence protocol;
- post-training objective that directly targets personalization failures.

Image-only personalization is retained only as an ancestor. Strict I2V, V2V
editing, pose/audio animation, style-only LoRA and motion-only customization are
kept as neighboring tasks unless a paper explicitly evaluates subject identity
under novel video content.

## 3. Frozen task boundary

The chapter uses the following operational definition:

> Given one or more subject reference sets and a text prompt, optionally with
> motion, camera, relation or temporal-presence controls, generate a new video in
> which each intended subject retains its identity and key attributes. The
> references do not have to occupy the output timeline, and there is no complete
> source video whose timeline must be preserved.

Consequences:

- a first frame that must be reproduced is strict I2V;
- a full source video that must be edited is V2V;
- a portrait plus pose/audio/driving signal is character or digital-human
  animation unless the work also studies open-context subject generation;
- per-subject tuning remains personalization, but its data, trainable parameters,
  steps, time, memory and stored state must be reported;
- `open-set` means generalization relative to a declared identity- or
  subject-disjoint evaluation split. It does **not** establish that an
  internet-pretrained foundation model has never seen a celebrity, product or
  object, and it does not mean open weights, open vocabulary or zero-shot.

## 4. Route taxonomy used in the chapter

| Route | Subject-specific work at deployment | Typical mechanism | Principal risk |
|---|---:|---|---|
| A. Per-subject tuning | token, LoRA, adapter or full tuning | bind an identifier to a few images | reconstructive overfit, static/copy shortcut, storage |
| B. Amortized encoder/adapter | none for a new subject | shared vision/face encoder and learned injection | domain bias, insufficient fine detail, hidden training scale |
| C. Frozen or in-context reference tokens | none or lightweight inference logic | concatenate/reference cross-attend to image tokens | prompt domination, order ambiguity, reference leakage |
| D. Image personalization plus motion module | separate reusable modules | combine spatial identity and temporal motion modules | module interference and motion forgetting |
| E. Multi-subject/relation composition | none or several subject modules | masks, anchored prompts, subject slots, relation/motion adapters | blend, drop, attribute/action swap |
| F. Reward/preference post-training | per-identity or global | identity, semantic, motion or VLM rewards | evaluator gaming, reward overfit, proxy blindness |
| G. Temporal/multi-shot memory | usually amortized | reference time windows, content anchors, cross-shot context | boundary softness, long-horizon drift, context conflicts |

These are alternatives and combinations, not a single upgrade chain.

## 5. Primary evidence ledger

### 5.1 Early route-forming work

| Work | Safe mechanism claim | Evidence boundary |
|---|---|---|
| Textual Inversion / DreamBooth | established learned tokens and per-subject image-model tuning | image generation ancestors, not video evidence |
| VideoDreamer | early multi-subject language-video customization preprint | no formal proceedings or released method code at freeze date |
| DreamVideo | 3–5 subject images plus motion videos; separate textual identity, identity adapter and motion adapter | authors' 20-subject/30-motion test set; first-attempt language is an author claim |
| VideoBooth | feed-forward image-prompt video: coarse CLIP embedding plus multi-scale image-latent attention injection | image-level CLIP/DINO; not a strict multi-subject binding protocol |
| CustomVideo | co-occurrence images, subject masks and attention control for two/three subjects | first public in 2024; formal TMM publication is 2026 |
| DisenStudio | spatial-disentangled cross-attention and masked/motion-preserved tuning | ACM MM 2024; custom training rather than open-set amortized inference |
| Magic-Me | identity tokens, 3D Gaussian noise prior, face/tiled refinement | ECCV 2024 workshop proceedings, not main ECCV; human identity focus |

### 5.2 2025 formal system milestones

| Work | Safe mechanism claim | Critical evidence boundary |
|---|---|---|
| ConsisID | CogVideoX DiT with global keypoint/face latent and local recognition+CLIP features; no per-person tuning | single human identity; 30-person benchmark; FID is worse than ID-Animator in its main table |
| Video Alchemist | entity word + reference image binding, separate personalization cross-attention, no test-time optimization | architecture supports multi-subject, but main quantitative subsets are 1,736 single-subject and 1,285 single-face videos |
| Movie Weaver | `[R1]/[R2]` anchored prompts plus concept embeddings encode text–image association and reference order | only five trained configurations; automatic comparison is 97 single-face cases; multi-concept evidence is mainly human ablation and qualitative comparison |
| VideoMage | spatial subject LoRAs, temporal motion LoRA, auxiliary-video regularization, negative guidance and collaborative sampling | six motions × three subject pairs × four backgrounds; not open-set amortized personalization |
| PersonalVideo | per-identity video reward tuning, semantic consistency to the base model and late-step identity injection | explicitly single identity and per-identity optimization; no multi-identity support |
| MagicMirror | face identity/structure dual branch plus cross-modal adapter and two-stage training | human face specialization; formal ICCV 2025 |
| MagicID | identity/dynamics preference pairs and hybrid sampling | identity-focused human customization; reward quality is a dependency |
| Phantom | joint text-image-video triplets and cross-modal alignment for single/multi-subject S2V | foundation-model scale and data make route-level comparison difficult |
| DreamRelation | learns a relation from exemplar videos with relational decoupling and dynamics enhancement | relation customization is not identical to open-set subject acquisition |
| DualReal | alternates identity/motion adaptation and controls both by denoising stage and DiT depth | joint tuning route; its reported relative gains are protocol-specific |
| OpenS2V-Nexus | OpenS2V-Eval plus OpenS2V-5M subject-text-video triples | automatic metrics remain proxies; data provenance and foundation-model contamination need separate audits |

### 5.3 2026 formal and frontier evidence

| Work | Safe mechanism claim | Critical evidence boundary |
|---|---|---|
| AlcheMinT | VAE reference tokens, entity text binding and interval-conditioned positional bias for subject appearance/disappearance | tables test one/two references; benchmark/training sample counts are not disclosed; timing and CLIP identity trade off |
| ID-Crafter | hierarchical intra/inter-subject and cross-modal attention, VLM semantic guidance and online RL | formal CVPR 2026; VLM/reward correctness must be audited separately |
| Gloria | compact character content anchors, extra/intra-clip cues against copying and weak positional offsets between anchors | authors report videos beyond 10 minutes; this is not an independent reproduction or arbitrary-character guarantee |
| SMRABooth | aligns self-supervised subject representation and optical-flow motion representation with sparsely injected LoRAs | subject+motion customization; requires per-concept tuning |
| ConsID-Gen | strict I2V neighbor: auxiliary unposed views and visual-geometric encoder for view-consistent identity | first image is a timeline anchor, so it does not redefine open-set S2V |
| IPRO | strict human I2V neighbor: differentiates a face reward through late sampling steps with KL regularization | reward improvement does not validate generic objects or non-face identity |
| ID-Sim | identity-focused feed-forward similarity metric trained for human-selective sensitivity | image/crop metric; no temporal or role-binding measurement by itself |
| VGBE 2026 challenge | consistent I2V evaluation with identity, geometry and perceptual quality | workshop challenge and I2V contract, not an S2V benchmark |
| Vera | million-pair cross-clip identity-aligned human data, masked identity supervision and layer-wise reference attention | 2026-07 preprint; human-specific and not formally published at freeze date |
| Keyframe-Anchored | training-free keyframe sequence plus multi-reference interpolation for timestamped actions | challenge solution ranked third; sequential I2V/keyframe route rather than general S2V evidence |

## 6. Red-team checks against overclaiming

### 6.1 Video Alchemist

- The paper's training reference and target are sampled from the same video. It
  explicitly documents pose, lighting, occlusion, crop and camera-view leakage,
  and uses blur, scaling, color, brightness, flip, shear and rotation to reduce,
  not eliminate, the shortcut.
- MSRVTT-Personalization has 2,130 clips and subject-level metrics, but the main
  comparisons use only the single-subject and single-face subsets. Multi-subject
  support is therefore an architectural and qualitative claim, not a complete
  role-binding benchmark result.
- The paper reports that more references can reduce text alignment. Its own
  augmentation ablation also shows a Pareto movement rather than improvement on
  every metric.
- The authors list residual reference-pose/expression copying, segmentation
  burden, unrealistic multi-subject scale/composition and absence of an automatic
  visual-quality metric as limitations.

### 6.2 Movie Weaver

- Training covers face, face-body, face-body-animal, two-face and
  two-face-body; a three-person prompt can collapse to two because that
  configuration is absent from training.
- The 228K pretraining videos and 651 curated fine-tuning videos are proprietary
  Shutterstock-derived data; the final model is 30B and the reported pretraining
  used 256 H100 GPUs for about five days.
- Its only automatic matched comparison is on 97 single-face cases. The
  multi-concept claim relies on qualitative comparison to Vidu 1.5 and a
  300-pair two-face human ablation.
- Anchored prompts and concept embeddings strongly improve two-face separation,
  but the protocol does not measure which person performs which action over
  time. The paper also documents big-face, reduced-motion and unseen-template
  failures.

### 6.3 PersonalVideo

- This is per-identity reward tuning: 800 AnimateDiff iterations or 4,000
  HunyuanVideo steps in the reported settings, not tuning-free open-set inference.
- The evaluation has 20 identities × 50 prompts = 1,000 videos. The method
  reports a joint increase in face similarity, dynamic degree and text score, but
  Dynamic Degree is motion magnitude, not action correctness.
- The paper explicitly states that it cannot generate multiple identities.
- The prose says it minimizes cosine similarity while the displayed identity
  objective is a positive cosine term in the total loss. Without code, the sign
  convention is ambiguous; the chapter therefore explains the intended reward
  without reproducing the questionable equation.

### 6.4 ConsisID

- The internal pipeline filters to 130K clips; the benchmark uses 30 people,
  five images each and 90 prompts. “Not in our training set” does not establish
  absence from CogVideoX or face-recognition pretraining.
- Its main table reports higher face/text scores than ID-Animator, but worse
  face-region FID (151.82 versus 117.46, where lower is better). The paper's
  statement that it wins across all metrics is inconsistent with the table.
- The supplement itself notes weak alignment between FID and human perception.
  The chapter reports metric directions rather than repeating the global claim.
- Loss symbols/names are reversed once in the setup section. The chapter uses
  full loss names and does not reuse those symbols.

### 6.5 AlcheMinT

- The main table evaluates one and two references. “Up to 15 tracks per training
  video” describes the annotation pipeline, not verified 15-reference generation.
- WeRoPE improves the reported temporal interval metrics but lowers both CLIP
  text and CLIP reference scores in the central ablation; it is a Pareto trade,
  not a free identity improvement.
- Standard RoPE provides relative phase structure, not a theorem that attention
  monotonically decays with distance. The paper's rotation-mixture equality and
  one right-interval timestamp also appear algebraically ambiguous. The chapter
  therefore describes WeRoPE as an empirical interval-dependent phase bias and
  does not present the disputed equality as fact.
- S2VTime uses synthetic T2I references and GroundingDINO+SAM2 for interval
  extraction; it does not test same-class identity swaps or full role/action
  binding.

## 7. Evaluation contract derived from the review

The chapter separates these measurements:

1. **Subject presence**: the intended subject is detected/tracked in expected
   frames; a missing subject gets zero rather than being omitted.
2. **Identity/attribute fidelity**: face or generic-subject similarity is
   computed on tracked subject crops, with per-frame curves and real-video
   calibration.
3. **Temporal drift and re-entry**: identity before/after occlusion or leaving the
   frame is compared directly.
4. **Text/action/camera following**: global semantics and subject-specific roles
   are scored separately.
5. **Motion**: optical-flow magnitude, trajectory diversity and action success
   are separate; temporal consistency cannot substitute for motion.
6. **Binding**: the full reference-to-generated-subject similarity matrix is
   measured; diagonal margin and swap tests reveal blending and role exchange.
7. **Leakage**: reference background, pose, crop and pixel-copy similarity are
   explicitly probed under counterfactual references.
8. **Base-capability retention**: the same prompts without personalization are
   compared to the frozen base model.
9. **Adaptation/system cost**: data, steps, trainable/stored parameters, wall
   time, peak VRAM, generation latency and failure rate are reported.
10. **Rights and contamination**: consent, license, biometric processing,
    near-duplicate search, revocation and model-state deletion are audited.

## 8. Artifact status at freeze date

| Work | Code / weights / data status checked |
|---|---|
| Magic-Me | official training/inference code and some identity embeddings; external base weights required; no training dataset |
| PersonalVideo | author repository remains a `Code Coming Soon` placeholder; no weights/data |
| Video Alchemist | official MSRVTT-Personalization benchmark/evaluation repository; no model code or weights |
| Movie Weaver | project page; no linked official code, weights or data |
| MAGREF | inference code and checkpoint; 480P/14B Pro and training code remain TODO; no training data |
| AlcheMinT | README/project placeholder; no method code, weights or downloadable benchmark |
| PoCo | project/demo repository; core model code, weights and data not released |
| OpenS2V-Nexus | evaluation code, generated comparison videos and released dataset subsets under the repository's stated terms |
| ConsisID | official code/model ecosystem and OpenS2V links; the full internal 130K training corpus is not equivalent to an open dataset |

Artifact availability is not model-quality evidence, and a project page is not
counted as code release.

## 9. Teaching visual record

### 9.1 Final asset

- File: `assets/diagrams/personalized-video-binding-contract.png`
- Built-in image generation tool; two text/semantics corrections followed by a
  single-panel failure correction.
- Pixel dimensions: **1672 × 941** (approximately 16:9)
- Color space: RGB
- File size: **1,991,841 bytes**
- SHA-256:
  `856761f1d0fc89c76b90aef4407879d59ab54a28a8d8c1ebcc9161e2d294d5b9`

### 9.2 Final generation/edit intent

The final image depicts three reference sets (human, dog, toy), three independent
subject slots, four parallel route choices (`TUNE`, `ADAPTER`, `IN-CONTEXT`,
`BIND`), a generated video in a novel context, five independent gates and four
failure panels (`BLEND`, `DROP`, `COPY`, `FREEZE`). The reference images are
visually outside the output timeline.

The base prompt requested an original vector-like scientific infographic with
exact uppercase labels, no paper/model names, scores, logos, watermark or copied
academic layout. The first correction changed the route column to four true
alternatives and changed `LEAKAGE` to `NO LEAKAGE`. The second correction changed
only the `BLEND` panel into a visible human–dog–toy hybrid.

### 9.3 Visual inspection

- [x] All exact labels are present and correctly spelled.
- [x] Route boxes are parallel alternatives, not a sequential upgrade chain.
- [x] Reference sets are not depicted as first frames.
- [x] Human, dog and toy remain separately bound in the successful output.
- [x] Blend, drop, reference-copy and static-freeze failures are distinguishable.
- [x] Color is backed by border pattern, icon and line-style differences.
- [x] Original-size inspection passed.
- [x] Generic Gray profile rendering remains readable; arrows, boxes, subject
  groups and failure panels do not rely on hue alone.
- [x] No paper figure, model logo, score or watermark appears.

The chapter also includes two editable Mermaid diagrams with `accTitle` and
`accDescr`: a task-boundary decision tree and a multi-subject binding/falsification
graph. Prose alternatives are included for narrow screens and screen readers.

## 10. Proposed experiment and reproduction boundary

`PersonaBind-1` in the chapter is a proposed falsification protocol, not a
completed benchmark run. It freezes reference sets, identity-disjoint splits,
prompts, seeds, adaptation budgets and evaluators, and changes only the
personalization route. No checkpoint was downloaded or executed here, so this
batch can claim a primary-source-grounded review, artifact audit and executable
evaluation design, but not independent confirmation of any paper's quality,
speed, memory, open-set generalization or safety.

## 11. Validation record

Final integration checks were run on 2026-08-30:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: the 13 non-README changed/new Markdown files passed with 0 issues; README retains exactly two pre-existing MD001/MD028 findings outside the edited lines |
| Local links and images | 307 relative references across 14 Markdown files resolve to 143 unique local targets; 0 missing |
| Reference anchors | The new chapter has 35 references, 86 citation occurrences and all 35 references cited; no missing, orphan, duplicate, mismatched or skipped anchor |
| External primary sources | All 35 chapter URLs were checked: 33 returned HTTP 200, the TMM DOI returned 202, and the ACM DOI's automation 403 was separately verified through DOI/Crossref metadata; no deterministic 404, 410, 5xx or timeout |
| Mermaid | 20 blocks across 11 affected documentation pages all contain `accTitle` / `accDescr` and rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Mermaid visual check | Both chapter diagrams and the revised eight-branch reading route were rendered to PNG and inspected in color and grayscale; no clipped node, broken branch or color-only meaning was found |
| Generated teaching visual | The 1672×941 RGB PNG was re-inspected at original size and in grayscale after integration; labels, route alternatives, three subject slots and blend/drop/copy/freeze failures remain legible; SHA-256 matches the value in section 9 |
| Timeline preservation | The same 75 HTML image `src` / `alt` pairs remain in the same order as `HEAD`; every alt is non-empty and every local image exists |
| Bibliography | Registry, metadata, BibTeX and generated index contain the same 132 citekeys; section N contains 8 entries, all 66 unique GitHub URLs match the Star snapshot, Python compile and updater `--check` passed |
| Patch, credential and size hygiene | `git diff --check` passed; strong credential-pattern scanning over all 20 changed/untracked files found no candidate; the largest file is the 1,991,841-byte teaching PNG and no changed file exceeds 5 MiB |

Temporary Mermaid renderings and grayscale derivatives remain outside the
repository. These checks validate the review, source boundaries and teaching
artifacts; they do not train, download or independently run a video-generation
checkpoint. `PersonaBind-1` remains a proposed protocol.
