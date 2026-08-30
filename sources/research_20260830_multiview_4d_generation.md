# Multi-view and 4D video generation research record

> Freeze date: **2026-08-30 (Asia/Shanghai)**. This is the evidence and visual audit for `docs/tasks/multiview-4d-generation.md`. It records what was searched, included, excluded and independently checked; it is not a model-reproduction report.

## 1. Why this became a separate chapter

The repository previously treated multi-view and 4D generation as a geometry-heavy branch of controllability. That was reasonable while most camera-conditioned work still produced one fixed-length path. By the freeze date, the boundary had become structurally distinct:

- CAT4D, 4DiM, SV4D/2.0 and GenXD expose camera and world time as separate query axes;
- explicit dynamic NeRF/Gaussian routes output a renderable state rather than only a video;
- 4C4D, DGGT, MoRel and 4DSurf make sparse capture, pose prediction, long-range memory and surface consistency separate contracts;
- Full-4D, MV-Forcing, Stream4D and 4DStreamCtrl extend the field toward full-scope, long multi-view, 4D-rewarded and online generation;
- the evaluation contract requires same-time cross-view, same-view cross-time, joint novel-view/novel-time and loop-closure slices that a generic control chapter cannot express compactly.

A read-only repository audit found 37 `docs/**/*.md` files, with `getting-started.md` the only page omitted from the coverage matrix. The audit also found that multi-view/4D evidence was scattered across controllable generation, I2V, V2V, causal streaming, interactive worlds and physical consistency. A separate frontier audit ranked scalable Video DiT/backbone systems as the other major missing axis; that is retained as the next structural candidate rather than folded into this task chapter.

## 2. Review questions

1. What distinguishes a video, a multi-view image set, a camera-controlled video, a multi-view video grid and a renderable 4D state?
2. Which methods reconstruct captured dynamics, which generate unobserved content and which do both?
3. How do canonical deformation, spacetime fields, dynamic Gaussians, multi-view video diffusion and geometric bridges differ?
4. What evidence can falsify a geometry or 4D-consistency claim?
5. Which 2026 results are formal publications, and which remain author-reported preprints?
6. Which timing numbers measure scene construction, rendering, generation or end-to-end streaming?

## 3. Search protocol

### 3.1 Source order

1. Official CVF, PMLR, ICLR/OpenReview and NeurIPS proceedings;
2. arXiv abstract/version pages for work not yet in official proceedings;
3. author project pages and repositories only for artifact status;
4. no survey, news article, vendor blog or leaderboard was used as evidence for a technical mechanism.

### 3.2 Query families

- `dynamic NeRF canonical deformation novel view time`
- `4D Gaussian Splatting dynamic scene rendering`
- `text-to-4D video diffusion score distillation`
- `multi-view video diffusion camera time 4D generation`
- `single video to 4D scene generation`
- `generalizable 4D video diffusion`
- `4D reconstruction sparse cameras feed-forward unposed`
- `long multi-view video generation 4D self-forcing`
- `streaming autoregressive diffusion 4D consistency reward`
- `interactive video generation online 4D control`

Searches were repeated with venue restrictions for CVPR/ICCV/ICLR/NeurIPS/ICML 2021–2026 and exact-title queries for every milestone.

## 4. Inclusion and exclusion

### 4.1 Included

- a distinct camera-time input/output or state contract;
- an explicit dynamic representation, view-time generative model or geometry bridge;
- official publication or clearly labelled preprint;
- enough public method detail to identify the mechanism and a falsifier;
- camera-only papers only when they define a boundary that prevents a 4D misclassification.

### 4.2 Excluded from direct milestones

- static 3D generation without world-time dynamics;
- ordinary T2V/I2V that never exposes camera pose or same-time multiple views;
- novel-view rendering of a static scene;
- action-conditioned world models whose main contract is action consequence rather than view/time query;
- product demos without a public technical mechanism;
- a paper name mentioned only by a later paper without a readable primary page.

### 4.3 Negative classification rules

- camera pose adherence does not prove same-time multi-view consistency;
- a $V\times T$ pixel grid does not prove an exported or editable 4D state;
- a renderable dynamic Gaussian does not prove watertight geometry or physical state;
- real-time rendering after per-scene optimization does not prove real-time construction;
- a static 3D reconstruction reward can penalize genuine motion and reward a frozen video;
- novel view is an observation intervention, not an environment action.

## 5. Primary-source status ledger

| Work | First public | Formal status by freeze date | Contract used in the chapter | Evidence boundary |
|---|---|---|---|---|
| D-NeRF | 2020 preprint | CVPR 2021 | canonical deformation + dynamic radiance field | per-scene reconstruction |
| Nerfies | 2020 preprint | ICCV 2021 | casual capture + elastic deformation | topology and pose limits |
| MAV3D | 2023 preprint | ICML 2023 | text-to-video score → dynamic NeRF | no paired 4D data; SDS cost |
| Consistent4D | 2023 preprint | ICLR 2024 | video-to-4D + interpolation consistency | object-centric and per-scene |
| 4D-GS | 2023 preprint | CVPR 2024 | explicit Gaussians + 4D voxels + deformation | author rendering speed only |
| Align Your Gaussians | 2023 preprint | CVPR 2024 | dynamic Gaussians + composed priors | generated asset, not reconstruction |
| 4Real | 2024 preprint | NeurIPS 2024 | reference/freeze-time video + deformable 3DGS | staged, per-scene pipeline |
| 4DiM | 2024 preprint | ICLR 2025 | metric camera pose + timestamp diffusion | pixel query, not exported state |
| SV4D | 2024 preprint | ICLR 2025 | unified multi-frame/multi-view video diffusion | object data and fixed grid |
| GenXD | 2024 preprint | ICLR 2025 | masked conditions + view/time modules | pose-mined data uncertainty |
| EG4D | 2024 preprint | ICLR 2025 | explicit 4D without SDS | object-level generation |
| 4K4DGen | 2024 preprint | ICLR 2025 | panoramic 4D generation | author 4K/360° protocol |
| CAT4D | 2024 preprint | CVPR 2025 | arbitrary camera/time queries → deformable 3DGS | native 16-frame model plus sampling |
| 4Real-Video | 2024 preprint | CVPR 2025 | generalizable 4D video diffusion | geometry still needs independent tests |
| DriveDreamer4D | 2024 preprint | CVPR 2025 | generated driving data → 4D representation | domain-specific downstream evidence |
| GenMOJO | 2025 preprint | CVPR 2025 | object decomposition + joint occlusion splatting | difficult multi-object scenes |
| SV4D 2.0 | 2025 preprint | ICCV 2025 | blended 3D/frame attention + progressive training | author LPIPS/FV4D protocol |
| Free4D | 2025 preprint | ICCV 2025 | tuning-free single-image scene-to-4D | model-prior hallucination |
| DimensionX | 2024 preprint | ICCV 2025 | spatial/temporal decoupled video diffusion | generated views then 4DGS |
| DiST-4D | 2025 preprint | ICCV 2025 | metric-depth 4D driving generation | driving-domain scope |
| 4C4D | 2026 preprint | CVPR 2026 | four-camera 4DGS, geometry-focused opacity decay | sparse calibrated capture |
| DGGT | 2025 preprint | CVPR 2026 | feed-forward unposed driving 4D reconstruction | pose is predicted output |
| MoRel | 2025 preprint | CVPR 2026 | long-range anchor relay and bounded memory | construction and rendering remain separate |
| 4DSurf | 2026 preprint | CVPR 2026 | dynamic surface flow + overlapping segments | surface reconstruction, not open generation |
| SpaceTimePilot | 2025 preprint | CVPR 2026 | generative rendering across camera and time | generated query pixels |
| Full-4D | 2026-05-25 | arXiv preprint | dense $T\times V$ synthesis + 4DGS | no formal proceedings located |
| MV-Forcing | 2026-07-06 | arXiv preprint | 3D bridge + temporal/view self-forcing | author acceptance claim not substituted for proceedings |
| Stream4D | 2026-08-20 | arXiv preprint | dynamic 4D reconstruction reward + motion prior | no independent reward/runtime reproduction |
| 4DStreamCtrl | 2026-08-26 | arXiv preprint | 3D point tracks + four-step causal student | code/SLO not independently verified |
| GEN3C | 2025 | CVPR 2025 | incremental 3D cache for camera control | camera route adjacent to full 4D |
| WorldForge | 2025 | CVPR 2026 | zero-shot camera control for 3D/4D use | no persistent action state implied |
| BulletTime | 2025 | CVPR 2026 | decoupled world time and camera pose | one camera path, not necessarily full grid |

## 6. Claim and timing audits

### 6.1 4D-GS speed

The CVPR page reports 82 FPS at 800×800 on an RTX 3090. The paper notes that speed depends on resolution, Gaussian count and deformation-field scale. The chapter therefore records it only as an author rendering result after scene optimization.

### 6.2 CAT4D output length

The paper states that the multi-view video diffusion model was trained to generate 16 frames at once, then uses a sampling strategy to produce a larger collection for reconstruction. “Unbounded collection” is a sampling construction, not proof of drift-free open-horizon dynamics.

### 6.3 SV4D 2.0 metrics

The official ICCV abstract reports relative LPIPS and FV4D improvements against SV4D. The chapter does not copy the percentages into a cross-paper leaderboard because evaluator version, dataset and query grid are paper-specific.

### 6.4 Static critic failure in Stream4D

The arXiv abstract explicitly states that a rigid 3D reconstruction critic can treat motion as reconstruction error and reward freezing. This is used as a negative-classification example, not as independently reproduced evidence that the proposed reward fixes every backbone.

### 6.5 4DStreamCtrl SLO

The preprint abstract reports four denoising steps, 480p, 20 FPS on one high-end GPU, length-independent memory and hundreds of frames. The chapter labels every number author-reported and requires separate TTFF, steady-state latency, decoder, external state, load and deadline measurements.

## 7. Paper-review structure

Each detailed review in the chapter uses the same six fields:

1. question or missing contract;
2. input and output;
3. representation or generation mechanism;
4. evidence actually shown;
5. unresolved limitation;
6. formal publication and artifact boundary.

The milestone criterion is a change in task input/output, representation, query interface or evidence standard. A higher visual score alone is not a milestone.

## 8. Visual record

### 8.1 Learning objective

Show, without relying only on color, that:

- video = one row across time;
- multi-view image = one column at a fixed time;
- camera-controlled video = one selected view per time, a diagonal path;
- multi-view video / 4D = a camera-by-time grid;
- a renderable state supports repeated arbitrary $(v,t)$ queries;
- a plausible diagonal path cannot prove a consistent 4D world.

### 8.2 Generation method

Tool: OpenAI image generation through the repository image-generation workflow. One candidate was generated and accepted after scientific inspection; no paper figure was supplied as a reference.

Prompt summary:

> Create a clean publication-quality educational infographic with camera view on the vertical axis and world time on the horizontal axis. Use a consistent paper-bird-and-cube scene in a 4×5 grid. Highlight a row, column, diagonal and complete grid as VIDEO, MULTI-VIEW IMAGE, CAMERA-CONTROLLED VIDEO and MULTI-VIEW VIDEO / 4D GRID. Add a renderable 4D state queried by $(v,t)$ and the warning that a plausible diagonal video does not prove a consistent 4D world. Use shape and line-style redundancy, exact short labels, no scores or model logos.

### 8.3 Accepted asset

- Project path: `assets/diagrams/multiview-4d-camera-time-contract.png`
- Generated-source path retained outside the repository: `/Users/xiaodong/.codex/generated_images/01a04c93-4978-7ad2-9956-339854046832/exec-74cc2cfe-ba6a-4762-993e-32b1d8df6136.png`
- Dimensions: `1536 × 1024`
- Format: PNG
- SHA-256: `c8475132b45324ca31cc8969ec35bc2ad90925d3396a4c5a1fbf0555b40c74e7`

The tool returned 3:2 rather than the requested 16:9. It was retained because every label, grid cell and warning is legible at the chapter width, while cropping to 16:9 would remove the contract cards or warning.

### 8.4 Visual inspection checklist

- [x] Vertical camera and horizontal world-time axes are correct.
- [x] Row, column, diagonal and full-grid claims are visually distinct.
- [x] The bird and cube maintain recognizable identity across cells.
- [x] The renderable-state box is downstream of the full grid, not a synonym for a diagonal video.
- [x] The warning explicitly blocks the main false implication.
- [x] No paper name, model logo, score, watermark or copied figure layout is present.
- [x] Grayscale rendering and final repository-width inspection passed after integration.

### 8.5 Editable companion diagrams

The chapter also contains two Mermaid diagrams with `accTitle` and `accDescr`:

1. five technical routes from observation/condition to pixel grid or renderable state;
2. six independent evidence gates from observed-region fidelity through system cost.

These are the editable and screen-reader-addressable alternatives to the raster overview.

## 9. Reproduction boundary

`GridFork-1` is a proposed protocol, not a reported run. No model checkpoint was downloaded or executed in this batch. The chapter can therefore claim a current, primary-source-grounded literature and evidence review, but not independent confirmation of model quality, geometry, 4D consistency, speed or memory.

## 10. Validation record

This section was completed after cross-page integration on 2026-08-30.

- [x] Markdown lint: 30 changed/new non-README Markdown files, 0 errors; README retains exactly two pre-existing warnings.
- [x] Reference closure: 32 references, 78 citation occurrences, 32 unique cited references, no orphan or numbering gap.
- [x] Local link and image closure: 465 relative links and 99 image targets checked across 31 changed/new Markdown files, 0 missing.
- [x] External URL audit: all 32 chapter reference URLs returned HTTP 200 after two deterministic 404 corrections.
- [x] Mermaid accessibility metadata and rendering: both chapter diagrams plus the updated reading-route and timeline diagrams rendered with Mermaid CLI 11.16.0 and system Chrome.
- [x] Original-size and grayscale visual inspection: the generated PNG and both chapter Mermaid diagrams remain interpretable without color; the exact prose fallback covers the wide route diagram on narrow screens.
- [x] Timeline media preservation: all 75 existing HTML images remain present with non-empty alt text.
- [x] `git diff --check`: passed.
- [x] Changed-file credential-pattern scan: passed with no candidate secret.
