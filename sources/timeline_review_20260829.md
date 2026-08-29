# Timeline review log

Review date: 2026-08-29

Target: `docs/timeline.md`

## Review method

- Used the first public paper or first official institutional announcement as the year anchor.
- Preferred papers, proceedings, author project pages, official model cards and official research announcements.
- Separated peer-reviewed/preprint evidence from vendor-reported demonstrations.
- Checked each node along four axes: representation, temporal mechanism, conditioning/control, and evidence type.
- Kept product releases out of the technical backbone when their architecture and training objective were not publicly documented.

## High-confidence corrections

- Changed the opening period from `1990s–2003` to `1981–2003`.
- Restored the specific 2014 node, [Video (Language) Modeling](https://arxiv.org/abs/1412.6604), instead of the generic label "video sequence modeling with RNNs".
- Recorded Dynamic Textures as `2001/2003`: [ICCV 2001](https://doi.org/10.1109/ICCV.2001.937658) was the first publication; [IJCV 2003](https://doi.org/10.1023/A:1021669406132) was the expanded journal version.
- Changed MAGVIT to 2022 and Latent Video Diffusion to 2023 under the first-publication rule.
- Changed PlaNet to 2018, Dreamer to 2019, and MuZero to 2019/2020.
- Replaced the ambiguous name "Cosmos 1" with NVIDIA Cosmos / the first Predict1 platform generation.
- Removed unsupported "depth" wording from the Veo 3 zero-shot capability summary.
- Reframed Cosmos 3 and GWM-1 as model families or separately post-trained branches, not single universal checkpoints.
- Separated Kling and Seedance creative audio-video models from closed-loop Physical AI evidence.

## Major additions

### Motion, prediction and probability

Video Rewrite, Action-Conditional Video Prediction, Video Pixel Networks, SV2P, FVD and VideoFlow.

### Discrete tokens and language-model-style generation

NÜWA, CogVideo and VideoPoet.

### Diffusion, DiT and Flow Matching

Latent Diffusion, Rectified Flow / Flow Matching, DiT, W.A.L.T., VideoCrafter, CogVideoX, Movie Gen, HunyuanVideo and Wan 2.1.

### Decision and interactive world models

UniSim, DINO-WM, Genie 2, Cosmos Predict2, Matrix-Game 2 and Marble.

### Current 2026 state

Cosmos 3, V-JEPA 2.1, LeWorldModel, EB-JEPA, Kling 3, Seedance 2.0, MiniMax H3 and Seedance 2.5.

## Evidence boundaries enforced in the rewrite

- A plausible video is not evidence of correct physics or causality.
- Feature prediction is not the same as pixel generation.
- Open-loop rollout is not the same as reliable closed-loop control.
- A real-time result in one game is not evidence of a general game engine.
- Vendor benchmarks and demonstrations remain attributed to the vendor unless independently reproduced.
- Open weights are not automatically equivalent to a fully open-source training system.

## Illustration policy

- One independent imagegen call per final timeline node.
- 16:9 scientific editorial illustrations with a shared visual language.
- No model logos, paper titles, labels or embedded prose.
- Images explain mechanisms and are explicitly marked as conceptual illustrations, not paper figures or actual model outputs.
- Project copies are normalized to 960×540 JPEG for practical README rendering; the original imagegen files remain in the Codex generated-image store.

## Resource-link audit

- Every one of the 75 nodes now has an explicit resource row for Paper/Report, Project, Code, Weights and Demo.
- Only author- or institution-maintained implementations are labeled `Code`; community reimplementations are excluded.
- MAGVIT-v2 keeps the MAGVIT v1 repository under `Related Code`, MuZero labels the author gist as `Pseudocode`, and the GameNGen website repository is not treated as model code.
- Retired, archived, unavailable and gated resources are marked in place instead of being presented as currently runnable.
- Detailed per-node evidence and caveats are stored in `timeline_resource_audit_01_25.*`, `timeline_resource_audit_26_50.*` and `timeline_resource_audit_51_75.*`.

## Final validation

- 75 milestone cards, 75 image references and 75 unique alt descriptions.
- Every card contains the same three evidence fields: representation/mechanism, control/task and significance/boundary.
- The resource rows contain 235 links across 226 unique URLs: 39 nodes link official code, 26 link weights and 20 link an official demo or executable notebook; the other nodes show an explicit unavailable/not-applicable state.
- All four local cross-references resolve in the current workspace.
- All 75 referenced images exist, are JPEG 960×540, and have unique SHA-256 hashes.
- HTML table tags are balanced: 75 tables, 75 rows and 150 cells.
- `git diff --check -- docs/timeline.md` reports no whitespace errors.
- Visual review used per-batch contact sheets plus `sources/timeline-contact-sheet-all.jpg`; two images with suspected pseudo-text were regenerated before final delivery.
- The normalized imagegen prompt template and scope rules are recorded in `sources/timeline_image_prompt_set.md`.

## Incremental addition: MiniMax H3

- Added MiniMax H3 under the 2026 native audio-video section using the first official release date, 2026-07-31.
- Distinguished the open H3-Base checkpoints from the full hosted system: H3-Context-IR, H3-Regenerate-2K and the initial sparse-attention implementation were not included in the first open release.
- Recorded architecture and I/O specifications only where the official release or open-source announcement provides them.
- Detailed source notes are stored in `sources/research_20260829_minimax_h3.md`.
