# Video reasoning 2026 summer refresh audit — 2026-08-30

## Scope and freeze

- **Target:** `docs/video-reasoning.md`
- **Search and verification date:** 2026-08-30, Asia/Shanghai
- **Purpose:** close the coverage-audit gaps for UniVR, the newer *Thinking in Video* paper and RuleMaze; normalize the formal publication status of VChain, MME-CoF, *Thinking with Video* and VIPER; add a mechanism figure that separates output, denoising and interaction time.
- **Previous evidence base:** `research_20260829_video_reasoning_vbvr.md` remains the full historical and VBVR forward/backward-citation audit. This record is an incremental refresh, not a replacement.

## Questions

1. Which July–August 2026 works materially change the chapter's account of visual reasoning trajectories, causal evidence or unseen-rule evaluation?
2. Which works are about video generators, and which are adjacent MLLM controls that must not be counted as VGM milestones?
3. What does “pure visual” mean in UniVR, and what language or external evaluator remains in the system?
4. Does a plausible generated future establish explicit causal understanding?
5. Which proceedings pages resolve the venue labels of VChain, MME-CoF, *Thinking with Video* and VIPER?
6. How can one diagram make three different reasoning time axes visible without presenting an unproved unified mechanism as fact?

## Search protocol

Discovery queries combined exact titles and topic terms:

- `site:arxiv.org UniVR thinking visual space unified visual reasoning`
- `site:github.com bytedance UniVR official`
- `"Thinking in Video" "reason about the real world"`
- `site:github.com BRZ911 Thinking-in-Video`
- `RuleMaze rule-compliant visual spatial planning`
- `site:huggingface.co/datasets RuleMaze`
- `VChain ACL Anthology 2026`
- `MME-CoF CVPR 2026 Findings`
- `Thinking with Video CVPR 2026 openaccess`
- `Beyond the Last Frame ACL 2026`

Inclusion required an arXiv record plus an official repository/data or formal proceedings page when available. Search snippets, community lists and aggregator pages were discovery aids only. Quantitative claims were included only when recoverable from the paper, official repository or official dataset card. The freeze treats unaccepted 2026 items as preprints.

## Primary-source ledger

| Work or record | Primary source | Verified contribution or metadata | Boundary used in the chapter |
|---|---|---|---|
| UniVR | [arXiv:2607.12800](https://arxiv.org/abs/2607.12800), [official repository](https://github.com/bytedance/UniVR) | Submitted 2026-07-14; Emu3.5 SFT/RL framework; repository describes VR-X with about 1.5M raw samples, about 310K SFT samples and 3K curated RL samples; VR-GRPO combines global and step-focal reward | “Pure visual demonstrations” describes the reasoning trajectory, not a language-free system: the task still has a text instruction, and reward construction uses a VLM plus visual features |
| Thinking in Video | [arXiv:2607.17523](https://arxiv.org/abs/2607.17523), [official repository](https://github.com/BRZ911/Thinking-in-Video) | Submitted 2026-07-20; Causal-Generative Dual-Judge separates explicit causal perception from implicit generated-future prediction; official materials describe 1,500 videos, including 900 Video-MME and 600 paired input/gold-future cases | A plausible future is not evidence that the generator explicitly recovered the correct causal relation; this paper must not be confused with 2025's *Thinking with Video* |
| RuleMaze | [arXiv:2608.20237](https://arxiv.org/abs/2608.20237), [official repository](https://github.com/oceanflowlab/RuleMaze), [dataset card](https://huggingface.co/datasets/Fish-03/RuleMaze) | Submitted 2026-08-20; programmatic rule/logic/validator generation; disentangled perception, execution and rule verification; dataset card showed 119,595 rows with train/seen/unseen-rule splits on the freeze date | Adjacent MLLM visual-planning benchmark, not a video-generator benchmark. It contributes split and validator methodology, not evidence that a VGM can plan under unseen rules |
| VChain | [ACL Anthology](https://aclanthology.org/2026.findings-acl.12/) | Findings of ACL 2026, pages 226–250, DOI 10.18653/v1/2026.findings-acl.12 | Formal venue replaces the arXiv-only citation |
| MME-CoF | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026F/html/Guo_Are_Video_Models_Ready_as_Zero-Shot_Reasoners_An_Empirical_Study_CVPRF_2026_paper.html) | Findings of CVPR 2026, pages 9175–9184 | It is a CVPR Findings paper, not a main-track CVPR paper |
| Thinking with Video | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tong_Thinking_with_Video_Video_Generation_as_a_Promising_Multimodal_Reasoning_CVPR_2026_paper.html) | CVPR 2026 main proceedings, pages 41121–41129 | Distinct title, authors and evidence from the newer *Thinking in Video* preprint |
| VIPER / Beyond the Last Frame | [ACL Anthology](https://aclanthology.org/2026.acl-long.934/) | ACL 2026 long paper, pages 20393–20409, DOI 10.18653/v1/2026.acl-long.934 | Formal venue replaces the arXiv-only citation |

## Evidence interpretation

### UniVR: trajectory modality is not system modality

UniVR is relevant because it trains a unified visual autoregressive model to carry intermediate reasoning in visual form. The repository makes the training stack auditable: SFT can be full-parameter or LoRA, RL uses VR-GRPO, and the released pipeline includes data and inference support. Its reward is not a direct program-only oracle for every task. The global evaluator and step selection involve a VLM and CLIP-feature uncertainty. The chapter therefore credits visual-trajectory learning while retaining the external-evaluator attribution.

### Causal-generative dual judging

The newer *Thinking in Video* asks whether explicit causal answers and generated futures agree. This is stronger than judging a future only for visual plausibility: a model may sample an acceptable-looking continuation while failing the explicit causal question, or answer a causal question without generating the correct future. The chapter uses this as a measurement result and warning, not as proof that either judge fully recovers real-world causality.

### RuleMaze as an adjacent control

RuleMaze contributes three reusable ideas: rule-family splits, executable validators and prefix-progress metrics. Those are directly useful for designing video-reasoning benchmarks. However, its primary model class and output protocol are multimodal-language planning rather than video generation. The chapter therefore labels the row “adjacent MLLM control” and excludes it from VGM rankings or milestone counts.

## Venue corrections

| Reference | Previous label | Frozen label |
|---|---|---|
| VChain | Findings of ACL with arXiv link | Findings of ACL 2026, ACL Anthology formal page, pages 226–250 |
| MME-CoF | CVPR | Findings of CVPR 2026, CVF formal page, pages 9175–9184 |
| Thinking with Video | CVPR with arXiv link | CVPR 2026 main proceedings, CVF formal page, pages 41121–41129 |
| VIPER | ACL with arXiv link | ACL 2026 long paper, ACL Anthology formal page, pages 20393–20409 |

## Generated mechanism figure

- **Asset:** `assets/diagrams/video-reasoning-three-clocks.png`
- **Generator:** built-in OpenAI image generation tool
- **Generated:** 2026-08-30
- **Dimensions:** 1672 × 941 RGB PNG
- **SHA-256:** `e516d6a14ac29968f17fe5982d6ccc6450624d54c454ebf29d24c2c78df2c36a`

Prompt summary: create a white-background, vector-like 16:9 scientific teaching schematic with three horizontal lanes. Output time shows a dot traversing maze frames and warns that an observable trajectory is not proof of causal computation. Denoising time shows noise, early plan, constraint binding and late rendering. Interaction time closes action, short rollout, verifier and replan into a feedback loop. A right-side ladder separates final answer, process validity, causal intervention and closed-loop return. No model names, benchmark numbers, logos, gradients or decorative imagery.

The raster image is explanatory rather than evidentiary: it contains no performance claim. The chapter keeps a deterministic Mermaid version, alt text, a caption and a sequential text alternative. The Mermaid is the editable source of exact relationships; the PNG is the fast visual overview.

## Acceptance matrix

| Requirement | Chapter location | Check |
|---|---|---|
| Freeze date and incremental audit link | chapter lead | date and two audit links present |
| UniVR scale, method and attribution boundary | §6.8 and §8.2 | visual trajectory separated from text input and external evaluator |
| Two similarly named papers disambiguated | §6.8 | titles, dates and references separated |
| Causal-generative gap | §6.8 and benchmark table | future plausibility not upgraded to causal understanding |
| RuleMaze scope | §6.8 and benchmark table | explicitly labeled adjacent MLLM control |
| Formal venue normalization | references 4, 5, 6 and 15 | formal proceedings pages and page ranges used |
| Three reasoning clocks | §10.5 | generated PNG, Mermaid, caption and text alternative |

## Validation results

| Check | Result |
|---|---|
| Markdown | `markdownlint-cli2` 0.23.2 / markdownlint 0.41.1 checked the chapter and this record: 0 issues |
| Reference closure | 58 unique in-text reference targets and 58 anchors; no missing or orphaned anchor |
| Local paths | 12 local links/images resolved from the chapter; no missing target |
| Mermaid syntax | The new block rendered with Mermaid CLI 11.16.0 and system Chrome to a non-empty 38,748-byte SVG |
| Mermaid visual check | A 1846 × 2088 audit PNG was inspected at original detail: all three clocks, the feedback loop, edge labels and the L1–L4 ladder are visible with no clipping |
| Generated PNG | 1672 × 941 RGB; SHA-256 matches the asset ledger; grayscale mean 0.918393 and standard deviation 0.220759 confirm non-blank contrast |
| PNG visual check | Inspected at original detail: maze states, three denoising phases, interaction feedback arrow and evidence ladder are legible; no model names, scores or contradictory capability claims appear |
| Diff hygiene | `git diff --check` passed after the chapter edit |

The Mermaid SVG/PNG outputs were created in a temporary directory for rendering review and were not added to the repository. The generated teaching PNG is the only new visual asset for this chapter. No commit or push is performed by this refresh task.
