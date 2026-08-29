# Research audit: causal, streaming and real-time video generation

This file records the search and verification trail behind `docs/generative-models/causal-streaming-generation.md`.

## Scope

- **Review date:** 2026-08-29 (Asia/Shanghai)
- **Primary question:** How did video generation move from offline bidirectional diffusion to causal, few-step, long-horizon and deadline-aware streaming systems?
- **Subquestions:** training/inference mismatch, teacher–student architectural mismatch, few-step distillation, KV/memory scaling, long-horizon drift, online serving, interactive condition changes and 4D consistency.
- **Time window:** 2024-01-01 through 2026-08-29, with older work included only when required to explain a mechanism.
- **Evidence policy:** peer-reviewed paper or official proceedings first; arXiv paper for new work without proceedings; project pages and model cards only for implementation or availability claims. Secondary surveys were not used as evidence for method claims.

## Search strategy

Queries combined the following phrases and close variants:

```text
"autoregressive video diffusion"
"causal video diffusion"
"streaming video generation"
"real-time interactive video generation"
"rolling KV cache" video diffusion
"self forcing" video
"causal forcing" video
"long-horizon" autoregressive video diffusion
```

The search was deliberately broader than the final chapter. API result counts below are **raw responses before deduplication and eligibility screening**, not PRISMA-style included-study counts.

| Source | Search use | Raw response observed | Notes |
|---|---|---:|---|
| arXiv search/API | exact phrases, title and abstract discovery | 65 for the main exact-phrase query | Primary full records and submission histories used for preprints |
| OpenAlex | citation-neighborhood and synonym discovery | 8,217 for the broad query | Very noisy; used only to find candidates, never as the source of a technical claim |
| Crossref | venue and DOI cross-check | 110,507 for the broad query | Very noisy; useful for proceedings metadata |
| DBLP | publication-title and venue discovery | 58 for the focused publication query | Used to find 2026 causal/streaming papers, then verified at the primary paper |
| MLSys proceedings | official venue verification | direct paper lookup | Used for StreamDiffusionV2 |
| CVF / NeurIPS / OpenReview | official venue verification where available | direct paper lookup | Used when the paper had a formal proceedings version |
| Semantic Scholar API | attempted metadata/citation lookup | HTTP 429 | Rate-limited; no claims or counts from this API were used |

## Inclusion criteria

A work was included when it changed or carefully evaluated at least one of the following:

1. causal/frame-wise/chunk-wise factorization of video generation;
2. self-generated-history training or teacher–student causal distillation;
3. few-step sampling needed for streaming latency;
4. bounded, compressed, recurrent, hierarchical or retrieved long-term memory;
5. time-to-first-frame, frame deadlines, jitter or multi-GPU streaming serving;
6. interactive prompt/action changes during rollout;
7. long-horizon geometry, motion or source-preservation objectives specific to streaming AR models.

## Exclusion criteria

- image-only streaming diffusion without a temporal video model;
- generic video acceleration that did not address causal or streaming generation;
- product announcements without a technical paper or system card;
- duplicate arXiv versions after the latest revision had been checked;
- papers found only in aggregators when the primary paper could not be verified;
- a speed claim lacking enough context to state at least its hardware or model setting.

## Primary-source ledger

### Foundations and first causal/few-step line

| Work | Primary source | Status used in chapter | Reason included |
|---|---|---|---|
| Diffusion Forcing | <https://arxiv.org/abs/2407.01392> | NeurIPS 2024 | Independent per-token noise and causal variable-horizon rollout |
| CausVid | <https://arxiv.org/abs/2412.07772> | CVPR 2025 | Bidirectional teacher to 4-step causal student, KV-cached streaming |
| Self Forcing | <https://arxiv.org/abs/2506.08009> | NeurIPS 2025 Spotlight | On-policy autoregressive rollout on self-generated history |
| LongLive | <https://arxiv.org/abs/2509.22622> | arXiv 2025 | Prompt recache, frame sink, train-long-test-long and author-reported single-H100 speed |
| Rolling Forcing | <https://arxiv.org/abs/2509.25161> | arXiv 2025 | Joint progressive-noise window, attention sink and long-window distillation |
| StreamDiffusionV2 | <https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html> | MLSys 2026 | SLO-aware online system, TTFF, deadline scheduling and multi-GPU pipeline |

### Distillation and train–test alignment

| Work | Primary source | Status used in chapter | Distinct claim verified |
|---|---|---|---|
| Causal Forcing | <https://arxiv.org/abs/2602.02214> | ICML 2026 | Frame-level injectivity problem and AR-teacher ODE initialization |
| Causal Forcing++ | <https://arxiv.org/abs/2605.15141> | arXiv 2026 | Causal consistency initialization for frame-wise 1–2-step generation |
| Causal-rCM | <https://arxiv.org/abs/2606.25473> | technical report 2026 | Teacher-forcing consistency plus self-forcing DMD recipe |
| Video-Mirai | <https://arxiv.org/abs/2606.03971> | arXiv 2026 | Training-only foresight supervision for causal representations |
| Self Gradient Forcing | <https://arxiv.org/abs/2607.20368> | arXiv 2026 | Two-pass restoration of future-to-context KV gradients |

### Long-horizon memory and efficiency

| Work | Primary source | Status used in chapter | Memory design |
|---|---|---|---|
| FAST-AR | <https://arxiv.org/abs/2602.01801> | ICML 2026 | Temporal cache compression plus ANN sparse cross/self-attention |
| Quant VideoGen | <https://arxiv.org/abs/2602.02958> | ICML 2026 | Training-free 2-bit KV-cache quantization |
| Light Forcing | <https://arxiv.org/abs/2602.04789> | ICML 2026 | Chunk-aware hierarchical sparse attention |
| Rolling Sink | <https://arxiv.org/abs/2602.07775> | arXiv 2026 | Training-free cache maintenance beyond the training horizon |
| Sparse Forcing | <https://arxiv.org/abs/2604.21221> | arXiv 2026 | Trainable persistent block-sparse attention |
| Forcing-KV | <https://arxiv.org/abs/2605.09681> | arXiv 2026 | Head-specialized static and dynamic KV pruning |
| ARL2 | <https://arxiv.org/abs/2605.16579> | arXiv 2026 | Intra-frame softmax plus fixed recurrent cross-frame state |
| VideoMLA | <https://arxiv.org/abs/2605.30351> | arXiv 2026 | Shared low-rank content cache and decoupled 3D-RoPE key |
| LongLive-RAG | <https://arxiv.org/abs/2606.02553> | arXiv 2026 | Content-addressable retrieval over self-generated latent history |
| FadeMem | <https://arxiv.org/abs/2606.10671> | arXiv 2026 | Dense-near, sparse-far hierarchical consolidation under fixed budget |
| VideoSSM | <https://arxiv.org/abs/2512.04519> | arXiv 2025 | Global state-space memory plus local context window |

### Geometry and task extensions

| Work | Primary source | Status used in chapter | Extension |
|---|---|---|---|
| Stream4D | <https://arxiv.org/abs/2608.19556> | arXiv 2026, submitted 2026-08-20 | Dynamic 4D reconstruction reward and motion prior; treated as provisional |
| MV-Forcing | <https://arxiv.org/abs/2607.05376> | ECCV 2026 | Temporal and view-wise autoregression bridged by 4D geometry |
| JoyAI-Video-Edit | <https://arxiv.org/abs/2608.03974> | arXiv 2026 | Open-ended causal video editing and source-anchored distillation |

## Candidate works intentionally not promoted to milestones

The search also surfaced FramePack, SkyReels-V2, MAGI-1, LongVie 2 and many 2026 cache/forcing variants. They remain important context, but the chapter's milestone table uses a stricter test: a milestone must change the problem definition, causal factorization, training distribution, memory abstraction or deployment protocol. A higher resolution, longer demo or another variant of an existing sparse-cache recipe is not by itself a milestone.

Primary records checked for this contextual group:

- FramePack: <https://arxiv.org/abs/2504.12626>
- SkyReels-V2: <https://arxiv.org/abs/2504.13074>
- MAGI-1: <https://arxiv.org/abs/2505.13211>
- LongVie 2: <https://arxiv.org/abs/2512.13604>

## Claim-handling rules

1. Performance numbers are introduced as **author-reported**, not independently replicated.
2. FPS is not compared across different GPU counts, resolutions, precisions, model sizes or denoising steps.
3. A long-duration demo is not evidence of real-time execution.
4. Dynamic prompting is not evidence of action-conditioned world modeling.
5. Causal attention is an information constraint, not proof of causal physical reasoning.
6. New August 2026 preprints are marked provisional and are not used to rewrite older peer-reviewed conclusions.

## Figure generation and verification

The chapter overview figure was generated with the built-in image-generation tool after the scientific-schematics script could not run because `OPENROUTER_API_KEY` was not configured.

- Workspace asset: `assets/diagrams/causal-streaming-video-generation.png`
- Dimensions: 1672 × 941
- SHA-256: `41a0f7564b40803164b92af3938c8a933f16d89e16fe5676d724d046db9bc079`
- Visual checks: all 11 requested labels rendered correctly; offline full-clip/bidirectional arrows are distinct from the causal chunk loop; the rolling memory is visibly bounded; no logo, watermark, extra text or malformed typography was observed.

## Review limitations

- This is a focused scoping review, not a formal meta-analysis; the field is too new and the evaluation setups are not homogeneous.
- Some 2026 papers had only arXiv versions at the review date.
- OpenAlex and Crossref counts were too broad to serve as screening statistics.
- Semantic Scholar rate limiting prevented its use as the third metadata cross-check; DBLP, Crossref and official proceedings were used instead.
- Independent reproduction of speed and long-horizon quality claims was outside this repository review; the chapter therefore supplies a fair-reporting protocol rather than a ranking.
