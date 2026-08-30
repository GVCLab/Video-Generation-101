# Research audit: causal, streaming and real-time video generation

This file records the search and verification trail behind `docs/generative-models/causal-streaming-generation.md`.

## Scope

- **Review date:** 2026-08-30 (Asia/Shanghai; refreshed from the original 2026-08-29 audit)
- **Primary question:** How did video generation move from offline bidirectional diffusion to causal, few-step, long-horizon and deadline-aware streaming systems, and which claims belong to codec, generator, commit or runtime layers?
- **Subquestions:** causal codec/generator separation, training-history exposure, teacher–student architectural mismatch, few-step versus measured NFE, streaming commit/revision, KV/resident-memory scaling, open-horizon drift, online serving/backpressure, interactive condition changes and world-model negative classification.
- **Time window:** 2024-01-01 through 2026-08-30, with older work included only when required to explain a mechanism.
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
"streaming commit" video generation lookahead revision
"first frame latency" video diffusion deadline jitter
"backpressure" real-time video generation
"open horizon" video generation constant memory
"separable causal diffusion" video
"flow cache" autoregressive video
"motion stream" interactive video generation
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
7. explicit streaming commit, lookahead, revision, deadline or recovery contracts;
8. long-horizon geometry, motion or source-preservation objectives specific to streaming AR models.

## Exclusion criteria

- image-only streaming diffusion without a temporal video model;
- generic video acceleration that did not address causal or streaming generation;
- product announcements without a technical paper or system card;
- duplicate arXiv versions after the latest revision had been checked;
- papers found only in aggregators when the primary paper could not be verified;
- a speed claim lacking enough context to state at least its hardware or model setting.
- a product or project label of "real-time" without a paper/system card or reproducible timing boundary;
- causal codec work whose only contribution was representation compression; it remains owned by the tokenizer audit unless it also changed the generator/serving contract.

## Primary-source ledger

### Foundations and first causal/few-step line

| Work | Primary source | Status used in chapter | Reason included |
|---|---|---|---|
| Diffusion Forcing | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html> | NeurIPS 2024 | Independent per-token noise; not self-rollout, few-step or a commit/SLO guarantee |
| CausVid | <https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html> | CVPR 2025 | Bidirectional teacher to 4-step causal student; history is mainly GT/noised GT, not on-policy matching |
| Self Forcing | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html> | NeurIPS 2025 | On-policy autoregressive rollout on self-generated history; detached history leaves a context-gradient gap |
| LongLive | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html> | ICLR 2026; first preprint 2025 | Prompt recache, frame sink, train-long-test-long and author-reported single-H100 speed |
| Rolling Forcing | <https://openreview.net/forum?id=IAyzXjbfwo> | ICLR 2026; first preprint 2025 | Joint progressive-noise window, attention sink and long-window distillation; window is not strictly frame-serial |
| StreamDiffusionV2 | <https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html> | MLSys 2026 | SLO-aware online system, TTFF, deadline scheduling and multi-GPU pipeline |
| Separable Causal Diffusion | <https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html> | CVPR 2026 | Once-per-frame causal encoder plus multi-step frame-wise diffusion renderer |
| FlowCache | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/85dc8f85ff978b9c606d3b2f5b0da69a-Abstract-Conference.html> | ICLR 2026 | Training-free per-chunk feature-cache decisions plus bounded resident KV compression |
| MotionStream | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/0cece806cd3d1dfad4a893f016ad3d7d-Abstract-Conference.html> | ICLR 2026 | Online motion/camera controls with Self Forcing, sinks and rolling KV |

### Distillation and train–test alignment

| Work | Primary source | Status used in chapter | Distinct claim verified |
|---|---|---|---|
| Causal Forcing | <https://arxiv.org/abs/2602.02214> | [ICML 2026 official accepted-paper list](https://icml.cc/Downloads/2026); PMLR page not located at freeze date | Frame-level injectivity problem and AR-teacher ODE initialization |
| Causal Forcing++ | <https://arxiv.org/abs/2605.15141> | arXiv 2026 | Causal consistency initialization for frame-wise 1–2-step generation |
| Causal-rCM | <https://arxiv.org/abs/2606.25473> | technical report 2026 | Teacher-forcing consistency plus self-forcing DMD recipe |
| Video-Mirai | <https://arxiv.org/abs/2606.03971> | arXiv 2026 | Training-only foresight supervision for causal representations |
| Self Gradient Forcing | <https://arxiv.org/abs/2607.20368> | arXiv 2026 | Two-pass restoration of future-to-context KV gradients |

### Long-horizon memory and efficiency

| Work | Primary source | Status used in chapter | Memory design |
|---|---|---|---|
| FAST-AR | <https://arxiv.org/abs/2602.01801> | [ICML 2026 accepted](https://icml.cc/Downloads/2026); technical claims from arXiv | Temporal cache compression plus ANN sparse cross/self-attention |
| Quant VideoGen | <https://arxiv.org/abs/2602.02958> | [ICML 2026 accepted](https://icml.cc/Downloads/2026); technical claims from arXiv | Training-free 2-bit KV-cache quantization |
| Light Forcing | <https://arxiv.org/abs/2602.04789> | [ICML 2026 accepted](https://icml.cc/Downloads/2026); technical claims from arXiv | Chunk-aware hierarchical sparse attention |
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
| MV-Forcing | <https://arxiv.org/abs/2607.05376> | arXiv; authors report ECCV 2026, official proceedings page not located | Temporal and view-wise autoregression bridged by 4D geometry |
| JoyAI-Video-Edit | <https://arxiv.org/abs/2608.03974> | arXiv 2026 | Open-ended causal video editing and source-anchored distillation |

## Release-surface and timing-boundary audit

| Work | Code / weights status checked 2026-08-30 | Timing or NFE boundary retained in chapter |
|---|---|---|
| Diffusion Forcing | <https://github.com/buoyancy99/diffusion-forcing>; code/checkpoints and a paper branch are public | No inherent few-step or real-time claim |
| CausVid | <https://github.com/tianweiy/CausVid>; training, inference and weights public | 640×352, 120 frames/10 s, 4 steps, one H100; 9.4 FPS and TTFF 1.3 s include text encoder, DiT and VAE |
| Self Forcing | <https://github.com/guandeh17/Self-Forcing>; code, checkpoints and training public | Wan2.1-1.3B, 832×480, 4 steps, one H100; chunk/frame results are not interchangeable |
| Rolling Forcing | <https://github.com/TencentARC/RollingForcing>; code, checkpoint and training public | Reported 0.76 s is steady-state latency, not TTFF; GPU/VAE boundary is insufficient for cross-paper comparison |
| LongLive | <https://github.com/NVlabs/LongLive>; code and weights public | 1.3B, 832×480, one H100, 20.7 FPS and 240 s demonstration; paper claims map to v1.0, not later v2 infrastructure |
| Causal Forcing / ++ | <https://github.com/thu-ml/Causal-Forcing>; code/config/checkpoints public | CF++ keeps the first latent frame at 4 steps; 1/2-step modes apply to later frames, and the A800 timing excludes VAE |
| Causal-rCM | <https://github.com/NVlabs/rcm>; recipe/code public, checkpoint completeness stated conservatively | Clean-context 4/2/1 steps require 5/3/2 NFE because context encoding adds a forward |
| SCD | CVF paper/supplement/project page public; no verifiable official code/checkpoint located | 832×480, batch 1, one H100; 11.1 FPS/0.29 s is encoder-once plus multi-step renderer, not few-step distillation |
| FlowCache | <https://github.com/mikeallen39/FlowCache>; MAGI-1 and SkyReels-V2 code public | 2.38× and 6.7× are A800 offline speedups from very long baselines, not a real-time FPS/TTFF result |
| MotionStream | <https://github.com/alex4727/motionstream>; repository explicitly says code remains under internal review, no runnable weights | 16.7 FPS uses the full VAE; 29.5 FPS depends on a separately trained Tiny VAE |
| StreamDiffusionV2 | <https://github.com/chenfengxu714/StreamDiffusionV2>; inference/PyPI/checkpoint public; training and parts of scheduler remain TODO | 4×H100 pipeline figures are aggregate system evidence, not one-GPU generator latency |

Formal acceptance, an open repository, downloadable weights and independent reproduction were treated as four separate evidence fields. A repository placeholder, branch with only inference, or README timing table was not upgraded to a reproduced result.

## Candidate works intentionally not promoted to milestones

The search also surfaced FramePack, SkyReels-V2, MAGI-1, LongVie 2 and many 2026 cache/forcing variants. They remain important context, but the chapter's milestone table uses a stricter test: a milestone must change the problem definition, causal factorization, training distribution, memory abstraction or deployment protocol. A higher resolution, longer demo or another variant of an existing sparse-cache recipe is not by itself a milestone.

Primary records checked for this contextual group:

- FramePack: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html>; fixed context does not make it real-time, and anti-drift/inverted ordering is not always strictly causal
- SkyReels-V2: <https://arxiv.org/abs/2504.13074>
- MAGI-1: <https://arxiv.org/abs/2505.13211>; reported streaming uses a 24-GPU pipeline and separates/overlaps the VAE, so it is not one-GPU latency evidence
- LongVie 2: <https://arxiv.org/abs/2512.13604>

## Claim-handling rules

1. Performance numbers are introduced as **author-reported**, not independently replicated.
2. FPS is not compared across different GPU counts, resolutions, precisions, model sizes or denoising steps.
3. A long-duration demo is not evidence of real-time execution.
4. Dynamic prompting is not evidence of action-conditioned world modeling.
5. Causal attention is an information constraint, not proof of causal physical reasoning.
6. New August 2026 preprints are marked provisional and are not used to rewrite older peer-reviewed conclusions.
7. Causal codec, causal generator, streaming commit and real-time SLO are separate contracts; success at one layer is not inherited by the next.
8. Sampling-step labels are not treated as NFE until clean-context encoding, CFG branches and recomputation are counted.
9. Bounded resident KV/GPU memory is not described as lossless long-term memory or constant total system cost; CPU, external index and retrieval latency remain separate.
10. Author priority claims such as "first" are omitted unless an exhaustive search could support them; the chapter instead states the concrete capability placed at the center of the paper.

## Proposed falsification package

The chapter now defines `StreamFork-1`, but the package is explicitly marked **proposal, not run**. Its four required probes are:

1. equal prefix/seed with perturbed hidden suffix or future condition, checked by per-commit hashes outside the declared revision window;
2. matched GT/noised-GT/self-rollout histories at equal backbone, teacher, cache and measured NFE, evaluated at 1×/2×/6×/12× training horizon;
3. 5 s/1 min/4 min resident-memory and long-recall traces with object-return, scene-cut, small-object and cache-poisoning probes;
4. end-to-end cold/warm and single/concurrent timing including text encoder, DiT, codec/VAE, queue, transport and display, with deadline recovery and at least a 60 s soak.

The completion surface is a frozen manifest, commands/environment lock, raw `trace.jsonl`, `commits.csv` and hashes, NFE hook log, all videos/failures and survival/memory/latency plots. A selected demo or copied author table does not complete the experiment.

## Figure generation and verification

The chapter overview figure was generated with the built-in image-generation tool after the scientific-schematics script could not run because `OPENROUTER_API_KEY` was not configured.

- Workspace asset: `assets/diagrams/causal-streaming-video-generation.png`
- Dimensions: 1672 × 941
- SHA-256: `41a0f7564b40803164b92af3938c8a933f16d89e16fe5676d724d046db9bc079`
- Visual checks: all 11 requested labels rendered correctly; offline full-clip/bidirectional arrows are distinct from the causal chunk loop; the rolling memory is visibly bounded; no logo, watermark, extra text or malformed typography was observed.

Exact mechanism relationships were added as editable Mermaid rather than another generated raster:

1. five-layer causal/streaming stack;
2. next-unit versus full-sequence versus rolling per-token noise and commit frontier;
3. bounded generator-memory write/age/compress/retrieve/drop/read lifecycle;
4. speculative generation → decode/gate → immutable commit/hash → display, with backpressure and reset branches.

All four contain `accTitle` and `accDescr`. Mermaid CLI 11.16.0 with system Chrome rendered them to non-empty SVG and PNG; the inspected PNG sizes were 936×726, 671×1932, 1584×828 and 1324×1189. Individual color renders and a grayscale contact sheet preserved every branch label, terminal state and arrow meaning; the diagrams do not rely on color alone. The rolling-noise chart deliberately uses a tall reading order so the A/B/C schedules remain legible instead of shrinking three mechanism panels into one wide strip.

## Documentation validation

The final causal/streaming integration batch produced the following checks:

- Markdown: 24 changed/new non-README files, 0 issues with markdownlint-cli2 0.23.2 / markdownlint 0.41.1; README retains exactly two pre-existing warnings at its historical heading/blockquote locations.
- Reference closure: causal chapter 28 references / 71 citation occurrences; no missing, orphan, duplicate or gapped reference anchors. The same check passed for every changed ref-based chapter.
- Local links/images: 417 relative targets across 25 changed/new Markdown files; 0 missing.
- Mermaid: 39 blocks across the changed/new Markdown set; all 39 contain accessibility metadata and render to non-empty SVG.
- Causal visuals: all four Mermaid charts inspected individually and in color/grayscale contact sheets; no clipped labels, broken terminals or color-only distinction.
- Timeline preservation: 75 HTML images retained and all 75 have non-empty `alt` text.
- External URLs in the causal chapter: 38 unique links. The first pass returned HTTP 200 for 37 and a policy 403 only for an OpenReview PDF endpoint; that link was replaced by its official forum page, which returned 200. A later rapid GitHub-only retry produced transport code `000` after those same repository pages had already returned 200 and had been read through the browser layer; no deterministic 404 remained.
- Patch hygiene: `git diff --check` returned no error.

These checks validate documentation structure and source accessibility, not model behavior. `StreamFork-1` remains a proposed experiment; no checkpoint, runtime trace or independent speed/long-horizon reproduction was created in this batch.

## Review limitations

- This is a focused scoping review, not a formal meta-analysis; the field is too new and the evaluation setups are not homogeneous.
- Some 2026 works had only arXiv/technical-report versions at the review date; formal status was upgraded only for works with an official venue page or accepted-paper list.
- OpenAlex and Crossref counts were too broad to serve as screening statistics.
- Semantic Scholar rate limiting prevented its use as the third metadata cross-check; DBLP, Crossref and official proceedings were used instead.
- Independent reproduction of speed, commit correctness and long-horizon quality claims was outside this repository review; the chapter therefore supplies `StreamFork-1` and a fair-reporting protocol rather than a ranking.
