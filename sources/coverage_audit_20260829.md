# Repository coverage, depth and visual audit — 2026-08-30

This is the working gap matrix for the long-term goal: every subfield should have an understandable visual summary, a current technical route, defensible milestones, detailed paper review and explicit evidence limits.

## Scoring and priority

- **Depth 0:** absent.
- **Depth 1:** outline or paper-name list.
- **Depth 2:** coherent introductory tutorial.
- **Depth 3:** review chapter with mechanisms, comparisons, evaluation and failure analysis.
- **Depth 4:** review chapter plus reproducible evidence trail, current primary sources, quantitative protocols and verified visuals.
- **Reproduction is a separate axis:** Depth measures the chapter/review artifact, not whether a large checkpoint was executed. Proposed-only experiments are labelled in each row and research log; actual smoke/matched/independent runs must never be inferred from Depth 4. Navigation pages and bibliography infrastructure use `N/A` when the subfield checklist is not applicable.
- **P0:** factual/classification error, unsupported claim, structural absence or field coverage clearly stale by 2026-08-29.
- **P1:** route is sound but misses mechanisms, milestones, quantitative evidence or a major recent branch.
- **P2:** presentation, visual coverage, metadata normalization or secondary breadth.

Counts are a repository snapshot after adding the causal-streaming, video post-training, native audio-video, fine-grained control, video degradation restoration, video-tokenizer, multi-view/4D and Video DiT/backbone chapters, with the mechanism, foundation, task-map, application, getting-started, interpolation, inpainting, text-to-video, image-to-video, video-to-video, video-prediction, action-conditioned prediction, interactive-world, digital-human, story/multi-shot, unconditional-video, open-model, video-reasoning, JEPA, diffusion/flow, physics, timeline, reading-route and bibliography rows refreshed on 2026-08-30. “Visual” counts Mermaid blocks plus embedded images; a visual can still fail the scientific-quality gate. The former mixed VAE/tokenizer page is now split: `variational-generation.md` owns ELBO, learned prior, posterior collapse and multi-future evidence, while `video-tokenizers.md` owns representation, quantization, causal/chunk boundaries, exact budgets, bitstream maturity and fixed-consumer replacement tests. `video-dit-backbones.md` separately owns post-codec patch/token geometry, mixer/mask, position/fusion, active capacity and execution scaling; it does not absorb objective derivations or streaming SLO.

## A. Generation mechanisms and foundation models

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `docs/generative-models.md` | 464 | 32 | 4 | 4 | Five-axis cross-classification now routes representation, objective, backbone and deployment to disjoint specialist contracts; cascade/cache no longer leak into the backbone taxonomy; next gate is controlled implementation comparison | P2 |
| `recurrent-prediction.md` | 368 | 23 | 3 | 4 | Stochastic recurrent latents now route ELBO/learned-prior semantics to variational generation and compression-only representations to the tokenizer contract; next gate is matched open-loop/closed-loop reproduction | P2 |
| `variational-generation.md` | 277 | 7 | 2 | 3.5 | ELBO/learned-prior/multi-future core is clear, but the milestone table stops at 2018 and `LatentFork-1` lacks a named dataset/checkpoint/threshold; refresh modern stochastic-latent interfaces before calling it current-complete | P1 |
| `video-tokenizers.md` | 416 | 25 | 3 | 4 | New representation/codec review spanning VQ/LFQ/FSQ/BSQ, exact finite-clip and bitstream ledgers, causal prefix/chunk contracts, adaptive/structured/aligned frontiers, BS0–BS3 and TokenizerFork-1; next gate is matched open-checkpoint and independent bitstream reproduction | P2 |
| `video-dit-backbones.md` | 576 | 35 | 3 | 4 | New primary-source review from U-Net/DiT ancestry through full/factorized/window/sparse/linear/hybrid attention, position/fusion, noise-time experts, distributed execution, cache and quantization; includes a verified original compute-contract visual plus accessible block/falsification diagrams and proposed BackboneFork-1/ServeFork-1; no model run was executed | P2 |
| `adversarial-generation.md` | 358 | 32 | 3 | 4 | Rewritten across full video GANs, tokenizer/decoder critics and explicit adversarial video distillation, including negative classifications; next gate is fixed-teacher/NFE/critic reproduction | P2 |
| `autoregressive-generation.md` | 647 | 20 | 2 | 4 | Rewritten across strict token, frame, set and chunk factorization, continuous heads, cache complexity and forcing; next gate is a matched decoder/cache implementation study | P2 |
| `masked-generation.md` | 399 | 18 | 3 | 4 | Rewritten across absorbing-state/discrete diffusion links, confidence calibration and token/frame/tube/block schedules; next gate is controlled schedule/calibration reproduction | P2 |
| `diffusion-models.md` | 409 | 24 | 2 | 4 | DDPM→score→reverse SDE→PF-ODE remains chapter-owned; duplicated tokenizer taxonomy is reduced to an integration boundary, and rate claims now require bitstream evidence; next gate is numerical notebook reproduction | P2 |
| `flow-consistency-models.md` | 636 | 20 | 4 | 4 | Rewritten across FM/RF/reflow, CM/sCM/rCM, Shortcut/MeanFlow/FACM/AlphaFlow and DMD/DMD2; a verified five-layer map now separates training objectives, continuous processes, training-free solvers, learned few-step routes and the orthogonal streaming axis; next gate is matched-budget reproduction | P2 |
| `causal-streaming-generation.md` | 658 | 28 | 5 | 4 | Four-layer codec→generator→commit→SLO contract, four clocks, rolling-noise/commit visual, bounded-memory lifecycle, formal-status/artifact audit and proposed StreamFork-1 are explicit; global dense cost and data-time KV cache are now separated from other mixers/inter-step reuse; no checkpoint/runtime run was executed | P2 |
| `video-post-training-alignment.md` | 385 | 38 | 2 | 4 | New focused review separating CPT/SFT, RM, pair construction, DPO/RWR, online feedback, policy-gradient RL, test-time guidance and reward-guided distillation, with credit, cost, hacking and evaluator-independence gates; next gate is matched-checkpoint reproduction | P2 |
| `foundation-models.md` | 360 | 42 | 3 | 4 | System view now separates tokenizer, objective, backbone/execution and deployment manifests; detailed acceptance adds parameter/FLOP-matched architecture and fixed-checkpoint serving rows while retaining 4D/streaming/bitstream contracts | P2 |

### Mechanism-level structural correction

The repository should stop treating these as one mutually exclusive taxonomy:

1. **representation:** pixel, continuous latent, discrete token, hybrid;
2. **factorization:** full-sequence, recurrent, autoregressive, masked, chunk/rolling;
3. **training objective:** likelihood/ELBO, adversarial, diffusion/score, flow matching, consistency/DMD, preference/RL;
4. **backbone:** CNN/RNN, U-Net, Transformer/DiT or recurrent/SSM; then mixer/mask, position/fusion and FFN/MoE routing inside that implementation;
5. **deployment:** offline, causal streaming, real-time interactive.

Modern systems routinely choose one item from several axes. Diffusion Forcing, CausVid, Self Forcing, MarDini and MaskFlow make this especially visible.

## B. Task chapters

| File | Lines | Refs | Visuals | Current depth | Main gap or correction | Priority |
|---|---:|---:|---:|---:|---|---|
| `text-to-video.md` | 455 | 36 | 4 | 4 | Rewritten around a strict task boundary, orthogonal route matrix, runtime condition contract, optional post-training, native AV, release surfaces and acceptance evidence; next gate is matched open-checkpoint reproduction | P2 |
| `image-to-video.md` | 465 | 34 | 3 | 4 | Rewritten around pixel/latent/soft temporal anchors, full condition tensors, motion/camera/audio routes, preservation–motion trade-offs and versioned evidence; next gate is matched checkpoint/control reproduction | P2 |
| `video-to-video.md` | 442 | 47 | 3 | 4 | Rewritten around a strict source-time-axis contract, route-specific preservation assumptions, formal/release status, eight-axis evaluation, multi-turn non-destructive editing and causal streaming; full-frame degradation restoration and mask completion are now separate adjacent contracts; next gate is matched-checkpoint editing reproduction | P2 |
| `digital-human.md` | 366 | 45 | 3 | 4 | Rewritten around seven non-interchangeable tasks, time/sync/identity/authorization contracts, release surfaces, loss conflicts, leakage, counterfactuals and 2026 long-duration/streaming/full-body work; old unlabelled-score PNG is no longer cited and a score-free contract visual was verified; next gate is matched open-checkpoint reproduction | P2 |
| `frame-interpolation.md` | 480 | 32 | 4 | 4 | Rewritten as correspondence/reconstruction versus generative routes, with hard endpoint constraints, milestone-date rules, protocol traps, failure analysis and a reproducible evidence log; next gate is matched-data/runtime reproduction | P2 |
| `video-prediction.md` | 466 | 30 | 3 | 4 | Rewritten around train-only versus deployment-GT boundaries, deterministic/stochastic/latent/diffusion routes, forcing distinctions, calibrated rollout and an evidence ladder; next gate is matched long-rollout and downstream-utility reproduction | P2 |
| `action-conditioned-prediction.md` | 447 | 33 | 3 | 4 | Rewritten around action clocks, intervention versus logged correlation, pixel/probabilistic/task/representation/generative routes, model exploitation, calibrated futures and real closed-loop utility; next gate is the preregistered ActionFork-2D reproduction | P2 |
| `video-restoration.md` | 361 | 30 | 3 | 4 | New inverse-problem review separating blur/downsample/noise/compression recovery from masked completion, spanning alignment, propagation, Transformer, generative prior and causal/one-step routes through formal 2026 work; includes degradation manifests, four-axis evaluation, hallucination audit and RestorationFork-1; next gate is matched open-checkpoint reproduction | P2 |
| `video-inpainting.md` | 546 | 38 | 3 | 4 | Rewritten from valid-pixel propagation through missing-region synthesis and global temporal verification, with mask-outside preservation, scene-cut reset, benchmark traps and a reproducible evidence log; now explicitly excludes full-frame degradation restoration; next gate is controlled mask/protocol reproduction | P2 |
| `story-multishot.md` | 391 | 34 | 3 | 4 | Rewritten around shot contracts, planner/joint/memory/storyboard/streaming routes, state transactions, conflict rollback, current formal venues and release surfaces; SVD misclassification removed; next gate is matched long-story reproduction | P2 |
| `interactive-world-generation.md` | 463 | 39 | 3 | 4 | Rewritten around action schedules, dual clocks, speculative versus authoritative memory, formal 2026 frontiers, model/product/artifact release surfaces, external interlock and post-hoc audit; next gate is common-hardware closed-loop reproduction | P2 |
| `unconditional-video-generation.md` | 456 | 35 | 3 | 4 | Rewritten around strict deployment-time $p(X)$ boundaries, GAN/token/diffusion/flow routes, fidelity–coverage–tail–memorization evidence, protocol traps and a reproducible experiment; Sora/Cosmos moved to adjacent conditional systems; next gate is matched-data reproduction | P2 |
| `native-audio-video-generation.md` | 479 | 33 | 2 | 4 | New task-contract review separating V2A/A2V, shared-condition, staged, inference-coupled and native joint generation, then auditing codecs, clocks, dual-stream/single-stream fusion, streaming memory, search and five independent acceptance gates; next gate is a fixed-output open-checkpoint comparison | P2 |
| `controllable-video-generation.md` | 517 | 45 | 2 | 4 | New task-contract review organizing camera, trajectory, pose/structure and identity controls by coordinate/time contract, injection point, conflict handling, release surface and falsifiable evaluation; next gate is a matched-backbone controller reproduction | P2 |
| `multiview-4d-generation.md` | 538 | 32 | 3 | 4 | New camera×world-time review separating video paths, multi-view grids and renderable states, with reconstruction/generative/streaming routes, 2021–2026 milestones, six evidence gates and proposed GridFork-1; no model run was executed | P2 |

## C. Reasoning, world models, physics and evaluation

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `video-reasoning.md` | 811 | 58 | 3 | 4 | Added UniVR, the causal-generative gap and RuleMaze as a scoped adjacent control; normalized MME-CoF/VChain/Thinking-with-Video/VIPER venues, separated output, denoising and interaction clocks, and renamed its local R1–R4 scale to avoid collision with the canonical L0–L7 evidence ladder; next gate is causal-state and closed-loop reproduction | P2 |
| `world-models.md` | 539 | 33 | 3 | 4 | Rewritten around cascaded versus joint world-action routes, receding-horizon execution, pseudo-action recovery, verifier scope, persistent memory, uncertainty and decision utility; its local WM0–WM6 reporting stack now maps explicitly to the canonical global L0–L7 ladder, and both Mermaid diagrams carry accessible metadata; next gate is independent closed-loop reproduction | P2 |
| `physical-consistency.md` | 594 | 41 | 5 | 4 | Rewritten with a six-level concept taxonomy, one canonical L0–L7 evidence ladder, 2024–26 method/benchmark registries and an explicit condition→state→rollout→measurement→falsification contract; training signals are separated from sealed evaluators, and the next gate is independent GAUGE/robotics replication | P2 |
| `evaluation.md` | 887 | 62 | 3 | 4 | Rewritten with 2025–26 task families, evaluator stress tests, pairwise statistics, SLO/energy, watermark and C2PA 2.4; restoration, streaming, 4D and generation-backbone claims now have distinct acceptance contracts and execution manifests | P2 |
| `jepa.md` | 340 | 19 | 4 | 4 | Rewritten around the exact teacher–student contract, collapse, dense/semantic trade-off, action-conditioned latent MPC, V2.1 attribution, multi-future uncertainty, release surfaces and claim-specific tests; next gate is matched latent-control reproduction | P2 |

## D. Navigation, applications and resources

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `getting-started.md` | 278 | link-only | 4 embedded images | N/A | Engineering and research routes now include a backbone/token-attention-cost checkpoint before system claims; navigation pages are not scored as subfield reviews | P2 |
| `taxonomy.md` | 295 | 12 | 3 | 4 | Condition source × source-content relation × interaction-horizon now includes a separate multi-view/4D row and camera-path→state boundary alongside restoration/completion | P2 |
| `applications.md` | 331 | 12 | 3 | 4 | Capability→workflow→acceptance now adds virtual production/AR/VR dynamic assets and separates camera path, multi-view grid and renderable 4D state | P2 |
| `timeline.md` | 904 | 0 local anchors | 2 Mermaid + 75 images | 4 | Representation/objective/backbone are now parallel labels; a first-public/formal backbone sub-lineage reaches hybrid/sparse/reuse work through formal 2026 while preserving preprint/artifact boundaries | P2 |
| `reading-list.md` | 552 | current external links | 2 | 4 | Tokenizer and Video DiT/backbone are shared trunk/deepening units, not extra task branches; the seven task branches retain one common falsifiable manifest | P2 |
| `resources/datasets.md` | 795 | 38 | 2 | 4 | Rewritten as a current scoping review with release-surface taxonomy, an exact ten-stage data-engine path plus governance loop, rights/provenance, action/physics branches, manifest and fixed-compute validation; release-unit claims passed independent audit, and the next gate is acquisition/decode-yield replication | P2 |
| `resources/open-models.md` | 451 | 35 | 4 | 4 | Dated release-surface audit now also prevents DiT names from implying topology/routing; Hunyuan SSTA, Wan2.2 total/active and SANA2 checkpoint boundaries are explicit; next gate is clean-environment reproduction | P2 |
| `bibliography.md` / registry | 171 / registry | 117 registered core items | 0 | N/A | Cross-chapter DiT/WALT/Latte/CogVideoX/SANA/LinVideo/TimeRipple anchors now have a dedicated registry section; metadata, Stars, BibTeX and generated index remain identity-consistent | P2 |

## P0 factual and evidence corrections

These should be fixed before broad stylistic expansion:

1. ~~Reclassify IFRNet as an efficient convolutional encoder–decoder/feature-refinement method, not a Transformer.~~ Corrected on 2026-08-30 in the interpolation chapter.
2. ~~Stop using general Video GAN, MoCoGAN, SVG, SVD or Video Diffusion Models as direct task-specific milestones without an explicit “technical ancestor” label.~~ Corrected on 2026-08-30 across the refreshed task chapters; direct milestones and technical ancestors are now distinguished.
3. ~~Remove or replace SVD as evidence for an LLM storyboard/multi-shot pipeline.~~ Corrected on 2026-08-30; the story chapter now separates SVD as an unrelated ancestor and audits current multi-shot routes.
4. ~~Remove Sora and Cosmos from the representative unconditional-generation route.~~ Corrected on 2026-08-30; both are now adjacent conditional systems, not direct milestones.
5. ~~Standardize dates as “first preprint year / formal publication year”; FramePack, Hallo, SadTalker, ShotAdapter and LDF-VFI require the same rule.~~ Corrected on 2026-08-30; LDF-VFI is recorded as 2026-01 first preprint → CVPR 2026 formal publication, and the timeline now applies the distinction systematically.
6. ~~Mark the digital-human overview's `Identity 0.92` and `Temporal Consistency 0.90` as illustrative or regenerate it without fabricated-looking scores.~~ Corrected on 2026-08-30: the legacy file is retained but uncited; the chapter uses a score-free generated contract visual.
7. ~~Update C2PA from 2.2 to the official 2.4 specification and preserve the version date.~~ Corrected on 2026-08-30 in the evaluation and application evidence boundaries.
8. ~~Split model technical significance from current product availability; a discontinued product does not erase its research milestone, but should not be presented as currently available.~~ Corrected on 2026-08-30 with independent timeline evidence axes and dated availability states.
9. ~~Stop routing full-frame super-resolution, deblurring, denoising and decompression to a mask-completion chapter, and stop reusing local L-level names for incompatible evidence scales.~~ Corrected on 2026-08-30 with a dedicated degradation-restoration contract, separate inpainting boundary, canonical global L0–L7 ladder and explicitly mapped WM/R local scales.
10. ~~Complete the tokenizer/VAE ownership split: move representation, quantization, causal-codec and bitstream accounting out of `variational-generation.md`, keep ELBO/learned-prior/multi-future semantics there, then recount both chapters and re-run link/reference/depth checks.~~ Corrected on 2026-08-30: the two independently readable pages now own disjoint task contracts, both have closed references and rendered accessible diagrams, and shared pages reserve bpp/bitrate for entropy-coded bitstreams.
11. ~~Stop conflating review depth with checkpoint reproduction, omit navigation pages from the inventory, and leave multi-view/4D as an unnamed control sub-branch.~~ Corrected on 2026-08-30: Depth now measures the review artifact, run status remains explicit, `getting-started.md`/bibliography are `N/A`, and camera×time 4D has its own chapter and evidence contract. The variational row was conservatively lowered to 3.5 pending a current frontier refresh.

## Visual implementation queue

Each visual must have an explicit learning objective, editable/accessible alternative and a post-generation visual check.

| Priority | Visual | Learning objective |
|---|---|---|
| Done | Orthogonal mechanism cube/map | Show representation, factorization, objective, backbone and deployment as composable axes |
| Done | Video token generation routes | Separate continuous latent, discrete AR and masked/discrete routes while keeping representation, factorization, objective and deployment claims orthogonal |
| Done | Tokenizer four-ledger finite-clip contract | Derive exact grid/element counts for a 9-frame example, then separate nominal token capacity from probability-model/entropy-coded bitstream |
| Done | Causal tokenizer first-frame/chunk contract | Show first-frame anchoring, past-only receptive fields, cache carry/reset, optional overlap/crop, committed frames and the non-implication chain to real-time SLO |
| Done | Per-token noise to rolling commit frontier | Separate next-unit, full-sequence and rolling schedules; show that noise levels do not imply causal access, self-history, few-step or SLO |
| Done | Bounded generator-memory lifecycle | Show write, aging, anchor/persistent/compressed/state/retrieval/drop routes; separate GPU working set from growing external storage and information loss |
| Done | Streaming commit/backpressure/recovery state machine | Trace speculative output through decode/gate/hash/immutable commit/display, with declared degradation, backpressure, cache reset and future-leak probes |
| Done | Modern foundation system stack | Track governance→representation→generator→post-train→decode→deployment and separate full GAN, codec critic and adversarial distillation roles |
| Done | DDPM–SDE–PF-ODE–FM/RF–CM/DMD map | Prevent the most common diffusion/flow/consistency terminology errors |
| Done | Offline bidirectional vs causal streaming | Separate full-clip denoising from chunk-causal output, bounded memory and new-condition loop |
| Done | Data/provenance pipeline | Source→shot split→dedup→filter→caption→license/provenance→versioned split; generated figure plus accessible Mermaid alternative verified |
| Done | Physics evidence loop | Condition→state→generator/simulator→measurement→constraint/reward→falsification |
| Done | WAM dual route | Compare cascaded planner+generator with joint action-video model and verifier |
| Done | Three-axis task map | Condition source, source-preservation strength and open-loop→closed-loop |
| Done | T2V prompt-to-evidence contract | Separate pure text, source-pixel hybrid and closed-loop tasks; connect runtime conditions, optional post-training, generation and acceptance gates |
| Done | I2V anchor-and-control contract | Separate pixel/latent/soft anchors and trace image, motion, camera and audio conditions into the denoiser and preservation/motion gates |
| Done | Video-prediction deployment contract | Separate train-only future supervision from deployment rollout and connect claim strength to calibration, intervention and closed-loop evidence |
| Done | Interactive-world closed-loop stack | Separate causal action flow, memory read/update, external pre-execution interlock and post-hoc evidence audit |
| Done | VFI dual branch | Flow/depth/occlusion deterministic branch vs diffusion/DiT generative branch |
| Done | Video inpainting pipeline | Valid-pixel propagation→missing-region synthesis→global consistency→outside-mask protection |
| Done | Video degradation restoration contract | Degradation observation→alignment/propagation/generative routes→fidelity/temporal/detail gates, with inpainting as a separate missing-support branch |
| Done | Video reasoning three clocks | Output-time, denoising-time and interaction-time reasoning with intervention points |
| Done | Evaluation evidence ladder | L0 visual plausibility through L7 real-world closed-loop utility |
| Done | Multi-shot memory workflow | Script/character bible→shot plan→references→generation→memory update→conflict/rollback |
| Done | Unconditional-generation evidence chain | No external condition→marginal model→sampling→fixed output contract→quality/coverage/tail/memorization evidence |
| Done | Digital-human condition and sync contract | Authorized inputs→timeline alignment→condition fusion→generation→separate sync/identity/motion/media gates |
| Done | Open-model release surface | Official identity→version→artifact split→license intersection→hardware→smoke test→reproducibility manifest |
| Done | V2V method selector | Local/global, appearance/motion, new-view and multi-turn requirements to method families |
| Done | Action-conditioned intervention-to-control loop | Observed history→action schedule→counterfactual rollouts→fresh-observation planning loop |
| Done | JEPA MPC loop | Encoder→action-conditioned predictor→cost→planner→action→new observation |
| Done | Video post-training parallel-route evidence map | Separate data/SFT, pairwise preference, reward/RL, test-time optimization and distillation without implying a compulsory pipeline |
| Done | Native audio-video generation contract | Separate V2A, A2V, shared-condition, staged and learned-joint factorization before evaluating synchronization |
| Done | Fine-grained control contract | Trace camera, trajectory, structure and identity signals through coordinates/conflicts, injection and four independent acceptance gates |
| Done | Camera-view × world-time 4D contract | Distinguish a video row, multi-view image column, camera-controlled diagonal and full view-time grid, then separate the grid from a renderable dynamic state |
| Done | Video DiT compute contract | Derive token growth, distinguish full/factorized/window-sparse/linear/hybrid topology, separate token reduction, noise-time experts and parallel/cache, then require output/FLOPs/latency/VRAM/communication/quality gates |
| Done | Video DiT block ownership map | Route codec latent→patch/position→mixer/fusion→FFN/experts→objective head while keeping tokenizer, backbone and sampler chapters disjoint; accessible Mermaid and sequential text supplied |
| Done | BackboneFork-1 evidence map | Freeze codec/data/objective/sampler/output, run parameter- and FLOP-matched forks, then retain a claim only after quality, long-range, grid and cost falsifiers pass |

## Primary-source refresh queues

Before editing each field, verify its queue at the primary paper or official proceedings; do not copy this audit's shorthand into final prose without a fresh read.

- **Representation/tokenizers:** MAGVIT-v2, CV-VAE, OmniTokenizer, VidTok, Causal VAE, BSQ-ViT, ElasticTok, LARP, CoordTok, VidTwin, Divot, InfoTok, AdapTok, NeRV-Diffusion, VideoRAE, V-RAE and KVAE; always freeze exact shape/dtype/token accounting, mark formal/preprint status and distinguish real entropy-coded bitstreams from nominal rates.
- **Mechanisms/foundation:** Diffusion Forcing, MarDini, MaskFlow, LTX-Video, VidTok, Pyramid Flow, T2V-Turbo, CausVid, Self/Causal Forcing, Rolling Forcing, LongLive, SCD, FlowCache, MotionStream, StreamDiffusionV2, Causal-rCM, Step-Video, Open-Sora 2, SkyReels-V2, MAGI-1, Wan2.2, HunyuanVideo 1.5, Ovi, LTX-2; VideoDPO, VideoAlign, DynamicsBoost, Dual-IPO, BranchGRPO and noisy-latent rewards. For causal/streaming work, separately verify history exposure, measured NFE, commit semantics, resident/external memory and end-to-end SLO.
- **Backbone/scaling (structural chapter completed 2026-08-30):** factorized/full/window/sparse/linear/hybrid attention, MMDiT/Expert AdaLN, 3D RoPE, noise-time MoE, context parallelism, inter-step reuse, quantization and current formal/artifact status are now separated in `video-dit-backbones.md`; next gate is to execute `BackboneFork-1` on matched small models and `ServeFork-1` on one fixed open checkpoint, without multiplying independent paper speedups.
- **Multi-view/4D:** D-NeRF, Nerfies, MAV3D, 4D-GS, 4Real, 4DiM, SV4D/2.0, GenXD, CAT4D, 4Real-Video, 4C4D, DGGT, MoRel, 4DSurf, Full-4D, MV-Forcing, Stream4D and 4DStreamCtrl; always separate camera path, view-time grid, renderable state, unseen-region generation, scene-build cost and query/render cost.
- **Physics/evaluation:** NewtonRewards, Physion-Eval, Physics-IQ Verified, Apple-π, GAUGE, Interpreting Physics in Video World Models, VBench 2.0, WorldReasonBench, WBench, MIND, EntityBench, HuM-Eval/HuM-Bench.
- **World/action:** WAM survey, World Action Verifier, DreamGen, ViPRA, WorldPack, Infinite-World, ReWorld.
- **Tasks:** IFRNet, EDEN, BiM-VFI, LDF-VFI; MCVD/FramePack; STTN/FuseFormer/E²FGVI/ProPainter; STIV/ReasonDiff; ShotAdapter/OneStory/MultiShotMaster; VACE/EasyV2V/Editto; Exploration-Driven GIE/Vid2World/Dexterous World Models; MM-Diffusion/Ovi/LTX-2/NAVA/Ripple; MotionCtrl/Motion Prompting/GEN3C/LAMP/BulletTime/FlashMotion/4DStreamCtrl.
- **Resources:** InternVid, OpenVid-1M, Physion/CLEVRER/PHYRE, NewtonBench-60K, VBVR, AgiBot World, RoboMIND/RH20T, modern audio-video and multi-view/3D sets.

## Batch verification — action/V2V/reasoning/JEPA refresh

Final checks on 2026-08-30 covered the four chapters, four dated research records and this audit:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 9 files, 0 issues |
| Reference closure | action 33 refs / 67 citations; V2V 47 / 119; reasoning 58 / 155; JEPA 19 / 60; no missing or orphan anchor |
| Local links and images | 31 chapter-local targets resolved; 0 missing |
| Mermaid | 8 blocks total; all contain accessibility metadata and all 8 rendered with Mermaid CLI plus system Chrome to non-empty PNG/SVG audit artifacts |
| Mermaid visual check | 2 action, 2 V2V, 1 reasoning and 3 JEPA charts inspected individually/contact-sheet; no clipping, broken branch or unreadable terminal node |
| Generated PNGs | 4 teaching assets inspected at original detail; dimensions, SHA-256, grayscale contrast, prompt and scientific boundary are recorded in the corresponding dated research file |
| Patch hygiene | `git diff --check` returned no error |

Temporary Mermaid renderings and the contact sheet remain outside the repository. This batch does not run model checkpoints, does not constitute R1/R2 reproduction, and does not commit or push.

## Batch verification — transport, physics, timeline, reading and bibliography refresh

Final checks on 2026-08-30 covered the six changed documentation pages, five dated audit/research records, generated assets and bibliography infrastructure:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 11 files, 0 issues |
| Reference closure | diffusion 24 / 24; flow 20 / 20; physics 41 / 41; no missing, orphan or mismatched anchor |
| Local links and images | All relative targets in the changed Markdown set resolved; 0 missing |
| Mermaid | 12 blocks across diffusion, flow, physics, reading list and timeline; all include `accTitle` / `accDescr`, and all 12 rendered with Mermaid CLI plus system Chrome to non-empty SVG audit artifacts |
| Timeline media preservation | 75 existing HTML images retained; all have non-empty alt text, and the referenced image set is unchanged from `HEAD` |
| Generated PNGs | Two 1672×941 teaching assets inspected at original size and in grayscale; SHA-256, final prompts, rejected variants and scientific boundaries are recorded in the dated visual research file |
| Bibliography integrity | Registry, metadata and BibTeX each contain the same 109 unique identities; JSON parse, Python compile, updater `--check` and `--dry-run` all passed |
| Patch hygiene | `git diff --check` returned no error |

This batch verifies documentation structure and generated artifacts, not model checkpoints. GAUGE, few-step video and closed-loop claims remain author-reported until an independent reproduction is added.

## Batch verification — post-training, native AV and fine-grained control

Final checks on 2026-08-30 covered the three new chapters, four dated research/audit records, three generated assets and the shared navigation updates:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 15 changed/new docs and logs, 0 issues; README retains two pre-existing warnings outside the changed lines |
| Reference closure | applications 12 refs / 23 citations; foundation 42 / 63; mechanism overview 32 / 48; taxonomy 12 / 22; post-training 38 / 71; control 45 / 98; native AV 33 / 75; no missing, orphan, duplicate or numbering gap |
| Local links and images | 279 relative targets checked across 16 changed/new Markdown files, 139 unique resolved paths, 0 missing |
| External evidence links | Chapter-level checks covered 40 post-training, 34 native-AV and 48 control URLs; site-policy/TLS/timeout cases were separated from deterministic 404 and re-read through the browser layer |
| Mermaid | 16 blocks across 10 changed Markdown files; all contain `accTitle` / `accDescr`, and all 16 rendered with Mermaid CLI 11.12.0 plus system Chrome to non-empty SVG audit artifacts |
| Mermaid visual check | The revised shared falsification loop and all three new chapter diagrams were rendered to PNG and inspected; the control-protocol and conclusion-downgrade nodes remain distinct, with no clipping or broken branch |
| Generated PNGs | Three 1672×941 teaching assets inspected at original detail and in grayscale; final SHA-256, prompts, rejected variants and scientific boundaries are recorded in the corresponding dated research files |
| Patch and secret hygiene | Staged diff whitespace check and changed-file credential-pattern scan passed before commit |

Temporary Mermaid and grayscale audit files remained outside the repository. This batch validates educational structure and evidence boundaries; it does not train or independently reproduce the cited video models.

## Batch verification — degradation restoration and evidence-scale correction

Final checks on 2026-08-30 covered the restoration chapter and research record, task-boundary corrections, WM/R scale normalization, shared navigation, timeline, reading route and generated asset:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 15 changed/new docs and logs, 0 issues; README retains two pre-existing warnings outside the changed lines |
| Reference closure | Restoration 30 refs / 56 citations; no missing, orphan, duplicate or numbering gap |
| External evidence links | All 30 restoration chapter reference URLs returned HTTP 200 during the final audit |
| Local links and images | 306 relative targets checked across 16 changed/new Markdown files; 0 missing |
| Mermaid | 23 blocks across 13 changed documentation pages; all include `accTitle` / `accDescr`, and all 23 rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Mermaid visual check | The six-branch course route, shared falsification loop, R1–R4 reasoning scale, two WM diagrams and both restoration diagrams were inspected individually/contact-sheet; no clipping or broken branch |
| Generated PNG | The 1672×941 opaque restoration contract was inspected at original detail and in grayscale; labels, task split and acceptance gates are legible; SHA-256 is recorded in the dated research log |
| Timeline media preservation | 75 existing HTML images retained; all still have non-empty alt text |
| Patch and secret hygiene | `git diff --check` and changed-file credential-pattern scan passed |

Temporary rendering artifacts remained outside the repository. This batch verifies task classification, sources and educational presentation; it does not train or independently reproduce the cited restoration models.

## Batch verification — variational/tokenizer ownership split

Final checks on 2026-08-30 covered the two independently readable chapters, the tokenizer evidence record, all shared navigation/integration pages and the updated milestone table:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 22 changed/new non-README Markdown files, 0 issues; README retains two pre-existing MD001/MD028 warnings |
| Reference closure | Variational 7 refs / 11 citations; tokenizer 25 / 62; no missing, orphan, duplicate or numbering gap |
| External evidence links | All 32 reference URLs across the two chapters returned HTTP 200 during the final audit |
| Local links and images | 404 relative targets checked across 23 changed/new Markdown files; 0 missing |
| Mermaid | 35 blocks across the changed/new Markdown set; all include `accTitle` / `accDescr`, and all 35 rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Mermaid visual check | The five-axis overview, both variational diagrams, both tokenizer contracts, both reading-route diagrams and the timeline overview were inspected as PNGs; no clipping, broken branch or unreadable terminal node |
| Timeline media preservation | 75 existing HTML images retained; all still have non-empty alt text; VidTok/V-RAE/VideoRAE/KVAE were added as text rows with provisional status |
| Patch and secret hygiene | `git diff --check` and changed-diff credential-pattern scan passed |

Temporary rendering artifacts remained outside the repository. This batch verifies classification, current source status, notation and educational presentation; it does not retrain tokenizers, rerun generation benchmarks or independently implement the cited entropy-coded bitstreams.

## Batch verification — causal, streaming and real-time contract refresh

Final checks on 2026-08-30 covered the causal/streaming chapter, refreshed evidence record, shared mechanism/foundation/evaluation pages, reading route, timeline and navigation:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 24 changed/new non-README Markdown files, 0 issues; README retains exactly two pre-existing MD001/MD028 warnings |
| Reference closure | Causal/streaming 28 refs / 71 citations; every changed ref-based chapter has no missing, orphan, duplicate or numbering gap |
| Formal/artifact status | Diffusion Forcing, CausVid, Self Forcing, Rolling Forcing, LongLive, SCD, FlowCache, MotionStream and StreamDiffusionV2 refreshed from official proceedings/venue pages; code/weights/placeholder surfaces are recorded separately |
| Local links and images | 417 relative targets checked across 25 changed/new Markdown files; 0 missing |
| External evidence links | 38 unique causal-chapter URLs checked; an OpenReview PDF policy 403 was replaced by its official forum page returning 200; later transient GitHub transport `000` retries were separated from earlier successful/browser reads, with no deterministic 404 |
| Mermaid | 39 blocks across the changed/new Markdown set; all contain `accTitle` / `accDescr`, and all 39 rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Causal visual check | Four chapter Mermaid charts rendered to PNG and inspected individually plus color/grayscale contact sheets; schedule, memory lifecycle, commit frontier, backpressure and recovery labels remain legible without color |
| Timeline media preservation | 75 HTML images retained; all still have non-empty alt text; the new causal lineage is text-only |
| Patch and secret hygiene | `git diff --check` returned no error; strong credential-pattern scan over changed/untracked files returned no candidate |

Temporary Mermaid, PNG and grayscale audit files remain outside the repository. The new `StreamFork-1` is a proposal rather than a reported run: the literature-review artifact is depth 4, while checkpoint-level commit/runtime evidence remains absent and must be tracked on the separate reproduction axis.

## Batch verification — multi-view/4D camera-time contract

Final checks on 2026-08-30 covered the new chapter and research record, generated camera×time asset, shared task/foundation/evaluation pages, course route and timeline:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.21.0 / markdownlint 0.40.0: 30 changed/new non-README Markdown files, 0 errors; README retains exactly two pre-existing MD001/MD028 warnings |
| Reference closure | Multi-view/4D 32 refs / 78 citation occurrences / 32 unique cited references; no missing, orphan, duplicate or numbering gap |
| Primary-source metadata | All 32 titles, first authors and formal venue/year fields were checked against official proceedings or arXiv; first-public dates are separated from later proceedings status |
| Local links and images | 465 relative links and 99 image targets checked across 31 changed/new Markdown files; 0 missing |
| External evidence links | All 32 chapter reference URLs returned HTTP 200 after correcting two deterministic 404 targets |
| Mermaid | Both chapter diagrams plus the updated reading-route and timeline diagrams rendered with Mermaid CLI 11.16.0 and system Chrome to non-empty SVG artifacts; accessibility metadata is present |
| Visual check | Generated 1536×1024 camera×time PNG and both chapter diagrams passed original-size and grayscale inspection; the wide route diagram has an exact sequential prose fallback for narrow screens |
| Timeline media preservation | 75 existing HTML images retained; all still have non-empty alt text |
| Patch and secret hygiene | `git diff --check` and the changed-file credential-pattern scan passed |

Temporary render and grayscale audit files remain outside the repository. `GridFork-1` is a proposed protocol: no model checkpoint was downloaded or executed, so geometry, view-time consistency, latency, memory and throughput remain literature-grounded rather than independently reproduced.

## Batch verification — Video DiT backbone and scaling contract

Final checks on 2026-08-30 covered the new backbone chapter and research record, generated compute-contract asset, shared navigation/evaluation pages, reading route, timeline and bibliography infrastructure:

| Check | Result |
|---|---|
| Markdown | markdownlint-cli2 0.23.2 / markdownlint 0.41.1: 15 changed/new non-README Markdown files, 0 issues; README retains exactly two pre-existing MD001/MD028 warnings |
| Reference closure | Video DiT backbone 35 refs / 102 citation occurrences / 35 unique cited references; no missing, orphan, duplicate or numbering gap |
| Local links and images | 378 relative references, 151 unique resolved targets and 90 image references checked across 16 changed/new Markdown files; 0 missing |
| External evidence links | All 36 unique external URLs in the new chapter checked: 34 returned 200; OpenAI and OpenReview returned anti-automation 403 but were browser/index verified; no deterministic 404, 410, 5xx or timeout |
| Mermaid | 26 blocks across the changed Markdown set; all contain accessibility titles/descriptions, and all 26 rendered with Mermaid CLI 11.16.0 plus system Chrome to non-empty SVG artifacts |
| Mermaid visual check | Six affected reading-route, timeline and backbone diagrams were rendered to PNG and inspected as a contact sheet; topology, ownership and falsification branches remain legible with no clipping |
| Generated PNG | The 1672×941 RGB compute-contract asset was inspected at original detail and in grayscale; labels, warning and topology distinctions are legible; SHA-256 is recorded in the dated research log |
| Bibliography integrity | Registry contains 117 core items; JSON parse, Python compile and updater `--check` passed, with registry, snapshots, BibTeX and generated index consistent |
| Timeline media preservation | The original 75 HTML image `src` / `alt` pairs were retained in the same order; every alt is non-empty and every local target exists |
| Patch and secret hygiene | `git diff --check` passed; strong credential-pattern scan over all 21 changed/untracked files returned no candidate; no changed file exceeds 5 MiB |

Temporary Mermaid and grayscale audit files remain outside the repository. `BackboneFork-1`, fixed-weight `ServeFork-1` and converted-checkpoint `ServeFork-1b` are proposed protocols rather than reported runs; no model checkpoint was trained, converted or benchmarked in this batch.

## Definition of done for one subfield

A subfield is not complete because a list of recent papers was appended. It must pass all of these gates:

1. task/mechanism definition and nearest-neighbor concepts are separated;
2. technical route is organized by mechanism, not chronology alone;
3. milestones use a stated criterion and include what remained unsolved;
4. current frontier is verified to the review date and new preprints are labelled provisional;
5. at least one visual materially clarifies the route and passes text/layout/scientific checks;
6. evaluation includes inputs, outputs, metrics, protocol, version, uncertainty and failure cases;
7. strong claims are mapped to the minimum evidence required;
8. primary sources, search date and exclusions are recorded in `sources/`;
9. links, anchors, image paths, Mermaid syntax and repository checks pass;
10. no author-reported performance is presented as independent confirmation.
