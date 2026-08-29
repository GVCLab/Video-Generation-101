# Repository coverage, depth and visual audit — 2026-08-30

This is the working gap matrix for the long-term goal: every subfield should have an understandable visual summary, a current technical route, defensible milestones, detailed paper review and explicit evidence limits.

## Scoring and priority

- **Depth 0:** absent.
- **Depth 1:** outline or paper-name list.
- **Depth 2:** coherent introductory tutorial.
- **Depth 3:** review chapter with mechanisms, comparisons, evaluation and failure analysis.
- **Depth 4:** review chapter plus reproducible evidence trail, current primary sources, quantitative protocols and verified visuals.
- **P0:** factual/classification error, unsupported claim, structural absence or field coverage clearly stale by 2026-08-29.
- **P1:** route is sound but misses mechanisms, milestones, quantitative evidence or a major recent branch.
- **P2:** presentation, visual coverage, metadata normalization or secondary breadth.

Counts are a repository snapshot after adding the causal-streaming chapter, with the mechanism, foundation, task-map, application, interpolation, inpainting, text-to-video, image-to-video, video-to-video, video-prediction, action-conditioned prediction, interactive-world, digital-human, story/multi-shot, unconditional-video, open-model, video-reasoning and JEPA rows refreshed on 2026-08-30. “Visual” counts Mermaid blocks plus embedded images; a visual can still fail the scientific-quality gate.

## A. Generation mechanisms and foundation models

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `docs/generative-models.md` | 449 | 32 | 4 | 4 | Rewritten as a five-axis cross-classification map with compatibility constraints, three token-generation routes, 2024–26 milestones and claim-specific evidence; next gate is controlled implementation comparison | P2 |
| `recurrent-prediction.md` | 368 | 23 | 3 | 4 | Rewritten across deterministic transforms, recurrent state, stochastic latents, RSSM, diffusion rollout and forcing boundaries with a reproducible evidence log; next gate is matched open-loop/closed-loop reproduction | P2 |
| `variational-generation.md` | 488 | 27 | 2 | 4 | Rewritten to separate stochastic latent generators from codec/tokenizer roles, with tensor/compression/causality audits; next gate is a fixed-generator codec ablation | P2 |
| `adversarial-generation.md` | 358 | 32 | 3 | 4 | Rewritten across full video GANs, tokenizer/decoder critics and explicit adversarial video distillation, including negative classifications; next gate is fixed-teacher/NFE/critic reproduction | P2 |
| `autoregressive-generation.md` | 647 | 20 | 2 | 4 | Rewritten across strict token, frame, set and chunk factorization, continuous heads, cache complexity and forcing; next gate is a matched decoder/cache implementation study | P2 |
| `masked-generation.md` | 399 | 18 | 3 | 4 | Rewritten across absorbing-state/discrete diffusion links, confidence calibration and token/frame/tube/block schedules; next gate is controlled schedule/calibration reproduction | P2 |
| `diffusion-models.md` | 413 | 24 | 2 | 4 | Rewritten across DDPM→score→reverse SDE→PF-ODE, parameterization, weighting, architecture and acceleration; next gate is numerical notebook reproduction | P2 |
| `flow-consistency-models.md` | 546 | 20 | 2 | 4 | Rewritten across FM/RF/reflow, CM/sCM/rCM, Shortcut/MeanFlow/FACM/AlphaFlow and DMD/DMD2 with video evidence boundaries; next gate is matched-budget reproduction | P2 |
| `causal-streaming-generation.md` | 374 | 25 | 2 | 3.5 | New focused review; next gate is independent reproduction/benchmark integration, not more paper names | P2 |
| `foundation-models.md` | 352 | 42 | 3 | 4 | Rewritten as a data→captioner→tokenizer→generator→post-train→distill→decode/audio/safety/API system with release-surface evidence; next gate is clean-environment manifest reproduction | P2 |

### Mechanism-level structural correction

The repository should stop treating these as one mutually exclusive taxonomy:

1. **representation:** pixel, continuous latent, discrete token, hybrid;
2. **factorization:** full-sequence, recurrent, autoregressive, masked, chunk/rolling;
3. **training objective:** likelihood/ELBO, adversarial, diffusion/score, flow matching, consistency/DMD, preference/RL;
4. **backbone:** CNN/RNN, U-Net, Transformer/DiT, SSM/hybrid;
5. **deployment:** offline, causal streaming, real-time interactive.

Modern systems routinely choose one item from several axes. Diffusion Forcing, CausVid, Self Forcing, MarDini and MaskFlow make this especially visible.

## B. Task chapters

| File | Lines | Refs | Visuals | Current depth | Main gap or correction | Priority |
|---|---:|---:|---:|---:|---|---|
| `text-to-video.md` | 455 | 36 | 4 | 4 | Rewritten around a strict task boundary, orthogonal route matrix, runtime condition contract, optional post-training, native AV, release surfaces and acceptance evidence; next gate is matched open-checkpoint reproduction | P2 |
| `image-to-video.md` | 465 | 34 | 3 | 4 | Rewritten around pixel/latent/soft temporal anchors, full condition tensors, motion/camera/audio routes, preservation–motion trade-offs and versioned evidence; next gate is matched checkpoint/control reproduction | P2 |
| `video-to-video.md` | 439 | 47 | 3 | 4 | Rewritten around a strict source-time-axis contract, route-specific preservation assumptions, formal/release status, eight-axis evaluation, multi-turn non-destructive editing and causal streaming; next gate is matched-checkpoint editing reproduction | P2 |
| `digital-human.md` | 366 | 45 | 3 | 4 | Rewritten around seven non-interchangeable tasks, time/sync/identity/authorization contracts, release surfaces, loss conflicts, leakage, counterfactuals and 2026 long-duration/streaming/full-body work; old unlabelled-score PNG is no longer cited and a score-free contract visual was verified; next gate is matched open-checkpoint reproduction | P2 |
| `frame-interpolation.md` | 480 | 32 | 4 | 4 | Rewritten as correspondence/reconstruction versus generative routes, with hard endpoint constraints, milestone-date rules, protocol traps, failure analysis and a reproducible evidence log; next gate is matched-data/runtime reproduction | P2 |
| `video-prediction.md` | 466 | 30 | 3 | 4 | Rewritten around train-only versus deployment-GT boundaries, deterministic/stochastic/latent/diffusion routes, forcing distinctions, calibrated rollout and an evidence ladder; next gate is matched long-rollout and downstream-utility reproduction | P2 |
| `action-conditioned-prediction.md` | 447 | 33 | 3 | 4 | Rewritten around action clocks, intervention versus logged correlation, pixel/probabilistic/task/representation/generative routes, model exploitation, calibrated futures and real closed-loop utility; next gate is the preregistered ActionFork-2D reproduction | P2 |
| `video-inpainting.md` | 545 | 38 | 3 | 4 | Rewritten from valid-pixel propagation through missing-region synthesis and global temporal verification, with mask-outside preservation, scene-cut reset, benchmark traps and a reproducible evidence log; next gate is controlled mask/protocol reproduction | P2 |
| `story-multishot.md` | 391 | 34 | 3 | 4 | Rewritten around shot contracts, planner/joint/memory/storyboard/streaming routes, state transactions, conflict rollback, current formal venues and release surfaces; SVD misclassification removed; next gate is matched long-story reproduction | P2 |
| `interactive-world-generation.md` | 463 | 39 | 3 | 4 | Rewritten around action schedules, dual clocks, speculative versus authoritative memory, formal 2026 frontiers, model/product/artifact release surfaces, external interlock and post-hoc audit; next gate is common-hardware closed-loop reproduction | P2 |
| `unconditional-video-generation.md` | 456 | 35 | 3 | 4 | Rewritten around strict deployment-time $p(X)$ boundaries, GAN/token/diffusion/flow routes, fidelity–coverage–tail–memorization evidence, protocol traps and a reproducible experiment; Sora/Cosmos moved to adjacent conditional systems; next gate is matched-data reproduction | P2 |

## C. Reasoning, world models, physics and evaluation

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `video-reasoning.md` | 809 | 58 | 3 | 4 | Added UniVR, the causal-generative gap and RuleMaze as a scoped adjacent control; normalized MME-CoF/VChain/Thinking-with-Video/VIPER venues and separated output, denoising and interaction clocks; next gate is causal-state and closed-loop reproduction | P2 |
| `world-models.md` | 523 | 33 | 3 | 4 | Rewritten around cascaded versus joint world-action routes, receding-horizon execution, pseudo-action recovery, verifier scope, persistent memory, uncertainty and decision utility; next gate is independent closed-loop reproduction | P2 |
| `physical-consistency.md` | 528 | 41 | 3 | 4 | Rewritten with a six-level concept taxonomy, one canonical L0–L7 evidence ladder, 2024–26 method/benchmark registries, grouped interventions, program measurements, counterfactuals and closed-loop validation; an independent primary-source audit closed four evidence-level P1s, and the next gate is independent GAUGE/robotics replication | P2 |
| `evaluation.md` | 845 | 62 | 3 | 4 | Rewritten with 2025–26 generation/editing/reasoning/world-model families, evaluator stress tests, pairwise statistics, judge/arena calibration, SLO/energy, watermark and C2PA 2.4; independent audit passed after citation and standards-boundary fixes, and the next gate is evaluator/inter-validator replication | P2 |
| `jepa.md` | 340 | 19 | 4 | 4 | Rewritten around the exact teacher–student contract, collapse, dense/semantic trade-off, action-conditioned latent MPC, V2.1 attribution, multi-future uncertainty, release surfaces and claim-specific tests; next gate is matched latent-control reproduction | P2 |

## D. Navigation, applications and resources

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `taxonomy.md` | 260 | 9 | 3 | 4 | Rewritten as condition source × source-content relation × interaction-horizon axes, with task contracts, boundary cases, milestone criteria and a reproducible evidence log; next gate is coverage expansion without collapsing the axes | P2 |
| `applications.md` | 324 | 12 | 3 | 4 | Rewritten as capability claim→workflow→acceptance→hard deployment gates→monitor/rollback, with digital-human, creative, world and Physical-AI protocols plus provenance limits; next gate is domain-specific production replication | P2 |
| `timeline.md` | 722 | 0 local anchors | 1 Mermaid + 76 images | 3 breadth | Image-rich but relation-poor; add mechanism evolution lanes and separate recent preprints as frontier observations | P1 |
| `reading-list.md` | 252 | 47 | 0 | 2.5 | Needs route-oriented prerequisites and current causal/streaming, post-training and native AV reading paths | P1 |
| `resources/datasets.md` | 795 | 38 | 2 | 4 | Rewritten as a current scoping review with release-surface taxonomy, an exact ten-stage data-engine path plus governance loop, rights/provenance, action/physics branches, manifest and fixed-compute validation; release-unit claims passed independent audit, and the next gate is acquisition/decode-yield replication | P2 |
| `resources/open-models.md` | 449 | 35 | 4 | 4 | Rewritten as a dated release-surface audit using a nine-axis vector and R0–R4 evidence levels, current version corrections, model/task/hardware selection, smoke tests, licenses and a reproducibility card; next gate is clean-environment checkpoint reproduction | P2 |
| `bibliography.md` / registry | 82 / registry | 58 registered core items | 0 | 2 | At least 31 arXiv IDs used by audited mechanism/foundation pages are not registered; metadata/code status inconsistencies remain | P0 infrastructure |

## P0 factual and evidence corrections

These should be fixed before broad stylistic expansion:

1. Reclassify IFRNet as an efficient convolutional encoder–decoder/feature-refinement method, not a Transformer.
2. Stop using general Video GAN, MoCoGAN, SVG, SVD or Video Diffusion Models as direct task-specific milestones without an explicit “technical ancestor” label.
3. ~~Remove or replace SVD as evidence for an LLM storyboard/multi-shot pipeline.~~ Corrected on 2026-08-30; the story chapter now separates SVD as an unrelated ancestor and audits current multi-shot routes.
4. ~~Remove Sora and Cosmos from the representative unconditional-generation route.~~ Corrected on 2026-08-30; both are now adjacent conditional systems, not direct milestones.
5. Standardize dates as “first preprint year / formal publication year”; FramePack, Hallo, SadTalker and ShotAdapter are now corrected, while LDF-VFI still requires the same audit.
6. ~~Mark the digital-human overview's `Identity 0.92` and `Temporal Consistency 0.90` as illustrative or regenerate it without fabricated-looking scores.~~ Corrected on 2026-08-30: the legacy file is retained but uncited; the chapter uses a score-free generated contract visual.
7. Update C2PA from 2.2 to the official 2.4 specification and preserve the version date.
8. Split model technical significance from current product availability; a discontinued product does not erase its research milestone, but should not be presented as currently available.

## Visual implementation queue

Each visual must have an explicit learning objective, editable/accessible alternative and a post-generation visual check.

| Priority | Visual | Learning objective |
|---|---|---|
| Done | Orthogonal mechanism cube/map | Show representation, factorization, objective, backbone and deployment as composable axes |
| Done | Video tokenizer tensor pipeline | Show causal 3D encoding, spatial/temporal compression, latent shape, generator and decoder bottlenecks |
| Done | Modern foundation system stack | Track governance→representation→generator→post-train→decode→deployment and separate full GAN, codec critic and adversarial distillation roles |
| P0 | DDPM–SDE–PF-ODE–FM/RF–CM/DMD map | Prevent the most common diffusion/flow/consistency terminology errors |
| Done | Offline bidirectional vs causal streaming | Separate full-clip denoising from chunk-causal output, bounded memory and new-condition loop |
| Done | Data/provenance pipeline | Source→shot split→dedup→filter→caption→license/provenance→versioned split; generated figure plus accessible Mermaid alternative verified |
| P0 | Physics evidence loop | Condition→state→generator/simulator→measurement→constraint/reward→falsification |
| Done | WAM dual route | Compare cascaded planner+generator with joint action-video model and verifier |
| Done | Three-axis task map | Condition source, source-preservation strength and open-loop→closed-loop |
| Done | T2V prompt-to-evidence contract | Separate pure text, source-pixel hybrid and closed-loop tasks; connect runtime conditions, optional post-training, generation and acceptance gates |
| Done | I2V anchor-and-control contract | Separate pixel/latent/soft anchors and trace image, motion, camera and audio conditions into the denoiser and preservation/motion gates |
| Done | Video-prediction deployment contract | Separate train-only future supervision from deployment rollout and connect claim strength to calibration, intervention and closed-loop evidence |
| Done | Interactive-world closed-loop stack | Separate causal action flow, memory read/update, external pre-execution interlock and post-hoc evidence audit |
| Done | VFI dual branch | Flow/depth/occlusion deterministic branch vs diffusion/DiT generative branch |
| Done | Video inpainting pipeline | Valid-pixel propagation→missing-region synthesis→global consistency→outside-mask protection |
| Done | Video reasoning three clocks | Output-time, denoising-time and interaction-time reasoning with intervention points |
| Done | Evaluation evidence ladder | L0 visual plausibility through L7 real-world closed-loop utility |
| Done | Multi-shot memory workflow | Script/character bible→shot plan→references→generation→memory update→conflict/rollback |
| Done | Unconditional-generation evidence chain | No external condition→marginal model→sampling→fixed output contract→quality/coverage/tail/memorization evidence |
| Done | Digital-human condition and sync contract | Authorized inputs→timeline alignment→condition fusion→generation→separate sync/identity/motion/media gates |
| Done | Open-model release surface | Official identity→version→artifact split→license intersection→hardware→smoke test→reproducibility manifest |
| Done | V2V method selector | Local/global, appearance/motion, new-view and multi-turn requirements to method families |
| Done | Action-conditioned intervention-to-control loop | Observed history→action schedule→counterfactual rollouts→fresh-observation planning loop |
| Done | JEPA MPC loop | Encoder→action-conditioned predictor→cost→planner→action→new observation |

## Primary-source refresh queues

Before editing each field, verify its queue at the primary paper or official proceedings; do not copy this audit's shorthand into final prose without a fresh read.

- **Mechanisms/foundation:** Diffusion Forcing, MarDini, MaskFlow, LTX-Video, VidTok, Pyramid Flow, T2V-Turbo, CausVid, Self/Causal Forcing, Causal-rCM, Step-Video, Open-Sora 2, SkyReels-V2, MAGI-1, Wan2.2, HunyuanVideo 1.5, Ovi, LTX-2.
- **Physics/evaluation:** NewtonRewards, Physion-Eval, Physics-IQ Verified, Apple-π, GAUGE, Interpreting Physics in Video World Models, VBench 2.0, WorldReasonBench, WBench, MIND, EntityBench, HuM-Eval/HuM-Bench.
- **World/action:** WAM survey, World Action Verifier, DreamGen, ViPRA, WorldPack, Infinite-World, ReWorld.
- **Tasks:** IFRNet, EDEN, BiM-VFI, LDF-VFI; MCVD/FramePack; STTN/FuseFormer/E²FGVI/ProPainter; STIV/ReasonDiff; ShotAdapter/OneStory/MultiShotMaster; VACE/EasyV2V/Editto; Exploration-Driven GIE/Vid2World/Dexterous World Models.
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
