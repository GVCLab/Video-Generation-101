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

Counts are a repository snapshot after adding the causal-streaming, video post-training, native audio-video and fine-grained control chapters, with the mechanism, foundation, task-map, application, interpolation, inpainting, text-to-video, image-to-video, video-to-video, video-prediction, action-conditioned prediction, interactive-world, digital-human, story/multi-shot, unconditional-video, open-model, video-reasoning, JEPA, diffusion/flow, physics, timeline, reading-route and bibliography rows refreshed on 2026-08-30. “Visual” counts Mermaid blocks plus embedded images; a visual can still fail the scientific-quality gate.

## A. Generation mechanisms and foundation models

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `docs/generative-models.md` | 451 | 32 | 4 | 4 | Rewritten as a five-axis cross-classification map with compatibility constraints, three token-generation routes, 2024–26 milestones and claim-specific evidence; now links a dedicated post-training contract, and the next gate is controlled implementation comparison | P2 |
| `recurrent-prediction.md` | 368 | 23 | 3 | 4 | Rewritten across deterministic transforms, recurrent state, stochastic latents, RSSM, diffusion rollout and forcing boundaries with a reproducible evidence log; next gate is matched open-loop/closed-loop reproduction | P2 |
| `variational-generation.md` | 488 | 27 | 2 | 4 | Rewritten to separate stochastic latent generators from codec/tokenizer roles, with tensor/compression/causality audits; next gate is a fixed-generator codec ablation | P2 |
| `adversarial-generation.md` | 358 | 32 | 3 | 4 | Rewritten across full video GANs, tokenizer/decoder critics and explicit adversarial video distillation, including negative classifications; next gate is fixed-teacher/NFE/critic reproduction | P2 |
| `autoregressive-generation.md` | 647 | 20 | 2 | 4 | Rewritten across strict token, frame, set and chunk factorization, continuous heads, cache complexity and forcing; next gate is a matched decoder/cache implementation study | P2 |
| `masked-generation.md` | 399 | 18 | 3 | 4 | Rewritten across absorbing-state/discrete diffusion links, confidence calibration and token/frame/tube/block schedules; next gate is controlled schedule/calibration reproduction | P2 |
| `diffusion-models.md` | 415 | 24 | 2 | 4 | Rewritten across DDPM→score→reverse SDE→PF-ODE, parameterization, weighting, architecture and acceleration, with an explicit bridge to the five-layer transport map; next gate is numerical notebook reproduction | P2 |
| `flow-consistency-models.md` | 636 | 20 | 4 | 4 | Rewritten across FM/RF/reflow, CM/sCM/rCM, Shortcut/MeanFlow/FACM/AlphaFlow and DMD/DMD2; a verified five-layer map now separates training objectives, continuous processes, training-free solvers, learned few-step routes and the orthogonal streaming axis; next gate is matched-budget reproduction | P2 |
| `causal-streaming-generation.md` | 374 | 25 | 2 | 3.5 | New focused review; next gate is independent reproduction/benchmark integration, not more paper names | P2 |
| `video-post-training-alignment.md` | 385 | 38 | 2 | 4 | New focused review separating CPT/SFT, RM, pair construction, DPO/RWR, online feedback, policy-gradient RL, test-time guidance and reward-guided distillation, with credit, cost, hacking and evaluator-independence gates; next gate is matched-checkpoint reproduction | P2 |
| `foundation-models.md` | 356 | 42 | 3 | 4 | Rewritten as a data→captioner→tokenizer→generator→post-train→distill→decode/audio/safety/API system with release-surface evidence and links to the three new contracts; next gate is clean-environment manifest reproduction | P2 |

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
| `native-audio-video-generation.md` | 479 | 33 | 2 | 4 | New task-contract review separating V2A/A2V, shared-condition, staged, inference-coupled and native joint generation, then auditing codecs, clocks, dual-stream/single-stream fusion, streaming memory, search and five independent acceptance gates; next gate is a fixed-output open-checkpoint comparison | P2 |
| `controllable-video-generation.md` | 517 | 45 | 2 | 4 | New task-contract review organizing camera, trajectory, pose/structure and identity controls by coordinate/time contract, injection point, conflict handling, release surface and falsifiable evaluation; next gate is a matched-backbone controller reproduction | P2 |

## C. Reasoning, world models, physics and evaluation

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `video-reasoning.md` | 809 | 58 | 3 | 4 | Added UniVR, the causal-generative gap and RuleMaze as a scoped adjacent control; normalized MME-CoF/VChain/Thinking-with-Video/VIPER venues and separated output, denoising and interaction clocks; next gate is causal-state and closed-loop reproduction | P2 |
| `world-models.md` | 523 | 33 | 3 | 4 | Rewritten around cascaded versus joint world-action routes, receding-horizon execution, pseudo-action recovery, verifier scope, persistent memory, uncertainty and decision utility; next gate is independent closed-loop reproduction | P2 |
| `physical-consistency.md` | 594 | 41 | 5 | 4 | Rewritten with a six-level concept taxonomy, one canonical L0–L7 evidence ladder, 2024–26 method/benchmark registries and an explicit condition→state→rollout→measurement→falsification contract; training signals are separated from sealed evaluators, and the next gate is independent GAUGE/robotics replication | P2 |
| `evaluation.md` | 845 | 62 | 3 | 4 | Rewritten with 2025–26 generation/editing/reasoning/world-model families, evaluator stress tests, pairwise statistics, judge/arena calibration, SLO/energy, watermark and C2PA 2.4; independent audit passed after citation and standards-boundary fixes, and the next gate is evaluator/inter-validator replication | P2 |
| `jepa.md` | 340 | 19 | 4 | 4 | Rewritten around the exact teacher–student contract, collapse, dense/semantic trade-off, action-conditioned latent MPC, V2.1 attribution, multi-future uncertainty, release surfaces and claim-specific tests; next gate is matched latent-control reproduction | P2 |

## D. Navigation, applications and resources

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `taxonomy.md` | 283 | 12 | 3 | 4 | Rewritten as condition source × source-content relation × interaction-horizon axes, with dedicated native-AV and visual-control contracts, eight boundary cases, milestone criteria and a reproducible evidence log; next gate is coverage expansion without collapsing the axes | P2 |
| `applications.md` | 326 | 12 | 3 | 4 | Rewritten as capability claim→workflow→acceptance→hard deployment gates→monitor/rollback, with direct routes into post-training, native-AV and control contracts plus domain-specific provenance limits; next gate is production replication | P2 |
| `timeline.md` | 821 | 0 local anchors | 2 Mermaid + 75 images | 4 | Added a four-axis evidence model and explicit 2026 post-training, native-AV and control frontier rows; first disclosure, formal publication, editorial milestone, artifact and current product availability remain separate; next gate is periodic primary-source refresh | P2 |
| `reading-list.md` | 417 | 55 unique primary links | 2 | 4 | Rebuilt as prerequisites→common trunk→five specialized routes→shared falsification→capstone, adding a control-signal route with minimum reproductions and explicit claim-downgrade rules; next gate is learner-run reproduction evidence | P2 |
| `resources/datasets.md` | 795 | 38 | 2 | 4 | Rewritten as a current scoping review with release-surface taxonomy, an exact ten-stage data-engine path plus governance loop, rights/provenance, action/physics branches, manifest and fixed-compute validation; release-unit claims passed independent audit, and the next gate is acquisition/decode-yield replication | P2 |
| `resources/open-models.md` | 449 | 35 | 4 | 4 | Rewritten as a dated release-surface audit using a nine-axis vector and R0–R4 evidence levels, current version corrections, model/task/hardware selection, smoke tests, licenses and a reproducibility card; next gate is clean-environment checkpoint reproduction | P2 |
| `bibliography.md` / registry | 162 / registry | 109 registered core items | 0 | 4 | Core mechanism/foundation arXiv coverage is 59/59, registry/metadata/BibTeX identity sets agree and 21 official-code relations are explicit; next gate is deliberate chapter-local expansion rather than indiscriminate ingestion | P2 |

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

## Visual implementation queue

Each visual must have an explicit learning objective, editable/accessible alternative and a post-generation visual check.

| Priority | Visual | Learning objective |
|---|---|---|
| Done | Orthogonal mechanism cube/map | Show representation, factorization, objective, backbone and deployment as composable axes |
| Done | Video tokenizer tensor pipeline | Show causal 3D encoding, spatial/temporal compression, latent shape, generator and decoder bottlenecks |
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

## Primary-source refresh queues

Before editing each field, verify its queue at the primary paper or official proceedings; do not copy this audit's shorthand into final prose without a fresh read.

- **Mechanisms/foundation:** Diffusion Forcing, MarDini, MaskFlow, LTX-Video, VidTok, Pyramid Flow, T2V-Turbo, CausVid, Self/Causal Forcing, Causal-rCM, Step-Video, Open-Sora 2, SkyReels-V2, MAGI-1, Wan2.2, HunyuanVideo 1.5, Ovi, LTX-2; VideoDPO, VideoAlign, DynamicsBoost, Dual-IPO, BranchGRPO and noisy-latent rewards.
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
