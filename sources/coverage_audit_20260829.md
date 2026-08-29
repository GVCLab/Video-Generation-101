# Repository coverage, depth and visual audit — 2026-08-29

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

Counts are a repository snapshot after adding the causal-streaming chapter. “Visual” counts Mermaid blocks plus embedded images; a visual can still fail the scientific-quality gate.

## A. Generation mechanisms and foundation models

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `docs/generative-models.md` | 441 | 32 | 3 | 4 | Rewritten as a five-axis cross-classification map with compatibility constraints, 2024–26 milestones, claim-specific evidence and an independently audited visual; next gate is controlled implementation comparison | P2 |
| `recurrent-prediction.md` | 89 | 5 | 0 | 2 | Missing SV2P/SVG-LP/SAVP/VideoFlow comparison and teacher/open-loop/self-forcing bridge | P1 |
| `variational-generation.md` | 77 | 5 | 0 | 1.5 | Mixes stochastic latent models with tokenizer/codec role; no tensor shapes, causal padding or compression trade-off | P0 |
| `adversarial-generation.md` | 63 | 5 | 0 | 2 | Stops at DVD-GAN; misses StyleGAN-V/DIGAN and GAN's current decoder/distillation role | P1 |
| `autoregressive-generation.md` | 74 | 5 | 0 | 2 | Does not separate pixel, discrete-token, continuous-latent and chunk-wise AR; no KV/cache complexity | P1 |
| `masked-generation.md` | 75 | 5 | 0 | 2 | Missing absorbing-state/discrete diffusion link, confidence calibration and tube/frame/block schedules | P1 |
| `diffusion-models.md` | 413 | 24 | 2 | 4 | Rewritten across DDPM→score→reverse SDE→PF-ODE, parameterization, weighting, architecture and acceleration; next gate is numerical notebook reproduction | P2 |
| `flow-consistency-models.md` | 546 | 20 | 2 | 4 | Rewritten across FM/RF/reflow, CM/sCM/rCM, Shortcut/MeanFlow/FACM/AlphaFlow and DMD/DMD2 with video evidence boundaries; next gate is matched-budget reproduction | P2 |
| `causal-streaming-generation.md` | 376 | 25 | 2 | 3.5 | New focused review; next gate is independent reproduction/benchmark integration, not more paper names | P2 |
| `foundation-models.md` | 236 | 22 | 1 | 2.5 | Needs data→captioner→tokenizer→generator→post-train→distill→SR/audio/safety/API system view and current open models | P0 |

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
| `text-to-video.md` | 207 | 12 | 0 | 2.5 | 2025–26 model claims lack direct citations; missing native audio-video, post-training/RL, long video and model-availability boundary | P0 |
| `image-to-video.md` | 63 | 9 | 0 | 1.5 | Generic ancestors are presented too close to I2V milestones; missing STIV, subject/reference-time, identity and camera/3D control protocols | P0 |
| `video-to-video.md` | 303 | 22 | 0 | 3 | Strongest task review; add mechanism-choice visual and split benchmark/data milestones; fresh JoyAI entry is now marked provisional | P1 |
| `digital-human.md` | 237 | 22 | 1 | 3 | Correct Hallo/SadTalker date convention; generated PNG contains unlabelled illustrative scores; add 2026 long-duration/camera-control work | P0 factual, then P1 |
| `frame-interpolation.md` | 59 | 7 | 0 | 1 | IFRNet is incorrectly placed in Transformer route; missing SoftSplat/IFRNet/AMT/EDEN/BiM-VFI and standard protocols | P0 |
| `video-prediction.md` | 72 | 13 | 0 | 1.5 | Missing MCVD; FramePack venue stale; must separate pixel/distribution/latent prediction and planning utility | P0 |
| `action-conditioned-prediction.md` | 76 | 15 | 0 | 2 | Needs action-intervention, counterfactual, model-exploitation and closed-loop benchmark protocols | P1 |
| `video-inpainting.md` | 61 | 8 | 0 | 1 | Jumps from optical flow to 2024–25 DiT; missing STTN, FuseFormer, E²FGVI and ProPainter lineage | P0 |
| `story-multishot.md` | 64 | 9 | 0 | 1 | SVD does not support the claimed LLM storyboard route; ShotAdapter venue stale; missing OneStory/MultiShotMaster and benchmark protocol | P0 |
| `interactive-world-generation.md` | 75 | 13 | 0 | 1.5 | Mixes demos, papers and platforms; needs FPS/resolution/horizon/loop-memory/closed-loop evidence matrix | P0 |
| `unconditional-video-generation.md` | 68 | 15 | 0 | 1.5 | Sora and Cosmos are conditional foundation/world models, not representative unconditional methods | P0 factual |

## C. Reasoning, world models, physics and evaluation

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `video-reasoning.md` | 746 | 55 | 1 | 3.5 | Integrate already-audited 2026 Thinking in Video, UniVR and RuleMaze; normalize MME-CoF/VChain/VIPER venue status | P1 |
| `world-models.md` | 304 | 26 | 1 | 2.5 | Add cascaded vs joint WAM, action verifier, persistent memory and decision-utility benchmarks; label repository synthesis | P0 |
| `physical-consistency.md` | 528 | 41 | 3 | 4 | Rewritten with a six-level concept taxonomy, one canonical L0–L7 evidence ladder, 2024–26 method/benchmark registries, grouped interventions, program measurements, counterfactuals and closed-loop validation; an independent primary-source audit closed four evidence-level P1s, and the next gate is independent GAUGE/robotics replication | P2 |
| `evaluation.md` | 845 | 62 | 3 | 4 | Rewritten with 2025–26 generation/editing/reasoning/world-model families, evaluator stress tests, pairwise statistics, judge/arena calibration, SLO/energy, watermark and C2PA 2.4; independent audit passed after citation and standards-boundary fixes, and the next gate is evaluator/inter-validator replication | P2 |
| `jepa.md` | 283 | 10 | 2 | 3 | Add quantitative comparison, uncertainty/multimodality and action-conditioned planning loop; distinguish probe evidence from physical law | P1 |

## D. Navigation, applications and resources

| File | Lines | Refs | Visuals | Current depth | Main gap | Priority |
|---|---:|---:|---:|---:|---|---|
| `taxonomy.md` | 76 | 0 | 1 | 1.5 | Linear task chain implies a false progression; replace with condition source × preservation × open/closed-loop task map | P0 |
| `applications.md` | 118 | 9 | 0 | 1.5 | Missing digital-human applications and capability→system requirement→acceptance metric→failure/safety gate mapping | P0 |
| `timeline.md` | 722 | 0 local anchors | 1 Mermaid + 76 images | 3 breadth | Image-rich but relation-poor; add mechanism evolution lanes and separate recent preprints as frontier observations | P1 |
| `reading-list.md` | 252 | 47 | 0 | 2.5 | Needs route-oriented prerequisites and current causal/streaming, post-training and native AV reading paths | P1 |
| `resources/datasets.md` | 795 | 38 | 2 | 4 | Rewritten as a current scoping review with release-surface taxonomy, an exact ten-stage data-engine path plus governance loop, rights/provenance, action/physics branches, manifest and fixed-compute validation; release-unit claims passed independent audit, and the next gate is acquisition/decode-yield replication | P2 |
| `resources/open-models.md` | 77 | 8 | 0 | 1 | Stale 2026 list; conflates paper/code/weights; missing license, memory, checkpoint date and reproducibility status | P0 |
| `bibliography.md` / registry | 82 / registry | 58 registered core items | 0 | 2 | At least 31 arXiv IDs used by audited mechanism/foundation pages are not registered; metadata/code status inconsistencies remain | P0 infrastructure |

## P0 factual and evidence corrections

These should be fixed before broad stylistic expansion:

1. Reclassify IFRNet as an efficient convolutional encoder–decoder/feature-refinement method, not a Transformer.
2. Stop using general Video GAN, MoCoGAN, SVG, SVD or Video Diffusion Models as direct task-specific milestones without an explicit “technical ancestor” label.
3. Remove or replace SVD as evidence for an LLM storyboard/multi-shot pipeline.
4. Remove Sora and Cosmos from the representative unconditional-generation route.
5. Standardize dates as “first preprint year / formal publication year”; this affects Hallo, SadTalker, FramePack, ShotAdapter and LDF-VFI.
6. Mark the digital-human overview's `Identity 0.92` and `Temporal Consistency 0.90` as illustrative or regenerate it without fabricated-looking scores.
7. Update C2PA from 2.2 to the official 2.4 specification and preserve the version date.
8. Split model technical significance from current product availability; a discontinued product does not erase its research milestone, but should not be presented as currently available.

## Visual implementation queue

Each visual must have an explicit learning objective, editable/accessible alternative and a post-generation visual check.

| Priority | Visual | Learning objective |
|---|---|---|
| P0 | Orthogonal mechanism cube/map | Show representation, factorization, objective, backbone and deployment as composable axes |
| P0 | Video tokenizer tensor pipeline | Show causal 3D encoding, spatial/temporal compression, latent shape, generator and decoder bottlenecks |
| P0 | DDPM–SDE–PF-ODE–FM/RF–CM/DMD map | Prevent the most common diffusion/flow/consistency terminology errors |
| Done | Offline bidirectional vs causal streaming | Separate full-clip denoising from chunk-causal output, bounded memory and new-condition loop |
| Done | Data/provenance pipeline | Source→shot split→dedup→filter→caption→license/provenance→versioned split; generated figure plus accessible Mermaid alternative verified |
| P0 | Physics evidence loop | Condition→state→generator/simulator→measurement→constraint/reward→falsification |
| P0 | WAM dual route | Compare cascaded planner+generator with joint action-video model and verifier |
| P0 | Three-axis task map | Condition source, source-preservation strength and open-loop→closed-loop |
| P0 | VFI dual branch | Flow/depth/occlusion deterministic branch vs diffusion/DiT generative branch |
| P0 | Video inpainting pipeline | Valid-pixel propagation→missing-region synthesis→global consistency→outside-mask protection |
| P1 | Video reasoning three clocks | Output-time, denoising-time and interaction-time reasoning with intervention points |
| P1 | Evaluation evidence ladder | L0 visual plausibility through L7 real-world closed-loop utility |
| P1 | Multi-shot memory workflow | Script/character bible→shot plan→references→generation→memory update→conflict/rollback |
| P1 | V2V method selector | Local/global, appearance/motion, new-view and multi-turn requirements to method families |
| P1 | JEPA MPC loop | Encoder→action-conditioned predictor→cost→planner→action→new observation |

## Primary-source refresh queues

Before editing each field, verify its queue at the primary paper or official proceedings; do not copy this audit's shorthand into final prose without a fresh read.

- **Mechanisms/foundation:** Diffusion Forcing, MarDini, MaskFlow, LTX-Video, VidTok, Pyramid Flow, T2V-Turbo, CausVid, Self/Causal Forcing, Causal-rCM, Step-Video, Open-Sora 2, SkyReels-V2, MAGI-1, Wan2.2, HunyuanVideo 1.5, Ovi, LTX-2.
- **Physics/evaluation:** NewtonRewards, Physion-Eval, Physics-IQ Verified, Apple-π, GAUGE, Interpreting Physics in Video World Models, VBench 2.0, WorldReasonBench, WBench, MIND, EntityBench, HuM-Eval/HuM-Bench.
- **World/action:** WAM survey, World Action Verifier, DreamGen, ViPRA, WorldPack, Infinite-World, ReWorld.
- **Tasks:** IFRNet, EDEN, BiM-VFI, LDF-VFI; MCVD/FramePack; STTN/FuseFormer/E²FGVI/ProPainter; STIV/ReasonDiff; ShotAdapter/OneStory/MultiShotMaster; VACE/EasyV2V/Editto; Exploration-Driven GIE/Vid2World/Dexterous World Models.
- **Resources:** InternVid, OpenVid-1M, Physion/CLEVRER/PHYRE, NewtonBench-60K, VBVR, AgiBot World, RoboMIND/RH20T, modern audio-video and multi-view/3D sets.

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
