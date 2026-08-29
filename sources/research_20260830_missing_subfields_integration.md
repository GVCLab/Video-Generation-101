# Structural gap audit for the next Video Generation 101 expansion

> Audit date: **2026-08-30 (Asia/Shanghai)**. This record explains why three new chapters were added and how they must be integrated without erasing task or evidence boundaries.

## 1. Review question

Which active video-generation research axes are already mentioned in the repository but still lack a dedicated chapter that can answer all of the following?

1. What is the exact task or optimization contract?
2. Which technical routes are genuinely different?
3. Which works changed the contract or evidence standard?
4. What is formally published, only a preprint, only an artifact, or only a product claim?
5. What minimum reproduction and falsifier would test the main claim?

The audit is structural rather than keyword-only. A subject was not considered covered merely because several paper names appeared in a timeline or reading list.

## 2. Repository audit

Read-only checks used `rg --files docs`, the current coverage matrix, and targeted searches for `audio-video`, `preference`, `reward`, `DPO`, `RL`, `camera`, `trajectory`, `pose`, `depth`, `ControlNet`, `MotionCtrl` and `CameraCtrl`.

| Candidate axis | Existing coverage before this batch | Missing contract | Decision |
|---|---|---|---|
| Native joint audio-video | Scattered in T2V, foundation models, timeline and reading route | No chapter separates video-to-audio, staged generation, coupled dual streams, shared joint denoising, product output and streaming AV | Add a task chapter |
| Video post-training and alignment | Scattered in the mechanism overview, datasets, evaluation and a combined few-step reading route | No chapter separates SFT, reward modeling, DPO/RWR, policy-gradient RL, inference guidance and distillation, or treats reward leakage and denoising credit assignment as first-class risks | Add a mechanism chapter |
| Fine-grained controllable generation | The taxonomy lists masks, trajectories and cameras, but the repository had no dedicated method lineage | No chapter organizes control signal, coordinate system, injection point, source-preservation contract, conflict handling or control-specific evaluation | Add a task chapter |

## 3. Inclusion and exclusion rules

A new chapter was included only when the axis had:

- a distinct input/output or optimization contract;
- at least three independently identifiable primary works or official releases;
- evaluation failures that cannot be diagnosed by generic T2V quality metrics;
- enough current activity that folding it into a paragraph would obscure the 2025–2026 frontier.

The following were not promoted to separate chapters in this batch:

- **long video**, because continuous single-shot generation and cross-shot narrative already have separate causal-streaming and story/multi-shot routes;
- **video-to-audio**, because it is used as the staged counterfactual inside the native AV chapter rather than mislabeled as joint generation;
- **multi-view and 4D generation**, because the current addition treats them as geometry-heavy branches of explicit controllability; a future audit may split them if their task and evaluation contracts diverge further;
- **generic safety and provenance**, because they already have explicit sections in evaluation, applications, datasets and foundation-model release-surface audits.

## 4. Independent primary-source frontier check

The purpose of this table is to verify that each proposed chapter is anchored in a current technical transition, not only in an old taxonomy label. Results remain author-reported unless an independent reproduction is explicitly cited.

| Axis | Primary source | What it changes | Evidence status at freeze date |
|---|---|---|---|
| Native AV | [NAVA](https://arxiv.org/abs/2605.30073) | Separates native AV alignment from external context conditioning with Align-then-Fuse MMDiT | 2026 preprint |
| Native AV | [Inference-Time Scaling for Joint Audio-Video Generation](https://arxiv.org/abs/2606.03183) | Uses multiple verifiers and adaptive reward weighting; explicitly reports single-verifier trade-offs and hacking risk | TMLR acceptance stated on arXiv; final journal metadata must still be version-checked |
| Native AV | [Ripple](https://arxiv.org/abs/2607.26818) | Adds modality-specific recurrent memories, cross-modal interaction and block-causal streaming | 2026 preprint; latency/FPS are author protocol results |
| Post-training | [Improving Video Generation with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76227feb18ea0ee40bd15cf02c33e18e-Abstract-Conference.html) | Jointly exposes multi-dimensional VideoReward, Flow-DPO, Flow-RWR and inference-time Flow-NRG | NeurIPS 2025 formal publication |
| Post-training | [Dual-IPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a0d3f77bb435817807d463c5dcef1ab-Abstract-Conference.html) | Iteratively updates both reward model and video generator instead of freezing one side | ICLR 2026 formal publication |
| Post-training | [Consistent Noisy Latent Rewards](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html) | Moves reward evaluation into noisy latent trajectories and tests preference consistency across timesteps | ICLR 2026 formal publication |
| Post-training | [BranchGRPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/233d16f17f809981763db2f01b7f9603-Abstract-Conference.html) | Amortizes shared rollout prefixes and assigns depth-wise advantages instead of broadcasting one terminal reward uniformly | ICLR 2026 formal publication; video evidence is one reported setting |
| Post-training | [DynamicsBoost](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.html) | Builds preference order from continuation context and excludes the shared prefix from the DPO loss | CVPR 2026 formal publication |
| Controllability | [Motion Prompting](https://openaccess.thecvf.com/content/CVPR2025/html/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.html) | Treats sparse or dense motion trajectories as a general motion prompt | CVPR 2025 formal publication |
| Controllability | [GEN3C](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.html) | Renders an incrementally updated 3D cache under a target camera path before conditioning generation | CVPR 2025 formal publication |
| Controllability | [LAMP](https://openaccess.thecvf.com/content/CVPR2026/html/Kizil_LAMP_Language-Assisted_Motion_Planning_for_Controllable_Video_Generation_CVPR_2026_paper.html) | Translates cinematography language into deterministic object/camera trajectory programs | CVPR 2026 formal publication |
| Controllability | [BulletTime](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_BulletTime_Decoupled_Control_of_Time_and_Camera_Pose_for_Video_Generation_CVPR_2026_paper.html) | Explicitly decouples world time from camera pose | CVPR 2026 formal publication |
| Controllability | [FlashMotion](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FlashMotion_Few-Step_Controllable_Video_Generation_with_Trajectory_Guidance_CVPR_2026_paper.html) | Shows that distilling a base generator does not preserve a control adapter automatically and adds post-distillation adapter tuning | CVPR 2026 formal publication |
| Controllability | [4DStreamCtrl](https://arxiv.org/abs/2608.25479) | Combines camera and object control with online streaming and long rollout | 2026-08-26 preprint; real-time and long-run numbers are author-reported |

## 5. Integration contract

The three chapters must remain connected but not collapsed:

1. **Post-training** may optimize AV synchronization or trajectory adherence, but the reward does not redefine the underlying task.
2. **Few-step distillation** changes the sampling budget; it is not preference alignment even when both appear in one recipe.
3. **Camera and trajectory conditions** specify a desired visual motion; they are not environment actions unless the system establishes a state transition and closes the action-observation loop.
4. **A video with sound** is not native joint AV unless the public mechanism shows audio and video exchange information during generation.
5. **Products, papers, code, weights and current endpoints** are separate release surfaces and receive separate labels.

The repository integration therefore requires links from the task taxonomy, mechanism overview, foundation-model overview, reading route and top-level README, plus a dated update to the coverage matrix. It does not require forcing every chapter-local paper into the core bibliography registry.

## 6. Evidence boundary

This batch is a literature, artifact and documentation audit. It does not independently train or reproduce the cited large models. Mermaid rendering, image inspection, link closure and metadata checks can prove that the educational artifacts are internally sound; they cannot convert author-reported quality, latency, synchronization or controllability into independently verified performance.
