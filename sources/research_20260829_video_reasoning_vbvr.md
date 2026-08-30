# Video reasoning literature and citation audit: from Zero-Shot Learners to VBVR

- **Audit date:** 2026-08-29
- **Purpose:** support the standalone review in [`docs/video-reasoning.md`](../docs/video-reasoning.md), using *Video models are zero-shot learners and reasoners* as the narrative origin and VBVR as a later infrastructure node
- **Scope:** reasoning through video generation, latent/diffusion visual reasoning, process-aware evaluation, verifiable training, collaborative inference, and the boundary with world models. “Reasoning about an existing video” is included only when it changes how the generation-side evidence should be interpreted.

## 1. Search and verification method

The review used a layered search rather than treating one citation database as complete.

1. Start from the official v2 HTML/PDF of [*Video models are zero-shot learners and reasoners*](https://arxiv.org/abs/2509.20328), and separate its stated hypothesis, system setup, quantitative results, qualitative demonstrations, and limitations.
2. Trace backward to controlled world-model scaling, continuous visual transformation, and visualized-thought work that made the hypothesis plausible.
3. Trace forward to papers that turn the original phenomenon into benchmarks, supervised or reward-based training, mechanism probes, inference-time scaling, and closed-loop systems.
4. Audit the current VBVR arXiv record and v3 HTML/PDF, its project page, benchmark page, and EvalKit as a major scale-up node rather than the field origin.
5. Query VBVR forward citations from Semantic Scholar and OpenAlex, then search arXiv HTML/PDF full text for `2602.20159`, the exact title, and `VBVR`.
6. Open each candidate paper and distinguish a reference-list hit from actual use of VBVR data, checkpoints, tasks, or scores.
7. Independently scan the official [Awesome Video Reasoning](https://github.com/Video-Reason/Awesome-Video-Reasoning) list for important parallel papers that do not cite VBVR.
8. Use each original paper as the authority for dataset sizes and metrics. When a comparison table in another paper disagrees, retain both values and label the mismatch.

### Focal-paper verification checkpoint

- The reviewed focal version is arXiv:2509.20328v2, revised 2025-09-29.
- The paper reports 62 qualitative tasks and 7 quantitative tasks. Its 18,384 generated videos decompose into 744 qualitative generations (62 tasks × 12 samples) and 17,640 quantitative generations.
- Its operational meaning of zero-shot is no task-specific fine-tuning or task head. Because Veo's training data and post-training are closed, this is not evidence that related patterns or demonstrations were absent from training.
- The tested object is a black-box system: closed-model Veo 2 or Veo 3 accessed through a publicly callable Google Vertex AI API, including its LLM prompt rewriter. Input is an initial image plus a text instruction; output is an 8-second, 16:9, 720p, 24 FPS video.
- Chain-of-Frames is introduced with cautious analogy language: successive frames can apply changes through space and time, paralleling the visible role of language tokens in Chain-of-Thought. The paper does not causally establish that one output frame equals one internal reasoning step.
- The authors explicitly judge that the simple Sudoku solution likely came from the LLM rather than the video model. Their standalone Gemini 2.5 Pro controls neither identify the hidden rewriter nor reproduce its actual rewritten prompts, so they reduce but do not eliminate the hidden-component attribution problem.
- The paper's `pass@k` protocols are task-specific: maze success is binary best-of-$k$; edge and segmentation take the best metric over candidates after within-video frame selection; visual analogy uses majority voting for $`k\gt1`$. They share an added-sampling budget, but do not have one universal probabilistic interpretation.
- The Veo 2 identifier is internally inconsistent: the main methods text uses `veo-2.0-generate-001`, while several appendix passages use `veo-2.0-generate-preview-001`. Veo 2 versus Veo 3 is a product-version comparison, not a controlled scaling law.
- The focal paper is therefore treated as a broad phenomenon-discovery and agenda-setting paper. MME-CoF converts breadth into a multidimensional benchmark; VR-Bench narrows CoF to programmable maze trajectories and training; VBVR scales to many tasks and rule scorers; Demystifying Video Reasoning challenges the frame-level mechanism with Chain-of-Steps.

### VBVR citation-index snapshot

- [Semantic Scholar paper page](https://www.semanticscholar.org/paper/d3e893d3d9a722aea0373ed05c282a81f310a52a) returned 22 citing papers on 2026-08-29.
- API lookup used:
  - `https://api.semanticscholar.org/graph/v1/paper/ARXIV:2602.20159?fields=paperId,title`
  - `https://api.semanticscholar.org/graph/v1/paper/d3e893d3d9a722aea0373ed05c282a81f310a52a/citations?fields=title,year,externalIds,contexts,intents,isInfluential&limit=100`
- OpenAlex split the arXiv/DOI identities and showed a stale count of zero on the main record, while full-text search still located citing papers. It was therefore used for discovery, not as the authoritative count.
- Crossref did not expose a useful arXiv citation graph; DataCite metadata counts were also stale.
- Combining databases and direct PDF checks produced a lower bound of **28 formal citing works**: 27 arXiv papers and one OpenReview workshop paper.

The number is a dated snapshot, not a permanently closed citation set.

---

## 2. VBVR version and numerical audit

**Primary source:** [A Very Big Video Reasoning Suite, arXiv:2602.20159v3](https://arxiv.org/abs/2602.20159), revised 2026-08-27 and accepted to ICML 2026.

### 2.1 Current main-text figures

| Item | v3 main-text value | Interpretation |
|---|---:|---|
| Curated and publicly released tasks | 150 | Current main data-generation pipeline |
| Images | 2,015,000 | Initial and target states used to form clips |
| Clips | 1,007,500 | 1,000,000 train + 7,500 test pool |
| Training tasks | 100 | 10,000 clips per task |
| Training clips | 1,000,000 | Full released training pool |
| Test-pool tasks | 150 | 50 clips per task |
| Test-pool clips | 7,500 | Distinct from the benchmark evaluation subset |
| VBVR-Bench tasks | 100 | 50 ID + 50 OOD |
| Evaluated cases | 500 | Five samples per benchmark task |

### 2.2 Internal version mismatches

- The arXiv abstract metadata and one stale appendix sentence still say **200 tasks**. The current method/data main text says **150**.
- The main text says task design began from more than 500 proposals; an appendix passage says an initial pool of more than 300.
- The full 7,500 test pool is not the same as the 500-case VBVR-Bench evaluation subset.
- The main scaling curve stops at 500K training clips even though the release contains 1M. It is incorrect to describe the plot as a complete 1M scaling curve.

### 2.3 Scorer and human-alignment boundary

- VBVR uses task-specific 0–1 rule scorers and reports Spearman $`\rho\gt0.9`$ against human judgments.
- The reported correlation is computed on **nine model-level win-ratio points**, not on all 4,500 video-level judgments.
- Scorer completeness differs by task. Some scorers verify action order and legal transitions; others mainly test final location, color retention, or absence of large jumps.
- The result supports more repeatable, interpretable model ranking than a generic VLM judge. It does not prove every intermediate causal constraint is checked.

### 2.4 Main reported results

| Model/reference | VBVR overall |
|---|---:|
| Human | 0.974 |
| Wan2.2 base | 0.371 |
| Sora 2 | 0.546 |
| Veo 3.1 | 0.480 |
| VBVR-Wan2.2 | 0.685 |
| VBVR-LTX2.3 | 0.516 |

- VBVR-Wan2.2 improves 84.6% relative to its base score.
- Across the displayed 0K–500K curve, ID rises from 0.412 to 0.760 and OOD from 0.329 to 0.610.
- A persistent gap of roughly 15% remains; returns visibly flatten after approximately 200K–400K.
- VBench-I2V overall changes little, 0.8816 to 0.8835. Camera-motion consistency rises from 0.5444 to 0.6592, while dynamic degree falls from 0.5285 to 0.4106. Reasoning score, motion, and general generation quality therefore need separate reporting.

### 2.5 Safest one-sentence interpretation

VBVR shows that large-scale, rule-verifiable visual task supervision can substantially improve structured visual state transformation and transfer some behavior to held-out task families; it does not show that scale alone yields general causal reasoning, a faithful frame-by-frame internal algorithm, or a closed-loop world model.

---

## 3. Backward genealogy: what VBVR consolidated

VBVR is infrastructure and consolidation, not the origin of video reasoning. The modern sequence is best described as parallel waves rather than a single chain.

### 3.1 Controlled world-model scaling precursor

| Work | Role before VBVR | Main evidence | Boundary |
|---|---|---|---|
| [How Far Is Video Generation from World Model: A Physical Law Perspective / PhyWorld](https://arxiv.org/abs/2411.02385) | Controlled ID/compositional/OOD scaling precedent | Box2D tasks; data 30K→3M; DiT about 22M→310M; ID and coverage improve, physical OOD does not reliably close | Controlled 2D physics, not general logic or planning |

### 3.2 Capability discovery and Chain-of-Frames framing

| Work | Role | Key facts | Evidence boundary |
|---|---|---|---|
| [Video models are zero-shot learners and reasoners](https://arxiv.org/abs/2509.20328) | Modern empirical trigger and CoF naming | 62 qualitative + 7 quantitative tasks; 18,384 generated videos; Veo 3 5×5 maze pass@10 78% vs Veo 2 14% | Closed model and prompt rewriter; qualitative tasks have few samples; best-frame/pass@10 is search upper bound |
| [MME-CoF](https://arxiv.org/abs/2510.26802) | Compact multidimensional diagnosis | 12 dimensions, 59 curated entries, six displayed model variants | Gemini 2.5 Pro judge; VBVR Table 1 says 120, conflicting with the native paper’s 59 |
| [Thinking with Video](https://arxiv.org/abs/2511.04570) | Extends video reasoning to multimodal/text-centric tasks | VideoThinkBench 4,149 samples; studies ICL, self-consistency and test-time scaling | Prompt rewriting can dominate: Wan 2.5 without rewriter fell to zero on GSM8K/MMLU in the reported test; visible process often remained wrong/unreadable |

### 3.3 Benchmark diversification in late 2025

| Work | What it added | Native scale or result | Critical caution |
|---|---|---|---|
| [TiViBench](https://arxiv.org/abs/2511.13704) | Four reasoning dimensions, three difficulty levels, VideoTPO | 595 image-prompt samples, 24 scenarios | Improvement partly comes from external LLM critique and prompt revision |
| [Gen-ViRe](https://arxiv.org/abs/2511.13853) | Six cognitive dimensions and mixed process/final-state judging | 24 subtasks, 72 prompts, over 2,500 generated videos | Small prompt set; heavy dependence on Gemini judge |
| [VR-Bench](https://arxiv.org/abs/2511.15065) | Programmatic maze supervision, SFT, trajectory metrics, sampling scaling | Native paper reports 7,920 videos, five maze families | VBVR train/test entries sum to 7,874; the mismatch is unresolved |
| [V-ReasonBench](https://arxiv.org/abs/2511.16668) | Deterministic final-frame evaluation across four dimensions | 326 instances, 652 initial/target images, 9,780 generated videos | Explicitly reveals “right endpoint, wrong process” cases |
| [RULER-Bench](https://arxiv.org/abs/2512.02622) | Rule categories and checklist judging | 622 instances, 40 tasks, six rule classes; judge agreement 0.8512 | “Rule-based” describes tested rules; scoring still relies on GPT-o3 |
| [MMGR](https://arxiv.org/abs/2512.14691) | Unified image/video generative reasoning across abstract, embodied and physical domains | 1,853 samples; ARC-AGI below 10%, Sudoku below 7% | Output modalities and judges vary across tasks |
| [SVBench](https://arxiv.org/abs/2512.21507) | Social intention and interaction | Main benchmark uses 15 tasks, 135 prompts | Seed suite is 30 tasks, but only 15 are in the short-video main evaluation; agent/judge dependence |

### 3.4 From evaluation to intervention and training

| Work | Intervention | Why it matters |
|---|---|---|
| TiViBench / VideoTPO | Multi-sample critique and prompt refinement, no VGM weight update | Separates base capability from system-level test-time compute |
| VR-Bench / Wan-R1 line | Programmatic trajectory SFT | First narrow but clean video-reasoning training testbed |
| [NewtonRewards](https://arxiv.org/abs/2512.00425) | Physics-grounded, verifiable post-training rewards | Turns optical-flow/physical proxies into an optimizable signal |
| [VIPER](https://arxiv.org/abs/2512.24952) | Process-Oriented Correctness | Shows final-state scoring can reward outcome hacking |

### 3.5 Historical placement warning

[Demystifying Video Reasoning](https://arxiv.org/abs/2603.16870) was first submitted on 2026-03-17, after VBVR v1 on 2026-02-23. It appears in the August v3 references because VBVR was revised later. It is a post-VBVR mechanism paper, not a precursor.

---

## 4. Forward citations that actually use or inherit VBVR assets

The following 12 papers do more than mention VBVR. They use its tasks, data, checkpoints, scores, or derived benchmark material.

| # | Work | VBVR dependency | New contribution |
|---:|---|---|---|
| 1 | [Demystifying Video Reasoning](https://arxiv.org/abs/2603.16870) | Uses VBVR-Wan/LTX models and samples 200 cases mainly from VBVR-Bench | Chain-of-Steps, denoising interventions, Training-Free Ensemble; VBVR 0.685→0.716 |
| 2 | [CollabVR](https://arxiv.org/abs/2605.08735) | Directly evaluates VBVR-Wan2.2 on VBVR-Bench | VLM plan → short VGM clip → VLM verify loop; VBVR-Wan 0.671→0.757 at matched compute |
| 3 | [SenseNova-U1](https://arxiv.org/abs/2605.12500) | Adds VBVR-Image preview evaluation and VBVR-derived comparisons | Unifies multimodal understanding and generation under NEO-unify |
| 4 | [Video Models Can Reason with Verifiable Rewards / VideoRLVR](https://arxiv.org/abs/2605.15458) | Evaluates VBVR-OOD transfer against VBVR-Wan2.2 | SDE-GRPO, dense decomposed rewards, Early-Step Focus; about 40% lower training latency |
| 5 | [VLMs are Good Teachers](https://arxiv.org/abs/2606.02564) | Uses VBVR domain-adaptation SFT and benchmark reward | VLM-derived differentiable reward + test-time LoRA; VBVR/RULER average +16.7 points |
| 6 | [World Model Self-Distillation](https://arxiv.org/abs/2606.12072) | Appendix VBVR OOD puzzle evaluation | VLM creates tasks/solutions, Demonstrator generates traces, Executor is distilled and RL-refined |
| 7 | [OpenCoF](https://arxiv.org/abs/2607.08763) | OpenCoF-17K includes 30 VBVR subtasks and 7,750 VBVR-derived videos | Visual/text reasoning tokens and organized intermediate-state supervision |
| 8 | [Apple-$\pi$](https://arxiv.org/abs/2607.16401) | Evaluates VBVR-Wan2.2 and discusses VBVR reasoning supervision | Law-grounded physical intelligence: perception → formulation → deduction |
| 9 | [Articulated Object Reconstruction from Rest-State Observation](https://arxiv.org/abs/2607.27749) | Uses Wan2.2 + VBVR LoRA to generate articulation hypotheses | Applies a video-reasoning checkpoint as a dynamic prior for a 3D inverse problem |
| 10 | [ChronoVision](https://arxiv.org/abs/2608.05631) | Builds Vbvr-VQA from VBVR-Dataset/Bench-Data | Latent final-state reconstruction, ROI evidence localization, joint RL reward |
| 11 | [VGI-Bench](https://arxiv.org/abs/2608.19583) | Evaluates multiple VBVR-tuned checkpoints and synthetic-transfer claims | 27 tasks, 810 instances; process validity, input sensitivity, limited late correction |
| 12 | [VBVR-Pro](https://arxiv.org/abs/2608.26105) | Rewrites 150 VBVR generators and adds 150 | 300 tasks, verifiable reward, multimodal comparison, multi-task RL, judge audit |

### Direct-use conclusions

- The first wave uses VBVR mainly as a diagnostic substrate for mechanism and closed-loop inference.
- The second wave converts deterministic scorers into RL or test-time rewards.
- Later work restructures data around intermediate states, creates derived QA tasks, or uses VBVR checkpoints as reusable dynamic priors.
- VBVR-Pro closes the loop from tasks and scoring to optimization, modality comparison, and mechanism probing.

---

## 5. Formal citations where VBVR is context rather than dependency

These 16 works formally cite VBVR, but their central experiment does not depend on VBVR assets. They are still important to the broader research route.

| # | Work | Main contribution | Actual role of VBVR |
|---:|---|---|---|
| 1 | [EndoCoT](https://arxiv.org/abs/2603.12252) | Iterative latent thought guidance and terminal grounding; four-task average 92.1%, +8.3 points | Related-work evidence that video priors can express logical transformations |
| 2 | [MME-CoF-Pro](https://arxiv.org/abs/2603.20194) | 303 samples, 16 categories, necessary-step Reasoning Score, text/visual hints | Background example of large-scale verifiable evaluation |
| 3 | [Video Models Reason Early](https://arxiv.org/abs/2603.30043) | Early plan commitment, ~12-step threshold, ChEaP; long maze 7%→67% | Benchmark background; experiments use VR-Bench |
| 4 | [How Far Are Video Models from True Multimodal Reasoning?](https://arxiv.org/abs/2604.19193) | CLVG-Bench, 1,000+ metadata, six categories, 47 subcategories | Benchmark taxonomy/comparison |
| 5 | [Evaluating Spatial World Modeling in Video Generators via 3D Camera Trajectory Generation](https://openreview.net/forum?id=fd80936a37b7aeb373d2dc4f88b2ae3311e8ac12) | Floor-plan navigation with jointly generated video and 3D camera pose | Related-work comparison |
| 6 | [Do multimodal models imagine electric sheep?](https://arxiv.org/abs/2605.09693) | Decodes latent visual states from a VLM; 16 visual tokens per step improve 83%→about 89% | Broad emergent-generalization evidence |
| 7 | [WorldReasonBench](https://arxiv.org/abs/2605.10434) | 436 open-world future-state cases, 22 subcategories, about 6K preference pairs | Synthetic-puzzle benchmark background |
| 8 | [Entity-Centric World Models / IA-JEPA](https://arxiv.org/abs/2605.15466) | Interaction-aware masking for collision/momentum; CLEVRER causal 3.22%→14.26% | Zero-shot puzzle benchmark background |
| 9 | [PaintBench](https://arxiv.org/abs/2606.00188) | 20 precise editing classes and pixel-level deterministic scoring | Borrows rule-based human-aligned scorer philosophy |
| 10 | [Physics-IQ Verified](https://arxiv.org/abs/2606.18943) | Audits physics benchmark; changes 57.6% samples/34.8% prompts; ranking $\tau=0.46$ | Synthetic reasoning benchmark background |
| 11 | [Video-MME-Logical](https://arxiv.org/abs/2606.27828) | Controlled temporal-logical operations and intermediate-state diagnosis for MLLMs | Generation-oriented reasoning comparison |
| 12 | [The Seriality Gap in Video Diffusion Models](https://arxiv.org/abs/2607.13031) | Shows degradation with dependency-chain length; denoising steps do not provide arbitrary serial depth | World-simulator/reasoner motivation |
| 13 | [Hierarchical Denoising for Multi-Step Visual Reasoning](https://arxiv.org/abs/2607.15278) | Coarse-to-fine latent tree; success 34.22→60.29; 54.2× faster than bidirectional diffusion search | Related benchmark/method lineage |
| 14 | [Visual prompt engineering for video models](https://arxiv.org/abs/2607.25537) | Edits visual problem presentation; VPCT Veo 3.1 41.3%→59.3% | Broad video-foundation/reasoning context |
| 15 | [Deferred Exposure of Future Trajectories](https://arxiv.org/abs/2608.01755) | Avoids future-trajectory leakage in autonomous-driving RLVR | Very broad benchmark context |
| 16 | [Visual General Intelligence: A White Paper](https://arxiv.org/abs/2608.25924) | Visual-centric general-intelligence agenda | Survey/background citation |

### Why this distinction matters

A citation edge is not a method-dependency edge. For example, Reason Early is central to the video-reasoning mechanism story, but it cites VBVR only as a new benchmark and runs its core experiments on VR-Bench. Conversely, Articulated Object Reconstruction is not a central reasoning paper, but it directly uses a VBVR LoRA checkpoint in its pipeline.

---

## 6. Important parallel work not recoverable from the VBVR citation graph

Filtering only by “cites VBVR” would miss several important branches:

| Work | Independent contribution |
|---|---|
| [From Perception to Action / CHAIN](https://arxiv.org/abs/2602.21015) | Interactive physics-driven 3D reasoning benchmark; submitted one day after VBVR v1 and currently does not cite it |
| [NEWTON](https://arxiv.org/abs/2605.18396) | Physics-aware agent toolchain with planner, scientific computation, keyframes, VGM and verifier |
| [World Reasoning Arena](https://arxiv.org/abs/2603.25887) | Evaluates action simulation fidelity, long-horizon forecast, simulative reasoning and planning |
| [V-Bridge](https://arxiv.org/abs/2603.13089) | Transfers video-generation priors into few-shot image restoration |
| [ImagiNav](https://arxiv.org/abs/2603.13833) | VLM subgoals + imagined video + inverse dynamics for robot navigation |
| [Wan-R1](https://arxiv.org/abs/2603.27866) | GRPO for flow-based VGM with trajectory/embedding-level verifiable rewards |
| [UniVR](https://arxiv.org/abs/2607.12800) | Global + step rewards across manipulation, spatial puzzles and physical reasoning |
| [Thinking in Video: Can Video Generators Really Reason About the Real World?](https://arxiv.org/abs/2607.17523) | Measures a Perception–Prediction Gap between explicit causal answers and plausible generated futures |
| [RuleMaze](https://arxiv.org/abs/2608.20237) | Natural-language rule generation, logical forms and executable validators for unseen-rule generalization |

There is also a naming collision: [Lumos-Nexus](https://arxiv.org/abs/2605.31603) calls its own 208-case, eight-dimension benchmark “VR-Bench.” It is not the 7,920-video maze benchmark in arXiv:2511.15065. Reviews should disambiguate by paper title or arXiv ID.

---

## 7. Evidence-led development route

### Phase 1 — emergence, September 2025

- Question: can a pretrained video generator perform a task without task-specific training?
- Evidence: qualitative behaviors, best-frame scores, pass@$k$.
- Unresolved: black-box prompt rewriting, single-run reliability, mechanism.

### Phase 2 — benchmark diversification, October–December 2025

- Question: which spatial, physical, logical, embodied, and social abilities are present?
- Evidence: MME-CoF, TiViBench, Gen-ViRe, VR-Bench, V-ReasonBench, RULER, MMGR, SVBench.
- Unresolved: VLM-judge calibration, final-state shortcuts, small prompt sets.

### Phase 3 — scalable supervised learning, February 2026

- Question: do task diversity and data scale create cross-task generalization?
- Evidence: VBVR’s million-clip training pool, deterministic scorers, ID/OOD curve.
- Unresolved: saturation, persistent OOD gap, process completeness, dynamics trade-off.

### Phase 4 — mechanism and process, March–April 2026

- Question: does reasoning happen along output frames, denoising steps, network depth, or early plan states?
- Evidence: MME-CoF-Pro, Demystifying, Reason Early, Seriality Gap, later VGI-Bench.
- Unresolved: causal generality across architectures and tasks.

### Phase 5 — verifiable optimization and closed loops, May–June 2026

- Question: how can the model be made reliable at inference time?
- Evidence: VideoRLVR, CollabVR, VLM-as-Teacher, World Model Self-Distillation.
- Unresolved: external-teacher attribution, reward hacking, matched compute.

### Phase 6 — structure, multimodality, and real-world laws, July–August 2026

- Question: what architecture and substrate support longer chains and real-world transfer?
- Evidence: OpenCoF, Hierarchical Denoising, Apple-$\pi$, ChronoVision, VGI-Bench, VBVR-Pro.
- Unresolved: long serial dependencies, causal faithfulness, sim-to-real, image/video/interleaved compute trade-offs.

---

## 8. Claim ledger used by the review

| Claim | Supported by | Safe wording | Unsafe extension |
|---|---|---|---|
| Video models show zero-shot reasoning-like behavior | Video-Zero-Shot, MME-CoF, TiViBench | Non-zero success on specified tasks under stated sampling budget | General reasoning or reliable one-shot solver |
| Visual generation can help some reasoning | VisWorld-Eval, MVoT, visual planning work | Helps tasks whose state is naturally spatial/visual | Video is universally superior to text |
| Training scale helps | VR-Bench, VBVR, PhyWorld | Improves ID and some held-out performance | More data will close OOD by itself |
| Intermediate frames can be useful | CoF benchmarks, process scores | Output frames expose a checkable state trajectory | One frame is necessarily one internal thought |
| Denoising contains plan information | Demystifying, Reason Early, VGI-Bench | Intermediate denoising states predict and causally affect outcomes in tested settings | All video architectures reason the same way |
| More frames/steps increase compute | Thinking in Frames, Seriality Gap | Can help when they enable additional effective dependency computation | More frames or denoising steps always mean deeper reasoning |
| External VLM feedback improves systems | VideoTPO, CollabVR, VLMs-as-Teachers | Improves the combined planner/generator/verifier system | Proves the base VGM independently reasons better |
| Physics-like output indicates some learned prior | NewtonRewards, PhyWorld, Apple-$\pi$ | Shows task-specific physical consistency under selected checks | Learned general physical laws or calibrated causal model |
| Closed-loop video can support planning | CHAIN, CollabVR, NEWTON, world-model work | Short rollouts plus feedback can improve task success | Open-loop generation alone is a decision world model |

---

## 9. Remaining limitations of this audit

- The field is moving quickly and papers are revised after initial submission; citation direction can appear cyclic across versions.
- Closed commercial systems hide prompt rewriters, audio/text solvers, safety filters, and rerankers.
- Many papers use “reasoning” for different objects: final state, visible process, latent state, or a complete agent pipeline.
- A paper’s reported metric may have multiple aggregation levels; this audit preserves the native wording where a denominator is not fully recoverable.
- Full independent reproduction of the reported model scores was outside this literature audit. The review verifies claims against primary papers and official repositories, not by rerunning proprietary models.
- Citation counts should be refreshed before publication or submission; the 28-paper set is a verified lower bound as of the audit date.
