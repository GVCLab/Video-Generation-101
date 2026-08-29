# 视频预测章节研究与验收记录

## 0. 冻结范围

- 冻结时间：**2026-08-30（Asia/Shanghai）**。
- 目标页面：`docs/tasks/video-prediction.md`。
- 目标图：`assets/diagrams/video-prediction-evidence-ladder.png`。
- 主问题：过去可见条件下的未来像素/状态预测；action-conditioned 与 world model 只作为边界和 decision-aware 分支。
- 非主问题：VFI、纯文本到视频、仅视频压缩/质量预测、动作识别、天气数值预报、仅当前帧理解。
- 版本原则：arXiv 首发、当前版本与正式 proceedings 分开记录；冻结日后的更新不倒灌。

本记录是为章节可复核性保留的 bounded literature review，不声称对 4,000 余条宽泛元数据结果做了系统综述。所有结论以一手论文、正式 proceedings 或作者项目为主；OpenAlex 只用于发现和交叉检查元数据。

## 1. 使用的技能与 fallback

开始写作前完整读取：

1. `literature-review/SKILL.md`，以及直接相关的 `references/database_strategies.md`；
2. `scientific-schematics/SKILL.md`，以及 `references/best_practices.md`；
3. 内置 `imagegen/SKILL.md`，以及 `references/prompting.md`、`references/sample-prompts.md`。

`scientific-schematics/SKILL.md` 指向的 `references/diagram_types.md` 在本机目录中不存在。Fallback 是按 `best_practices.md` 的信息层级、箭头语义、色盲/灰度冗余、最小文字与视觉验收原则自行设计“rollout contract + evidence ladder”，没有伪造缺失文件的内容。

## 2. 问题拆解与证据等级

### 2.1 研究问题

1. 什么输入可见性才构成 video prediction？
2. Pixel、transform、latent、token、diffusion 与 decision state 的输出合同如何对齐？
3. Teacher forcing、scheduled sampling、self forcing、Diffusion Forcing 与 causal distillation 分别改变什么？
4. 单步像素指标、长期 rollout、条件分布、干预与闭环 utility 各能支持多强的主张？
5. MCVD、FramePack 及 2026 工作的标题、ID、首发、venue 与功能边界是什么？

### 2.2 证据等级

| 等级 | 定义 | 本章用途 |
|---|---|---|
| **E1** | 正式 conference proceedings、PMLR、CVF、NeurIPS 或 OpenReview 的接收页面/论文 | 标题、作者、正式 venue、论文机制与实验边界的首选 |
| **E2** | arXiv 当前论文/版本史，或作者官方项目/代码 | 首发日期、最新预印本、尚无正式 venue 的前沿、实现补充 |
| **E3** | OpenAlex 等索引元数据 | 检索计数、候选发现、venue/引用元数据交叉检查；不单独支撑技术主张 |
| **S** | 本章基于 E1/E2 的综合或推论 | 明示为边界、风险或验收建议，不冒充论文原结论 |

## 3. 检索源、检索式与结果数

### 3.1 arXiv API

冻结日查询 `https://export.arxiv.org/api/query`；计数取 Atom feed 的 `opensearch:totalResults`。

| 查询 | 总结果 | 本次查看 | 用途 |
|---|---:|---:|---|
| `all:"video prediction"` | 416 | 按提交时间最新 30 条 metadata | 发现 2026 前沿；识别 action/world-model 误命中 |
| `ti:"video prediction"` | 163 | 按提交时间最新 30 条 metadata | 收紧到标题直指任务的工作 |
| `all:"future frame prediction"` | 63 | 按提交时间最新 30 条 metadata | 补 future-frame 同义表述 |
| `all:"autoregressive video diffusion"` | 65 | 按提交时间最新 30 条 metadata | 补 causal/streaming/long-rollout diffusion |

四个“最新 30”是 120 个检索槽位，不是 120 篇互异论文；同一论文会跨查询重复。随后以标题/摘要筛选和 backward chaining 检查 foundational work，再对最终候选按 arXiv ID 定点回读。

定点 ID 查询至少覆盖：`1511.05440`、`1605.08104`、`2104.10157`、`2106.13195`、`2205.09853`、`2407.01392`、`2412.07772`、`2412.14169`、`2504.12626`、`2506.08009`、`2506.09985`、`2602.02214`、`2604.11707`、`2606.03971`、`2607.02087`、`2607.25984`、`2608.19556`、`2608.26794`。

### 3.2 OpenAlex API

使用 `https://api.openalex.org/works?filter=title.search:<query>&per-page=1` 获取总数；计数可能随索引更新而变化，以下只代表冻结日快照。

| `title.search` 查询 | 总结果 |
|---|---:|
| `video prediction` | 4,481 |
| `future frame prediction` | 63 |
| `autoregressive video diffusion` | 89 |
| `next-frame prediction` | 36 |

另试过 OpenAlex 的宽泛 `search=`，返回数达到数十万量级并混入 video coding、activity recognition、quality prediction 等；因此没有把它当作领域规模。对前三个 title query 分别查看按 cited-by 排序的前 8 个结果，共 24 个结果槽位，主要用于发现漏项和检查误命中。

### 3.3 正式 proceedings 与官方项目

逐条打开或定点核验以下一手入口：

- NeurIPS proceedings：ConvLSTM、Scheduled Sampling、DNA/CDNA/STP、Video Diffusion Models、MCVD、Diffusion Forcing、FramePack、Self Forcing；
- PMLR：SVG-LP、PlaNet；
- CVF Open Access：SimVP、CausVid、PHANTOM、*Envisioning the Future, One Step at a Time*、PhysInOne；
- ICML 2026 官方会议页：Causal Forcing；
- OpenReview/ICLR：SV2P、NOVA；ICLR 2016 archive 用于核对 Beyond MSE；
- 作者项目：[FramePack](https://github.com/lllyasviel/FramePack)、[MCVD implementation](https://github.com/voletiv/mcvd-pytorch)。

正式 landing page 直接核验了 18 条主参考的 venue；Beyond MSE 另以 arXiv 正文和 ICLR 2016 官方 archive 交叉检查。作者项目仅用于实现/功能补充，不用 README 的自报数字替代论文协议。

## 4. 纳入、排除与范围审计

### 4.1 纳入规则

至少满足一项：

1. 明确建模 $p(x_{future}\mid x_{past})$ 或其 latent/trajectory 等价物；
2. 对 video rollout 的 uncertainty、forcing、context/memory、diffusion sampling 或 evaluation 有关键贡献；
3. 是 action-conditioned/decision-aware 邻接工作，且能说明预测何时才升级为 world model；
4. 2026 工作直接触及长期记忆、动态奖励、future-aware training、物理状态或 representation-first prediction。

最终章节纳入 30 条编号主参考，其中 18 条有直接正式 proceedings/接收入口，12 条主要以 arXiv/CoRR 版本呈现；部分正式论文同时有 arXiv，但按最高证据等级计一次。

### 4.2 排除规则

排除或只作边界说明：

- 未来端点可见的 VFI；
- 只做 video coding、quality prediction、recognition、anticipation label，而不生成未来观测/状态；
- 只凭文本/图像 prompt 任意续写且没有真实条件未来评价；
- 只在 action-conditioned simulator 上报告控制、却被误写成被动 video prediction；
- 只有二手榜单/博客、无法回到论文或正式页面的结论；
- 冻结日后出现的版本、venue 或数字。

在 OpenAlex top-result screen 中明确遇到并排除了 multiview video coding、activity prediction/recognition、video quality prediction 等标题误命中；在 arXiv 最新页中，action/world-model、天气/交通专域和视频理解也很多。它们说明“搜到 video prediction”不等于进入本章。

### 4.3 术语判定

| 术语 | 本记录采用的判定 |
|---|---|
| Video prediction | 推理只见过去；预测更晚像素或对应状态 |
| Future-GT boundary | 未来真值/其编码不得进入部署推理；训练期 corruption、posterior、teacher forcing、foresight teacher/target 与损失均可作为明示的 train-only 路径 |
| VFI | 目标时刻两侧存在真实条件；不是外推 |
| Future frame synthesis | 只有 past-only 时才等同 prediction；否则是更宽泛条件生成 |
| Action-conditioned prediction | $p(Y\mid X,A)$；动作需显式进入 tensor/时间对齐合同 |
| World model | 至少有 action/state transition，并以 planning/control utility 评价 |
| Latent prediction | 预测未来 representation；没有 decoder/probe 就不能声称像素/状态正确 |
| Generative continuation | 追求合理续写；若无真实条件分布证据，不自动成为 forecasting |

## 5. 逐条标题、ID 与 venue 核验

“首发/ID”取 arXiv metadata 或正式论文标识；“venue”只在正式入口存在时填写，不把投稿状态写成接收。

| Ref | 一手核验标题 | ID / 首发 | 正式 venue | 等级与本章用途 |
|---:|---|---|---|---|
| 1 | *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting* | NeurIPS hash `07563...` / 2015 | NeurIPS 2015 | E1；deterministic recurrent grid state |
| 2 | *Deep multi-scale video prediction beyond mean square error* | arXiv:1511.05440 / 2015-11-17 | ICLR 2016 | E1/E2；MSE 模糊与锐化边界 |
| 3 | *Unsupervised Learning for Physical Interaction through Video Prediction* | NeurIPS hash `d9d4...` / 2016 | NeurIPS 2016 | E1；DNA/CDNA/STP、动作条件 |
| 4 | *Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning* | arXiv:1605.08104 | ICLR 2017 conference version | E2；predictive coding |
| 5 | *Stochastic Variational Video Prediction* | OpenReview `rk49Mg-CW` | ICLR 2018 | E1；SV2P |
| 6 | *Stochastic Video Generation with a Learned Prior* | PMLR v80 | ICML 2018 | E1；SVG-LP |
| 7 | *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | NeurIPS hash `e995...` | NeurIPS 2015 | E1；forcing gap |
| 8 | *Deep Kalman Filters* | arXiv:1511.05121 | 未写正式 venue | E2；stochastic state-space 模板 |
| 9 | *Learning Latent Dynamics for Planning from Pixels* | PMLR v97 | ICML 2019 | E1；PlaNet 与 utility |
| 10 | *VideoGPT: Video Generation using VQ-VAE and Transformers* | arXiv:2104.10157 | 未核到正式 venue | E2；discrete token AR |
| 11 | *FitVid: Overfitting in Pixel-Level Video Prediction* | arXiv:2106.13195 / CoRR | 未写 ICLR 接收 | E2；强简化基线、venue 纠错 |
| 12 | *SimVP: Simpler Yet Better Video Prediction* | CVPR paper page | CVPR 2022 | E1；纯 CNN 基线 |
| 13 | *Video Diffusion Models* | NeurIPS hash `39235...` | NeurIPS 2022 | E1；video diffusion |
| 14 | *MCVD - Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation* | arXiv:2205.09853 / 2022-05-19 | NeurIPS 2022 | E1/E2；mask 接口、block rollout |
| 15 | *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* | arXiv:2407.01392 / 2024-07-01 | NeurIPS 2024 | E1/E2；token-wise noise forcing |
| 16 | *NOVA: Autoregressive Video Generation without Vector Quantization* | OpenReview `JE9tCwe3lp` / arXiv:2412.14169 | ICLR 2025 | E1；continuous token/frame AR |
| 17 | *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models* | arXiv:2412.07772 | CVPR 2025 | E1；CausVid 因果蒸馏 |
| 18 | *Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models* | arXiv:2504.12626 | NeurIPS 2025 | E1/E2；context packing 与 anti-drift |
| 19 | *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion* | arXiv:2506.08009 / 2025-06-09 | NeurIPS 2025 | E1/E2；self-generated rollout training |
| 20 | *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* | arXiv:2506.09985 / 2025-06-11 | 冻结日按 arXiv | E2；action-free 表征与 AC planning 分层 |
| 21 | *PHANTOM: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics* | CVPR paper page | CVPR 2026 | E1；Ying Shen 等；物理状态 + RGB |
| 22 | *Envisioning the Future, One Step at a Time* | CVPR paper page | CVPR 2026 | E1；sparse trajectory future |
| 23 | *PhysInOne: Visual Physics Learning and Reasoning in One Suite* | CVPR paper page | CVPR 2026 | E1；Siyuan Zhou 等；统一物理 suite |
| 24 | *Video-Mirai: Autoregressive Video Diffusion Models Need Foresight* | arXiv:2606.03971 / 2026-06-02 | 无正式 venue | E2；training-only foresight |
| 25 | *Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation* | [ICML poster `65646`](https://icml.cc/virtual/2026/poster/65646) / arXiv:2602.02214 | ICML 2026 | E1；causal teacher/ODE init |
| 26 | *Ring Forcing: Towards Precise Long-Term Memory for Autoregressive Video Diffusion* | arXiv:2608.26794 / 2026-08-27 | 无正式 venue | E2；远距 retrieval，极新预印本 |
| 27 | *Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models* | arXiv:2608.19556 / 2026-08-20 | 无正式 venue | E2；4D consistency reward |
| 28 | *Schrödinger's Cat: Probabilistic Representation and Prediction of Potential Scene Kinematics* | arXiv:2607.25984 / 2026-07-28 | 作者在 arXiv 自报 accepted to ECCV 2026；冻结日未核到官方记录 | E2；潜在动力学分布 |
| 29 | *Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction* | arXiv:2604.11707 / 2026-04-13 | 无正式 venue | E2；representation-first/Re2Pix |
| 30 | *SUNTA: Hierarchical Video Prediction with Surprise-based Chunking* | arXiv:2607.02087 / 2026-07-02 | 无正式 venue | E2；surprise-based temporal abstraction |

表中大小写按一手 landing page/arXiv metadata 保留；正文可以使用论文给出的缩写，但参考文献标题不意译、不扩写。

## 6. 重点论文机制与边界核验

### 6.1 MCVD

- 一手来源：NeurIPS 2022 proceedings + arXiv:2205.09853。
- 条件机制：past frames 和 future frames 两组 condition 分别整体 mask/drop；不同组合对应 prediction、backward prediction、interpolation、unconditional generation。
- 网络/rollout：论文强调非递归 2D convolution；模型生成 frame block，任意更长视频依靠 autoregressive block rollout。
- 可以支持：统一 conditional score model 的任务接口与多样采样。
- 不能直接支持：四任务同难度、任意长无 drift、跨论文 FVD 数字天然可比。

### 6.2 FramePack 的首发、定稿与功能边界

- arXiv v1：2025-04-17 04:02:31 UTC；初题 *Packing Input Frame Context in Next-Frame Prediction Models for Video Generation*；作者为 Lvmin Zhang、Maneesh Agrawala。
- 当前 arXiv v3：题名改为 *Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models*；作者表为 Lvmin Zhang、Shengqu Cai、Muyang Li、Gordon Wetzstein、Maneesh Agrawala。
- 正式入口：NeurIPS 2025 proceedings，同当前主标题。
- 核心功能：按时间邻近、feature similarity 或 hybrid importance 压缩输入帧上下文，在固定 token/context budget 中保留更长历史；另以 endpoint、sampling order、discrete history 等设计控制 drift。
- 作者声称可在推理中使用数千输入帧；本章不把该系统数字写成跨硬件/跨任务常数。
- 功能边界：这是 context packing + sampling/drift prevention，不是物理辨识、校准未来分布或 closed-loop control。
- 因果边界：先生成 endpoint/far section 再补中间 section 的变体改变输出顺序，不等价于严格在线逐时刻 causal emission。

### 6.3 Diffusion Forcing、Self Forcing 与 CausVid

- Diffusion Forcing：每个序列 token 可有独立 diffusion noise level，以同一因果模型组合 next-token、full-sequence diffusion、variable-length rollout 与 guidance；不是 scheduled sampling 的简单重命名。
- Self Forcing：训练时直接条件于自身生成 rollout，使用 KV cache、少步 diffusion、video-level objective 与 stochastic gradient truncation；正式 venue 为 NeurIPS 2025。
- CausVid：双向 teacher 到 few-step causal student 的 distillation，包含 distribution matching 与 ODE initialization；正式 venue 为 CVPR 2025。
- 三者都触及 train–test/causality，但分别改变 loss/noise schedule、训练输入分布、teacher–student 推理结构，不能合并成一个“forcing 技术”。

### 6.4 2026 前沿的证据边界

- PHANTOM、trajectory work、PhysInOne 有 CVPR 2026 正式页面，Causal Forcing 有 ICML 2026 正式接收页，均作为 E1。
- Video-Mirai、Ring Forcing、Stream4D、Schrödinger's Cat、Representations Before Pixels、SUNTA 截止冻结日按 arXiv E2；其中 Schrödinger's Cat/GARFIELD 的 arXiv 页有作者自报 ECCV 2026 接收，但未核到官方 proceedings/会议页，故不升级。所有性能、分钟级、实时或长度主张写成作者在特定协议的结果。
- Ring Forcing 首发仅距冻结日 3 天，尤其需要后续正式评审和独立复现。

## 7. 综合框架的推导记录

### 7.1 三条正交轴

把方法只按 CNN/RNN/Transformer 分类会混淆问题。本章改为：

1. **表示**：pixel、transform、continuous latent、discrete token、trajectory/state；
2. **不确定性**：point estimate、latent variable、score/diffusion、explicit state distribution；
3. **rollout**：one-shot、frame AR、block AR、recurrent state、packed/sliding context。

例如 latent diffusion 同时占据 latent 表示与 score-based 不确定性；FramePack 主要改 rollout memory，而不是创造新的未来概率模型。

### 7.2 Evidence ladder

| 层级 | 核心问题 | 不能越级推出 |
|---|---|---|
| L0 one-step fidelity | 单步/短段是否像 GT | 长期稳定、多峰覆盖 |
| L1 multi-step open loop | 自回灌时是否持续正确 | 分布校准、干预正确 |
| L2 distribution/calibration | 多样样本是否覆盖真实条件分布 | OOD dynamics、控制有效 |
| L3 OOD/intervention | 外观/动力学变化与 action swap 是否正确 | planner 中一定有用 |
| L4 closed-loop utility | 是否提升 success/return/regret/safety | 仍需报告系统成本与安全尾部 |

该阶梯是 S 级综合：它把不同论文散落的评价目标按主张强度组织，不声称来自某一篇论文。

## 8. 图片生成与 QA 记录

### 8.1 生成方式

按任务要求使用内置 `image_gen`，没有用脚本绘制或编辑。先生成主体，再做一次局部修订：初稿已正确隔离**部署推理**中的 GT，但缺少“预测结果进入 metrics”的显式连接；修订只新增从 `EVALUATE` 到 `LOSS / METRICS` 的绿色箭头。该 PNG 是“部署推理 + 离线评价”摘要，不是完整训练 DAG；训练期的 corruption/noising、posterior、teacher forcing 与 foresight target 可使用 future GT，但必须标成 train-only 并在部署时移除。

### 8.2 原始生成 prompt

```text
Use case: scientific-educational
Asset type: publication-quality textbook schematic, 16:9 landscape
Primary request: Explain the inference rollout contract and the evidence ladder for passive video prediction. The scientific message must be exact: at inference only the observed past enters the model; generated frames or blocks are fed back during a self-conditioned rollout; stochastic models branch into multiple plausible futures; ground-truth future frames are available only to training losses or offline evaluation and must never leak into inference.
Scene/backdrop: clean white academic figure background
Style/medium: crisp vector-like scientific infographic, flat shapes, precise arrows, generous whitespace, readable sans-serif typography
Composition/framing: left 70% is a horizontal pipeline; right 30% is a vertical five-step evidence staircase. On the left show exactly these stages, left-to-right: "OBSERVED PREFIX" (three simple film-frame thumbnails) -> "MODEL" -> "PREDICT FRAME / BLOCK" -> "SELF-CONDITIONED ROLLOUT" (feedback arrow from generated output back to the next prediction) -> "K PLAUSIBLE FUTURES" (three visibly branching future strips) -> "EVALUATE". Add a separate dashed lower lane labeled exactly "TRAIN / TEST ONLY: FUTURE GROUND-TRUTH" feeding only a box labeled "LOSS / METRICS"; include a clear no-leak barrier between this lane and MODEL. Above the main input place the exact label "PAST ONLY AT INFERENCE". On the right title the staircase exactly "EVIDENCE LADDER" and label its five ascending steps exactly: "L0 ONE-STEP FIDELITY", "L1 MULTI-STEP OPEN LOOP", "L2 DISTRIBUTION & CALIBRATION", "L3 OOD & INTERVENTION", "L4 CLOSED-LOOP UTILITY". Use an upward arrow labeled "STRONGER CLAIM".
Color palette: colorblind-safe Okabe-Ito blue, orange, green and purple plus dark charcoal; every category also differs by shape, border, or line style so the figure remains interpretable in grayscale
Text (verbatim): [same labels]
Constraints: render every required phrase verbatim exactly once; no other prose; all arrow directions scientifically correct; no arrow from future ground-truth into MODEL; no model names, no benchmark scores, no invented numbers, no logos, no watermark, no decorative characters, no figure number; minimum readable text; high contrast; no overlaps or cropped labels
Avoid: photorealism, gradients, dark background, 3D perspective, tiny labels, ambiguous branching, circular loop that implies access to future ground-truth
```

### 8.3 局部修订 prompt

```text
Use case: precise-object-edit
Input images: Image 1: edit target, the scientific infographic just generated
Primary request: Add one thin dark-green directional arrow from the bottom of the "EVALUATE" clipboard to the top edge of the "LOSS / METRICS" box, so it is explicit that generated predictions and future ground-truth are compared by the metrics. Keep the existing dashed blue ground-truth arrow into "LOSS / METRICS".
Constraints: change only this connection; preserve every label verbatim, all icons, colors, shapes, spacing, dimensions, the no-leak barrier, and every existing arrow; no new text; no arrow from future ground-truth to MODEL; no watermark; do not crop anything.
```

### 8.4 文件、哈希与验收

| 项目 | 结果 |
|---|---|
| 生成原图 | `generated_images/<thread>/<artifact>.png` |
| 项目副本 | `assets/diagrams/video-prediction-evidence-ladder.png` |
| 尺寸/模式 | 1672 × 941 px；PNG；RGB；non-interlaced |
| SHA-256 | `cbb336d1a262c8ef92f57542057a5b165b588a2f49f5f1ebb68f813805dff36e` |
| 原图视觉验收 | 通过：无裁切/重叠/水印；主路径、反馈箭头、$K$ 分支、GT 隔离与 evidence ladder 可读 |
| 灰度验收 | 通过：用 ImageMagick 临时转换后以 original detail 检查；线型、边框、形状仍可区分，临时文件已删除 |
| 文本验收 | 通过：逐项人工核对下列 16 个固定标签；无虚构 benchmark 数字 |

文本 checklist：`PAST ONLY AT INFERENCE`、`OBSERVED PREFIX`、`MODEL`、`PREDICT FRAME / BLOCK`、`SELF-CONDITIONED ROLLOUT`、`K PLAUSIBLE FUTURES`、`EVALUATE`、`TRAIN / TEST ONLY: FUTURE GROUND-TRUTH`、`LOSS / METRICS`、`EVIDENCE LADDER`、`L0 ONE-STEP FIDELITY`、`L1 MULTI-STEP OPEN LOOP`、`L2 DISTRIBUTION & CALIBRATION`、`L3 OOD & INTERVENTION`、`L4 CLOSED-LOOP UTILITY`、`STRONGER CLAIM`。

## 9. 最终验证记录

以下项目在正文冻结后执行：

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Markdown lint | `npx --yes markdownlint-cli2 docs/tasks/video-prediction.md sources/research_20260830_video_prediction.md` | **通过**：markdownlint-cli2 0.23.2 / markdownlint 0.41.1，0 issues |
| 引用锚点 | 提取 `[[n]](#ref-n)` 与 `<a id="ref-n">`，检查缺失/未使用/重复 | **通过**：1–30 全部使用且全部定义；missing 0、unused 0、duplicate 0 |
| 链接 | 检查本地相对路径；并发请求正文全部外部入口 | **通过**：2/2 本地相对路径存在；32/32 唯一外部 URL 返回 HTTP 200；Causal Forcing ICML 2026 页及 PHANTOM/PhysInOne CVF 页 metadata 分别确认正式标题与 Ying Shen/Siyuan Zhou；arXiv export metadata 确认 GARFIELD 的 ECCV 2026 信息是作者 comment |
| Mermaid | 提取两段 Mermaid，用 Mermaid CLI 11.16.0 实际渲染 | **通过**：2/2 生成 SVG；均含 `flowchart-v2`、`aria-describedby` 和正确 `<title>`；train-only/deployment 图另渲染为 PNG 并以 original detail 检查，未把 future GT 画入部署模型输入；独立终验后删除了无条件 `pred → update` 旁路，仅 `stop=no` 回灌，`stop=yes` 终止 |
| PNG | `file`、`identify`、`shasum -a 256` | **通过**：1672 × 941，8-bit RGB PNG，non-interlaced；SHA-256 与上表一致；不额外声称嵌入 ICC profile |
| 文本/灰度 | 原图与临时灰度图 original-detail 视觉检查 | 已通过 |
| Diff scope | `git diff --check`、限定路径 diff 与 `git status --short` | **通过**：本轮审计修复只编辑视频预测正文与研究记录，未修改 PNG、coverage 或 nav；工作区另有并行任务文件，未触碰 |

## 10. 冻结声明

本章的“2026 frontier”只表示截至 2026-08-30 可定位的一手状态。尤其是 2026-08 的预印本，后续可能更名、增删作者或进入正式 venue；更新章节时必须重新读取 version history 与正式页面，不能沿用本记录中的冻结状态。
