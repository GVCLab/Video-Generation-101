# 生成运输机制与物理证据闭环：研究和图示审计

> 检索与核验日期：2026-08-30（Asia/Shanghai）；对应正文：`docs/generative-models/diffusion-models.md`、`docs/generative-models/flow-consistency-models.md`、`docs/physical-consistency.md`
> 目标：补齐 coverage audit 中两项 P0 图示——DDPM–SDE–PF-ODE–FM/RF–CM/DMD 关系图，以及 condition→state→generator/simulator→measurement→constraint/reward→falsification 证据闭环。

## 1. 研究问题

### 1.1 生成运输机制

1. DDPM/score、reverse SDE、PF-ODE、FM/RF、solver、CM/PD、DMD 和 streaming 分别属于哪一个设计层？
2. 哪些边表示数学导出，哪些只表示常见监督来源，哪些能力是正交的系统选择？
3. 2025–2026 的 sCM/rCM、MeanFlow、FACM、AlphaFlow 是否改变这些边界？

### 1.2 物理一致性

1. 如何把“更物理”改写成实验前可冻结、实验后可证伪的 claim contract？
2. 训练 reward、测量器、VLM judge、真实参照和独立环境应怎样隔离？
3. 何时证据只能支持 plausibility，何时能推进到 physical fidelity、counterfactual fidelity 或 decision fidelity？

## 2. 检索策略

### 2.1 数据库与入口

- 正式论文：NeurIPS、ICLR、ICML/PMLR、CVPR/CVF、MLSys 官方 proceedings；
- 开放审稿与会议页：OpenReview；
- 最新预印本：arXiv abstract/version page；
- 代码状态：作者或论文官方仓库；
- 交叉检查：正文已有参考文献、时间线和 coverage audit。

### 2.2 代表性搜索式

```text
"Denoising Diffusion Probabilistic Models" objective sampler
"Score-Based Generative Modeling through Stochastic Differential Equations" probability flow ODE
"Flow Matching for Generative Modeling" conditional probability path velocity
"Rectified Flow" reflow coupling trajectory
"Consistency Models" consistency distillation training
"Distribution Matching Distillation" target score fake score
site:proceedings.iclr.cc 2026 "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency"
site:proceedings.iclr.cc 2026 "FACM: Flow-Anchored Consistency Models"
site:proceedings.iclr.cc 2026 "AlphaFlow: Understanding and Improving MeanFlow Models"
site:arxiv.org/abs/2608.05948 GAUGE physical fidelity video generation
```

### 2.3 纳入标准

- 原始方法论文或正式 proceedings；
- 对“训练目标、连续动力学、solver、student、部署轴”至少一层有明确数学定义；
- 物理论文必须给出可检查的数据、测量或评测协议；
- 2026 工作若已有正式 ICLR/CVPR/MLSys 页面，优先使用正式页面而不是旧预印本；
- 最新 arXiv 工作必须明确标为预印本，不把作者结论升级为独立共识。

### 2.4 排除标准

- 二手博客、搜索摘要和模型营销页不用于方法定义；
- 只展示样例、没有可复现实验协议的“物理”声明不用于证据层升级；
- 图像侧一步生成结果不外推为视频一步、流式或无限时长证据；
- 只因网络输出叫 `velocity`，不将工作归入 FM/RF；
- 只因轨迹是 ODE，不将工作归入 Flow Matching；
- 只因 NFE 少，不将工作归入 causal/streaming。

## 3. 一手来源与可支持断言

| 一手来源 | 状态 | 本次采用的断言 | 不采用的外推 |
|---|---|---|---|
| [DDPM](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | NeurIPS 2020 | 离散前向加噪、学习反向过程；与 denoising score matching 的联系 | DDPM 不等于所有随机/确定 sampler |
| [Score-SDE](https://openreview.net/forum?id=PxTIG12RRHS) | ICLR 2021 | 同一 score 定义 reverse SDE 与 PF-ODE；理想条件下边缘分布一致 | 两条过程不共享逐样本轨迹 |
| [DDIM](https://openreview.net/forum?id=St1giarCHLP) | ICLR 2021 | 可复用 DDPM 训练目标的非马尔可夫采样构造 | 不把它写成 FM objective |
| [DPM-Solver](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) | NeurIPS 2022 | 面向 diffusion ODE 的 training-free solver | 图像 10–20 NFE 不外推为任意视频延迟 |
| [Flow Matching](https://openreview.net/forum?id=PqvMRDCJT9t) | ICLR 2023 | 条件概率路径/条件速度回归得到边缘速度场 | 条件路径简单不保证学习轨迹一步无损 |
| [Rectified Flow](https://openreview.net/forum?id=XVjTT1nw5z) | ICLR 2023 | 直线条件桥与 reflow 重耦合 | “直线”不等于语义空间任意轨迹都线性 |
| [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) | ICML 2023 | flow-map/终点一致性；distillation 与 standalone training 两条路线 | consistency 不等于 teacher-free，也不等于固定 1 NFE |
| [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html) | CVPR 2024 | target score 与 fake score 驱动分布层蒸馏 | 不与逐轨迹 consistency 混写 |
| [DMD2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | NeurIPS 2024 | two-time-scale fake-score、GAN、on-policy/multi-step 改进 | 去掉固定回归集不等于去掉教师 |
| [sCM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html) | ICLR 2025 | 连续时间 consistency 的稳定化与可扩展训练 | JVP 训练成本不算采样 NFE |
| [MeanFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) | NeurIPS 2025 | 学习区间平均速度；正式证据以图像为主 | 不外推为视频一步生成 |
| [rCM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html) | ICLR 2026 | score regularization + continuous-time consistency；在作者设置中扩展至 14B、5 秒视频、1–4 步 | 15–50× 是特定设置结果，不是通用常数；“mitigating”不写成消除 mode collapse |
| [FACM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d0dac08f4199f0c348dd2feace0305a-Abstract-Conference.html) | ICLR 2026 | FM anchor 与 CM shortcut 联合；强结果使用预训练 teacher | 联合目标不让 FM 与 CM 成为同义词 |
| [AlphaFlow](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html) | ICLR 2026 | 在作者理论中统一 trajectory FM、Shortcut 与 MeanFlow，使用 curriculum 缓解目标冲突 | 正式实验是 ImageNet 图像，不外推到视频 |
| [Physics-IQ Verified](https://arxiv.org/abs/2606.18943) | 2026 预印本 | ground truth、prompt、权重与聚合修订可改变排名 | 不把新版排名当永久真值 |
| [GAUGE](https://arxiv.org/abs/2608.05948) | 2026-08 预印本 | 22 个受控任务族、真实轨迹、物理 metadata、不确定性；3 引擎评 14 族、6 I2V 模型评 5 个刚体任务 | 仍待独立复现；视频部分不能代表所有材料/场景 |

## 4. 综合结论

### 4.1 为什么采用“五层”而不是单时间线

1. **训练统计量：** diffusion/score 常学习可换算的 $\epsilon,x_0,v,$ score；FM/RF 学由概率路径和耦合诱导的速度场。
2. **连续过程：** score 可同时驱动随机 reverse SDE 与确定 PF-ODE；FM/RF 的学习场积分成 ODE。都是 ODE 不表示来源相同。
3. **推理求解：** DDIM、DPM-Solver 或时间网格通常复用旧参数。它们改变积分误差和 NFE，不产生新 student。
4. **新训练少步模型：** PD/CM 主要利用教师轨迹或 flow map；DMD/DMD2 主要对齐分布；Shortcut/MeanFlow/AlphaFlow 学跨步长或区间运输。2026 的 FACM/rCM 组合多个信号，但仍需逐项声明教师与目标。
5. **部署：** causal mask、chunking、KV cache、首帧延迟、稳态吞吐和持续发帧属于视频时间/系统轴，与噪声时间 NFE 正交。

图中的边因此只表达“数学导出”或“常见监督来源”，不表达唯一父子关系。正文显式注明 standalone CT、联合目标和跨层方法，避免把教学图误写成封闭分类法。

### 4.2 物理 evidence loop 的必要 gate

1. **条件 gate：** 初态、动作、物性、边界、相机与随机种子必须可追溯；否则无法归因。
2. **状态 gate：** 不同材料使用匹配的状态接口；刚体 pose 不足以表示布料 strain 或流体场。
3. **测量 gate：** 报告提取器版本、不确定性、有效 coverage 和失败样本；不允许静默删掉难例。
4. **反事实 gate：** 固定非干预因素，扫描动作/参数，并检查 effect size 与 branch locality。
5. **独立性 gate：** 训练 reward 与封存终评隔离；真实参照或独立环境不能被生成器反复查询。
6. **证伪 gate：** 预先冻结阈值、seeds、ID/OOD 和 stopping rule；任一必要 gate 失败时不得用视觉质量加权平均补偿。
7. **claim gate：** 只按实际证据报告 L0–L7；VLM plausibility 最多支持 L2，标定测量和反事实可到 L3–L4，持续闭环、策略排序和真实收益才推进 L5–L7。

## 5. 图像生成与验收记录

两张图均使用 Codex 内置 image generation，最终文件复制进仓库；未使用外部 CLI。每张图都有正文 Mermaid 与顺序化文字替代，生成图不承担唯一语义来源。

### 5.1 Diffusion–Flow–Few-Step 五层图

- 最终文件：`assets/diagrams/diffusion-flow-few-step-five-layers.png`
- 尺寸：1672 × 941
- SHA-256：`fa9c9c7e9247441c545407700d3ac75c71458bb2ca240b0e61d8b1780fb18656`
- 最终 prompt 摘要：16:9 白底、五行三列分类矩阵；行分别为 training statistic、continuous process、inference-only、trained few-step、deployment；要求逐字渲染代表方法、变化对象和三个不等式警告；禁止跨行因果箭头。
- 迭代记录：前两版采用谱系箭头，但生成器把 reverse SDE 连到 solver、solver 连到 CM/PD、PF-ODE 连到 DMD。因这些错误会改变技术含义，两版均弃用；最终版改为无跨层箭头矩阵，谱系只由 Mermaid 表达。
- 视觉检查：标题、五行、三列和三个警告完整；无裁切、重叠、水印或乱码；字号在原始分辨率下可读。
- 灰度检查：灰度范围 0–255、均值 218.05、标准差 74.42；逐行文字、分隔线、编号和三个警告仍可区分。

### 5.2 Physics Evidence Loop

- 最终文件：`assets/diagrams/physics-evidence-falsification-loop.png`
- 尺寸：1672 × 941
- SHA-256：`8f630ced47e0e733f3d78651ecbd5c0f4ab00983e464d6432c83470b553942d6`
- 最终 prompt 摘要：16:9 白底闭环；condition contract→state→generator/simulator→rollout→measurement→constraint/reward→falsification gate；封存参照只进入测量/gate；PASS 报 claim level，FAIL 回到条件合同；附三个证据警告。
- 视觉检查：独立参照没有流入 generator；PASS/FAIL 分支、失败反馈回路和 L0–L7 输出清晰；所有指定文字完整，无裁切、重叠、水印或额外步骤。
- 灰度检查：灰度范围 0–255、均值 238.82、标准差 46.21；实线主流程、虚线独立参照、菱形 gate 和反馈回路在去色后仍可辨。

### 5.3 最终 prompt：五层分类矩阵

```text
Use case: scientific-educational
Asset type: landscape textbook taxonomy card for an advanced video generation course
Primary request: create a scientifically accurate five-row comparison matrix titled exactly
"Diffusion–Flow–Few-Step: Five Separate Layers". This image must classify concepts without
drawing causal arrows between rows.
Scene/backdrop: pure white technical-paper background
Style/medium: publication-quality flat vector-like matrix, crisp sans-serif typography,
strong grid, generous whitespace
Composition/framing: 16:9 landscape. Three columns titled exactly "LAYER",
"REPRESENTATIVE CONCEPTS", "WHAT ACTUALLY CHANGES".

Five rows, using all text verbatim:
1. "TRAINING STATISTIC" | "DIFFUSION / SCORE   |   FLOW MATCHING" |
   "DENOISER OR SCORE   |   VELOCITY FIELD"
2. "CONTINUOUS PROCESS" | "REVERSE SDE   |   PF-ODE   |   LEARNED ODE" |
   "RANDOM PATH   |   DETERMINISTIC PATH"
3. "INFERENCE-ONLY" | "DDIM   |   DPM-SOLVER" |
   "SOLVER OR TIME GRID — NO RETRAINING"
4. "TRAINED FEW-STEP" | "CM / PD   |   DMD / DMD2   |   SHORTCUT / MEANFLOW" |
   "FLOW MAP   |   DISTRIBUTION   |   INTERVAL TRANSPORT"
5. "DEPLOYMENT" | "CAUSALITY   |   CHUNKING   |   KV CACHE" |
   "ORTHOGONAL TO NFE"

At the bottom, three compact warning badges exactly:
"v-PREDICTION ≠ FLOW VELOCITY"
"PF-ODE ≠ FLOW MATCHING"
"FEW-STEP ≠ STREAMING"

Color palette: colorblind-friendly Okabe-Ito inspired, with blue accent for diffusion
concepts, orange for flow concepts, purple for few-step training, green for deployment,
dark charcoal text. Use icons or row numbers only as redundant cues.
Constraints: no causal arrows between rows; render every supplied label verbatim; no Chinese
text; no extra methods; no equations; no logo; no watermark; no tiny text; no decorative
illustration; do not merge rows.
```

### 5.4 最终 prompt：物理证据闭环

```text
Use case: scientific-educational
Asset type: landscape advanced-textbook schematic for physics-aware video generation
Primary request: create a closed-loop scientific diagram titled exactly
"Physics Evidence Loop: Claim Must Survive Falsification".
Scene/backdrop: clean white technical-paper background
Style/medium: publication-quality flat vector-like systems diagram, crisp sans-serif text,
no decorative imagery
Composition/framing: 16:9 landscape. Main flow left-to-right across the center, then a clear
failure feedback loop returning underneath.

Main nodes and exact text in order:
1. "CONDITION CONTRACT"
   small second line: "s0 • action u • parameters θ • boundary b • seed"
2. "STATE"
   small second line: "track • 3D/4D • field • latent"
3. "GENERATOR / SIMULATOR"
4. "ROLLOUT"
5. "MEASUREMENT"
   small second line: "trajectory • contact • conservation • uncertainty"
6. "CONSTRAINT / REWARD"
7. diamond "FALSIFICATION GATE"
The pass branch goes to "REPORT CLAIM LEVEL L0–L7".
The fail branch goes down to "FAIL → CHANGE DATA / MODEL / EVALUATOR", then loops back to
CONDITION CONTRACT.

Add an independent side input above MEASUREMENT titled exactly
"SEALED REFERENCE / INDEPENDENT ENVIRONMENT", with a dashed arrow into MEASUREMENT and
FALSIFICATION GATE.
Add three warning badges at the bottom exactly:
"VLM JUDGE ≠ PHYSICAL TRUTH"
"LOW RESIDUAL ≠ CORRECT BOUNDARY"
"PASS ONE SEED ≠ EVIDENCE"

Color palette: Okabe-Ito inspired, blue inputs/state, orange generator, purple
measurement/constraints, red falsification, green passed claim. Use shape plus color redundancy
and clearly labeled PASS and FAIL arrows.
Constraints: render all supplied text verbatim; no extra steps; no equations beyond the supplied
symbols; no logos; no watermark; no tiny text; no arrows crossing boxes; the feedback loop must
be obvious; the independent reference must not feed the generator.
```

## 6. 残余风险与后续复现

- rCM、FACM、AlphaFlow 的正式页面已核验，但其跨模型/跨数据集普遍性仍需独立训练复现；本次只记录作者设置。
- GAUGE 是 2026-08 新预印本，协议方向重要，但视频模型只覆盖 5 个刚体任务；不能据此声称物理评测问题已经解决。
- 两张 PNG 已完成像素、路径、SHA-256、原色与灰度视觉检查；最终批次仍需在全仓合并后重跑本地链接和 `git diff --check`。
- 本批不改变具体 benchmark 排名，也不把动态图示当成数值实验结果。
