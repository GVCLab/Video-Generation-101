# 交互式世界生成研究日志

## 1. 任务、范围与冻结日

- 检索与核验日期：**2026-08-30（Asia/Shanghai）**
- 对应章节：[交互式世界生成：从动作条件视频到可验证的持久世界](../docs/tasks/interactive-world-generation.md)
- 图像资产：[interactive-world-closed-loop-stack.png](../assets/diagrams/interactive-world-closed-loop-stack.png)
- 核心问题：
  1. interactive world generation 与 action-conditioned prediction、learned game engine、world model、4D/scene simulation、WAM/policy 的最小边界是什么？
  2. latent action、自回归/扩散 game engine、memory/compression、joint video–action 与 persistent world 怎样组成机制谱系？
  3. FPS、accepted-action Hz、hold/drop 规则、p95 input-to-first-affected-frame latency、horizon、memory、revisitation、action validity、uncertainty 与 decision utility 应怎样分层验收？
  4. Genie、UniSim、GameNGen、DIAMOND、Oasis、Genie 2/3、Project Genie、Matrix 3/3.5 与 2026 正式 frontier 的 paper/demo/checkpoint/product 开放面分别是什么？

本日志是面向技术综述的 structured search，不宣称为医学式系统综述。相同工作在 arXiv、OpenAlex、会议页、项目页、GitHub 和 Hugging Face 中出现时只算一个研究对象；不同入口用于核验不同证据面，不把跨库重复记录相加成伪精确 PRISMA 数字。

## 2. 技能执行与图示 fallback

执行前完整读取并遵循：

1. `literature-review/SKILL.md`，以及其直接要求的 `references/database_strategies.md` 与 `references/citation_styles.md`：用于多库检索、纳排、citation chaining、证据分级和逐项核验。
2. `scientific-schematics/SKILL.md`，以及 `references/best_practices.md`、`references/QUICK_REFERENCE.md`：用于先定义信息层级，再生成、语义复核、原图/灰度检查和 provenance 记录。
3. `imagegen/SKILL.md`，以及 `references/prompting.md`、`references/sample-prompts.md`：用于内置 image generation、项目资产保存和迭代验收。

`scientific-schematics` 指向的 `references/diagram_types.md` 在当前技能目录中不存在。执行时明确采用 `best_practices.md` 与 `QUICK_REFERENCE.md` 的布局、颜色、可访问性、灰度与验收规则作为 fallback；没有虚构缺失文件的内容。

## 3. 来源、检索式与结果数

### 3.1 OpenAlex

调用 OpenAlex Works API，使用 `title.search`，并限制 `from_publication_date:2018-01-01,to_publication_date:2026-08-30`。结果数为冻结日 API 的 `meta.count`。

| `title.search` 检索式 | 返回总数 | 用途 |
|---|---:|---|
| `interactive world model` | 152 | 发现系统、memory、benchmark 与同名工作 |
| `interactive world generation` | 30 | 检查生成/游戏/交互的术语重叠 |
| `video world model` | 379 | 发现 action-conditioned video、robotics 与长期 rollout |
| `neural game engine` | 6 | 定位 GameNGen、DIAMOND 类 learned engine |
| `persistent world model` | 35 | 发现回访、对象状态和 memory 工作 |

OpenAlex 只用于发现、去重以及日期/venue/DOI 元数据交叉检查；机制、性能与开放状态均回到论文、正式 proceedings、机构页或官方 repo。

### 3.2 arXiv API

使用 arXiv API 的精确短语或布尔字段检索；结果数来自冻结日 Atom feed 的 `opensearch:totalResults`。

| arXiv query | 返回总数 | 实际筛选 |
|---|---:|---|
| `ti:"interactive world model"` | 25 | 逐条检查全部 25 项标题/摘要；排除无关同名理论工作 |
| `all:"interactive world generation"` | 4 | 检查术语本身；数量少说明很多系统使用 world model / game engine 命名 |
| `all:"neural game engine"` | 5 | 与正式/作者页交叉定位可玩系统 |
| `all:"persistent world" AND all:model` | 25 | 筛 long-horizon memory、stateful world 与非视频叙事系统 |

对点名与入选工作再按 arXiv ID 精确打开：`2310.06114`、`2408.14837`、`2508.13009`、`2512.08931`、`2512.14614`、`2512.02473`、`2602.02393`、`2602.15922`、`2604.08995`、`2604.18564`、`2605.03941`、`2606.17730`、`2606.31672`、`2607.00836`、`2608.06332`、`2608.09449`、`2608.23565`。

### 3.3 正式 proceedings、机构页、项目与开放 artifact

至少覆盖五类互补一手来源：

| 来源类型 | 核验对象 | 承担的证据 |
|---|---|---|
| PMLR / ICLR / ICML / NeurIPS official proceedings | Genie、UniSim、GameNGen、DIAMOND、Astra、WorldPlay、Infinite-World、iWorld-Bench、Vid2World、WorldGym、World-In-World | 正式题名、venue、论文摘要、机制与实验 |
| arXiv primary paper | Matrix、WorldPack、ActWorld、MultiWorld、ReWorld、WorldRoamBench、DreamZero、GeniWorld、Sekai2；另用于已发表工作的版本追踪 | 最新版本、摘要机制、作者实验与日期 |
| 机构官方页 | Sora framing、Genie 2、Genie 3 model page、Project Genie 2026-01 首发/05 扩展与帮助页 | release/demo 能力、官方限制、访问级别；模型面与产品面分开 |
| 作者项目页 | Oasis、DIAMOND、Matrix-Game 2/3/3.5、ReWorld | demo、硬件/速度细节、失败样例、artifact 链接 |
| 官方代码/权重面 | open-oasis GitHub + Oasis 500M HF、Astra GitHub/HF、DIAMOND repo、Matrix 2/3/3.5、ReWorld repo | code/checkpoint/license/缺失权重状态 |

二手媒体、榜单聚合、SEO 页面和无作者/机构归属的 demo 没有进入结论证据。

## 4. 纳入、排除与实际筛选

### 4.1 纳入标准

1. 提出或严格评测在线 action-conditioned observation generation、learned game engine、persistent memory 或 joint video–action system 的一手工作。
2. 能改变本章边界、闭环合同或评测设计的 foundational world-model 工作。
3. 任务点名的 Genie、UniSim、GameNGen、DIAMOND、Oasis、Genie 2/3，以及冻结日前可由一手入口定位的 Project Genie 产品节点、正式 2026 frontier 与 Matrix 3/3.5 release surface。
4. 机构 release 在没有等价公开 paper/checkpoint 时可纳入，但必须降为 official claim/demo 级。
5. 代码或权重只有在 repo/model card 可实际定位时才标“开放”；占位 repo、未来计划或项目页按钮不等于 artifact。

### 4.2 排除标准

1. 只做普通 T2V/I2V、没有动作或状态闭环的工作；Sora 仅用于概念边界，不用于证明交互能力。
2. 只做固定场景 novel-view/4D rendering、没有动作干预的工作。
3. 名称含 world/game/action，但实际与环境转移、视觉 rollout 或策略无关的工作。
4. 仅有产品营销、转载、社交媒体片段且找不到机构/作者原始入口的数字。
5. 同一论文的 arXiv、会议、项目和代码不重复计为多个研究结果。
6. 截止日之后才发生的 venue 月份或未来硬件能力不写成当前事实；例如 WorldPack 页面标注的 `09/2026` 不在 2026-08-30 的已发生时间范围内。

### 4.3 实际筛选与去重说明

- 完整检查 arXiv `ti:"interactive world model"` 的 25 条结果，保留与在线视觉 world、memory、robotics 和 benchmark 直接相关者。
- 对 OpenAlex 五个 query 的新近记录做标题/摘要初筛，再以 exact title/ID 回到 primary source。
- 通过正式 paper → project → code/weights 做 backward/forward chaining；例如 DIAMOND 用 proceedings 支持论文实验，用项目页支持约 10 FPS、连续跳跃失败和 playable artifact。
- 正文章节最终建立 39 个引用入口，对应少于 39 个独立工作，因为部分工作有 paper + project/repo/product 多类互补证据。
- 未声称“排除 N 篇”的统一数字：五个 OpenAlex query、四个 arXiv query 和 artifact 入口高度重叠，强行相加会重复计算。

## 5. 证据等级

| 等级 | 定义 | 正文允许的表述 |
|---|---|---|
| **A** | 正式同行评审 proceedings，且题名/摘要/实验可定位 | 可写 venue、论文机制与作者实验；仍不称独立复现 |
| **B+** | arXiv + 可运行 code + 对应 checkpoint/weights | 可写作者结果与具体开放面；复现状态仍需本项目实测 |
| **B** | arXiv + 作者项目或代码，开放面不完整 | 可写预印本机制和 author-reported 数字，并明确缺口 |
| **B−** | 很新的 arXiv；代码或权重部分缺失 | 只作 frontier signal，避免建立成熟共识 |
| **C** | 机构/作者 release 或 curated demo，无等价公开 paper/checkpoint | 只写“官方展示/主张/限制”，不写独立验证 |
| **D** | 二手发现线索 | 不进入正文结论 |

等级衡量可核查性和开放面，不是模型能力排名。

## 6. 核心证据账本

| 工作 | 机制/用途 | 关键一手证据 | 冻结日开放面与边界 | 等级 |
|---|---|---|---|---|
| World Models / Dreamer | latent dynamics、imagination control | arXiv/ICLR paper | 不要求开放域像素实时生成 | A |
| UniSim | 异构 image/navigation/robot 数据组成 simulator | ICLR 2024 proceedings；Outstanding Paper | 未发现等价通用公开 checkpoint | A/C |
| Genie | tokenizer + AR dynamics + latent action | ICML/PMLR | 正式 paper；latent action 无物理单位 | A |
| GameNGen | action-conditioned next-frame diffusion | ICLR 2025 official poster/proceedings | DOOM、20 FPS 单 TPU、多分钟均为论文作者结果 | A |
| DIAMOND | diffusion world + agent training；CS:GO engine | NeurIPS + project/code/playable models | 论文与 artifact 都可定位；项目展示规则漏洞 | A/B+ |
| Oasis | spatial AE + latent DiT + Diffusion Forcing | official project + GitHub + HF | 500M code/weights 与更大 live demo 分开 | B+/C |
| Genie 2 | autoregressive latent diffusion、counterfactual、memory demo | DeepMind official release | 最长 1 分钟、多数 10–20 秒；无等价 open checkpoint | C |
| Genie 3 model | text-to-interactive world、promptable events | DeepMind official model/limitations page | 720p、20–24 FPS、few minutes 为官方 model claim；无公开 paper/checkpoint/API | C |
| Project Genie product | Genie 3 驱动的 web research prototype | Google 2026-01-29 launch、2026-05-19 expansion、Labs help | Ultra 成人订阅者：美国首发后全球逐步扩展；60 s；模型能力子集；Street View 美国地点先行 | C |
| Matrix-Game 2.0 | few-step causal AR diffusion、frame-level action | arXiv + project links | 作者报 25 FPS/minute；code/weights links 可定位 | B+ |
| Astra | temporal causal attention、noise history、action experts | ICLR 2026 + GitHub/HF | checkpoint/部分 pipeline 已开；长期记忆仍 TODO | A/B+ |
| WorldPlay | dual action、reconstituted context、Context Forcing | ICML 2026 + project/code link | 论文作者报 720p/24 FPS | A/B |
| Vid2World | video diffusion causalization + causal action guidance | ICLR 2026 proceedings | 跨机器人、3D 游戏、导航的正式论文实验；不自动证明权威长时状态 | A |
| WorldPack | trajectory packing + geometric selection | arXiv v3 | 4→22 effective frames；16% packing overhead；未来 venue 不前置 | B |
| Infinite-World | pose-free hierarchical memory、tri-state action labels | ICML 2026 + project | 作者报 1000+ frames；未发现等价完整 code/weights | A/C |
| Matrix-Game 3.0 | error injection/self-correction、camera memory、DMD | technical report + project + official repo | 720p/40 FPS 需带 8+1 GPU；已发 code、Unreal first-person base/distilled 两个 5B；mixed-real/28B 未发 | B+ |
| Matrix-Game 3.5 | Patch Memory、Warped PRoPE、static/dynamic decoupling、few-step distillation | 2026-07 project/report + official repo | code、first/third-person 5B base、3-step first-person distilled 已发；依赖 Wan2.2/DA3、≥40 GB；无正式 venue/独立复现 | B |
| ActWorld | action-aware hierarchy + event/object tokens | arXiv v1 + project | 100K interaction video；对象事件而非只相机漫游 | B− |
| MultiWorld | multi-agent condition + global-state encoder | arXiv v2 + project | 多视角/多主体作者实验 | B− |
| ReWorld | mixed windows、bounded KV + landmark bank、4-step LoRA | arXiv v1 + project + repo | inference code 已发；核心权重 Coming soon | B− |
| iWorld-Bench | 视觉、轨迹、memory 与统一动作生成 | ICML 2026 | 330K clips、2.1K samples、14 models 为论文作者报告 | A |
| WorldRoamBench | per-frame action、drift、physics、memory | arXiv v3 | 600+ cases、10–60 s、10+ models 为作者报告 | B− |
| DreamZero | joint video–action WAM/policy | arXiv + project | 14B/7 Hz 是 author-reported policy rate，不是 render FPS | B |
| GeniWorld | URDF visual action、robot policy evaluator | arXiv v1 | 很新；只按作者实验表述 | B− |
| Sekai2 | 长视频、pose、caption、loop/revisit 数据 | arXiv v2 | 数据集报告，不冒充完成的 interactive model | B− |
| WorldGym | autoregressive action-conditioned robot rollouts + VLM rewards | ICLR 2026 proceedings | 支持论文内 policy-success correlation/ranking；不是无误差真实环境替代品 | A |
| World-In-World | unified online planning/action API + four closed-loop environments | ICLR 2026 proceedings | task success 揭示 controllability/post-training/inference-compute 作用；不以视觉分数替代闭环效用 | A |

## 7. 关键 fresh-check

### 7.1 Genie 2、Genie 3 model 与 Project Genie product

- Genie 2 官方页写明每步输入 keyboard/mouse action、模拟 next observation；一致世界最长约 1 分钟，多数展示为 10–20 秒。
- Genie 2 官方页把样本标为 undistilled base，并说明实时 distilled 版本会降低输出质量。
- Genie 2 页面有 counterfactual、long-horizon memory 和 interesting outtakes；这些仍是官方 curated demo，而非公开 benchmark/checkpoint。
- Genie 3 当前官方模型页写明 720p、20–24 FPS 与 sustained interaction/few-minute 能力；这些仍是 model claim，未发现等价公开 paper、checkpoint 或 API。
- Project Genie 是单独的产品/平台证据面：2026-01-29 先向美国 18+ Google AI Ultra 订阅者推出实验原型；2026-05-19 起向全球符合条件的 18+ Ultra 订阅者逐步扩展，并加入美国地点先行的 Street View grounding。
- Project Genie 官方发布仍写单次生成限制 60 秒，且明确 promptable events 等部分 Genie 3 model demo 能力未进入该原型；因此不能拿 60 秒产品配额否定模型页的分钟级 claim，也不能拿模型页能力冒充产品现状。
- Google Labs 当前帮助页仍把访问限定为 Google AI Ultra 会员，并列出生成限制/政策入口；日志记录的是冻结日访问面，不把它写成开放 API 或可复现 artifact。

### 7.2 Oasis

- 项目页明确：开放 Oasis 500M code/weights，在线 demo 使用更大 checkpoint。
- GitHub README 提供 500M inference script 和 Hugging Face 下载命令；HF model card 可定位、需要同意条件后访问文件。
- 项目页的 20 FPS 是 official performance claim；4K/Sohu 是未来硬件设想，不是当前 500M checkpoint 的已验证能力。
- 项目主动列出 fuzzy distance、uncertain-object temporal inconsistency、domain generalization、inventory/object control 和 long-context failure。

### 7.3 DIAMOND、Matrix 与时钟

- DIAMOND 正式 NeurIPS 入口支持 mean HNS 1.46、CS:GO standalone engine 和 release；项目页支持约 10 FPS RTX 3090、381M CS:GO 模型与 repeated-jump exploitation。
- Matrix-Game 2.0 的 25 FPS/minute-level 来自作者 paper/project；开放链接存在，但正文不把数字称为独立复现。
- Matrix-Game 3.0 项目把 720p@40 FPS 的推理条件写为 8 GPUs for DiT + 1 GPU for VAE；正文保留该硬件条件。
- Matrix-Game 3.0 官方 repo 冻结日已有 inference code 和两个 Unreal first-person 5B 权重（base、distilled）；mixed Unreal+real-world 模型和 28B 模型仍写作 future release。
- Matrix-Game 3.5 GitHub API 最早 commit 为 2026-07-18，早于冻结日；repo 开放 code、first/third-person 5B base 与 three-step first-person distilled checkpoint，并说明 Wan2.2-TI2V-5B、Depth-Anything-3 和至少 40 GB VRAM 依赖。它是 project/technical-report 证据，不是正式 venue 或独立复现，也不回写成 3.0 的历史里程碑。
- DreamZero 的 7 Hz 是机器人 policy closed-loop rate，不与 20/24/40 render FPS 混表比较。

### 7.4 Memory 与开放面

- WorldPack 当前标准标题是 *Dynamic Frame Compression for Long-context Video World Modeling*；冻结日前版本为 v3。
- Infinite-World 标准标题完整包含 *1000-Frame Horizons via Pose-Free Hierarchical Memory*。
- ReWorld arXiv 为 2026-08-24 v1；repo 明写 inference code released、pretrained checkpoints on the way，两个核心权重均 Coming soon。
- ActWorld 保存 event-update 与 object-identity tokens，解决的是交互事件遗忘；不能用相机 loop-closure 指标完全替代。

### 7.5 正式 venue、benchmark 与用途

- UniSim 为 ICLR 2024 Outstanding Paper；GameNGen 为 ICLR 2025；Astra、Vid2World、WorldGym、World-In-World 为 ICLR 2026；WorldPlay、Infinite-World、iWorld-Bench 为 ICML 2026。正文不再把这些工作降写成仅 arXiv。
- Vid2World 的正式摘要支持 video diffusion architecture/objective causalization 与 causal action guidance；正文只把它作为预训练视频模型转 interactive model 的路径，不外推长期权威状态。
- iWorld-Bench 的正式摘要直接支持 330K clips、2.1K selected samples、4.9K tests、six task types 和 14 models。
- WorldRoamBench 把 action、vision、physics、memory 拆成四维，并用 controllability-gated physics 与 action-decoupled memory 避免错误动作掩盖物理/记忆分数。
- WorldGym 用 Monte Carlo world-model rollouts + VLM rewards 评估 robot policies；World-In-World 用统一 online planning/action API 与四个 closed-loop environments 报 task success。二者补足 decision utility，但仍是作者实验，不把 policy ranking 或成功率反推为逐状态全正确。
- 任何 benchmark 都只覆盖分层协议的一部分；正文没有将其总分写成真实 transfer 或安全证明。

### 7.6 K 步动作、双时钟与记忆权限

- 正文将预测条件从单一 $a_t$ 改为与未来 observation steps 对齐的动作 schedule $A_t=(a_t,\ldots,a_{t+K-1})$；原始 accepted-action event 使用独立时钟，相邻 schedule 项可由同一动作 hold 得到，必须冻结 hold duration 与覆盖规则。
- 视频 FPS 和动作接受时钟分开：最低报告 accepted-action Hz、hold/drop/coalescing 规则，以及从动作被接受到首个受影响显示帧的 p95 latency。
- 预测 rollout 只能写可回滚、带 provenance 的 speculative memory；真实观测或权威 engine state 才能写 authoritative memory。没有真实/权威分支时，不能把自写记忆称作 ground truth。
- 闭环 Mermaid 将 Human/Policy 标为提出/选择动作、Real Environment 标为执行者；预测与真实结果的 audit 只影响后续展示/训练/决策，不倒推为当前已执行动作的安全互锁。

## 8. AI 图生成、迭代与视觉验收

### 8.1 初始生成 prompt

使用内置 image generation 工具，brand-new generation，不传参考图。原始 prompt：

```text
Create a publication-quality scientific systems schematic in a clean vector-like infographic style, 16:9 landscape, light warm-white background, high contrast, colorblind-safe Okabe-Ito palette. Topic: “Interactive World Generation: Closed-Loop Stack”.

Audience: machine learning researchers. The figure must communicate an operational contract, not marketing.

Layout:
- A left-to-right main pipeline across the center:
  1) “Inputs” with three small items: “Observation”, “Action”, “Goal / Event”.
  2) “Causal World Core” with: “Action Encoder”, “Autoregressive / Diffusion Dynamics”, “State Predictor”.
  3) “Predicted Outputs” with: “Next Observation”, “State / Reward / Done”.
- Above the core, a distinct blue memory module titled “Persistent State & Memory” with exactly: “Recent Context”, “Entities & Events”, “Spatial / Landmark Memory”. Show dashed retrieve and update arrows between memory and the world core.
- Below the core, a distinct amber verification band titled “Evidence & Guardrails” with exactly: “Action Validity”, “Counterfactual Check”, “Uncertainty”, “Safety Gate”.
- At far right, an external green module titled “Real Environment” with: “Execute Action”, “Actual Observation”. Draw a solid feedback loop from the actual observation back to Inputs, clearly distinct from the predicted-output path.
- Add a small bottom-right telemetry box containing exactly: “FPS ≠ Action Hz ≠ Latency”.
- Use arrow legend at bottom-left: solid arrow = online causal flow; dashed arrow = memory read / update; dotted arrow = audit signal.
- Show dotted audit signals from Predicted Outputs and Real Environment to Evidence & Guardrails.
- Keep “Human / Policy” as a small actor above Inputs that proposes an action; do not make the world model itself look like a policy.
- Include a small branch from Causal World Core to two alternative candidate futures, both feeding “Counterfactual Check”, to show branching evaluation.

Visual requirements:
- crisp typography, correct spelling, large readable text, no tiny paragraphs;
- generous spacing and alignment, no overlaps or clipped edges;
- consistent rounded rectangles, subtle shadows, thin dark-gray strokes;
- avoid photorealism, 3D, gradients, decorative icons, robots, game characters, logos, watermarks, citations, or numeric performance claims;
- ensure the diagram remains intelligible in grayscale by combining color with borders, patterns, and labels;
- exactly preserve the requested English labels; no additional prose.
```

初始生成文件：

`generated_images/01a04ece-2192-7241-b4c7-03dc15c32d27/exec-492833bd-f3ab-474e-afed-67332245f6a0.png`

初检发现文字、布局、颜色和灰度均清楚，但 `Predicted Outputs` 的实线直接指向 `Real Environment`，可能误读为预测结果本身执行动作，未通过科学语义验收。

### 8.2 两轮定点编辑与独立审计

以初始生成图作为本地参考图，进行一次最小语义修正：

```text
Edit the provided scientific schematic while preserving its exact 16:9 layout, typography, labels, colors, modules, legend, and visual style. Make only the following causal-wiring correction, with generous routing space and no overlaps:

1. REMOVE the solid arrow from “Predicted Outputs” to “Real Environment”. Predicted outputs must never look as if they execute an action.
2. ADD a distinct solid action path originating at “Human / Policy” (or from the “Action” item within Inputs) and ending at “Execute Action” inside “Real Environment”. Route this line cleanly around the top or outside edge of the central modules; label it exactly “execute selected action”.
3. KEEP the solid feedback path from “Actual Observation” in “Real Environment” back to “Observation” in “Inputs”; label it exactly “observe”.
4. KEEP the model path from Inputs through Causal World Core to Predicted Outputs.
5. KEEP the dotted audit arrows from both “Predicted Outputs” and “Actual Observation” to “Evidence & Guardrails”.
6. KEEP candidate futures feeding “Counterfactual Check”.
7. Do not add any other arrows, labels, modules, prose, icons, or numeric claims.
8. Verify all existing labels remain correctly spelled, large, unclipped, and fully readable. Preserve the legend: solid = online causal flow, dashed = memory read / update, dotted = audit signal.
9. Ensure grayscale intelligibility through borders, arrow styles, and labels, not color alone.

The corrected semantics must be unambiguous: a human or policy proposes/selects an action; the learned world model predicts its consequences; the real environment independently executes the selected action; predicted and actual outcomes are compared by evidence/guardrails; the actual observation closes the real loop.
```

第一轮修订文件：

`generated_images/01a04ece-2192-7241-b4c7-03dc15c32d27/exec-b96a861e-a813-4f88-9011-ea619b9d9bdc.png`

该版已把预测与真实执行分开，但独立审计发现两个残余因果问题：底部 `Safety Gate` 实际只接收执行后的 audit signal，名称却暗示能阻断当前动作；Candidate Future A/B 到审计带仍画成实线。由于科学图应能脱离正文独立阅读，不能只靠 caption 豁免，故继续进行第二轮定点编辑。

第二轮编辑冻结以下约束：

```text
1. Replace the direct Human / Policy to Real Environment action path with:
   Human / Policy -> EXTERNAL PRE-EXECUTION INTERLOCK -> ALLOW -> Real Environment.
   Add a separate REJECT / FALLBACK branch that never enters Real Environment.
2. Rename the amber band to EVIDENCE & POST-HOC AUDIT.
3. Rename Safety Gate to POST-HOC SAFETY AUDIT.
4. Change Candidate Future A/B audit arrows from solid to dotted.
5. Preserve solid online flow, blue dashed memory flow, dotted audit flow,
   the legend, all other labels, and FPS ≠ Action Hz ≠ Latency.
```

接受版 artifact：

`generated_images/<thread>/exec-5fd66bd3-8677-4873-bce7-f59ef423487e.png`

项目交付路径：

`assets/diagrams/interactive-world-closed-loop-stack.png`

### 8.3 文件与验收记录

| 项 | 结果 |
|---|---|
| 格式 | PNG，8-bit/color RGB，non-interlaced |
| 尺寸 | 1672 × 941 px（约 16:9） |
| 文件大小 | 1,219,607 bytes |
| SHA-256 | `b30cdf8cfac6306b855ab5c48f386275aca14106557efcfa7e76dd0747863a79` |
| 原图回读 | 使用原始分辨率回读；无裁切、重叠、断字或异常符号 |
| 文字检查 | 人工逐项检查标题、模块、`EXTERNAL PRE-EXECUTION INTERLOCK`、`ALLOW`、`REJECT / FALLBACK`、`EVIDENCE & POST-HOC AUDIT`、`POST-HOC SAFETY AUDIT`、legend 与 `FPS ≠ Action Hz ≠ Latency`，拼写正确 |
| 语义检查 | Human / Policy 只提出/选择动作；外部 interlock 在真实执行前允许或拒绝；Real Environment 执行；world core 只预测；候选未来、预测与实际均以点线进入事后 audit；实际观测闭环回写。事后 audit 只影响后续展示/训练/决策，不冒充当前动作的 pre-execution interlock |
| 灰度检查 | 用 ImageMagick 转 8-bit grayscale 后原分辨率回读；实线/虚线/点线、边框和文字仍可区分 |
| OCR | 本机无 `tesseract`；未把缺失 OCR 当通过证据，采用人工逐项文字核对 |
| 可访问性 | 正文含描述性 alt、图注和 7 步顺序化文字替代；两张 Mermaid 均含 `accTitle`/`accDescr` |

## 9. 引用逐项核验账本

| # | 标准题名/入口 | 核验内容 | 状态 |
|---:|---|---|---|
| 1 | World Models / arXiv 1803.10122 | 题名、作者、2018、latent world-model framing | 通过 |
| 2 | Dream to Control / arXiv 1912.01603 | 题名、作者、ICLR 2020、latent imagination | 通过 |
| 3 | Learning Interactive Real-World Simulators / ICLR 2024 | UniSim、高/低层控制、sim-to-real author experiment、Outstanding Paper | 通过 |
| 4 | Genie / PMLR 235 | 11B、tokenizer、AR dynamics、latent action、ICML 2024 | 通过 |
| 5 | Video generation models as world simulators / OpenAI | 仅用于被动 video-generator framing | 通过 |
| 6 | Diffusion Models Are Real-Time Game Engines / ICLR 2025 | GameNGen、20 FPS、单 TPU、多分钟、29.4 PSNR | 通过 |
| 7 | DIAMOND / NeurIPS official proceedings | 标准题名、HNS 1.46、CS:GO engine、release | 通过 |
| 8 | Oasis project | architecture、20 FPS official claim、限制、500M vs demo | 通过 |
| 9 | etched-ai/open-oasis | inference code、500M 下载与使用路径 | 通过 |
| 10 | Etched/oasis-500m | checkpoint model card、gated file access、500M 边界 | 通过 |
| 11 | Genie 2 / Google DeepMind | 1 minute、10–20 s majority、counterfactual、memory、distillation trade-off、outtakes | 通过 |
| 12 | Genie 3 model / Google DeepMind | 720p、20–24 FPS、few minutes、limitations；无公开 paper/checkpoint/API | 通过 |
| 13 | Matrix-game 2.0 / arXiv 2508.13009 | 标准题名、1200 h、25 FPS、minute-level、open claim | 通过 |
| 14 | WorldPlay / ICML 2026 | Dual Action、Reconstituted Context、Context Forcing、720p/24 FPS | 通过 |
| 15 | Astra / ICLR 2026 | causal attention、noise history、action adapter、mixture experts | 通过 |
| 16 | EternalEvan/Astra | checkpoint、inference、training/open TODO 边界 | 通过 |
| 17 | WorldPack / arXiv 2512.02473 v3 | 当前题名、4→22 frames、16% overhead、冻结日 venue 边界 | 通过 |
| 18 | Infinite-World / ICML 2026 | 完整题名、HPMC、tri-state labels、30-min revisit、1000+ frames claim | 通过 |
| 19 | Matrix-Game 3.0 project/arXiv 2604.08995 | memory、自纠错、DMD、720p@40 FPS、8+1 GPU | 通过 |
| 20 | ActWorld / arXiv 2606.17730 | 100K data、action-aware memory、event/object tokens | 通过 |
| 21 | MultiWorld / arXiv 2604.18564 | multi-agent condition、global-state encoder、多视角 | 通过 |
| 22 | ReWorld / arXiv 2608.23565 | mixed windows、fixed KV、landmark bank、64 s/384 latents | 通过 |
| 23 | zhifeichen097/ReWorld | inference code、weights Coming soon、license | 通过 |
| 24 | iWorld-Bench / ICML 2026 | 330K/2.1K/4.9K、six tasks、14 models | 通过 |
| 25 | WorldRoamBench / arXiv 2606.31672 | action/vision/physics/memory、600+、10–60 s | 通过 |
| 26 | World Action Models are Zero-shot Policies / arXiv 2602.15922 | DreamZero、14B、joint video/action、7 Hz | 通过 |
| 27 | GeniWorld / arXiv 2608.06332 | URDF visual action、OOD generalization、policy evaluator | 通过 |
| 28 | Sekai2 / arXiv 2608.09449 | 2,826 h、trajectory/caption、982 loop/revisit sequences | 通过 |
| 29 | WAM tutorial / arXiv 2607.00836 v7 | 当前题名、tutorial 定位，不作 benchmark | 通过 |
| 30 | DIAMOND project | playable models、约 10 FPS RTX 3090、381M、multiple-jump failure | 通过 |
| 31 | Matrix-Game 2.0 project | GitHub/HF 链接、作者 benchmark 与开放面 | 通过 |
| 32 | Vid2World / ICLR 2026 | architecture/objective causalization、causal action guidance、三类域实验 | 通过 |
| 33 | WorldGym / ICLR 2026 | action-conditioned robot rollout、VLM reward、policy success/ranking | 通过 |
| 34 | World-In-World / ICLR 2026 | unified planning/action API、four closed-loop environments、task success | 通过 |
| 35 | Project Genie launch / Google 2026-01-29 | 美国 Ultra 18+ 首发、60 s、模型能力子集 | 通过 |
| 36 | Project Genie expansion / Google 2026-05-19 | 全球符合条件 Ultra 18+ 逐步扩展、Street View 美国地点先行 | 通过 |
| 37 | Project Genie / Google Labs Help | 当前 Ultra 访问、政策与支持边界 | 通过 |
| 38 | Matrix-Game 3.0 official repo | inference code、Unreal first-person base/distilled 5B 已发；mixed-real/28B 未发 | 通过 |
| 39 | Matrix-Game 3.5 official repo | 2026-07 日期、Patch Memory/Warped PRoPE、三项 5B 权重、依赖与 ≥40 GB | 通过 |

## 10. 验证命令与最终结果

冻结前在仓库根目录执行，结果如下。

| 验收项 | 命令/方法 | 结果 |
|---|---|---|
| Markdown | `npx --yes markdownlint-cli2` 检查正文与本日志 | `markdownlint-cli2 v0.23.2`、`markdownlint v0.41.1`；0 issues |
| 引用锚点 | Ruby 提取 `[[n]](#ref-n)` 与 `<a id="ref-n">`，检查缺失、未引用和重复 | 68 个引用实例；cited unique 39；defined 39；missing 0；uncited 0；duplicates 0 |
| arXiv 题名/ID | 从正文提取全部 arXiv ID，以 arXiv API 当前 Atom 元数据逐字比较链接题名 | 12/12 完全一致；批量 API 默认返回前 10 项，ActWorld 与 Sekai2 再单独查询通过 |
| 相对链接 | Ruby 解析正文与日志中的相对目标并在文件系统逐个解析 | 5 个实例均存在；missing 0 |
| 外部入口 | 提取 39 个唯一外链并复核一手页面；对更新的正式 venue、Google 与 GitHub 入口做 fresh-check | 39/39 均定位到一手入口；9 个新增/替换的 ICLR/ICML proceedings URL 经 `curl -L` 返回 200；Google/Matrix 页面经浏览器读取成功；无 404，未把局部 TLS 超时伪报成页面失效 |
| Mermaid | 提取 2 个 Mermaid block；`@mermaid-js/mermaid-cli` 配置本机 Google Chrome 后分别渲染 SVG 与 PNG，并原图回读 | 2/2 成功；SVG 为 31,455 B、42,377 B；PNG 为 710 × 1318、369 × 1258；文字、分支和箭头可读；两张 SVG 均保留 `<title>` 与 `<desc>` |
| PNG | `file`、ImageMagick `identify`/decode、字节数、SHA-256；另生成灰度副本原尺寸回读 | 1672 × 941、RGB 8-bit、non-interlaced、1,219,607 bytes；彩色与灰度均 decode exit 0；SHA 与 8.3 节一致 |
| 空白/补丁 | tracked 正文用 `git diff --check --`；新日志用 `git diff --no-index --check /dev/null …`，PNG 另做二进制解码 | 两个 Markdown 均无 whitespace diagnostic；`--no-index` 因“文件不同”按设计返回 1，不代表 whitespace error；PNG decode exit 0 |
| 作用域 | `git status --short` 与限定路径 `git diff --numstat` | 交互世界交付物为正文、研究日志与修订 PNG；仓库 coverage audit 在同一未提交批次中另行刷新；未修改导航，未 commit、未 push |

Mermaid CLI 首次使用默认 Puppeteer 配置时因本机缺少指定版本的 `chrome-headless-shell` 失败；改为显式指定已安装的 Google Chrome 后实际渲染成功。因此这里记录的是最终成功渲染，而不是仅做源码解析。

剩余不确定性有三项：本机缺少 `tesseract`，图中文字以原图人工逐项回读替代 OCR；少数 Google/DeepMind 页面在命令行检查中会 TLS 超时，因此外链结论同时依赖浏览器一手页打开与 API/proceedings 交叉核验；厂商/作者给出的 FPS、分辨率、horizon 与 Matrix 3/3.5 repo 能力未在统一硬件独立复现，正文始终保留 claim 主语和证据等级。
