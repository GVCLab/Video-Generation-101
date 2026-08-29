# 视频 World Model 与 World Action Model 研究日志

## 1. 任务、范围与截止时间

- 检索日期：**2026-08-30（Asia/Shanghai）**
- 对应章节：[从视频生成器到 World Action Model](../docs/world-models.md)
- 研究问题：
  1. video generator、predictive world model、WAM、controllable simulator、policy、environment model 的最小边界是什么？
  2. deterministic/stochastic latent dynamics、RSSM、Dreamer/TD-MPC 与现代 video world model 如何衔接？
  3. cascaded planner/policy + generator 与 joint action–video model 的机制和证据怎样比较？
  4. verifier、persistent state、counterfactual、uncertainty 和 receding-horizon loop 怎样组成可验收闭环？
  5. 2025–2026 的 WAM、memory 和 evaluation 工作有哪些可核验 artifact？

本日志是面向技术综述的 structured search，不宣称为医学式系统综述。为避免虚假的精确性，同一论文在 arXiv、OpenAlex、会议页、项目页和代码仓库间只作为一项工作计入证据，不把跨库重复项强行汇总成一个 PRISMA 数字。

## 2. 技能与研究协议

执行前完整读取并遵循：

- literature-review：多库检索、纳排标准、citation chaining、逐项核验；
- scientific-schematics：先定义信息层级，再生成、目视检查、灰度检查和记录 provenance。

结论优先级为：正式会议/期刊入口 → arXiv 原文 → 作者项目/官方代码 → 机构发布。二手文章只可发现线索，不进入结论证据。

## 3. 数据源、检索式与结果数

### 3.1 OpenAlex

通过 OpenAlex Works API，以标题检索并限制 publication date 为 2018-01-01 至 2026-08-30。

| 检索式（title.search） | 返回总数 | 实际使用方式 |
|---|---:|---|
| world model robotics | 391 | 发现机器人 survey、policy evaluation 与 transfer 线索 |
| world action model | 608 | 按 newest 检查前 25 项，再做 exact-title 核验 |
| interactive world model | 152 | 按 newest 检查前 25 项，重点筛 memory 与 controllable simulator |
| video world model | 379 | 查 DreamGen、evaluation、长期 rollout 与同名工作 |

OpenAlex 用于发现、日期/DOI/venue 交叉检查，不单独承担机制或性能数字的证据。

### 3.2 arXiv

通过 arXiv API 查询标题字段；结果数为检索日 API 的 totalResults。

| arXiv query | 返回总数 | 筛选用途 |
|---|---:|---|
| ti:"world model" AND all:robot | 442 | 机器人 world model、WAM 与评测 |
| ti:"world action model" | 131 | WAM 定义、DreamZero、WAV 与同名项 |
| ti:"interactive world model" | 25 | Genie/Oasis 类交互系统与 long-horizon memory |
| ti:"video world model" | 72 | DreamGen、WorldPack、WorldEval、RoboWM-Bench |

对章节点名工作进一步使用 exact-title 或 arXiv ID：2607.00836、2605.00080、2604.01985、2505.12705、2511.07732、2602.15922、2512.02473、2602.02393、2608.23565、2601.12428、2505.19017、2506.00613、2605.29360、2604.19092。

### 3.3 正式会议、作者项目与代码

| 来源类型 | 核验对象 |
|---|---|
| PMLR / official proceedings | PlaNet、Genie、DreamGen |
| OpenReview | Dreamer、TD-MPC2、GameNGen、UniSim、ViPRA |
| NeurIPS proceedings | Action-Conditional Video Prediction、DIAMOND |
| 作者项目页 | Genie、Oasis、DreamGen、WAV、ViPRA、DreamZero、Infinite-World |
| 官方代码仓库 | DreamGen/GR00T-Dreams、WAV、ViPRA、DreamZero、DIAMOND、Oasis、ReWorld |
| 机构发布 | Sora、Genie 2、Genie 3、Cosmos；只作为 release/demo 级证据 |

至少三类互补一手来源已覆盖：**arXiv 原文、OpenAlex 元数据、正式会议入口、作者项目与官方代码**。

## 4. 纳入、排除与去重

### 4.1 纳入

1. 提出动作条件 world dynamics、latent planning、video world model、WAM、persistent memory 或 decision-utility evaluation 的一手工作。
2. 能改变本章机制分类或证据边界的 foundational paper。
3. 2025–2026 被任务明确点名，且可定位 arXiv、正式会议、作者项目或代码的工作。
4. 产品/机构发布只有在没有等价论文且能说明 release surface 时纳入，并降为 C 级。

### 4.2 排除

1. 二手榜单、媒体转述、SEO 聚合页和无可追溯数字。
2. 只做普通 video generation、没有 action/state/planning 联系的工作。
3. 名称含 action model 但实际与环境转移、机器人动作或互动无关的工作。
4. 同一论文的 arXiv 与会议版本不重复计为两项结论。
5. demo 中可见但 paper/checkpoint 未说明的能力，不写成模型通用能力。
6. “开源”若只有网页或占位 repo、没有 code/weights/license，则不按完整 artifact 纳入。

### 4.3 实际筛选

- 对 OpenAlex “world action model”与“interactive world model”分别检查 newest 前 25 项的标题、摘要和入口，共 50 个高新鲜度记录。
- 对任务点名工作逐项 exact-title 检索，并沿参考文献/项目页做 backward/forward chaining。
- 章节最终保留 33 个参考入口；其中 survey/tutorial 仅用于 taxonomy，数值优先回到原始论文。
- 没有给出“总排除 N 项”的伪精确数字，因为四个检索式重叠严重，且同一工作跨 arXiv、OpenAlex、会议与项目页多次出现。

## 5. 证据等级

| 等级 | 定义 | 可支持的表述 |
|---|---|---|
| A | 正式同行评审入口，且有可定位 primary artifact 或 evaluation | 可写 venue、机制和论文实验；仍保留任务边界 |
| B | arXiv + 作者项目或官方代码 | 可写作者报告与开放面，不称独立复现 |
| C | 机构/作者 release 或 demo，无同等级论文/checkpoint | 只写“官方展示/报告”，不写通用能力 |
| D | 二手检索线索 | 只用于发现，不进入正文结论 |

等级衡量可核查性，不是模型能力排名。

## 6. 核心证据账本

| 工作 | 一手入口 | 用于支持 | 关键边界 | 等级 |
|---|---|---|---|---|
| World Models / PlaNet / Dreamer | arXiv、PMLR、OpenReview | latent dynamics、RSSM、imagination control | 小环境与 model exploitation | A |
| MuZero | Nature | task-relevant dynamics 不必重建像素 | 不等于视觉 simulator | A |
| TD-MPC2 | OpenReview + project/code | multi-task latent MPC | 不是开放世界单 checkpoint | A |
| Genie | PMLR | action-free video 中学习 latent action | latent action 不等于机器人控制量 | A |
| GameNGen | OpenReview | action-conditioned diffusion 达实时交互 | 单一游戏，FPS 不等于 action latency | A |
| DIAMOND | NeurIPS + code/demo | diffusion world model 可训练 agent；暴露规则利用 | learned simulator 中高分可能不迁移 | A |
| Oasis | 作者项目 + code/500M weights | interactive Minecraft-style world | 开放 500M 与更大在线 demo 分开 | B/C |
| DreamGen | CoRL/PMLR + project + GR00T-Dreams | world model → synthetic video → pseudo-action → policy | 离线数据路线，不是在线 planner | A |
| WAM tutorial | arXiv 2607.00836 v7 | WAM 定义与四范式 | 教程不是实验基准 | B |
| World Action Verifier | arXiv 2604.01985 v2 + project/code | state plausibility + action reachability | workshop 荣誉，不是 ICLR 主会；非通用安全证书 | B |
| ViPRA | ICLR 2026 OpenReview + project/code/weights | future visual + latent action joint pretraining | 22 Hz 能力与约 3.5 Hz 实验设置分开 | A |
| DreamZero | arXiv + project/code | joint video–action backbone 与真机闭环 | zero-shot 仅限论文设定 | B |
| WorldPack | arXiv 2512.02473 v3 | dynamic frame compression + geometry selection | 当前标题已改变；压缩不是实体记忆 | B |
| Infinite-World | arXiv v2 + project | pose-free hierarchy、tri-state action uncertainty、1000+ frames | 未发现等价完整代码/权重 | B/C |
| ReWorld—memory | arXiv 2608.23565 + repo | mixed windows、fixed KV、landmark bank | weights “Coming soon” | B- |
| ReWorld—reward | arXiv 2601.12428 | 235K+ preference、四维 HERO reward | 与 long-memory ReWorld 是不同工作 | B |
| WorldGym / WorldEval / MiraBench / RoboWM-Bench | arXiv primary papers | policy ranking、action fidelity、optimism、物理约束仿真执行 | 任何单一 benchmark 都不覆盖 L0–L6 | B |

## 7. 数字与命名的 fresh-check

### 7.1 World Action Verifier

- 当前标题：*World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry*。
- 机制：state plausibility 与 action reachability 分解；action-free subgoal、稀疏 inverse model、forward rollout cycle。
- 作者实验：9 个 MiniGrid、RoboMimic、ManiSkill 任务；约 2× sample efficiency，超过 22% downstream improvement。
- Venue 边界：项目页列出 ICLR 2026 World Models workshop outstanding paper 与 Recursive Self-Improvement workshop spotlight；未把它写成 ICLR main-track paper。

### 7.2 DreamGen

- 正式入口：CoRL 2025，PMLR 305。
- 官方 GR00T-Dreams repo 当前覆盖 world-model finetune、synthetic video generation、IDM/latent pseudo-action 与 policy training。
- 正文只用它支持“离线合成数据改善 policy”的路线，不推断在线 MPC。

### 7.3 ViPRA

- 标准标题：*ViPRA: Video Prediction for Robot Actions*。
- 正式入口：ICLR 2026 OpenReview forum w3Ik8HUyTT。
- 机制：future visual observation + motion-centric latent action；flow decoder 输出连续动作块。
- 性能表述：论文讨论最高 22 Hz 能力，但真实实验最高约 3.5 Hz；action chunk 14，执行 7 步后重规划。两种数字分开记录。

### 7.4 DreamZero

- 14B autoregressive video diffusion backbone；联合未来视频与动作。
- 作者报告约 7 Hz closed-loop 和 38× inference speedup。
- 新真实观测回写 KV cache；“zero-shot”不外推到任意机器人或任务。

### 7.5 WorldPack

- 当前 arXiv v3 标题：*WorldPack: Dynamic Frame Compression for Long-context Video World Modeling*。
- 2 帧不压缩 + 4 帧 4× + 16 帧 16×，在约 4-frame token budget 中暴露 22 帧历史（5.5×）。
- 论文报告 trajectory packing 约增加 16% diffusion inference time；geometric selection 另有成本。

### 7.6 Infinite-World

- 标准标题：*Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory*。
- HPMC hierarchical pose-free memory compressor。
- tri-state uncertainty-aware action labels 与 30-minute revisit-dense finetuning。
- “1000+ frames”按作者报告表述；未把项目 demo 写成独立复现。

### 7.7 ReWorld 名称冲突

1. arXiv:2608.23565：*An Interactive World Model with Long-Horizon Memory*；mixed per-head windows、random routing、fixed KV cache、pose-indexed landmark bank、chunk-drop training。作者报告 704×1280 streaming 与固定 12-chunk cache 下 64 秒/384 latents。repo 有 inference code，weights 在检索日仍为 “Coming soon”。
2. arXiv:2601.12428：*Multi-Dimensional Reward Modeling for Embodied World Models*；235K+ preference data、HERO 四个 reward heads 与 HERO-FPO。

章节始终使用完整副标题或功能括注，不把二者合并。

### 7.8 Demo 与 checkpoint

- Oasis：公开 500M artifact 与更大在线 demo 分开。
- Genie 2/3：按 Google DeepMind official release/demo 级证据，不写成已开放论文/checkpoint。
- Sora“world simulator”是技术报告中的研究 framing，不作为动作条件或 planning 证据。

### 7.9 提交前一手页面复核

独立审计发现初稿存在“arXiv ID 正确、标题改写或错配”的阻塞问题。2026-08-30 重新打开 arXiv 一手页面后：

- 2604.01985 修正为 *World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry*；正文机制与 2× sample efficiency、超过 22% 下游提升和 9-task 范围均由当前摘要支持。
- 2602.02393 补全为 *Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory*。
- 2506.00613 修正为 *WorldGym: World Model as An Environment for Policy Evaluation*；删除原先错配的“分布内低估、OOD 高估”转述，改为摘要明确支持的 Monte Carlo rollout、VLM reward、真机相关性与 policy ranking。
- 2605.29360 修正为 *MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models*。
- 2604.19092 修正为 *RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation*；明确其执行环境是 physically grounded simulation，不写成真机执行。
- 2511.07732 一并修正为复数标题 *ViPRA: Video Prediction for Robot Actions*。

## 8. 章节综合判断

1. deterministic/stochastic 是未来表示选择；是否有 action intervention 和 external utility 是另一条轴。
2. 级联路线更容易插入 verifier、uncertainty 与安全约束；联合路线降低接口开销，却需更强消融。
3. IDM 可恢复伪动作或检查可达性，但不等同 planner；planner 仍负责候选搜索与任务优化。
4. persistent memory 必须支持状态覆盖、地址化检索与置信度，而不只是长 KV。
5. FPS、action Hz 与 sense-to-act latency 必须分开。
6. 最终证据阶梯是 open-loop perceptual → state/action fidelity → counterfactual → planning utility → real-world transfer。

## 9. AI 科学示意图生成记录

### 9.1 文件与 provenance

- 生成方式：OpenAI image generation，经 scientific-schematics 信息层级约束；未使用第三方图片素材。
- 初始生成源：本次会话的 image generation 输出。
- 最终生成源：
  `generated_images/01a04ece-2192-7241-b4c7-03dc15c32d27/exec-2f63b7df-c5b0-434d-ac56-5bdf9a95c9ce.png`
- 项目文件：[world-action-model-dual-route.png](../assets/diagrams/world-action-model-dual-route.png)
- 尺寸：**1672 × 941 px**
- 色彩：RGB，8-bit，non-interlaced PNG
- SHA-256：`7535851726699655b69303874f23aba109fc1374c1af479bede410d12b3cff49`

### 9.2 首次生成 prompt

~~~text
Create a publication-quality scientific systems diagram for a Chinese technical textbook, 16:9 landscape, clean light background, high contrast, colorblind-safe Okabe–Ito palette, professional sans-serif typography, generous white space, no logos, no decorative robots, no benchmark numbers, no fake citations. Title at top: “World–Action Modeling: Two Routes, One Closed Loop”. Use two clearly separated vertical panels feeding one shared bottom closed-loop band.

LEFT PANEL header: “A. Cascaded: Imagine / Evaluate → Act”. Flow top-to-bottom with compact labeled blocks: “Observation + Goal” → “World / Video Predictor” → branching “Candidate Futures” → “Planner / Reward / IDM” → “Action Chunk”. Show a dashed secondary path “Offline synthetic trajectories → Policy training”. Add a small warning badge: “Module errors compound”.

RIGHT PANEL header: “B. Joint: Predict Future + Action”. Flow top-to-bottom: “Observation + Goal + Proprioception” → large shared block “Joint Video–Action Backbone” → two aligned outputs “Future State / Video” and “Executable Action Chunk”. Connect them with an alignment bracket labeled “shared predictive objective”. Add a small warning badge: “Alignment is not causal proof”.

BETWEEN PANELS place a narrow verification column labeled “Action Verifier” with two checklist items: “State plausibility” and “Action reachability”; show it can score candidate futures/actions from both routes.

BOTTOM BAND across full width: “Receding-Horizon Closed Loop”. Sequence: “Roll out K candidates” → “Score task value − uncertainty − verifier penalty” → “Execute first action chunk only” → “Observe real world” → “Re-anchor memory / state” → curved arrow back to both panel inputs. Beneath it, show three persistent-state tokens as small boxes: “Recent context”, “Entity / spatial memory”, “Uncertainty”.

Use solid arrows for data flow, dashed arrows for training-only/offline flow, and a tiny legend. Make every requested label spelled exactly. Ensure no cropped text, no overlapping arrows, balanced alignment, readable at 100% and in grayscale. The visual message must be that neither route is complete without verification and real-world feedback.
~~~

### 9.3 迭代与修复 prompt

首次图的 offline synthetic trajectories 箭头方向错误，可能被读成“离线轨迹训练 predictor”。目视发现后只改该连接：

~~~text
Edit this scientific diagram while preserving every other label, layout, color, and connection exactly. Fix only the LEFT PANEL offline training branch so its direction is scientifically correct: draw a dashed gray arrow FROM the “World / Video Predictor” block TO the “Offline synthetic trajectories” box, then a dashed gray arrow FROM “Offline synthetic trajectories” DOWN TO “Policy training”. Remove the current incorrect dashed arrow that points from “Offline synthetic trajectories” into the predictor. Keep the main online flow Observation + Goal → World / Video Predictor → Candidate Futures → Planner / Reward / IDM → Action Chunk unchanged. Do not alter any text, do not add elements, do not crop anything, and retain the full 16:9 canvas and bottom legend.
~~~

### 9.4 视觉检查

- 原图按 1672 × 941 原分辨率查看：标题、两条路线、verifier、底部 loop、persistent-state 标签均清晰。
- 箭头：online 主路径方向正确；offline 分支为 Predictor → synthetic trajectories → Policy training。
- 布局：无裁切、无块间重叠、无箭头穿过正文；左右面板对齐。
- 文字：所有指定英文标签可读，无乱码；IDM 在正文另行解释为伪动作恢复/候选可达性模块，不等同 planner。
- 灰度：转换为临时灰度图后检查，橙/蓝分区仍可通过位置、边框与标题区分。
- 可访问性：章节提供完整 alt text、图注和 4 步顺序化文字替代。

## 10. 交付验证记录

验证日期：2026-08-30。

| 检查 | 命令/方法 | 精确结果 |
|---|---|---|
| markdownlint | `npx --yes markdownlint-cli@0.47.0 docs/world-models.md sources/research_20260830_video_world_action_models.md` | exit 0；0 条错误 |
| 引用锚点 | Ruby 扫描 `<a id="...">` 与 `](#...)` | 33 个定义、50 次引用；missing 0，unused 0 |
| 相对链接 | Ruby 按各 Markdown 文件目录解析目标 | 章节 9 个、日志 2 个，共 11 个；missing 0 |
| Mermaid 语法与实际渲染 | `@mermaid-js/mermaid-cli@11.12.0` + 本机 Google Chrome | 2/2 成功；PNG 分别为 784×133、784×150 |
| Mermaid 视觉回读 | 原始 PNG 查看 | 2/2 无裁切、无节点重叠，箭头与标签可读 |
| 图像结构 | `file` + `shasum -a 256` | 1672×941 RGB PNG；SHA 与 9.1 一致 |
| diff whitespace | `git diff --check -- docs/world-models.md sources/research_20260830_video_world_action_models.md assets/diagrams/world-action-model-dual-route.png` | exit 0；无输出 |
