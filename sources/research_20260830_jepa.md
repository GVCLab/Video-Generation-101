# JEPA / latent predictive world model research audit

## 0. 审计范围

- **冻结日：** 2026-08-30（Asia/Shanghai）。
- **输出：** 支持 `docs/jepa.md` 的技术重写；主线整合阶段另加入一张经过语义回读的教学 PNG。不改 coverage audit。
- **问题边界：** 精确定义 JEPA 训练合同；区分 representation learner、action-conditioned latent dynamics、planner 与 pixel generator；核验 2026 年版本和公开工件；建立可证伪评测。
- **相邻页面只读：** `docs/world-models.md` 与 `docs/tasks/action-conditioned-prediction.md` 用于术语对齐。后者的“同一状态改变动作”与“闭环效用”被保留，但本页补上 target encoder、EMA、energy、MPC 和 release surface 的细节。

## 1. 证据等级与使用规则

| 等级 | 来源 | 本次用途 |
|---|---|---|
| E1 | 正式 proceedings / OpenReview venue 论文 | 机制、实验协议、正式接收状态 |
| E2 | 作者 arXiv 论文或 working paper | 2025–2026 新工作、公式、消融和限制 |
| A1 | 作者 / 机构官方项目页、博客 | release 声明、作者口径、部署演示边界 |
| A2 | 官方仓库、checkpoint、model card | 实际代码、权重、配置、许可证和版本面 |
| E3 | 独立第三方论文 | 只作邻近基线，不用来代替被评方法的一手证据 |
| X | 搜索摘要、聚合站、媒体转述 | 只用于发现，不进入技术结论 |

技术问题只由 E1 / E2 / A1 / A2 支撑。Meta 博客中的机器人成功率按“作者报告”处理，并用论文协议复核；GitHub / Hugging Face 只证明工件存在，不证明论文主张正确。

## 2. 检索式与筛选过程

### 2.1 广检索

2026-08-30 运行 arXiv API，按 submitted date 倒序取前 100 条并人工看标题、摘要与作者：

```text
search_query=all:"joint embedding predictive architecture"
sortBy=submittedDate&sortOrder=descending&max_results=100

search_query=all:JEPA
sortBy=submittedDate&sortOrder=descending&max_results=100
```

当次索引分别返回总量 237 与 402；总量会随 arXiv 索引变化，只作为检索覆盖记录。广检索用于发现 2026 年候选，随后回到精确 arXiv ID、作者仓库和正式 venue 核验。

网页 / 站点检索式：

```text
site:openaccess.thecvf.com I-JEPA CVPR 2023
site:openreview.net "Revisiting Feature Prediction" V-JEPA TMLR
site:ai.meta.com/research V-JEPA 2 planning
site:ai.meta.com/blog V-JEPA 2 world model benchmarks
site:github.com/facebookresearch vjepa2 V-JEPA 2.1
site:huggingface.co/facebook "V-JEPA 2"
site:openreview.net TD-JEPA ICLR 2026
site:openreview.net EB-JEPA World Models Workshop
site:arxiv.org Branch-JEPA
site:arxiv.org Var-JEPA
site:github.com LeVJEPA VideoMix Large
```

精确条目复核：

- `2301.08243` I-JEPA
- `2307.12698` MC-JEPA
- `2404.08471` V-JEPA
- `2411.04983` DINO-WM
- `2506.09985` V-JEPA 2 / 2-AC
- `2510.00739` TD-JEPA
- `2511.08544` LeJEPA
- `2602.03604` EB-JEPA
- `2603.14482` V-JEPA 2.1
- `2603.19312` LeWorldModel
- `2603.20111` Var-JEPA
- `2607.05238` Branch-JEPA
- `2608.27395` LeVJEPA

### 2.2 纳入标准

1. 直接定义或实证检验 JEPA、energy-based joint embedding、动作条件 latent dynamics、collapse regularization、dense feature 或多未来。
2. 能落到可核验公式、训练信息流、消融、控制协议或公开工件。
3. 2026 年新条目必须在冻结日前公开，且标注 preprint / workshop / conference 状态。
4. 相邻方法只有在构成必要对照时纳入，例如 DINO-WM 证明“预训练 visual feature + latent dynamics + planning”不是 JEPA 专属。

### 2.3 排除标准

- 标题含 JEPA 但属于语音、生物、金融等跨领域移植，且不改变本页视觉 / 动作技术结论。
- 只有搜索摘要、二手媒体或 benchmark 聚合，没有论文或作者工件。
- 只有 image / video probe，没有动作干预或闭环证据，却把结果称作控制型 world model。
- 只有像素可视化 decoder，却将其误写成基础 JEPA predictor。
- 只有 best-of-K / oracle sample，却没有 proper score、分支利用率或校准的多模态主张。
- 冻结日之后的版本、commit 或产品声明。

## 3. 一手来源账本

| 来源 | 等级 | 本页提取的证据 | 不外推的内容 |
|---|---|---|---|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | E2 | 分层 world model、actor、cost、memory 的蓝图 | working paper 不是完成系统或 benchmark |
| [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) | E1 | energy 是未归一化兼容性标量；推理可最小化 energy | energy 不等于已校准概率 |
| [I-JEPA CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | E1 | context / target blocks、EMA teacher、非像素预测 | 静态图像结果不等于 dynamics |
| [V-JEPA OpenReview](https://openreview.net/forum?id=QaCCuDfBk2) / [arXiv](https://arxiv.org/abs/2404.08471) | E1 | TMLR 接收；时空 mask、L1 target loss、点预测分析、独立 diffusion decoder | probe 不等于 action model；decoder 不属于预训练 predictor |
| [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) | E2 | action-free V2 与 V2-AC 分工；DROID 数据；多步 latent loss；CEM / MPC；机器人协议和限制 | “zero-shot”不等于没见过同类 embodiment |
| [Meta V-JEPA 2 release article](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | A1 | >1M 小时预训练、<62h DROID、作者报告的 pick / place 范围 | 新闻口径不代替论文表格与样本数 |
| [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482) | E2 | dense context loss、deep self-supervision、multi-modal tokenizer、dense / semantic 消融、导航生成头 | 2.1 encoder 本身不是 action predictor |
| [LeJEPA paper](https://arxiv.org/abs/2511.08544) | E2 | SIGReg、单 encoder、无 stop-gradient / EMA 的训练路线、论文假设下保证 | 保证不无条件覆盖所有数据和控制任务 |
| [LeWorldModel paper](https://arxiv.org/abs/2603.19312) | E2 | 像素到 action latent 的端到端训练、SIGReg、CEM / MPC、状态与 surprise probes | 小控制协议不等于 foundation-scale 泛化 |
| [EB-JEPA OpenReview](https://openreview.net/forum?id=ZVAMdXGCUC) | E1 | ICLR 2026 World Models Workshop；单卡教程、图像 / 视频 / Two Rooms | 不是 ICLR main-track 大规模 benchmark |
| [TD-JEPA OpenReview](https://openreview.net/forum?id=SzXDuBN8M1) | E1 | ICLR 2026；reward-free offline transition、policy-conditioned 长期 latent dynamics、13 数据集 | 不是视频生成或机器人部署 |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | E3 | frozen DINOv2 patch feature + action dynamics + visual-goal planning，六环境 | 不是 JEPA 预训练证据 |
| [LeVJEPA](https://arxiv.org/abs/2608.27395) | E2 | 2026-08-27 提交；SIGReg、random token dropping、block-causal attention、作者 compute 报告 | 距冻结日仅 3 天，尚无正式 venue 和控制结果 |
| [Branch-JEPA](https://arxiv.org/abs/2607.05238) | E2 | 有权重的有限 latent successor set；Energy Score | 新预印本，不能据此声称通用视频多未来 |
| [Var-JEPA](https://arxiv.org/abs/2603.20111) | E2 | 变分 / ELBO 形式的 latent uncertainty | 冻结日实证主要为表格设置，不外推视频或控制 |

## 4. 逐项技术核验

### 4.1 Context、target、mask、stop-gradient 与 EMA

- I-JEPA / V-JEPA 的 context encoder 只读取可见 token；target encoder 读取未遮挡样本后抽 target 位置。
- target 分支的 `stop-gradient` 阻断反向传播。在线 encoder 与 predictor 由损失更新，target encoder 由在线权重 EMA 更新。
- mask / target query 提供位置，不提供内容。I-JEPA 强调足够大的 target blocks 与空间分散 context；V-JEPA 将短、长时空 block mask 的 union 沿时间重复，平均遮挡约 90%。
- 损失只在 target token 上计算。若 context 路径读取 target 像素，即使 loss 很好也属于信息泄漏。
- EMA、stop-gradient 和 predictor asymmetry 是经过实证的 recipe；本页没有把它们写成单独的数学防坍塌保证。

### 4.2 非生成式边界

V-JEPA 论文明确把基础方法称为 non-generative feature prediction。像素可视化流程是：

1. 冻结训练好的 context encoder 和 predictor；
2. 对被遮挡区域得到预测 feature；
3. 另训 conditional diffusion decoder；
4. decoder 只负责从 feature 采样可能像素。

因此“V-JEPA 可视化出视频”不能改写成“V-JEPA predictor 生成 RGB”。Hugging Face 的 V2 encoder card同样把用途写成 representation、classification / retrieval 等 feature use，而非 pixel generation。

### 4.3 确定性点预测

V-JEPA 对 L1 目标给出条件中位数的点预测解释。这里支持的结论是“基础目标没有显式多模态分布”，而不是“表示一定模糊”。Branch-JEPA、Var-JEPA、V2.1 下游 diffusion 分别对应有限集合、变分 latent 和生成式轨迹头；三者的证据与归属保持分开。

### 4.4 V-JEPA 2-AC 的动作与规划

论文协议核对：

- action-free V2 encoder 与 action-conditioned predictor 分开；
- V2-AC 冻结 ViT-g visual encoder，训练约 300M predictor；
- 数据少于 62 小时 DROID，4 秒 clip、4 fps、256×256；动作 / state 为 7D end-effector 表示；
- predictor 使用 block-causal attention，训练有 teacher forcing 与两步 rollout；
- 规划 energy 是预测终点 latent 到 visual-goal latent 的 L1；
- CEM 采样并更新动作分布，只执行第一步，再读取真实观测重规划。

机器人表格按两个实验室、每任务 10 次试验汇总。作者报告 reach 100%，cup / box grasp 65% / 25%，cup / box pick-place 80% / 65%。这里没有计算额外置信区间，也没有把 10 次协议外推成跨实验室总体概率。

“zero-shot”按论文限制为：没有部署环境 / 对象 / 任务特定训练数据与奖励；但训练数据来自 DROID 同类 Franka embodiment，且组合任务使用人工视觉子目标。正文据此删除了“完全没见过机器人 / 无人工任务结构”等过强表述。

### 4.5 V-JEPA 2.1 的 dense / semantic 消融

论文消融记录：

| 配置变化 | ADE20K mIoU | Something-Something-v2 |
|---|---:|---:|
| 基线 masked loss | 22.2 | 72.8 |
| + context loss | 33.8 | 62.5 |
| + deep self-supervision | 38.6 | 72.1 |

该表支持“dense 与 temporal semantic 存在权衡，deep supervision 恢复平衡”，不支持“context loss 对所有任务单调提升”。

机器人对照还存在归因混杂：同 horizon 1 / sample budget 下，2.1 路线的 grasp 从 60% 到 70%；调到 horizon 8、300 samples 后为 80%。摘要的 +20 points 合并了 encoder / retraining 与 planner setting 变化，正文已明确不能全部归因给 backbone。

V2.1 导航使用 conditional diffusion Transformer 在 representation 上生成轨迹。多模态属于下游 trajectory model；基础 2.1 JEPA 仍是确定性 feature learner。

### 4.6 Collapse 与动力学可辨识性

- LeJEPA 的 SIGReg 通过随机一维投影的 characteristic-function 匹配，使 embedding 接近各向同性高斯；正文只写“在论文假设下保证”。
- LeWorldModel 联合训练 encoder 与 action predictor，使用 prediction + SIGReg，无预训练 encoder、EMA teacher 或 stop-gradient。论文限制中提到：高维 SIGReg 目标在简单数据上可能弱化，紧凑表示可能漏掉控制信息。
- EB-JEPA 的 Two Rooms 消融显示，动作设置还需 inverse-dynamics 类约束来保持可规划信息；非坍塌不等于动作可辨。

### 4.7 评测证据不跨级

本次把以下协议独立：

1. frozen linear / attentive probe：表征可读性；
2. partial / full finetune：适配性与训练预算；
3. held-out latent prediction：点预测准确性与 rollout 误差；
4. paired action intervention：动作敏感性；
5. state readout：位置、速度、接触和对象状态充分性；
6. fixed-budget closed-loop MPC：真实效用；
7. controlled OOD：规律迁移；
8. proper score / calibration：不确定性。

任何第 1 项成绩都不能替代第 4–8 项。页面中的最小实验给出预注册示例门槛，并明确这些不是论文通用阈值。

## 5. 版本与公开工件核验

### 5.1 Git HEAD 快照

2026-08-30 执行 `git ls-remote <official-repo> HEAD`：

| 项目 | 官方仓库 | HEAD |
|---|---|---|
| I-JEPA | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` |
| V-JEPA | [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` |
| V-JEPA 2 / 2-AC / 2.1 | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `204698b45b3712590f06245fbfba32d3be539812` |
| LeJEPA | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) | `c293d291ca87cd4fddee9d3fffe4e914c7272052` |
| LeWorldModel | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` |
| EB-JEPA | [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | `966e61e9285b3a876f49b9774e9720d9a99a7925` |
| LeVJEPA | [MLO-lab/LeVJEPA](https://github.com/MLO-lab/LeVJEPA) | `941526c428aa513a5bdfa38697724fda794d8496` |

GitHub API 在连续请求后出现 rate limit 403；未用缓存字段补猜，最后三个 HEAD 改用 `git ls-remote` 验证。commit 是冻结日远端指针，不保证以后不变。

### 5.2 Release surface

| 项目 | 论文 | 训练 / 评测代码 | 权重与配置 | 数据 | 许可证注意 |
|---|---|---|---|---|---|
| I-JEPA | CVPR | 有 | 多个 checkpoint / log | 数据说明 | 仓库 archived；API SPDX 为 NOASSERTION，页面不猜许可证 |
| V-JEPA | TMLR | 有 | checkpoints / configs | 数据说明 | API SPDX 为 NOASSERTION；不猜许可证 |
| V-JEPA 2 | arXiv | 有 | 300M、600M、1B、1B@384；probe | 数据配方 | 仓库称主体 MIT，特定文件 Apache-2.0 |
| V-JEPA 2-AC | 同 V2 论文 | 有 | checkpoint、config、PyTorch Hub | DROID recipe | 同仓库条款 |
| V-JEPA 2.1 | arXiv `2603.14482` | 同仓库有 | 80M、300M、1B、2B@384 直链 | VisionMix163M 描述 | 同仓库条款 |
| LeWorldModel | arXiv | 有 | checkpoint / data links | 公开链接 | 仓库标 MIT |
| EB-JEPA | workshop | 有 | 教程示例 | 示例数据 | Apache-2.0 |
| LeVJEPA | 新 arXiv | 有 | VideoMix-Large card | 训练说明 | repo 总体 MIT；适配文件与权重条款另列 |

官方 Hugging Face [V-JEPA 2 collection](https://huggingface.co/collections/facebook/v-jepa-2) 在冻结日列出 V2 encoder / probe cards，未列 V2.1。此负面结果只说明“官方 collection 没有 2.1 card”，不说明 2.1 权重未发布；后者已由官方 GitHub README 的直链验证。

## 6. 负面核验与纠偏

| 检查 | 结果 | 页面处理 |
|---|---|---|
| V-JEPA 是否直接生成 RGB | 否；另训 diffusion decoder | 明确拆出可视化模块 |
| V2 encoder 是否动作条件 | 否 | 只把 V2-AC predictor 称为 action model |
| V2.1 是否 V-JEPA 3 / 新动作模型 | 否 | 定位为 dense representation recipe |
| V2.1 摘要 +20 grasp 是否纯 backbone 增益 | 否；规划 horizon / samples 同时改变 | 写明混杂；不做纯归因 |
| V2.1 是否有官方 HF card | collection 未发现 | 记录未列，但同时记录 GitHub 直链权重 |
| V2.1 README 论文链接是否正确 | 仍残留 `arxiv.org/abs/TODO` | 正文给正确 arXiv `2603.14482` |
| I-JEPA / V-JEPA 许可证能否由 API 明确 | GitHub API 为 NOASSERTION | 不凭印象写 SPDX |
| MC-JEPA 是否有作者论文专属公开仓库 | 精确搜索未发现 | 不声称存在；仅保留论文里程碑 |
| EB-JEPA 是否 ICLR main track | 否，World Models Workshop | 论文状态写全 |
| TD-JEPA 是否正式 ICLR 2026 | 是，OpenReview venue 核验 | 与纯 arXiv 条目区分 |
| LeVJEPA 是否成熟控制 world model | 否；新视频表征预印本 | 标 E2、新近、无 action planning |
| feature probe 是否能证明物理规律 | 否 | 评测矩阵禁止跨级 |
| raw energy 是否是概率 / 置信度 | 否 | 要求校准或 proper score |

## 7. 图示设计与可访问性

正文使用 3 个 Mermaid 和 1 张生成式教学 PNG：

1. teacher–student JEPA 信息流：突出 context 可见域、target full-view、stop-gradient、EMA 与梯度边界；
2. 能力阶梯：主线是 representation → action dynamics → search → MPC，横线分支不暗示自动升级；
3. 动作训练 / 部署双环：区分 teacher-forced latent loss 与真实环境 receding-horizon loop。

每个 Mermaid 都包含单行 `accTitle`、`accDescr`，节点标签不用颜色承载语义，且图后有完整顺序化文字替代。PNG 同时提供精确 alt、图注和五步顺序化替代；Mermaid 仍承担可编辑、可搜索的确定性关系。

生成式教学图记录：

- **文件：** `assets/diagrams/jepa-latent-prediction-planning-loop.png`
- **工具与日期：** built-in OpenAI image generation，2026-08-30
- **尺寸：** 1536 × 1024 RGB PNG
- **SHA-256：** `47ba76771768ae581313df389cea541540e06d82c7bb8af95b2a3054fc4c61c0`
- **灰度统计：** mean 0.904244，standard deviation 0.231017，min 0，max 1

Prompt 要求上半部分严格区分 context encoder、EMA/stop-gradient target encoder、mask-position token、predicted/target latent loss 与可选 pixel decoder；下半部分画出多候选动作序列、latent rollout、goal/value score、只执行第一步和真实新观测回环；右侧用四级证据阶梯阻止把 probe 跨级解释成 world model。首版回读发现 pixel decoder 的虚线可能被误读为参与 latent loss，因此用同一原图定点编辑：decoder 改为只从 predicted latent 接收虚线并输出 optional visualization；latent loss 只保留 $\hat z_T$ 与 $z_T$ 两路输入。最终图在原尺寸回读，箭头、公式标签和警示框均清晰。

## 8. 验证记录

完成正文后执行：

```bash
# markdownlint
markdownlint-cli2 docs/jepa.md sources/research_20260830_jepa.md

# 内联 Python：引用闭合、重复 anchor、相对链接和外链状态
python -c '<read-only checks>'

# 抽取每个 mermaid fenced block，逐个实际渲染到临时 SVG
mmdc -p /tmp/puppeteer.json \
  -i /tmp/jepa-mermaid-N.mmd \
  -o /tmp/jepa-mermaid-N.svg

# whitespace / patch sanity
git diff --check -- docs/jepa.md sources/research_20260830_jepa.md
```

实际结果：

| 检查 | 结果 |
|---|---|
| markdownlint-cli2 v0.23.2 / markdownlint v0.41.1 | PASS，主线最终复跑 2 files，0 errors |
| 引用闭合 | PASS，19 个 anchor、60 次引用；使用集合与定义集合相等，无重复和 label mismatch |
| 相对链接 | PASS，主线插图后 2 / 2 存在 |
| 外部链接 | PASS，30 个唯一 URL 全部返回 HTTP 2xx；使用并发只读 GET，未以搜索摘要替代内容核验 |
| Mermaid 可访问性 | PASS，3 / 3 `accTitle`、3 / 3 `accDescr`，且每图后有文字替代 |
| Mermaid 实际渲染 | PASS，3 / 3 用 Mermaid CLI 渲染为临时 SVG；首轮 CLI 未自动发现 Chrome，显式指向本机 Chrome 后全部通过 |
| 仓库产物 | PASS，Mermaid SVG 与 puppeteer config 仅在 `/tmp`；主线整合新增 1 张教学 PNG，并保留生成记录、hash、alt 和文字替代 |
| `git diff --check` | PASS，无 whitespace error |

共享工作区另有其他 agent 的改动；审稿子任务只创建 / 修改两份 Markdown，主线随后增加上述 PNG 与图文记录。未修改 coverage，未 commit 或 push。

## 9. 审计结论

页面的最高证据结论限定为：

1. masked joint-embedding prediction 能学习强图像 / 视频表征；
2. action-conditioned predictor 可以把冻结视觉先验接到 latent MPC；
3. dense、collapse、长期 dynamics 与多未来是彼此可组合但需独立验收的分支；
4. 截至冻结日，真实机器人证据仍是同类 embodiment、小样本、视觉目标和有限 horizon 协议；
5. 没有任何 frozen probe、像素可视化或单一 energy 数值足以证明完整物理理解、校准不确定性或开放世界控制。
