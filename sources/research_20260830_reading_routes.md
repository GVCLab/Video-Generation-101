# 阅读路线研究日志（2026-08-30）

## 1. 范围与冻结点

- **目标文件：** `docs/reading-list.md`
- **检索与核验日期：** 2026-08-30（Asia/Shanghai）
- **时间范围：** 经典先修不设起始年；前沿分支重点看 2024–2026
- **主题范围：** 因果/流式视频，少步生成/后训练/蒸馏，原生音视频，World Action Model / JEPA
- **冻结规则：** 只按冻结日能核验的发表状态标注；之后的版本、转正或撤回不倒灌

本日志是可追溯的检索记录，不是第二份书目库。路线页只保留足以形成“先修 → 主干 → 分支 → 验证”的代表作，而不追求网罗所有新论文。

## 2. 研究问题

1. 一个初学者进入四条前沿路线前，最少需要哪些表示、生成目标、时间建模、评测与控制先修？
2. 哪些工作构成方法主干，哪些只是新近 recipe 或产品声明？
3. 每篇论文的主张可以被哪个最小实验重现，又可被哪个负对照推翻？
4. “因果”、“实时”、“少步”、“联合音视频”、“世界模型”和“零样本策略”各需要什么证据？

## 3. 检索策略

### 3.1 发现性宽搜索

2026-08-30 使用 arXiv API 题录宽搜索下列查询。返回总数只用于说明候选池量级，不是 PRISMA 式精确样本数，也不是证据。

| 路线 | arXiv API 查询 | 返回总数 |
|---|---|---:|
| causal / streaming | `all:("causal video diffusion" OR "streaming video generation" OR "autoregressive video diffusion")` | 98 |
| few-step / post-training | `all:"video generation" AND all:("consistency distillation" OR "distribution matching distillation" OR "preference optimization" OR "human feedback")` | 100 |
| native audio-video | `all:("audio-video generation" OR "joint audio visual generation") AND all:(synchronized OR joint)` | 62 |
| world-action / JEPA | `all:("world action model" OR "video world model" OR "joint embedding predictive architecture")` | 563 |

宽搜索后用标题、作者、方法名交叉去重，再转到论文原文和正式发表入口。

### 3.2 正式状态定点查询

使用以下模板定点核验，并在正式入口中回读标题、作者与会议/期刊：

```text
site:proceedings.neurips.cc "<exact paper title>"
site:openaccess.thecvf.com "<exact paper title>"
site:proceedings.mlr.press "<exact paper title>"
site:openreview.net "<exact paper title>"
site:nature.com/articles "<exact paper title>"
site:icml.cc/Downloads/2026 "Causal Forcing"
site:arxiv.org/abs "<exact paper title>"
```

主题补充查询：

```text
"causal video generation" self forcing diffusion forcing
"real-time streaming video generation" TTFF latency
"video diffusion" distillation preference optimization reward model
"joint audio video generation" twin DiT single stream
"world action model" video action policy
"V-JEPA" action-conditioned world model planning
```

### 3.3 来源优先级

1. 出版方/会议 proceedings 或期刊原文；
2. 官方接收名单；
3. 作者 arXiv 稿、官方项目仓库或模型卡；
4. 机构 system card 或技术报告，仅用于该机构自述。

搜索引擎摘要、媒体、论坛、二手博客和论文聚合站只可导航，不承担路线页的方法、数字或发表状态断言。

## 4. 纳入、排除与去重

### 4.1 纳入标准

- 能填充一个明确教学节点：先修、方法起点、重要分叉、证据边界或可验证前沿；
- 可从一手页面核验标题、方法主张和冻结日的发表状态；
- 与视频生成、视频动力学、音视频生成或动作条件世界模型直接相关；
- 能提出最小复现或负对照，而不只是展示样例。

### 4.2 排除标准

- 只有二手摘要，找不到作者原文、官方仓库或正式入口；
- 同一方法的轻微版本更新，不会改变路线结构或证伪任务；
- 纯音频生成、纯图像加速或纯语言世界模型，且没有必要的视频迁移作用；
- 只有产品演示，无法核验时序因子化、训练目标、版本或释放面；
- 没有封闭回路或 action intervention 证据，却仅凭画面可视化声称已是可行动 world model。

### 4.3 去重规则

- 有正式版时，路线主链指向 proceedings/期刊，不再把 arXiv 稿单独算一篇；
- 技术报告与产品页只要证据合同不同，可以分别保留；
- 同一论文可服务两条路线，但只给一个发表状态；例如 CausVid 同时连接因果生成与少步蒸馏。

## 5. 证据分级

| 级别 | 可核验条件 | 路线页允许的语气 |
|---|---|---|
| **A·正式发表** | proceedings、期刊或同行评议入口已经出现 | “论文提出/报告；发表于……” |
| **A*·正式接收** | 官方名单可核验接收，但冻结日未找到正式 proceedings 页 | “已被接收；正文当前见作者稿” |
| **B·预印本** | 作者 arXiv 稿，未核验正式发表 | “作者提出/报告；截至冻结日为预印本” |
| **C·技术报告** | 机构/团队技术报告，无同等正式论文 | “报告披露/展示” |
| **D·官方发布** | 官方 system card、模型卡、仓库或产品页 | “提供方声明/发布” |
| **S·课程综合** | 本课程的路线、实验与停止规则 | “本课程要求” |

正式发表等级只说明同行评议/出版状态，不保证结论正确、工件开放或能在本地复现。

## 6. 主干与分支的一手证据台账

### 6.1 共同主干

| 节点 | 一手入口 | 状态 | 纳入理由/边界 |
|---|---|---|---|
| Beyond MSE | [作者稿](https://arxiv.org/abs/1511.05440)、[ICLR 2016 archive](https://iclr.cc/archive/www/2016.html) | A | 多未来与像素平均的入口；不把感知清晰等同动力学正确 |
| VQ-VAE | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | A | 离散表示与压缩损失的基线 |
| DDPM | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | A | 扩散训练与多步采样的基线 |
| Flow Matching | [OpenReview](https://openreview.net/forum?id=PqvMRDCJT9t) | A | 连续时间向量场目标；步数仍需独立报告 |
| Video Diffusion Models | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html) | A | 视频扩散、联合生成与级联设计入口 |
| MAGVIT | [CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html) | A | 视频 tokenization 与 masked generation |
| VBench | [CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) | A | 用分维度评估代替“一个总分说明一切” |

### 6.2 因果/流式路线

| 作品 | 一手入口 | 状态 | 本路线采用的主张 |
|---|---|---|---|
| Diffusion Forcing | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html) | A | per-token noise 把生成、预测与 rolling 放进同一框架 |
| CausVid | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) | A | 双向教师到因果学生，DMD、4-step 与 KV cache |
| Self Forcing | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) | A | 在自生成历史上训练，直接改变 exposure-bias 条件 |
| Causal Forcing | [作者稿](https://arxiv.org/abs/2602.02214)、[ICML 2026 官方名单](https://icml.cc/Downloads/2026) | A* | 冻结日已接收，未用作者稿伪装 proceedings |
| StreamDiffusionV2 | [MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | A | 把 TTFF、deadline、jitter 和 pipeline 纳入生成系统证据 |
| LongLive / Rolling Forcing | [LongLive](https://arxiv.org/abs/2509.22622)、[Rolling Forcing](https://arxiv.org/abs/2509.25161) | B | 长时窗口、记忆与 train-long-test-long 前沿 |
| Causal Forcing++ / Causal-rCM | [Causal Forcing++](https://arxiv.org/abs/2605.15141)、[Causal-rCM](https://arxiv.org/abs/2606.25473) | B | frame-wise 1–2 step 的新 recipe；不写成已经形成共识 |

### 6.3 少步/后训练/蒸馏路线

| 作品 | 一手入口 | 状态 | 本路线采用的主张 |
|---|---|---|---|
| Consistency Models | [ICML 2023](https://proceedings.mlr.press/v202/song23a.html) | A | 一步/少步一致映射的基础 |
| DMD2 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | A | 分布匹配、两时间尺度与输入失配 |
| VideoLCM | [作者稿](https://arxiv.org/abs/2312.09109) | B | 视频 latent consistency 基线；不提高其发表状态 |
| T2V-Turbo | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a57aa8e8b57e64a42e95f7dceb0adb9-Abstract-Conference.html) | A | consistency distillation 与 reward 的联合训练 |
| InstructVideo | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_InstructVideo_Instructing_Video_Diffusion_Models_with_Human_Feedback_CVPR_2024_paper.html) | A | 视频反馈微调的早期路径 |
| VideoPrefer / VideoRM | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html) | A | 视频 preference data 与 reward model |
| VideoDPO | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html) | A | 多维 preference pair 用于 diffusion DPO |
| DynamicsBoost | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.html) | A | 无人工标注的 continuation preference；仍需独立盲评 |

### 6.4 原生音视频路线

| 作品 | 一手入口 | 状态 | 本路线采用的主张 |
|---|---|---|---|
| VideoPoet | [ICML 2024](https://proceedings.mlr.press/v235/kondratyuk24a.html) | A | 统一多模态 token 与多任务；不由此推出同时联合采样 |
| Movie Gen | [技术报告](https://arxiv.org/abs/2410.13720) | C | 视频模型与独立音频模型家族，作 staged 对照 |
| MMAudio | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_MMAudio_Taming_Multimodal_Joint_Training_for_High-Quality_Video-to-Audio_Synthesis_CVPR_2025_paper.html) | A | 给定视频的同步音频生成；不是联合生成视频 |
| Ovi | [作者稿](https://arxiv.org/abs/2510.01284) | B | twin-DiT 与逐块双向融合 |
| LTX-2 | [作者稿](https://arxiv.org/abs/2601.03233) | B | 非对称双流、cross-attention 与共享时间条件 |
| MiniMax H3 | [官方发布](https://www.minimax.io/blog/minimax-h3)、[官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) | D | 单流联合 AV 的提供方声明；不当作正式论文 |
| Sora 2 | [官方 system card](https://openai.com/index/sora-2-system-card/) | D | 产品与安全发布证据；不由页面反推未披露架构 |

### 6.5 World Action / JEPA 路线

| 作品 | 一手入口 | 状态 | 本路线采用的主张 |
|---|---|---|---|
| PlaNet | [ICML 2019](https://proceedings.mlr.press/v97/hafner19a.html) | A | latent dynamics + CEM + receding-horizon 控制基线 |
| DreamerV3 | [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2) | A | learned world model 中 dynamics、actor、critic 的合同 |
| Genie | [ICML 2024](https://proceedings.mlr.press/v235/bruce24a.html) | A | 无动作标签视频的 latent action；不等于真实机器人控制量 |
| DIAMOND | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html) | A | diffusion world model 与 agent 评估；保留 model exploitation 边界 |
| GameNGen | [ICLR 2025](https://openreview.net/forum?id=P8pqeEkn1H) | A | action-conditioned 交互视频；不外推至通用控制 |
| V-JEPA | [TMLR](https://openreview.net/forum?id=QaCCuDfBk2) | A | masked latent prediction 的表征学习；不是生成器或策略 |
| DINO-WM | [ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | A | frozen feature + action dynamics + visual-goal planning |
| V-JEPA 2 / 2-AC | [作者稿](https://arxiv.org/abs/2506.09985) | B | action-free encoder 与 action-conditioned predictor 需分开记账 |
| V-JEPA 2.1 / LeWorldModel | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)、[LeWorldModel](https://arxiv.org/abs/2603.19312) | B | dense feature 与端到端 action latent 前沿 |
| DreamGen | [CoRL 2025](https://proceedings.mlr.press/v305/jang25a.html) | A | world model 生成数据后训 policy；不写成在线 planner |
| DreamZero | [作者稿](https://arxiv.org/abs/2602.15922) | B | joint video-action prediction 到 zero-shot policy 的最新汇合；需审计 zero-shot 口径 |

## 7. 关键断言与证据边界

| 断言 | 一手核验基础 | 路线页的保守结论 |
|---|---|---|
| causal video 不等于物理因果推理 | Diffusion Forcing、CausVid、Self Forcing 方法定义 | “causal”通常先指信息访问/生成顺序；物理因果需 action intervention 或反事实实验 |
| streaming 不等于 real-time | StreamDiffusionV2 的系统定义 | 上线声明至少要报 TTFF、稳态节拍、p95/p99、deadline miss 与完整 pipeline |
| few-step 不等于无损加速 | Consistency Models、DMD2、T2V-Turbo | 网络调用数、wall time、质量、多样性与 CFG 能力必须分开测 |
| reward 上升不等于整体质量提升 | InstructVideo、VideoPrefer、VideoDPO、DynamicsBoost | 训练 reward 不得同时充当唯一验收者；要查奖励投机和多样性损失 |
| “有声视频”不等于 native joint AV | Movie Gen、MMAudio、Ovi、LTX-2 的因子化/架构 | 必须标明 staged V2A、统一 token 多任务、耦合双流或单流联合 latent |
| unified tokens 不等于 simultaneous joint sampling | VideoPoet 方法与任务设定 | 共用词表/模型是表示证据，同一采样过程中的 AV 交换需另行核验 |
| JEPA representation 不等于 World Action Model | V-JEPA、V-JEPA 2-AC、DINO-WM | 要分开 encoder、action predictor、planner 和 policy 证据 |
| 好看的 rollout 不等于好 policy | PlaNet、DreamerV3、DIAMOND、DreamGen | 必须给 action intervention、receding-horizon 反馈、成功率与 model-to-real/original-env gap |
| Causal Forcing 为 A* 而不是 A | ICML 2026 官方名单 + arXiv 作者稿 | 已接收，但冻结日未用作者稿代替正式 proceedings |
| 最新不等于证据最强 | 上述所有 B/D 级条目 | 前沿论文可以必读，但必须显式标成预印本或官方发布 |

## 8. 验证任务的设计原则

1. **对准论文主张：** 质量主张用盲评与分维指标，速度主张用完整 pipeline 计时，联合 AV 主张用跨模态干预，WAM 主张用 action 对照和闭环任务。
2. **冻结混淆项：** 模型/权重版本、prompt、seed、时长、分辨率、CFG、VAE、硬件、精度和预处理。
3. **先写 falsifier：** 实验前预注册什么结果会让声明降级，避免看到样例后移动标准。
4. **保存失败：** 长时漂移、卡顿、声音张冠李戴、action-insensitive rollout 与 simulator exploitation 都是结果。
5. **不越级结论：** 作者实验只能写“论文报告”；课程复现只支持自己已运行的版本、任务和硬件。

## 9. Mermaid 路线图设计记录

`docs/reading-list.md` 使用两张 Mermaid 图，不生成 PNG：

1. **课程依赖图：** 只显示先修、共同主干、四分支、共同验收与结课项目；避免把每篇论文塞进图中。
2. **证伪回路图：** 将 claim 到 measurable proposition、protocol、frozen conditions、reproduction、falsifier 和结论降级串成闭环。

两图都提供 `accTitle` / `accDescr` 和图后中文文字替代。颜色只用于辅助分组；所有关系仍由节点文字、箭头和“是/否”标签表达。

## 10. 局限与更新触发器

- arXiv 题录宽搜索会包含不相关或同版记录；返回总数不可作为系统综述流程图的纳入数。
- 一部分 2026 工作只有预印本，或只能从官方名单核验接收；路线为此保留 A* 与 B 级。
- 商业产品的训练数据、完整架构和 serving pipeline 可能不公开；D 级只能支持提供方声明。
- 本路线刻意选代表作而不穷尽所有工作。若出现正式版、公开 checkpoint、独立复现或反证结果，应更新证据等级与最小任务。

## 11. 文档验证记录

下列检查在路线文本定稿后执行；本节将记录实际结果，不用“应该通过”代替。

- Markdown lint：`markdownlint-cli 0.47.0` 检查两份文件，**0 错误**
- Markdown 引用/围栏闭合：围栏计数与链接语法扫描，**通过**
- 仓库内相对链接：**14 处**逐一解析，目标全部存在
- 外部一手链接：**93 次引用、46 个去重 URL**；39 个直接 HTTP 200，其余 7 个因 `curl` 的 TLS/403 限制未直接返回 200，已通过浏览器入口逐一打开并回读标题
- Mermaid 实际 SVG 渲染：`mermaid-cli 11.12.0` 将两个图块分别渲染为 SVG，**2/2 通过**；SVG 均含 `aria-labelledby` 和 `aria-describedby`，未生成 PNG
- `git diff --check -- docs/reading-list.md sources/research_20260830_reading_routes.md`：**通过**
