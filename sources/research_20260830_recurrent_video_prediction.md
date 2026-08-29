# 递归/循环视频预测：截至 2026-08-30 的检索、筛选与证据审计

> 冻结时间：2026-08-30（Asia/Shanghai）
> 对应正文：[`docs/generative-models/recurrent-prediction.md`](../docs/generative-models/recurrent-prediction.md)
> 研究类型：机制导向的 scoping literature review，不是穷尽式 meta-analysis，也不是模型排行榜

## 📋 1. 改写对象、问题与邻章边界

### 1.1 改写前快照

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `docs/generative-models/recurrent-prediction.md` | 89 | `fadb109d2397ccfcadb7d1a39de83b5e5db35d5c29e94c10a5df6b2bc4574fac` |

[`coverage_audit_20260829.md`](coverage_audit_20260829.md) 将该页评为 depth 2，明确缺口是：SV2P / SVG-LP / SAVP / VideoFlow 比较，以及 teacher forcing → open-loop → self forcing 的桥梁。原文只有 5 条参考文献、0 张图，且把 recurrent、AR、causal 与流式部署放得过近。

### 1.2 本次研究问题

1. recurrent state update、概率 AR factorization、causal access、streaming/deadline 的最小判据分别是什么？
2. 直接像素、显式运动变换、空间递归状态、随机 latent、normalizing flow、state-space 与 diffusion 如何复用同一 rollout 外壳？
3. SV2P、SVG-LP、SAVP 与 VideoFlow 对“多种未来”改变了什么，posterior collapse 如何验证？
4. MCVD、Diffusion Forcing、FramePack、SkyReels-V2、MAGI-1、LongLive 的 state、commit unit 与 inner sampler 分别是什么？
5. teacher forcing、scheduled sampling、complete teacher forcing、self forcing 与 Causal Forcing 是否改变同一层？
6. 怎样以 open-loop drift、diversity–accuracy、action intervention 与 closed-loop planning 协议限制强主张？

### 1.3 与邻章的所有权划分

先全文阅读以下页面，避免重复搬运：

- [`autoregressive-generation.md`](../docs/generative-models/autoregressive-generation.md)：拥有 strict/set/frame/chunk AR、commit 粒度、KV 复杂度与完整 teacher/self-forcing 时序推导；
- [`causal-streaming-generation.md`](../docs/generative-models/causal-streaming-generation.md)：拥有少步蒸馏、bounded memory、端到端 SLO 与 causal/streaming/real-time 的系统展开；
- [`variational-generation.md`](../docs/generative-models/variational-generation.md)：拥有 ELBO、learned prior、posterior collapse 与 codec/tokenizer 的数学展开；
- [`coverage_audit_20260829.md`](coverage_audit_20260829.md)：给出改写前缺口与 depth-4 完成条件。

本页只保留理解递归视频预测必需的交叉边界，并把独有篇幅用于状态机制、经典 stochastic prediction 与闭环证据协议。

## 🔎 2. 检索表面、日期与可复现检索式

### 2.1 表面 A：正式 proceedings、OpenReview 与 arXiv

2026-08-30 使用标题精确检索与作者/会议页回查。代表性检索式：

~~~text
site:proceedings.neurips.cc "Convolutional LSTM Network"
site:openreview.net "Deep Predictive Coding Networks for Video Prediction"
site:proceedings.neurips.cc PredRNN Spatiotemporal LSTMs
site:openreview.net "Stochastic Variational Video Prediction"
site:proceedings.mlr.press "Stochastic Video Generation with a Learned Prior"
site:openreview.net VideoFlow stochastic video generation
site:proceedings.neurips.cc MCVD masked conditional video diffusion
site:proceedings.neurips.cc "Diffusion Forcing"
site:openaccess.thecvf.com "Taming Teacher Forcing"
site:proceedings.neurips.cc "Self Forcing"
site:proceedings.neurips.cc "Frame Context Packing"
site:openreview.net LongLive ICLR 2026
site:icml.cc/Downloads/2026 "Causal Forcing"
~~~

对新技术报告使用的 arXiv API 请求：

~~~text
https://export.arxiv.org/api/query?
id_list=2602.02214,2504.13074,2505.13211
~~~

该请求在冻结日返回 3 条：Causal Forcing v5、SkyReels-V2 v3、MAGI-1 v1；用于核对题名、版本、作者、摘要、官方代码链接和 venue comment。正式会场仍须回到 proceedings / conference index，不从 arXiv comment 单独升级。

### 2.2 表面 B：OpenAlex 元数据检索；Semantic Scholar 为失败回退记录

OpenAlex 请求模板：

~~~text
https://api.openalex.org/works?search=<query>&per-page=1
https://api.openalex.org/works?filter=title.search:<exact-title>&per-page=5
~~~

冻结日实际返回量如下；数字是索引命中数，查询彼此重叠，不是独立筛选样本或 PRISMA 总数。

| OpenAlex query | 返回量 | 用法与处置 |
|---|---:|---|
| `recurrent video prediction ConvLSTM PredRNN` | 455 | 用于发现同义词；按 title / abstract 初筛，不用 cited-by 排名决定纳入 |
| `stochastic video prediction SV2P SVG SAVP VideoFlow` | 7 | 命中核心随机预测线及邻近工作，再回查原论文 |
| `video prediction diffusion MCVD Diffusion Forcing` | 53 | 区分视频 diffusion、动作 diffusion 等同名噪声 |
| `teacher forcing scheduled sampling self forcing video` | 46,579 | 精度过低，排除该 broad query，改用标题精确检索 |
| `title.search: Convolutional LSTM Network` | 618 | 前五条含同名扩展；原论文存在 W1485009520 / W2953118818 两条记录，按同题同年去重 |
| `title.search: Stochastic Variational Video Prediction` | 5 | 核心记录 W2765363933；OpenAlex 年份为 arXiv 索引年 2017，venue 采用 ICLR 2018 正式页 |
| `title.search: MCVD Masked Conditional Video Diffusion` | 2 | arXiv W4281706822 与 proceedings W7133227252 去重，正文引用正式页 |
| `title.search: Diffusion Forcing Next-token Prediction Meets Full-Sequence Diffusion` | 2 | arXiv W4400373517 与 proceedings W4415798523 去重 |
| `title.search: Frame Context Packing and Drift Prevention` | 1 | W4415273172，DOI `10.52202/085713-1024`，核对 NeurIPS 2025 五作者版 |

Semantic Scholar Graph API 也以标题精确式尝试：

~~~text
https://api.semanticscholar.org/graph/v1/paper/search?
query=Diffusion%20Forcing%3A%20Next-token%20Prediction%20Meets%20Full-Sequence%20Diffusion
&limit=5
&fields=title,authors,year,venue,url,externalIds
~~~

冻结日返回 HTTP 429 `TooManyRequests`，因此**没有任何正文断言来自该失败响应**。OpenAlex 完成了“OpenAlex/Semantic Scholar”索引表面的去重和 venue 漂移提示，所有机制事实仍回到一手论文。

### 2.3 表面 C：官方项目、代码与发布面

| 工作 | 官方 artifact | 本次核对项 |
|---|---|---|
| PredNet | [coxlab/prednet](https://github.com/coxlab/prednet) | 作者组织、模型与评测实现 |
| SVG-LP | [edenton/svg](https://github.com/edenton/svg) | learned-prior 代码归属 |
| SAVP | [项目页](https://alexlee-gk.github.io/video_prediction/) / [代码](https://github.com/alexlee-gk/video_prediction) | arXiv/投稿状态、best-of-many 展示语境 |
| VideoFlow | [作者项目页](https://sites.google.com/view/videoflow/home) | 多帧 flow 主张；未把第三方同名仓库写成官方代码 |
| MCVD | [voletiv/mcvd-pytorch](https://github.com/voletiv/mcvd-pytorch) / [项目页](https://mask-cond-video-diffusion.github.io/) | block rollout 与 `one_frame_at_a_time` 可选配置 |
| Diffusion Forcing | [buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing) / [项目页](https://boyuan.space/diffusion-forcing/) | per-token noise、rolling task 与代码范围 |
| Self Forcing | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | self rollout、few-step 与 gradient truncation 同时存在 |
| Causal Forcing | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) / [项目页](https://thu-ml.github.io/CausalForcing.github.io/) | AR teacher 初始化、ICML 2026 标签、基础模型长窗限制 |
| FramePack | [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) / [项目页](https://lllyasviel.github.io/frame_pack_gitpage/) | vanilla causal 与 anti-drift / inverted 非严格因果变体 |
| SkyReels-V2 | [SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) | last-frame segment extension、模型/代码发布、作者“infinite-length”措辞 |
| MAGI-1 | [SandAI-org/MAGI-1](https://github.com/SandAI-org/MAGI-1) | 24-frame chunk、block-causal attention、并行去噪范围 |
| LongLive | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) / [项目页](https://nvlabs.github.io/LongLive/) | v1 frame sink、KV re-cache、硬件绑定 FPS/时长；v2 infra 不倒写为 v1 论文机制 |

项目页只支持“作者公开了什么、配置怎样写、作者怎样描述系统”。没有独立复现时，不把 demo、Stars、README 速度或“infinite”措辞升级为普适实验结论。

## 🧾 3. 纳入、排除、去重与证据等级

### 3.1 纳入标准

- 原始方法直接改变 recurrent state、预测 head、随机未来、训练历史来源、rolling commit 或决策使用；
- 机制事实能在正式论文全文/摘要或作者技术报告中定位；
- 新系统主张至少有作者 arXiv + 官方 artifact 双表面；
- 非视频 sequence work 仅在其定义 scheduled sampling 等通用训练问题时纳入，并明确不是视频效果证据；
- 评测协议允许 repository synthesis，但不伪装成某篇论文的实验结论。

### 3.2 排除或转交邻章

| 候选 | 处置 | 原因 |
|---|---|---|
| Professor Forcing | 不进正文主线 | NeurIPS 2016 的 hidden-dynamics 对抗对齐很相关，但原实验不是视频；避免训练策略名录膨胀 |
| CausVid | 仅在邻章展开 | CVPR 2025 的 bidirectional teacher → 4-step causal student 属于蒸馏/流式主线，已由 causal-streaming 章节拥有 |
| Rolling Forcing | 转交 causal-streaming 章节 | rolling window 与 frame sink 是部署/蒸馏扩展；不是本章经典随机预测缺口的必要节点 |
| LongLive 2.0 | 不回写 LongLive v1 机制 | 2026 infra、NVFP4 与并行发布是后续系统；正文只用 ICLR 2026 v1 论文可归因事实 |
| Deep Forcing / FlowCache / Context Forcing 等 | 不堆名词 | 属于记忆/缓存/后训练的相邻前沿；未改变本章所需的最小 taxonomy |
| 只展示长视频 demo 的系统 | 排除强结论 | 没有无重置 horizon curve、失败尾部和隐藏 reset/anchor 说明 |
| OpenAlex broad-query 非相关条目 | title/abstract 排除 | 例如 action/motion diffusion 或通用 teacher forcing 论文，名称命中但任务不符 |

去重后正文证据注册表含 **23 篇 primary works**。OpenAlex 的 arXiv/proceedings 双记录合并为同一工作；项目页和代码作为 artifact，不重复计为论文。由于表面是定向 scoping search、broad query 高度重叠，本记录不虚构“去重前唯一文献总数”。

### 3.3 证据等级

| 等级 | 来源组合 | 可支持 | 不可自动支持 |
|---|---|---|---|
| **E1** | 正式 proceedings / accepted OpenReview + 原论文 | 标题、作者、venue、机制、论文设置内结果 | 跨设置普适最优、商业可用、独立复现 |
| **E2** | 作者 arXiv 技术报告 + 官方项目/代码 | 作者提出的机制、发布面、作者报告结果 | 正式同行评审共识或独立确认 |
| **E3** | 官方项目、代码、model card、conference index | artifact、配置、版本、作者声明、venue 交叉核验 | 论文所有实验可复现、权重/数据权利、一般性能 |
| **E4** | OpenAlex / Semantic Scholar 等索引 | 候选发现、重复记录、年份/DOI 异常提示 | 最终 venue、作者拼写或机制断言 |
| **S** | 本次跨来源综合 | 术语边界、协议与章节结构 | 新的实证结果 |

写作规则：E1/E2 的结果写“论文/作者报告”；E3 的 README 主张写“官方项目称”；E4 只导航；S 结论明确为最小报告协议或逻辑边界。

## 📚 4. Primary evidence registry

| ID | 工作与一手页 | 等级 | 本章使用与边界 |
|---|---|---|---|
| R01 | [Mathieu et al., Deep Multi-Scale Video Prediction](https://arxiv.org/abs/1511.05440), ICLR 2016 | E1 | 多峰未来下 point estimate / pixel objective 的平均化直觉；不写成“MSE 必然模糊” |
| R02 | [Finn et al., Unsupervised Learning for Physical Interaction](https://proceedings.neurips.cc/paper/2016/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html), NeurIPS 2016 | E1 | DNA/CDNA/STP 与 action-conditioned prediction；机器人任务证据不外推为普适物理因果 |
| R03 | [Shi et al., ConvLSTM](https://proceedings.neurips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html), NeurIPS 2015 | E1 | 卷积 input/state transition 与空间 hidden state |
| R04 | [Lotter et al., PredNet](https://openreview.net/forum?id=B1ewdt9xe), ICLR 2017 | E1 | predictive-coding error hierarchy；预测与表征迁移证据分开 |
| R05 | [Wang et al., PredRNN](https://proceedings.neurips.cc/paper_files/paper/2017/hash/e5f6ad6ce374177eef023bf5d0c018b6-Abstract.html), NeurIPS 2017 | E1 | ST-LSTM 与 zigzag spatiotemporal memory |
| R06 | [Bengio et al., Scheduled Sampling](https://proceedings.neurips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html), NeurIPS 2015 | E1 | curriculum 混合 ground truth / model output；通用 sequence 方法，不是视频专属 |
| R07 | [Huszár, How (not) to Train your Generative Model](https://arxiv.org/abs/1511.05101), 2015 | E2 | scheduled sampling objective 不一致批评；作为理论边界，不当成视频 benchmark |
| R08 | [Babaeizadeh et al., SV2P](https://openreview.net/forum?id=rk49Mg-CW), ICLR 2018 | E1 | sequence-level / time-varying stochastic latent 与条件未来 |
| R09 | [Denton and Fergus, SVG-LP](https://proceedings.mlr.press/v80/denton18a.html), ICML 2018 | E1 | time-varying learned prior；作者按 PMLR 写 Emily Denton |
| R10 | [Lee et al., SAVP](https://arxiv.org/abs/1804.01523), 2018 | E2 | ConvLSTM + local posterior + VAE/GAN 互补目标；只写 arXiv technical report，不声称正式 venue |
| R11 | [Kumar et al., VideoFlow](https://openreview.net/forum?id=rJgUfTEYvH), ICLR 2020 | E1 | conditional normalizing flow、exact likelihood 与 latent dynamics；exact 只相对模型表示成立 |
| R12 | [Hafner et al., PlaNet](https://proceedings.mlr.press/v97/hafner19a.html), ICML 2019 | E1 | RSSM deterministic/stochastic state 与 latent online planning |
| R13 | [Voleti et al., MCVD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/944618542d80a63bbec16dfbd2bd689a-Abstract-Conference.html), NeurIPS 2022 | E1 | masked conditional diffusion、non-recurrent 2D backbone、block rollout；代码 one-frame 选项不改论文主分类 |
| R14 | [Chen et al., Diffusion Forcing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html), NeurIPS 2024 | E1 | independent per-token noise 与 rolling tasks；不推出 self-history、few-step 或 deadline |
| R15 | [Zhou et al., Taming Teacher Forcing / MAGI](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html), CVPR 2025 | E1 | CTF = complete unmasked ground-truth history；仍是 off-policy history |
| R16 | [Huang et al., Self Forcing](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html), NeurIPS 2025 | E1 | self-generated rollout history + rolling KV；few-step/DMD/gradient truncation 另列 |
| R17 | [Zhu et al., Causal Forcing](https://arxiv.org/abs/2602.02214), ICML 2026 | E1 + E3 | AR teacher 做 ODE initialization，再使用 self-forcing 式 DMD；venue 由 [ICML 2026 official downloads index](https://icml.cc/Downloads/2026) 与官方 repo 交叉核验 |
| R18 | [Zhang et al., FramePack](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html), NeurIPS 2025 | E1 | 五作者正式版、fixed context packing 与 drift prevention；不混用早期两作者 arXiv 元数据 |
| R19 | [Chen et al., SkyReels-V2](https://arxiv.org/abs/2504.13074), 2025 | E2 | per-token Diffusion Forcing、last-frame segment extension；“infinite-length”保留为作者/系统主张，且记录论文结论承认 error accumulation 限制实际高质量时长 |
| R20 | [Sand.ai et al., MAGI-1](https://arxiv.org/abs/2505.13211), 2025 | E2 | 24-frame chunk、block-causal attention 与 chunk-wise denoising；world-model 自称不是闭环规划证据 |
| R21 | [Yang et al., LongLive](https://openreview.net/forum?id=nCAODkpsPJ), ICLR 2026 | E1 + E3 | frame-level AR、short window/frame sink、KV re-cache；20.7 FPS 与 240 秒绑定单 H100 论文设置 |
| R22 | [Ebert et al., Visual Planning](https://proceedings.mlr.press/v78/frederik-ebert17a.html), CoRL 2017 | E1 | video prediction 进入 visual MPC 的 downstream evidence |
| R23 | [He et al., Lagging Inference Networks](https://openreview.net/forum?id=rylDfnCqF7), ICLR 2019 | E1 | posterior collapse、mutual information / inference lag；提供通用 VAE 诊断，不充当视频效果比较 |

## ⚖️ 5. 关键术语与 venue 裁决

### 5.1 Recurrent、AR、causal 与 streaming

- `recurrent`：计算状态或已提交输出被反复反馈；可以是 LSTM、RSSM、KV、frame packing。
- `autoregressive`：联合概率按有序 unit 条件分解；不要求显式固定维度 $h_t$。
- `causal`：当前 unit 不读取未来干净 unit；不等于 intervention correctness。
- `streaming`：完整输出结束前持续发出结果；`real-time` 还要端到端满足 deadline。
- diffusion 内部 solver step 是 noise time；frame/chunk commit 是 data time，必须分开计数。

### 5.2 Forcing 名称

- teacher forcing：真实前缀；
- scheduled sampling：随机混合真实/模型前一步，是 curriculum，不是严格 on-policy sequence objective；
- CTF：MAGI 的 complete **ground-truth** history，不是 complete rollout；
- Self Forcing：训练时真实提交自生成 history；few-step 与 history source 是两个改动；
- Causal Forcing：AR teacher 初始化 + DMD 的方法专名，处理 bidirectional teacher → causal student 的 flow-map / architecture gap。

### 5.3 Venue 与作者漂移

1. SV2P 使用 ICLR 2018 OpenReview；不能用 OpenAlex/arXiv 2017 索引年当 venue 年。
2. SVG-LP 使用 PMLR 的 **Emily Denton**、ICML 2018 元数据。
3. SAVP 截至冻结日按 arXiv 1804.01523 技术报告；项目页的 submission 状态不写成已接收会场。
4. VideoFlow 是 ICLR 2020 `rJgUfTEYvH`，与 optical-flow 软件/论文同名项去重。
5. FramePack 正式 NeurIPS 2025 版有 Lvmin Zhang、Shengqu Cai、Muyang Li、Gordon Wetzstein、Maneesh Agrawala 五位作者；早期 arXiv 题录不可覆盖正式元数据。
6. Causal Forcing 的 ICML 2026 由 conference downloads index 与作者 repo 交叉核验；机制仍引用 arXiv v5 原文。
7. SkyReels-V2 与 MAGI-1 是 2025 arXiv technical reports；官方代码存在不等于正式 proceedings。
8. LongLive 使用 ICLR 2026 OpenReview；后续 LongLive 2.0 README 数字不倒写为原论文结果。

## 🧪 6. 证据协议如何形成

以下是跨 R02、R08–R14、R16、R21–R23 形成的 **S 级综合协议**，不是声称某篇论文已完整执行全部项目。

### 6.1 Open-loop drift

1. 固定 observation、commit unit、总 horizon、memory policy、overlap、reset 与 re-anchor。
2. 同时跑 codec-only、teacher-forced 与 fully open-loop 三条路径。
3. 在等比增长的 commit horizon 报均值、区间/分位数和预注册首次失败时间。
4. 单独测 seam、对象存在、身份、几何、运动、动作/prompt 响应。
5. 消融 window、sink、packing 和 recurrent state；不以单个长视频 demo 代替曲线。
6. 对 open-ended/infinite 主张写最长实际测量点，点外标记 unmeasured。

### 6.2 Posterior collapse 与 latent usage

1. 逐时间步 KL、active units 与必要时条件互信息；
2. 相同历史下重采样、置换、屏蔽 latent；
3. prior rollout 与 posterior rollout 差距；
4. 语义事件多样性与时间一致性，不把闪烁当 latent 被有效使用；
5. decoder-strength / latent-path ablation。

### 6.3 Diversity–accuracy

- 所有模型相同每条件样本数 $N$ 与 seed 规则；
- 单样本 fidelity 分布、样本间 diversity、mode coverage 与 calibration 同时报告；
- average-of-$N$、best-of-$N$ 和人工选择样本明确分栏；
- 若只能获得一个真实未来，不能把到该样本的最近距离当成完整分布得分。

### 6.4 Action intervention 与 closed-loop planning

- 从相同/可匹配初始状态，只替换动作，固定其他条件与随机种子；
- 包含 no-op、反向动作、作用时延与 held-out action sequence；
- 与真实环境或保真 simulator 的效应方向、幅度、时序对齐；
- MPC 每轮只执行第一步并读回新观测；
- 对照 open-loop script、无模型 baseline、oracle dynamics（若有）；
- 同时报任务成功、碰撞/约束违反、模型不确定性、model exploitation 与端到端 deadline。

## 🚧 7. 未决问题与冻结日后的更新规则

- 经典数据集常只有一个观测未来，coverage/calibration 需要可控模拟器、多未来标注或事件级 proxy；不存在单一完美指标。
- self-generated history 降低 exposure gap，但超长 horizon 仍受到 state compression、codec、chunk seam 与分布尾部共同影响。
- fixed context / constant peak memory 不等于总时间常数，也不证明记忆质量不随时长下降。
- causal attention 不证明动作干预正确；“world model”名称不能替代 closed-loop benefit。
- SkyReels-V2、MAGI-1 的正式 venue 状态和后续版本可能变化；冻结日后更新时必须重查 proceedings、arXiv version 与官方 artifact，不能只改年份。
- 官方项目后续可能修改 README、checkpoint 或速度数字；章节只保留与论文版本可归因的边界。

## ✅ 8. 本次写作验收映射

| 要求 | 正文位置 | 验收方式 |
|---|---|---|
| 确定性 pixel / transform | §3 | MSE 边界 + DNA/CDNA/STP 公式与局限 |
| ConvLSTM / PredNet / PredRNN | §4 | 三种状态机制分别定义 |
| SV2P / SVG-LP / SAVP / VideoFlow | §5 | 同表比较 latent、目标、venue 与证据边界 |
| latent/state-space | §6 | RSSM 公式 + planning 使用边界 |
| MCVD / Diffusion Forcing / modern rolling | §7 | state、commit、inner sampler 与测量上限分栏 |
| forcing 精确边界 | §8 | 五行 history-source 表，邻章交叉链接 |
| open-loop / latent / diversity | §9–10 | 可复核控制变量与报告输出 |
| action intervention / planning | §11 | Mermaid 闭环时序 + 对照协议 |
| 机制里程碑 | §12 | 每行“为何里程碑 / 未解决” |
| 可访问视觉 | §1、§2、§11 | 系统图精确 alt + 两张 Mermaid `accTitle` / `accDescr` + 紧随文字替代 |
| 引用与 venue | §15、本记录 §4–5 | `ref-N` 一手页；arXiv/正式版去重 |

最终静态检查与实际 Mermaid 渲染结果在完成改写后追加到本记录末尾；验证命令不修改仓库内其他文件。

## ✅ 9. 冻结日最终验证

2026-08-30 对最终两份文件执行：

| 验证项 | 命令/方法 | 结果 |
|---|---|---|
| Markdown | `npx --yes markdownlint-cli2 <两文件>` | 0 issues |
| 引用锚点 | Node 脚本交叉检查 `](#ref-N)` 与 `<a id="ref-N">` | 23 个定义、44 次引用；missing 0，unused 0 |
| 相对链接 | 从各自目录解析本地 Markdown/image target | 正文 8 个、研究记录 6 个；broken 0 |
| Mermaid 可访问元数据 | 统计 fenced chart、`accTitle`、`accDescr` 与紧随文字替代 | 2 / 2 / 2；另有共享系统图文字替代，共 3 处 |
| Mermaid 实际渲染 | Mermaid CLI + 本机 Chrome，Markdown 输入、PNG 输出，scale 2 | 2 张均成功生成；人工查看无截断、标签可读、箭头与 loop 正确 |
| tracked diff whitespace | `git diff --check -- docs/generative-models/recurrent-prediction.md` | 通过，无 whitespace error |
| 作用域 | `git status --short` 与 target-specific diff | 本任务只改正文并新增本研究记录；共享工作区的其他协作者文件未触碰 |

渲染产物只放在 `/tmp/recurrent-prediction-final-artifacts/` 用于验收，没有加入仓库；正文保留可编辑 Mermaid 源码。共享系统 PNG 由并行章节任务提供，本章只按要求引用，并链接其独立研究记录。
