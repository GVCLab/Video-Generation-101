# 无条件视频生成研究记录（冻结于 2026-08-30）

对应正文：[无条件视频生成：从边际分布到可审计的动态先验](../docs/tasks/unconditional-video-generation.md)。

本文件记录本轮检索、纳排、证据等级、任务归类冲突和验证结果。它不是把搜索引擎结果数当作领域规模，也不把作者自称的 “unconditional” 自动等同于严格部署合同 $p(X)$。

## 1. 研究问题与判定规则

### 1.1 核心问题

1. 哪些工作在部署采样时真正从独立随机源生成**含首帧在内的完整视频**？
2. 从 Video Textures/Dynamic Textures 到 GAN、token、masked、diffusion、DiT、flow 的可验证技术转折是什么？
3. 2025–2026 年有哪些正式发表的直接证据，哪些只是预印本、条件系统或可迁移组件？
4. FVD/IS、precision/recall、density/coverage 和复制审计怎样组成闭环，而不是单指标榜单？

### 1.2 本轮操作性定义

- **Direct / strict pure $p(X)$：** 采样函数不接收类别、域、文本、参考图、视频前缀、动作、状态或样本特定 token；首帧也由模型采样。
- **Restricted-domain unconditional：** 在固定窄域数据内无样本特定输入；正文允许作为域内无条件证据，但明确写成 $p(X\mid D=d)$，不外推开放世界。
- **Mechanism/background：** 只提供 tokenizer、DiT、diffusion、flow、评价等基础机制，不直接证明视频 pure $p(X)$。
- **Adjacent conditional：** class/T2V/I2V/prediction/world model；只在澄清边界或说明可迁移组件时纳入。
- **Self-conditioning：** 若历史完全由模型自身从固定 prior 生成，仍是 $p(X)$ 的因子分解；若历史来自真实数据，就是 prefix-conditioned。

任务归类以**部署数据流和官方采样代码**优先于摘要中的任务名称。论文结果只在其实际数据、分辨率、长度、条件和评价协议内成立。

## 2. 检索入口与实际执行

检索日期均为 2026-08-30（Asia/Shanghai）。本轮使用四类入口，满足“不依赖单一数据库”的要求：预印本 API、文献元数据 API、正式 proceedings/出版页、官方代码/项目页。

### 2.1 预印本 API：arXiv

| 查询 | API 返回 | 实际筛选 | 用途 |
|---|---:|---:|---|
| `all:"unconditional video generation"` | 15 | 15 个标题全部查看 | 找直接自称无条件的候选与 2025–2026 边界 |
| `all:"video generation" AND all:"masked autoregressive"` | 4 | 4 个标题全部查看 | 查 MAGI/MarDini 等 masked-AR 路线 |
| `all:"video generation" AND all:"flow matching"` | 93 | 按最新排序前 50 | 查 flow/ODE 是否出现严格 pure $p(X)$ 新节点 |

精确短语查询中进入全文/官方页复核的直接候选包括：Video Diffusion Models（2204.03458）、RAVEN（2401.06035）、StyleInV（2308.16909）、Generative Video Bi-flow（2503.06364）、Inference-based GAN Video Generation（2512.21776）。其余标题多为预测、条件生成、特定医学/行为域或只在正文中附带无条件分支；未因摘要出现关键词就自动纳入。

Masked-AR 查询返回 CanvasMAR、异构 action-video 工作、MAGI、MarDini。只有 MAGI 同时具备正式 CVPR 2025 页面和本章需要的 UCF-101 无条件实验；其他工作不是本章里程碑所需的严格 pure $p(X)$ 直接证据，或仍为预印本/条件任务。

Flow-matching 前 50 个结果多数为 T2V、I2V、world model、视频编辑或采样加速。该切片没有提供一个可替代本章现有正式证据链的“开放域 pure $p(X)$ foundation milestone”。这只是本次查询的结论，不声称穷尽 93 个结果。

### 2.2 文献元数据 API：OpenAlex、Crossref、Semantic Scholar

| 入口 | 查询/响应 | 使用方式 | 限制处理 |
|---|---|---|---|
| OpenAlex | `unconditional video generation`，1999-01-01 至 2026-08-30；宽泛 count 27,244 | 只作候选发现 | 词法召回极宽，后续请求触发 rate limit；不把 count 写成相关论文数 |
| Crossref | title query `unconditional video generation`；宽泛 total 862,735 | DOI/题名交叉核验 | fuzzy total 明显不可解释；仅保留精确题名/DOI 命中 |
| Crossref | title query `masked autoregressive video generation` | 核验 MAGI DOI/venue 元数据 | 后续请求受 rate limit；venue 仍回到 CVF 官方页确认 |
| Semantic Scholar | `unconditional video generation`, limit 20 | 尝试独立发现入口 | HTTP 429；记录为失败入口，不声称“零结果” |

元数据入口的原则是“发现候选，不承担关键技术事实”。所有进入正文的架构、条件合同和评价主张都回到论文正文、正式 proceedings、出版商页面或官方代码。

### 2.3 正式 proceedings / 出版商入口

定向检查了以下一手站点：

- [NeurIPS Proceedings](https://proceedings.neurips.cc/)：VGAN、VQ-VAE、DDPM、VDM、MCVD、LongVideoGAN、IS、precision/recall；
- [CVF Open Access](https://openaccess.thecvf.com/)：TGAN、MoCoGAN、StyleGAN-V、MAGVIT、MAGI、PVDM、DiT、FVD content bias、replication、Video Bi-flow；
- [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5950_ECCV_2022_paper.php)：TATS；
- [OpenReview](https://openreview.net/)：DIGAN、Flow Matching、Latte；
- [PMLR](https://proceedings.mlr.press/v119/naeem20a.html)：Density/Coverage；
- [Springer DOI / version of record](https://doi.org/10.1007/s00354-026-00326-8)：2026 temporal SSM 视频 diffusion；
- [ICIP 2025 program record](https://cmsworkshops.com/ICIP2025/view_paper.php?PaperNum=1891)：RAVEN venue 状态。

### 2.4 官方代码与项目页入口

以下官方仓库/项目页用于核对“实际 sample API 喂什么”，不把 README 的宣传语提升为论文结论：

| 系统 | 官方入口 | 核查重点 |
|---|---|---|
| TGAN | [pfnet-research/tgan](https://github.com/pfnet-research/tgan) | 作者实现存在、训练/采样路线 |
| VideoGPT | [wilson1yan/VideoGPT](https://github.com/wilson1yan/VideoGPT) | `n_cond_frames=0` 与可选 `class_cond` 是不同开关 |
| MAGVIT | [google-research/magvit](https://github.com/google-research/magvit) | 代码归档状态；仓库说明 model weights 未获发布批准 |
| Latte | [Vchitect/Latte](https://github.com/Vchitect/Latte) | 类别条件与无条件配置/脚本并存 |
| MAGI | [magivideogen.github.io](https://magivideogen.github.io/) | 项目样例与论文链接；未发现同页公开官方代码链接 |
| Video Bi-flow | [ryushinn/ode-video](https://github.com/ryushinn/ode-video) | 采样从测试 split 读取首帧，故严格合同为 prefix/first-frame conditional |
| SSM Meets VDM | [shim0114/SSM-Meets-Video-Diffusion-Models](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models) | 低分辨率配置、长序列任务和代码可用性 |
| Sora | [OpenAI technical report](https://openai.com/index/video-generation-models-as-world-simulators/) | 官方公开条件接口与技术范围，不用二手榜单 |

## 3. 纳入、排除与证据等级

### 3.1 纳入标准

至少满足一项：

1. 提供 direct pure/restricted-domain $p(X)$ 的新训练或采样能力；
2. 是 GAN、VQ/token、masked、diffusion、DiT、flow 的必要祖先机制；
3. 是评测/记忆化的原始方法或直接视频审计；
4. 是容易被误列为无条件里程碑、必须由一手证据纠正的邻接系统；
5. 对 2025–2026 边界提供正式发表或明确 preprint 状态。

每条进入正文的里程碑必须能回答：输入合同是什么、能力转折是什么、数据/长度/分辨率边界是什么、证据是什么级别。

### 3.2 排除标准

- 仅 T2V、I2V、V2V、插值、视频编辑、动作条件预测或 world model，且不承担边界教学功能；
- 真实首帧/前缀是采样必需输入，却没有独立 $p(x_1)$ 的完整系统；
- 只提供精选 demo、没有论文/官方技术说明支撑可比较结论；
- survey、媒体报道、博客排行榜用于替代原论文；
- 相同方法的重复 arXiv/proceedings 记录；正式版优先；
- 模型名含 “foundation / world / diffusion / flow” 但没有本章严格任务证据；
- 只报告静态图像指标或不能恢复视频采样合同。

### 3.3 证据等级

| 等级 | 定义 | 正文允许承担的主张 |
|---|---|---|
| **E1** | 正式 proceedings / 期刊 version of record / 原始方法论文 | 架构、实验范围、正式 venue、论文明确局限 |
| **E2** | arXiv/技术报告，无本轮可核正式版 | 探索方向与作者报告结果；必须标 preprint |
| **E3** | 作者/机构官方代码或项目页 | API、配置、代码/权重可用性；不能单独证明性能 |
| **E4** | 官方公司技术报告/产品说明 | 产品/系统自述与公开条件接口；不作为独立 benchmark |
| **E5** | 元数据 API/搜索结果 | 发现候选、核对 DOI；不承担技术结论 |

## 4. 正文引用闭环与分类登记

下表的 Ref 与正文参考文献编号一一对应。`Direct?` 指是否直接支持严格或已声明窄域的无条件视频生成，而不是“这篇论文是否重要”。

| Ref | 证据 | 等级 | Direct? | 本章承担的事实 |
|---:|---|---|---|---|
| 1 | Video Textures, SIGGRAPH 2000 | E1 | 否 | 给定 source 内重排，不是 dataset-level $p(X)$ |
| 2 | Dynamic Textures, IJCV 2003 | E1 | 窄域 | 线性动态随机纹理源 |
| 3 | VGAN, NeurIPS 2016 | E1 | 是 | 从噪声一次生成完整短 clip；背景/前景拆分 |
| 4 | TGAN, ICCV 2017 + code | E1+E3 | 是 | temporal latent generator 与 image generator 分工 |
| 5 | MoCoGAN, CVPR 2018 | E1 | 是 | content/motion latent 与双 critic 归纳偏置 |
| 6 | DVD-GAN, arXiv 2019 | E2 | **否** | class embedding；双判别器尺度；UCF 记忆化警告 |
| 7 | DIGAN, ICLR 2022 | E1 | 是 | 坐标隐式连续视频与 dynamics-aware critic |
| 8 | StyleGAN-V, CVPR 2022 | E1 | 是 | 连续时间与稀疏训练；不等于无限期语义一致 |
| 9 | LongVideoGAN, NeurIPS 2022 | E1 | 是 | 长低分辨率 + 短高分辨率分层生成 |
| 10 | RAVEN, arXiv + ICIP 2025 record | E1/E2 | 窄域 | 高效混合 3D 表示；formal program 状态 |
| 11 | Inference-based GAN Video Generation | E2 | 是 | 2025 GAN 探索；冻结日仍为 preprint |
| 12 | VQ-VAE, NeurIPS 2017 | E1 | 机制 | 离散 codebook + learned prior 基础 |
| 13 | VideoGPT, arXiv + code | E2+E3 | 可配置 | VQ 视频 token + AR；`n_cond_frames=0` 可核 |
| 14 | TATS, ECCV 2022 | E1 | 是 | 3D-VQGAN + 时间敏感 Transformer 长续写 |
| 15 | MAGVIT, CVPR 2023 + code | E1+E3 | 相关表否 | masked 并行生成；UCF 表为 class conditional |
| 16 | MAGVIT-v2, ICLR 2024 | E1 | 机制 | tokenizer 质量改变后续生成上限 |
| 17 | MAGI, CVPR 2025 + project | E1+E3 | 是 | UCF 无条件 masked-frame / causal-time 实验 |
| 18 | DDPM, NeurIPS 2020 | E1 | 机制 | diffusion 目标/采样祖先 |
| 19 | Video Diffusion Models, NeurIPS 2022 | E1 | 是 | UCF/Kinetics 无条件分支与 prediction 适配分开 |
| 20 | MCVD, NeurIPS 2022 | E1 | 是/多任务 | mask 合同切换 prediction/interpolation/unconditional |
| 21 | PVDM, CVPR 2023 | E1 | 是 | projected latent diffusion 的无条件视频生成 |
| 22 | DiT, ICCV 2023 | E1 | 机制 | 图像 latent Transformer scaling 祖先，不是视频直接结果 |
| 23 | Latte, TMLR 2025 + code | E1+E3 | 是/可配置 | factorized video DiT 的无条件与类别条件配置 |
| 24 | Flow Matching, ICLR 2023 | E1 | 机制 | 连续 flow/ODE 训练框架 |
| 25 | Video Bi-flow, ICCV 2025 + code | E1+E3 | **否，需首帧** | 官方 sampling path 读取 first frame |
| 26 | SSM Meets VDM, NGC 2026 + code | E1+E3 | 窄域 | 低分辨率 256 帧、temporal SSM 效率探针 |
| 27 | FVD original | E2 | 评价 | FVD 定义与视频 feature 分布距离 |
| 28 | Inception Score original, NeurIPS 2016 | E1 | 评价 | IS 定义与分类器依赖 |
| 29 | PRD, NeurIPS 2018 | E1 | 评价 | precision/recall 分离 fidelity/diversity |
| 30 | Improved P/R, NeurIPS 2019 | E1 | 评价 | 局部流形 support 估计 |
| 31 | Density/Coverage, ICML 2020 | E1 | 评价 | 局部密度和覆盖指标 |
| 32 | FVD content bias, CVPR 2024 | E1 | 评价 | 时间扰动敏感度不足的直接审计 |
| 33 | Video replication, WACV 2025 | E1 | 评价 | 无条件/条件视频 diffusion 的时空复制 |
| 34 | Sora official report | E4 | **否** | 条件 video system；只能作为邻接技术 |
| 35 | Cosmos official paper | E2/E4 | **否** | physical-AI/world family；不是 pure benchmark |

## 5. 关键冲突的逐项处理

### 5.1 DVD-GAN

论文生成器输入包含随机 $z$ 与 learned class embedding；Kinetics-600 主结果是类别条件。处理：正文保留其“空间/时间双判别器的尺度设计”，但 milestone 表明确标 **class conditional，不是 pure $p(X)$**。论文对 UCF 训练复制的讨论同时进入评价章节。

### 5.2 MAGVIT 与 MAGVIT-v2

MAGVIT 把多任务放在一个框架中，但 UCF generation 与 Kinetics prediction 的条件不同。处理：不把多任务表整块称为无条件结果；MAGVIT-v2 只承担 tokenizer 机制转折，不承担 pure unconditional milestone。

### 5.3 MAGI

对 CVPR PDF 的方法和实验页进行了逐页检查：帧内 masked、帧间 causal；UCF-101 表使用 2,048 样本；不同 VAE 显著改变 FVD；超过 100 帧的展示主要位于简单/首帧条件场景，非周期运动有退化。处理：肯定其 formal pure-UCF 实验，但不把条件长样例扩写成“解决长期无条件生成”。

### 5.4 Video Bi-flow

论文用“unconditional”描述部分设置，但官方代码的 sampling 说明要求首帧由数据提供或另行生成。处理：以部署路径优先，写为 $p(x_{2:T}\mid x_1)$。只有未来若补齐可独立采样并联合评估的 $p(x_1)$，才能形成完整 $p(X)$。

### 5.5 Sora 与 Cosmos

Sora 的公开报告聚焦文本/媒体条件视频与 world-simulator 讨论；Cosmos 是 physical-AI world foundation 平台/家族。处理：二者只出现在任务边界和邻接组件讨论，不进入 direct unconditional milestone。公开的大模型规模、条件能力或“world”命名都不能替代 pure $p(X)$ 采样证据。

### 5.6 null condition / classifier-free guidance

条件模型的 null embedding 是训练分布中的一个分支，可能带 condition dropout 与 guidance 偏置。处理：正文写成 $p_\theta(X\mid\varnothing)$，要求与专门训练的 $p_\theta(X)$ 分开报告；不把空 prompt 当作自动任务转换。

## 6. 明确排除项

以下类别没有作为 direct milestone 计入，但必要时可在别章讨论：

- Stable Video Diffusion、Lumiere、AnimateDiff、Open-Sora 等以 I2V/T2V 或条件接口为主要证据的系统；
- 未来帧预测、插值、视频补全、视频编辑与 reference-driven identity generation；
- action-conditioned prediction、robotics world model、game/physical simulator；
- 仅需真实首帧的 rollout（包括当前官方 Video Bi-flow sampling path）；
- StyleInV 等未给本章带来独立、正式、可验证转折且会重复连续 GAN 叙事的候选；
- CanvasMAR、MarDini 等本轮查询命中的预印本/条件路线，因 formal/direct 条件不足未提升为里程碑；
- Sora/Cosmos 的模型家族名称，不作为 pure unconditional benchmark；
- 二手 survey、博客、leaderboard、媒体报道，不用来支持原始架构/性能事实。

## 7. 数据与评价证据的保守解释

1. 不跨论文直接排序 FVD，除非 encoder/checkpoint、$T$/FPS/分辨率、样本数、真实 split 和预处理一致。
2. MAGI 的 2,048-sample 表只作为该论文合同内结果；不与 10,000-sample 或 train-set FVD 混排。
3. PVDM、DVD-GAN 等论文采用的 UCF/Kinetics 划分和 preprocessing 可能不同；正文不抄单一数值榜单。
4. IS 不比较真实分布且可被记忆化提高；只能作为域适配 classifier 下的诊断。
5. Precision/recall 与 density/coverage 原始方法主要针对图像 embedding；视频应用必须新增 video feature 与时间扰动 sanity check。
6. 长尾分析需预先分层并做 macro 指标；没有标签时采用冻结 embedding 聚类并人工复核，不事后为某模型挑分组。
7. 复制审计同时查训练集和 held-out 集，覆盖单帧、局部 motion、整 clip 以及轻微时空变换。

## 8. 搜索覆盖与不可声称内容

- arXiv 三组查询存在重叠，targeted proceedings 又来自 citation chasing；因此不输出伪 PRISMA “唯一论文总数”。
- OpenAlex/Crossref 的宽泛 totals 是检索器词法行为，不代表相关文献规模。
- Semantic Scholar 429 代表入口不可用，不代表没有文献。
- flow query 只筛了按最新排序前 50/93；结论限定为“该切片没有发现直接开放域 pure milestone”。
- 正式 venue、代码存在与模型权重可用是三个不同事实；例如 MAGVIT 仓库存在但权重发布受限。
- 截止日结论是本轮证据审计结果，不声称证明某个未公开工业系统绝对没有无条件能力。

## 9. 交付验证记录

2026-08-30 实际执行结果：

1. `npx --yes markdownlint-cli2 docs/tasks/unconditional-video-generation.md sources/research_20260830_unconditional_video.md`，版本 `markdownlint-cli2 v0.23.2` / `markdownlint v0.41.1`，结果 **0 issues**。
2. Python 引用审计：正文 35 个 citation numbers、35 个唯一 anchors，范围均为 1–35；missing、unused、duplicate 均为空。
3. 本地链接审计：正文、研究记录与教学图的相对链接均可解析到现存文件；`git diff --check` 通过。
4. 外链 GET 检查：去重后 46 个 URL 中 44 个返回 `<400`；ACM DOI 跳转页与 OpenAI Sora 页面各返回 403（反爬/访问限制），没有其他坏链。两项均保留官方 canonical URL，不把机器 403 误记为文献不存在。
5. Mermaid CLI `11.16.0` 配合本机 Google Chrome，两个 Mermaid block 均实际渲染为临时 SVG：约 28 KB 与 41 KB；SVG 含对应 `<title>` / `<desc>`。临时文件位于 `/tmp`，未纳入仓库。

实际渲染采用独立随机输入之外的纯文档解析，不改变正文。最终复跑 markdownlint 和引用审计，以确认作者元数据修正没有引入回归。

## 10. 图像资产记录

- 项目文件：[`assets/diagrams/unconditional-video-evidence-chain.png`](../assets/diagrams/unconditional-video-evidence-chain.png)
- 生成提示核心：白底、16:9、无模型名；主链为“无外部条件 → 学习视频边际分布 → 采样动态片段 → 统一输出合同 → 多维评测”，并将文本、首帧或动作输入标为任务越界，将质量、覆盖、长尾和未复制拆开。
- 像素尺寸：1672 × 941，RGB PNG。
- SHA-256：`3992c5637ef25fe2e42d821f6a710082de7ceb15ec0c477a2224e864e09cbf42`。
- 视觉回读：五步证据链、越界出口、四个评测轴和三项固定条件均清晰；没有模型名、分数或与正文冲突的能力主张。正文同时保留 Mermaid 与顺序化文字替代。
