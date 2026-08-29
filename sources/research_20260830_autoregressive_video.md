# 自回归视频生成：截至 2026-08-30 的检索、核验与证据边界

> 截止时间：2026-08-30（Asia/Shanghai）
> 用途：支撑 docs/generative-models/autoregressive-generation.md 的深度重写
> 性质：一手来源研究记录，不是模型排行榜，也不是“开源可商用”清单

## 📋 1. 审计对象与结论摘要

### 1.1 改写前快照

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| docs/generative-models/autoregressive-generation.md | 74 | c0af6b2f5ca6d5e184187614828e664409a658f11b9987eca0783a5beba5471b |

改写前正文只有 5 条参考文献，主要问题是：

1. 把 AR 近似限定为离散 token + cross-entropy，没有 continuous token + diffusion head。
2. 把 pixel、token、frame 与 chunk 放在同一列表中，却没有定义 commit granularity。
3. 将 Phenaki 写成“因果生成”，没有区分时间因果 tokenizer 与 bidirectional masked generator。
4. 没有说明 MAGVIT-v2 同时讨论 AR-LM 与 MLM，容易把 tokenizer 贡献误算成 strict AR 证据。
5. 没有 strict token AR 与 generalized set/frame/chunk AR 的教材约定。
6. teacher forcing、complete teacher forcing、self forcing 与 exposure bias 只有名词，没有历史分布公式。
7. KV cache 只有一句结论，没有累计 attention work、内存增长与当前 noisy unit 的缓存边界。
8. 没有区分 per-token small diffusion head 与每个 chunk 调用 full video diffusion backbone。
9. 把 causal、few-step、streaming 与 real-time 相邻描述，缺少正交维度。
10. 缺 2024–2026 的正式里程碑、发布面与速度数字不可横比的说明。

### 1.2 核心裁决

本次采用以下局部术语：

$$
\text{strict AR}:
\quad
p(y_{1:N}\mid c)
=
\prod_{i=1}^{N}
p(y_i\mid y_{<i},c),
$$

$$
\text{block / set AR}:
\quad
p(Y_{1:K}\mid c)
=
\prod_{k=1}^{K}
p(Y_k\mid Y_{<k},c).
$$

第二式不保证 $Y_k$ 内部也是 fixed-order next-token；它可能由双向 attention、masked refinement、discrete diffusion 或 continuous diffusion 联合/并行生成。

本次把系统拆成五列：

$$
\text{representation}
\times
\text{factorization}
\times
\text{conditional objective}
\times
\text{backbone}
\times
\text{deployment}.
$$

其中 commit unit 是 factorization 与 deployment 之间的关键接口：只有已经确定、不再被当前单元内部迭代改写的输出，才可安全成为下一单元的因果历史。

## 🔎 2. 检索面、检索式与日期

### 2.1 四种检索表面

| 检索表面 | 2026-08-30 使用的检索式 / 路径 | 用途 | 局限 |
|---|---|---|---|
| 正式 proceedings | site:proceedings.mlr.press、site:proceedings.neurips.cc、site:proceedings.iclr.cc、site:openaccess.thecvf.com，搭配论文全名 | 核验正式标题、venue、年份、作者、摘要主张 | CVF HTML 偶发 403，改用正式 PDF URL 或会议索引 |
| arXiv API | all:"autoregressive video generation"；all:"masked autoregressive video generation"；all:"autoregressive video diffusion" | 找到 2025–2026 新工作及术语冲突 | 命中数动态变化；预印本不自动升级为正式结论 |
| 官方项目 / 代码 / 模型页 | 论文全名 + official GitHub / project / Hugging Face；再回查论文作者与仓库组织 | 核验代码、权重、配置、许可证标签与发布空白 | 项目页可以晚于论文更新；代码许可证不等于权重、数据许可 |
| 正文与引用链 | 从 MAR、NOVA、MAGI、CausVid、Self Forcing、Lumos-1 正式 PDF 的方法与参考文献反向追踪 | 核验 factorization、训练前缀、两类 step 与 cache 语义 | 单篇论文的相关工作表述不能替代被引论文原文 |

arXiv API 的实际请求模板为：

~~~text
https://export.arxiv.org/api/query?
search_query=all:%22autoregressive%20video%20generation%22
&start=0
&max_results=10
&sortBy=submittedDate
&sortOrder=descending
~~~

另外两次只替换 search_query 为 masked autoregressive video generation 与 autoregressive video diffusion。每个 query 只初筛按提交日期排序的前 10 条，再回到正式 venue 或作者页核验；不记录不可复现的搜索结果总数。

### 2.2 代表性正式页检索式

~~~text
site:proceedings.mlr.press "Video Pixel Networks" OR "VideoPoet"
site:proceedings.neurips.cc "Autoregressive Image Generation without Vector Quantization"
site:proceedings.iclr.cc "Autoregressive Video Generation without Vector Quantization"
site:proceedings.iclr.cc "Lumos-1" OR "LongLive" OR "Flow Caching"
site:openaccess.thecvf.com "Taming Teacher Forcing" OR
  "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models"
~~~

### 2.3 纳入与排除

纳入：

- 原始方法论文；
- 正式同行评审 proceedings；
- 作者/机构项目页；
- 论文作者或官方组织的代码、模型卡与配置文档；
- 只要用于定义通用 objective，可纳入图像论文，但必须标明不是视频效果证据。

排除：

- 聚合排行榜、新闻稿、二手教程；
- 名称相同但作者不一致的社区复现，除非只用于说明“不是官方发布”；
- 只有 demo、没有论文或任务设置的速度宣传；
- 无法区分 batch throughput 与逐流 latency 的比较；
- 截至截止日仍只有预印本、且不影响核心 taxonomy 的新增 paper name 堆叠。

## 🧾 3. 证据等级与断言规则

| 等级 | 来源 | 可以支持 | 不能自动支持 |
|---|---|---|---|
| A | 正式会议 proceedings / publisher page | 标题、venue、方法定义、论文设置内结果 | 普适最优、商业可用、跨硬件速度比较 |
| B | 作者 arXiv / 正式技术报告 | 原作者提出的方法与报告结果 | 已同行评审共识 |
| C | 官方项目、GitHub、Hugging Face model card | 当时公开物、配置、代码/权重许可证标签 | 论文所有实验可复现、训练数据权利、代码与权重同许可 |
| S | 本记录的综合判断 | 术语裁决、章节结构、不可横比项 | 新的实验结论 |

断言写法：

- “论文提出 / 作者报告”保留原始主张主体；
- “正式 ICLR/CVPR/NeurIPS/ICML/MLSys”只在 proceedings 核验后使用；
- “公开代码 / 权重”必须给官方 artifact URL；
- “未发现”限定为本次检查的正式页与项目页，不证明整个互联网绝对不存在；
- “可商用”不从 Apache/MIT 代码许可证外推到模型权重、训练数据或生成内容。

## 🧠 4. 概念事实表

### 4.1 Representation 与 factorization

| 路线 | 生成变量 | 外层分解 | 条件分布 | 直接一手证据 | 裁决 |
|---|---|---|---|---|---|
| PixelRNN / Video Pixel Networks | 原始离散像素通道 | fixed total order | categorical | ICML 2016 / 2017 | strict pixel AR |
| VideoGPT | VQ 视频 token | fixed spacetime order | categorical | 2021 作者 arXiv + repo | strict discrete-token AR |
| VideoPoet | 文本/图像/视频/音频离散 token | decoder-only sequence | categorical mixture objectives | ICML 2024 | strict discrete-token AR |
| Phenaki | C-ViViT 离散 token | 生成器多轮 masked completion | bidirectional masked Transformer | ICLR 2023 / Google Research | 不是 strict token AR；只有 tokenizer 时间因果 |
| MAGVIT-v2 | lookup-free 离散 token | AR-LM 或 MLM 均可 | categorical next-token 或 masked iterative | ICLR 2024 | tokenizer 不决定 factorization |
| MAR | 连续 image latent token | strict 或 generalized masked AR | per-token Diffusion Loss | NeurIPS 2024 | continuous AR 的通用 objective 证据，主要不是视频证据 |
| NOVA | 连续 video latent | frame-by-frame + spatial set-by-set | per-token diffusion head | ICLR 2025 | outer temporal AR + generalized spatial set AR |
| MAGI | 视频 latent / frame unit | inter-frame causal | intra-frame masked modeling | CVPR 2025 | hybrid frame AR，不是 strict token AR |
| CausVid | 连续视频 latent | causal frame/chunk rollout | DMD-distilled few-step video diffusion | CVPR 2025 | factorization + distillation + deployment 三层 |
| Self Forcing | 连续视频 latent | causal frame/chunk rollout | few-step diffusion + video-level loss | NeurIPS 2025 | history source 改为 self rollout |
| InfinityStar | 离散 spacetime token | spacetime-pyramid next-scale AR | scale-wise discrete prediction | NeurIPS 2025 | 高分辨率纯离散 generalized AR，非 strict next-token |
| Lumos-1 | 离散视频 token | inter-frame causal | intra-frame mask-based discrete diffusion | ICLR 2026 | frame-wise outer AR，非 strict next-token |

### 4.2 两种 diffusion head 不能混写

MAR/NOVA：

$$
h_i=F_\theta(x_{<i},c),
$$

$$
\mathcal L
=
\mathbb E
\left\|
\epsilon-D_\phi(x_i^\tau,\tau,h_i)
\right\|^2.
$$

这里 $D_\phi$ 是为一个连续 token 条件密度服务的小 denoising head。外层 Transformer 产生 $h_i$，内层做 $D$ 次小 head evaluation。

CausVid/Self Forcing 类：

$$
\hat B_k
=
\operatorname{VideoDenoise}_\theta
\left(
z_k;\hat B_{<k},c
\right),
$$

每个 frame/chunk 内部调用的是完整或主要视频 denoising backbone。两者都有 data-time commit 与 noise-time NFE，但网络调用大小和 cache 行为不同。

### 4.3 NOVA 的嵌套口径

NOVA 正式摘要明确使用 temporal frame-by-frame 与 spatial set-by-set。方法层可表示为：

$$
p(\mathbf S_{1:F}\mid c)
=
\prod_f
p(\mathbf S_f\mid\mathbf S_{<f},c),
$$

$$
p(\mathbf S_f\mid\mathbf S_{<f},c)
=
\prod_k
p(\mathbf S_{f,k}\mid
\mathbf S_{f,<k},\mathbf S_{<f},c).
$$

官方 repo 又分别暴露 num_inference_steps 与 num_diffusion_steps。前者是 AR/set 提交轮次，后者是 per-token diffusion head 的内部步数；不能合并为一个 NFE 数。

### 4.4 InfinityStar 的准确分类

InfinityStar 正文明确说明其工作受 VAR / Infinity 的 next-scale prediction 启发，并将其扩展为 Spacetime Pyramid Modeling。视频被组织成 image pyramid 与多个 clip pyramids，外层从粗尺度走向细尺度。

处理：

- 表示仍是 purely discrete；
- AR 顺序发生在 spacetime pyramid scales；
- 一个尺度内的 token map 不是 raster-scan next-token；
- 论文的 720p / 5 秒 / 约 10 倍速度是这一 next-scale 配置的作者报告。

### 4.5 Lumos-1 的准确分类

ICLR 2026 正式摘要同时给出：

- inter-frame causal attention；
- intra-frame bidirectional attention；
- parallel mask-based discrete diffusion；
- Autoregressive Discrete Diffusion Forcing 与 temporal tube masking。

因此教材写为：

$$
\text{discrete representation}
\times
\text{frame-level outer AR}
\times
\text{within-frame discrete diffusion}.
$$

“不是 strict next-token AR”不是贬义判断，而是对其并行化贡献的准确描述。

## 🎓 5. Teacher forcing 争议表

| 名称 | 训练历史来源 | 历史是否完整 | 是否直接暴露在自身错误上 | 主要处理的问题 |
|---|---|---|---|---|
| 标准 teacher forcing | ground truth | 依实现 | 否 | 稳定、并行的条件训练 |
| MAGI masked teacher forcing | ground truth | 历史中有 mask | 否 | 与 masked frame objective 配合 |
| MAGI complete teacher forcing | ground truth | 完整、未 mask | 否 | 移除训练上下文的人为遮挡 |
| Self Forcing | model rollout | 自生成历史 | 是 | train–test history gap |
| CausVid | teacher / real data 构造的蒸馏监督 | 取决于蒸馏路径 | 不等同于完整 on-policy rollout | 双向 teacher 到 causal few-step student |

CTF 不等于 self forcing。它让 ground-truth history 更完整，但没有让模型在训练时承受自己早先生成的偏差。

Self Forcing 同时使用 few-step model 与 stochastic gradient truncation；它的结果不能只归因于 history source。CausVid 则同时改变 teacher–student objective、视频时间 factorization 与 noise-time step 数。

## ⚙️ 6. KV cache、串行成本与部署边界

### 6.1 复杂度口径

strict token AR 若每一步重算长度 $i$ 的完整前缀，仅 attention 累计工作为

$$
\sum_{i=1}^{N}O(i^2)=O(N^3).
$$

有 KV cache 时，每步新 query 对历史 key 的工作为

$$
\sum_{i=1}^{N}O(i)=O(N^2).
$$

这是 attention 部分的渐近式，不含 MLP、projection、kernel launch 与 bandwidth。缓存没有消除 $N$ 次 commit 依赖。

保存全部历史的 KV 内存近似

$$
O(LN_{\mathrm{hist}}d_{\mathrm{KV}}b).
$$

block/frame/chunk AR 中，clean committed history 可缓存；当前 noisy 或 masked unit 在各 refinement/denoising round 中会变化，通常不能直接永久缓存。

### 6.2 2026 cache / deployment 工作应放在哪一列

| 工作 | Factorization | Objective | Cache / deployment 改动 | 本章使用方式 |
|---|---|---|---|---|
| LongLive | frame-level causal AR | 基础视频 diffusion 的训练与 long tuning | KV-recache、short window、frame sink | factorization 与 prompt/cache 一起说明 |
| FlowCache | 继承 MAGI-1 / SkyReels-V2 等基础模型 | training-free，不改训练 objective | chunkwise adaptive feature reuse + bounded KV compression | 作为 cache 层代表，不称新 AR 家族 |
| AR-Drag | few-step AR video diffusion | motion reward + reinforcement learning | 目标是实时 motion control | 作为 objective/control 扩展 |
| StreamDiffusionV2 | 继承基础 video diffusion | training-free | rolling KV、SLO scheduler、pipeline | 作为 serving 反例：streaming 不定义 AR |

FlowCache 的 “cache” 同时涉及两种对象：

1. denoising step 之间的 feature reuse；
2. 已完成历史 chunk 的 KV compression。

它不是标准 LLM next-token KV cache 的简单同义词。

## 📊 7. 速度、分辨率与时长为什么不可横比

| 论文主张 | 原文口径 | 不能直接比较的因素 |
|---|---|---|
| NOVA 高效连续 AR | 论文摘要为 0.6B；repo 可分别调 AR 与 diffusion steps | 模型、step、batch、分辨率与 per-token head 成本 |
| CausVid 4-step / 9.4 FPS | 双向 teacher 到 causal student；单 GPU 作者报告 | 摘要未给可与所有方法对齐的端到端设置 |
| Self Forcing sub-second latency | 单 GPU、few-step、self-rollout | latency 起止点、分辨率、帧率与 decoder |
| InfinityStar 720p / 5 s / 约 10 倍 | next-scale spacetime pyramid；论文特定模型与 diffusion baselines | tokenizer、硬件、尺度数、采样配置、质量匹配 |
| LongLive 20.7 FPS / 240 s | 单 H100、1.3B、作者设置 | 分辨率、精度、attention window、解码与 prompt switch |
| FlowCache 2.38 倍 / 6.7 倍 | 分别在 MAGI-1 / SkyReels-V2 上 | base model 不同，VBench 改变量方向也不同 |
| StreamDiffusionV2 58.28 / 64.52 FPS | 4×H100、14B / 1.3B、系统 pipeline | 多 GPU、并行流数、NFE、SLO 与 batch scheduler |

这些数字只用于说明论文优化对象，不生成跨论文速度排序。可比实验至少要锁定：

- 相同模型与 checkpoint；
- 相同输出分辨率、时长与目标 FPS；
- 相同 guidance、outer commits 与 inner NFE；
- 相同精度、batch、GPU 与 warm-up；
- 相同 tokenizer / decoder 是否计时；
- 相同 TTFF、稳态 latency 或吞吐定义；
- 相同质量或 mode-coverage 约束。

## 📚 8. 正式论文与原始方法来源 registry

| ID | 年份 / 状态 | 一手来源 | 用于核验 |
|---|---|---|---|
| P01 | ICML 2016 | [Pixel Recurrent Neural Networks](https://proceedings.mlr.press/v48/oord16.html) | pixel strict AR |
| P02 | ICML 2017 | [Video Pixel Networks](https://proceedings.mlr.press/v70/kalchbrenner17a.html) | 四维视频像素依赖链 |
| P03 | NeurIPS 2017 | [Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | VQ representation |
| P04 | arXiv preprint 2021 | [VideoGPT](https://arxiv.org/abs/2104.10157) | VQ video + GPT prior；venue 边界 |
| P05 | CVPR 2022 | [MaskGIT](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html) | masked iterative 对照 |
| P06 | ICLR 2023 | [Phenaki](https://openreview.net/forum?id=vOEXS39nOF) | causal tokenizer / masked generator |
| P07 | ICLR 2024 | [Language Model Beats Diffusion - Tokenizer is key to visual generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html) | MAGVIT-v2 tokenizer 与 MLM/AR-LM 边界 |
| P08 | ICML 2024 | [VideoPoet](https://proceedings.mlr.press/v235/kondratyuk24a.html) | decoder-only multimodal AR |
| P09 | NeurIPS 2024 | [Autoregressive Image Generation without Vector Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html) | MAR continuous token + Diffusion Loss |
| P10 | ICLR 2025 | [Autoregressive Video Generation without Vector Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6e5112eaa45f8c30b242c5f576213a92-Abstract-Conference.html) | NOVA frame/set factorization |
| P11 | CVPR 2025 | [Taming Teacher Forcing for Masked Autoregressive Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html) | MAGI、MTF/CTF |
| P12 | CVPR 2025 | [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) | CausVid、DMD、few-step causal student |
| P13 | NeurIPS 2025 | [Self Forcing](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) | self-rollout、video-level loss、rolling KV |
| P14 | NeurIPS 2025 | [InfinityStar](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f832f6d70ea73779369142dac61a389f-Abstract-Conference.html) | 纯离散 next-scale spacetime pyramid 与发布声明 |
| P15 | ICLR 2026 | [Lumos-1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59ad89d72559dd4ce557d56f36313724-Abstract-Conference.html) | inter-frame causal + intra-frame discrete diffusion |
| P16 | ICLR 2026 | [LongLive](https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html) | frame AR、recache、sink、速度边界 |
| P17 | ICLR 2026 | [Flow Caching for Autoregressive Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85dc8f85ff978b9c606d3b2f5b0da69a-Abstract-Conference.html) | chunkwise reuse 与 KV compression |
| P18 | ICLR 2026 | [Real-Time Motion-Controllable Autoregressive Video Diffusion](https://proceedings.iclr.cc/paper_files/paper/2026/hash/71c1d6ec1f0003d8ea10bbea4291002d-Abstract-Conference.html) | AR-Drag、few-step + RL |
| P19 | CVPR 2026 | [Causality in Video Diffusers is Separable from Denoising](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html) | data-time causality 与 noise-time denoising 分离 |
| P20 | MLSys 2026 | [StreamDiffusionV2](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | training-free streaming system 与 SLO |

## 📦 9. 官方项目、代码、权重与许可证快照

> 下表只记录 2026-08-30 检查到的公开表面。GitHub 的 LICENSE 只约束对应仓库内容；模型权重、基础模型、训练数据和 demo 输出可能另有条款。

| 工作 | 官方 artifact | 实际可见物 | 许可证 / 边界 |
|---|---|---|---|
| VideoGPT | [wilson1yan/VideoGPT](https://github.com/wilson1yan/VideoGPT) | 训练、采样、评测、Colab 与 VQ-VAE 使用入口 | repo 显示 MIT；论文仍是 arXiv preprint |
| Phenaki | [Google Research publication](https://research.google/pubs/phenaki-variable-length-video-generation-from-open-domain-textual-descriptions/) / [project](https://sites.research.google/gr/phenaki/) | 论文、方法说明、demo | 检查页未给第一方完整训练代码/权重；LAION/lucidrains 等是社区复现 |
| MAGVIT-v2 | [google-research/magvit](https://github.com/google-research/magvit) | repo 自述为 CVPR 2023 MAGVIT 的代码与模型表 | Apache-2.0 且已 archived；不能称完整 MAGVIT-v2 官方发布 |
| VideoPoet | [Google project](https://sites.research.google/videopoet/) | 论文说明与 demo | 检查页未识别到第一方代码/权重入口 |
| MAR | [LTH14/mar](https://github.com/LTH14/mar) | 训练/评测代码与模型入口 | GitHub API / LICENSE 显示 MIT |
| NOVA | [baaivision/NOVA](https://github.com/baaivision/NOVA) | 代码、model zoo、T2I/T2V 推理；分别暴露 AR / diffusion steps | repo 声明 code and models Apache-2.0 |
| MAGI | [magivideogen project](https://magivideogen.github.io/) | 论文与 demo | 页面 Code 按钮 href 在快照中为空；未识别第一方训练代码/权重 |
| CausVid | [GitHub](https://github.com/tianweiy/CausVid) / [Hugging Face](https://huggingface.co/tianweiy/CausVid) | 代码、AR / bidirectional checkpoints 与推理命令 | HF metadata 写 cc-by-nc-4.0；卡片正文链接/文字写 CC BY-NC-SA 4.0，二者冲突；按更严格条款复核，不推断商用许可 |
| Self Forcing | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | 训练/推理代码与模型链接 | repo 显示 Apache-2.0；基础权重仍需查自身条款 |
| InfinityStar | [FoundationVision/InfinityStar](https://github.com/FoundationVision/InfinityStar) | 代码与模型入口 | repo 显示 MIT；论文速度仍为特定设置 |
| Lumos-1 | [alibaba-damo-academy/Lumos](https://github.com/alibaba-damo-academy/Lumos) | inference、fine-tuning 与 checkpoints | repo 显示 Apache-2.0 |
| LongLive | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) / [project](https://nvlabs.github.io/LongLive/) | 训练、推理、模型权重、互动脚本 | repo news 记录许可证改为 Apache-2.0；仍查基础 Wan 权重条款 |
| FlowCache | [mikeallen39/FlowCache](https://github.com/mikeallen39/FlowCache) | MAGI-1 / SkyReels-V2 加速实现 | 仓库根目录未从 GitHub license API 解析到许可证；依赖基础模型许可 |

许可证冲突处理原则：

1. model card 顶部 metadata 与正文许可不一致时，不选择更宽松的一项；
2. code license 不覆盖 checkpoint，除非发布方明确写明；
3. checkpoint license 不解决训练数据版权、隐私或商标风险；
4. community reproduction 的许可不能代表原作者模型许可；
5. 无 LICENSE 不等于默认可自由使用。

## ⚠️ 10. 主要争议与最终处理

### 10.1 “Masked autoregressive” 是否自相矛盾

社区存在两种用法：

- 窄义 AR：固定全序的 next-token factorization；
- 广义 MAR：按随机或学习到的 set 顺序提交，set 内并行。

处理：保留论文原名，但教材第一次出现时声明 strict token 与 generalized set/frame AR 的局部约定。

### 10.2 Phenaki 是不是 AR

论文明确写 tokenizer 在时间上 autoregressive，token generator 是 bidirectional masked Transformer。

处理：不把整个 generator 列为 strict AR；作为“codec causality 不推出 generator causality”的代表。

### 10.3 MAGVIT-v2 是否是 AR 模型

它是视觉 tokenizer 论文，可配 AR-LM，也明确描述 MLM 的 non-autoregressive iterative decoding。

处理：列在离散表示史中，不能把所有 video results 统一称 strict AR。

### 10.4 Continuous token 是否违反 AR

AR 约束联合分布分解，不要求每个条件分布 categorical。MAR 用 per-token Diffusion Loss，NOVA 给出视频正式证据。

处理：分开 outer AR backbone 与 inner density head；增加两个时钟和两类 NFE。

### 10.5 Frame/chunk AR 是否等于 token AR

frame/chunk 只要求过去单元到未来单元 causal；当前单元可 bidirectional denoise。

处理：用 commit unit 明确外层串行深度；Lumos-1 明写“非 strict next-token AR”。

### 10.6 KV cache 是否解决串行瓶颈

KV cache 将历史 attention 重算降为增量 attention，但下一 commit 仍依赖前缀；历史 KV 内存还会增长。

处理：正文同时给累计 attention work、KV memory 和 current noisy unit 不能永久缓存的边界。

### 10.7 Few-step、causal、streaming 与 real-time

它们分别是 noise-time NFE、信息访问、交付方式和 SLO 证据。

处理：单独五列表；用 AR-Drag、FlowCache、StreamDiffusionV2 说明 objective/cache/deployment 可以叠加但不重新定义 factorization。

### 10.8 MAGI 与 MAGI-1 名称碰撞

本记录的 MAGI 指 CVPR 2025 Taming Teacher Forcing for Masked Autoregressive Video Generation。FlowCache repo 支持的 MAGI-1 是另一发布系统。

处理：正文只在正式论文上下文使用 MAGI；FlowCache 速度行保留 “MAGI-1” 原名，不把两者当同一 artifact。

## ✅ 11. 可复核的写作约束

正文必须满足：

- VideoGPT 明标 arXiv preprint；
- Phenaki 明写 causal tokenizer + masked generator；
- MAGVIT-v2 明写 tokenizer / AR-LM / MLM 边界；
- MAR 明标主要图像证据，NOVA 才是直接视频 evidence；
- NOVA 分开 AR steps 与 per-token diffusion steps；
- Lumos-1 明写不是 strict next-token AR；
- MAGI 的 CTF 不得写成 self forcing；
- CausVid 分开 teacher、DMD、few-step、causal factorization 与 KV；
- Self Forcing 分开 self-generated history 与 few-step training；
- FlowCache 只放 cache/deployment，不称新 factorization；
- LongLive / InfinityStar / CausVid / FlowCache 的速度只按作者配置转述；
- StreamDiffusionV2 作为 SLO / system 证据，不作为 AR 定义；
- 至少两张 Mermaid 都有 accTitle、accDescr 与顺序化文字替代；
- 所有参考文献锚点一一对应，无未引用条目。

## 🔗 12. 后续研究边界

本轮没有尝试：

- 穷尽 2026-08-30 当天所有 autoregressive video 预印本；
- 复现实验或重测 FPS；
- 审计每个 checkpoint 的完整训练数据 provenance；
- 把 VBench、FVD、用户研究与 latency 强行聚成单一排名；
- 从 causal attention 推导物理因果或可干预 world-model 能力。

新论文进入正文的门槛应是至少改变一个可验证维度：

1. 新的 representation 或 compression–generation trade-off；
2. 新的 factorization / commit unit；
3. 新的 conditional density objective；
4. 直接处理 self-history distribution gap；
5. 有界 memory 或可复现实测的 serving SLO；
6. 相同设置下的 mode coverage、长期漂移或控制证据。

仅增加一个作者自报 FPS、一个更长 demo 或一个未开放的模型名，不足以改变本章 taxonomy。
