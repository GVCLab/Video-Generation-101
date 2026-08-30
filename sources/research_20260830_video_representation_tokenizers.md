# 视频潜表示、Representation Tokenizer 与 Learned/Generative Codec 研究轨迹（2026-08-30）

> 范围：同时为 `docs/generative-models/variational-generation.md` 与 `docs/generative-models/video-tokenizers.md` 提供可复核证据。本文区分 stochastic latent future model、representation tokenizer、learned codec 与 generative codec，并跟踪至 **2026-08-30** 的正式论文、作者技术报告和官方开放实现。

## 1. 研究问题与边界

本轨迹围绕八个问题：

1. 同样被称为“video VAE”的系统，何时在学未来的条件分布，何时只是为上层 generator 提供压缩表示？
2. ELBO、learned prior 与 posterior collapse 在条件视频预测中分别承担什么角色？
3. 连续、离散与混合 tokenizer 的数据类型、训练目标和上层 generator 接口如何区分？
4. 输入张量、潜张量、网格压缩、元素数压缩、token 数和 bitrate 如何不混淆？
5. causal 3D VAE 的左侧时间填充、首帧锚定和 chunk 边界会带来哪些工程约束？
6. 2025–2026 的自适应 token 预算与 INR-weight latent 是新表示分支，还是旧式特征网格的换名？
7. representation tokenizer 何时只是上层 generator/理解模型的表示接口，何时已具备 learned codec 的概率模型、熵编码和 bitstream 合同？
8. generative decoder 或 diffusion de-tokenizer 何时只是表示解码器，何时才能称为可传输的 generative codec？

不纳入的内容：只有产品演示而无公开技术细节的系统；第三方博客中无法追溯到一手材料的数值；把图像-only 结果直接外推为视频运动结论的主张；只报 latent 尺寸却称已实现可传输 codec 的工作。

## 2. 检索表面、检索式与日期

所有检索及本轮 tokenizer/codec 专项复核均于 **2026-08-30（Asia/Shanghai）** 执行。搜索结果只用于发现，最终事实必须回到正式 proceedings、作者 arXiv 稿或官方仓库/项目页。“未找到正式 venue/bitstream”只表示在本截止日和检索表面中未发现可核验证据，不是对所有未公开材料的否定。

| 检索表面 | 代表检索式 | 用途 | 纳入方式 |
|---|---|---|---|
| ICLR / OpenReview 正式页 | `site:proceedings.iclr.cc 2025 "High-Quality Joint Image and Video Tokenization with Causal VAE"`；`site:proceedings.iclr.cc 2026 InfoTok adaptive video tokenizer`；`site:proceedings.iclr.cc 2026 "NeRV-Diffusion"` | 核对 2025–2026 标题、作者、正式接收状态 | 仅用 proceedings 页确认 venue；OpenReview 用于读全文 |
| NeurIPS / PMLR / CVF / ECVA 正式 proceedings | `site:proceedings.mlr.press Denton Fergus "Stochastic Video Generation"`；`site:proceedings.neurips.cc OmniTokenizer`；`site:openaccess.thecvf.com CVPR 2025 Divot`；`site:ecva.net TATS ECCV 2022` | 核对 2017–2025 基础论文与正式 venue | 优先引用 abstract/conference HTML，必要时读 PDF |
| arXiv 作者稿 | 论文全名加 `arXiv`；`"VidTok" 2412.13061`；`"VideoRAE" 2607.14088`；`"KVAE" 2608.05798`；`"V-RAE" 2608.13556` | 核对尚无正式 venue 的 tokenizer 技术报告及全文 | 明标“preprint / author technical report”，不写成已发表 |
| 官方 GitHub / 模型卡 / 项目页 | `site:github.com/NVIDIA/Cosmos-Tokenizer`；`site:github.com/Tencent-Hunyuan/HunyuanVideo 3D causal VAE`；`site:github.com/kandinskylab/kvae`；`site:huggingface.co microsoft/VidTok` | 核对代码、权重、license、张量形状和压缩配置是否真正公开 | 只支撑仓库已显式写出的操作事实，不把 README 性能宣传视为独立复现 |
| 官方项目研究页 | `Cosmos Tokenizer first temporal token first frame`；`InfoTok ICLR 2026 oral`；`NeRV-Diffusion ICLR 2026` | 补足 proceedings 摘要未展开的方法图和开放实现链接 | 证据等级低于正式论文，不单独支撑宽泛性能结论 |

## 3. 筛选标准与证据等级

### 3.1 纳入标准

- 能直接支撑 VAE/ELBO、learned prior、VQ/VQGAN、causal video autoencoder 或 adaptive/implicit tokenizer 的机制主张。
- 有可核验标题、作者、时间和 URL；对正式发表必须能在官方 proceedings 找到。
- 对“开放模型”，至少有官方代码或权重之一，并在表中分别标记；“可下载权重”不自动等于训练数据、训练脚本和评测全部开放。
- 数值只在它能说明机制或实现边界时纳入，并保留作者评测协议限定。

### 3.2 排除标准

- 无正式论文、作者稿或官方仓库背书的新闻转述。
- 只展示最佳样本、无 reconstruction 和 downstream generation 分离评估的说法。
- 把 compression factor、token reduction 或 float latent 元素数直接当作 bpp/bitrate。
- 把 causal encoder/decoder 外推为上层 generator 也 causal，或外推为端到端实时。

### 3.3 证据等级

| 等级 | 定义 | 可支撑的主张 | 不可支撑的主张 |
|---|---|---|---|
| A | 官方会议 proceedings 或出版社正式论文 | 标题、venue、论文明示方法与实验 | 未公开的产品实现、跨协议通用性 |
| B | 作者 arXiv 稿或机构技术报告 | 公开方法细节与作者实验 | 同行评审或独立复现 |
| C | 作者/机构官方仓库、项目页或模型卡 | 代码与权重可用性、API 形状、配置和 license | README 中“SOTA”或速度宣传的独立有效性 |
| X | 第三方聚合页、博客、新闻、搜索摘要 | 只用于发现线索 | 任何最终教材事实 |

## 4. 核心角色分解

### 4.1 Stochastic latent future model

这一角色的对象是条件分布 $p(x_{C+1:K}\mid x_{1:C})$。随机潜变量用来表达给定同一历史时仍可能出现的多个合理未来。训练时的近似后验 $q_\phi(z_k\mid x_{\le k})$ 可看到目标帧；部署时只能从 learned prior $`p_\psi(z_k\mid x_{\lt k})`$ 采样。SVG-LP 于 ICML 2018 对这一 posterior–prior 分工给出了经典实例。

验收对象是未来分布：条件一致性、多样性、覆盖、校准与长期 rollout。单个最好样本或重建分数不能验证这一角色。

### 4.2 Representation tokenizer

这一角色的对象是变换 $z=E(x)$ 与 $\hat x=D(z)$，目标是让后续 diffusion、flow、AR、masked generator 或理解模型在更小表示上工作。它可以输出 continuous latent、VQ/LFQ/FSQ/BSQ 等 discrete symbol 或受控的 hybrid latent；可以用 KL、感知/对抗损失、semantic alignment 或 generator-aware prior 训练。这些选择都不会自动建立可传输 bitrate 合同。

验收对象是 reconstruction ceiling、表示分布、时间/空间边界、编解码成本和对固定上层消费者的影响。对 continuous latent 只能先报张量形状、数值精度和存储成本；对 discrete token 也不能把 token 数或 $\log_2K$ 直接写成实际码率。

### 4.3 Learned codec

learned codec 在 representation tokenizer 之上增加传输/存储合同：可量化的符号 $q$ 、可用于编码的概率模型 $p(q)$、熵/算术编码器 $C$ 与可解析 bitstream $b=C(q;p)$。码率必须由实际字节数、像素数、帧数与帧率计算，并计入 mask、shape、长度、header 和其他 side information。

只有 entropy loss、learned prior、binary latent 或“compression ratio”不足以证明 codec 合同。例如 LARP 的轻量 AR prior 是 tokenizer 训练约束，MAGVIT-v2 的 entropy penalty 用于 code utilization，InfoTok 的 BPP$_{16}$ 是 nominal 账本；它们都不等于发布了熵编码 bitstream。

### 4.4 Generative codec

generative codec 是 learned codec 的子类：在完整 bitstream 合同上，使用生成式先验或 decoder 优化低码率下的感知真实性。它必须同时验收 rate、输入忠实性、感知质量与生成式幻觉；锐利但虚构的细节不能当作无损恢复。

Divot 的 diffusion de-tokenizer 证明 continuous representation 可由生成式 decoder 反解码，但在没有量化、概率模型、熵编码和 bitstream 时，它仍是“使用 generative detokenizer 的 representation tokenizer”，不是已完成的 generative codec。

### 4.5 同一个“VAE”词为何会跨越多个角色

这些系统都可以有 encoder、decoder、Gaussian latent 或 KL 项，但随机变量、消费者和传输合同不同：

| 问题 | Stochastic latent future model | Representation tokenizer | Learned codec | Generative codec |
|---|---|---|---|---|
| $z/q$ 表示什么 | 同一历史下不可约简的未来不确定性 | 输入视频的紧凑可解码表示 | 可由概率模型和熵编码传输的符号 | 同左，但 decoder 允许以生成式先验改善感知质量 |
| 训练时看什么 | posterior 看历史加真实目标未来 | 待表示的视频；causal 编码器限制未来视野 | 同左，加 rate 或概率估计 | 同左，加 perception/generative objective |
| 部署时输入来自哪里 | 从只看历史的 learned prior 采样 | encoder 计算，或由上层 generator 预测 | 来自解析实际 bitstream | 来自解析实际 bitstream，decoder 可有随机性 |
| 核心失败 | prior–posterior gap、collapse、漏 mode | 表示丢失、闪烁、边界伪影、上层难学 | 实际 rate 超标、码流不可解、RD 劣化 | 同左，再加生成细节与输入不一致 |

## 5. 表示类型与压缩账本

### 5.1 连续、离散与混合

| 表示 | 存储的数学对象 | 常见上层模型 | 主要瓶颈 |
|---|---|---|---|
| Continuous | $z\in\mathbb R^{B\times C_z\times T'\times H'\times W'}$ | Gaussian diffusion、flow matching、DMD、continuous AR head | 通道数、潜分布尺度、float 精度、decoder 忠实性 |
| Discrete | $`i\in\lbrace1,\ldots,K_c\rbrace^{B\times T'\times H'\times W'}`$ 或 bit/scalar groups | categorical AR、masked prediction、discrete diffusion | 量化误差、code usage/dead codes、词表/序列长度 |
| Hybrid | 离散粗语义 $i$ 加连续残差 $r$，或两条互补流 | 离散 AR 加 residual diffusion，或双流解码器 | 两流的传输成本、对齐、融合与分别验收 |

HART 在 ICLR 2025 对“离散粗表示 + 连续残差”给出清晰定义，但它的主要证据是图像生成，不能直接当作视频运动结论。TVC 在 2025 的视频压缩设定中实例化了 discrete/continuous 双流，但它的目标是超低码率 codec，不等于所有上层生成器都应使用双流。

一个项目分别发布 continuous 和 discrete 型号，不等于单个样本使用 hybrid latent。Cosmos Tokenizer 和 VidTok 都支持连续/离散变体，但应按具体 checkpoint 判定输出类型。Divot 用 continuous representation 和 diffusion de-tokenizer；“decoder 是 diffusion”也不会把连续 tokenizer 自动变成 hybrid。

### 5.2 张量形状和四种不同的“压缩”

设输入为

```math
x\in\mathbb R^{B\times C\times T\times H\times W},
```

连续 latent 为

```math
z\in\mathbb R^{B\times C_z\times T'\times H'\times W'},
```

离散 token map 为

```math
i\in\{1,\ldots,K_c\}^{B\times T'\times H'\times W'}.
```

必须分别报告：

```math
r_t=\frac{T}{T'},\qquad
r_{hw}=\frac{HW}{H'W'},\qquad
r_{grid}=\frac{THW}{T'H'W'},
```

```math
r_{elem}=\frac{CTHW}{C_zT'H'W'}.
```

- $r_t$ 是有限 clip 的实际时间网格比，不一定等于配置里的 nominal factor。
- $r_{grid}$ 只计时空位置数，常被写成 $f_t\times f_h\times f_w$。
- $r_{elem}$ 还计入 RGB 通道 $C$ 与 latent 通道 $C_z$。例如 nominal $4\times8\times8$ 和 $C_z=16$ 在 RGB 输入上的渐近元素数降幅是 $3\times4\times8\times8/16=48$，不是 256。
- 对离散 token，未做概率编码时只能给出 nominal $\log_2K_c$ bits/token；实际码率需要概率模型、熵编码、bitstream 开销与帧率。
- 对 continuous latent，若没有量化精度和熵编码器，则不能从 $r_{elem}$ 得到 bitrate。

Cosmos 官方示例将 `[1, 3, 9, 512, 512]` 编码为 continuous `[1, 16, 3, 64, 64]` 或 discrete index `[1, 3, 64, 64]`。这个有限 clip 的实际时间比为 $9/3=3$，虽然型号名中的 nominal temporal factor 是 4；首帧独立锚定使短 clip 不能用渐近比例粗暴代替。

### 5.3 Bitstream 成熟度 BS0–BS3

为避免把 latent/token 缩小统称为“视频压缩”，本轨迹对传输合同单独使用 **BS0–BS3（bitstream stage）**：

| 级别 | 最低可核验合同 | 可报指标 | 不能越界的说法 |
|---|---|---|---|
| BS0：表示缩小 | continuous latent 或只有 shape/element reduction | latent shape、dtype、元素数、内存 | 不能写实际 bpp/bitrate 或 codec |
| BS1：离散符号 | 可量化 token/bit/scalar group，但无已验证的熵编码位流 | token 数、词表、nominal bits/token，可另报 mask/length 理论开销 | $N\log_2K$ 不是实际文件字节数；entropy regularizer 不是 entropy coder |
| BS2：实验性位流 | 概率模型、熵/算术编码、可解析字节流与实际 bpp–RD | 包含 side information 的字节数、bpp/bitrate、encode/decode 速度 | 论文内实验不自动证明开源 artifact 完整或与他人实现互操作 |
| BS3：可交换码流 | 版本化 syntax/header、独立编解码器、错误处理、随机访问/分片规则与跨实现互操作 | BS2 全部指标加兼容性、seek/streaming 行为 | 不能仅凭一个论文库的内部 tensor 序列化声称标准化 codec |

BS0–BS3 **只是本 source 文件内的位流成熟度标尺，不是全书的证据层级 L0–L7**，也不等同于本文第 3.3 节的来源等级 A/B/C/X。三者分别回答“位流合同成熟到哪里”、“全书证据链完整到哪里”和“来源是否经同行评审/官方发布”。

本轮目标工作中，continuous autoencoder 通常只到 BS0；MAGVIT/MAGVIT-v2、VidTok-FSQ、LARP、ElasticTok-FSQ、InfoTok、AdapTok 和 VideoRAE-discrete 通常只到 BS1；BSQ-ViT 在论文实验中用 AR 概率模型与 adaptive arithmetic coding 报告实际 bpp–RD，因而到 BS2。本轮未确认任一目标工作达到 BS3。

### 5.4 Causal padding 和首帧锚定

“causal”在 codec 中表示输出时刻 $k$ 不使用 $k$ 之后的输入帧。3D 时间卷积因此使用左侧填充或等价 cache；block-causal Transformer 则让当前时间块只读当前与过去块。一种常见的首帧保留映射是

```math
T'=1+\left\lfloor\frac{T-1}{f_t}\right\rfloor
=\left\lceil\frac{T}{f_t}\right\rceil,
```

但不同实现的 pad/crop 约定可不同，必须以实际 API 形状为准。Cosmos 官方页明确说第一个 temporal token 表示第一帧，用同一 latent 空间处理图像和视频。

这一性质只消除 codec 对未来帧的依赖。它不证明上层 diffusion/flow/AR 生成器也不看未来，不证明 decoder 没有 chunk 预热，也不证明端到端达到 streaming SLO。

### 5.5 Reconstruction ceiling 的精确边界

对固定的 $E,D$，可忠实保留的信息受 $D(E(x))$ 限制。若文字、小物体、高速运动或微小视差在 $E(x)$ 中已不可区分，上层 generator 不能从同一 latent 可靠地恢复原输入。

但这不是“主观美学分数的数学上界”。VQGAN 或 diffusion/generative decoder 可以合成看似锐利的新细节，却不保证这些细节忠实于输入。因此应把“感知真实”与“输入忠实”分开验收，并把 tokenizer reconstruction 与 generator sampling 分开报告。

## 6. Benchmark 与事实 registry

| ID | 一手来源 | 状态 | 可支撑的事实 | 证据边界 |
|---|---|---|---|---|
| R01 | [Auto-Encoding Variational Bayes](https://iclr.cc/archive/2014/old-site/conference-proceedings.html) | ICLR 2014，A | ELBO、重参数化和 amortized posterior | 图像基础模型，未定义视频 codec |
| R02 | [Stochastic Variational Video Prediction](https://openreview.net/forum?id=rk49Mg-CW) | ICLR 2018，A | 随机 latent 为同一历史产生多种未来 | 设定为视频预测，不是大型文生视频 tokenizer |
| R03 | [Stochastic Video Generation with a Learned Prior](https://proceedings.mlr.press/v80/denton18a.html) | ICML 2018，A | 每时刻随机 latent、训练 posterior 与测试 learned prior | 作者数据上的多未来预测证据 |
| R04 | [Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://openreview.net/forum?id=ryLDfnCqF7) | ICLR 2019，A | collapse 中模型忽略 latent、$q$ 贴近 prior；训练动力学是一个成因 | 不证明某一种 warm-up 对所有视频模型必然有效 |
| R05 | [Neural Discrete Representation Learning](https://papers.nips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | NeurIPS 2017，A | VQ-VAE、nearest-code lookup、straight-through、learned prior | 其视频实验不等于现代高清视频 tokenizer 的性能 |
| R06 | [Taming Transformers for High-Resolution Image Synthesis](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html) | CVPR 2021，A | 感知/对抗目标与离散 codebook 结合的 VQGAN | 图像证据；锐利度不代表像素忠实 |
| R07 | [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5950_ECCV_2022_paper.php) | ECCV 2022，A | 3D-VQGAN 与 Transformer 用于长视频生成 | 长 rollout 不证明无漂移 |
| R08 | [MAGVIT: Masked Generative Video Transformer](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html) | CVPR 2023，A | 3D video tokenizer 加 masked token modeling | 生成速度数值限定于论文硬件和基线 |
| R09 | [Language Model Beats Diffusion - Tokenizer is key to visual generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html) | ICLR 2024，A | MAGVIT-v2 用更强离散 tokenizer 支撑 LM 视觉生成 | 标题不能外推为所有 LM 普遍胜过 diffusion |
| R10 | [CV-VAE: A Compatible Video VAE for Latent Generative Video Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) | NeurIPS 2024，A | 连续 3D VAE 与现有 image-VAE latent 的兼容正则 | “生成四倍帧数”限作者微调协议 |
| R11 | [OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html) | NeurIPS 2024，A | 空间 window attention、时间 causal attention、图像–视频渐进联合训练 | 其 VQVAE 和 VAE 是两种训练/输出模式，不是自动 hybrid |
| R12 | [High-Quality Joint Image and Video Tokenization with Causal VAE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/03df5246cc78af497940338dd3eacbaa-Abstract-Conference.html) | ICLR 2025，A | causal 3D convolution、图像/视频联合、时空下采样与 flow regularization | 摘要中的“outperforms”不应脱离数据集和配置写成通用排名 |
| R13 | [Image and Video Tokenization with Binary Spherical Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e25198b6a75f74277ee3a2bd4165d9ef-Abstract-Conference.html) | ICLR 2025，A | BSQ 先将 latent 投影并球面归一化，再作 sign/binary quantization，无显式 codebook；block-wise causal ViT；AR 概率模型加 adaptive arithmetic coding 后评估实际 bpp–RD | 论文证据到 BS2，是本轮目标中少数真正连接 token 与 bitstream 的工作；未证明 BS3 互操作或公开 artifact 已覆盖论文全部编码链 |
| R14 | [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ab4e1e704c68b0f476c265996f08d283-Abstract-Conference.html) | ICLR 2025，A | 离散粗代码加连续残差的 hybrid tokenizer 定义 | 主实验是图像，不直接证明视频时间一致性 |
| R15 | [Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Ge_Divot_Diffusion_Powers_Video_Tokenizer_for_Comprehension_and_Generation_CVPR_2025_paper.html) | CVPR 2025，A | continuous video representation；diffusion 反解码；LLM 以 GMM 建模连续特征分布 | diffusion decoder 不意味 latent 是 discrete/hybrid；无量化、熵编码与 bitstream，因而是带 generative detokenizer 的 representation tokenizer（BS0），不是已完成的 generative codec |
| R16 | [TVC: Tokenized Video Compression with Ultra-Low Bit Rate](https://link.springer.com/article/10.1007/s44267-025-00098-7) | Visual Intelligence 2025，A | discrete/continuous 双流、熵模型和位流评估 | codec 任务，不是视频生成排名 |
| R17 | [InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression](https://proceedings.iclr.cc/paper_files/paper/2026/hash/432f048a844654ba981953491e6dc80e-Abstract-Conference.html) | ICLR 2026 Oral，A | Cosmos discrete causal base 加 block-causal adaptive compressor；FSQ levels $[8,8,8,5,5,5]$ 产生 64,000 个状态；基于 ELBO/reconstruction error 的 router 按样本信息量保留 token | 真正的样本自适应长度，但需额外 decoder pass；论文只验证 reconstruction，未测 generation/理解；BPP$_{16}$ 是 token 和 mask 的 nominal 账本（BS1），64,000 也不严格等于 $2^{16}=65,536$ |
| R18 | [AdapTok: Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html) | CVPR 2026，A | block-wise tail-token masking、causal scorer 与预算约束下的内容自适应分配 | UCF-101/Kinetics-600 协议；1D latent 不代表无位置结构 |
| R19 | [NeRV-Diffusion: Diffuse Implicit Neural Representation for Video Synthesis](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1a17a06de88cf77f25cda0da91615a54-Abstract-Conference.html) | ICLR 2026，A | 整段视频编码为 INR 网络权重，DiT 在 weight latent 上去噪 | 不是 frame-grid latent；作者效率结果不是任意分辨率/时长下的保证 |
| R20 | [KVAE: Family of Tokenizers for Multimodal Generative Models](https://arxiv.org/abs/2608.05798) | arXiv 2608.05798，B；本轮未找到正式 venue | continuous Gaussian VAE family；video 变体为 attention-free causal Conv3D，固定 $4\times8\times8$/$4\times16\times16$ 网格；官方实现提供 per-stream cache/chunk inference | causal 只约束未来依赖，cache 不自动证明端到端 streaming SLO；固定 token 数、BS0，无量化/熵编码/bitstream；性能为预印本作者报告 |
| R21 | [ElasticTok: Adaptive Tokenization for Image and Video](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html) | ICLR 2025，A | 训练时随机丢弃每帧尾部 token，支持条件于既往帧的变长表示 | token 数节省不自动等于变长 batch 的同比加速 |
| R22 | [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](https://proceedings.iclr.cc/paper_files/paper/2025/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html) | ICLR 2025 Oral，A | learned-codebook stochastic vector quantization；holistic queries 读取完整视频；训练 tokenizer 时加入轻量 AR prior，提高 latent 对后续 AR 建模的适配 | **输入时间上非因果**；AR 是 latent 顺序，不是时间因果或 entropy coder；query 数可配置但不是每样本自适应；BS1 |
| R23 | [Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Efficient_Long_Video_Tokenization_via_Coordinate-based_Patch_Reconstruction_CVPR_2025_paper.html) | CVPR 2025，A | CoordTok 用 $xy/yt/xt$ triplane 与坐标 patch reconstruction 表示长视频 | token 数只在相同 clip、分辨率和 decoder 协议下可比 |
| R24 | [VidTwin: Video VAE with Decoupled Structure and Dynamics](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VidTwin_Video_VAE_with_Decoupled_Structure_and_Dynamics_CVPR_2025_paper.html) | CVPR 2025，A | continuous 固定结构/动态双分支；backbone temporal attention 使用 causal mask，structure Q-Former 则对时间序列做查询压缩 | Q-Former 是否引入未来依赖未由 prefix-invariance 实验闭合，因而只能写“backbone causal，端到端因果性未证”；架构命名不证明可识别 disentanglement；0.20% 是 latent/input 元素比，不是 bpp；BS0 |
| R25 | [VidTok: A Versatile and Open-Source Video Tokenizer](https://arxiv.org/abs/2412.13061) | arXiv 2412.13061，B；本轮未找到正式 venue | 分别提供 continuous-KL 与 discrete-FSQ 型号，且有 causal 和 noncausal 配置；每个 checkpoint 使用固定网格/压缩配置 | 多型号不是单个 hybrid latent；配置可选不是样本自适应；VCR 是网格压缩率，KL 型号为 BS0、FSQ 型号为 BS1，无实际熵编码 bitstream |
| R26 | [V-RAE: Rethinking Video Latent Spaces for Generation](https://arxiv.org/abs/2608.13556) | arXiv 2608.13556，B；本轮未找到正式 venue | 在 frozen visual foundation encoder 特征上学 continuous representation autoencoder；固定 latent grid 与固定数量 temporal pooling 输出；用匹配 token 预算的 DiT 检查 generative learnability | 因果性依 variant 而异：DINO/SigLIP/EUPE 路线使用 chunk-wise causal decoder，V-JEPA2.1 路线非因果；content-weighted pooling 不改变 token 数，不是 adaptive length；BS0，tFVD 为论文内诊断而非通用共识 |
| R27 | [VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders](https://arxiv.org/abs/2607.14088) | arXiv 2607.14088，B；本轮未找到正式 venue | frozen VideoMAEv2/V-JEPA2 层级特征加 1D self-attention projector；分别训练 continuous 与 discrete Multi-codebook SimVQ 模式；用 local/global REPA 对齐语义特征 | 两种模式不是单个 hybrid latent；都是固定 $N_{\mathrm{latent}}$，论文未提出 causal 合同；continuous 为 BS0、discrete 为 BS1，无熵编码 bitstream；代码/模型的未来发布承诺不等于截止日已可复现 |

## 7. 代表开放实现 registry

| 实现 | 论文/仓库状态 | 截止日可公开核验的细节 | 不做的推断 |
|---|---|---|---|
| [CV-VAE](https://github.com/AILab-CVC/CV-VAE) | NeurIPS 2024；官方训练/推理代码与权重 | 连续 video VAE；仓库列出 SD2.1/SVD 和 SD3/SD3.5 兼容变体 | 不假设它与任意未公开 image VAE 都无缝兼容 |
| [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer) | NeurIPS 2024；MIT 代码仓库 | 同一架构支持 image/video；仓库给出 VQVAE 与 VAE 训练配置 | 不将两种模式误称为单个 hybrid latent |
| [Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | 官方仓库已转为 read-only，指向 NVIDIA Cosmos；代码 Apache-2.0，模型用 NVIDIA Open Model License | image/video 的 continuous/discrete 变体；temporal 4/8 与 spatial 8/16 配置；明示 API 形状 | README 的速度/质量排名不当作独立复现；“2048×”是网格乘积语境 |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 作者技术报告；官方代码与权重 | CausalConv3D VAE；时间 4×、空间 8×、latent 16 channels | 不把 $4\times8\times8=256$ 写成实际元素数或 bitrate 压缩 |
| [VidTok](https://huggingface.co/microsoft/VidTok) | arXiv 2412.13061；官方 Microsoft 项目/模型卡 | 公开 continuous 与 FSQ-based discrete 变体、causal/noncausal 配置及多个固定压缩设置 | 截止日不写为已在某会议正式发表；不将多配置写成 adaptive length，不将 FSQ token 写成实际 bitstream |
| [KVAE](https://github.com/kandinskylab/kvae) | arXiv 2608.05798；官方代码与 Hugging Face 权重 | video 2.0 有固定 $t4s8$ 和 $t4s16$，因果 cache 与 chunk 推理接口公开 | 不把 2026-08 技术报告写成同行评审 venue；不将 cache 或 continuous latent（BS0）写成已完成 streaming codec |

## 8. 2022–2026 技术路线与里程碑

| 年份 | 代表节点 | 真正改变的设计轴 | 应保留的证据边界 |
|---|---|---|---|
| 2022 | TATS，ECCV | 3D-VQGAN 将时空量化与长序列 Transformer 结合 | “数千帧”是作者数据和 rollout 协议 |
| 2023 | MAGVIT，CVPR | 3D tokenizer 和 masked parallel generation 结合 | tokenizer 与 generator 贡献需分开 |
| 2024 | MAGVIT-v2，ICLR | lookup-free 离散化与更大隐式词表 | 受控对比不是普遍的 LM-vs-diffusion 定理 |
| 2024 | CV-VAE，NeurIPS | 连续时空 latent 与既有 image-VAE latent 对齐 | 兼容性是训练目标，不是任意 checkpoint 无损互换 |
| 2024 | OmniTokenizer，NeurIPS | 图像–视频联合、spatial window 与 temporal causal attention | continuous/discrete 为可选模式 |
| 2024 | VidTok，arXiv 预印本 | 在同一项目系列公开 continuous-KL/discrete-FSQ 与 causal/noncausal 配置 | 截止日未找到正式 venue；系列多配置不是 hybrid 或 adaptive length |
| 2025 | Causal VAE，ICLR | causal 3D convolution、长序列时空采样与 motion regularization | causal codec 不等于 causal generator |
| 2025 | BSQ，ICLR | 从显式 codebook 走向球面二值量化，并把熵模型接到位流 | 压缩标准比较只在同 bpp 协议下有意义 |
| 2025 | Divot，CVPR | 从纯重建 tokenizer 走向语义连续表示加生成式 de-tokenizer | 不能只用 PSNR 评估理解用 latent |
| 2026 | InfoTok，ICLR；AdapTok，CVPR | **fixed-rate → content-adaptive token budget** | token 数减少需与 router/scorer 开销、结构保留和 downstream 质量一起报告 |
| 2026 | NeRV-Diffusion，ICLR | **frame-wise feature-map latent → whole-video INR-weight latent** | 是表示单元的变化，不是证明网格 latent 已过时 |
| 2026 | V-RAE、VideoRAE、KVAE，arXiv 预印本 | frozen foundation representation、representation alignment 与 diffusability 把评估从纯重建推向 generative learnability | 截止日均必须保留 preprint 标签；三者均未发布实际 bitstream |

### 8.1 四类综合，而非单线替代

| 综合路线 | 代表一手来源 | 被改变的轴 | 公平验收问题 |
|---|---|---|---|
| Fixed-grid | Causal VAE、BSQ-ViT、CV-VAE | 规则 $T'\times H'\times W'$ 网格中的编码器、量化或兼容性 | 相同 shape、dtype 与 decoder 下，重建和下游生成是否改善？ |
| Adaptive budget | ElasticTok、InfoTok、AdapTok | 每帧或每块使用的 token 数由内容决定 | router/scorer、变长 batching 与最坏情况长度计入后，端到端 SLO 是否改善？ |
| Structured latent | CoordTok、VidTwin、NeRV-Diffusion | 从规则局部网格转向 triplane、双分支或 INR weights | 结构归纳偏置是否经干预验证，并能跨时长/分辨率泛化？ |
| Generator-aware tokenizer | LARP、Divot | tokenizer 训练显式考虑上层 prior 或生成式 de-tokenizer | 固定 generator 与总算力后，收益是否仍来自表示本身？ |
| Representation-aligned tokenizer | V-RAE、VideoRAE、KVAE | 引入 frozen foundation features、representation alignment 或 downstream diffusability 选型 | 固定上层架构、token 预算、数据和总 FLOPs 后，语义/收敛收益能否跨数据集保持？ |

五类并不互斥：一个系统可以既 causal fixed-grid，又使用 generator-aware loss；也可能在 structured latent 上再做 adaptive allocation。教材应把它们放在正交设计轴上，而不是写成“旧 tokenizer 被新 tokenizer 依次取代”。

## 9. 关键争议与证据边界

### 9.1 “VAE”是生成模型还是 codec

两种用法都存在，不能靠名字判断。应检查 $z$ 是否表示对未来的随机选择，推理时是否从条件 prior 采样，以及上层是否另有 diffusion/flow/AR generator。

### 9.2 “压缩倍数”是否可比

不可直接比。论文可能报 $f_t\times f_h\times f_w$、张量元素比、token 数比、nominal bits/token、bpp 或带帧率的 bitrate。除非分子、分母、通道、数值精度、熵编码和 clip 边界全部一致，否则不应排名。

### 9.3 Causal 是否意味 streaming

不意味。causal 只约束信息访问。软件是否维护 cache、每个 chunk 有多大预热、decoder 是否能逐帧提交、上层 generator 是否需要完整窗口，都需另外证明。

### 9.4 Reconstruction 好是否必然 generation 好

不必然。重建太差会明确限制生成，但更高 PSNR/LPIPS 不能确保 latent 分布容易被上层模型学习。需要在固定 generator 或受控算力下做 downstream 比较，同时查看 latent 尺度、数值分布、序列长度和 code usage。

### 9.5 自适应 token 是否一定更快

不一定。InfoTok 和 AdapTok 表明 fixed budget 不是唯一选择，但 router/scorer、变长 batching、位置恢复和缓存碎片都会增加系统成本。必须报 end-to-end wall-clock、显存、token 分布与质量曲线，不只报平均 token reduction。

### 9.6 INR-weight latent 是否还是 tokenizer

是，如果 encoder 把输入映射到可由 decoder/INR 恢复的紧凑表示。NeRV-Diffusion 的差别是 latent 不再与局部时空网格一一对应，而是实例专属 INR 的权重。这改变了 denoiser 看到的结构先验，也引入权重空间可生成性与实例 decoder 误差的新问题。

### 9.7 2026-08-30 专项复核的六项纠偏

1. **LARP 不是 causal video tokenizer。** holistic queries 读取完整视频；论文中的 learned AR prior 建模 latent token 顺序，不是 encoder 的输入时间因果性。query 数能以配置改变，也不是按样本内容预测长度。
2. **VidTwin 只能写“backbone temporal attention causal，端到端因果性未证”。** structure Q-Former 压缩时间序列，论文未用前缀不变性实验排除未来泄漏；双分支架构也不构成可识别语义解耦的证明。
3. **InfoTok 的 BPP$_{16}$ 是 BS1 nominal 账本，不是实际 bitstream。** 它使用 FSQ levels $[8,8,8,5,5,5]$，状态数是 64,000，不严格等于 $2^{16}=65,536$；计入 token/mask 理论位数也不等于熵编码后文件大小。
4. **V-RAE 的 causal 结论必须按 variant 写。** DINO/SigLIP/EUPE 等 framewise encoder 路线搭配 chunk-wise causal decoder；V-JEPA2.1 路线使用非因果视频编码器。attention pooling 的权重依赖内容，但输出 token 数固定，不应标为 adaptive length。
5. **VidTok 和 VideoRAE 都是分别提供 continuous/discrete 模式，不是单个 hybrid latent。** hybrid 必须在同一样本/解码合同中同时消费离散主码和连续残差或两条互补流。两者的目标配置也都是固定 token 预算，不是 content-adaptive tokenization。
6. **BSQ-ViT 到达 BS2，但不能写成 BS3 标准化 codec。** 论文确实用 AR 概率模型与 adaptive arithmetic coding 报告实际 bpp–RD；但这不自动证明版本化 header、随机访问、异常处理、跨实现互操作或完整开源复现链。

## 10. 建议的教材验收协议

### 10.1 先按任务角色与消费者分流

1. 若目标是给定历史采样多个未来，进入 **stochastic future** 协议；消费者是未来分布本身。
2. 若目标是将已知视频变成更短/更小的上层变量，进入 **representation tokenizer** 协议；必须明示消费者是 discrete AR/masked generator、continuous diffusion/flow 还是理解模型。
3. 若目标包含传输或存储码率，进入 **learned codec** 协议；消费者是独立 bitstream decoder，不能只看 latent tensor。
4. 若位流 decoder 还使用生成式先验或随机采样，进入 **generative codec** 协议；除 rate 外必须拆分感知质量与输入忠实性。
5. 若系统跨越多个角色，必须拆成 encoder/tokenizer、概率模型/码流、latent generator、pixel decoder 和最终输出分段验收。

### 10.2 Stochastic future 协议

- 同一历史下报告多次采样，不只报 best-of-$N$。
- 分别检查条件一致性、样本间多样性、真实未来覆盖和长期漂移。
- 报告 KL 时同时报 decoder 对 $z$ 的敏感性、active units/互信息代理和条件样本多样性；小 KL 本身不是 collapse 的充分证据。
- 分开训练 posterior 重建与测试 prior 采样，显式测量 prior–posterior gap。

### 10.3 Representation tokenizer 消费者协议

- 完整报告 `[B,C,T,H,W] -> [B,C_z,T',H',W']` 或 index shape，以及 nominal 与 exact 网格/元素/token 账本。
- 将 PSNR/SSIM、LPIPS/感知质量、rFVD/时间分布、OCR/人脸/小物体、快速运动和边界伪影分栏。
- 测试训练长度内外的短 clip、长 clip、任意尺寸、首帧、末帧、chunk 切分和 streaming cache reset。对 causal 声称，构造前缀相同而未来不同的视频，验证 encoder/decoder 前缀不变；block-causal 只在块边界要求该性质。
- 报告 encode/decode wall-clock、峰值显存、设备、dtype、batch、帧数和分辨率。对 adaptive tokenizer，将 router/scorer、额外 decoder pass、search NFE、ILP、ragged batching 与 p95 延迟全部计入。
- 对 discrete AR/masked 消费者，报 token order、code usage、经验熵和序列长度；对 diffusion/flow 消费者，报 latent scale/distribution 和训练稳定性；对理解消费者，报固定 probe 与跨数据集语义保留。
- 固定同一上层 generator/理解模型、训练步数与总 FLOPs 做替换对照，避免 tokenizer 改进与消费者规模改进混在一起。

### 10.4 Learned codec 消费者协议

- 在 tokenizer 协议之上，必须产生可持久化的 bitstream，并由独立进程在不读取 encoder 内部 tensor 的情况下重建。
- 实际字节数要包含 arithmetic/range code、mask、shape、长度、header、model selection 和其他 side information；同时报 bpp、bitrate、帧率、RD 曲线与 encode/decode FPS。
- 明示 BS0–BS3 阶段；只有 $N\log_2K$、compression factor 或 entropy loss 的工作仍留在 BS0/BS1。
- 对 random access/streaming 声称，验证分片边界、seek 开销、错码扩散、cache reset 和跨进程/跨实现解码。

### 10.5 Generative codec 消费者协议

- 必须首先通过 learned codec 的实际 bitstream 协议；仅有 diffusion/generative decoder 不足以进入该分支。
- 同时报 rate–distortion、rate–perception 与输入忠实性；对文字、人脸、小物体、快速运动和细微时序作定向 hallucination 审计。
- 若 decoder 有随机性，对同一 bitstream 重复解码，分开报告样本间方差、最好/最差忠实度与感知评分，不使用 best-of-$N$ 掩盖失真。

## 11. 图示规范

### 11.1 四种角色与消费者分流图

- 决策节点：`What consumes the latent or bitstream?`。
- stochastic future 支：`Observed history` → `Training posterior sees target` / `Test prior sees history` → `Sample possible futures` → `Evaluate future distribution`。
- representation tokenizer 支：`Encode known video` → `Compact latent/token` → `Fixed downstream generator or understanding model` → `Evaluate reconstruction and consumer utility`。
- learned codec 支：`Quantized symbols` → `Probability model` → `Entropy coder` → `Actual bitstream` → `Independent decoder` → `Evaluate rate-distortion`。
- generative codec 作为 learned codec 子支：`Actual bitstream` → `Generative decoder/prior` → `Evaluate rate-distortion-perception and hallucination`。
- 禁止把这些分支画成先后进化关系；一个系统可跨多个角色，但验收必须分段。

### 11.2 压缩账本与 causal 边界图

- 节点顺序：`RGB [B,C,T,H,W]` → `Left-only temporal context` → `Encoder/downsample` → `Continuous / discrete / hybrid` → `Decoder` → `Reconstruction`。
- 并列账本节点：`Grid ratio`、`Element ratio`、`Nominal token bits`、`BS0–BS3 bitstream stage`、`Actual bytes only with entropy coding`。
- 首帧节点标注“first temporal token anchors first frame”，不用 $T/f_t$ 模糊短 clip 边界。
- Mermaid 必须含 `accTitle` 和单行 `accDescr`，下方用顺序化文字完整复述。

## 12. 更新触发条件

出现以下任一情况时，应重新审计本章：

- VidTok、V-RAE、VideoRAE、KVAE 或其他预印本获得正式 venue，标题/配置/数据发生变化。
- 主流开放模型公布可复现 bitrate、entropy coder 或标准化 bitstream，可将“tokenizer”与“codec”的边界改写得更具体。
- adaptive tokenizer 出现统一的变长 batching 与 end-to-end SLO 评测，能验证 token reduction 是否真正转化为延迟/显存收益。
- INR-weight latent 在更多数据、分辨率和条件生成任务中被独立复现，可以评估它是否超越特定 benchmark 的可迁移分支。

## 13. 本轮核验记录与主张边界

- **核验日期**：2026-08-30（Asia/Shanghai）。
- **一手来源范围**：CVF/ICLR/NeurIPS/PMLR/ECVA 正式 proceedings、作者 arXiv 稿、官方项目页/代码库/模型卡；第三方榜单和博客不作最终证据。
- **正式/预印本边界**：MAGVIT、MAGVIT-v2、Causal VAE、LARP、ElasticTok、BSQ-ViT、CoordTok、VidTwin、Divot、InfoTok、AdapTok 有正式 proceedings；VidTok、V-RAE、VideoRAE、KVAE 在本轮仅按 arXiv 预印本/技术报告处理。“未找到正式 venue”是截止日检索结果，不是永久性断言。
- **causal 主张边界**：只表示 encoder/decoder 在指定帧或 block 不访问未来输入；不自动包含上层 generator 因果性、chunk/full 等价、实时性或 streaming SLO。
- **adaptive 主张边界**：只有 token 数依赖当前样本/时间块内容时才标为 adaptive length；多 checkpoint、可配 query 数、可变输入时长或 content-weighted 但固定数量 pooling 都不足以成立。
- **bitstream 主张边界**：BS0–BS3 仅审计传输合同，不等于全书 L0–L7 证据层级。本轮对 BSQ-ViT 的 BS2 判定来自论文报告的概率模型、adaptive arithmetic coding 和实际 bpp–RD；未判定任一目标工作到达 BS3。
- **复现边界**：本轮是论文/官方 artifact 只读核验，没有重训模型、重跑 benchmark 或独立生成/解码位流；作者报告的性能不得改写为独立复现结论。

## 14. 教材拆章与交付验收

本轮最终把原混合页面拆为 277 行的 stochastic-future 变分章和 416 行的 tokenizer/codec 章，并同步总览、基础模型、入门、阅读路线、时间线与相邻机制页。验收结果如下：

| 检查 | 结果 |
|---|---|
| Markdown | 22 个变更/新增的非 README Markdown 文件以 markdownlint-cli2 0.23.2 / markdownlint 0.41.1 检查，0 问题；README 仍只有两个位于本轮修改前既已存在的 MD001/MD028 提示 |
| 引用闭合 | 变分章 7 refs / 11 citations；tokenizer 章 25 / 62；均无缺失、孤儿、重复或编号断裂 |
| 一手来源 URL | 两章共 32 个参考文献 URL 以重定向 GET 复核，32/32 返回 HTTP 200 |
| 本地链接 | 23 个变更/新增 Markdown 文件的 404 个相对链接与图片目标全部存在 |
| Mermaid | 23 个文件共 35 个 Mermaid，全部含 `accTitle` / `accDescr`；以 Mermaid CLI 11.16.0 和系统 Chrome 全部渲染为非空 SVG |
| 视觉检查 | 五轴总览、两张变分图、两张 tokenizer 合同图、两张阅读路线图与时间线总图已渲染为 PNG 并逐张检查；未发现截断、断路或不可读终点 |
| 时间线媒体 | 75 个既有 HTML 图片保留，空 alt 为 0；新增前沿使用文本表，避免制造无来源的论文插画 |
| 补丁卫生 | `git diff --check` 与 changed-diff credential pattern scan 均通过 |

临时 Mermaid/PNG 审计文件位于仓库外。本轮闭合的是教学结构、来源与可证伪合同，不构成对 32 篇工作的模型重训、benchmark 复现或实际 bitstream 独立实现。
