# 视频潜表示与 Tokenizer 研究轨迹（2026-08-30）

> 范围：为 `docs/generative-models/variational-generation.md` 的重写提供可复核证据。本文区分 stochastic latent future model 与 video codec/tokenizer，并跟踪至 **2026-08-30** 的正式论文、作者技术报告和官方开放实现。

## 1. 研究问题与边界

本轨迹围绕六个问题：

1. 同样被称为“video VAE”的系统，何时在学未来的条件分布，何时只是为上层 generator 提供压缩表示？
2. ELBO、learned prior 与 posterior collapse 在条件视频预测中分别承担什么角色？
3. 连续、离散与混合 tokenizer 的数据类型、训练目标和上层 generator 接口如何区分？
4. 输入张量、潜张量、网格压缩、元素数压缩、token 数和 bitrate 如何不混淆？
5. causal 3D VAE 的左侧时间填充、首帧锚定和 chunk 边界会带来哪些工程约束？
6. 2025–2026 的自适应 token 预算与 INR-weight latent 是新表示分支，还是旧式特征网格的换名？

不纳入的内容：只有产品演示而无公开技术细节的系统；第三方博客中无法追溯到一手材料的数值；把图像-only 结果直接外推为视频运动结论的主张；只报 latent 尺寸却称已实现可传输 codec 的工作。

## 2. 检索表面、检索式与日期

所有检索均于 **2026-08-30（Asia/Shanghai）** 执行。搜索结果只用于发现，最终事实必须回到正式 proceedings、作者稿或官方仓库。

| 检索表面 | 代表检索式 | 用途 | 纳入方式 |
|---|---|---|---|
| ICLR / OpenReview 正式页 | `site:proceedings.iclr.cc 2025 "High-Quality Joint Image and Video Tokenization with Causal VAE"`；`site:proceedings.iclr.cc 2026 InfoTok adaptive video tokenizer`；`site:proceedings.iclr.cc 2026 "NeRV-Diffusion"` | 核对 2025–2026 标题、作者、正式接收状态 | 仅用 proceedings 页确认 venue；OpenReview 用于读全文 |
| NeurIPS / PMLR / CVF / ECVA 正式 proceedings | `site:proceedings.mlr.press Denton Fergus "Stochastic Video Generation"`；`site:proceedings.neurips.cc OmniTokenizer`；`site:openaccess.thecvf.com CVPR 2025 Divot`；`site:ecva.net TATS ECCV 2022` | 核对 2017–2025 基础论文与正式 venue | 优先引用 abstract/conference HTML，必要时读 PDF |
| arXiv 作者稿 | 论文全名加 `arXiv`；`"KVAE" 2608.05798 video tokenizer`；`"VidTok" video tokenizer` | 检索尚无正式 venue 的 2026 技术报告及全文 | 明标“preprint / author technical report”，不写成已发表 |
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

这一角色的对象是条件分布 $p(x_{C+1:K}\mid x_{1:C})$。随机潜变量用来表达给定同一历史时仍可能出现的多个合理未来。训练时的近似后验 $q_\phi(z_k\mid x_{\le k})$ 可看到目标帧；部署时只能从 learned prior $p_\psi(z_k\mid x_{<k})$ 采样。SVG-LP 于 ICML 2018 对这一 posterior–prior 分工给出了经典实例。

验收对象是未来分布：条件一致性、多样性、覆盖、校准与长期 rollout。单个最好样本或重建分数不能验证这一角色。

### 4.2 Modern video codec/tokenizer

这一角色的对象是变换 $z=E(x)$ 与 $\hat x=D(z)$，目标是让后续 diffusion、flow、AR 或 masked generator 在更小表示上工作。它可以用 KL 正则、VQ/LFQ/FSQ/BSQ 量化、感知损失或对抗损失，但它不因此自动学得未来分布。

验收对象是重建、压缩账本、边界处理、编解码性能和 downstream generation。没有量化、概率模型、熵编码和 bitstream 定义的 continuous latent 只证明张量维度下降，不证明实际 bitrate。

### 4.3 同一个“VAE”词为何会跨越两个角色

两者都可以有 encoder、decoder、Gaussian latent 和 KL 项，但随机变量的语义不同：

| 问题 | 随机未来模型 | codec/tokenizer |
|---|---|---|
| $z$ 表示什么 | 同一历史下不可约简的未来不确定性 | 输入视频的紧凑可解码表示 |
| 训练时 posterior 看什么 | 历史加真实目标未来 | 待压缩的完整图像/视频（causal 编码器例外地限制时间视野） |
| 测试时 latent 从哪里来 | 从只看历史的 learned prior 采样 | 由 encoder 从要重建的已知视频计算，或由上层 generator 产生 |
| 核心失败 | prior–posterior gap、collapse、漏 mode | 量化/压缩丢失、闪烁、边界伪影、decoder 幻觉 |

## 5. 表示类型与压缩账本

### 5.1 连续、离散与混合

| 表示 | 存储的数学对象 | 常见上层模型 | 主要瓶颈 |
|---|---|---|---|
| Continuous | $z\in\mathbb R^{B\times C_z\times T'\times H'\times W'}$ | Gaussian diffusion、flow matching、DMD、continuous AR head | 通道数、潜分布尺度、float 精度、decoder 忠实性 |
| Discrete | $i\in\{1,\ldots,K_c\}^{B\times T'\times H'\times W'}$ 或 bit/scalar groups | categorical AR、masked prediction、discrete diffusion | 量化误差、code usage/dead codes、词表/序列长度 |
| Hybrid | 离散粗语义 $i$ 加连续残差 $r$，或两条互补流 | 离散 AR 加 residual diffusion，或双流解码器 | 两流的传输成本、对齐、融合与分别验收 |

HART 在 ICLR 2025 对“离散粗表示 + 连续残差”给出清晰定义，但它的主要证据是图像生成，不能直接当作视频运动结论。TVC 在 2025 的视频压缩设定中实例化了 discrete/continuous 双流，但它的目标是超低码率 codec，不等于所有上层生成器都应使用双流。

一个项目分别发布 continuous 和 discrete 型号，不等于单个样本使用 hybrid latent。Cosmos Tokenizer 和 VidTok 都支持连续/离散变体，但应按具体 checkpoint 判定输出类型。Divot 用 continuous representation 和 diffusion de-tokenizer；“decoder 是 diffusion”也不会把连续 tokenizer 自动变成 hybrid。

### 5.2 张量形状和四种不同的“压缩”

设输入为

$$
x\in\mathbb R^{B\times C\times T\times H\times W},
$$

连续 latent 为

$$
z\in\mathbb R^{B\times C_z\times T'\times H'\times W'},
$$

离散 token map 为

$$
i\in\{1,\ldots,K_c\}^{B\times T'\times H'\times W'}.
$$

必须分别报告：

$$
r_t=\frac{T}{T'},\qquad
r_{hw}=\frac{HW}{H'W'},\qquad
r_{grid}=\frac{THW}{T'H'W'},
$$

$$
r_{elem}=\frac{CTHW}{C_zT'H'W'}.
$$

- $r_t$ 是有限 clip 的实际时间网格比，不一定等于配置里的 nominal factor。
- $r_{grid}$ 只计时空位置数，常被写成 $f_t\times f_h\times f_w$。
- $r_{elem}$ 还计入 RGB 通道 $C$ 与 latent 通道 $C_z$。例如 nominal $4\times8\times8$ 和 $C_z=16$ 在 RGB 输入上的渐近元素数降幅是 $3\times4\times8\times8/16=48$，不是 256。
- 对离散 token，未做概率编码时只能给出 nominal $\log_2K_c$ bits/token；实际码率需要概率模型、熵编码、bitstream 开销与帧率。
- 对 continuous latent，若没有量化精度和熵编码器，则不能从 $r_{elem}$ 得到 bitrate。

Cosmos 官方示例将 `[1, 3, 9, 512, 512]` 编码为 continuous `[1, 16, 3, 64, 64]` 或 discrete index `[1, 3, 64, 64]`。这个有限 clip 的实际时间比为 $9/3=3$，虽然型号名中的 nominal temporal factor 是 4；首帧独立锚定使短 clip 不能用渐近比例粗暴代替。

### 5.3 Causal padding 和首帧锚定

“causal”在 codec 中表示输出时刻 $k$ 不使用 $k$ 之后的输入帧。3D 时间卷积因此使用左侧填充或等价 cache；block-causal Transformer 则让当前时间块只读当前与过去块。一种常见的首帧保留映射是

$$
T'=1+\left\lfloor\frac{T-1}{f_t}\right\rfloor
=\left\lceil\frac{T}{f_t}\right\rceil,
$$

但不同实现的 pad/crop 约定可不同，必须以实际 API 形状为准。Cosmos 官方页明确说第一个 temporal token 表示第一帧，用同一 latent 空间处理图像和视频。

这一性质只消除 codec 对未来帧的依赖。它不证明上层 diffusion/flow/AR 生成器也不看未来，不证明 decoder 没有 chunk 预热，也不证明端到端达到 streaming SLO。

### 5.4 Reconstruction ceiling 的精确边界

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
| R13 | [Image and Video Tokenization with Binary Spherical Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e25198b6a75f74277ee3a2bd4165d9ef-Abstract-Conference.html) | ICLR 2025，A | BSQ 无显式 codebook；block-wise causal ViT；自回归先验加算术编码后评估 bpp | 是少数真正连接 token 与 bitstream 的证据；不能代替所有神经 codec 比较 |
| R14 | [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ab4e1e704c68b0f476c265996f08d283-Abstract-Conference.html) | ICLR 2025，A | 离散粗代码加连续残差的 hybrid tokenizer 定义 | 主实验是图像，不直接证明视频时间一致性 |
| R15 | [Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Ge_Divot_Diffusion_Powers_Video_Tokenizer_for_Comprehension_and_Generation_CVPR_2025_paper.html) | CVPR 2025，A | continuous video representation；diffusion 反解码；LLM 以 GMM 建模连续特征分布 | diffusion decoder 不意味 tokenizer 的代码是 diffusion 噪声时间轴 |
| R16 | [TVC: Tokenized Video Compression with Ultra-Low Bit Rate](https://link.springer.com/article/10.1007/s44267-025-00098-7) | Visual Intelligence 2025，A | discrete/continuous 双流、熵模型和位流评估 | codec 任务，不是视频生成排名 |
| R17 | [InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression](https://proceedings.iclr.cc/paper_files/paper/2026/hash/432f048a844654ba981953491e6dc80e-Abstract-Conference.html) | ICLR 2026，A | 基于 ELBO 的 router 按信息量分配变长 token；官方摘要报告不影响性能时节省 20% token，并在 2.3× 压缩率下仍优于先前启发式自适应方法 | 作者协议数值；token 节省不等于端到端同比加速 |
| R18 | [AdapTok: Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html) | CVPR 2026，A | block-wise tail-token masking、causal scorer 与预算约束下的内容自适应分配 | UCF-101/Kinetics-600 协议；1D latent 不代表无位置结构 |
| R19 | [NeRV-Diffusion: Diffuse Implicit Neural Representation for Video Synthesis](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1a17a06de88cf77f25cda0da91615a54-Abstract-Conference.html) | ICLR 2026，A | 整段视频编码为 INR 网络权重，DiT 在 weight latent 上去噪 | 不是 frame-grid latent；作者效率结果不是任意分辨率/时长下的保证 |
| R20 | [KVAE: Family of Tokenizers for Multimodal Generative Models](https://arxiv.org/abs/2608.05798) | arXiv 2608.05798，B | 2026-08 公开的 causal video tokenizer 技术报告；$4\times8\times8$ 和 $4\times16\times16$ 变体 | 截止日无正式 venue；性能为作者报告 |
| R21 | [ElasticTok: Adaptive Tokenization for Image and Video](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html) | ICLR 2025，A | 训练时随机丢弃每帧尾部 token，支持条件于既往帧的变长表示 | token 数节省不自动等于变长 batch 的同比加速 |
| R22 | [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](https://proceedings.iclr.cc/paper_files/paper/2025/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html) | ICLR 2025，A | 训练 tokenizer 时加入轻量 AR prior，提高 latent 对后续 AR 建模的适配 | prior 是 tokenizer 训练约束，不等于最终生成模型的全部能力 |
| R23 | [Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Efficient_Long_Video_Tokenization_via_Coordinate-based_Patch_Reconstruction_CVPR_2025_paper.html) | CVPR 2025，A | CoordTok 用 $xy/yt/xt$ triplane 与坐标 patch reconstruction 表示长视频 | token 数只在相同 clip、分辨率和 decoder 协议下可比 |
| R24 | [VidTwin: Video VAE with Decoupled Structure and Dynamics](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VidTwin_Video_VAE_with_Decoupled_Structure_and_Dynamics_CVPR_2025_paper.html) | CVPR 2025，A | 结构与动态使用不同 latent 分支 | 架构命名不构成语义解耦的充分证据 |

## 7. 代表开放实现 registry

| 实现 | 论文/仓库状态 | 截止日可公开核验的细节 | 不做的推断 |
|---|---|---|---|
| [CV-VAE](https://github.com/AILab-CVC/CV-VAE) | NeurIPS 2024；官方训练/推理代码与权重 | 连续 video VAE；仓库列出 SD2.1/SVD 和 SD3/SD3.5 兼容变体 | 不假设它与任意未公开 image VAE 都无缝兼容 |
| [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer) | NeurIPS 2024；MIT 代码仓库 | 同一架构支持 image/video；仓库给出 VQVAE 与 VAE 训练配置 | 不将两种模式误称为单个 hybrid latent |
| [Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | 官方仓库已转为 read-only，指向 NVIDIA Cosmos；代码 Apache-2.0，模型用 NVIDIA Open Model License | image/video 的 continuous/discrete 变体；temporal 4/8 与 spatial 8/16 配置；明示 API 形状 | README 的速度/质量排名不当作独立复现；“2048×”是网格乘积语境 |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 作者技术报告；官方代码与权重 | CausalConv3D VAE；时间 4×、空间 8×、latent 16 channels | 不把 $4\times8\times8=256$ 写成实际元素数或 bitrate 压缩 |
| [VidTok](https://huggingface.co/microsoft/VidTok) | arXiv 2412.13061；官方 Microsoft 项目/模型卡 | 公开 continuous 与 FSQ-based discrete 变体及多压缩配置 | 截止日不写为已在某会议正式发表 |
| [KVAE](https://github.com/kandinskylab/kvae) | arXiv 2608.05798；官方代码与 Hugging Face 权重 | video 2.0 有 $t4s8$ 和 $t4s16$，因果 cache 与 chunk 推理接口公开 | 不把 2026-08 技术报告写成同行评审 venue |

## 8. 2022–2026 技术路线与里程碑

| 年份 | 代表节点 | 真正改变的设计轴 | 应保留的证据边界 |
|---|---|---|---|
| 2022 | TATS，ECCV | 3D-VQGAN 将时空量化与长序列 Transformer 结合 | “数千帧”是作者数据和 rollout 协议 |
| 2023 | MAGVIT，CVPR | 3D tokenizer 和 masked parallel generation 结合 | tokenizer 与 generator 贡献需分开 |
| 2024 | MAGVIT-v2，ICLR | lookup-free 离散化与更大隐式词表 | 受控对比不是普遍的 LM-vs-diffusion 定理 |
| 2024 | CV-VAE，NeurIPS | 连续时空 latent 与既有 image-VAE latent 对齐 | 兼容性是训练目标，不是任意 checkpoint 无损互换 |
| 2024 | OmniTokenizer，NeurIPS | 图像–视频联合、spatial window 与 temporal causal attention | continuous/discrete 为可选模式 |
| 2025 | Causal VAE，ICLR | causal 3D convolution、长序列时空采样与 motion regularization | causal codec 不等于 causal generator |
| 2025 | BSQ，ICLR | 从显式 codebook 走向球面二值量化，并把熵模型接到位流 | 压缩标准比较只在同 bpp 协议下有意义 |
| 2025 | Divot，CVPR | 从纯重建 tokenizer 走向语义连续表示加生成式 de-tokenizer | 不能只用 PSNR 评估理解用 latent |
| 2026 | InfoTok，ICLR；AdapTok，CVPR | **fixed-rate → content-adaptive token budget** | token 数减少需与 router/scorer 开销、结构保留和 downstream 质量一起报告 |
| 2026 | NeRV-Diffusion，ICLR | **frame-wise feature-map latent → whole-video INR-weight latent** | 是表示单元的变化，不是证明网格 latent 已过时 |
| 2026 | KVAE，arXiv 技术报告 | 公开更高空间压缩的 causal continuous video VAE 变体 | 截止日必须保留 preprint 标签 |

### 8.1 四类综合，而非单线替代

| 综合路线 | 代表一手来源 | 被改变的轴 | 公平验收问题 |
|---|---|---|---|
| Fixed-grid | Causal VAE、BSQ-ViT、CV-VAE | 规则 $T'\times H'\times W'$ 网格中的编码器、量化或兼容性 | 相同 shape、dtype 与 decoder 下，重建和下游生成是否改善？ |
| Adaptive budget | ElasticTok、InfoTok、AdapTok | 每帧或每块使用的 token 数由内容决定 | router/scorer、变长 batching 与最坏情况长度计入后，端到端 SLO 是否改善？ |
| Structured latent | CoordTok、VidTwin、NeRV-Diffusion | 从规则局部网格转向 triplane、双分支或 INR weights | 结构归纳偏置是否经干预验证，并能跨时长/分辨率泛化？ |
| Generator-aware tokenizer | LARP、Divot | tokenizer 训练显式考虑上层 prior 或生成式 de-tokenizer | 固定 generator 与总算力后，收益是否仍来自表示本身？ |

四类并不互斥：一个系统可以既 causal fixed-grid，又使用 generator-aware loss；也可能在 structured latent 上再做 adaptive allocation。教材应把它们放在正交设计轴上，而不是写成“旧 tokenizer 被新 tokenizer 依次取代”。

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

## 10. 建议的教材验收协议

### 10.1 先选任务角色

1. 若目标是给定历史采样多个未来，进入 stochastic future 协议。
2. 若目标是将已知视频变成更短/更小的上层变量，进入 tokenizer 协议。
3. 若系统两者都做，必须拆成 codec reconstruction、latent generator 和 pixel output 三段验收。

### 10.2 Stochastic future 协议

- 同一历史下报告多次采样，不只报 best-of-$N$。
- 分别检查条件一致性、样本间多样性、真实未来覆盖和长期漂移。
- 报告 KL 时同时报 decoder 对 $z$ 的敏感性、active units/互信息代理和条件样本多样性；小 KL 本身不是 collapse 的充分证据。
- 分开训练 posterior 重建与测试 prior 采样，显式测量 prior–posterior gap。

### 10.3 Tokenizer/codec 协议

- 完整报告 `[B,C,T,H,W] -> [B,C_z,T',H',W']` 或 index shape，以及 nominal 与 exact 压缩比。
- 将 PSNR/SSIM、LPIPS/感知质量、rFVD/时间分布、OCR/人脸/小物体、快速运动和边界伪影分栏。
- 测试训练长度内外的短 clip、长 clip、任意尺寸、首帧、末帧、chunk 切分和 streaming cache reset。
- 报告 encode/decode wall-clock、峰值显存、设备、dtype、batch、帧数和分辨率。
- 只有实际量化、熵模型与 bitstream 时才报 bpp/bitrate；否则只报 latent/token 账本。
- 固定同一上层 generator 做 downstream generation 对照，避免 tokenizer 改进与 generator 规模改进混在一起。

## 11. 图示规范

### 11.1 两种角色分流图

- 输入节点：`Observed history`。
- 左支：`Training posterior sees target` → `KL-align learned prior` → `Sample possible futures`。
- 右支：`Encode known video` → `Compact latent/token` → `Downstream generator` → `Decode pixels`。
- 终端分别标注：`Evaluate future distribution` 和 `Evaluate reconstruction first`。
- 禁止把两条支路画成先后进化关系。

### 11.2 压缩账本与 causal 边界图

- 节点顺序：`RGB [B,C,T,H,W]` → `Left-only temporal context` → `Encoder/downsample` → `Continuous / discrete / hybrid` → `Decoder` → `Reconstruction`。
- 并列账本节点：`Grid ratio`、`Element ratio`、`Nominal token bits`、`Actual bitstream only with entropy coding`。
- 首帧节点标注“first temporal token anchors first frame”，不用 $T/f_t$ 模糊短 clip 边界。
- Mermaid 必须含 `accTitle` 和单行 `accDescr`，下方用顺序化文字完整复述。

## 12. 更新触发条件

出现以下任一情况时，应重新审计本章：

- KVAE 或其他 2026 技术报告获得正式 venue，标题/配置/数据发生变化。
- 主流开放模型公布可复现 bitrate、entropy coder 或标准化 bitstream，可将“tokenizer”与“codec”的边界改写得更具体。
- adaptive tokenizer 出现统一的变长 batching 与 end-to-end SLO 评测，能验证 token reduction 是否真正转化为延迟/显存收益。
- INR-weight latent 在更多数据、分辨率和条件生成任务中被独立复现，可以评估它是否超越特定 benchmark 的可迁移分支。
