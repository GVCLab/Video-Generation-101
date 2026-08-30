# 视频 Tokenizer、Codec 与生成式压缩：表示接口、位流边界与生成上限

> 本章资料核验截至 **2026-08-30**。正式论文构成主干；V-RAE、VideoRAE、KVAE 与 VidTok 保留预印本标签。这里的“压缩”先指 latent/token 表示变小，只有量化、概率模型、熵编码、位流语法与独立解码合同全部闭合时，才上升为实际 codec 主张。

## 1. 任务合同：先问它交付张量、符号，还是位流

视频 tokenizer 的最小接口是

```math
z=E(x),\qquad \hat x=D(z),
```

其中 $x$ 是已知视频，$z$ 是供重建或上层 generator 使用的表示。这个定义没有自动包含未来分布、熵编码或实时服务。

### 1.1 三个经常混用的层级

| 名称 | 最小组成 | 可以报告 | 不能自动报告 |
|---|---|---|---|
| **representation tokenizer** | encoder、latent/token 接口、decoder | shape、token 数、重建、encode/decode 成本 | bpp、bitrate、可交换文件 |
| **learned codec** | tokenizer + 量化 + 概率模型 + entropy coder + bitstream syntax | 实际文件大小、bpp/bitrate、rate–distortion | 随机访问、标准兼容或生成质量，除非另测 |
| **generative codec / compression** | learned codec + 感知、对抗或生成式先验补偿被丢信息 | rate–distortion–perception、时间稳定、幻觉率 | “补出的细节就是原始事实” |

因此：continuous VAE 没有量化与熵编码时只是 latent 压缩接口；VQGAN 有离散代码和感知/对抗 decoder，也不自动成为可传输 codec [[2]](#ref-2)；Divot 的 diffusion de-tokenizer 是生成式反解码机制，但没有位流合同就仍不是 learned codec [[11]](#ref-11)。

若任务是“同一历史之后采样多个合理未来”，latent 表达的是未来不确定性，应转到[变分视频生成](variational-generation.md)。两者都可能有 Gaussian latent 和 KL，却不是同一个问题。

### 1.2 表示、因子分解、训练目标是三条轴

![三条视频 token 生成路线：连续 latent 路线将 RGB 视频编码为连续时空 latent，再由 diffusion、flow 或 continuous autoregressive head 建模并解码；离散 token 路线先量化为 code IDs，再用 categorical AR 或 masked generator 预测；masked/discrete diffusion 路线对离散格点做遮盖或离散噪声恢复。图中强调 representation、factorization、training head 与 deployment claim 不等价。](../../assets/diagrams/video-token-generation-routes.png)

图的阅读顺序是：先确定 representation 是 continuous latent、discrete code ID 还是 patch token；再确定联合分布由 AR、masked、diffusion 或 flow 怎样分解；随后看 training objective；最后才讨论 latency、streaming 和 codec。一个 tokenizer 可以接多个 generator，一个 generator family 也可以换 tokenizer。

本章固定三个术语，避免把所有单元都叫 token：

- **continuous latent**：浮点或量化后的连续张量元素；
- **discrete code / token ID**：来自有限状态集合的离散符号；
- **patch token**：Transformer 接收的序列单元，可能承载连续向量，也可能承载离散 code embedding。

## 2. 表示谱系：连续、离散与真正的 hybrid

### 2.1 Continuous latent

连续 tokenizer 输出

```math
z\in\mathbb R^{B\times C_z\times T'\times H'\times W'}.
```

Gaussian diffusion 或 flow 通常直接在该欧氏空间建模。优点是无最近邻量化误差、梯度路径直接；代价是 latent 尺度、通道数、dtype 和分布几何都会影响上层学习。没有量化精度与 entropy coder，就不能从 shape 推出 bitrate。

CV-VAE 以连续 3D VAE 对齐已有 image-VAE latent，核心是兼容现有 latent generator [[6]](#ref-6)。OmniTokenizer 用 spatial window attention 与 temporal causal attention 联合处理图像和视频，并分别发布 VAE 与 VQVAE 模式 [[7]](#ref-7)。“一个项目有两类 checkpoint”不等于一个样本同时使用 hybrid latent。

### 2.2 VQ：learned codebook

VQ-VAE 将 encoder 输出映射到最近的 codebook 向量 [[1]](#ref-1)：

```math
j^*=\arg\min_j\lVert z_e(x)-e_j\rVert_2^2,
\qquad z_q(x)=e_{j^*}.
```

典型目标是

```math
\mathcal L_{\mathrm{VQ}}
=\mathcal L_{\mathrm{rec}}
+\lVert \mathrm{sg}[z_e]-e_{j^*}\rVert_2^2
+\beta\lVert z_e-\mathrm{sg}[e_{j^*}]\rVert_2^2,
```

其中 stop-gradient 与 straight-through estimator 让 decoder 梯度回到 encoder。VQ 的主要风险是量化误差、dead codes、usage 不均和序列过长。TATS 把 3D-VQGAN 与 time-sensitive Transformer 结合 [[3]](#ref-3)，MAGVIT 则把 3D tokenizer 接到 masked parallel generator [[4]](#ref-4)；两者都说明 tokenizer 与 generator 应拆开归因。

### 2.3 Lookup-free：LFQ、FSQ 与 BSQ 不是同一个算法

| 机制 | 离散化核心 | 显式 learned codebook | 关键边界 |
|---|---|---|---|
| **LFQ** | 逐维 sign / binary assignment | 否 | MAGVIT-v2 的 entropy penalty 是训练正则，不是 entropy coder [[5]](#ref-5) |
| **FSQ** | 每个标量量化到有限 levels，笛卡尔积形成状态 | 否 | nominal 状态数与实际码长需分开；项目的 FSQ checkpoint 不等于 hybrid |
| **BSQ** | 投影、$\ell_2$ 球面归一化后 sign quantization | 否 | 与 LFQ 都 lookup-free，但球面几何不同；BSQ-ViT 另接 AR 概率模型与算术编码 [[9]](#ref-9) |

MAGVIT-v2 用 LFQ 支撑更大隐式词表与 LM 视觉生成 [[5]](#ref-5)，但论文标题不构成“任意 LM 普遍胜过 diffusion”的定理。BSQ-ViT 是本章目标集合中唯一明确跨到实验性实际位流层的正式论文：约 300M 的 AR 概率模型加 adaptive arithmetic coding，并报告实际 bpp / rate–distortion [[9]](#ref-9)。这仍不等于已定义标准容器、随机访问和跨实现兼容。

### 2.4 Hybrid 必须发生在同一个样本合同内

真正的 hybrid 至少要求一个样本同时传递两类互补表示：

1. **离散主码 + 连续残差**：HART 用离散 token 表示粗结构，再由连续 residual diffusion 补细节 [[10]](#ref-10)；主要证据来自图像，不能直接外推视频时序。
2. **离散流 + 连续流**：TVC 在超低码率视频压缩中联合编码两条流，并实际评估位流 [[16]](#ref-16)。这是 codec 证据，不是视频生成排名。

VidTok、OmniTokenizer、Cosmos 或 VideoRAE 各自提供 continuous/discrete 模式时，应按具体 checkpoint 判断；两种模式并列发布不是 hybrid。

## 3. 四本压缩账：从 shape 到真实位流

设 RGB 视频、连续 latent 和离散 code map 分别为

```math
x\in\mathbb R^{B\times C\times T\times H\times W},\quad
z\in\mathbb R^{B\times C_z\times T'\times H'\times W'},\quad
i\in\{1,\ldots,K_c\}^{B\times T'\times H'\times W'}.
```

必须分别报告

```math
r_t=\frac{T}{T'},\qquad
r_{hw}=\frac{HW}{H'W'},\qquad
r_{grid}=\frac{THW}{T'H'W'},
```

```math
r_{elem}=\frac{CTHW}{C_zT'H'W'},\qquad
N_{token}=T'H'W'.
```

![图 034：视频 tokenizer 的张量与码率四本账](assets/imagegen-diagrams/034/diagram.png)
顺序化文字替代：输入 `[B,C,T,H,W]` 经 encoder 形成 continuous `[B,Cz,T',H',W']` 或 discrete `[B,T',H',W']`。第一本账只计算时间、空间和时空网格比；第二本账加入输入/latent 通道与 dtype；第三本账用 $N_{token}$ 与 $\log_2K_c$ 计算离散代码的名义容量；第四本账必须再经过概率模型、熵或算术编码，并计入 header、chunk、index 与 side information，才得到 actual bits、bpp 和带 duration/FPS 的 bitrate。Cosmos 官方示例把 `[1,3,9,512,512]` 变成 continuous `[1,16,3,64,64]` 或 discrete `[1,3,64,64]` [[20]](#ref-20)：精确 $r_t=3$、$r_{hw}=64$、$r_{grid}=192$、continuous $r_{elem}=36$；名义 $t4s8s8$ 的长片渐近值才是 $r_{grid}=256$、$r_{elem}=48$。配置名不能替代有限 clip 的实际 shape。

### 3.1 四本账各自回答什么

- $r_t$：该有限 clip 的实际时间位置降幅，不保证等于配置 nominal factor；
- $r_{hw}$：每帧空间位置数降幅；
- $r_{grid}$：时空位置总数降幅，常被粗写成 $f_t f_h f_w$；
- $r_{elem}$：再计输入通道 $C$ 与 latent 通道 $C_z$，仍未计 dtype；
- $N_{token}$ 与 $\log_2K_c$：只给固定长度代码的名义容量；
- actual bpp / bitrate：必须来自真实 bitstream 文件；bitrate 还要给 FPS 或 duration。

若 continuous latent 是 FP16、输入是 8-bit RGB，存储字节账与 $r_{elem}$ 仍不同。若离散符号分布偏斜，熵编码后的平均码长可低于 $\log_2K_c$；但 header、mask、长度和随机访问索引又会增加真实文件大小。

### 3.2 位流成熟度 BS0–BS3

这里的 **BS0–BS3** 只描述 bitstream maturity，不等于全书[评测证据 L0–L7](../evaluation.md)梯度。

| 等级 | 合同 | 能否叫实际 codec |
|---|---|---|
| BS0 | 只缩小 continuous tensor | 否；只能报 shape、elements、dtype/storage |
| BS1 | 产生 discrete symbols/bits，只报 token 数、词表位数或 nominal bpp | 否；还没有真实 entropy-coded 文件 |
| BS2 | 有概率模型、entropy/arithmetic coder 和实际 bpp–RD 实验 | 可称实验性 learned codec，但仍需核开放 artifact |
| BS3 | 可交换 bitstream、header、随机访问、版本语法与独立 decoder | 可审计部署合同；本章目标列表中无完整 BS3 证据 |

InfoTok 的 $\mathrm{BPP}_{16}$ 是 token 数乘 nominal bits 并加 mask cost，不是 entropy-coded bitstream；其 FSQ levels 为 $8^3 5^3=64{,}000$，也不严格等于 $2^{16}=65{,}536$ [[17]](#ref-17)。MAGVIT-v2 的 entropy penalty 是正则项，不是 coder。只有把 `.bin` 交给另一个进程，后者仅凭位流与公开 decoder 恢复视频，才闭合真实 codec 验收。

## 4. 时间合同：causal codec 不推出 streaming 系统

### 4.1 四种不能混写的时间边界

| 合同 | 当前输出可看什么 | 代表性提醒 |
|---|---|---|
| noncausal / full-clip | 整段过去与未来 | LARP 的 holistic queries 读完整视频；AR prior 只描述 latent 次序 [[13]](#ref-13) |
| frame-causal | 当前和更早帧 | 需用任意未来扰动做 prefix-invariance |
| block-causal | 当前块内可双向，只禁止读取未来块 | ElasticTok 是 4-frame block-causal，不应写成逐帧 causal [[12]](#ref-12) |
| chunk-causal implementation | 以 cache、warm-up、overlap/crop 处理分块 | 是否等价于 full-clip、是否可逐帧提交要另测 |

一种常见首帧锚定规则是

```math
T'=1+\left\lfloor\frac{T-1}{f_t}\right\rfloor
=\left\lceil\frac{T}{f_t}\right\rceil,
```

但不同实现有不同 padding、stride、chunk 和 crop，**实际 API 输出 shape 优先于公式或型号名**。Cosmos 明确第一个 temporal token 锚定第一帧 [[20]](#ref-20)；HunyuanVideo 公开 CausalConv3D、名义 $t4s8s8$ 与 16 latent channels，这足以核 shape，不足以推 bitrate [[21]](#ref-21)。

![图 035：因果视频 tokenizer 的首帧与分块边界](assets/imagegen-diagrams/035/diagram.png)
顺序化文字替代：输入 $x_1\ldots x_9$ 经过只读当前与过去的 codec，形成 $z_0,z_1,z_2$；第一个 temporal token 锚定第一帧，后续 receptive field 可以重叠，不能画成永远独占的四帧桶。实现可从 cold start/reset 进入 chunk $k$，也可携带 chunk $k-1$ 的 cache；随后可能有 warm-up、overlap 和 crop，只有 committed frames 可见，并须比较 cache carry 与 reset 的接缝。causal codec 不推出 causal upper generator，后者也不推出 streaming commit 或 real-time SLO；TTFF、稳态 FPS/deadline、峰值显存和质量漂移都要测量。

### 4.2 Prefix-invariance 是最小因果证据

构造 $x,x'$，让二者在时间块 $b$ 之前完全相同、未来任意不同。frame-causal 合同要求

```math
E(x)_{\le b}=E(x')_{\le b},\qquad
D(E(x))_{\le b}=D(E(x'))_{\le b}.
```

block-causal 只在块边界验收。还要比较 full-clip 与不同 chunk size/cache 策略的输出；任何前缀变化或 full/chunk 不一致，都否定对应 causal/streaming 合同。

VidTwin 的 backbone temporal attention 有 causal mask，但 structure Q-Former 查询整段时间序列，不能据此声称端到端 prefix-causal [[15]](#ref-15)。KVAE 的 attention-free causal Conv3D 和公开 cache/chunk 接口是较强实现证据，但并发 cache、chunk 等价性、reset seam 与延迟仍需实测 [[22]](#ref-22)。

## 5. Reconstruction ceiling：输入忠实上限与生成可学性是两件事

固定 $E,D$ 后，$D(E(x))$ 决定输入中的哪些差异还能被可靠恢复。如果文字、小物体、高速运动或微小视差在 latent 中不可区分，上层 generator 无法从同一个 latent 稳定恢复原事实。这是 tokenizer 对**输入忠实度**的表示上限。

它不是“美学质量”的严格上限：perceptual/adversarial decoder 或 diffusion de-tokenizer 可以补出锐利、合理却不是输入事实的纹理。至少分栏报告：

- pixel fidelity：PSNR、SSIM 与区域误差；
- perceptual similarity：LPIPS 等，但不把锐利当忠实；
- temporal stability：运动边缘、闪烁、切镜、循环重编码；
- factual detail：OCR、人脸、小物体、数量、手部与关系；
- boundary robustness：短片、首末帧、非整除尺寸、chunk/reset seam；
- encode/decode system cost：wall-clock、峰值显存、设备、batch、dtype、分辨率。

更高 reconstruction 分数也不保证更好 generation：latent 可能尺度不稳、几何不规则、序列过长或含上层难以预测的局部细节。正确归因必须固定 generator 架构、训练数据、训练 FLOPs、sampling protocol 与 latent budget，只替换 tokenizer，并同时报告 reconstruction 与 downstream generation。

## 6. 2023–2026 技术路线：四条正交分支

### 6.1 Fixed-grid 与量化仍是主干

规则 $T'\times H'\times W'$ 网格最容易 batch、并行与位置对齐。MAGVIT、MAGVIT-v2、CV-VAE、Causal VAE 与 BSQ-ViT分别推进 3D tokenization、lookup-free 量化、image-VAE 兼容、图像—视频联合因果编码和实际熵编码 [[4]](#ref-4) [[5]](#ref-5) [[6]](#ref-6) [[8]](#ref-8) [[9]](#ref-9)。固定网格的代价是：静止背景与快速运动得到相同位置预算。

### 6.2 Adaptive budget：配置可选不等于按样本自适应

必须区分：

- **多 checkpoint / 多配置**：用户选一个固定网格；VidTok、Cosmos、KVAE 的多压缩型号属于此类 [[20]](#ref-20) [[22]](#ref-22) [[23]](#ref-23)。
- **可截断的有序前缀**：ElasticTok 以随机 tail masking 学会变长表示，但部署仍需 length search/selection [[12]](#ref-12)。
- **样本或块自适应**：InfoTok 用 reconstruction-error/ELBO router 选择高信息 token，需额外 decoder pass [[17]](#ref-17)；AdapTok 用 causal scorer 与 IPAL 在预算下分配长度 [[18]](#ref-18)。

“平均 token 少”不是系统效率。必须计入 router/scorer、search NFE、额外 decode、ILP、padding/packing、ragged batching、p95 latency、最坏 token 数与质量尾部。LARP 的 query 数可配置，但一个配置内固定，且 holistic query 看完整视频，不应误归为 causal adaptive tokenizer [[13]](#ref-13)。

### 6.3 Structured latent：改变一个表示单元是什么

- **Holistic queries**：LARP 用 learned queries 聚合整段视频，并以轻量 AR prior 让 latent 更适合后续 AR 建模 [[13]](#ref-13)。其 AR 是 latent 次序约束，不是输入时间因果。
- **Coordinate triplanes**：CoordTok 用 $xy$、$yt$、$xt$ 三个 plane 与坐标 patch reconstruction 表示长视频 [[14]](#ref-14)。作者的 1280-token 数值只在相同 128-frame、128×128 和 decoder 协议内可比较。
- **Structure/dynamics branches**：VidTwin 将两类 latent 分支分开 [[15]](#ref-15)；架构命名不是语义解耦充分证据，必须 swap、probe 与干预。
- **INR-weight latent**：NeRV-Diffusion 把整段视频编码成实例 INR 的网络权重，再让 DiT 在 weight latent 上去噪 [[19]](#ref-19)。这改变了 denoiser 的几何先验，也引入权重对称性、实例拟合误差与跨时长泛化问题。

### 6.4 Generator-aware 与 semantic representation

只优化 reconstruction 的 latent 未必易于生成或理解。LARP 在 tokenizer 训练中加入轻量 AR prior [[13]](#ref-13)；Divot 学 continuous semantic representation，并用 diffusion generative de-tokenizer 与 GMM prior 支持理解/生成 [[11]](#ref-11)。这类方法必须在固定上层模型和总算力下对照，否则收益可能来自额外 generator 容量。

2026 的 V-RAE 与 VideoRAE 进一步用 frozen video foundation features 或 representation alignment，让 latent 同时保留感知与语义结构 [[24]](#ref-24) [[25]](#ref-25)。两者截至核验日均是预印本：

- V-RAE 的时间合同依 encoder variant 而变；DINO/SigLIP/EUPE 变体是 chunk-causal，V-JEPA2.1 变体非因果，chunk 内仍可双向；
- VideoRAE 分别提供 continuous 与 multi-codebook SimVQ 模式，不是一个 hybrid latent；其完整视频 encoder 没有提出 causal 合同；
- 两者的受控 generation 实验支持“更可生成的表示”假设，但尚不能写成社区共识。

## 7. 里程碑与正式状态

| 首次公开 / 正式状态 | 节点 | 真正改变的轴 | 位流级别 | 不能越界的结论 |
|---|---|---|---|---|
| 2017 / NeurIPS 2017 | VQ-VAE [[1]](#ref-1) | learned codebook 离散表示 | BS1 | 基础视频实验不等于现代高清性能 |
| 2021 / CVPR 2021 | VQGAN [[2]](#ref-2) | 感知/对抗离散 decoder | BS1 | 锐利不等于输入忠实或实际 codec |
| 2022 / ECCV 2022 | TATS [[3]](#ref-3) | 3D-VQGAN + 时序 Transformer | BS1 | 长 rollout 不等于无漂移 |
| 2022 / CVPR 2023 | MAGVIT [[4]](#ref-4) | 3D VQ + masked parallel generation | BS1 | tokenizer 与 generator 收益需拆分 |
| 2023 / ICLR 2024 | MAGVIT-v2 [[5]](#ref-5) | LFQ 与大隐式词表 | BS1 | entropy regularizer 不是 coder |
| 2024 / NeurIPS 2024 | CV-VAE；OmniTokenizer [[6]](#ref-6) [[7]](#ref-7) | continuous compatibility；joint image-video | BS0 / BS0–1 | checkpoint 兼容与 hybrid 不可混写 |
| 2024 / ICLR 2025 | Causal VAE；BSQ-ViT；ElasticTok；LARP [[8]](#ref-8) [[9]](#ref-9) [[12]](#ref-12) [[13]](#ref-13) | causal joint、BSQ、可截断预算、generator-aware | BS0 / **BS2** / BS0–1 / BS1 | 只有 BSQ-ViT 明确实验性实际码流 |
| 2024 / CVPR 2025 | CoordTok；VidTwin；Divot [[14]](#ref-14) [[15]](#ref-15) [[11]](#ref-11) | triplane、双分支、semantic detokenizer | BS0 | token 数与“解耦”需受控验证 |
| 2025 / ICLR 2026 | InfoTok；NeRV-Diffusion [[17]](#ref-17) [[19]](#ref-19) | 样本自适应；INR weights | BS1 / BS0 | nominal BPP 不是位流；权重 latent 非局部网格 |
| 2025 / CVPR 2026 | AdapTok [[18]](#ref-18) | block-causal 1D adaptive latent | BS1 | scorer、IPAL、ragged cost 必须计入 |
| 2026 / preprint watchlist | V-RAE；VideoRAE；KVAE [[24]](#ref-24) [[25]](#ref-25) [[22]](#ref-22) | semantic representation；更高压缩 causal VAE | BS0–1 | 尚无正式 venue，不写成定论 |

“首次公开年 / 正式发表年”双列避免时间线漂移：例如 MAGVIT 是 2022 arXiv / CVPR 2023，MAGVIT-v2 是 2023 arXiv / ICLR 2024。专题比较优先使用正式状态；全书[时间线](../timeline.md)按首次公开定位并附 venue。

## 8. 代表开放实现：artifact 也要分层

| 实现 | 截止日状态 | 可核验内容 | 不应据此推断 |
|---|---|---|---|
| CV-VAE [![GitHub: AILab-CVC/CV-VAE](https://img.shields.io/github/stars/AILab-CVC/CV-VAE?style=social)](https://github.com/AILab-CVC/CV-VAE) | NeurIPS 2024；官方代码与权重 | continuous video VAE；仓库列出与若干 image-VAE latent 兼容型号 | 任意未公开 image VAE 都可无损替换 |
| OmniTokenizer [![GitHub: FoundationVision/OmniTokenizer](https://img.shields.io/github/stars/FoundationVision/OmniTokenizer?style=social)](https://github.com/FoundationVision/OmniTokenizer) | NeurIPS 2024；官方仓库 | joint image-video；VQVAE 与 VAE 配置 | 两种 checkpoint 自动构成 hybrid |
| Cosmos Tokenizer [![GitHub: NVIDIA/Cosmos-Tokenizer](https://img.shields.io/github/stars/NVIDIA/Cosmos-Tokenizer?style=social)](https://github.com/NVIDIA/Cosmos-Tokenizer) | 官方仓库与模型；代码/模型 license 分开 | continuous/discrete、temporal 4/8、spatial 8/16、API shape | README 性能宣传已被独立复现 |
| HunyuanVideo [![GitHub: Tencent-Hunyuan/HunyuanVideo](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanVideo?style=social)](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 作者报告；官方代码与权重 | CausalConv3D VAE、$t4s8s8$、16 channels | 256× 元素压缩或实际 bitrate |
| [VidTok](https://huggingface.co/microsoft/VidTok) | arXiv 2412.13061；项目与模型公开 | continuous-KL 与 discrete-FSQ 型号、causal/noncausal 配置 [[23]](#ref-23) | 截止日已有正式 venue；VCR 就是 bpp |
| KVAE [![GitHub: kandinskylab/kvae](https://img.shields.io/github/stars/kandinskylab/kvae?style=social)](https://github.com/kandinskylab/kvae) | arXiv 2608.05798；代码与权重 | video 2.0 的 $t4s8$、$t4s16$、causal cache/chunk 接口 | 已同行评审；公开 cache 即端到端实时 |

有推理代码不等于有训练代码；有权重不等于训练数据与配方公开；代码许可证、模型许可证和依赖许可证也可能不同。

## 9. TokenizerFork-1：一套实验，拆开五种主张

### 9.1 冻结清单

- 数据 snapshot、train/test split、clip 抽样、分辨率与 FPS；
- encoder/decoder 参数量级、latent positions、训练 FLOPs 与优化步数；
- downstream generator 架构、input projection、训练数据、步数、采样器和 seed；
- 评测脚本、硬件、dtype、batch、warm-up 与计时边界。

### 9.2 五个最小可证伪分叉

| 分叉 | 只改变什么 | 必须报告 | 证伪条件 |
|---|---|---|---|
| **量化器替换** | VQ / LFQ / FSQ / BSQ | 匹配 nominal bits 与经验 symbol entropy；重建、usage、吞吐、生成 | 优势在同码长同算力下消失，就不能归因量化机制 |
| **因果合同** | full、frame/block causal、chunk/cache | prefix-invariance、full/chunk 等价、reset seam | 未来扰动改变前缀，或 chunk 输出不等价 |
| **adaptive 成本** | fixed、adaptive、oracle，在同平均/最坏预算下 | router/search/ILP、packing、p95、VRAM、质量尾部 | token 少但 wall-clock/VRAM 不降，或尾部质量崩坏 |
| **生成/语义对齐** | 只换 tokenizer | reconstruction、semantic probe、gFVD/VBench、收敛 AUC | 匹配 generator/FLOPs 后收益消失 |
| **structured 干预** | plane/branch 删除、置换、跨视频 swap | 身份/运动 probe、选择性迁移、重建与生成 | 各分支同样携带全部属性，强解耦主张失败 |

### 9.3 真实 codec 验收

codec 分支必须额外交付：

1. 编码器实际写出的独立 `.bin`；
2. 另一进程只凭 `.bin`、公开 decoder 与版本 manifest 重建；
3. 文件大小包含 symbols、adaptive mask、长度、header、index 与 side information；
4. 报实际 bpp、rate–distortion、encode/decode FPS、峰值显存和随机访问；
5. 损坏或版本不匹配时给出确定性错误，而非静默解码。

若数字只能由 $N\log_2K$、latent tensor bytes 或“compression ratio”推算，就停留在 BS0/BS1，不能称 actual bitrate。

## 10. 失败定位与停止规则

| 症状 | 优先怀疑 | 最小诊断 | 不应立即下的结论 |
|---|---|---|---|
| 平均重建好，文字/人脸坏 | 高频与事实细节被压掉 | OCR、identity、小物体分层评测 | “PSNR 高所以 tokenizer 足够” |
| 静态好，运动闪烁 | temporal bottleneck、chunk seam | 运动边缘、reset/carry、循环重编码 | “单帧 LPIPS 好所以视频好” |
| latent 更短但训练不快 | router、ragged batch、kernel 利用率 | 端到端 wall-clock、p95、padding 浪费 | “token 少等于同比加速” |
| reconstruction 更好，generation 更差 | latent 几何或尺度难学 | 固定 generator 的收敛 AUC 与分布统计 | “decoder 还不够大” |
| branch 名称清晰但 swap 无选择性 | 未真正解耦 | probe、交换、干预 | “结构/动态标签就是因果因素” |
| nominal bpp 很低但无文件 | 缺概率模型/coder/side-info | 导出 `.bin` 并独立解码 | “已达到 codec 码率” |
| causal full-clip 好，chunk 接缝坏 | cache、pad/crop、warm-up 不一致 | 多 chunk size、reset/carry、prefix test | “causal 就能无缝 streaming” |

停止继续堆 tokenizer 容量的条件：若在相同 latent budget 与 generator 下，连续两轮只提高感知锐度，却不改善 OCR/身份/运动忠实、下游生成或端到端成本，应先定位 encoder 信息损失、decoder 幻觉、latent 几何和系统 packing，而不是继续扩大 decoder。

## 11. 与其他章节的接口

- continuous latent 常接[扩散模型](diffusion-models.md)或 [Flow / Consistency](flow-consistency-models.md)；
- discrete code ID 常接[自回归生成](autoregressive-generation.md)或[掩码生成](masked-generation.md)；
- stochastic future posterior/prior 属于[变分视频生成](variational-generation.md)，不是本章 codec 账本；
- causal codec 只有与 causal generator、cache、commit policy 和 SLO 共同闭合，才进入[因果流式生成](causal-streaming-generation.md)；
- 系统中的 checkpoint、训练、部署与 license 边界见[视频基础模型](../foundation-models.md)；
- 全局评测预算、人工评测与证据梯度见[评测指南](../evaluation.md)。

建议阅读顺序：第 1 节先确定交付物，第 3 节核四本账，第 4 节做因果边界测试，第 5–7 节理解生成上限和技术分叉，最后用第 9–10 节写可证伪实验。完整检索式、纳排、证据等级和实现快照见[研究日志](../../sources/research_20260830_video_representation_tokenizers.md)。

## 参考文献

<a id="ref-1"></a>[1] [Neural Discrete Representation Learning](https://papers.nips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html). Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. NeurIPS 2017.

<a id="ref-2"></a>[2] [Taming Transformers for High-Resolution Image Synthesis](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html). Patrick Esser, Robin Rombach, Björn Ommer. CVPR 2021.

<a id="ref-3"></a>[3] [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5950_ECCV_2022_paper.php). Songwei Ge et al. ECCV 2022.

<a id="ref-4"></a>[4] [MAGVIT: Masked Generative Video Transformer](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html). Lijun Yu et al. CVPR 2023.

<a id="ref-5"></a>[5] [Language Model Beats Diffusion - Tokenizer is key to visual generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html). Lijun Yu et al. ICLR 2024.

<a id="ref-6"></a>[6] [CV-VAE: A Compatible Video VAE for Latent Generative Video Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html). Sijie Zhao et al. NeurIPS 2024.

<a id="ref-7"></a>[7] [OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html). Junke Wang et al. NeurIPS 2024.

<a id="ref-8"></a>[8] [High-Quality Joint Image and Video Tokenization with Causal VAE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/03df5246cc78af497940338dd3eacbaa-Abstract-Conference.html). Dawit Mureja Argaw et al. ICLR 2025.

<a id="ref-9"></a>[9] [Image and Video Tokenization with Binary Spherical Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e25198b6a75f74277ee3a2bd4165d9ef-Abstract-Conference.html). Yue Zhao, Yuanjun Xiong, Philipp Krähenbühl. ICLR 2025.

<a id="ref-10"></a>[10] [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ab4e1e704c68b0f476c265996f08d283-Abstract-Conference.html). Haotian Tang et al. ICLR 2025.

<a id="ref-11"></a>[11] [Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Ge_Divot_Diffusion_Powers_Video_Tokenizer_for_Comprehension_and_Generation_CVPR_2025_paper.html). Yuying Ge et al. CVPR 2025.

<a id="ref-12"></a>[12] [ElasticTok: Adaptive Tokenization for Image and Video](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html). Wilson Yan et al. ICLR 2025.

<a id="ref-13"></a>[13] [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](https://proceedings.iclr.cc/paper_files/paper/2025/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html). Hanyu Wang et al. ICLR 2025 Oral.

<a id="ref-14"></a>[14] [Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Efficient_Long_Video_Tokenization_via_Coordinate-based_Patch_Reconstruction_CVPR_2025_paper.html). Huiwon Jang et al. CVPR 2025.

<a id="ref-15"></a>[15] [VidTwin: Video VAE with Decoupled Structure and Dynamics](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VidTwin_Video_VAE_with_Decoupled_Structure_and_Dynamics_CVPR_2025_paper.html). Yuchi Wang et al. CVPR 2025.

<a id="ref-16"></a>[16] [TVC: Tokenized Video Compression with Ultra-Low Bit Rate](https://link.springer.com/article/10.1007/s44267-025-00098-7). Visual Intelligence, 2025.

<a id="ref-17"></a>[17] [InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression](https://proceedings.iclr.cc/paper_files/paper/2026/hash/432f048a844654ba981953491e6dc80e-Abstract-Conference.html). Haotian Ye et al. ICLR 2026 Oral.

<a id="ref-18"></a>[18] [AdapTok: Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space](https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html). Yan Li et al. CVPR 2026.

<a id="ref-19"></a>[19] [NeRV-Diffusion: Diffuse Implicit Neural Representation for Video Synthesis](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1a17a06de88cf77f25cda0da91615a54-Abstract-Conference.html). Yixuan Ren et al. ICLR 2026.

<a id="ref-20"></a>[20] [Cosmos Tokenizer](https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/). NVIDIA Research. Official project and repository documentation, accessed 2026-08-30.

<a id="ref-21"></a>[21] HunyuanVideo: A Systematic Framework For Large Video Generation Model [![GitHub: Tencent-Hunyuan/HunyuanVideo](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanVideo?style=social)](https://github.com/Tencent-Hunyuan/HunyuanVideo). Tencent Hunyuan. Author technical report and official repository, accessed 2026-08-30.

<a id="ref-22"></a>[22] [KVAE: Family of Tokenizers for Multimodal Generative Models](https://arxiv.org/abs/2608.05798). arXiv:2608.05798, preprint, 2026.

<a id="ref-23"></a>[23] [VidTok: A Versatile and Open-Source Video Tokenizer](https://arxiv.org/abs/2412.13061). Microsoft Research. arXiv preprint, 2024.

<a id="ref-24"></a>[24] [V-RAE: Rethinking Video Latent Spaces for Generation](https://arxiv.org/abs/2608.13556). Minghui Guo, Shengqiong Wu, Hao Fei. arXiv preprint, 2026.

<a id="ref-25"></a>[25] [VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders](https://arxiv.org/abs/2607.14088). Zhihao Xie, Junfeng Wu, Xinting Hu, et al. arXiv preprint, 2026.
