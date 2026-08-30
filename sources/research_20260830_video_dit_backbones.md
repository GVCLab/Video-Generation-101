# Video DiT / Backbone Scaling 研究记录（2026-08-30）

> 对应正文：[`docs/generative-models/video-dit-backbones.md`](../docs/generative-models/video-dit-backbones.md)。本记录保存问题冻结、检索与筛选口径、first-public/formal 分离、数值协议、原创图生成与视觉验收，以及尚未运行的可证伪实验。它不是独立 benchmark，也不把作者自报结果升级为独立复现。

## 1. 为什么新增独立专章

仓库原来的 backbone 内容主要是总览表，尚未系统回答：

- latent grid、patch 与 backbone token 的数量关系；
- full、factorized、window、sparse、linear、recurrent、hybrid attention 的真实连边与复杂度；
- cross-attention、AdaLN、Expert AdaLN、MMDiT/dual-stream→single-stream 与 MoE 的差异；
- 3D RoPE、FPS、宽高比、frame packing 与训练网格外推；
- token reduction、active/total parameters、NFE、quantization、parallelism、cache 的独立成本轴；
- matched-backbone 与 fixed-checkpoint serving 两类公平实验。

现有总览把 cascade、rolling cache、autoregressive 和 MoE 放在同一 backbone 表中，容易把系统组合、执行方式、factorization 与网络内部路由混为一类。本轮因此新增机制专章，但不把 backbone 误列为新的应用任务。

## 2. 研究问题冻结

检索前固定以下问题，避免只追逐最新模型名：

1. Video DiT 的最小接口是什么？它与 codec、objective、sampler 和 deployment 的责任边界在哪里？
2. 当 $N=T_pH_pW_p$ 时，dense、factorized、window、sparse 与 linear attention 的计算/显存主项分别是什么？
3. “3D attention”“3D RoPE”“causal mask”和“3D VAE”是否是四个不同属性？
4. 文本/视频融合中的 cross-attention、joint attention、dual→single stream 与 Expert AdaLN 分别做什么？
5. noise-timestep MoE 是否是 token-routed MoE？`total parameters`、`active parameters` 和 resident memory 怎样分账？
6. context/sequence parallel 是否降低算法总复杂度？通信、head 可整除与强扩展怎样报告？
7. inter-step feature/cache、attention sparsity reuse、data-time KV cache 和 cross-depth residual reuse 是否相同？
8. 线性、hybrid、稀疏、量化、少步蒸馏的作者速度怎样保留 protocol，避免倍数相乘？
9. 什么反证会推翻“全局更一致”“线性无损扩展”“位置可外推”“cache 精确”等主张？

## 3. 检索日期、表面与代表查询

专项检索与 fresh-read 于 **2026-08-30（Asia/Shanghai）** 完成。搜索引擎只用于发现，最终教材事实回到正式 proceedings、作者稿、机构技术报告、官方代码或官方模型文档。

| 检索表面 | 代表查询 | 目的 |
|---|---|---|
| CVF / ECCV / ICLR / PMLR / MLSys 正式页 | `site:openaccess.thecvf.com CVPR 2026 video diffusion attention sparse linear`；`site:proceedings.iclr.cc 2026 video generation attention`；论文全名 | 核对正式标题、venue、作者摘要与协议数字 |
| arXiv 作者稿 | `Video DiT full spatiotemporal attention 3D RoPE`；`SANA-Video 2.0`；`HunyuanVideo 1.5 SSTA` | 读取正式摘要未展开的方法、公式、消融和限制 |
| 官方 GitHub / 文档 | `Wan2.2 timestep expert routing`；`Open-Sora STDiT temporal RoPE`；`SANA Video 2 checkpoint` | 核对 release surface、配置、路由代码与“coming soon”状态 |
| 机制定向查询 | `video diffusion inter-step cache attention sparsity`；`distributed sparse attention video DiT`；`hybrid linear softmax video diffusion` | 找到 PAB、AdaCache、RAPID、DSA、BLADE、ReHyAt 等分支 |
| 反例查询 | `video DiT full attention divided attention ablation`；`linear attention unstructured sparse kernel limitation` | 主动找与“更高效必然更好”相冲突的作者证据 |

## 4. 纳入、排除与证据等级

### 4.1 纳入标准

- 能直接支撑 backbone interface、attention topology、position/fusion、MoE、parallelism、cache 或 quantization 的机制主张。
- 有可核验的 first-public 时间或正式 proceedings；正式状态与预印本/仓库状态分开。
- 速度或质量数字必须能找到硬件、分辨率/帧数、步数、precision、计时边界或 baseline 中的至少主要项；缺项会在正文明确。
- 开放性按 code、weights、config、training recipe、license 分面记录，不用“开源”一词代替全部发布面。

### 4.2 排除标准

- 只凭产品名、演示或第三方博客猜测 block 结构。
- 把图像 DiT/MMDiT 的实验直接写成视频运动证据；只把它们作为架构祖先。
- 把 codec compression、patch size、attention sparsity、NFE、quantization 或多卡加速合并成一个“模型效率”。
- 缺少同输出合同，却把不同分辨率、时长、步数、CFG、GPU 与计时范围的速度横排。
- 把 TimeRipples 的比例估算写成已由兼容 sparse kernel 实测，或把 Wan2.2 的基础 Wan citation 当作独立 MoE 论文。

### 4.3 来源等级

| 等级 | 定义 | 可支撑 | 不可支撑 |
|---|---|---|---|
| A | 官方会议/期刊 proceedings | 正式状态、公开论文的机制与作者实验 | 未公开产品实现、跨协议通用排序 |
| B | 作者 arXiv/技术报告 | 作者公开的结构、公式、消融、限制 | 同行评审与独立复现 |
| C | 作者/机构官方仓库、代码、模型卡或文档 | release surface、配置、代码路径、运行条件 | README 的质量/速度宣传成为独立结论 |
| X | 搜索摘要、第三方聚合/新闻/博客 | 发现候选 | 任何最终教材事实 |

## 5. 机制定义与命名消歧

| 易混名称 | 本轮操作定义 | 明确不等于 |
|---|---|---|
| Video DiT | 在视频 latent/patch token 上实现 denoising/score/velocity 等条件映射的 Transformer backbone | diffusion objective、flow matching、sampler、video tokenizer |
| 3D full attention | 所有时空 video token 在一次 attention 中形成全连接 score graph | 3D RoPE、3D VAE、文本也一定进入同一 self-attention |
| 3D RoPE | 在 head channel 子空间按 $t,h,w$ 坐标旋转 $Q,K$ | 改变 attention topology 或降低复杂度 |
| Expert AdaLN | 文本/视频等模态使用专属 adaptive normalization 参数 | token/top-k MoE、按噪声阶段切整套 expert |
| Noise-time MoE | 根据 diffusion/flow 时间 $\tau$ 选择 denoiser expert | 沿视频时间 $k$ 分段、content-adaptive token routing |
| Context/sequence parallel | 将同一样本的 token/context 分给多设备 | 把全局 $O(N^2)$ 总工作变成 $O(N)$ |
| Inter-step cache | 跨 denoising step $\tau_i\rightarrow\tau_{i+1}$ 复用 activation、residual 或 attention 结果 | causal AR 在视频时间上的 KV cache |
| Cross-depth reuse | 在同一次 denoiser forward 的不同 block/layer 之间复用 feature | inter-step cache；SANA2 AttnRes 属于这一类 |
| Intra-attention reuse | 在一次 attention 内复用相邻 token/channel 的 score 计算 | inter-step cache；TimeRipples 属于这一类 |

## 6. 一手来源账本

### 6.1 2022–2025：从 U-Net 前史到 full/joint Video DiT

| First-public → formal | 工作 | 纳入机制 | 关键边界 |
|---|---|---|---|
| 2022-04-07 → NeurIPS 2022 | [Video Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html) | 3D U-Net；空间/时间操作分解；图像/视频联合训练 | 不是 DiT；扩展采样会累积漂移 |
| 2022-09-29 → ICLR 2023 | [Make-A-Video](https://iclr.cc/virtual/2023/poster/11001) | T2I U-Net 中函数保持地加入时间卷积/attention | 不是 DiT；级联插帧/超分收益不能归给单 backbone |
| 2022-10-05 → 无正式 venue | [Imagen Video](https://arxiv.org/abs/2210.02303) | 七模型时空超分级联 | 不是单一 backbone；代码/权重未发布 |
| 2022-12-19 → ICCV 2023 | [DiT](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) | latent patch ViT、adaLN-Zero、图像 scaling | 图像架构桥梁，不是第一个 Video DiT |
| 2023-12 → ECCV 2024 | [W.A.L.T.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10270_ECCV_2024_paper.php) | spatial window + spatiotemporal window | 局部窗口限制直接全局通信；三级系统含多项贡献 |
| 2024-01-05 → TMLR 2025 | [Latte](https://openreview.net/forum?id=ntGPYNUF3t) | 四种 factorized Video DiT；space/time 交替 | 主要公开实验的时长/分辨率/数据规模有限 |
| 2024-02-15 → 技术报告 | [Sora report](https://openai.com/index/video-generation-models-as-world-simulators/) | compressed latent + spacetime patch Transformer | 未公开足以确认具体 block 首创的细节 |
| 2024-03 → ICML 2024 | [Stable Diffusion 3 / MMDiT](https://proceedings.mlr.press/v235/esser24a.html) | separate modality weights + bidirectional joint fusion | 图像架构祖先，不是视频时序实验 |
| 2024-03-17 release → 无正式 venue | [Open-Sora 1.0 / STDiT](https://github.com/hpcaitech/Open-Sora/releases/tag/v1.0.0) | 分解 spatial/temporal attention；后续 temporal/3D RoPE 与 QK norm | Open-Sora、Open-Sora Plan、Open-Sora 2.0 不可混称 |
| 2024-08 → ICLR 2025 | [CogVideoX](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce31378e9f41d8907e97dab172b6c559-Abstract-Conference.html) | joint full attention、Expert AdaLN、3D RoPE、frame packing | Expert AdaLN 不是 MoE；作者消融不等于跨模型定理 |
| 2024-10-22 → 官方 artifact | [Mochi 1 / AsymmDiT](https://huggingface.co/genmo/mochi-1-preview) | 文本/视觉不同 width/MLP，joint attention | Asymm 指模态容量，不是 causal/asymmetric attention |
| 2024-12-03 → 作者报告/开放实现 | [HunyuanVideo](https://arxiv.org/abs/2412.03603) | 20 dual-stream + 40 single-stream full-attention blocks、3D RoPE | 不是 60 层始终双流；13B/full attention 代价高 |
| 2024-12 → 作者预印本/开放实现 | [LTX-Video](https://arxiv.org/abs/2501.00103) | 高压缩 latent + full spatiotemporal attention | 作者 H100 速度依赖输出、步数和精度 |
| 2025-02-14 → 作者报告/开放实现 | [Step-Video-T2V](https://arxiv.org/abs/2502.10248) | 30B；video token 3D full self-attention；文本独立 cross-attention | “full”不表示文本与视频一定是 CogVideoX 式联合序列 |
| 2025-02-25 → 作者报告/开放实现 | [Wan2.1](https://arxiv.org/abs/2503.20314) | full spatiotemporal video self-attention + umT5 cross-attention | 1.3B/14B 是模型族；长序列 attention 仍是瓶颈 |
| 2025-07-28 → 官方 artifact | [Wan2.2](https://github.com/Wan-Video/Wan2.2) | high/low-noise two-expert hard switch；总约 27B、每步 active 约 14B | 无独立 Wan2.2 正式论文；不是 token-routed MoE |

### 6.2 2025–2026：复杂度、复用、并行与量化

| 状态 | 工作 | 机制 | 保留的作者协议 / 不能越界之处 |
|---|---|---|---|
| CVPR 2025 | [LinGen](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_LinGen_Towards_High-Resolution_Minute-Length_Text-to-Video_Generation_with_Linear_Computational_Complexity_CVPR_2025_paper.pdf) | 线性复杂度长视频架构 | “分钟级”是作者设置，需 matched quality/coverage |
| ICLR 2025 | [PAB](https://proceedings.iclr.cc/paper_files/paper/2025/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html) | attention output broadcast；broadcast sequence parallel | 最高 10.5× 是作者协议，cache 为近似 |
| ICCV 2025 | [AdaCache](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html) | motion/content-adaptive cache schedule | 最高 4.7× 为作者协议，需高速运动/scene cut 分桶 |
| MLSys 2025 | [ScaleFusion](https://openreview.net/pdf?id=anZWBeWnWh) | distributed spatiotemporal attention、sharding 与通信重叠 | 32 vs 8 A100 强扩展 3.60× 为作者设置；不减少总 FLOPs |
| ICLR 2026 Oral | [SANA-Video](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41b93c59da0d0f835907fd661d419db2-Abstract-Conference.html) | ReLU-kernel Linear DiT；block-wise cumulative $O(d^2)$ state | 训练 12 天×64 H100、分钟级/720p/5090 数字均为不同作者协议 |
| 2026-07-23 preprint/部分开放 | [SANA-Video 2.0](https://arxiv.org/abs/2607.21553) / [official docs](https://nvlabs.github.io/Sana/docs/sana_video2/) | 75% gated linear + 25% softmax anchor；AttnRes | 严格渐近仍含 $O(N^2)$；快照日只核验到 5B checkpoint，14B coming soon |
| CVPR 2026 | [LinVideo](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_LinVideo_A_Post-Training_Framework_towards_On_Attention_in_Efficient_Video_CVPR_2026_paper.html) | selective post-training conversion to Hedgehog linear attention；ADM | 单 H100/batch1/50-step：1.43–1.71×；4-step 15.9–20.9×，不能拆掉 NFE 条件 |
| CVPR 2026 | [RAPID](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.pdf) | 一次 dense block-score 后缓存 mask/score，跨 step 调密度 | 单 A100 Turbo：Wan 1.79×、Hunyuan 2.01×；两模型 prompt protocol 不同 |
| CVPR 2026 | [TimeRipples](https://openaccess.thecvf.com/content/CVPR2026/html/Mao_TimeRipples_Accelerating_vDiTs_by_Understanding_the_Spatio-Temporal_Correlations_in_Latent_CVPR_2026_paper.html) | 同一 attention 内按 $Q/K$ 时空局部相似复用部分 score | 论文说 FA 不支持该非结构稀疏；2.31–2.66× 含比例估算，不是部署 kernel 实测 |
| ICLR 2026 | [DSA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c3728248f3c627d1f16ca5726cdf83f5-Abstract-Conference.html) | distributed sparse attention + scheduling | 8 GPU：vs distributed 1.43×、vs single GPU 10.79×，不可与单卡算法横排 |
| ICLR 2026 | [BLADE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5bcb807ae43ad0851a6ba6162a866404-Abstract-Conference.html) | data-free joint adaptive block sparsity + step distillation | 8.89–14.10× 同时改变 sparsity 与 NFE，不能当 attention-only 加速 |
| ICLR 2026 | [VMoBA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d6c4014ff8d95025aa35d831c0f81faa-Abstract-Conference.html) | recurrent 1D/2D/3D block partition、global/threshold selection | 训练/推理 FLOPs 与 latency speedup 必须分开 |
| CVPR 2026 | [Attention Surgery](https://openaccess.thecvf.com/content/CVPR2026/html/Ghafoorian_Attention_Surgery_An_Efficient_Recipe_to_Linearize_Your_Video_Diffusion_CVPR_2026_paper.html) | cost-aware softmax/linear hybrid post-training | 不是 from-scratch universal linear architecture |
| CVPR 2026 | [ReHyAt](https://openaccess.thecvf.com/content/CVPR2026/html/Ghafoorian_ReHyAt_Recurrent_Hybrid_Attention_for_Video_Diffusion_Transformers_CVPR_2026_paper.html) | recurrent chunk-wise hybrid attention、constant-memory state | 作者训练预算/速度需随协议引用 |
| ICLR 2025 | [ViDiT-Q](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html) | image/video DiT quantization | 量化证据不能自动外推到所有 kernel/GPU |
| ICLR 2026 | [QuantSparse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) | quantization + attention sparsification 联合压缩 | HunyuanVideo-13B 的 3.68× storage、1.88× E2E 是作者设置 |

TimeRipples 的两个官方记录存在源数据差异：arXiv `2511.12035` 当前元数据写作 `Timeripple`、第一作者为 Wenxuan Miao，CVPR 2026 正式页写作 `TimeRipples`、第一作者为 Wenxuan Mao。书目 registry 保留 first-public arXiv 快照，正文优先引用正式版标题与作者；这不是本仓库自行改名或推断作者身份。
| CVPR 2026 | [DeltaQuant](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DeltaQuant_4-bit_Video_Diffusion_Models_with_Spatiotemporal_Delta_Smoothing_CVPR_2026_paper.html) | 4-bit + spatiotemporal delta smoothing | 需保留 calibration、precision 与硬件协议 |

## 7. 数值协议审计样例

| 工作 | 作者数字 | 同时发生的变量 | 教材允许的表述 |
|---|---|---|---|
| LinVideo | 50-step 1.43–1.71×；4-step 15.9–20.9× | attention conversion + 后者 DMD2/step reduction | 分别写；不能称 linear attention 单独 20.9× |
| RAPID | Turbo 1.79× / 2.01× | density schedule、模型、帧数和 prompt protocol | “单 A100 作者设置”并保留模型名 |
| DSA | 8 GPU 1.43× vs distributed、10.79× vs single | device count + sparse/distributed execution | 不能和单卡 sparse 数字横排 |
| SANA2 DiT-forward | 3.2× | 单 H100 BF16、batch1、warm-up、CUDA event；排除 text/VAE，AttnRes off | 不能改写为完整生成 3.2× |
| SANA2 Sol-Engine | B200 3.58× / H100 2.84× | sparse + cache + NFE 50→33 等集成栈 | 只称作者集成 serving recipe，不拆成可相乘模块 |
| TimeRipples | 2.31–2.66× | attention saving 比例估算；缺兼容非结构 sparse kernel | 标成估算，不称已实测 E2E deployment |
| Wan2.2 5B | 单消费 GPU，5 秒 720p 少于 9 分钟 | 具体 checkpoint、offload/precision 与官方命令 | 不能赋给 A14B；A14B 单卡命令另需至少 80GB |

## 8. 原创图记录

### 8.1 文件与完整性

- 文件：`assets/diagrams/video-dit-compute-contract.png`
- 类型：PNG，RGB，1672×941，8-bit sRGB
- 大小：1,334,186 bytes（ImageMagick 显示约 1.272 MiB）
- SHA-256：`290dea8037c5660c28cae37820bc7851127e7c706e75e56971942ccf12721697`
- 生成方式：OpenAI image generation；全新原创教学图，不要求模仿任何论文图或品牌视觉。

### 8.2 最终提示词（原文）

> Use case: scientific-educational
> Asset type: original handbook infographic for an advanced Video Generation 101 chapter
> Primary request: Create a clean, publication-quality 16:9 landscape scientific infographic titled exactly "VIDEO DiT: WHERE THE COMPUTE GOES". Explain the compute path of a video diffusion transformer without copying any paper figure.
> Scene/backdrop: bright white classroom/technical-report background, generous whitespace, crisp vector-like flat design.
> Composition/framing: four numbered panels flowing left to right, then a full-width warning strip at the bottom.
> Panel 1 exact heading: "1  TOKEN BUDGET". Show a small video/latent cube labeled exactly "T' × H' × W'" being patchified into a sequence labeled exactly "N TOKENS". Visually show that longer duration and higher resolution increase N.
> Panel 2 exact heading: "2  ATTENTION TOPOLOGIES". Show five small, clearly different connection/matrix icons with exact labels: "FULL  O(N²)", "FACTORIZED", "WINDOW / SPARSE", "LINEAR  O(N)", "HYBRID". Use distinct shapes and line patterns, not color alone. Do not imply one is always best.
> Panel 3 exact heading: "3  SCALING LEVERS". Show four stacked levers with exact labels: "TOKEN REDUCTION", "ATTENTION DESIGN", "NOISE-TIME EXPERTS", "PARALLELISM + CACHE". Make clear that noise-time experts route between experts across denoising stages, while parallelism distributes work rather than removing all work.
> Panel 4 exact heading: "4  EVIDENCE GATES". Show a checklist with exact labels: "SAME OUTPUT CONTRACT", "TOTAL FLOPs", "WALL LATENCY", "PEAK VRAM", "COMMUNICATION", "QUALITY + COVERAGE".
> Bottom warning text, verbatim and prominent: "Fewer attention FLOPs do not automatically mean faster end-to-end video generation."
> Style/medium: polished technical infographic, consistent bold sans-serif typography, dark navy text, restrained Okabe-Ito colorblind-safe blue/orange/green/purple accents, high contrast, 1-2 pt dark outlines, no gradients required.
> Scientific constraints: attention topology and scaling levers are separate; token reduction, attention sparsity/linearity, timestep-expert routing, distributed parallelism, denoising-step reduction, and caching are not interchangeable. No fabricated metrics, no model logos, no paper names, no benchmark scores.
> Accessibility constraints: readable at repository page width, direct labels, redundant shapes and line styles, grayscale interpretable.
> Avoid: tiny text, decorative circuitry, pseudo-code, excessive arrows, overlapping elements, cropped labels, extra words, watermarks, logos.

### 8.3 视觉与语义验收

2026-08-30 使用原分辨率彩色图与 ImageMagick 灰度副本分别人工检查：

- **通过：文字。** 标题、四个 panel、五种 topology、四种 scaling lever、六项 evidence gate 和底部警告均可读；未见乱码、伪词、裁切或重叠。
- **通过：结构。** 四栏从 token budget → topology → scaling lever → evidence gate；noise-time expert 图沿 denoising steps 路由，不会误读为视频帧路由。
- **通过：边界。** parallelism + cache 与 attention design 分栏；图中没有论文名、模型 Logo、benchmark 或虚构数字。
- **通过：灰度。** 边框、编号、线型、矩阵/节点形状和复选框在去色后仍可区分，语义不依赖颜色。
- **通过：尺寸。** 16:9 近似比例，正文页宽下标题与关键标签清晰。
- **保留限制。** “LINEAR O(N)”是教学层的序列渐近标签；正文已说明具体 kernel 可能是 $O(Nrd)$ 或 $O(Nd^2)$，hybrid 只要保留固定比例 dense layer，严格渐近仍含 $O(N^2)$。

灰度副本仅用于本地 QA，路径 `/tmp/video-dit-compute-contract-gray.png`，不作为仓库资产提交。

## 9. 尚未运行的实验

### 9.1 `BackboneFork-1`

从头训练、同时做 parameter-matched 与 training-FLOP-matched 的 U-Net/full/factorized/window-sparse/linear-recurrent 比较。冻结 codec、patch、数据及顺序、text encoder、objective、sampler/NFE/CFG、输出、训练 tokens、precision 和 hardware。按 $\tau$ 分桶 target error，同时测质量/覆盖、长程/绑定、网格外推和成本斜率。

### 9.2 `ServeFork-1`

固定 checkpoint 权重、prompt、seed、sampler、NFE、CFG、输出和 decode，逐项改变等价 fused kernel 或无需训练的 sparse 近似执行、inter-step cache、quantization 和 device parallelism；最后再测明确列出全部变更的集成 serving recipe。需要校准、蒸馏或后训练优化的 linearization / learned sparse router 进入独立的 `ServeFork-1b` converted-checkpoint 分叉，另报转换数据、更新步数、训练 FLOPs、参数变化和转换时间，不能记作固定 checkpoint 的纯 kernel 加速。

两者目前均为**预注册协议草案，尚未运行**。正文只把它们作为可证伪方法，不声称已有实验结果。

## 10. 本轮结论边界

1. 2024–2025 的规模化路线从 factorized/window 扩展到 full/joint、多流与非对称容量，但不存在一个经完全匹配实验确认的全局最佳 topology。
2. 2025–2026 的 frontier 是多条正交线：linear/hybrid、structured/dynamic sparse、inter-step reuse、cross-depth reuse、distributed execution、quantization 与少步；不能把论文倍数直接相乘。
3. Video DiT 的公平比较必须先冻结 representation、objective、sampler、condition 与 output contract，再分别报告算法成本、kernel 实现和多卡通信。
4. 本轮没有训练大模型、没有运行公开 checkpoint benchmark，也没有独立复现作者速度；所有定量结果均标为作者协议。
