# 视频扩散模型的底层算子、并行与在线服务综述

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本文优先采用正式会议论文、作者公开论文与官方实现说明。文中的加速、显存和吞吐数字均来自作者在特定模型、输出规格、精度、硬件和计时范围下的实验；本文尚未在统一环境中复现这些结果，因此不进行跨论文速度排名，也不将不同模块的加速比直接相乘。

## 摘要

算法层面的压缩必须经过实际执行，才会体现为可观测的速度提升。底层算子与编译优化可减少访存和启动开销；显存调度与 VAE 优化决定目标配置能否运行；并行策略影响单请求延迟和集群扩展能力；在线服务系统还需同时考虑首帧延迟、逐帧时限、吞吐和尾延迟。近年来，研究重点逐步从独立的注意力算子扩展到适配时空结构的多卡通信，以及面向特定模型、硬件和请求配置的全栈调优。

本文属于叙述性范围综述，重点讨论**执行层基础设施**，不追求穷尽全部文献。少步生成见[蒸馏综述](distillation.md)，低比特模型见[量化综述](quantization.md)，结构压缩见[剪枝综述](pruning.md)，动态连边见[Token 与注意力稀疏化综述](sparsity.md)，跨步复用见[缓存综述](caching.md)。

## 1. 研究边界：系统优化的对象与指标

离线生成的一次端到端延迟可拆为：

```math
L_{\mathrm{e2e}}
=L_{\mathrm{text}}
+\sum_{i=1}^{N_{\mathrm{NFE}}}
\left(L_{\mathrm{denoiser},i}+L_{\mathrm{guidance},i}\right)
+L_{\mathrm{VAE}}
+L_{\mathrm{post}}
+L_{\mathrm{I/O}}.
```

系统优化主要改变各环节在硬件上的执行方式，但未必减少数学操作数。评测时应区分以下四个层级：

| 层级 | 典型指标 | 可以说明 | 尚不足以说明 |
|---|---|---|---|
| 算子 | kernel time、OPS、带宽、占用率 | 某个 attention/GEMM/conv 更快 | 整个 denoiser 或视频生成同倍加速 |
| 模块 | DiT forward、VAE decode、单步延迟 | 一个主要模块的真实收益 | text、VAE、I/O 后仍有同一收益 |
| 流水线 | end-to-end latency、peak VRAM | 固定输出下整条路径的结果 | 多用户吞吐或尾延迟 |
| 服务 | TTFF、p50/p95/p99、deadline miss、FPS/QPS | 在线体验和容量 | 单样本质量没有变化 |

峰值显存至少包含权重、激活、attention/KV、cache、workspace 与 runtime。`OOM → 可运行` 是重要的**容量收益（capacity gain）**，但不能直接折算为 latency speedup。

## 2. 研究进展：从单算子到完整服务

### 2.1 2024：单卡内存优化与 DiT 并行

Streamlined Inference 将 feature slicing、连续算子分组和跨去噪时间步的 Step Rehash 结合起来，使 AnimateDiff 在作者给定的实验设置中由 42 GB 峰值显存降至 11 GB，并能在 2080 Ti 上运行 [[1]](#ref-1)。同年提出的 xDiT 将 sequence parallel、patch-level PipeFusion 与 CFG parallel 组织为可混合的并行引擎，系统分析了 DiT 在 NVLink 与以太网集群上的可扩展性 [[2]](#ref-2)。其论文中的视频生成实验仅使用了部分并行配置，不能将所有视频结果都归因于三种并行方式同时启用。前者侧重单卡容量，后者侧重多卡执行，二者所报告的速度收益对应不同变量。

### 2.2 2025：kernel、VAE 与跨机通信分化

SageAttention2 通过 per-thread INT4 的 $Q,K$、FP8 的概率矩阵/$V$ 路径和分级累加，将低比特设计实现为 attention kernel；论文报告的是相对 FlashAttention2、xFormers 的算子 OPS，而非视频流水线的同倍加速 [[3]](#ref-3)。SpargeAttention 在此基础上加入两级在线过滤，将 training-free 稀疏选择、低比特计算与 kernel 执行结合起来 [[4]](#ref-4)。

同一时期，ScaleFusion 针对时空 attention 的跨机通信，利用层内与层间调度隐藏传输开销；其 32 张 A100 相对 8 张 A100 的 3.60× strong scaling，以及相对既有方案的平均 1.36× 加速采用了不同基线 [[5]](#ref-5)。LeanVAE 将研究对象从 denoiser 扩展至 latent 编解码：该方法以轻量 NAF 模块、非重叠 patch、wavelet 与 compressed sensing 重构 video VAE，作者报告其相对所选 VAE 基线最多减少 50× FLOPs，并获得 8–44× 的模块推理加速 [[6]](#ref-6)。这些结果表明，随着 DiT 加速，VAE 在端到端耗时中的占比可能上升。

### 2.3 2026：SLO、动态负载与具体配置的全栈优化

StreamDiffusionV2 不再仅优化整段视频的完成时间，而是显式约束 time-to-first-frame、逐帧 deadline 与 jitter，并结合 SLO-aware batching、rolling KV、step/layer pipeline 和异构 GPU 编排 [[7]](#ref-7)。db-SP 处理稀疏 attention 与 sequence parallel 组合后的 head/block 双重负载不均，并分别报告 attention-specific speedup 与 end-to-end speedup [[8]](#ref-8)。

FlashDecoder 将 latent-to-pixel 阶段改造为逐帧 Transformer decoder，表明实时系统的关键路径并不限于 denoiser [[9]](#ref-9)。SANA-Video 将 linear attention、常量内存状态与 NVFP4 部署纳入同一架构 [[10]](#ref-10)。TurboDiffusion 和 Sol 则将蒸馏、低比特、稀疏、token pruning、cache 与 kernel fusion 组合为全栈方案 [[11]](#ref-11) [[12]](#ref-12)。截至 2026-09-02，二者均为作者预印本，可作为全栈组合优化的案例，但不能替代统一基准或独立复现。

## 3. 底层算子与编译：从理论 FLOPs 到实际耗时

底层算子优化不改变模型语义，主要通过减少 HBM 数据往返、提高 Tensor Core 利用率、融合中间张量和降低算子启动开销来缩短运行时间。常见路径包括融合注意力、低精度 GEMM、归一化与激活融合、因果三维卷积优化、计算图捕获，以及针对特定张量形状的编译。

评测中常见的三类混淆如下：

1. **模拟量化/稀疏**只证明数值质量，不证明硬件加速；
2. **kernel benchmark**只测一个算子，不包括 MLP、VAE、通信和 I/O；
3. **end-to-end benchmark**可能同时改变 step、precision、checkpoint 与输出，因此不能全归因于 kernel。

SageAttention2 采用“数值格式—线程粒度—累加策略”的协同设计；SpargeAttention 则采用“selector—稀疏 layout—低比特 kernel”的协同设计 [[3]](#ref-3) [[4]](#ref-4)。如果目标硬件不支持相应的 INT4/FP8 路径，或输入 shape 触发 fallback，其性能收益未必能够保持。

一个可复核的 kernel 报告至少应写清：GPU/加速器、驱动与编译器版本、dtype、layout、输入 shape、warm-up、重复次数、是否含 selector/pack/dequant、累加精度，以及 kernel 与端到端两个层级的 latency。

## 4. 显存、内存卸载与 VAE：区分容量收益与时延收益

降低 peak VRAM 的方法包括 feature slicing、attention/latent chunking、operator grouping、CPU offload、流水化 decode 与 tiled VAE。这些方法的容量收益通常以更多切片、同步或数据搬运为代价，对时延的影响可分为三类：

- 显存下降、速度近似不变；
- 显存下降但 PCIe/统一内存搬运令速度变慢；
- 原 baseline OOM，新方法首次使目标规格可运行。

Streamlined Inference 通过 feature slicing 与 operator grouping 改变激活生命周期，但其 Step Rehash 同时近似跳过计算，因此整体速度不能只归因于内存调度 [[1]](#ref-1)。LeanVAE 与 FlashDecoder 则修改解码器结构，需要新 checkpoint 或专门训练；它们不属于对原 VAE 的无损 kernel 替换 [[6]](#ref-6) [[9]](#ref-9)。

VAE/内存实验应报告 `peak allocated`、`peak reserved`、host memory、传输字节、decode latency、首帧时间、重建 PSNR/SSIM/LPIPS/时序指标，以及是否包含 encode。仅报告“显存降低 X×”，无法判断该收益是否以延迟或重建质量为代价。

## 5. 多 GPU 并行：通信轴与视频结构的匹配

| 并行轴 | 切分对象 | 主要收益 | 主要瓶颈 |
|---|---|---|---|
| Tensor parallel | channel/head/矩阵 | 分摊大层权重与 GEMM | 高频 collective、细粒度同步 |
| Sequence/context parallel | 时空 token | 直接容纳长序列 | all-to-all/ring 通信、负载不均 |
| Pipeline / PipeFusion | layer、patch 或 denoising stage | 重叠计算与通信 | bubble、阶段不平衡、状态管理 |
| CFG parallel | 条件/无条件分支 | 两个独立前向并发 | guidance-distilled 模型可能没有该收益 |
| Request/data parallel | 请求或样本 | 提升吞吐 | 不直接降低单请求 latency |

xDiT 的主要贡献在于混合这些并行轴，而非认定某一种并行方式对所有张量形状均最优 [[2]](#ref-2)。ScaleFusion 进一步利用时空注意力的结构隐藏跨机通信，表明长视频并行需要适配网络拓扑与注意力数据布局 [[5]](#ref-5)。db-SP 表明，即使算法稀疏率相同，若密集计算块在设备间分布不均，实际运行时间仍由最慢的设备决定 [[8]](#ref-8)。

多卡结果应同时报告单卡或最小可运行基线、GPU 数、互连方式、批量大小、总 FLOPs、计算/通信时间、强扩展效率、峰值显存、p50/p95 时延与吞吐。`32 GPU 对 8 GPU 快 3.6×` 表示扩展效率，不能表述为算法复杂度降低 3.6×。

## 6. 在线服务：离线吞吐与交互式实时性

离线系统关心整段完成时间与批吞吐；交互流式系统至少还要满足：

```text
TTFF / inter-frame latency / jitter
p50, p95, p99 latency / deadline-miss rate
concurrent streams / admission control / backpressure
rolling state bytes / reset and recovery behavior
quality under 1, 2, 4... denoising steps
```

StreamDiffusionV2 的主要贡献是将这些 SLO 与模型流水化统一考虑。作者在四张 H100、BF16、未使用 TensorRT 或量化的实验设置下，分别为 14B 与 1.3B 模型报告 58.28 与 64.52 FPS 的单步结果；另在单张 H100、$512^2$、单步在线 V2V 的 SLO 表中报告 0.37–0.47 秒 TTFF、585 ms p99、0.2% 的 1 秒时限超时率与 21 ms 平均抖动 [[7]](#ref-7)。多卡 FPS 与单卡 SLO 属于两组不同实验，不应据此推断任意 14B 模型均可实现实时生成。平均 FPS 也不足以反映首帧等待、周期性抖动或 p99 超时情况。

在线服务还会改变离线场景下的最优配置：dynamic batching 可以提高吞吐，但会增加等待；rolling cache 可减少重复计算，但会引入 eviction/reset；异构 GPU 能提高资源利用率，同时也增加调度复杂度。因此，系统评测应同时给出质量、TTFF、尾延迟、吞吐和成本的 Pareto 前沿，而非仅报告平均 FPS。

## 7. 全栈组合：优化顺序与瓶颈转移

TurboDiffusion 在作者给定的实验设置中组合 low-bit attention、trainable sparse-linear attention、rCM step distillation、W8A8 与其他工程优化，并报告 100–200× 端到端加速 [[11]](#ref-11)。该结果反映的是完整配置相对于指定 baseline 的组合收益，不能据此认定每个模块均独立贡献相同倍数。

Sol 将缓存、稀疏注意力、token 剪枝、量化和算子融合组织为面向模型、硬件与服务配置的搜索空间；其三个模型实例均报告超过 2× 的端到端收益 [[12]](#ref-12)。这些结果提示，较优的加速配置可能随具体模型和硬件而变化，因而可以通过自动化搜索选择方法组合。不过，目前证据仍来自作者选定的模型和配置，其通用性尚需跨实验室复现。

SANA-Video 采用了另一种全栈设计：在训练阶段引入 Linear DiT、block-wise 自回归和 constant-memory state，并进一步采用 NVFP4 部署 [[10]](#ref-10)。由于模型架构、训练过程和执行层均发生变化，该方法不能与仅替换既有 checkpoint 之 kernel 的方法直接比较。

## 8. 代表工作比较

| 工作 | 层级 | 是否改权重/架构 | 原文实验中的主要证据 | 解释边界 |
|---|---|---:|---|---|
| Streamlined Inference [[1]](#ref-1) | 单卡内存 + 跨步复用 | 否 | AnimateDiff 42→11 GB；2080 Ti 可运行 | 容量与近似跳算的组合 |
| xDiT [[2]](#ref-2) | 混合多 GPU 并行 | 否 | 多种 DiT、NVLink/以太网扩展 | 并行收益依赖拓扑和 shape |
| SageAttention2 [[3]](#ref-3) | attention kernel | 否 | 对 FA2/xFormers 的 OPS 对比 | 算子倍数不等于 E2E 倍数 |
| SpargeAttention [[4]](#ref-4) | 稀疏 + 低比特 kernel | 否 | 多类模型端到端指标近似保持 | selector 和 kernel 需一并计时 |
| ScaleFusion [[5]](#ref-5) | 跨机 ST-DiT 并行 | 否 | 8→32 A100 strong scaling 3.60× | 不是减少总 FLOPs |
| LeanVAE [[6]](#ref-6) | video VAE | 是 | 最高 50× FLOPs、8–44×模块速度 | 需同时验收重建与生成质量 |
| StreamDiffusionV2 [[7]](#ref-7) | 在线服务 | 否 | 四 H100 下 FPS；单 H100 下 TTFF/SLO | 结果仅适用于原文的模型和评测设置 |
| db-SP [[8]](#ref-8) | 稀疏 sequence parallel | 否 | 平均 E2E 1.25×、attention 1.40× | 两种层级不能混报 |
| FlashDecoder [[9]](#ref-9) | 流式 VAE decoder | 是 | 原文报告最高 12× decoder 加速 | 不等于整条生成 12× |
| SANA-Video [[10]](#ref-10) | 架构 + 状态 + 低精度 | 是 | RTX 5090、长视频与 NVFP4 结果 | 是共同设计，不是单项消融 |
| TurboDiffusion [[11]](#ref-11) | 全栈组合 | 是 | 作者报告 100–200× E2E | 预印本中的组合结果，不能拆分相乘 |
| Sol [[12]](#ref-12) | 自动化全栈优化 | 可选 | 三实例 >2× E2E | 通用性仍待独立验证 |

## 9. 共识、争议与证据边界

### 9.1 较为一致的观察

- 注意力计算加速后，MLP、VAE、通信或启动开销在总耗时中的占比可能上升，因此需要持续进行性能分析。
- 稀疏、低比特和并行策略只有匹配底层算子、数据布局与硬件，才会缩短实际运行时间。
- 多卡并行主要改变 latency/throughput 与容量，通常不减少总计算量。
- 在线视频系统的核心指标包括 TTFF、deadline 和 jitter，平均 FPS 只是其中一项。
- 全栈（full-stack）加速比应来自最终组合的实测结果，不能由各模块加速比相乘得到。

### 9.2 尚待解决的问题

- 同一 kernel 在不同 GPU 代际、序列长度和 head dimension 上能否稳定占优？
- 动态稀疏/动态 cache 的 selector 与调度成本在短视频、小 batch 下是否值得？
- VAE 改造造成的轻微重建误差，是否会在下游人评、文字和高速运动中被放大？
- 多卡低延迟能否在真实多租户流量下保持 p99，而不是只在单请求 benchmark 中成立？
- 自动化全栈搜索能否跨模型、硬件和软件版本迁移，而无需重新进行大规模调参？

## 10. 公平评测规范

### 10.1 算子级评测

固定输入 shape、dtype、layout、硬件、软件版本、warm-up 和重复次数。分别报告 kernel latency、有效带宽/OPS、selector/pack/dequant 成本、数值误差与 fallback 比例。

### 10.2 基础设施模块评测

固定 checkpoint、prompt、seed、sampler/NFE、CFG、输出分辨率/帧数/FPS、VAE、batch 与计时边界。一次只改变 kernel、编译、offload 或并行策略；报告 denoiser forward、VAE、通信和端到端分解。

### 10.3 在线服务评测

固定请求到达分布、并发、SLO 与硬件预算，报告 cold/warm TTFF、p50/p95/p99、deadline miss、吞吐、峰值显存、GPU-hours、energy/video、失败恢复与质量退化。

### 10.4 全栈归因评测

按 `基线 → +少步 → +量化/小模型 → +稀疏/缓存 → +底层算子/并行 → +在线服务` 逐级评测。每一级均应保留模型和产物哈希、实际 NFE/稠密度/位宽、模型与输出设置、硬件、延迟分解、质量/运动/多样性，以及异常场景分类。最终实测结果用于评估全栈组合，中间结果用于分析各模块的贡献。

## 11. 研究空白与未来方向

1. **跨硬件可移植 benchmark。** 需要同时覆盖消费 GPU、数据中心 GPU、边缘 NPU，以及不同数值格式和内存层级。
2. **VAE 的独立系统评测。** 在少步 DiT 场景中，应联合优化 decoder、postprocess 与输出编码，并单列 latent-to-first-frame。
3. **稀疏—并行共同调度。** 动态 pattern、head/block 负载、通信拓扑和 kernel tile 应在同一个优化器内决定。
4. **多租户视频 SLO。** 需要公开到达轨迹、突发流量、抢占、cache 隔离和故障恢复，而非只有理想稳定流。
5. **能耗和总成本。** 延迟之外还应报告 energy/video、峰值功率、热降频、GPU-hours 和跨区域通信成本。
6. **可验证的自动调优。** 面向具体配置的搜索应保留配置文件、编译产物、质量阈值和回退条件，以支持结果复核。
7. **统一的异常场景分类。** scene cut、快运动、小物体、文字、局部控制、长时漂移和异质 batch 应纳入基础设施回归集。

## 12. 建议阅读路线

1. 先读 Streamlined Inference 与 xDiT，分别考察单卡容量优化和混合并行 [[1]](#ref-1) [[2]](#ref-2)。
2. 再读 SageAttention2 与 SpargeAttention，理解数值、稀疏 pattern 与 kernel 协同设计的必要性 [[3]](#ref-3) [[4]](#ref-4)。
3. 用 ScaleFusion 与 db-SP 理解通信隐藏、strong scaling 和稀疏负载均衡 [[5]](#ref-5) [[8]](#ref-8)。
4. 用 LeanVAE、FlashDecoder 与 SANA-Video 补齐 decoder、长视频状态和架构级效率 [[6]](#ref-6) [[9]](#ref-9) [[10]](#ref-10)。
5. 最后读 StreamDiffusionV2、TurboDiffusion 与 Sol，将 SLO、组合归因和针对具体配置的调优纳入同一部署分析框架 [[7]](#ref-7) [[11]](#ref-11) [[12]](#ref-12)。

检索范围、证据等级和重构记录见[研究日志](../../../sources/research_20260902_video_diffusion_infrastructure.md)。

## 参考文献

<a id="ref-1"></a>[1] [Fast and Memory-Efficient Video Diffusion Using Streamlined Inference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/18b0b4c788c8f2cf6c2943b989ad18c8-Abstract.html). Zheng Zhan et al. NeurIPS. 2024.

<a id="ref-2"></a>[2] [xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism](https://arxiv.org/abs/2411.01738). Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, Jiannan Wang. Author paper and official implementation. 2024.

<a id="ref-3"></a>[3] [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://proceedings.mlr.press/v267/zhang25ae.html). Jintao Zhang et al. ICML. 2025.

<a id="ref-4"></a>[4] [SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference](https://proceedings.mlr.press/v267/zhang25ch.html). Jintao Zhang et al. ICML. 2025.

<a id="ref-5"></a>[5] [ScaleFusion: Scalable Inference of Spatial-Temporal Diffusion Transformers for High-Resolution Long Video Generation](https://proceedings.mlsys.org/paper_files/paper/2025/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html). Jiacheng Yang et al. MLSys. 2025.

<a id="ref-6"></a>[6] [LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_LeanVAE_An_Ultra-Efficient_Reconstruction_VAE_for_Video_Diffusion_Models_ICCV_2025_paper.html). Yu Cheng, Fajie Yuan. ICCV. 2025.

<a id="ref-7"></a>[7] [StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html). Tianrui Feng et al. MLSys. 2026.

<a id="ref-8"></a>[8] [db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism](https://proceedings.mlsys.org/paper_files/paper/2026/hash/db988b089d8d97d0f159c15ed0be6a71-Abstract-Conference.html). Siqi Chen et al. MLSys. 2026.

<a id="ref-9"></a>[9] [FlashDecoder: Real-Time Latent-to-Pixel Streaming Decoder with Transformers](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_FlashDecoder_Real-Time_Latent-to-Pixel_Streaming_Decoder_with_Transformers_CVPR_2026_paper.html). Minguk Kang, Suha Kwak. CVPR. 2026.

<a id="ref-10"></a>[10] [SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41b93c59da0d0f835907fd661d419db2-Abstract-Conference.html). Junsong Chen et al. ICLR. 2026.

<a id="ref-11"></a>[11] [TurboDiffusion: Accelerating Video Diffusion Models by 100–200 Times](https://arxiv.org/abs/2512.16093). Jintao Zhang et al. Author preprint and official implementation. 2025.

<a id="ref-12"></a>[12] [Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation](https://arxiv.org/abs/2606.23743). Yitong Li et al. Author preprint. 2026.
