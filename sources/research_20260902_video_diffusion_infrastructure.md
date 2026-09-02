# 视频扩散模型压缩与推理基础设施研究记录（2026-09-02）

> 对应正文已重构为一页总览与六篇独立专题综述。本记录说明研究问题、检索与筛选方法、一手来源、原论文实验数据、尚未完成的复现工作及建议实验。它属于问题驱动的**叙述性范围综述**，并未采用预注册协议、穷尽检索和逐篇排除流程，也不构成独立的基准测试。原论文报告的加速比仅作为文献结果引用，不视为本仓库的复现结论。

对应的七个文件为：

1. [压缩与推理加速综述导航](../docs/generative-models/inference-acceleration.md)
2. [少步与蒸馏综述](../docs/generative-models/inference-acceleration/distillation.md)
3. [量化综述](../docs/generative-models/inference-acceleration/quantization.md)
4. [模型剪枝综述](../docs/generative-models/inference-acceleration/pruning.md)
5. [Token 与注意力稀疏化综述](../docs/generative-models/inference-acceleration/sparsity.md)
6. [缓存与特征复用综述](../docs/generative-models/inference-acceleration/caching.md)
7. [底层算子、并行与在线服务综述](../docs/generative-models/inference-acceleration/systems.md)

## 1. 为什么增设推理基础设施总览与专题综述

仓库已有三块相关内容：

- Diffusion 章已分清 solver、consistency/trajectory distillation 和 DMD；
- Video DiT 章已覆盖 token 预算、attention topology、稀疏、并行、cache 与量化；
- 因果流式章已覆盖 data-time KV cache、TTFF、deadline 和 SLO。

但这些内容分散在不同机制章节中，尚缺少统一的推理成本分析，无法集中回答 `NFE 多少 / 单次前向开销多大 / 显存是否足够 / kernel 是否支持 / 多卡是否有效 / 质量损失发生在哪里`。其中，参数、channel、head、block 和 layer 剪枝此前也没有形成完整的专题脉络。

本次重构以总览页说明各环节的成本构成及其相互关系，并以六篇独立综述分别讨论蒸馏、量化、模型剪枝、token/attention 稀疏化、cache，以及 kernel、显存、并行和 serving。原有机制章节继续保留更完整的数学与架构背景。拆分后，每篇综述均可独立说明研究范围、方法分类与演进、代表性工作、证据争议、复现要求和未来方向；总览页则用于比较不同路线并提供阅读索引。

## 2. 预先确定的研究问题

1. 视频扩散的端到端延迟和峰值显存应如何分项分析？
2. nominal step、solver evaluation、CFG forward 和真实 NFE 怎样区分？
3. sampler、step distillation、size distillation 和 causal student 分别改变什么？
4. 非结构权重、结构化 channel/layer/block、token 与 attention edge 剪枝是否需要分类？
5. weight、activation、attention 与 KV/cache 量化的硬件与质量证据是否相同？
6. inter-step feature cache、data-time KV cache 和 cross-layer reuse 分别跨哪个时间轴？
7. attention-kernel、DiT-forward、DiT-only、end-to-end、单卡和多卡倍数怎样避免混报？
8. 多项优化同时启用时，怎样通过逐级消融做归因？
9. 需要哪些反例来推翻“FLOPs 少即更快”“低 bit 即无损”“平均 FPS 高即实时”？

## 3. 检索日期、检索来源与证据等级

专项文献核查于 **2026-09-02（Asia/Shanghai）** 完成。搜索引擎仅用于发现候选论文，标题、发表状态、方法和实验信息均以 CVF、ICLR、NeurIPS、PMLR、AAAI、MLSys 等正式页面，以及作者 arXiv 论文或官方实现为准。本次检索旨在覆盖各技术路线的代表性工作及其证据边界，不宣称穷尽全部论文；正文因此按照叙述性/范围综述理解。

| 检索来源 | 代表查询 | 目的 |
|---|---|---|
| CVF / ICLR / NeurIPS / PMLR / AAAI / MLSys | `video diffusion distillation quantization pruning cache sparse attention inference system`；论文全名 | 核对标题、发表场所、方法与实验设置 |
| arXiv 作者稿 | `video diffusion model pruning`、`video DiT W4A8`、`few-step causal video diffusion` | 补充正式摘要未展开的方法、限制和预印本前沿 |
| 官方仓库/项目页 | 论文标题 + `official code`、checkpoint 与运行配置 | 确认 kernel、低比特或 cache 是否有真实实现 |
| 反例定向 | `memory compression latency overhead`、`attention speedup end-to-end`、`mobile video pruning ablation` | 主动寻找省显存但不降时延、kernel 快但 E2E 收益小等结果 |

| 等级 | 定义 | 可以支撑 | 不可以支撑 |
|---|---|---|---|
| A | 正式 proceedings/期刊 | 正式状态、公开机制、作者实验 | 跨模型/硬件通用排名、独立复现 |
| B | 作者 arXiv/技术报告 | 作者公开的方法、消融、限制 | 同行评审状态、无条件泛化 |
| C | 作者/机构官方仓库与文档 | 发布面、kernel 支持、checkpoint/配置 | README 宣传成为独立性能结论 |
| X | 二手新闻、聚合页、搜索摘要 | 发现候选 | 最终教材事实 |

## 4. 成本模型与操作定义

正文的主分解为：

```math
L_{\mathrm{e2e}}
=L_{\mathrm{text}}
+\sum_{i=1}^{N_{\mathrm{NFE}}}
\left(L_{\mathrm{denoiser},i}+L_{\mathrm{guidance},i}\right)
+L_{\mathrm{VAE}}+L_{\mathrm{post}}+L_{\mathrm{I/O}}.
```

所有方法先按它改变的项归类：

| 方法 | 主要改变 | 仍需另行验证 |
|---|---|---|
| sampler / solver | NFE/时间节点 | 模型参数或 resident weights 变小 |
| step/guidance distillation | NFE、guidance forward、学生分布 | 单次 DiT forward 更快 |
| size/structured pruning | layer/channel/head/block/宽度 | 若无兼容 kernel，非结构稀疏不保证 wall time |
| token pruning/merging | 序列长度、activation | 小物体、快运动或局部控制无损 |
| sparse/linear attention | attention edge/复杂度 | MLP、VAE、selector、通信成本同比例下降 |
| quantization | weight/activation/cache bytes 与部分算子 | 硬件没有低精度 kernel 时仍会更快 |
| inter-step cache | 去噪轴上的重复计算 | 严格数值等价 |
| causal KV cache | 视频时间上已提交历史的重算 | 普通双向 Video DiT 可直接使用 |
| parallelism / serving | 单请求延迟、吞吐、通信与 SLO | 算法总 FLOPs 降低 |

## 5. 一手文献矩阵

### 5.1 少步蒸馏

| 路线 | 代表工作 | 纳入理由 | 主要局限 |
|---|---|---|---|
| training-free solver | [DPM-Solver](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) | 不改权重，先建立同 checkpoint 的 NFE–质量曲线 | 极少步积分误差；不降低每次前向成本 |
| 逐级轨迹压缩 | [Progressive Distillation](https://openreview.net/forum?id=TIdIXIpzhoI) | 每轮让学生模型以 $N/2$ 步逼近教师模型的 $N$ 步采样 | 多轮串行蒸馏，视频高维轨迹更难拟合 |
| consistency | [Consistency Models](https://proceedings.mlr.press/v202/song23a.html)、[MCM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c859b99b5d717c9035e79d43dfd69435-Abstract-Conference.html)、[rCM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html) | 从通用轨迹一致扩展到运动–外观拆分和大规模视频 teacher | 细节、运动、多样性与低 NFE 同时受压 |
| 分布匹配 | [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html)、[DMD2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | 匹配 student/target 分布，不要求逐样本复制 teacher 轨迹 | 反向 KL 的 mode-seeking、颜色/饱和偏移、多样性下降 |
| 视频对抗/奖励修复 | [APT](https://proceedings.mlr.press/v267/lin25m.html)、[DOLLAR](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.html) | 用 discriminator/reward 补少步的感知与语义退化 | 训练不稳定、reward bias、训练成本不可忽略 |
| 因果学生 | [CausVid](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) | 同时改变 NFE、视频时间 factorization 与 KV cache | 整体加速不能全归因于蒸馏；长时漂移和多样性仍需检验 |
| 分层 transition | [Transition Matching Distillation](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_Transition_Matching_Distillation_for_Fast_Video_Generation_CVPR_2026_paper.html) | 重 backbone 负责外层 transition，轻 flow head 负责内层更新 | effective NFE 是层参与折算，不是严格完整前向次数 |

正文组织方式：不再把“少步”作为单一类别。先区分免训练求解器与需要新模型权重的蒸馏方法，再将后者分为轨迹/一致性、分布匹配、对抗/奖励和因果/结构化学生模型。

### 5.2 模型与结构剪枝

| 工作 | 剪什么 | 为什么纳入 | 证据边界 |
|---|---|---|---|
| [F³-Pruning](https://doi.org/10.1609/aaai.v38i5.28300) | 时间 attention/block 冗余 | 早期 training-free 视频 diffusion 剪枝锚点 | 方法所用 U-Net/时间模块不能无消融外推到大型 full-attention DiT |
| [ICMD](https://arxiv.org/abs/2411.18375) | 按深浅层内容/运动敏感性剪 block | 强调剪枝需要运动保真的恢复目标 | 作者预印本；需再训练且骨干特定 |
| [Mobile Video Diffusion](https://openaccess.thecvf.com/content/ICCV2025/html/Yahia_Mobile_Video_Diffusion_ICCV_2025_paper.html) | channel funnel、temporal block、时间多尺度 | 真实移动端 full-stack 案例，有逐项消融 | 1.7 s 同时受低分辨率、结构、剪枝、单步和手机 kernel 影响 |
| [V.I.P.](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_V.I.P.__Iterative_Online_Preference_Distillation_for_Efficient_Video_Diffusion_ICCV_2025_paper.html) | pruned student + preference distillation | 表明剪枝后不必盲目复制 teacher 的所有维度 | 偏好模型会带来新的评估偏置 |
| [FastLightGen](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_FastLightGen_Fast_and_Light_Video_Generation_with_Fewer_Steps_and_CVPR_2026_paper.html) | layer/parameter pruning + few-step distillation | 明确展示模型规模与采样步数的联合优化 | 论文采用的 4-step + 30% pruning 是一项组合设置，不能拆成两个独立倍数 |

剪枝必须进一步分为：非结构权重、结构化 layer/channel/head/block、运行时 token、attention edge。只有前两类直接改变模型参数/结构；后两类主要改变特定输入上的序列或连边。

### 5.3 量化

| 工作 | 主要对象 | 原论文中的直接证据 | 必须保留的限制 |
|---|---|---|---|
| [QVD](https://doi.org/10.1145/3664647.3681050) | video diffusion PTQ；时间 feature skew 与 channel range | ACM Multimedia 2024；W8A8 质量证据 | 端到端算子和时延证据不足，尚不能确认实际加速 |
| [ViDiT-Q](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html) | timestep-aware W8A8/W4A8 DiT PTQ + GPU kernel | 2–2.5× 内存、1.4–1.7× E2E latency | 仅代表其模型、GPU、kernel 和 baseline |
| [Q-VDiT](https://proceedings.mlr.press/v267/feng25q.html) | token-aware estimator + temporal maintenance distillation | 极低 bit 下显式保持时间关系 | 蒸馏已改变 checkpoint，不是纯 PTQ 归因 |
| [QuantSparse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) | model quantization + attention sparsification | HunyuanVideo-13B：3.68× storage、1.88× E2E | 量化和稀疏误差/收益已耦合，不与 ViDiT-Q 直接排名 |
| [DeltaQuant](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DeltaQuant_4-bit_Video_Diffusion_Models_with_Spatiotemporal_Delta_Smoothing_CVPR_2026_paper.html) | core/delta spatiotemporal cube + 专用 kernel | 把视频局部冗余纳入 4-bit 执行 | 受 cube、timestep、长度和 kernel 支持影响 |
| [Quant VideoGen](https://arxiv.org/abs/2602.02958) | AR/streaming 视频模型的 2-bit KV cache | KV 约 6–7× 压缩 | 作者表中 E2E 约增 1.5%–4.3%；只适用增长 KV 历史 |

这一结果说明，下列四个目标需要分别考察：模型文件大小、峰值显存、单算子速度和端到端时延。

### 5.4 Token、attention 稀疏与 cache

| 工作 | 重用/删除对象 | 纳入目的 | 限制 |
|---|---|---|---|
| [ADAPTOR](https://openaccess.thecvf.com/content/CVPR2025W/EDGE/papers/Peruzzo_ADAPTOR_Adaptive_Token_Reduction_for_Video_Diffusion_Transformers_CVPRW_2025_paper.pdf) | 按局部运动减少 temporal token | 代表 token 级动态剪枝 | selector/gather/scatter 计时；小物体/快运动/控制对齐 |
| [Sparse Video-Gen](https://proceedings.mlr.press/v267/xi25c.html) | spatial/temporal head 的 block sparsity | 将 pattern 与可执行 layout/kernel 一起设计 | 聚类和重排有成本；稀疏率不是 E2E speedup |
| [RAPID](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.html) | 跨 denoising step 复用 block sparsity | 表明 sparsity pattern 本身也可 cache | 重用后仍要随 step 适配 density |
| [BLADE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5bcb807ae43ad0851a6ba6162a866404-Abstract-Conference.html) | block sparsity + step distillation | 组合压缩代表 | 作者倍数同时改变 attention 和 NFE |
| [PAB](https://proceedings.iclr.cc/paper_files/paper/2025/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html) | 跨步广播 attention output | 不同 attention block 用不同刷新规则 | 最高 10.5× 仅适用于原论文的特定实验设置；该方法属于近似缓存 |
| [TeaCache](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) | timestep-modulated input 指示 residual 刷新 | training-free 阈值/质量曲线代表 | 最高 4.41× 绑定 Open-Sora-Plan 作者设置 |
| [FasterCache](https://proceedings.iclr.cc/paper_files/paper/2025/hash/518046d86bbc41a0707727c38301ad8e-Abstract-Conference.html) | 相邻步 feature + 同步 CFG 分支 | 区分两种冗余来源 | CFG-cache 只适用相应 guidance 执行结构 |
| [AdaCache](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html) | 内容/运动自适应 block cache | 把 motion 纳入刷新规则 | 最高 4.7× 来自原论文的实验设置；还需分别测试镜头切换和快速运动场景 |
| [MagCache](https://proceedings.neurips.cc/paper_files/paper/2025/hash/311207bb626e36a8f1d3eb92aa67af22-Abstract-Conference.html) | residual 幅值比驱动刷新 | 一个 prompt 标定和异常步保护 | 论文报告的 2.10–2.68× 仍受模型、阈值、硬件和计时方式限制 |

### 5.5 底层算子、显存、并行与在线服务

| 工作 | 系统层机制 | 需要避免的误读 |
|---|---|---|
| [SageAttention2](https://proceedings.mlr.press/v267/zhang25ae.html) | INT4 $QK^\top$、FP8/FP16 $PV$ 与低精度 attention kernel | kernel OPS 不是端到端 video speedup |
| [SpargeAttention](https://proceedings.mlr.press/v267/zhang25ch.html) | 两级在线过滤 + 稀疏/量化 kernel | 稀疏必须包含 selector 和可执行 layout |
| [Streamlined Inference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/18b0b4c788c8f2cf6c2943b989ad18c8-Abstract.html) | feature slicing、operator grouping、Step Rehash | 显存调度与跳步复用的组合收益不归因给 offload 单项 |
| [xDiT](https://arxiv.org/abs/2411.01738) | sequence、PipeFusion、CFG parallel 组合 | 多卡降低单请求时间，不意味总 FLOPs 降为 $1/n$ |
| [ScaleFusion](https://proceedings.mlsys.org/paper_files/paper/2025/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html) | 时空 attention 分片与跨机通信重叠 | 必须报 GPU 数、互连和 strong scaling efficiency |
| [StreamDiffusionV2](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | SLO-aware batching、rolling KV、step/layer pipeline、多 GPU | 平均 FPS 不能代替 TTFF、deadline、jitter 和 p95/p99 |
| [LeanVAE](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_LeanVAE_An_Ultra-Efficient_Reconstruction_VAE_for_Video_Diffusion_Models_ICCV_2025_paper.html) | 轻量 video VAE 与重建效率 | VAE 模块倍数不能直接当作整条生成流水线倍数 |
| [FlashDecoder](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_FlashDecoder_Real-Time_Latent-to-Pixel_Streaming_Decoder_with_Transformers_CVPR_2026_paper.html) | 逐帧 latent-to-pixel Transformer decoder | decoder 变快不等于 denoiser 或 E2E 同倍加速 |
| [db-SP](https://proceedings.mlsys.org/paper_files/paper/2026/hash/db988b089d8d97d0f159c15ed0be6a71-Abstract-Conference.html) | 面向动态 block sparsity 的 head/block 双重负载均衡 | attention-specific 与 E2E speedup 必须分报 |
| [SANA-Video](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41b93c59da0d0f835907fd661d419db2-Abstract-Conference.html) | Linear DiT、常量内存状态与低精度部署共同设计 | 不能与固定 checkpoint 的单项 kernel 替换直接比较 |
| [TurboDiffusion](https://arxiv.org/abs/2512.16093) | 少步、低比特、稀疏/线性 attention 和工程组合 | 100–200× 是作者 full-stack 预印本结果，不是可分配给模块的独立倍数 |
| [Sol Video Inference Engine](https://arxiv.org/abs/2606.23743) | 面向模型/硬件实例组合 cache、sparse、token pruning、quant 和 fusion | 仅作为集成优化案例；截至 2026-09-02 仍为作者预印本 |

## 6. 原论文数值核对

| 工作 | 论文报告值 | 同时变化的变量 | 正文采用的表述 |
|---|---|---|---|
| ViDiT-Q | 2–2.5× memory；1.4–1.7× E2E latency | model、bit-width、GPU kernel、baseline | 仅说明原论文设置下的实际算子收益 |
| QuantSparse | 3.68× storage；1.88× E2E | quant + attention sparsity + distillation/reparameterization | 称组合压缩点，不评判单项优劣 |
| MobileVD | 14 latent frames、512×256，手机约 1.7 s | 分辨率、cross-attention、时间多尺度、block/channel pruning、单步 | 作为移动端组合部署结果，不将整体收益归于剪枝 |
| CausVid | 50 → 4 steps；作者报告 1.3 s TTFF、9.4 FPS | distill + causal factorization + KV + 特定输出/硬件 | 分别报 NFE、TTFF、吞吐，不只写“蒸馏加速” |
| PAB | 最高 10.5× | cache schedule + broadcast parallel + 特定模型/输出 | 仅说明原论文设置下的最高值，不用于跨论文排名 |
| TeaCache | 最高 4.41×，VBench 作者报告降 0.07% | Open-Sora-Plan、阈值、提示集与计时边界 | 完整保留 baseline 和质量口径 |
| AdaCache | 最高 4.7× | Open-Sora 作者设置、运动阈值和 cache schedule | 要求快运动/scene cut 额外验收 |
| MagCache | 2.10–2.68× | Open-Sora/CogVideoX/Wan/Hunyuan 的各自设置 | 不简化为通用 2.68× |
| Quant VideoGen | KV 约 6–7× 压缩；E2E 约 +1.5%–4.3% | 2-bit dequant 与特定 AR 模型 | 作为省内存但时延略增的反例 |
| xDiT | 多卡相对单卡作者报告多种倍数 | GPU 数、互连、parallel axis、模型与输出 | 说明系统扩展效率，不表述为算法 FLOPs 下降 |
| StreamDiffusionV2 | 4×H100 设置下的 TTFF/FPS/deadline 结果 | 多 GPU pipeline、batching、KV、模型/输出 | 引用时同时说明 SLO 与硬件条件 |

本文没有将不同论文中的 FLOPs、kernel、DiT-forward、E2E、单卡、多卡、latency 或 throughput 加速比放在同一列表中排序。

## 7. 公平复现方案

### 7.1 固定模型权重的免训练执行对比

固定 model/artifact hash、prompt、seed、sampler/NFE、CFG、分辨率、帧数/FPS、VAE、precision baseline、batch、hardware/software、warm-up 和计时范围。每次只改变 solver、cache、training-free token/attention sparsity、kernel、parallel 或 offload 中的一项。

记录：

```text
nominal steps / measured denoiser calls / CFG calls
text / denoiser / VAE / post / I/O latency
peak allocated / reserved / host memory / transfer bytes
attention density / selector overhead / cache hit and reset
cold and warm p50/p95; throughput; TTFF and jitter if streaming
same-seed numerical delta; frame/motion/text/diversity/failure metrics
```

### 7.2 涉及新 checkpoint 的模型压缩对比

对 distillation、QAT/PTQ 或 pruning 额外报告：训练/校准数据、timestep/prompt/length 采样、更新步数、GPU hours、teacher/student hash、参数变化、转换时间、bit metadata、未量化层、实际稀疏度和兼容 kernel。

目标是比较转换后系统在质量、时延和显存之间的权衡，不把它表述为固定权重下的纯执行优化。

### 7.3 全栈系统的逐级消融

```text
baseline
→ + fewer NFE / distilled checkpoint
→ + smaller structure or quantization
→ + sparse attention or token reduction
→ + cache
→ + fused/low-bit kernel
→ + offload / parallel / serving scheduler
```

每一级都重新测量质量、单次前向、端到端时延和峰值显存，并确认主要耗时是否转移。只有最终组合的实测结果可以称 full-stack speedup，模块倍数不相乘。

## 8. 未完成的复现与局限

1. 本次整理没有下载大型模型权重，没有在统一 GPU 环境中进行基准测试，也没有独立复现论文报告的加速比。
2. 部分 2025–2026 前沿仅能按作者预印本解读；正文已明标 B，不与正式 proceedings 同等对待。
3. 即使是正式发表的论文，其结果仍来自论文自身的实验，并非独立复现；硬件、输出、NFE、数值精度和软件版本不同时，不进行数字排名。
4. 剪枝中的运动—外观层敏感性仍没有统一测量；量化校准也缺少覆盖不同提示词、时间步、长度和分辨率的公开评测规范。
5. cache/稀疏论文的平均指标对 scene cut、快相机、小物体、手部/文字、强局部控制和长时漂移覆盖不足。
6. 训练成本、energy/video、峰值功耗、热降频和运维成本仍是 infra 文献的常见缺口。

## 9. 文档组织方式

- 将原综合 infra 章重构为“总览页 + 六篇独立综述”，不把剪枝/量化内容塞进 Diffusion 数学章。
- 总览页从 NFE、单次前向计算、数值精度与访存、计算复用和硬件执行五个方面分析推理成本；各专题则按照方法分类、技术演进、证据争议和未来方向组织，而不是简单按年份罗列论文。
- 所有定量结论均注明原论文的实验条件；综合案例不做单项归因。
- 在导航、工程师路线、Diffusion/Video DiT 专章与阅读清单中同时保留总览入口和专题直达链接，原有引用不删减。
