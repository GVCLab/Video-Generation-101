# 视频扩散模型量化综述

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本综述优先采用经过同行评审的会议论文，并参考作者论文和官方实现；预印本仅用于补充近期进展。文中引用的质量、显存和速度数据均以原文使用的模型、输出规格、硬件、底层算子和计时方法为限，本项目尚未在统一环境中独立复现。

量化通过降低权重、激活或缓存的数值精度，减少模型存储、显存占用和数据搬运；它本身并不减少去噪步数。只有当低比特算子、数据布局和完整生成流程均能有效利用低精度表示时，较小的模型文件才可能转化为实际的视频生成加速。

## 1. 综述范围与文献依据

本章主要讨论四个问题：量化对象是哪些张量；采用 PTQ、QAT 还是带训练的压缩方法；如何处理视频 DiT 在去噪时间步、token 和时空位置上的分布差异；比特数降低能否转化为实际运行时间的改善。

本文纳入直接压缩视频扩散模型去噪网络、注意力、特征缓存或因果视频模型 KV 缓存的研究。图像扩散量化仅在其为时间步感知校准、异常值处理或底层算子协同设计提供方法基础时加以讨论 [[1]](#ref-1) [[9]](#ref-9)。

视频 tokenizer 的离散码本量化、训练数据压缩、仅调整 sampler 或 solver 以减少采样步数，以及没有低比特算子实现而只报告 FLOPs 的工作，不属于本章的主要范围。

现有证据主要来自各方法论文自身的实验，目前尚无统一的视频扩散量化 benchmark。因此，本文侧重比较研究问题与方法设计，不对不同论文给出脱离实验条件的速度排名。

## 2. 量化对象及精度说明

对实值张量 $x$ 的常见仿射量化形式为：

```math
q=\operatorname{clip}\!\left(\operatorname{round}(x/s)+z,q_{\min},q_{\max}\right),
\qquad \hat{x}=s(q-z),
```

其中 $s$ 为 scale，$z$ 为 zero-point。实际存储和显存占用还包括 scale、zero-point、分组元数据、padding、未量化层以及运行时 workspace，因此仅按权重位宽计算的 $b/16$ 比例不能代表峰值显存的下降比例。

| 对象 | 常见写法 | 主要收益 | 主要风险 |
|---|---|---|---|
| Linear/MLP 权重 | W8、W4，per-channel/group | checkpoint 与 HBM 读取减少 | 只有 W4A16 时计算仍可受 activation 制约 |
| activation | A8、A6、A4，per-token/channel | 低比特 GEMM 与激活显存 | outlier、层间和 timestep 间范围剧变 |
| Attention $Q,K,V$ | INT8/INT4/FP8 | attention 投影与带宽下降 | $QK^\top$ 误差经 softmax 非线性放大 |
| Attention logits/probability | 常保留较高精度 | 若 kernel 支持可再降成本 | 小概率远距依赖易被截断 |
| 跨去噪步 feature cache | cache INT8/INT4 | cache bytes 与搬运减少 | 量化误差被后续多步复用 |
| 因果生成 KV cache | KV8/KV4/KV2 | 长时生成的历史内存下降 | 解量化开销、历史误差累积 |
| VAE / text encoder | 独立 dtype | 端到端常驻权重和固定成本下降 | 文本对齐、解码颜色或细节受损 |

报告 `W4A8` 时，还需说明量化范围是否涵盖 attention projection、MLP、modulation 和 output head；若使用“平均 4-bit”的说法，则应列出保留高精度的例外层，并给出按参数量加权的 effective bits。

还需区分两类缓存：**inter-step feature cache** 在不同去噪 timestep 之间复用，**autoregressive KV cache** 则随视频时间不断增长。后者只适用于具有因果 attention 和 KV 接口的生成器，其结论不能直接用于普通的双向 Video DiT。

## 3. PTQ、QAT 与训练辅助量化

| 范式 | 权重是否更新 | 所需数据与成本 | 适用情形 | 需要说明的条件 |
|---|---:|---|---|---|
| PTQ | 原始模型权重通常冻结 | 少量 calibration activation | 快速转换已有 checkpoint | 若优化 scale 或 rounding，仍应报告相应计算成本 |
| Data-free PTQ | 冻结 | 不使用真实校准视频 | 数据不可用或受隐私限制 | 仍可能使用合成代理数据、统计量或网格搜索 |
| QAT | 是 | fake-quant forward 与梯度训练 | 极低 bit 下的性能恢复 | 应报告训练数据、GPU hours 与新 checkpoint |
| 量化+蒸馏/微调 | 是，或更新辅助分支 | 教师特征或视频损失 | 恢复时空关系 | 效果不能完全归因于 PTQ |

PTQ4DM 较早将扩散模型在不同 timestep 上的分布变化纳入校准数据和量化指标设计 [[1]](#ref-1)。视频方法还需处理不同帧、空间 token 和时间 token 之间的差异，因此校准数据的采样方式与量化器的分组策略同样重要。

DVD-Quant 研究 data-free 的 Video DiT PTQ，并采用 Bounded-init Grid Refinement、Auto-scaling Rotated Quantization 和 $\delta$-Guided Bit Switching，降低方法对重新校准流程的依赖 [[8]](#ref-8)。这里的“data-free”仅指不使用真实校准数据，并不意味着没有搜索开销、先验信息或针对特定模型的设计。

## 4. 视频扩散的特有量化难点

### 4.1 不同去噪时间步的激活分布变化

同一网络层在高噪声和低噪声阶段的数值幅度、异常值通道及条件信号强度可能不同。若只使用单个 timestep 或均匀采样的 timestep 进行校准，便难以反映实际采样过程中不均匀的量化敏感性。ViDiT-Q 因此引入 timestep-aware channel balancing 和 mixed precision，而非对所有层与时间步使用同一静态量化规则 [[3]](#ref-3)。

### 4.2 Token 与时空位置的分布差异

视频 token 同时承载前景与背景、静止与快速运动、空间细节和跨帧关系。QVD 观察到 temporal feature skew 和通道数值范围不均，并以 HTDQ 保留时间特征的区分度，以 SCRI 改善单个通道对量化 level 的利用 [[2]](#ref-2)。

Q-VDiT 进一步从 token 和 feature 两个维度分析误差，使用 Token-aware Quantization Estimator 进行补偿，并通过 Temporal Maintenance Distillation 保持帧间的时空关系 [[4]](#ref-4)。由于该方法同时使用量化与蒸馏，其性能变化不能仅归因于 PTQ。

### 4.3 Attention 的量化误差

$QK^\top$ 中的偏差经 softmax 后可改变注意力排序；当稀疏化同时删除 attention edge 时，两类误差还会耦合。QuantSparse 以 multi-scale salient attention distillation 和 second-order sparse-attention reparameterization 恢复这种联合压缩下的偏移 [[5]](#ref-5)。

因此，权重或激活的 MSE 只能作为代理目标。视频评测还应包括长程主体一致性、运动质量、文本遵循和不同随机种子下的多样性，以发现数值误差较小但生成分布已经发生变化的情况。

### 4.4 时空冗余及其适用条件

DeltaQuant 将局部三维时空块划分为 core token 和 delta token：前者保留为 FP8，后者使用 4-bit 表示，并结合低秩权重分支与定制 kernel [[6]](#ref-6) [[9]](#ref-9)。其基本假设并非所有 token 都相似，而是在特定时空块、时间步和模型中，delta 相较原始 activation 更适合低比特表示。

在快速运动、镜头切换、局部编辑、高频纹理或更长视频中，这一假设可能不再充分成立。若校准集以静态镜头为主，便可能高估时空差分方法的适用性。

### 4.5 两种时间累积误差

在去噪过程中，早期时间步产生的偏差会改变后续步骤的输入；在因果视频生成过程中，量化后的 KV 则会持续参与后续帧的计算。Quant VideoGen 针对后一问题提出 semantic-aware smoothing 和 progressive residual quantization，以由粗到细的残差形式表示 2-bit KV cache [[7]](#ref-7)。

## 5. 方法演进：从静态 PTQ 到视频系统的联合设计

| 工作 | 文献状态 | 量化对象/范式 | 视频特化机制 | 原文实验结果 | 适用范围与限制 |
|---|---|---|---|---|---|
| PTQ4DM [[1]](#ref-1) | CVPR 2023 | 扩散 U-Net PTQ | multi-timestep calibration | 8-bit 生成质量可维持 | 原文未验证视频 DiT 的端到端系统 |
| QVD [[2]](#ref-2) | ACM MM 2024 | Video DM PTQ，W/A | HTDQ + SCRI | W8A8 在原文评估中接近全精度结果 | 尚无统一的真实低比特 kernel 评测 |
| ViDiT-Q [[3]](#ref-3) | ICLR 2025 | DiT PTQ，W8A8/W4A8 | timestep-aware balancing + mixed precision | 2–2.5× 内存优化，1.4–1.7× 端到端时延改善 | 结果限于原文的 GPU、kernel 与基线设置 |
| Q-VDiT [[4]](#ref-4) | ICML 2025 | W3A6 + distillation | TQE + TMD | scene consistency 23.40，原文称较当时量化方法提升 1.9× | 指标比值不能解释为时延加速比 |
| QuantCache [[10]](#ref-10) | ICCV 2025 | 量化+缓存+层剪枝 | 按 timestep/layer 分配精度 | Open-Sora 设置下端到端加速 6.72× | 收益同时来自量化、缓存与层剪枝 |
| QuantSparse [[5]](#ref-5) | ICLR 2026 | 量化+attention sparsity | salient distillation + reparameterization | HunyuanVideo-13B：3.68× 存储压缩、1.88× 端到端加速 | 收益同时来自量化与稀疏化 |
| DeltaQuant [[6]](#ref-6) | CVPR 2026 | W4A4/FP8 core + kernel | spatiotemporal delta smoothing | Wan2.2：模型压缩 2.9×、显存下降 2.3× | 111.8× 是多项技术共同作用的系统结果 |
| DVD-Quant [[8]](#ref-8) | ICLR 2026 | data-free W4A4 PTQ | BGR + ARQ + $\delta$-GBS | 原文报告约 2× 加速 | 需在相同数据可用性和 kernel 条件下比较 |
| Quant VideoGen [[7]](#ref-7) | 2026 预印本，已被 ICML 2026 接收 | 因果视频 2-bit KV cache PTQ | semantic smoothing + progressive residual | KV 最高约 7× 压缩，端到端时延开销小于 4% | 不适用于 KV 不随时间增长的双向模型 |

现有研究已从面向去噪时间步的校准，扩展到 token 与时空关系的保持，并进一步探索量化、缓存、稀疏化与底层算子的联合设计。

## 6. 硬件与底层算子实现

伪量化（fake quantization）使用 FP16 或 BF16 张量模拟取整与截断，可用于评估质量损失，但不能据此判断低比特计算的实际吞吐。要获得真实加速，至少需要完成权重打包，硬件支持目标数据类型，GEMM 或注意力算子能够直接使用相应的打包格式，并且动态缩放、解量化及数据布局转换的开销没有抵消低比特计算的收益。

ViDiT-Q 同时提供 GPU kernel，因此能够报告实际显存占用与端到端时延 [[3]](#ref-3)。SVDQuant 在图像扩散模型中说明了类似问题：若高精度低秩分支单独运行，额外的 activation 搬运可能抵消量化收益，Nunchaku 因此将该分支融合进低比特 kernel [[9]](#ref-9)。

DeltaQuant 也为 core/delta 分解实现了定制算子，而没有将理论上的误差改善直接解释为速度提升 [[6]](#ref-6)。这说明量化算法、数据布局与底层算子需要协同设计；若缺少其中任一环节，通常只能验证存储、质量或理论计算量中的部分收益。

实验报告还应列出 GPU 或加速器型号、Tensor Core 支持的 dtype、CUDA、驱动与编译器版本、batch、输入 shape、预热方式、graph capture、回退到高精度 kernel 的层，以及计时是否包含 CFG 双前向和 VAE 解码。

## 7. 质量、时延与显存的评测设置

### 7.1 固定生成条件

- 使用相同的基础 checkpoint 或 hash、sampler、实际 NFE、CFG/guidance、prompt、seed 和 negative prompt。
- 固定帧数、FPS、分辨率、宽高比、batch 和 VAE 解码设置。
- 比较 PTQ 方法时采用相同的 calibration budget，并将校准 prompt 与评测 prompt 分开。
- QAT 或蒸馏方法还应报告数据来源、更新步数、GPU hours 和教师模型，不宜与 training-free PTQ 直接合并比较。

### 7.2 说明各组件的数值精度

```text
weights / activations / QKV / logits-softmax / feature cache / KV cache
VAE / text encoder / normalization / first-last-sensitive layers
granularity / group size / symmetric-asymmetric / static-dynamic scale
packed format / accumulator dtype / fallback operators / effective average bits
```

### 7.3 分项报告生成质量

至少应分别报告单帧视觉质量、主体与背景一致性、运动幅度和平滑度、文本遵循、长程漂移、颜色或饱和度偏移，以及生成多样性。FVD、VBench、CLIP 和 PSNR 衡量的对象不同，不能相互替代；人工评测则应说明 prompt 数量、评审人数和结果的不确定性。

### 7.4 区分速度收益与容量收益

| 类别 | 最小报告项 |
|---|---|
| 存储 | 打包 checkpoint bytes、scale/metadata、effective bits/parameter |
| 显存 | peak allocated、peak reserved、权重/激活/cache/workspace 拆分 |
| 算子 | GEMM/attention microbenchmark，包含 dequant/reorder，不只报理论 TOPS |
| Denoiser | 在 token 数和 NFE 相同的条件下报告单次前向时延 |
| 端到端 | cold/warm p50/p95，包含 text encode、CFG、VAE、数据搬运和同步 |
| 容量 | 基线发生 OOM 时，报告可生成的最大帧数或分辨率，不换算为加速比 |

一组可比较的结果至少应同时包含质量指标、打包后的模型大小、峰值显存、单次前向时延和端到端时延。单独报告一个加速或压缩倍数，难以判断方法的实际收益。

## 8. 结果解读中的常见问题

1. **更低位宽未必带来更低时延。** 在 Quant VideoGen 的实验设置中，2-bit KV 将 cache 最高压缩约 7×，但解量化仍产生小于 4% 的端到端时延开销 [[7]](#ref-7)。这一结果主要体现容量收益，而非时延加速。
2. **“近无损”取决于评价指标与 prompt 分布。** 平均视频指标无法保证文字、手部、小物体、快速运动或镜头切换等困难样例不受影响。
3. **组合系统的收益需要通过消融实验分析。** QuantCache 的 6.72× 同时来自层级 cache、自适应量化、结构剪枝和 CUDA 实现 [[10]](#ref-10)；QuantSparse 也同时采用稀疏 attention [[5]](#ref-5)。
4. **模型大小的压缩比例不等于峰值显存的下降比例。** 在长时、高分辨率视频中，activation、attention workspace、cache 和运行时内存分配可能占据主要显存。
5. **不同论文对 PTQ 的训练范围定义并不完全一致。** 优化 scale 或 rounding、学习 rotation、增加低秩分支和使用教师蒸馏所需的成本不同，应分别说明。
6. **KV-cache 量化只适用于部分视频扩散模型。** 因果或自回归系统中的历史状态会随视频长度增长，而离线双向去噪中的 $K,V$ 只在当前前向过程中使用，二者对应不同的部署条件。
7. **校准过程可能过拟合评测分布。** 如果使用相同 prompt 选择 timestep、bit-width 和敏感层，量化方法的泛化能力可能被高估；应另外保留未见过的风格、运动类型和视频长度用于测试。

## 9. 研究空白与未来方向

1. **统一的系统评测基准。** 需要选用多个开放的 Video DiT，固定硬件代际、输出长度和端到端计时代码，同时公布权重打包格式及回退到高精度 kernel 的情况。
2. **校准集的迁移能力。** 需要系统考察视频由短变长、任务由 T2V 扩展到 I2V 或控制生成，以及场景由静态变为快速运动时产生的 calibration shift。
3. **两类时间累积误差的理论分析。** 现有工作多以经验方法处理 denoising 时间上的量化误差，对去噪误差与自回归历史误差共同作用下的长程稳定性仍缺少可预测的界限。
4. **覆盖完整生成流程的量化。** 除 denoiser 外，VAE、text encoder、conditioning/control branch 和 post-processing 等组件仍缺少统一的 Pareto 分析。
5. **动态精度与可预测时延。** 根据 timestep、token 或内容切换 bit 有助于改善质量，但也会引入条件分支、重新打包和尾时延；动态策略需要与服务 SLO 共同优化。
6. **量化与蒸馏、剪枝及 cache 的联合误差分析。** 需要在相同总计算成本下进行逐项消融，而不能直接相乘不同论文报告的加速比。
7. **端侧部署与跨厂商可移植性。** 同一 W4A4 设计在 NVIDIA GPU、移动 NPU 和其他加速器上可能需要不同的 accumulator、group size 与数据布局。
8. **面向视频质量的敏感性分析。** 网络层和 token 的重要性不应只由 reconstruction error 决定；后续研究需在 bit allocation 中直接考虑运动、身份、文字、局部控制和长程一致性。
9. **复现所需的公开材料。** 除 checkpoint 外，还应发布 calibration prompt 与 timestep 索引、打包后的权重、kernel、编译参数、计时脚本和典型失效样例。

## 10. 建议阅读顺序

1. 首先阅读 PTQ4DM，理解扩散模型为何不能只用单一时间点进行 calibration [[1]](#ref-1)。
2. 随后对照 QVD 与 ViDiT-Q，区分时序和通道分布处理与 DiT 敏感性分析及 kernel 实现两类思路 [[2]](#ref-2) [[3]](#ref-3)。
3. 阅读 Q-VDiT，了解极低 bit 量化为何需要引入时空蒸馏 [[4]](#ref-4)。
4. 结合 QuantSparse 与 QuantCache，分析量化与 sparsity、cache、pruning 之间的收益和误差耦合 [[5]](#ref-5) [[10]](#ref-10)。
5. 对照 DeltaQuant 和 DVD-Quant，比较利用时空差分与不使用真实校准数据两种 4-bit 量化思路 [[6]](#ref-6) [[8]](#ref-8)。
6. 最后阅读 Quant VideoGen，并注意区分因果模型的 KV cache 与离线双向 Video DiT 中的注意力状态 [[7]](#ref-7)。

## 参考文献

<a id="ref-1"></a>[1] [Post-Training Quantization on Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.html). Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, Yan Yan. CVPR. 2023.

<a id="ref-2"></a>[2] [QVD: Post-training Quantization for Video Diffusion Models](https://doi.org/10.1145/3664647.3681050). Shilong Tian et al. ACM Multimedia. 2024.

<a id="ref-3"></a>[3] [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html). Tianchen Zhao et al. ICLR. 2025.

<a id="ref-4"></a>[4] [Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers](https://proceedings.mlr.press/v267/feng25q.html). Weilun Feng et al. ICML. 2025.

<a id="ref-5"></a>[5] [QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html). Weilun Feng et al. ICLR. 2026.

<a id="ref-6"></a>[6] [DeltaQuant: 4-bit Video Diffusion Models with Spatiotemporal Delta Smoothing](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DeltaQuant_4-bit_Video_Diffusion_Models_with_Spatiotemporal_Delta_Smoothing_CVPR_2026_paper.html). Xingyang Li et al. CVPR. 2026.

<a id="ref-7"></a>[7] [Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](https://arxiv.org/abs/2602.02958). Haocheng Xi et al. Author paper; accepted at ICML 2026. 2026.

<a id="ref-8"></a>[8] [DVD-Quant: Data-free Video Diffusion Transformers Quantization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3bd28dd5cc4e15f9e019da13cc0c4844-Abstract-Conference.html). Zhiteng Li et al. ICLR. 2026.

<a id="ref-9"></a>[9] [SVDQuant: Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html). Muyang Li et al. ICLR. 2025.

<a id="ref-10"></a>[10] [QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_QuantCache_Adaptive_Importance-Guided_Quantization_with_Hierarchical_Latent_and_Layer_Caching_ICCV_2025_paper.html). Junyi Wu et al. ICCV. 2025.
