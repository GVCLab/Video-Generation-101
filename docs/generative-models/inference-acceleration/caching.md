# 视频扩散模型的缓存与特征复用综述

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本综述区分正式会议论文、作者公开论文和官方实现的证据等级。文中加速比均对应原论文所采用的模型、阈值、输出规格、硬件和计时方式；本仓库尚未在统一 GPU 环境中复现这些结果。

## 摘要

视频扩散模型在多个去噪时间步（denoising timestep）反复调用同一个大型去噪网络。相邻时间步的部分注意力输出、残差或网络块输出变化较小，为无需重新训练的特征复用提供了依据。然而，“缓存”并非单一机制：既可以跨去噪时间步复用特征，也可以在一次前向计算中跨层复用，还可以在因果视频模型中保留已经生成的历史键值（KV）。三类机制在近似误差、显存开销和失效条件方面存在显著差异。

2024–2026 年的研究大致经历了如下变化：早期方法采用固定刷新规则，随后转向由时间步、内容、运动、网络块和浅层探针共同决定的样本自适应策略，并进一步引入轨迹对齐、误差校正及针对不同特征维度的更新方法，以减轻过期特征带来的偏差。

## 1. 三个时间轴与概念边界

设视频时间为 $k$，denoising 时间为 $\tau$，网络深度为 $\ell$。

| 机制 | 复用跨度 | 典型对象 | 是否通常为近似 | 主要风险 |
|---|---|---|---:|---|
| inter-step feature cache | $\tau_i\rightarrow\tau_{i+1}$ | block/residual/attention/CFG feature | 是 | 过期特征、误差累积、敏感步损失 |
| cross-layer reuse | $\ell_i\rightarrow\ell_j$ | 相似中间特征 | 是 | 层职责异质，前景/背景或细节恢复受损 |
| causal KV cache | $k_i\rightarrow k_{i+1}$ | 已提交历史的 $K,V$ | 可以算法等价 | 显存随时长增长、eviction/reset 错误 |

普通双向 Video DiT 的跨步缓存不同于自回归模型的 KV 缓存。前者复用同一段视频在不同噪声水平下的中间结果；后者避免对已经生成的历史重复计算注意力。如果因果掩码、位置编码、滑动窗口或历史提交规则不同，KV 状态便不能直接沿用到新的请求。

## 2. 决策问题：何时复用、复用什么以及如何校正

对第 $i$ 个 denoising step，可将缓存决策写成：

```math
c_{i,\ell}=\mathbb{1}\!
\left[s_{i,\ell}<\delta_{i,\ell}\right],
```

其中 $s_{i,\ell}$ 表示当前时间步或网络层的变化程度或风险，$\delta_{i,\ell}$ 是与计算预算相关的阈值。现有方法主要在以下方面有所区别：

1. $s$ 取自 timestep embedding、input/residual difference、运动、前景/背景重要性，还是浅层 probe；
2. 缓存对象是整个 DiT block、attention、MLP、CFG 分支，还是部分 feature dimension；
3. 命中缓存后直接复制旧特征、进行线性或高阶校正，还是利用多步历史对齐轨迹；
4. 阈值由数据集离线标定、单个 prompt 标定，还是根据当前 sample 在线确定。

## 3. 技术路线及其演进

### 3.1 变化规律与预定义调度

Pyramid Attention Broadcast（PAB）分析空间、时间和交叉注意力在去噪过程中的变化，并以不同频率复用已有的注意力输出 [[1]](#ref-1)。该工作表明，不同类型注意力的冗余程度并不相同，因此整个网络通常不宜采用统一的刷新间隔。

TeaCache 用 timestep-embedding-modulated input 近似预测当前 residual 的变化，将固定间隔跳过计算改为基于累积误差的刷新 [[3]](#ref-3)。MagCache 从相邻 residual 的幅值比中归纳出较稳定的 magnitude law，并结合少量标定与异常时间步保护决定是否复用 [[7]](#ref-7)。二者均比固定 interval 更灵活，但仍存在阈值、质量与延迟之间的权衡，尚无可普遍保证“无损”的阈值。

### 3.2 利用 CFG、内容和运动信号

FasterCache 同时利用相邻时间步的特征冗余，以及同一时间步 CFG 条件分支与无条件分支之间的冗余 [[2]](#ref-2)。这种 CFG 缓存依赖相应的引导执行结构；对于已经蒸馏引导信息的学生模型、单分支推理或其他引导方式，相关策略未必适用。

AdaCache 根据每个样本的内容与运动信息安排缓存，并通过 Motion Regularization 为动态视频分配更多计算 [[4]](#ref-4)。因此，在低运动提示词上确定的最优阈值，不宜直接用于快速镜头、遮挡、镜头切换或多主体交互等场景。

### 3.3 块级、层级与分层复用

ProfilingDiT 观察到不同 block 对前景与背景的响应存在差异，因而根据语义重要性和时间步间的稳定性选择复用对象 [[5]](#ref-5)。BWCache 将复用粒度细化至 DiT block，以相邻时间步差异低于阈值作为触发条件，并依据中间稳定、两端变化较大的 U-shaped 规律分配计算预算 [[9]](#ref-9)。

QuantCache 联合使用 hierarchical latent cache、按层和时间步重要性分配的量化，以及结构冗余剪枝 [[6]](#ref-6)。其整体加速比反映的是多种优化的组合效果，不能据此推断任一缓存模块单独具有相同收益。

### 3.4 从固定规则转向在线探针与轨迹校正

DiCache 使用浅层在线探针估计深层缓存误差，再利用多步历史进行动态缓存轨迹对齐，据此确定复用时机与复用方式 [[8]](#ref-8)。ERTACache 通过离线残差分析确定可复用的时间步，并以轨迹感知系数和闭式残差线性化方法修正缓存引入的误差 [[10]](#ref-10)。

HyCa 将不同特征维度的演化建模为多类 ODE，并分别采用不同的缓存与更新规则 [[11]](#ref-11)。这些结果说明，仅以单一的整体特征差异评价整层可能过于粗略；与此同时，探针、轨迹拟合和校正也会产生额外计算，应计入端到端延迟。

### 3.5 因果 KV 缓存：保留流式生成中的已提交历史

CausVid 将双向教师模型蒸馏为因果学生模型，并在流式生成中结合少步采样与 KV 缓存 [[12]](#ref-12)。这种 KV 缓存依赖视频时间维度上的因果分解，与 TeaCache 等跨去噪时间步的特征复用方法并不等价，也不能相互替换。

Quant VideoGen 将自回归视频的 KV 压缩至 2-bit；在作者给定的实验设置中，cache 空间约缩小 6–7 倍，但解压开销使端到端时延约增加 1.5%–4.3% [[13]](#ref-13)。这一结果说明，缓存占用降低并不必然带来生成延迟下降。

## 4. 代表工作比较

| 方法 | 决策信号 | 缓存对象 | 校正/刷新 | 是否需训练 | 主要外推限制 |
|---|---|---|---|---:|---|
| PAB [[1]](#ref-1) | attention 变化规律 | spatial/temporal/cross attention | 分类 broadcast schedule | 否 | 不宜将作者报告的最高倍数视为所有骨干的固定收益 |
| FasterCache [[2]](#ref-2) | 步间与 CFG 分支冗余 | feature / CFG output | dynamic reuse | 否 | 收益不宜直接外推至无 CFG 或 guidance-distilled 模型 |
| TeaCache [[3]](#ref-3) | timestep-modulated input | residual/block output | 累积误差超阈刷新 | 否，需标定 | 单一模型的系数未必能跨长度或分辨率泛化 |
| AdaCache [[4]](#ref-4) | 内容与运动 | 可选 layer/module | sample-specific schedule | 否 | 低运动 prompt 的阈值不宜直接用于快动作或镜头切换 |
| ProfilingDiT [[5]](#ref-5) | 前景/背景职责 + 步间差 | block feature | 层/步双粒度 | 否，需 profiling | block 的语义职责未必能在不同骨干间直接迁移 |
| MagCache [[7]](#ref-7) | residual magnitude ratio | residual output | 幅值规律+异常步保护 | 否，少量标定 | 少量样本标定不能替代异常场景测试 |
| DiCache [[8]](#ref-8) | 浅层 online probe | 深层 feature | 多步轨迹对齐 | 否 | probe overhead 仍需计入端到端延迟 |
| BWCache [[9]](#ref-9) | block 步间差 | DiT block | 阈值触发刷新 | 否 | 相同 U-shaped 规律未必适用于所有 scheduler |
| ERTACache [[10]](#ref-10) | 离线 residual profile | cached residual | 轨迹校正+误差整流 | 否，需 profiling | 离线 prompt 分布未必代表线上请求 |
| Quant VideoGen [[13]](#ref-13) | 历史多尺度 KV | causal KV | 逐级残差量化/解压 | 否 | 低 bit 的显存收益不等同于 latency 收益 |

## 5. 主要结论与未决问题

### 5.1 较为一致的观察

1. **敏感性随 denoising 时间步和 layer 改变。** 语义快速建立阶段、中间稳定阶段和后期细节恢复阶段不宜共用固定的 cache interval。
2. **复用时机与误差校正同样重要。** 直接复制能够减少更多计算，但也更容易产生累积误差。
3. **Training-free 不等于无需配置或没有偏差。** 阈值、标定 prompt、cache 粒度和 reset 规则仍与模型及输出规格相关。
4. **缓存涉及质量、计算与显存之间的权衡。** 额外的 cache bytes、workspace 和指示量计算可能提高峰值显存。

### 5.2 尚待解决的问题

- **“近无损”的含义仍需明确。** 同 seed 下的输出接近并不等同于生成分布不变。PSNR/LPIPS 对 dense output 很敏感，但不能代替跨 seed 多样性、运动强度和文本遵循评测。
- **少量 prompt 标定的泛化范围有限。** magnitude 或 timestep 规律可能较为稳定，但镜头切换、极端运动和强控制场景仍需单独验证。
- **在线 probe 的净收益取决于模型与采样步数。** probe 可以减少误跳，但在 few-step student 或小模型上，指示量计算本身可能占据更高的延迟比例。
- **batch 内异质 schedule 会增加服务调度难度。** 当不同 sample 在不同 step/layer 刷新时，dynamic batching 的效率和 GPU 利用率可能下降。

## 6. 公平评测与服务环境评估

建议记录的最小缓存配置如下：

```text
model/checkpoint / task / prompt set / seeds
sampler / nominal steps / measured NFE / CFG structure
resolution / frames / FPS / batch / precision
cache object / tensor shape / dtype / bytes
indicator / threshold / calibration data / refresh layers and steps
hit or skip rate / reset triggers / correction overhead
GPU / software / compile / warm-up / timing boundary
text / denoiser / VAE / post / I/O latency
peak allocated / reserved / host memory
same-seed delta / motion / text / consistency / diversity / failure cases
```

评测集至少应覆盖低运动、快运动、相机环绕、scene cut、多主体遮挡、小物体、文字/手部、强 prompt 切换、轨迹或局部控制突变，以及超过标定长度的 rollout。对于 streaming 系统，还应报告 TTFF、逐帧 deadline、jitter、p95/p99、cache reset 和请求取消后的状态清理情况。

建议按以下四个层级报告结果：

1. cache 命中率与净减少的 block/attention 计算；
2. 指示量、重排、校正和内存操作在内的单步延迟；
3. 包含 text encoder、VAE 与 I/O 的端到端延迟；
4. 相同质量容差下的 latency–VRAM–quality Pareto 前沿。

## 7. 未来方向

1. **从调节阈值转向可校准的风险估计。** 预测“本次复用导致质量超出容差的概率”，而非仅给出未经标定的差异分数。
2. **从平均 prompt 转向面向异常场景的调度。** 在 scene cut、运动突变、控制信号改变或内容安全策略触发时强制 reset。
3. **联合调度 cache 与 dynamic batching。** 将 refresh schedule 相近的请求合批，以降低 branch divergence 和 straggler。
4. **统一 denoising cache 与 data-time memory 的状态管理规范。** 明确 key、版本、position、eviction、reset、容错和隐私边界。
5. **分析多种近似误差的联合传播。** 研究量化噪声、稀疏误差和 cache 误差在多步复用过程中的累积及其上界。
6. **评估缓存的能耗与带宽成本。** 在边缘设备上报告 energy/video、DRAM 读写量、温度和持续吞吐，而非仅报告单次 latency。

## 8. 建议阅读路线

1. 先读 PAB，理解不同 attention 模块的步间冗余 [[1]](#ref-1)。
2. 用 TeaCache、FasterCache 和 AdaCache 对照 timestep、CFG 和 motion/content 三种决策信号 [[2]](#ref-2) [[3]](#ref-3) [[4]](#ref-4)。
3. 再读 ProfilingDiT、MagCache 与 BWCache，比较不同网络块的作用、幅值规律和 U 形变化 [[5]](#ref-5) [[7]](#ref-7) [[9]](#ref-9)。
4. 用 DiCache、ERTACache 和 HyCa 理解在线探针、轨迹校正和按特征维度选择更新方法 [[8]](#ref-8) [[10]](#ref-10) [[11]](#ref-11)。
5. 最后对照 CausVid 与 Quant VideoGen，区分跨去噪时间步的特征复用与视频时间维度上的 KV 记忆 [[12]](#ref-12) [[13]](#ref-13)。

返回[压缩与推理加速综述导航](../inference-acceleration.md)；与量化、稀疏化和在线服务的交叉分别见[量化综述](quantization.md)、[Token 与注意力稀疏化综述](sparsity.md)和[系统与部署综述](systems.md)。

## 参考文献

<a id="ref-1"></a>[1] [Real-Time Video Generation with Pyramid Attention Broadcast](https://proceedings.iclr.cc/paper_files/paper/2025/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html). Xuanlei Zhao et al. ICLR. 2025.

<a id="ref-2"></a>[2] [FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://proceedings.iclr.cc/paper_files/paper/2025/hash/518046d86bbc41a0707727c38301ad8e-Abstract-Conference.html). Zhengyao Lyu et al. ICLR. 2025.

<a id="ref-3"></a>[3] [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html). Feng Liu et al. CVPR. 2025.

<a id="ref-4"></a>[4] [Adaptive Caching for Faster Video Generation with Diffusion Transformers](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html). Kumara Kahatapitiya et al. ICCV. 2025.

<a id="ref-5"></a>[5] [Model Reveals What to Cache: Profiling-Based Feature Reuse for Video Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.html). Xuran Ma et al. ICCV. 2025.

<a id="ref-6"></a>[6] [QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_QuantCache_Adaptive_Importance-Guided_Quantization_with_Hierarchical_Latent_and_Layer_Caching_ICCV_2025_paper.html). Junyi Wu et al. ICCV. 2025.

<a id="ref-7"></a>[7] [MagCache: Fast Video Generation with Magnitude-Aware Cache](https://proceedings.neurips.cc/paper_files/paper/2025/hash/311207bb626e36a8f1d3eb92aa67af22-Abstract-Conference.html). Zehong Ma et al. NeurIPS. 2025.

<a id="ref-8"></a>[8] [DiCache: Let Diffusion Model Determine Its Own Cache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/78288ef33b18a351c3cd679dc9a15c8d-Abstract-Conference.html). Jiazi Bu et al. ICLR. 2026.

<a id="ref-9"></a>[9] [BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Abstract-Conference.html). Hanshuai Cui et al. ICLR. 2026.

<a id="ref-10"></a>[10] [ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion](https://proceedings.iclr.cc/paper_files/paper/2026/hash/84d395725a9b40cb4a49d84478ac24c7-Abstract-Conference.html). Xurui Peng, Chenqian Yan, Hong Liu, Rui Ma, Fangmin Chen, Xing Wang, Zhihua Wu, Songwei Liu, Mingbao Lin. ICLR. 2026.

<a id="ref-11"></a>[11] [Let Features Decide Their Own Solvers: Hybrid Feature Caching for Diffusion Transformers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html). Shikang Zheng et al. ICLR. 2026.

<a id="ref-12"></a>[12] [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html). Tianwei Yin et al. CVPR. 2025.

<a id="ref-13"></a>[13] [Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](https://arxiv.org/abs/2602.02958). Haocheng Xi et al. Author paper; accepted at ICML. 2026.
