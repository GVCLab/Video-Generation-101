# 视频扩散模型的 Token 与注意力稀疏化综述

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本文优先采用 CVPR、ICCV、ICML、NeurIPS 和 ICLR 的正式论文。文中的速度与质量数据均来自作者在特定模型、序列长度、GPU、底层算子和计时设置下的实验；由于尚未在统一环境中复现，本文不对不同工作的性能进行直接排名。

## 摘要

视频 Diffusion Transformer（DiT）将时间、空间和条件信息编码为长序列。当 token 数为 $N$、隐维为 $d$ 时，密集自注意力中注意力分数的主要计算量随 $N^2d$ 增长，线性投影与 FFN 的计算复杂度则为 $O(Nd^2)$。现有方法主要从两个方面降低计算量：一是减少 token 数量，从而同时降低注意力、MLP 和激活的开销；二是减少注意力连边，仅计算部分 token 对之间的交互。

近期研究主要关注三个相互关联的问题：如何识别必要的 token 或连接，如何将稀疏模式组织为适合 GPU 执行的布局，以及如何根据提示词、网络层、注意力头和去噪时间步动态调整稀疏模式。

## 1. 相关概念与研究范围

| 对象 | 主要变化 | 参数量是否变化 | 是否属于本文重点 |
|---|---|---:|---:|
| weight / channel / head / layer pruning | 权重或网络结构 | 是或形成稀疏参数 | 否，见[模型剪枝综述](pruning.md) |
| token merge / prune | 运行时序列长度 $N$ | 否 | 是 |
| attention edge sparsity | $QK^\top$ 中实际参与计算的 token 对 | 否 | 是 |
| linear / recurrent mixer | 以线性或状态机制替代原有交互算子 | 可能需预训练或后训练 | 作为相关方向讨论 |

不同方法需要采用相应的评测指标。Token 缩减方法应报告实际序列长度 $N$、合并位置以及选择、聚合和分散操作的开销；注意力稀疏方法则应报告实际稠密度、分块大小、检索或重排成本和所用稀疏算子。理论 FLOPs 的下降并不足以说明实际运行时延随之降低。

## 2. 方法分类与代表工作

### 2.1 Token Merge / Prune：缩短运行时序列

Token merge/prune 方法首先估计可合并的空间或时间单元，再在部分 layer 中使用较短序列进行计算。ADAPTOR 根据局部运动信号自适应地减少 temporal token，使静态区域与动态区域获得不同的计算预算 [[1]](#ref-1)。

这类方法可以同时降低 attention、FFN 和 activation 的开销，但过早合并可能损害小物体、文字、手部、遮挡关系和快速运动的建模。对于接收轨迹、mask、ControlNet 或局部编辑条件的模型，token map 的变化还需与控制坐标保持一致。

### 2.2 Training-free Sparse Attention：利用预训练模型中的连接冗余

Sparse Video-Gen 观察到不同 head 分别侧重空间或时间交互，采用 online profiling 对 head 进行分类，并协同设计布局变换与定制 kernel [[2]](#ref-2)。AdaSpa 进一步考虑稀疏模式随 input、layer 和 head 的变化，采用 online precise search、LSE 复用以及 head-adaptive block 预算 [[4]](#ref-4)。RAPID 利用相邻 denoising step 之间 sparsity pattern 的稳定性复用模式，同时保留 inter-step adaptation [[7]](#ref-7)。

这类方法需要在计算预算与连接质量之间取得平衡。部分工作在最初若干 step 或敏感 layer 中保留 dense attention，并在较稳定的区间提高稀疏度，以减少重要连接被误删的风险。

### 2.3 语义重排与路由：兼顾连接选择与执行效率

Sparse VideoGen2 指出，当重要 token 分散在序列的不同位置时，GPU 仍可能因 padding 和非连续访存产生额外开销。该方法利用语义聚类与 permutation 将重要 token 重排为较密集的 block，再通过动态 kernel 进行计算 [[3]](#ref-3)。VORTA 在多种 sparse attention 变体之间进行路由，以兼顾局部和长距离交互 [[5]](#ref-5)。VMoBA 则通过 1D/2D/3D block partition、global block selection 和阈值预算，建模不同 head 的时空局部性 [[10]](#ref-10)。

上述方法同时涉及稀疏模式选择和系统实现。若语义选择无法转换为高效的执行布局，收益可能仅停留在理论 FLOPs；若布局规则但 token 选择不准确，则可能损害长程身份一致性、遮挡后的主体重现以及镜头内的关系建模。

### 2.4 可训练稀疏机制与后训练转换

Training-free 方法便于应用于已有模型，但 selector 受原始 attention 分布的限制。VSA 将 coarse tile 选择和 fine token attention 纳入可微 kernel，使稀疏计算可同时用于训练与推理 [[6]](#ref-6)。BLADE 联合训练 adaptive block sparsity 与 step distillation；其实验表明，先将模型蒸馏为少步模型、再独立进行稀疏化，未必能够补偿两种近似产生的联合误差 [[8]](#ref-8)。QuantSparse 则对稀疏 attention 与低比特模型进行联合恢复，进一步显示稀疏误差与量化误差之间存在耦合 [[13]](#ref-13)。

LinVideo 选择性地将部分 softmax attention 转换为 linear attention，并利用 distribution matching 恢复性能 [[12]](#ref-12)。SANA-Video 则在骨干设计阶段引入 block linear DiT [[11]](#ref-11)。前者属于已有 checkpoint 的后训练转换，后者属于新架构设计；两者均不宜与 training-free sparse mask 直接归为同一类实验。

### 2.5 分布式稀疏计算：通信与负载均衡

DSA 对 sparse attention、多 GPU 切分、调度和负载均衡进行联合设计 [[9]](#ref-9)。在多卡环境中，如果部分 head 或 block 保留更多连接，整个 attention 的时延将受到最慢设备的限制；若动态稀疏进一步引入 all-to-all 或频繁的数据重分布，通信开销可能抵消理论上的计算节省。

因此，多卡实验需要同时报告设备数量、互连方式、各卡的 density 分布、通信时间、straggler 和 strong-scaling efficiency，而不能只给出全局平均 sparsity。

## 3. 2025—2026 年的方法演进

| 阶段 | 代表工作 | 主要问题 | 方法进展 |
|---|---|---|---|
| 运动感知 token 缩减 | ADAPTOR（2025） | 哪些时间 token 可被合并 | 从静态规则转向视频内容自适应 |
| online pattern + kernel | Sparse Video-Gen（2025） | 分类空间/时间 head | 将 pattern 与可执行 layout 同时设计 |
| 精确搜索与动态预算 | AdaSpa、VORTA（2025） | input/head/layer 变化与长距交互 | 由固定 mask 转向动态稀疏模式 |
| 语义重排 | Sparse VideoGen2（2025） | 关键 token 分散导致的 GPU 浪费 | 区分 identification accuracy 与 hardware efficiency |
| 训练期稀疏 | VSA（2025） | 训练和推理共用可微 kernel | 将稀疏计算纳入模型训练 |
| 跨步模式复用 | RAPID（2026） | 降低 search 自身的开销 | 结合 attention sparsity 与 denoising-time cache |
| 联合压缩 | BLADE、QuantSparse（2026） | 稀疏与少步/量化的联合误差 | 对多种压缩误差进行联合优化 |
| 线性化与分布式执行 | LinVideo、SANA-Video、DSA（2026） | 替换算子与跨卡执行 | 扩展至 architecture/system co-design |

这些工作并非在相同实验条件下逐年改进。较新的方法通常同时引入语义聚类、稀疏 kernel、少步蒸馏或多卡调度，因此其整体加速比不能仅归因于 sparsity。

## 4. 代表方法比较

| 工作 | 主要对象 | 是否需要新 checkpoint | 动态性 | 实现实际加速的条件 | 主要质量风险 |
|---|---|---:|---|---|---|
| ADAPTOR [[1]](#ref-1) | temporal token | 否/可选调参 | 局部运动自适应 | selector 之后实际缩短序列 | 小物体、快运动、条件坐标错位 |
| Sparse Video-Gen [[2]](#ref-2) | spatial/temporal head 的连边 | 否 | 在线分类 | layout transform + custom kernel | pattern 错分和长距交互丢失 |
| Sparse VideoGen2 [[3]](#ref-3) | 语义关键 token 对 | 否 | 在线聚类与 top-p | permutation + dynamic block kernel | 聚类误差、重排开销 |
| AdaSpa [[4]](#ref-4) | head-adaptive block | 否 | input/layer/head/step 自适应 | fused search + LSE reuse | search step 与 density 对新 prompt 的适应性下降 |
| VORTA [[5]](#ref-5) | 多种 sparse mixer | 否 | 按 layer/step 路由 | 路由开销与各算子 kernel | 路由不当损害全局上下文 |
| VSA [[6]](#ref-6) | tile 内 critical tokens | 是 | 通过训练学习 | 可微 block kernel | 预训练成本与对特定骨干的依赖 |
| RAPID [[7]](#ref-7) | block mask | 否 | 跨 step 适配 | 通过 mask 复用降低搜索开销 | 跨步复用的 pattern 不再适用 |
| BLADE [[8]](#ref-8) | block attention + NFE | 是 | 内容自适应 | sparsity-aware distillation | 稀疏与少步近似共同造成质量损失 |
| DSA [[9]](#ref-9) | 分布式 sparse block | 否/实现绑定 | 动态调度 | 负载均衡与通信重叠 | straggler、跨卡重分布 |
| LinVideo [[12]](#ref-12) | softmax 层转 linear attention | 是（后训练） | 选层转换 | 线性 attention kernel | 复杂运动和全局依赖表达不足 |

## 5. 已形成的共识与尚未解决的争议

### 5.1 共识

1. **稀疏模式具有输入和层次依赖性。** prompt、layer、head、视频长度和 denoising step 均可能改变重要连接的分布。
2. **实际加速取决于 algorithm、layout 与 kernel 的协同。** 非连续的理想 mask 可能比密度稍高但便于 block 化执行的 mask 更慢。
3. **不同去噪阶段需要不同的计算预算。** 语义形成、稳定生成和细节恢复阶段对 attention 的需求不同，固定 density 通常难以兼顾各阶段。
4. **视频质量需要多维评估。** 除平均图像质量外，还应考察小物体、遮挡、快速运动、镜头变化和长程身份一致性等典型失效现象。

### 5.2 争议

- **近似误差与生成质量。** 相对于 dense output 的 PSNR 可用于检测近似误差，但不能替代感知质量、文本遵循和跨 seed 多样性评测。
- **Training-free 方法的迁移性。** 这类方法无需重新训练模型，但仍会受到特定 attention topology、RoPE、mask 和 kernel 的限制，其跨模型迁移能力尚无一致结论。
- **线性 attention 与稀疏 attention 的关系。** 线性化具有较明确的复杂度优势，但不同层的可替换性和表达能力并不均匀。现有研究更倾向于将其与 sparse/local/global attention 混合使用。
- **多种加速方法的组合。** 稀疏化会改变缓存相似性、量化分布和多卡负载，因此组合收益需要通过逐项消融和端到端实验确定。

## 6. 评测原则与报告规范

为保证实验可比性，应固定或完整报告以下设置：

```text
checkpoint / prompt set / seeds / sampler / NFE / CFG
resolution / frames / FPS / batch / precision
GPU / interconnect / software / compile / warm-up
token count before and after / density / block size
selector / clustering / permutation / gather-scatter latency
attention-kernel / DiT-forward / DiT-only / end-to-end latency
peak VRAM / communication / per-device load balance
frame quality / motion / text / consistency / diversity / failures
```

评测结果可分为三个层次：

1. **近似保真度**：在相同提示词和随机种子下，相对于密集计算基线的 PSNR、LPIPS、SSIM 或特征差异；
2. **生成质量**：文本、美学、运动、主体/背景一致性、多样性和人评；
3. **系统效率**：计入选择操作的预热后 p50/p95、峰值显存、吞吐、通信和单视频能耗。

若仅报告 attention kernel 的加速比，应明确该结果不包含 MLP、VAE、text encoder、聚类和布局转换的耗时。若实验比较的是 50-step dense baseline 与 4-step sparse student，则所得结果反映的是多项技术的组合收益。

## 7. 研究空白与未来方向

1. **由稀疏率转向误差预算。** 后续 selector 可进一步预测删除特定 token/edge 对运动、文本遵循和长程一致性的边际影响。
2. **面向控制条件的 token 缩减。** 轨迹、视差、姿态、mask 和音频等条件可能需要分别设置不可合并的 token 集合。
3. **统一的动态布局接口。** 现有方法通常依赖特定的 Triton/CUDA 算子和分块大小，迁移到不同 GPU 或模型骨干时仍有较高成本。
4. **稀疏化与 cache、quantization、parallelism 的联合校准。** 后续研究需在统一质量约束下考察不同技术的组合 Pareto 前沿，而非分别优化单个模块。
5. **可重复的长视频压力测试。** 应公开镜头切换、镜头环绕、多主体交互、小物体重现和跨分钟身份一致性等测试的退化曲线，并减少对精选演示样例的依赖。

## 8. 推荐阅读顺序

1. 先读 ADAPTOR，理解减 token 与减 attention edge 的区别 [[1]](#ref-1)。
2. 再对照 Sparse Video-Gen、AdaSpa 和 Sparse VideoGen2，比较 pattern discovery、精确搜索、语义重排与 kernel 设计 [[2]](#ref-2) [[3]](#ref-3) [[4]](#ref-4)。
3. 阅读 VORTA 和 VMoBA，理解路由与混合 sparse topology [[5]](#ref-5) [[10]](#ref-10)。
4. 对照 VSA 与 RAPID，比较训练所得稀疏模式和跨 denoising step 稀疏复用 [[6]](#ref-6) [[7]](#ref-7)。
5. 最后阅读 BLADE、QuantSparse、LinVideo 和 DSA，了解联合压缩、线性化与分布式执行 [[8]](#ref-8) [[9]](#ref-9) [[12]](#ref-12) [[13]](#ref-13)。

返回[压缩与推理加速综述导航](../inference-acceleration.md)。有关 GPU kernel、多卡计算与 serving 的讨论见[系统与部署综述](systems.md)。

## 参考文献

<a id="ref-1"></a>[1] [ADAPTOR: Adaptive Token Reduction for Video Diffusion Transformers](https://openaccess.thecvf.com/content/CVPR2025W/EDGE/papers/Peruzzo_ADAPTOR_Adaptive_Token_Reduction_for_Video_Diffusion_Transformers_CVPRW_2025_paper.pdf). Elia Peruzzo et al. CVPR Workshops. 2025.

<a id="ref-2"></a>[2] [Sparse Video-Gen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity](https://proceedings.mlr.press/v267/xi25c.html). Haocheng Xi et al. ICML. 2025.

<a id="ref-3"></a>[3] [Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation](https://papers.neurips.cc/paper_files/paper/2025/hash/8bd5148caced2d73cea7b6961a874a49-Abstract-Conference.html). Shuo Yang et al. NeurIPS. 2025.

<a id="ref-4"></a>[4] [Training-free and Adaptive Sparse Attention for Efficient Long Video Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Training-free_and_Adaptive_Sparse_Attention_for_Efficient_Long_Video_Generation_ICCV_2025_paper.html). Yifei Xia et al. ICCV. 2025.

<a id="ref-5"></a>[5] [VORTA: Efficient Video Diffusion via Routing Sparse Attention](https://proceedings.neurips.cc/paper_files/paper/2025/file/0badcb4e95306df76a719409155e46e8-Paper-Conference.pdf). Wenhao Sun et al. NeurIPS. 2025.

<a id="ref-6"></a>[6] [Faster Video Diffusion with Trainable Sparse Attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html). Peiyuan Zhang et al. NeurIPS. 2025.

<a id="ref-7"></a>[7] [RAPID: Reusing Attention Sparsity with Inter-step Adaptation for Efficient Video Diffusion](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_RAPID_Reusing_Attention_Sparsity_with_Inter-step_Adaptation_for_Efficient_Video_CVPR_2026_paper.html). Shangran Lin, Lu Lu, Jian Chen, Qiang Liu. CVPR. 2026.

<a id="ref-8"></a>[8] [BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5bcb807ae43ad0851a6ba6162a866404-Abstract-Conference.html). Youping Gu et al. ICLR. 2026.

<a id="ref-9"></a>[9] [DSA: Efficient Inference for Video Generation Models via Distributed Sparse Attention](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c3728248f3c627d1f16ca5726cdf83f5-Abstract-Conference.html). Shenggui Li et al. ICLR. 2026.

<a id="ref-10"></a>[10] [VMoBA: Mixture-of-Block Attention for Video Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d6c4014ff8d95025aa35d831c0f81faa-Abstract-Conference.html). Jianzong Wu, Liang Hou, Haotian Yang, Ye Tian, Pengfei Wan, Di Zhang, Yunhai Tong. ICLR. 2026.

<a id="ref-11"></a>[11] [SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41b93c59da0d0f835907fd661d419db2-Abstract-Conference.html). Junsong Chen et al. ICLR. 2026.

<a id="ref-12"></a>[12] [LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_LinVideo_A_Post-Training_Framework_towards_On_Attention_in_Efficient_Video_CVPR_2026_paper.html). Yushi Huang et al. CVPR. 2026.

<a id="ref-13"></a>[13] [QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html). Weilun Feng et al. ICLR. 2026.
