# 视频扩散模型剪枝与轻量化综述：从权重稀疏到结构化学生模型

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本文讨论视频扩散模型和 flow-matching 生成器中的模型剪枝与轻量化。资料以正式会议论文、作者预印本和官方项目页为主。文中的参数量、FLOPs、延迟和质量数据均沿用原论文的模型、任务、输出规格、硬件及计时设置；由于尚未在统一环境中复现，本文不对不同工作的速度进行直接排名。

## 1. 研究范围与基本概念

模型剪枝通过移除预训练生成器中的部分参数或结构，降低模型的存储与计算开销。对于视频生成模型，需要区分以下四类操作：

| 对象 | 操作 | checkpoint 是否变小 | 是否可直接降低 dense kernel 开销 | 本文处理方式 |
|---|---|---:|---:|---|
| 单个权重 | 置零、掩码或稀疏重训练 | 仅在稀疏格式中 | 通常不能 | 作为非结构权重剪枝讨论 |
| channel / head / block / layer | 物理删除宽度或深度 | 是 | 较可能，但需重建 tensor shape | 重点讨论 |
| 运行时 token | 按输入合并或丢弃 latent token | 通常否 | 取决于 gather/scatter 与变长 kernel | 仅作边界说明 |
| attention edge | 删除 token 间连边或 attention value | 通常否 | 依赖 block-sparse kernel | 仅作边界说明 |

后两类方法改变单次前向传播中的序列或连接关系，不宜与参数剪枝合并统计。相关内容见 [Video DiT 的 token 与 sparse-attention 讨论](../video-dit-backbones.md)以及[压缩与推理加速综述导航](../inference-acceleration.md)。F³-Pruning 虽以 pruning 命名，但其主要操作对象是 temporal-attention 权重或连接，因此处于模型剪枝与 attention sparsity 的交界处 [[1]](#ref-1)。

若教师模型的参数量为 $P_T$、学生模型的**物理参数量**为 $P_S$，可将参数压缩率定义为

```math
r_P=1-\frac{P_S}{P_T}.
```

不过，`mask 后非零参数比例`、序列化文件大小、dense tensor shape 和运行时 resident memory 分别反映不同性质。只有实际缩小矩阵维度、降低网络深度，或采用硬件支持的稀疏 kernel，参数压缩才可能稳定转化为时延收益。

## 2. 方法演进：从时间冗余分析到结构化压缩与部署

### 2.1 2024：时间冗余的初步研究

F³-Pruning 观察到，CogVideo 与 Tune-A-Video 中 temporal attention 的 aggregate attention score 会随网络层次或去噪进程降低，并据此在无需再训练的条件下移除得分较低的 temporal-attention 权重 [[1]](#ref-1)。该工作较早将视频模型中的时间冗余用作压缩依据，但其操作对象更接近运行时 attention edge，而非现代 Video DiT 中的 channel、head 或完整 block。论文在 CogVideo 的 50% temporal-attention 设置下报告约 44% 的 FLOPs 降幅和约 1.35× 的加速；Tune-A-Video 表 2 中的时间则由 45.08 s 降至 43.83 s。这组结果也表明，FLOPs 降幅与实际运行时间的改善并不具有固定对应关系。

### 2.2 2024—2025：内容与运动的差异化恢复

ICMD/VDMini 以 U-Net 视频扩散模型为研究对象。作者实验显示，浅层对单帧内容的影响相对较大，深层则更多关系到整段视频的运动一致性。基于这一观察，该方法优先裁减浅层冗余 block，再通过 Individual Content Distillation 和 Multi-frame Content Adversarial loss 恢复学生模型 [[2]](#ref-2)。论文在 SF-V I2V 和 T2V-Turbo-v2 T2V 的实验设置下，分别报告平均约 2.5× 和 1.4× 的加速。上述结论仅适用于所测试的 backbone 与任务，尚不足以说明浅层结构在其他模型中也可安全删除。

这一研究形成了结构化视频剪枝中常见的流程：先分析 layer/block sensitivity，据此构建较小的学生模型，再分别利用 frame-level appearance 与 multi-frame motion 信号进行性能恢复。在此过程中，剪枝与知识蒸馏承担不同作用：前者确定学生模型的结构，后者用于补偿容量下降造成的性能损失。

### 2.3 2025：面向移动端的系统级协同优化

MobileVD 以 Stable Video Diffusion 的时空 U-Net 为基础，综合采用较低的 latent 分辨率、时间多尺度、channel funneling、temporal transformer/residual block 剪枝和一步 adversarial fine-tuning [[3]](#ref-3)。作者报告该系统可在 Xiaomi 14 Pro 上用约 1.7 s 生成 `14 × 512 × 256` latent clip。该结果来自多项技术的共同作用，不能单独归因于剪枝。

V.I.P. 重点研究剪枝后的性能恢复：该方法在 VideoCrafter2 中删除完整 U-Net block，在 AnimateDiff 中仅剪除 motion module，并通过分阶段数据筛选以及结合 DPO/SFT 的 ReDPO，优先恢复退化最明显的属性 [[4]](#ref-4)。作者在两种模型上分别报告 36.2% 和 67.5% 的参数缩减，同时在若干指标上维持或超过完整模型。该结果并不意味着小模型通常优于教师模型，而是说明单纯采用 feature matching 或 SFT 可能使容量有限的学生模型牺牲原本表现较好的维度；偏好恢复本身也可能受到 reward model 和数据筛选偏差的影响。

Mobile Video DiT 将结构化剪枝扩展到 DiT 的 block、attention head 和 FFN channel，并利用知识蒸馏指导 sensitivity-aware pruning [[5]](#ref-5)。截至本文资料截止日期，该工作的主要依据仍是作者预印本与项目页。论文报告的约 15 FPS 移动端结果还结合了高压缩 VAE、四步 adversarial distillation 和移动端实现，因而不能视为 attention-head 剪枝的独立收益。

### 2.4 2026：模型规模与采样步数的联合优化

Neodragon 面向移动端 MMDiT 删除完整 block，并通过两阶段 distillation 恢复性能；同时，该系统还压缩 text encoder、替换 VAE decoder 并减少 NFE。作者在 Qualcomm Hexagon NPU 上报告 `49 帧、640×1024、约 6.7 s` 的端到端结果 [[6]](#ref-6)。这一结果表明，部署成本不仅取决于 denoiser 的参数量；完成剪枝后，文本编码器、VAE、首帧生成器和内存搬运均可能成为新的瓶颈。

FastLightGen 面向 HunyuanVideo-ATI2V 和 WanX-TI2V，在同一训练框架中结合 layer importance、dynamic probabilistic pruning 与 few-step distribution matching [[7]](#ref-7)。作者给出的代表性配置为 4-step 与 30% parameter pruning，即保留约 70% 的参数。论文图 1 所示的约 35.71× 是相对于 50-step、启用 CFG 的 baseline 得到的理论计算结果，并非统一设置下的端到端实测加速比。

PARE 在固定学生模型之外进一步引入 width pruning 和 input/timestep-adaptive depth routing：根据不同 head 的空间或时间作用保留 motion-critical head，并由轻量 router 决定各去噪步需要执行的 block [[8]](#ref-8)。这一 2026 年预印本展示了动态结构的潜力，但动态执行会引入控制流、batch divergence 和 router 开销，其收益仍需在实际 serving 设置中验证。

## 3. 剪枝粒度与可部署性

### 3.1 非结构权重剪枝

非结构剪枝为标量权重设置 mask，具有粒度细、可在保持 tensor shape 不变的情况下获得较高 sparsity 等特点。然而，dense GEMM 仍会读取零值并参与乘法。若实现中没有采用压缩存储、索引调度以及相应的 2:4、block-sparse 或其他硬件路径，即使 90% 的权重被置零，也未必能够降低显存占用或推理时延。

因此，实验应分别报告 `dense parameters`、`non-zero parameters`、稀疏格式字节数、metadata/indices、实际 kernel density，以及单层和整网时延。随机稀疏通常受访存与索引开销限制；规则化的 N:M 或 block sparsity 更便于部署，但相应缩小了子网络的搜索空间。

### 3.2 Channel 与 FFN-width 剪枝

Channel/FFN 剪枝直接降低投影维度，从而缩小 dense matrix。实际部署时，还需同步压缩后续的 norm、residual 和 QKV/MLP 权重，并使维度符合设备适用的 tile 倍数；否则，参数量虽然下降，kernel 执行效率却可能受到影响。MobileVD 的 channel funneling 和 Mobile Video DiT 的 FFN-channel 裁减分别代表 U-Net 与 DiT 中的相关实践 [[3]](#ref-3) [[5]](#ref-5)。

### 3.3 Attention-head 剪枝

评估 head importance 时，需要区分 head 数量和总 embedding width。若只对部分 head 设置 mask 而保留完整的 QKV 投影，参数量与 GEMM 开销可能不会下降；若进行物理压缩，则需重新配置 projection、reshape 和 fused-attention。视频模型中的不同 head 还可能分别侧重空间或时间信息。PARE 的实验表明，motion-aware scoring 可能比仅依据平均 attention magnitude 更合适，但这种功能分工尚未在不同模型间得到系统验证 [[8]](#ref-8)。

### 3.4 Block / layer 剪枝

删除完整 residual block 或 transformer layer 能够直接降低网络深度，是 ICMD、V.I.P.、NeoDragon 和 FastLightGen 共同采用的思路 [[2]](#ref-2) [[4]](#ref-4) [[6]](#ref-6) [[7]](#ref-7)。但完整 block 的删除会同时移除 attention、MLP、norm 和条件注入，误差还可能在后续 denoising step 中累积。因此，中等及以上比例的结构剪枝通常需要配合 fine-tuning、feature matching、consistency、adversarial 或 preference recovery。

## 4. 模型剪枝与尺寸蒸馏（Size Distillation）的关系

参数量较小的学生模型并不一定由剪枝获得。根据学生模型的构建与训练方式，可区分为三类：

- **Prune-only**：从 teacher 中删除结构，不进行恢复训练，主要用于测量预训练模型的结构冗余。
- **Prune + recovery distillation**：student 结构由剪枝得到，并通过蒸馏目标恢复性能；ICMD、V.I.P. 和 NeoDragon 属于此类。
- **Designed-small / architecture distillation**：独立设计小型网络，再利用 teacher 进行训练；这类方法属于模型蒸馏或轻量架构设计，不应将其全部收益归于 pruning。

Step distillation 主要减少 denoiser 的调用次数 $N_{\mathrm{NFE}}$，结构剪枝则降低单次调用的成本。联合使用两者时，应至少比较 `full/many-step`、`pruned/many-step`、`full/few-step` 和 `pruned/few-step` 四种设置。若缺少这一消融，只能报告组合方法的 Pareto 结果，无法确定各模块对总加速比的独立贡献。

## 5. 视频生成模型中的层敏感性（Layer Sensitivity）

ICMD 在特定 U-Net 和任务上观察到 shallow-content/deep-motion 的层级差异 [[2]](#ref-2)；FastLightGen 在两个大规模 DiT 上发现首尾层较为重要，而部分中间层存在较多冗余 [[7]](#ref-7)；PARE 则将 sensitivity 建模为随 timestep 和输入变化的动态函数 [[8]](#ref-8)。这些观察为研究层级功能提供了依据，但尚不足以形成适用于所有视频生成模型的统一结论。

造成差异的因素包括 U-Net skip connection 与 DiT residual topology、独立 temporal module 与 joint space-time attention、T2V/I2V 条件、noise/flow timestep、视频长度、运动幅度、camera motion，以及 teacher 是否经过 step distillation。较为可靠的 sensitivity study 应在 prompt × seed × timestep × length 的组合上对单个结构进行干预，并分别评估 appearance、motion、condition 和 diversity。

V.I.P. 还指出一项评测问题：较高的 dynamic degree 可能掩盖 consistency 的下降，而静态 prompt 本身也不要求明显运动 [[4]](#ref-4)。VBench 的多维指标适合分析不同能力，但不宜仅保留单一总分 [[9]](#ref-9)；I3D-FVD 存在 content bias，也不能单独证明时间建模能力得到保留 [[10]](#ref-10)。

## 6. 代表工作比较

| 工作 | 发表状态 | Backbone / 任务 | 主要压缩对象 | 恢复机制 | 实验结果及适用范围 |
|---|---|---|---|---|---|
| F³-Pruning [[1]](#ref-1) | AAAI 2024 | CogVideo；Tune-A-Video | temporal-attention weight/edge | 无训练 | CogVideo 约 1.35×；Tune-A-Video 45.08→43.83 s，不是现代 DiT 参数剪枝结论 |
| ICMD/VDMini [[2]](#ref-2) | ACM MM 2025；2024 预印本 | SF-V I2V；T2V-Turbo-v2 | U-Net shallow block | frame feature + multi-frame adversarial | 作者分别报告约 2.5×/1.4×；layer 功能结论限于所测模型 |
| MobileVD [[3]](#ref-3) | ICCV 2025 | SVD I2V U-Net | channel + temporal block | adversarial one-step fine-tuning | 手机 1.7 s 同时受分辨率、时间多尺度、一步化与 kernel 影响 |
| V.I.P. [[4]](#ref-4) | ICCV 2025 | VideoCrafter2；AnimateDiff | U-Net block；motion module | staged curation + ReDPO | 36.2%/67.5% 参数减少；不同剪枝对象不能直接横比 |
| Mobile Video DiT [[5]](#ref-5) | 2025 预印本 | 移动端 Video DiT | block + head + FFN channel | KD + adversarial four-step | 约 15 FPS 来自 VAE、剪枝、少步生成与移动端实现的共同作用 |
| Neodragon [[6]](#ref-6) | ICLR 2026 | 移动端 MMDiT T2V | denoiser block，另压 text/VAE | 两阶段 feature/flow recovery + step distill | 6.7 s 为完整系统的端到端结果 |
| FastLightGen [[7]](#ref-7) | CVPR 2026 | HunyuanVideo/WanX TI2V | DiT layer + sampling step | probabilistic pruning + distribution matching | 4-step + 30% pruning；约 35.71× 为理论组合估计 |
| PARE [[8]](#ref-8) | 2026 预印本 | Wan2.1-14B T2V/I2V | head width + routed depth | progressive distillation + router | 动态执行需计 router、负载分歧与真实 batch latency |

## 7. 评测原则与报告规范

比较不同剪枝设置时，至少应固定 teacher/student hash、任务、prompt 集、seed、帧数、分辨率、FPS、sampler、名义 step、实际 NFE、CFG、dtype、batch、硬件、编译器和 kernel 版本。若 text encoder、VAE、offload 或后处理发生变化，应将其作为系统配置差异单独说明。

计算效率至少包括以下四类指标：

1. 参数：dense / non-zero / trainable / serialized bytes；
2. 单步：MACs/FLOPs、实际执行 block/head/channel、forward latency；
3. 全程：NFE、denoiser 总时间、text/VAE/I/O、E2E latency 与峰值显存；
4. 系统：cold/warm、p50/p95、吞吐、功耗/能耗，以及 baseline OOM 时的 capacity 收益。

生成质量至少应分别考察 appearance、text/image condition、motion amplitude、temporal consistency、camera motion、长程 identity、diversity 和 prompt-correct stillness。自动指标需采用相同的版本、帧采样方式和样本数量，并辅以盲法人评；单一 VBench 总分或 FVD 无法反映所有退化类型 [[9]](#ref-9) [[10]](#ref-10)。

基本消融应比较 `teacher → prune-only → prune+recovery`。若同时采用少步生成，还需加入完整模型与剪枝模型在多步和少步设置下的四组对照。进一步加入参数量相当、从头设计的小模型作为基线，有助于区分剪枝所利用的预训练冗余与重新训练小型模型带来的收益。

## 8. 典型失效现象

- **运动塌缩**：画面清晰但主体不动、周期动作消失，或 prompt 要求的相机运动被削弱。
- **时间抖动**：单帧指标稳定，但纹理、手部、身份或背景在帧间跳变。
- **动态指标偏差**：无意义的大幅运动可能提高 dynamic degree，同时损害动作正确性与 consistency。
- **条件退化**：容量受限的 student 可能优先牺牲文字、参考图、轨迹或局部控制等条件信息。
- **稀有模式遗失**：平均指标保持，复杂组合、快速遮挡、小物体与长尾动作从分布中消失。
- **长视频误差累积**：模型在短 clip 上表现正常，但误差会随帧数或滚动生成过程持续增长。
- **Kernel 失配**：非结构零值缺少相应的稀疏 kernel，或 channel/head 维度破坏 tile 对齐，使理论 FLOPs 降幅无法转化为实际加速。
- **恢复过拟合**：student 只适配蒸馏 prompt、reward model 或 teacher 常见模式，多样性下降。

## 9. 共识、争议与证据强度

现有研究较为一致地表明：与任意非结构稀疏相比，结构化删除 channel、block 或 layer 更容易在 dense hardware 上取得实际收益；中高比例的压缩通常需要恢复训练；视频质量评估应分别考察 appearance 与 motion；模型尺寸压缩和少步蒸馏的贡献需要独立分析。这些认识已见于多种 U-Net 和 DiT，但相关证据仍主要来自各方法论文，尚缺少统一 benchmark 下的系统比较。

仍有四个问题缺乏明确结论。第一，layer/head 的运动功能能否跨架构、timestep 和任务保持稳定；第二，training-free saliency 能否替代成本较高的干预式 sensitivity 分析；第三，feature/score distillation 与 preference recovery 何者更有利于保持分布覆盖；第四，理论稀疏度、单个 kernel、单次 forward、E2E latency 和 energy 中，哪些指标最能反映部署性能。现有证据尚不足以回答这些问题。

现有证据的分布并不均衡。2024—2026 年的相关方法主要集中于少数开放 U-Net/DiT、短 clip 和作者自定的硬件环境。F³-Pruning、ICMD、MobileVD、V.I.P.、NeoDragon 和 FastLightGen 已正式发表；截至 2026-09-02，PARE 和 Mobile Video DiT 仍主要依据作者预印本。正式发表有助于资料追溯，但其中的速度数据并不等同于独立复现结果。

## 10. 研究空白与未来方向

1. **建立统一的运动—外观敏感性图谱**：在 U-Net、联合时空 Video DiT、T2V/I2V、不同时间步和视频长度上，系统发布逐网络块、注意力头和通道的干预曲线。
2. **面向底层算子的剪枝目标**：在结构搜索阶段考虑计算块对齐、N:M 或分块稀疏模式、显存带宽和通信开销，使所得结构能够直接适配目标硬件。
3. **条件控制与长尾样本评测**：围绕组合提示词、小物体、快速运动、镜头切换、参考图身份和局部控制构建困难样例集，避免平均分数掩盖局部退化。
4. **分布覆盖评测**：除教师样本匹配外，还应测量精确率/召回率、不同随机种子下的多样性和稀有动作召回，以识别恢复蒸馏导致的模式丢失。
5. **不同压缩方法的相互作用**：系统研究 pruning 后的量化校准漂移、cache 误差累积，以及 few-step student 中不同层功能的变化。
6. **动态深度模型的在线服务评估**：报告路由开销、批内执行分歧、尾延迟和多请求调度结果，以判断输入自适应方法是否优于固定学生模型。
7. **全生命周期成本**：将敏感性扫描、恢复训练、模型转换和部署能耗纳入成本分析，而不仅考察单次推理。

## 11. 推荐阅读顺序

1. 先读 F³-Pruning，理解 temporal redundancy 的早期观察，同时识别它与参数剪枝的边界 [[1]](#ref-1)。
2. 再读 ICMD/VDMini，理解结构敏感性与 appearance/motion 双恢复目标 [[2]](#ref-2)。
3. 对照 MobileVD 与 V.I.P.，理解移动端系统级设计和偏好式恢复的差异 [[3]](#ref-3) [[4]](#ref-4)。
4. 阅读 Mobile Video DiT 和 Neodragon，考察 channel/head/block 剪枝与 NPU 部署 [[5]](#ref-5) [[6]](#ref-6)。
5. 最后阅读 FastLightGen 与 PARE，了解 size×step 联合优化和动态结构的最新进展 [[7]](#ref-7) [[8]](#ref-8)。
6. 评测时同时阅读 VBench 与 FVD content-bias 分析，不以总分替代失败诊断 [[9]](#ref-9) [[10]](#ref-10)。

## 参考文献

<a id="ref-1"></a>[1] Sitong Su, Jianzhi Liu, Lianli Gao, Jingkuan Song. [F³-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis](https://doi.org/10.1609/aaai.v38i5.28300). AAAI, 2024.

<a id="ref-2"></a>[2] Yiming Wu, Huan Wang, Zhenghao Chen, Dong Xu. [Individual Content and Motion Dynamics Preserved Pruning for Video Diffusion Models](https://doi.org/10.1145/3746027.3755081). ACM Multimedia, 2025. [作者预印本](https://arxiv.org/abs/2411.18375)首次公开于 2024。

<a id="ref-3"></a>[3] Haitam Ben Yahia, Denis Korzhenkov, Ioannis Lelekas, Amir Ghodrati, Amirhossein Habibian. [Mobile Video Diffusion](https://openaccess.thecvf.com/content/ICCV2025/html/Yahia_Mobile_Video_Diffusion_ICCV_2025_paper.html). ICCV, 2025.

<a id="ref-4"></a>[4] Jisoo Kim, Wooseok Seo, Junwan Kim, Seungho Park, Sooyeon Park, Youngjae Yu. [V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_V.I.P.__Iterative_Online_Preference_Distillation_for_Efficient_Video_Diffusion_ICCV_2025_paper.html). ICCV, 2025.

<a id="ref-5"></a>[5] Yushu Wu et al. [Taming Diffusion Transformer for Efficient Mobile Video Generation in Seconds](https://arxiv.org/abs/2507.13343). arXiv preprint, 2025.

<a id="ref-6"></a>[6] Animesh Karnewar et al. [Neodragon: Mobile Video Generation using Diffusion Transformer](https://openreview.net/forum?id=XBzIhhwv8d). ICLR, 2026.

<a id="ref-7"></a>[7] Shitong Shao, Yufei Gu, Zeke Xie. [FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_FastLightGen_Fast_and_Light_Video_Generation_with_Fewer_Steps_and_CVPR_2026_paper.html). CVPR, 2026.

<a id="ref-8"></a>[8] Yutong Wang, Yunke Wang, Tianfan Xue, Yu Qiao, Yaohui Wang, Xinyuan Chen, Chang Xu. [PARE: Pruning and Adaptive Routing for Efficient Video Generation](https://arxiv.org/abs/2605.27336). arXiv preprint, 2026.

<a id="ref-9"></a>[9] Ziqi Huang et al. [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html). CVPR, 2024.

<a id="ref-10"></a>[10] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, Jia-Bin Huang. [On the Content Bias in Fréchet Video Distance](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html). CVPR, 2024.
