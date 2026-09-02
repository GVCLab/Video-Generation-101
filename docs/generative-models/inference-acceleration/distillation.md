# 视频扩散模型蒸馏与少步生成综述

> **文献更新至 2026-09-02（Asia/Shanghai）。** 本文以视频扩散模型和流模型为主要讨论对象；图像生成工作仅用于说明 Progressive Distillation、Consistency 与 DMD 等基础目标。文献以正式会议论文为主，必要时补充作者预印本。文中引用的步数、速度和质量数据均以原论文所采用的模型、分辨率、帧数、硬件、数值精度及计时范围为限，本仓库尚未在统一环境中复现这些结果。

视频扩散蒸馏通常被概括为“学生模型模仿教师模型”，但不同方法所压缩的对象并不相同，包括数值求解轨迹、从噪声到数据的有限步映射、最终生成分布、视频时序上的因果分解，以及模型本身的层数与宽度。不同路线都可能得到 1–4 步生成器，其训练信号、典型失效情形和部署成本却有明显差异，因而不能仅按最终 NFE 将其视为同一类方法。

## 1. 综述范围与文献依据

本综述围绕四个问题展开：少步学生模型从教师模型学习什么；视频任务相较图像任务增加了哪些运动、长时和因果约束；步数蒸馏与模型规模蒸馏如何结合；不同方法之间如何进行公平比较。

讨论范围包括作为基础对照的免训练求解器、轨迹蒸馏与一致性蒸馏、分布匹配、对抗蒸馏与奖励引导蒸馏、因果或视频感知的学生模型，以及步数与模型规模的联合蒸馏。量化、缓存和稀疏化仅在影响蒸馏效果归因时涉及，具体方法见相应专题综述。

| 等级 | 文献类型 | 可据此讨论 | 尚不能据此判断 |
|---|---|---|---|
| A | 正式会议论文 | 方法定义、公开实验及原文设置下的结果 | 跨论文的通用排名、独立复现结果 |
| B | 作者发布的 arXiv 论文或技术报告 | 新方法、作者提供的消融实验和局限性 | 同行评审结论、脱离实验条件的推广 |
| C | 官方项目页或实现 | checkpoint、配置与实现可用性 | 仅凭 README 中的描述判断质量或速度 |

截至上述日期，直接针对视频的研究主要发表于 2024–2026 年，实验多围绕少数开源视频教师模型和 VBench 类指标展开，因此结论对其他模型与任务的适用性仍需检验。

## 2. 基本概念与计算口径

若第 $i$ 个采样节点实际调用 denoiser $m_i$ 次，采样主成本可写为：

```math
C_{\mathrm{sample}}\approx\sum_{i=1}^{S}m_i\,C_{\theta,i}.
```

因此，名义采样步数 $S$、完整网络调用次数 NFE、CFG 所需的条件与无条件前向次数，以及每次前向所用模型的规模，都应分别记录。

| 概念 | 是否训练新权重 | 直接影响 | 通常不直接影响 |
|---|---:|---|---|
| Solver / schedule | 否 | 时间节点、积分误差与 NFE | 常驻权重、单次前向成本 |
| Step/guidance distillation | 是 | 少步映射、生成分布或 CFG 前向次数 | 学生模型的层数、宽度 |
| Size distillation / pruning | 是，或需恢复训练 | 参数量、层数、通道数、模块数 | 所需采样步数 |
| Causal distillation | 是 | 视频时序分解方式、历史状态接口 | 长时漂移或系统尾延迟 |

DDIM 与 DPM-Solver 可以直接复用已有 checkpoint，适合作为同一模型下质量与 NFE 关系的基础对照 [[1]](#ref-1) [[2]](#ref-2)。若求解器在目标 NFE 下仍无法满足质量或时延要求，再考虑通过训练新 checkpoint 学习跨越较大时间区间的生成映射。

步数蒸馏与模型规模蒸馏作用于相对独立的两个方面：前者减少模型调用次数，后者降低单次调用的计算量。**Step-size conditioning** 指以积分步长或时间区间作为网络条件；FastLightGen 所讨论的 **step-and-size co-distillation** 则同时压缩采样步数和模型规模。二者名称相近，但研究问题不同 [[19]](#ref-19) [[20]](#ref-20)。

## 3. 方法的发展脉络

### 3.1 2022–2023：从合并教师模型的采样步到一致性映射

Progressive Distillation 以学生模型的一次更新逼近教师模型的两次确定性更新，并逐轮将 $N$ 个采样步压缩为 $N/2$ 步 [[3]](#ref-3)。Imagen Video 随后将该方法用于视频级联系统。早期的视频少步生成通常需要分别蒸馏基础模型和超分辨率模型，而非直接蒸馏单一的端到端生成器 [[4]](#ref-4)。

Consistency Models 改变了监督对象：同一 probability-flow ODE 轨迹上的不同噪声点应映射到相同终点；模型既可以由教师模型蒸馏得到，也可以独立训练 [[5]](#ref-5)。由此，一步或少步生成不再只是求解器的选择，而成为模型通过训练获得的能力。不过，其在低 NFE 下的局部细节、轨迹误差和分布覆盖仍需通过视频实验验证。

### 3.2 2024：图像目标迁移到视频，并出现视频特有修正

DMD 不要求逐样本复制教师模型的轨迹，而是利用 target score 与 fake/student score 之差，使学生模型的生成分布接近数据分布；DMD2 在此基础上加入 on-policy 生成数据、双时间尺度更新和 GAN loss [[6]](#ref-6) [[7]](#ref-7)。两项工作为后续视频分布蒸馏提供了重要基础，但其原始实验主要面向图像，尚不足以说明视频运动能够得到同等程度的保持。

T2V-Turbo 将多种可微奖励直接引入视频一致性蒸馏，并在原文的实验设置下比较了 4 步模型与 50 步 DDIM 教师模型 [[8]](#ref-8)。Motion Consistency Model 则将运动蒸馏与单帧外观增强分开处理，并混合来自不同质量域的轨迹，以缓解图像蒸馏目标直接迁移到视频时外观监督与运动监督之间的冲突 [[9]](#ref-9)。

AnimateDiff-Lightning 采用 progressive adversarial diffusion distillation，是直接面向视频、但目前仍以作者技术报告为主要依据的代表性工作 [[10]](#ref-10)。该工作表明，对抗目标可以弥补少步回归在感知质量上的不足；这一结论并不意味着所有 consistency 或 DMD 方法都使用真假判别器。

### 3.3 2025：分布匹配、奖励引导与因果建模

APT 在扩散模型预训练之后，直接利用真实数据进行对抗式后训练，目标是以一步生成高分辨率视频。论文所称的“单次前向、实时生成”仅适用于其给定的两秒视频、分辨率、帧率和硬件设置 [[11]](#ref-11)。

CausVid 将双向、多步教师模型蒸馏为 4 步因果学生模型，并同时引入自回归视频分解和 KV cache [[12]](#ref-12)。因此，论文中的性能来自蒸馏、因果结构和系统复用的共同作用，不能全部归因于蒸馏。其 asymmetric DMD 虽借鉴了 DMD2 的更新节奏，但并未因此采用显式的 GAN loss。

DOLLAR 结合 variational score distillation、consistency distillation 与 latent reward optimization，并将 1 步或 4 步学生模型与 50 步教师模型进行比较 [[13]](#ref-13)。V.I.P. 面向经过剪枝的学生模型开展在线偏好蒸馏，说明小模型的性能恢复未必需要在每个维度上逐点复制教师模型 [[14]](#ref-14)。两类方法都会受到评价器偏差的影响：奖励模型或偏好模型所改善的维度，可能未能覆盖运动多样性等未测能力，也可能掩盖特定失效情形。

### 3.4 2026：规模化、分段映射与联合压缩

rCM 在 continuous-time consistency 中引入 score regularization，并以规模最高达 14B 的图像和视频教师模型研究 1–4 步蒸馏。论文报告的 15–50 倍提升针对扩散采样过程，并非完整生成流程的端到端速度 [[15]](#ref-15)。该方法将 consistency 的 mode-covering 倾向与基于 score 的 reverse divergence 所表现出的 mode-seeking 倾向纳入同一框架，用以研究清晰度与多样性之间的权衡。

TMD 并不让每个少步 transition 都调用一次完整 backbone，而是由主干在 outer transition 中提取语义，再由轻量 flow head 完成多次内部更新 [[16]](#ref-16)。Phased DMD 将 SNR 范围划分为若干子区间，通过 progressive distribution matching 和区间内 score matching，缓解多步 DMD 中训练链路过深、梯度截断和运动减弱等问题 [[17]](#ref-17)。比较这类方法时，应以等价计算量为准，而不能只看名义采样步数。

Adversarial Self-Distillation 以 $n$ 步学生模型和 $(n+1)$ 步自教师模型构造一步因果视频生成器，从而减少对独立大型教师模型的依赖 [[18]](#ref-18)。不过，短视频上的一步因果生成尚不能说明该方法已经解决无限时长流式生成中的状态漂移、镜头切换和交互条件变化。

FastLightGen 联合考虑采样步数与模型规模：先识别并动态剪除不重要的模块，再配置强、弱教师模型与 fake model 进行细粒度分布匹配。论文在 HunyuanVideo 和 Wan 系列模型的实验中重点报告了 4 步生成与 30% 参数剪枝的组合结果 [[19]](#ref-19)。该结果反映的是联合优化后在质量与效率之间的权衡，无法将整体收益分别归因于蒸馏和剪枝。

AnyFlow 将 endpoint consistency 扩展为任意区间 $z_t\rightarrow z_r$ 的 flow-map transition，并通过 on-policy backward simulation，使同一视频模型能够使用不同推理步数，且生成质量有望随步数增加继续改善 [[21]](#ref-21)。截至本文更新日期，该工作仍为作者预印本。其结果提示，在固定 1–4 步设置下表现最优的学生模型，增加测试时采样步数后未必继续受益。

## 4. 四类方法的共同点与差异

### 4.1 Trajectory / consistency：轨迹拟合与分布覆盖

Progressive Distillation、Consistency、MCM、rCM 与 AnyFlow 都利用 trajectory 或 flow-map 信息，但监督粒度各不相同，分别涉及合并相邻教师步、约束共同终点、单独蒸馏运动表示、加入 score regularizer，以及学习任意区间的 transition [[3]](#ref-3) [[5]](#ref-5) [[9]](#ref-9) [[15]](#ref-15) [[21]](#ref-21)。

这类方法通常训练较为稳定，也与 ODE 的数学表述紧密相连，但可能过度拟合多模态条件下的平均行为。评估多样性时，除观察单个随机种子的清晰度外，还应比较不同随机种子所生成的动作、镜头、构图和事件路径。

### 4.2 DMD、对抗学习与奖励引导的差异

DMD 中的 fake score network、GAN 中的真假判别器和固定的 reward model 属于三类不同的反馈机制。DMD 不一定包含二元对抗损失，reward-guided consistency 也不一定联合训练 critic；APT 则明确使用真实数据进行对抗式后训练 [[6]](#ref-6) [[8]](#ref-8) [[11]](#ref-11)。

分布匹配和对抗学习通常有助于提高一步样本的锐度与感知真实性，但也容易出现 mode-seeking、过饱和、颜色偏移或训练不稳定。奖励引导可以直接优化文本对齐或审美指标，同时也需要独立评价和人工检查，以判断指标改善是否伴随 reward hacking，或是否仅发生在部分质量维度。

### 4.3 Causal / video-aware：视频时序带来的额外约束

MCM 处理运动与外观监督之间的冲突；CausVid 与 ASD 改变视频时序上的条件结构；DOLLAR 和 Phased DMD 则直接针对长视频或复杂运动在极少步生成时的退化 [[9]](#ref-9) [[12]](#ref-12) [[13]](#ref-13) [[17]](#ref-17) [[18]](#ref-18)。

因此，视频学生模型至少需要同时保持单帧外观、局部与全局运动、主体和背景的一致性、时序事件的完整性、文本遵循以及不同随机种子之间的多样性。仅使用图像 FID、单帧美学分数或平均 VBench，难以判断具体损失了哪一方面的能力。

### 4.4 Step、step-size 与 size 的联合优化

Shortcut Models 代表“以 desired step size 为输入”的可变步长目标，但正式强证据主要仍是图像 [[20]](#ref-20)。FastLightGen 代表“采样步数与参数规模共同蒸馏”；TMD 则让不同层承担不同频率的更新；AnyFlow 进一步要求同一模型覆盖多个推理预算 [[16]](#ref-16) [[19]](#ref-19) [[21]](#ref-21)。

这些工作将研究重点从追求唯一的最少步数，转向在给定硬件、时延、显存和质量要求下，共同选择完整主干调用次数、轻量更新次数、模型规模与采样步数。

## 5. 代表工作对比

| 工作 | 主要机制 | 新 checkpoint | NFE / 原文实验设置 | 主要质量风险 | 证据 |
|---|---|---:|---|---|---:|
| Progressive Distillation [[3]](#ref-3) | 以一个学生步逼近两个教师步 | 是 | 逐轮 $N\rightarrow N/2$；图像基础实验 | 多轮训练误差累积 | A |
| Consistency Models [[5]](#ref-5) | 同轨迹点映射到共同终点 | 是 | 1-step 或少步 refinement；图像基础实验 | 细节与分布覆盖 | A |
| T2V-Turbo [[8]](#ref-8) | consistency + 混合可微 reward | 是 | 4-step 对 50-step DDIM 教师模型 | 奖励模型偏差、动作覆盖 | A |
| MCM [[9]](#ref-9) | motion consistency + appearance discriminator | 是 | few-step；视频与图像数据联合训练 | 外观与运动目标冲突 | A |
| AnimateDiff-Lightning [[10]](#ref-10) | progressive adversarial distillation | 是 | 原文报告 1/2/4/8-step | 架构依赖、对抗不稳定 | B |
| APT [[11]](#ref-11) | 真实数据 adversarial post-training | 是 | 1 NFE；两秒 720p、24 FPS | mode collapse、训练稳定性 | A |
| CausVid [[12]](#ref-12) | asymmetric DMD + causal student | 是 | 50→4 steps；系统结果还包括 KV 与因果结构的作用 | 长时漂移、教师模型的分布覆盖损失 | A |
| DOLLAR [[13]](#ref-13) | VSD + consistency + latent reward | 是 | 1/4-step；10 秒、128 帧 | 奖励模型偏差、训练成本 | A |
| rCM [[15]](#ref-15) | continuous-time consistency + score regularization | 是 | 1–4 steps；最高 14B、5 秒视频 | JVP 成本、跨教师模型泛化 | A |
| TMD [[16]](#ref-16) | outer transition + 轻量 flow-head rollout | 是 | outer step 不等于完整主干 NFE | 计算折算与结构改造 | A |
| Phased DMD [[17]](#ref-17) | 分区间 progressive DMD + score matching | 是 | few-step；原文实验包括 Wan2.2-28B | 区间路由与训练复杂度 | A |
| ASD [[18]](#ref-18) | $n$ 步 student / $(n+1)$ 步自教师对抗蒸馏 | 是 | 1-step causal video | 任意时长 rollout 尚未验证 | A |
| FastLightGen [[19]](#ref-19) | block pruning + step-and-size co-distillation | 是 | 原文联合使用 4-step 与 30% pruning | 收益归因耦合、硬件迁移 | A |
| AnyFlow [[21]](#ref-21) | 任意区间 flow-map、on-policy simulation | 是 | any-step；原文实验覆盖 1.3B–14B 模型 | 仍待同行评审与独立复现 | B |

只有在完整 denoiser 调用的定义一致时，表中的 NFE 才能直接比较。TMD 的轻量内层更新、CausVid 的因果生成与 KV 路径，以及 FastLightGen 的小模型前向都会改变单次调用成本，因此不能仅按采样步数排列速度高低。

## 6. 已有共识与开放问题

| 议题 | 当前较强共识 | 仍有争议或条件 |
|---|---|---|
| Solver 与蒸馏 | 同一 checkpoint 的 solver 曲线是必要基线 | 极低 NFE 下是否值得训练新学生模型取决于质量要求 |
| 轨迹与分布 | 二者监督对象不同，失败模式也不同 | consistency 的覆盖与 DMD 的清晰度/多样性能否稳定兼得 |
| 视频特有损失 | 单帧质量不能替代运动评测 | 哪种运动表征与长时指标最可靠 |
| 一步与多步 | 一步适合极端低延迟，多步保留更大修正容量 | 最佳点随 prompt、长度和硬件变化，不存在统一步数 |
| Reward/preference | 可以直接补充质量或对齐目标 | evaluator bias、reward hacking 和多样性损失常被平均分掩盖 |
| Causal student | 有利于滚动生成与历史复用 | 如何控制与双向教师模型的分布差异及长期误差 |
| Any-step scaling | 可变预算更适合具有不同延迟要求的服务场景 | 固定少步最优与更多步持续改善之间仍缺统一研究 |
| Step-and-size | 联合优化可以得到不同的 Pareto 点 | 在一种教师模型或骨干上的协同效果能否迁移至其他架构和设备 |

这些方法的主要差异不宜简化为损失函数的优劣。Trajectory 方法侧重路径一致性，DMD 和对抗学习侧重输出分布与感知质量，奖励引导方法优化选定的语义或审美维度，因果方法还需满足视频时序上的状态传递要求。在任务目标尚未统一的情况下，单一排行榜难以全面比较这些方法。

## 7. 公平评测与结果报告

每个蒸馏后 checkpoint 至少应报告以下信息：

```text
teacher/student hash; architecture and parameter count
distillation data, objective, update steps and training GPU-hours
nominal steps; full-backbone NFE; lightweight inner updates; CFG calls
resolution; frames; FPS; task; prompts; seeds; precision; batch
GPU/accelerator; software; warm-up; text/VAE/I/O inclusion
frame, motion, text, identity, long-horizon and diversity metrics
cold/warm p50/p95 latency; peak VRAM; throughput; TTFF if streaming
```

公平比较可分为三个层次：首先比较同一教师模型或 checkpoint 在不同求解器下的质量—NFE 曲线；其次，在教师模型和输出设置相同的条件下比较不同学生模型；最后，再比较允许架构、数据和系统同时变化的完整系统 Pareto 前沿。三个层次的结果不宜合并为单一的“加速倍数”。

| 评测维度 | 最低报告要求 | 需要单独检查的困难样例 |
|---|---|---|
| 外观 | 感知质量、审美、伪影 | 手、文字、小物体、纹理重复、过饱和 |
| 运动 | 动态程度、光流/跟踪、自然性 | 快运动、遮挡、相机急转、静态坍缩 |
| 时序 | 身份/背景一致、事件完成 | 镜头切换、目标离开画面后重新进入、长时漂移 |
| 条件 | prompt、图像或轨迹遵循 | 组合属性、罕见动作、冲突控制 |
| 覆盖 | 不同随机种子下的动作、镜头和构图分布 | 教师模型可以生成、但学生模型丢失的提示词对应模式 |
| 系统 | 完整 NFE、端到端时延、显存、吞吐 | 分别统计 CFG、VAE、I/O、cache 和多卡通信开销 |

“50→4 步”“1 NFE”或“15–50×”等原文结果都依赖各自的实验设置。若输出长度、数值精度、硬件和计时范围不同，这些数字不足以支持跨方法排序，也不能将蒸馏、剪枝、量化与 cache 的加速倍数直接相乘。

## 8. 研究空白与未来方向

1. **跨教师模型的统一少步评测框架。** 需要在相同 prompt、seed、输出规格、数值精度和硬件条件下，覆盖 U-Net 与 DiT、diffusion 与 flow，以及双向与因果教师模型，并公开 NFE 与各模块耗时。
2. **适用于视频的分布覆盖指标。** 现有平均指标不易发现动作、镜头和事件路径上的模式丢失。后续评测应考察不同随机种子下的条件覆盖，并统计典型失效情形。
3. **从固定步数转向可变计算预算。** Any-step 模型需要验证增加推理计算能否带来稳定的质量改善，并研究 endpoint consistency 在何种条件下会限制 test-time scaling。
4. **运动与外观表征的可解释分析。** 有必要分析不同网络层、时间区间和损失项分别如何影响外观、短期运动与长程事件，避免只依赖平均分数解释性能退化。
5. **因果学生模型的长时稳定性。** 在训练窗口内取得较好结果，并不代表模型在任意生成时长下都同样可靠。镜头切换、提示词变化、滚动缓存、误差恢复和状态重置应纳入标准评测。
6. **步数与规模联合压缩的可迁移性。** 联合压缩应在不同骨干、分辨率和硬件上重新评测，并分别报告参数量或 FLOPs、单次前向时延、端到端时延和训练成本。
7. **控制与编辑能力的保持。** 基础生成器完成少步化后，原有的轨迹、姿态、相机和身份 adapter 往往需要重新适配；总体画质指标不应掩盖控制误差。
8. **模型全生命周期的计算成本。** 推理成本降低并不意味着总成本更低。教师模型 rollout、fake score/discriminator、JVP、奖励模型及多轮蒸馏所需的 GPU-hours、能耗和数据成本仍缺少统一报告。

## 9. 建议阅读顺序

1. 首先阅读 DDIM、DPM-Solver、Progressive Distillation 与 Consistency Models，厘清求解器、轨迹蒸馏和 flow-map 学习之间的差别 [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3) [[5]](#ref-5)。
2. 随后阅读 DMD 与 DMD2，理解分布匹配与逐点复现教师模型之间的区别 [[6]](#ref-6) [[7]](#ref-7)。
3. 结合 T2V-Turbo、MCM、APT、CausVid 和 DOLLAR，比较奖励引导、运动与外观解耦、真实数据对抗训练、因果学生模型及 latent reward 等设计 [[8]](#ref-8) [[9]](#ref-9) [[11]](#ref-11) [[12]](#ref-12) [[13]](#ref-13)。
4. 最后阅读 rCM、TMD、Phased DMD、FastLightGen 与 AnyFlow，了解少步蒸馏如何进一步扩展到大规模模型、分段映射、模型规模联合压缩和 any-step scaling [[15]](#ref-15) [[16]](#ref-16) [[17]](#ref-17) [[19]](#ref-19) [[21]](#ref-21)。

## 参考文献

<a id="ref-1"></a>[1] [Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP). Jiaming Song, Chenlin Meng, Stefano Ermon. ICLR. 2021.

<a id="ref-2"></a>[2] [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html). Cheng Lu et al. NeurIPS. 2022.

<a id="ref-3"></a>[3] [Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI). Tim Salimans, Jonathan Ho. ICLR. 2022.

<a id="ref-4"></a>[4] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho et al. Author technical report. 2022.

<a id="ref-5"></a>[5] [Consistency Models](https://proceedings.mlr.press/v202/song23a.html). Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever. ICML. 2023.

<a id="ref-6"></a>[6] [One-Step Diffusion with Distribution Matching Distillation](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html). Tianwei Yin et al. CVPR. 2024.

<a id="ref-7"></a>[7] [Improved Distribution Matching Distillation for Fast Image Synthesis](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html). Tianwei Yin et al. NeurIPS. 2024.

<a id="ref-8"></a>[8] [T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a57aa8e8b57e64a42e95f7dceb0adb9-Abstract-Conference.html). Jiachen Li et al. NeurIPS. 2024.

<a id="ref-9"></a>[9] [Motion Consistency Model: Accelerating Video Diffusion with Disentangled Motion-Appearance Distillation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c859b99b5d717c9035e79d43dfd69435-Abstract-Conference.html). Yuanhao Zhai et al. NeurIPS. 2024.

<a id="ref-10"></a>[10] [AnimateDiff-Lightning: Cross-Model Diffusion Distillation](https://arxiv.org/abs/2403.12706). Shanchuan Lin, Xiao Yang. Author preprint. 2024.

<a id="ref-11"></a>[11] [Diffusion Adversarial Post-Training for One-Step Video Generation](https://proceedings.mlr.press/v267/lin25m.html). Shanchuan Lin et al. ICML. 2025.

<a id="ref-12"></a>[12] [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html). Tianwei Yin et al. CVPR. 2025.

<a id="ref-13"></a>[13] [DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.html). Zihan Ding et al. ICCV. 2025.

<a id="ref-14"></a>[14] [V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_V.I.P.__Iterative_Online_Preference_Distillation_for_Efficient_Video_Diffusion_ICCV_2025_paper.html). Jisoo Kim et al. ICCV. 2025.

<a id="ref-15"></a>[15] [Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html). Kaiwen Zheng et al. ICLR. 2026.

<a id="ref-16"></a>[16] [Transition Matching Distillation for Fast Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_Transition_Matching_Distillation_for_Fast_Video_Generation_CVPR_2026_paper.html). Weili Nie et al. CVPR. 2026.

<a id="ref-17"></a>[17] [Phased DMD: Few-Step Distribution Matching Distillation via Score Matching within Subintervals](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Phased_DMD_Few-step_Distribution_Matching_Distillation_via_Score_Matching_within_CVPR_2026_paper.html). Xiangyu Fan et al. CVPR. 2026.

<a id="ref-18"></a>[18] [Towards One-step Causal Video Generation via Adversarial Self-Distillation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html). Yongqi Yang et al. ICLR. 2026.

<a id="ref-19"></a>[19] [FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_FastLightGen_Fast_and_Light_Video_Generation_with_Fewer_Steps_and_CVPR_2026_paper.html). Shitong Shao, Yufei Gu, Zeke Xie. CVPR. 2026.

<a id="ref-20"></a>[20] [One Step Diffusion via Shortcut Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html). Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel. ICLR. 2025.

<a id="ref-21"></a>[21] [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://arxiv.org/abs/2605.13724). Yuchao Gu et al. Author preprint. 2026.
