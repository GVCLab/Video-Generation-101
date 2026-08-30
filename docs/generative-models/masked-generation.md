# 掩码生成：目标、采样器与时间因果不能混为一谈

掩码生成不是一种单独的概率模型。它至少包含四个可分别选择的层：怎样遮挡数据、用什么目标学习条件分布、推理时怎样决定下一批 token、视频帧之间是否因果。MaskGIT 式迭代解码、吸收态离散扩散、next-set 自回归和“帧间 AR + 帧内掩码”可能共享部分公式，却不因此成为同一种算法。

本章截至 **2026-08-30**，聚焦离散视频 token；连续 latent 中“遮帧规划、扩散渲染”的方法会作为边界案例单列。正文所说的“并行”均指同一轮网络调用内可并行预测多个位置，不等于单次前向、低总计算量或实时生成。

## 1. 先按四层拆开

![图 023：掩码生成的四层关系](assets/imagegen-diagrams/023/diagram.png)
顺序化文字替代：第一，训练端可以只做随机缺失条件预测，也可以先定义 clean token 逐渐进入 `[MASK]` 吸收态的前向过程；第二，前者常接置信度排序的 MaskGIT 式采样，后者有由时间或转移速率定义的概率化反向过程；第三，在特定吸收态参数化下，扩散目标可化成加权 masked 交叉熵，但采样器仍未因此相同；第四，若每轮提交的 token 集合形成有序分区，采样可用 next-set AR 语言描述，不过动态选择策略也是生成过程的一部分；第五，以上任一采样器还可选择整段双向注意力，或“帧间因果、帧内双向”的视频时间结构。

这张图对应四个不能互相替代的问题：

1. **腐败**：哪些 token、帧或时空块被替换为 `[MASK]`，比例怎样采样？
2. **目标**：模型只是预测缺失条件分布，还是优化某个离散扩散的变分目标？
3. **采样**：按置信度提交、按反向转移采样，还是按预定集合顺序生成？
4. **视频因果**：当前 token 能否看未来帧，长视频按整段、帧还是 chunk 推进？

只写“masked model”无法回答后面三个问题。

## 2. Masked modeling 只定义条件预测任务

设视频经 tokenizer 得到 $N$ 个离散 token：

```math
y^0=(y_1^0,\ldots,y_N^0)\in\{1,\ldots,K\}^N,
```

$c$ 是文本、首帧、动作等条件，$`M\subseteq\lbrace1,\ldots,N\rbrace`$ 是被遮挡位置。最常见的目标是

```math
\mathcal L_{\mathrm{mask}}
=\mathbb E_{y^0,M}
\left[-\sum_{i\in M}
\log p_\theta\!\left(y_i^0\mid y^0_{\bar M},c,M\right)\right].
```

它训练“给定可见位置，恢复缺失位置”的条件预测器。$M$ 可以来自随机 token、tube、整帧、时空块或任务特定分布。这个损失本身没有指定：

- 一个归一化联合分布 $p(y^0\mid c)$ 怎样分解；
- 推理时从全 `[MASK]` 到完整样本的转移概率；
- 一轮接受多少 token，以及接受后能否重新打开；
- 视频帧间是否因果。

因此，VideoMAE 的 tube masking 是视频表征预训练证据，不是一个视频生成 sampler。它用 90%–95% 的高比例 tube mask 学表征，不能据此推出生成时也应采用同样比例或 schedule [[4]](#ref-4)。反过来，MAGVIT 的 masked token objective 与 COMMIT 解码共同构成生成系统；只复现 loss 而不复现 condition embedding、调度与采样规则，不是同一个方法 [[6]](#ref-6)。

## 3. 吸收态离散扩散：何时能化成 masked 交叉熵

### 3.1 前向过程必须先被定义

令 $m$ 是额外的 `[MASK]` 吸收态，$\tau\in[0,1]$ 是**噪声时间**，不是视频帧索引。由吸收态马尔可夫链导出的单位置前向边缘可写为

```math
q_\tau(y_i^\tau\mid y_i^0)
=\alpha_\tau\,\mathbf 1[y_i^\tau=y_i^0]
+(1-\alpha_\tau)\,\mathbf 1[y_i^\tau=m],
```

其中 $\alpha_0=1$，并随 $\tau$ 增大而下降。一旦进入 $m$，继续向前只保持在 $m$，这才是“吸收态”的含义。D3PM 将这种转移矩阵作为离散扩散的一种选择，并建立它与 mask-based 模型的联系；**均匀类别噪声、邻接转移和 absorbing mask 都属于离散扩散，但彼此不是同一前向过程** [[1]](#ref-1)。

在特定 $x_0$ 参数化和连续时间推导下，吸收态模型的变分目标可写成一族 masked 交叉熵的加权积分：

```math
\mathcal L_{\mathrm{absorb}}
=\int_0^1 \lambda(\tau)\,
\mathbb E_{y^0,y^\tau}
\left[-\sum_{i:y_i^\tau=m}
\log p_\theta(y_i^0\mid y^\tau,c)\right]\,\mathrm d\tau,
```

其中权重 $\lambda(\tau)$ 由前向 schedule 与所选目标决定。NeurIPS 2024 的 *Simplified and Generalized Masked Diffusion for Discrete Data* 明确给出这一“加权交叉熵积分”桥，并允许状态相关 masking schedule [[9]](#ref-9)；同年的 MDLM 把 Rao–Blackwellized 目标写成经典 masked-language-modeling losses 的混合 [[8]](#ref-8)。这些结果主要来自离散文本与像素建模，支持的是**目标层关系**，不是视频质量的外推。

### 3.2 等价到哪一层，必须逐条限定

| 命题 | 结论 | 成立所需条件 | 不能推出 |
|---|---|---|---|
| absorbing diffusion 与 masked CE | 条件成立 | 指定 absorbing 前向过程、参数化、时间采样和权重 | 任意随机 mask loss 都是 diffusion likelihood |
| absorbing diffusion 与 any-order AR | 目标或界层可联系 | RADD 把 concrete score 写成 clean-data conditional 乘解析时间因子，并把 diffusion NLL 上界解释为 any-order AR 的期望 NLL [[14]](#ref-14) | 两者逐步转移、采样轨迹和视频性能相同 |
| masked diffusion 与 time-agnostic masked model | 在论文所研究参数化与 first-hitting sampler 下成立 | 必须采用相应 absorbing 模型和 sampler [[15]](#ref-15) | 所有带 mask 的视觉生成器都不需要时间，或 MaskGIT heuristic 是精确反向扩散 |
| MaskGIT 与 absorbing diffusion | 可共享网络输入和交叉熵形态 | 还需证明 schedule、接受规则与反向转移相容 | 看到 `[MASK]`、多轮 unmask 就可互换名称 |
| MaskGIT 与 next-set AR | 可作受限解释 | 每轮新提交集合互不重叠并覆盖全部位置，且把集合选择策略纳入生成过程 | 随机-mask 训练就是该自适应顺序的精确最大似然 |
| masked AR 与离散 token | 不必绑定 | MAR 可对连续 token 用 per-token diffusion loss [[10]](#ref-10) | “AR”必然等于离散交叉熵 |

ICLR 2025 的 time-agnostic 分析还发现，常见低温 categorical sampling 的数值误差会降低有效温度与 token 多样性，足以改变评价结论 [[15]](#ref-15)。因此“理论上可去时间变量”不等于实现细节无关；精度、温度、categorical sampler 和多样性指标必须共同报告。这个结论同样主要来自语言模型实验，应作为实现警示，而非视频 benchmark 结论。

最后，URSA 从均匀类别噪声出发，对整段离散时空 token 做全局迭代 refinement，是 ICLR 2026 的视频离散扩散实例，却不是 absorbing-mask diffusion [[18]](#ref-18)。“离散扩散”与“掩码扩散”不能互作同义词。

## 4. MaskGIT 式迭代解码究竟做了什么

MaskGIT 在训练时随机 mask 图像 token，在推理时从全 mask 开始，每轮并行预测当前未知位置，采样候选 token，再按置信度与剩余-mask schedule 决定保留哪些候选。原论文比较多种 schedule，并在其图像实验中选择 cosine；这是一项经验设计，不是所有 masked model 的定律 [[2]](#ref-2)。Phenaki 把相似的双向 masked transformer 用到视频 token，并报告通常使用 12–48 个采样步骤；其 tokenizer 则在时间上因果，这是 tokenizer 可变长能力与生成器双向补全的组合 [[5]](#ref-5)。

![图 024：MaskGIT 式一轮解码的状态变化](assets/imagegen-diagrams/024/diagram.png)
顺序化文字替代：第一，输入由已经提交的 token 与当前 `[MASK]` 位置组成；第二，模型在一次前向中预测所有当前 mask 位置并分别采样候选；第三，为候选计算概率置信度，并可加入随轮次退火的随机扰动；第四，schedule 决定下一轮仍需 mask 的数量，跨位置排序后提交高分候选、重掩低分候选；第五，原版 MaskGIT 给已提交位置高置信度，使其后续保持冻结；第六，若仍有 mask 就重复，否则将完整 token 解码成视频；第七，Token-Critic 是另训的接受/拒绝模型，不应冒充原版 max-probability 规则。

设共有 $J$ 轮，$\gamma:[0,1]\rightarrow[0,1]$ 单调下降，可用

```math
n_j=\left\lceil \gamma(j/J)N\right\rceil
```

指定第 $j$ 轮后仍保持 mask 的总数。这里要区分两个分布：

- **训练 mask-ratio 分布** $\rho\sim\pi_{\mathrm{train}}(\rho)$ 决定训练样本看见哪些缺失率；
- **推理 remaining-mask schedule** $\gamma(j/J)$ 决定每轮保留多少未知位置。

二者可以采用同一函数族，但角色不同。训练覆盖 $`5\%`$–$`95\%`$ 的随机 mask，不保证模型见过“由自己错误且按置信度筛选”的上下文。

### 4.1 Confidence 不是 correctness

常用候选分数

```math
c_i=p_\theta(\tilde y_i\mid y^{(j)},c)
```

只是当前条件分布对已采样类别的局部信心。它可能因位置熵、纹理难度、temperature、guidance 或 token 频率而不可横向比较。一个高置信错误被早早提交后，冻结策略会让后续轮次围绕它自洽，而不是纠正它。

推荐至少报告：

- 每轮、每种 mask unit 的 reliability diagram 与 ECE；
- top-$q$ 提交集合的真实正确率和错误保留率；
- 高置信错误进入后续轮次后的持续比例；
- temperature、Gumbel 扰动和 guidance 改变时的质量—多样性曲线；
- 与随机提交、熵、learned critic 的消融。

Token-Critic 另训判别器识别真实与采样 token，并用它接受、拒绝和重采样，是“可以重新质疑生成 token”的独立 image-side 方案 [[3]](#ref-3)。它证明 remasking policy 可以学习，但不是视频收益或概率校准的自动保证。

## 5. Mask unit 决定模型学到哪一种缺失结构

| Mask unit | 遮挡方式 | 适合的问题 | 主要风险与证据边界 |
|---|---|---|---|
| 随机 token | 在时空 token 中独立抽点 | 通用补全、全序列初始化 | 容易利用近邻冗余；与推理的置信度偏置 mask 不同 |
| temporal tube | 同一空间位置跨多帧一起遮挡 | 迫使使用运动与对象持续性线索 | VideoMAE 只证明表征预训练有效 [[4]](#ref-4)；Lumos-1 的 tube policy 才是生成训练证据 [[17]](#ref-17) |
| 整帧 | 一帧内所有空间 token 共用 mask 状态或比例 | 插帧、未来帧、chunk rollout | 帧级难度差异会造成 loss imbalance；一步内仍有大量空间 token |
| 时空 block | 连续时间与空间区域一起缺失 | 视频补全、局部编辑、对象区域控制 | 训练 block 尺寸与测试编辑区域不匹配时易退化 |
| 因果 prefix / chunk | 过去帧可见，未来帧或下一 chunk 隐藏 | 在线预测、长视频 rollout | 错误跨 chunk 累积，速度取决于 chunk 数与每 chunk 轮数 |
| mixed task mask | 首帧、内部区域、尾帧等条件组合 | 一个模型统一生成、预测、插值与补全 | 必须阻断 tokenizer 或 condition embedding 泄漏；任务采样比例影响能力 |

MAGVIT 用 3D tokenizer 和 COMMIT 条件掩码在同一个模型中统一多种任务；其 SSv2 实验覆盖十种任务，并以约 12 轮非自回归解码为例 [[6]](#ref-6)。这说明 mask 可以成为任务接口，但并不表示任意 mask 混合都会泛化。

Lumos-1 针对另一种问题：它采用帧内双向、帧间因果的 mask-based discrete diffusion，并指出空间冗余会造成 frame-wise loss imbalance；Autoregressive Discrete Diffusion Forcing 在训练中加入 temporal tube masking，并配套推理 masking policy [[17]](#ref-17)。这里 tube mask 的作用是平衡帧级学习与推理兼容性，不能与 VideoMAE 的表示学习动机合并成一条未经限定的结论。

## 6. “Masked AR”至少有三种含义

### 6.1 Next-set AR：先声明有序集合

令 $S_1,\ldots,S_J$ 是位置集合的一个有序分区，则

```math
p(y\mid c)=\prod_{j=1}^{J}
p\!\left(y_{S_j}\mid y_{S_{<j}},c\right).
```

若进一步把同一集合内各位置条件独立化，才得到

```math
p\!\left(y_{S_j}\mid y_{S_{<j}},c\right)
\approx\prod_{i\in S_j}p(y_i\mid y_{S_{<j}},c).
```

这与 MaskGIT 的一轮并行分类很像，但有两处缺口：MaskGIT 的 $S_j$ 常由本轮样本置信度动态决定；随机-mask 训练也没有显式最大化这条自适应选择策略的联合似然。可以称作“动态 next-set 视角”，不能无条件称为等价概率分解。

MAR 进一步说明 next-set AR 与离散码本并不绑定：它在连续图像 token 上用小型 diffusion loss 表示每个 token 的条件分布，并研究 generalized masked AR [[10]](#ref-10)。这是图像侧的结构证据，不是视频 benchmark 证据。

### 6.2 帧间 AR、帧内 masked：MAGI 与 Lumos-1

![图 025：帧间因果与帧内掩码的双层串行深度](assets/imagegen-diagrams/025/diagram.png)
顺序化文字替代：第一，把已经完成的第 $1$ 到 $k-1$ 帧作为因果历史；第二，将第 $k$ 帧的空间 token 设为全部或部分 mask；第三，在当前帧内部做 $J$ 轮双向预测、提交与重掩；第四，当前帧完成后写入历史或 KV cache；第五，对第 $k+1$ 帧重复；第六，训练时还要选择历史是完整真值帧还是被 mask 的帧，MAGI 的 Complete Teacher Forcing 选择前者，以缩小其设定中的训练—推理上下文差异。

MAGI 把帧内 masked modeling 与帧间 causal modeling 结合。其 Complete Teacher Forcing（CTF）让目标帧的 mask 预测条件于**完整 observation frames**，而不是 Masked Teacher Forcing（MTF）中的残缺历史；官方 CVPR 2025 论文在首帧条件预测协议下报告 FVD 相对改善 23%，并展示从 16 帧训练窗口 rollout 超过 100 帧 [[12]](#ref-12)。这些数字属于论文设置，不能写成所有数据和分辨率上的保证。

Lumos-1 同样采用帧间 causal、帧内 bidirectional masked discrete diffusion，但以 LLM 架构、MM-RoPE 和 temporal tube masking 处理效率及 frame-wise loss imbalance，是 ICLR 2026 的正式路线 [[17]](#ref-17)。两者结构相近，不表示训练目标、mask policy 或概率解释完全相同。

### 6.3 Masked planner + continuous renderer：MarDini

MarDini 的 “MAR” 在低分辨率连续 VAE latent 上遮掉整帧，让大规划器预测每帧 planning signal；随后轻量 continuous diffusion model 负责高分辨率空间生成 [[11]](#ref-11)。它说明“masked autoregressive video”也可能只是外层时间规划，内层既不是离散 CE，也不是 absorbing mask reverse process。该工作于 2025 年 5 月正式发表于 TMLR。

## 7. 并行度要报告串行深度，而不只写“并行生成”

设视频共有 $N$ 个 token、$T$ 帧，masked sampler 每个生成单元用 $J$ 轮，长视频分成 $C$ 个因果 chunk。忽略 guidance 与额外 critic 前向时：

| 生成组织 | 理想化串行网络调用深度 | 同一调用内并行 | 仍然存在的代价 |
|---|---:|---|---|
| 逐 token AR | $N$ | 训练可并行，采样不可 | KV cache、长序列与错误累积 |
| 全序列 masked | $J$ | 当前所有 mask token | 每轮全局 attention、$J$ 次前向、置信度排序 |
| next-set AR | 集合数 $J$ | 同一集合 token | 集合内条件独立近似与顺序策略 |
| 帧间 causal + 帧内 masked | 约 $T\times J$ | 当前帧空间 token | 每帧重新走迭代链；KV cache 只省历史重算 |
| chunk-causal masked | 约 $C\times J$ | 当前 chunk token | chunk 边界、上下文窗口和跨段漂移 |
| masked + Token-Critic / guidance | 基线再乘额外前向 | 依实现 | critic、条件/无条件双前向等 |

“$J$ 与 $N$ 无关”只说明串行轮数可能不随 token 数线性增长。若 dense attention 单轮成本随序列长度快速增长，长视频仍然昂贵。比较系统时至少同时报告：模型前向次数（NFE）、有效串行深度、每次处理的 token 数、attention 形式、guidance/critic 调用、batch、精度、硬件、端到端 wall-clock 与首帧延迟。

MaskFlow 很好地展示了这个权衡：它用逐帧独立 mask ratio 的训练支持 full-sequence 与 chunk/frame AR 两种 rollout，并可用 MGM-style sampling；更小 stride 往往更稳，却需要更多 NFE [[16]](#ref-16)。该工作截至截止日是 arXiv 预印本，且其核心是 discrete flow matching，不应自动归入 absorbing D3PM。

## 8. 训练—推理错配：六个独立来源

| 错配 | 训练端 | 推理端 | 可审计的缓解方式 |
|---|---|---|---|
| 上下文来源 | 可见 token 都是真值 | 可见 token 含模型错误 | self-conditioning、生成上下文扰动、长 rollout 验证 |
| mask 选择 | 随机或任务预设位置 | 由模型置信度偏置到“难位置” | 记录 mask-set 分布差异；训练中模拟选择策略 |
| 历史帧 | 完整或 masked 真值历史 | 完整但有误的生成历史 | CTF 对齐“完整历史”形态；噪声注入与动态时间间隔 [[12]](#ref-12) |
| 比例与 schedule | $\rho\sim\pi_{\mathrm{train}}$ | 固定 $\gamma(j/J)$ 轨迹 | 覆盖推理到达的比例和中间状态；做 schedule 外推测试 |
| 时间范围 | 短训练窗口 | 多 chunk 或多帧递归 | 训练 rollout、跨段状态记忆、按长度报告漂移 |
| codec / condition | 编码时可能看见未来或完整区域 | 真实部署只允许因果条件 | 因果 tokenizer、裁断感受野、条件泄漏单测 |

CTF 只修正其中“历史帧的形态”一项，不能消除模型历史从真值变成自生成所带来的 exposure bias。Lumos-1 的 temporal tube masking 主要针对 frame-wise loss imbalance 与兼容推理策略，也不能被概括成解决了全部 train–test gap。

## 9. 2021–2026 的一手路线与正确边界

| 年份 | 工作与状态 | 这条路线真正增加了什么 | 不应误写成 |
|---:|---|---|---|
| 2021 | D3PM，NeurIPS [[1]](#ref-1) | 离散转移矩阵、absorbing state 与 mask-based/AR 联系 | 所有离散 diffusion 都使用 `[MASK]` |
| 2022 | MaskGIT，CVPR [[2]](#ref-2) | 随机 masked token 训练 + 置信度 schedule 的图像迭代解码 | 精确 reverse D3PM |
| 2022 | Token-Critic，ECCV [[3]](#ref-3) | 学习接受、拒绝和重采样 token | 原版 MaskGIT 的默认置信度 |
| 2022 | VideoMAE，NeurIPS [[4]](#ref-4) | 高比例 tube mask 的视频表征学习 | 视频生成 sampler |
| 2023 | Phenaki，ICLR [[5]](#ref-5) | 时间因果 C-ViViT tokenizer + 双向 masked 视频生成器，支持变长/动态文本 | 时间因果的逐 token 生成器 |
| 2023 | MAGVIT，CVPR [[6]](#ref-6) | 3D tokenizer、COMMIT 与统一多任务 masked 视频生成 | 仅仅把 MaskGIT 换成 3D token |
| 2024 | MAGVIT-v2，ICLR [[7]](#ref-7) | 图像/视频共享的高质量视觉 tokenizer，支持 LM 式生成 | “MAGVIT masked sampler v2” |
| 2024 | MDLM 与 simplified masked diffusion，NeurIPS [[8]](#ref-8) [[9]](#ref-9) | absorbing diffusion 与 masked CE 的严格目标桥 | 视频生成质量已被证明 |
| 2024 | MAR，NeurIPS [[10]](#ref-10) | 连续 token 的 generalized masked AR 与 per-token diffusion loss | 离散 masked video model |
| 2025 | MarDini，TMLR [[11]](#ref-11) | 低分辨率 masked frame planning + 连续 diffusion renderer | absorbing-state discrete diffusion |
| 2025 | MAGI，CVPR [[12]](#ref-12) | 帧内 masked、帧间 causal，以及 Complete Teacher Forcing | 整段完全并行 |
| 2025 | MotionAura，ICLR [[13]](#ref-13) | VQ 离散视频 diffusion、spectral denoiser 与 codec full-frame masking | 所有模块都共享一个 mask 过程 |
| 2025 | RADD / time-agnostic MDM，ICLR [[14]](#ref-14) [[15]](#ref-15) | clean conditional、any-order AR 与 time-free sampler 的理论边界 | 所有 MaskGIT 视频模型等价 |
| 2025 | MaskFlow，arXiv 预印本 [[16]](#ref-16) | frame-level discrete flow matching，兼容 MGM/FM 与长视频 chunk rollout | 已正式同行评审的 absorbing D3PM |
| 2026 | Lumos-1，ICLR [[17]](#ref-17) | LLM 内帧间 causal + 帧内 masked discrete diffusion，temporal tube policy | 逐 token AR，或全序列双向模型 |
| 2026 | URSA，ICLR [[18]](#ref-18) | uniform discrete diffusion 的全局时空 refinement | masked / absorbing diffusion |

MAGVIT-v2 特别容易被误读。论文的核心贡献是 tokenizer，并用它支持 language-model-style visual generation；它不是对 MAGVIT 的 COMMIT/MaskGIT sampler 做“第二版” [[7]](#ref-7)。同理，MotionAura 的 codec 训练含 full-frame masking，而生成器采用 vector-quantized discrete diffusion；两个 mask 出现在不同模块，不能合并成一个算法描述 [[13]](#ref-13)。

## 10. 实验与复现最低清单

### 10.1 训练配置

- token 是离散 index、连续 latent，还是分层组合？单独报告 codec 重建上限。
- 写清 mask unit、位置分布、$\pi_{\mathrm{train}}(\rho)$、是否按帧独立采样。
- 若称 diffusion，给出前向 transition/rate、时间变量、参数化与 loss weight。
- 写清 attention：帧内、帧间各是双向还是因果，tokenizer 是否因果。
- 说明 teacher forcing 的历史是完整、masked、加噪还是真正模型 rollout。

### 10.2 推理配置

- 给出轮数 $J$、remaining-mask schedule、temperature、top-k/top-p、guidance。
- 写明 confidence 的定义、跨位置排序方式、tie handling 与随机扰动。
- 明确“remask”只作用于本轮候选，还是能够重新打开早期提交 token。
- 若称 probabilistic reverse process，给出实际 transition；若是 heuristic，就直接标明。
- 长视频报告 frame/chunk stride、上下文长度、缓存策略与总 NFE。

### 10.3 评测配置

- 同时报告质量、多样性、时序一致性和条件遵循，不只挑一个 FVD。
- 按视频长度画退化曲线，区分单窗口、$2\times$、$5\times$ 与更长 rollout。
- 用同一硬件、分辨率、帧数、精度和 guidance 比较 wall-clock。
- 对 confidence 做 held-out calibration；对 categorical sampler 检查数值精度与有效温度。
- 进行 mask-unit、训练比例、推理 schedule、可否 reopen、frame/chunk stride 消融。

## 11. 与其他章节的关系

- [自回归生成](autoregressive-generation.md)讨论固定 token 顺序与 teacher forcing；本章补充 next-set 和帧级混合分解。
- [扩散模型](diffusion-models.md)以连续状态为主；本章只在 absorbing 或 uniform categorical 转移已定义时使用“离散扩散”。
- [因果与流式生成](causal-streaming-generation.md)区分因果、少步与持续出帧；“帧内并行”本身不等于流式。
- [生成模型总览](../generative-models.md)把 objective、representation、temporal factorization、sampler 和 deployment 作为交叉分类轴；本章是该框架下的 masked 分支。

## 参考文献

<a id="ref-1"></a>[1] [Structured Denoising Diffusion Models in Discrete State-Spaces](https://proceedings.neurips.cc/paper_files/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html). Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg. NeurIPS. 2021.

<a id="ref-2"></a>[2] [MaskGIT: Masked Generative Image Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html). Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman. CVPR. 2022.

<a id="ref-3"></a>[3] [Improved Masked Image Generation with Token-Critic](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2901_ECCV_2022_paper.php). José Lezama, Huiwen Chang, Lu Jiang, Irfan Essa. ECCV. 2022.

<a id="ref-4"></a>[4] [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html). Zhan Tong, Yibing Song, Jue Wang, Limin Wang. NeurIPS. 2022.

<a id="ref-5"></a>[5] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://iclr.cc/virtual/2023/poster/12256). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.

<a id="ref-6"></a>[6] [MAGVIT: Masked Generative Video Transformer](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html). Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. CVPR. 2023.

<a id="ref-7"></a>[7] [Language Model Beats Diffusion - Tokenizer is key to visual generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html). Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. ICLR. 2024.

<a id="ref-8"></a>[8] [Simple and Effective Masked Diffusion Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html). Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, et al. NeurIPS. 2024.

<a id="ref-9"></a>[9] [Simplified and Generalized Masked Diffusion for Discrete Data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html). Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias. NeurIPS. 2024.

<a id="ref-10"></a>[10] [Autoregressive Image Generation without Vector Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html). Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He. NeurIPS. 2024.

<a id="ref-11"></a>[11] [MarDini: Masked Auto-regressive Diffusion for Video Generation at Scale](https://openreview.net/forum?id=fuOHI59rUW). Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, et al. TMLR. 2025.

<a id="ref-12"></a>[12] [Taming Teacher Forcing for Masked Autoregressive Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html). Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, et al. CVPR. 2025.

<a id="ref-13"></a>[13] [MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ad996b5c45130de2bc00b60d8607904-Abstract-Conference.html). Onkar Susladkar, Jishu Sen Gupta, Chirag Sehgal, Sparsh Mittal, Rekha Singhal. ICLR. 2025.

<a id="ref-14"></a>[14] [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a365e37c18fb91af547a2f0012a89e98-Abstract-Conference.html). Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, Chongxuan Li. ICLR. 2025.

<a id="ref-15"></a>[15] [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html). Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, Qinsheng Zhang. ICLR. 2025.

<a id="ref-16"></a>[16] [MaskFlow: Discrete Flows For Flexible and Efficient Long Video Generation](https://arxiv.org/abs/2502.11234). Michael Fuest, Vincent Tao Hu, Björn Ommer. arXiv preprint. 2025.

<a id="ref-17"></a>[17] [Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59ad89d72559dd4ce557d56f36313724-Abstract-Conference.html). Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, et al. ICLR. 2026.

<a id="ref-18"></a>[18] [Uniform Discrete Diffusion with Metric Path for Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/daf8364f0715a41a469c677c0adc4754-Abstract-Conference.html). Haoge Deng, Ting Pan, Fan Zhang, Yang Liu, Zhuoyan Luo, Yufeng Cui, et al. ICLR. 2026.
