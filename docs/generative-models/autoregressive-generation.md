# 自回归视频生成：表示、提交粒度与串行代价

> 本章截至 2026-08-30，依据正式会议论文、作者论文与官方发布面整理。阅读主线是：先判断联合分布按什么单位分解，再判断每个条件分布用 categorical、diffusion、flow 还是 masked refinement 实现，最后才讨论 KV cache、few-step 与 streaming。

“自回归视频生成”不是“离散 tokenizer + 交叉熵”的同义词。生成变量可以是原始像素、离散视觉码或连续 latent；一次提交可以是一个 token、一组 token、一帧或一个时间块；单个条件分布还可以由 diffusion/flow head 建模。反过来，使用 causal attention、少步去噪或流式服务，也不自动使系统成为严格的 next-token AR。

## 📋 1. 先定义：AR 是 factorization，不是模型品牌

### 1.1 严格自回归：每个变量都有一个全序

给定条件 $c$，把待生成视频表示成有序变量 $y_{1:N}$。严格自回归分解为

```math
p(y_{1:N}\mid c)
=
\prod_{i=1}^{N}
p(y_i\mid y_{<i},c).
```

这里的 $y_i$ 可以是一个像素通道、一个离散码或一个连续 latent token。定义只规定“第 $i$ 个变量不能读取未来变量”，没有规定：

- $y_i$ 必须离散；
- 损失必须是 categorical cross-entropy；
- 网络必须是 decoder-only Transformer；
- 顺序必须按视频时间从前到后；
- 推理必须实时或流式。

训练时，因果 mask 允许一次前向并行计算所有位置的条件损失；采样时，第 $i+1$ 个变量仍要等待第 $i$ 个变量被提交。训练并行与推理串行并不矛盾。

### 1.2 广义集合、帧与 chunk 自回归

许多论文把变量划分成按顺序提交的集合 $Y_1,\ldots,Y_K$：

```math
p(Y_{1:K}\mid c)
=
\prod_{k=1}^{K}
p(Y_k\mid Y_{<k},c).
```

这是一种有效的 block factorization，但 $Y_k$ 内部可能用双向注意力、并行独立 head、masked refinement 或联合 diffusion。为避免命名冲突，本章采用以下约定：

- **strict token AR**：每个生成变量处于全序中，每次提交一个变量；
- **set/frame/chunk AR**：只要求外层单元按顺序提交；单元内部机制必须另写；
- **masked iterative generation**：未知位置多轮并行补全，不默认称为 strict AR。

“下一组 token”可以被社区称为 generalized autoregression，但不能把它的并行深度、条件独立假设和 KV cache 行为直接等同于 next-token decoding。

### 1.3 提交粒度是最容易漏掉的变量

本章把 **commit unit** 定义为：在进入下一次外层生成决策前，已经确定、不会再被后续内部迭代改写，并可作为因果历史的最小输出单元。

若当前帧内的 token 仍会被重新 mask 或继续去噪，它们还不是稳定历史。只有整个帧或 chunk 完成并提交后，其 KV 才可能被下一单元安全复用。提交粒度同时决定：

1. 外层串行深度；
2. 新结果多久可见；
3. 哪些状态可以放进 KV cache；
4. 单个错误会以多大单位进入未来上下文；
5. “一步”究竟指 token、refinement round、noise-time NFE，还是一次外层 frame/chunk commit。

### 1.4 一张图定位一个 AR 系统

![图 012：自回归视频生成的三层选择](assets/imagegen-diagrams/012/diagram.png)
**图的顺序化文字替代：**

1. 先确定生成变量是原始像素、离散视觉码还是连续 latent。
2. 再确定进入不可变历史的最小单位是 token、集合、金字塔尺度、帧还是时间块。
3. 然后确定每个条件分布由 categorical head、逐 token diffusion/flow head、masked refinement 或完整视频 backbone 的去噪过程实现。
4. VideoGPT、NOVA、Lumos-1 与 CausVid 是这些轴的不同组合，不是四个互斥 objective。

## 🧱 2. Pixel AR：定义最纯粹，视频代价最直接

PixelRNN 将图像像素按固定扫描顺序分解，证明深度网络可以直接学习离散像素通道的条件分布 [[1]](#ref-1)。Video Pixel Networks 把这种思路扩展到视频，在时间、高度、宽度与颜色通道上建立四维依赖链 [[2]](#ref-2)。

若视频 $`v\in\lbrace0,\ldots,255\rbrace^{T\times H\times W\times C}`$ 被展平为 $v_{1:N}$，其中 $N=THWC$，则

```math
p(v\mid c)
=
\prod_{i=1}^{THWC}
p(v_i\mid v_{<i},c).
```

它不需要学习 tokenizer，因此没有量化误差；但代价是把局部纹理、颜色相关性和跨帧运动都交给同一条极长序列。即使只是 $16\times64\times64$ 的 RGB clip，也有 196,608 个通道级决策。历史 pixel AR 的价值主要是给出清晰 likelihood 与因果依赖基线，而不是证明高分辨率视频应继续逐通道采样。

Pixel AR 还暴露了一个通用事实：**表示长度与提交粒度共同决定串行下界**。更强的并行硬件可以加速每一步，却不能让第 $i+1$ 个严格条件在第 $i$ 个尚未采样时成立。

## 📦 3. 离散 token AR：tokenizer、顺序与生成目标要分开

### 3.1 两阶段概率模型

VQ-VAE 把连续 encoder 输出量化到有限 codebook，再由 decoder 重建像素 [[3]](#ref-3)。对视频 $v$，可写成

```math
z=Q(E(v)),\qquad
\hat v=D(z),
```

而严格离散 token 生成器学习

```math
p(z_{1:N}\mid c)
=
\prod_{i=1}^{N}
\mathrm{Cat}\!\left(
z_i;\pi_\theta(z_{<i},c)
\right).
```

训练目标通常是 token cross-entropy，但整体系统至少有两个误差源：

```math
\text{最终误差}
\approx
\text{tokenizer 重建误差}
+
\text{token prior 建模误差}.
```

因此，生成结果的模糊、文字损坏或闪烁不能一律归因于 Transformer；tokenizer 已经删除的信息，后续 prior 无法恢复。反过来，重建指标更好也不保证 token prior 更容易学：词表过大、序列过长或码分布严重失衡都可能增加生成难度。

### 3.2 VideoGPT：严格时空离散 AR 的可复现基线

VideoGPT 使用带 3D convolution 与 axial attention 的 VQ-VAE 压缩视频，再用 GPT-like Transformer 严格自回归地预测时空离散 latent [[4]](#ref-4)。它的重要性在于把“视频 tokenizer + strict token prior”拆得足够清楚；其论文截至本章日期仍应标为 arXiv preprint，而不是补写成正式会议论文。作者仓库公开训练与评测代码，但这不改变论文 venue 状态。

### 3.3 VideoPoet：统一词表仍是严格 next-token 训练

VideoPoet 把文本、图像、视频与音频离散化，在 decoder-only Transformer 中混合多种自回归生成目标 [[8]](#ref-8)。统一 token 接口便于把不同条件任务写成同一序列，但“都能成为 token”不表示它们具有相同：

- tokenizer 失真；
- 每 token 所覆盖的物理时间与空间范围；
- 错误传播代价；
- 采样温度与校准特性。

它是离散 strict AR 的重要扩展，不是“多模态一经 token 化就消除了模态差异”的证据。

### 3.4 Phenaki 与 MAGVIT-v2：最常见的误归类

Phenaki 的 C-ViViT tokenizer 在时间上是 causal / autoregressive，从而支持可变长度编码；但从文本生成视频 token 的上层模型是 **bidirectional masked Transformer** [[6]](#ref-6)。因此应写成“因果视频 tokenizer + masked token generator”，不能只因 tokenizer 有 autoregressive 性就把整个生成器称为 strict next-token AR。

MAGVIT-v2 的核心贡献是 lookup-free quantization 与视频/图像共享 tokenizer。论文同时讨论 AR language model 与 masked language model，并明确描述后者从全 mask 开始、反复采样与重 mask 的非自回归解码 [[7]](#ref-7)。所以：

- MAGVIT-v2 可以为 AR-LM 提供离散 token；
- tokenizer 论文的生成结果不能全部自动记到 strict AR 名下；
- 官方 google-research/magvit 仓库标明自己发布的是 CVPR 2023 MAGVIT，而非一套完整的 MAGVIT-v2 训练发布。

MaskGIT 是理解这种边界的经典对照：它每轮并行预测多个被 mask 的 token，再按置信度保留或重 mask，不采用固定 next-token 全序 [[5]](#ref-5)。

### 3.5 纯离散路线并未在高分辨率阶段消失

InfinityStar 在统一离散时空表示上把 next-scale prediction 扩展为 spacetime pyramid modeling，并公开代码与模型 [[14]](#ref-14)。它按从粗到细的图像/clip pyramid 提交尺度，不是 raster-scan strict next-token。论文报告 720p、5 秒以及相对 diffusion baseline 的速度优势，但这些数字绑定其 tokenizer、模型、采样设置与硬件，不能外推成“离散 AR 普遍快十倍”。它更可靠地支持的结论是：截至 2025 年，纯离散的 generalized scale AR 仍能扩展到高分辨率视频，而非只存在于早期低分辨率基线。

| 系统 | 表示 | 上层生成 factorization | 是否 strict next-token AR | 证据边界 |
|---|---|---|---|---|
| VideoGPT | 离散 VQ 视频码 | 固定时空序列 | 是 | 论文是 2021 arXiv preprint；有作者代码 |
| Phenaki | 离散 C-ViViT 码 | 多轮 masked completion | 否 | tokenizer 时间因果不等于 generator strict AR |
| MAGVIT-v2 | lookup-free 离散码 | 可接 AR-LM 或 MLM | 取决于所接生成器 | 主要是 tokenizer 证据；需逐实验辨认 objective |
| VideoPoet | 多模态离散 token | decoder-only next-token | 是 | 正式 ICML 2024；项目页不等于开放权重 |
| InfinityStar | 统一离散时空 token | spacetime-pyramid next-scale AR | 否；scale-level generalized AR | 正式 NeurIPS 2025；速度为作者特定配置报告 |

## 🔄 4. 连续 latent AR：条件 head 不必是 softmax

### 4.1 从 categorical head 换成条件密度 head

连续 token $x_i\in\mathbb R^d$ 没有有限类别标签，不能直接用普通 softmax 覆盖其条件密度。Autoregressive Image Generation without Vector Quantization（MAR）把两件事分开 [[9]](#ref-9)：

1. AR Transformer 根据已知前缀产生条件表示

   ```math
   h_i=F_\theta(x_{<i},c);
   ```

2. 一个小型 denoising MLP 用 Diffusion Loss 建模 $p(x_i\mid h_i)$。

令

```math
x_i^\tau
=
\alpha_\tau x_i+\sigma_\tau\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
```

典型噪声预测目标可写为

```math
\mathcal L_{\mathrm{head}}
=
\mathbb E_{i,\tau,\epsilon}
\left[
\left\|
\epsilon-
D_\phi(x_i^\tau,\tau,h_i)
\right\|_2^2
\right].
```

训练时可对许多位置并行采样噪声时间；推理时，每个待提交 token 或 token set 内部还要运行 $D$ 次小 head evaluation。这里存在两个独立时钟：

- $i$ 或 $k$：数据变量的 AR 提交顺序；
- $\tau$：单个条件密度内部的 diffusion noise time。

把二者都叫“step”会掩盖真实计算。

### 4.2 Flow head 是同一 factorization 的另一种实现

条件分布也可以由 flow head 生成。给定 $h_i$，从先验 $u_{i,0}$ 出发积分

```math
\frac{\mathrm d u_{i,s}}{\mathrm ds}
=
v_\phi(u_{i,s},s,h_i),
\qquad
u_{i,1}=x_i.
```

外层仍满足 $`p(x_{1:N}\mid c)=\prod_i p(x_i\mid x_{\lt i},c)`$；变化的只是每个 $p(x_i\mid\cdot)$ 的学习目标和采样器。MAR 与 NOVA 给出的直接一手证据是 **diffusion head**，不是 flow head。若某实现改用 flow matching，必须另报其条件路径、solver 与真实 NFE，不能仅凭连续输出就把它写成 flow。

### 4.3 NOVA：frame-by-frame 与 set-by-set 的嵌套分解

NOVA 将 MAR 的连续 token 思路扩展到视频，使用非量化连续 latent，并把时间与空间分成两层 [[10]](#ref-10)。令 $\mathbf S_f$ 表示第 $f$ 帧的空间 latent 集合，$\mathbf P,\mathbf m,\mathbf B$ 表示文本、mask 与其他条件，则外层可写为

```math
p(\mathbf S_{1:F}\mid\mathbf P,\mathbf m,\mathbf B)
=
\prod_{f=1}^{F}
p(\mathbf S_f\mid
\mathbf P,\mathbf m,\mathbf B,\mathbf S_{<f}).
```

帧内再把 token 分成 $K$ 个集合：

```math
p(\mathbf S_f\mid\cdot)
=
\prod_{k=1}^{K}
p(\mathbf S_{f,k}\mid
\mathbf S_{<f},\mathbf S_{f,<k},\cdot).
```

关键边界是：

- 帧间是时间因果的 outer AR；
- 帧内按 set 提交，不是 raster-scan strict token AR；
- 当前集合内可双向建模、并行预测；
- 每个连续 token 的条件密度由 diffusion head 实现；
- “AR steps”与“diffusion steps”是两个可独立调整的参数。

官方 NOVA 推理接口分别暴露这两类步数，正好说明不能只报一个模糊的“采样步数”。MAR 的主要实验证据来自图像；NOVA 才是该连续条件 head 在视频上的直接正式证据。

## 🧩 5. Token、set、scale、frame 与 chunk：谁先被提交

| 提交单位 | 外层因果关系 | 单元内部可能机制 | 严格性 | 主要收益 | 主要代价 |
|---|---|---|---|---|---|
| 像素通道 | 每通道依赖全部前缀 | categorical | strict | 无 tokenizer 误差 | 串行深度极大 |
| 离散 token | 每 token 依赖全部前缀 | categorical | strict | likelihood 清晰、兼容 LLM 栈 | tokenizer 上限、序列仍长 |
| 连续 token | 每 token 依赖全部前缀 | 小 diffusion/flow head | strict | 去掉量化，条件分布更灵活 | 每 token 有内部 NFE |
| token set | 下一集合依赖已提交集合 | 并行 head、双向 attention | generalized | 降低外层串行深度 | 集合内联合性与校准需说明 |
| pyramid scale | 细尺度依赖已提交粗尺度 | 整个 token map 并行预测 | generalized | 以较少外层步数逐级细化 | 顺序在尺度而非视频 token 全序 |
| frame | 下一帧依赖已提交帧 | masked、discrete diffusion、continuous diffusion | 仅 outer AR | 天然匹配播放顺序 | 一帧完成前未必可见 |
| temporal chunk | 下一块依赖已提交块 | full-backbone denoising | 仅 outer AR | 长视频与 KV 复用更自然 | chunk 边界、漂移和每块 NFE |

### 5.1 Lumos-1：frame-wise AR 与帧内并行离散 diffusion

Lumos-1 使用 inter-frame causal、intra-frame bidirectional 的 attention mask，并在帧内采用并行 mask-based discrete diffusion；其 Autoregressive Discrete Diffusion Forcing 还引入 temporal tube masking [[15]](#ref-15)。准确分类是：

```math
\text{离散表示}
\times
\text{frame-level outer AR}
\times
\text{intra-frame discrete diffusion}.
```

它不是 strict next-token AR。论文正是为避免 next-token latency 而让一帧内多个离散 token 并行 refinement。

### 5.2 MAGI：frame-causal 与 within-frame masked 的另一种组合

MAGI 把帧间 causal prediction 与帧内 masked modeling 组合，并重点研究 teacher forcing 的具体形式 [[11]](#ref-11)。它的论文标题虽然包含 “Masked Autoregressive”，但教材中应写清 outer frame factorization 与 inner masked objective，不能从标题直接推出固定 token 全序。

### 5.3 CausVid / Self Forcing：每个外层单元内部仍是视频 diffusion

CausVid 把双向视频 diffusion teacher 蒸馏成 causal autoregressive student，并把多步 teacher 压缩为 few-step student [[12]](#ref-12)。Self Forcing 进一步让训练 rollout 条件于模型自身生成的历史 [[13]](#ref-13)。二者的典型计算结构不是“小 diffusion MLP 逐 token采样”，而是：

```math
\text{frame/chunk commits}
\times
\text{每个 commit 的 full-backbone denoising NFE}.
```

因此，MAR/NOVA 的 per-token diffusion head 与 CausVid/Self Forcing 的 video diffusion backbone 必须分栏；仅凭 “autoregressive diffusion” 四个字无法判断成本。

## 🎓 6. Teacher forcing、complete teacher forcing 与 self forcing

### 6.1 训练前缀与推理前缀来自不同分布

标准 teacher forcing 在真实样本 $x\sim p_{\mathrm{data}}$ 的前缀上训练：

```math
\mathcal L_{\mathrm{TF}}
=
\mathbb E_{x\sim p_{\mathrm{data}}}
\sum_{k=1}^{K}
\ell_\theta(x_k;x_{<k},c).
```

推理却递归使用模型样本：

```math
\hat x_k
\sim
p_\theta(\cdot\mid\hat x_{<k},c).
```

训练看到的是 $`x_{\lt k}`$，部署看到的是 $`\hat x_{\lt k}`$。后者可能包含前面生成造成的身份偏移、色调漂移、几何错误、运动冻结与 chunk 接缝；这些状态在纯 teacher-forced 训练中概率很低。这一 history-distribution gap 才是 exposure bias 的核心。

### 6.2 MAGI 的 CTF 修正的是哪一层

MAGI 对比 masked teacher forcing（MTF）与 complete teacher forcing（CTF）：CTF 为预测未来帧提供完整、未 mask 的 ground-truth 历史帧，减少训练上下文被人为遮挡造成的信息缺失 [[11]](#ref-11)。但 CTF 的历史仍来自数据，不来自模型 rollout。因此：

- CTF 可以改善“训练时历史是否完整”；
- 它不等于 on-policy training；
- 它不能单独消除“真实历史 vs 自生成历史”的一般 exposure gap。

### 6.3 Self Forcing 直接训练在自生成历史上

Self Forcing 在训练阶段执行带 KV cache 的 autoregressive rollout，让后续单元真实条件于前面自生成的输出，并用视频级整体损失评价 rollout [[13]](#ref-13)。为了控制反向传播成本，它结合 few-step 模型与 stochastic gradient truncation。

这同时引入两个独立变化：

1. history source 从 ground truth 改为 model rollout；
2. 单个单元的 diffusion 采样被压缩到较少 NFE。

若实验只比较最终速度，不能把两者的贡献都归到 “self forcing” 这个训练前缀选择上。

### 6.4 CausVid 的 teacher 依赖与模式覆盖边界

CausVid 依赖高质量双向 teacher，并使用 distribution matching distillation 训练 causal student [[12]](#ref-12)。它带来少步与因果生成能力，但 student 的：

- 可达分布受 teacher 与蒸馏目标共同约束；
- mode coverage 不能只靠少量视觉样例判断；
- 50-step 到 4-step 等数字必须绑定论文设置；
- causal factorization、DMD objective 与 streaming KV cache 是三个不同贡献层。

### 6.5 一张时序图看训练—推理差异

![图 013：Teacher forcing 与 self forcing 的历史来源](assets/imagegen-diagrams/013/diagram.png)
**图的顺序化文字替代：**

1. Teacher forcing 对每个位置都把 ground-truth 前缀交给生成器。
2. 模型可以并行计算各位置损失，但从未在这些位置看到自己的历史错误。
3. Self forcing 只从初始条件开始，之后每次采样并提交当前单元。
4. 已提交单元进入 rollout 状态与 KV cache，成为下一单元的真实训练上下文。
5. 完整自生成序列接受视频级监督；实际方法可用 few-step 与梯度截断控制成本。

## ⚙️ 7. 串行复杂度与 KV cache：缓存减少重算，不删除依赖

### 7.1 strict token decoding 的注意力工作

设一共生成 $N$ 个 token。若每一步都把长度 $i$ 的完整前缀重新送入 causal Transformer，仅 self-attention 的累计工作近似

```math
\sum_{i=1}^{N}O(i^2)
=
O(N^3).
```

KV cache 保存历史层的 key/value，使第 $i$ 步只为新 query 与历史 key 计算注意力：

```math
\sum_{i=1}^{N}O(i)
=
O(N^2).
```

这是累计 attention work 的渐近比较，不是完整墙钟公式；projection、MLP、kernel、batch 与 memory bandwidth 仍会影响实测。最重要的是，缓存后依然有 $N$ 次依赖相连的 commit，不能变成一次全序列并行采样。

保存全部历史时，KV 内存近似为

```math
M_{\mathrm{KV}}
\propto
L\,N_{\mathrm{hist}}\,d_{\mathrm{KV}}\,b,
```

其中 $L$ 是层数，$N_{\mathrm{hist}}$ 是保留的历史 token 数，$d_{\mathrm{KV}}$ 是每层 KV 宽度，$b$ 是每元素字节数。长视频若无限保留历史，内存仍随时长线性增长。

### 7.2 block commit 与双向单元的缓存边界

若每次提交 $M$ 个 token，外层有 $K$ 次 commit，已提交历史通常可以缓存；但当前集合内部若反复 bidirectional refinement，其表示会随每轮改变，不能像已确定的历史那样一次计算后永久复用。

这给出一条实用判断：

- **past clean units**：若模型结构保持 causal 且条件不变，通常可复用；
- **current noisy/masked unit**：每轮状态变化，通常要重算或采用近似 feature cache；
- **future unknown units**：尚未生成，不能进入 causal cache；
- **prompt 发生变化**：旧 KV 未必与新条件一致，可能需要 recache。

### 7.3 三类 “内循环” 的真实代价

设外层有 $K$ 个 commit：

| 路线 | 外层串行深度 | 单元内循环 | 主要 backbone 调用 | 可缓存部分 |
|---|---:|---:|---|---|
| strict categorical token AR | $K$ | 1 | 每 commit 一次增量 Transformer | 已提交 token KV |
| set/frame masked refinement | $K$ | $R$ 轮 | 通常约 $K\times R$ 次当前单元更新 | 已提交单元；当前 mask 状态受限 |
| continuous token + small diffusion head | $K$ | $D$ 次小 head NFE | 约 $K$ 次 AR backbone，加 $K\times D$ 次小 head | 历史 AR KV；head 条件表示 |
| frame/chunk video diffusion | $K$ | $D$ 次 full-backbone NFE | 约 $K\times D$ 次大 backbone | clean history KV；当前 noisy 单元通常变化 |

所以报告效率时至少要同时给：

- commit 数 $K$；
- 单元大小与视频时长；
- refinement rounds 或 noise-time NFE；
- NFE 调用的是小 head 还是完整视频 backbone；
- guidance 是否导致额外前向；
- tokenizer/decoder 是否计入；
- 硬件、精度、batch、分辨率与端到端计时边界。

### 7.4 FlowCache 与 LongLive 改的是 cache / deployment 层

FlowCache 针对 autoregressive video diffusion 的不同 chunk 处于不同 denoising 状态这一现象，采用 chunkwise adaptive feature reuse，并用 importance–redundancy 策略压缩历史 KV 到有界预算 [[17]](#ref-17)。它是 training-free cache framework，不重新定义 AR factorization；作者报告的 2.38 倍与 6.7 倍加速分别绑定 MAGI-1、SkyReels-V2 及其配置。

LongLive 采用 frame-level causal AR，并加入 prompt 切换时的 KV-recache、短窗口 attention、frame-level attention sink 与 train-long–test-long 训练 [[16]](#ref-16)。其单 H100 上 20.7 FPS、最长 240 秒等数字是论文设置内的作者报告，不能与不同分辨率、模型大小、精度和解码口径的结果横比。

更完整的长期 memory、TTFF、deadline、jitter 与 SLO 讨论见[因果、流式与实时视频生成](causal-streaming-generation.md)。

## 🔀 8. AR、causal、few-step、streaming 与 real-time 是五个问题

| 术语 | 它真正规定什么 | 它不自动保证什么 |
|---|---|---|
| Autoregressive | 联合分布按变量或 block 的条件顺序分解 | 离散表示、交叉熵、causal video time、低延迟 |
| Causal in video time | 当前提交单元不读取未来视频单元 | strict next-token、结构因果理解、few-step |
| Few-step | noise/transport time 所需 NFE 较少 | 外层 commit 少、KV 有界、流式 deadline |
| Streaming | 结果按帧或 chunk 增量交付，系统可持续运行 | 每帧按时、低 jitter、长期不漂移 |
| Real-time | 在指定硬件和 SLO 下满足 TTFF 与逐帧 deadline | 方法在其他配置仍实时、质量无损 |

Causality in Video Diffusers is Separable from Denoising 进一步从架构上强调：视频时间的 causal reasoning 与 noise time 的 iterative denoising 可以拆开 [[19]](#ref-19)。这与本章的两时钟记号一致。

AR-Drag 在 few-step AR video diffusion 上加入 motion control 与 reinforcement learning [[18]](#ref-18)。它说明 reward/objective 可以叠加到既有 AR factorization 上；不是因为用了 RL，联合分解就变成了另一类 AR。

StreamDiffusionV2 则是 training-free streaming system，组合 rolling KV、noise controller、SLO-aware scheduler 与多 GPU pipeline [[20]](#ref-20)。它的贡献位于 deployment 层，不能用作“所有 streaming video diffusion 都是 AR”的依据。

读一篇 “autoregressive video diffusion” 论文时，建议写成五列：

| 系统 | 表示 | 数据时间 factorization | 单元内 objective / sampler | cache 与 deployment |
|---|---|---|---|---|
| VideoGPT | 离散 token | strict spacetime token AR | categorical sampling | 离线；标准 KV 可用 |
| NOVA | 连续 latent | frame AR + spatial set AR | per-token diffusion head | 历史帧 KV；双步数接口 |
| Lumos-1 | 离散 token | inter-frame causal | intra-frame discrete diffusion | 帧内并行 refinement |
| CausVid | 连续视频 latent | causal frame/chunk AR | DMD-distilled few-step video diffusion | KV streaming |
| Self Forcing | 连续视频 latent | causal frame/chunk AR | self-rollout training + few-step diffusion | rolling KV |
| FlowCache | 继承基础模型 | 继承基础模型 | 不改训练 objective | chunkwise reuse + bounded KV compression |
| StreamDiffusionV2 | 继承基础模型 | 继承基础模型 | training-free serving | scheduler + rolling KV + pipeline |

## 🗓️ 9. 2016–2026 里程碑：按概念变化，不按榜单排序

| 年份 | 工作 | 正式状态 | 里程碑 | 必须保留的边界 |
|---:|---|---|---|---|
| 2016 | PixelRNN [[1]](#ref-1) | ICML | 原始像素 strict AR | 图像证据，不是高分辨率视频可扩展性证明 |
| 2017 | Video Pixel Networks [[2]](#ref-2) | ICML | 四维视频像素依赖链 | 历史低分辨率设置 |
| 2017 | VQ-VAE [[3]](#ref-3) | NeurIPS | 用离散 latent 缩短生成序列 | representation，不规定上层 factorization |
| 2021 | VideoGPT [[4]](#ref-4) | arXiv preprint | VQ video tokenizer + strict GPT prior | 不得写成正式 venue |
| 2022 | MaskGIT [[5]](#ref-5) | CVPR | 多 token masked iterative decoding | 是对照路线，不是 strict next-token |
| 2023 | Phenaki [[6]](#ref-6) | ICLR | causal tokenizer 支持变长，masked generator 支持故事提示 | causal tokenizer 与 generator objective 分开 |
| 2024 | MAGVIT-v2 [[7]](#ref-7) | ICLR | lookup-free visual tokenizer，统一图像/视频码 | tokenizer 贡献不等于所有实验 strict AR |
| 2024 | VideoPoet [[8]](#ref-8) | ICML | decoder-only 多模态 discrete AR | 项目演示不等于开放模型 |
| 2024 | MAR [[9]](#ref-9) | NeurIPS | continuous token + per-token Diffusion Loss | 主要是图像证据 |
| 2025 | NOVA [[10]](#ref-10) | ICLR | non-quantized frame/set AR + diffusion head | AR steps 与 head diffusion steps 分开 |
| 2025 | MAGI [[11]](#ref-11) | CVPR | inter-frame causal + intra-frame masked；CTF | 不是 strict token AR；CTF 仍用真实历史 |
| 2025 | CausVid [[12]](#ref-12) | CVPR | bidirectional teacher → few-step causal student | teacher、DMD、factorization 与 KV 分栏 |
| 2025 | Self Forcing [[13]](#ref-13) | NeurIPS | 在 self-generated history 上训练 rollout | few-step 与 history source 是两项变化 |
| 2025 | InfinityStar [[14]](#ref-14) | NeurIPS | next-scale spacetime pyramid 的纯离散 generalized AR | 不是 strict next-token；速度与 720p 为作者配置报告 |
| 2026 | Lumos-1 [[15]](#ref-15) | ICLR | inter-frame AR + intra-frame bidirectional discrete diffusion | 明确不是 strict next-token AR |
| 2026 | LongLive [[16]](#ref-16) | ICLR | frame AR + recache/sink + long streaming | FPS 与时长不可跨配置横比 |
| 2026 | FlowCache [[17]](#ref-17) | ICLR | chunkwise denoising cache + bounded KV compression | cache 方法，不是新 factorization |
| 2026 | AR-Drag [[18]](#ref-18) | ICLR | few-step AR diffusion + motion reward/RL | objective 与 control 扩展 |
| 2026 | Separable Causal Diffusion [[19]](#ref-19) | CVPR | causal data time 与 denoising time 可拆 | causal 不等于 strict AR 或结构因果 |
| 2026 | StreamDiffusionV2 [[20]](#ref-20) | MLSys | 把 TTFF、deadline、jitter 与 pipeline 纳入系统 | deployment 证据，不定义 AR |

这张表中的质量、速度与分辨率数字没有被做成横向排行榜，因为数据、条件任务、时长、分辨率、tokenizer、模型规模、NFE、guidance、硬件和计时边界均不同。

## ✅ 10. 训练、推理与评测的最小报告协议

### 10.1 Representation

- 生成变量是 RGB、离散 code index 还是连续 latent？
- tokenizer 的时间与空间压缩率、词表大小和重建质量是多少？
- tokenizer 是否 causal；若是，这只属于 codec 还是也属于 generator？
- 生成质量是否与 tokenizer oracle reconstruction 分开报告？

### 10.2 Factorization 与 commit

- strict token、random set、frame 还是 chunk factorization？
- 单元内部是 causal、bidirectional、masked 还是 joint denoising？
- commit 后是否还会被后续轮次修改？
- 训练时可并行的 loss 位置与推理时必须串行的 commit 数分别是多少？

### 10.3 Objective 与教师

- 条件 head 是 categorical、diffusion、flow、discrete diffusion 还是 masked CE？
- 监督来自真实下一单元、bidirectional teacher、autoregressive teacher 还是 self-rollout？
- teacher forcing 使用完整历史还是 masked 历史？
- self forcing 是否对完整 rollout 反传；哪些状态 stop-gradient？

### 10.4 真实推理代价

- 外层 commit 数 $K$；
- 每 commit 的 refinement rounds / diffusion NFE $D$；
- $D$ 次调用的是小 head 还是 full backbone；
- classifier-free guidance 是否使实际 forward 数翻倍；
- KV cache 的精度、窗口、压缩、recache 与 eviction policy；
- 首帧、稳态帧延迟、端到端 FPS、jitter 与 deadline miss。

### 10.5 长期质量与模式覆盖

- 固定短片 FVD/VBench 之外，报告随 rollout 长度变化的身份、背景、色调、运动与语义漂移；
- 将 exposure bias、memory forgetting、chunk-boundary artifact 与 tokenizer error 分开诊断；
- 对少步蒸馏报告相同条件下的 teacher–student quality、diversity 与 mode coverage；
- 对 interactive 系统报告 prompt/action 改变后的响应延迟，而不仅是平均吞吐；
- 不把 causal attention 写成物理因果、可干预性或 world-model 正确性的证明。

## 🔍 11. 常见误读与快速修正

- **“离散 token 就是 AR。”** 离散 token 也可用于 MaskGIT、discrete diffusion；先查 factorization。
- **“Phenaki 是 causal AR generator。”** 它的 tokenizer 时间因果，上层 token generator 是 bidirectional masked Transformer。
- **“MAGVIT-v2 证明 AR 全面胜过 diffusion。”** 它首先是 tokenizer 论文，且正文包含 MLM 式迭代生成；要逐实验识别生成 objective。
- **“continuous AR 不再需要 diffusion。”** MAR/NOVA 正是用 diffusion head 表示连续 token 条件密度。
- **“InfinityStar 是逐 token AR。”** 它使用 spacetime-pyramid next-scale prediction，提交的是尺度，不是单个 token。
- **“4-step AR” 只有四次网络调用。** 还要乘外层 frame/chunk commit 数，并辨认小 head 与 full backbone。
- **“KV cache 消除了 AR 的串行性。”** 它消除历史重算，不消除下一 commit 对已生成前缀的依赖。
- **“CTF 已解决 exposure bias。”** CTF 修正 ground-truth 历史的完整性；self-generated history gap 仍需 on-policy rollout 等方法。
- **“causal = streaming = real-time。”** causal 是信息约束，streaming 是交付方式，real-time 是绑定硬件与 SLO 的实测结论。

## 🔗 12. 参考文献

<a id="ref-1"></a>[1] [Pixel Recurrent Neural Networks](https://proceedings.mlr.press/v48/oord16.html). van den Oord et al. ICML. 2016.

<a id="ref-2"></a>[2] [Video Pixel Networks](https://proceedings.mlr.press/v70/kalchbrenner17a.html). Kalchbrenner et al. ICML. 2017.

<a id="ref-3"></a>[3] [Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html). van den Oord et al. NeurIPS. 2017.

<a id="ref-4"></a>[4] [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). Yan et al. arXiv preprint. 2021.

<a id="ref-5"></a>[5] [MaskGIT: Masked Generative Image Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html). Chang et al. CVPR. 2022.

<a id="ref-6"></a>[6] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://openreview.net/forum?id=vOEXS39nOF). Villegas et al. ICLR. 2023.

<a id="ref-7"></a>[7] [Language Model Beats Diffusion - Tokenizer is key to visual generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html). Yu et al. ICLR. 2024.

<a id="ref-8"></a>[8] [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://proceedings.mlr.press/v235/kondratyuk24a.html). Kondratyuk et al. ICML. 2024.

<a id="ref-9"></a>[9] [Autoregressive Image Generation without Vector Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html). Li et al. NeurIPS. 2024.

<a id="ref-10"></a>[10] [Autoregressive Video Generation without Vector Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6e5112eaa45f8c30b242c5f576213a92-Abstract-Conference.html). Deng et al. ICLR. 2025.

<a id="ref-11"></a>[11] [Taming Teacher Forcing for Masked Autoregressive Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html). Zhou et al. CVPR. 2025.

<a id="ref-12"></a>[12] [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html). Yin et al. CVPR. 2025.

<a id="ref-13"></a>[13] [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html). Huang et al. NeurIPS. 2025.

<a id="ref-14"></a>[14] [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f832f6d70ea73779369142dac61a389f-Abstract-Conference.html). Liu et al. NeurIPS. 2025.

<a id="ref-15"></a>[15] [Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59ad89d72559dd4ce557d56f36313724-Abstract-Conference.html). Yuan et al. ICLR. 2026.

<a id="ref-16"></a>[16] [LongLive: Real-time Interactive Long Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html). Yang et al. ICLR. 2026.

<a id="ref-17"></a>[17] [Flow Caching for Autoregressive Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85dc8f85ff978b9c606d3b2f5b0da69a-Abstract-Conference.html). Ma et al. ICLR. 2026.

<a id="ref-18"></a>[18] [Real-Time Motion-Controllable Autoregressive Video Diffusion](https://proceedings.iclr.cc/paper_files/paper/2026/hash/71c1d6ec1f0003d8ea10bbea4291002d-Abstract-Conference.html). Zhao et al. ICLR. 2026.

<a id="ref-19"></a>[19] [Causality in Video Diffusers is Separable from Denoising](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html). Bai et al. CVPR. 2026.

<a id="ref-20"></a>[20] [StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html). Feng et al. MLSys. 2026.
