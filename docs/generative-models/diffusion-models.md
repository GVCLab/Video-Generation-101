# 扩散模型：从 DDPM、Score-SDE 到视频生成系统

扩散模型不是“把噪声逐帧变成视频”的单一算法，而是一套可以拆开的设计：用什么前向扰动定义训练分布，网络预测什么量，怎样把预测换算成 score 或去噪方向，推理时采用随机链还是确定性常微分方程，以及模型究竟在像素还是压缩表示中工作。DDPM 给出了现代离散去噪框架，Score-SDE 把它推广到连续随机过程并导出 probability-flow ODE（PF-ODE），DDIM、DPM-Solver 等则主要改变采样过程而非训练目标 [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3) [[6]](#ref-6)。

对视频而言，还必须再加一条独立的轴：画面中的物理时间。一次视频生成既可能沿视频帧推进，也会在每一帧或整段视频上沿噪声等级反复去噪。若把这两个“时间”都写成 $t$，就容易把视频自回归误认为反向扩散，或把少步去噪误认为流式输出。本章因此固定用 $k$ 表示视频时间，用 $\tau$ 表示噪声时间，并在这一符号约定下讨论训练、架构与部署。

## 1. 两条时间轴：视频时间 $k$ 与噪声时间 $\tau$

设一段含 $K$ 帧的干净视频为

```math
X_0=\left(x_0^{(1)},x_0^{(2)},\ldots,x_0^{(K)}\right).
```

上标 $`k\in\lbrace1,\ldots,K\rbrace`$ 表示第几帧，即现实画面怎样随时间演化；下标 $\tau\in[0,T]$ 表示这整段视频处在多强的噪声下。于是 $x_\tau^{(k)}$ 是“第 $k$ 帧在噪声等级 $\tau$ 的状态”，$X_0$ 是整段干净视频而不是第一帧，$X_T$ 则接近先验噪声。条件 $c$ 可以是文本、首帧、参考视频、深度、姿态、相机轨迹或动作。

![图 018：视频与噪声双时间轴](assets/imagegen-diagrams/018/diagram.png)
顺序化文字替代：第一，固定一个噪声等级 $\tau$，按 $k=1,\ldots,K$ 排列视频帧；第二，固定整段视频的帧索引结构，增加 $\tau$，把 $X_0$ 逐渐扰动为 $X_T$；第三，从 $X_T$ 出发降低 $\tau$；第四，可沿带随机项的 reverse SDE 或不带随机项的 PF-ODE 生成 $\hat X_0$；第五，在精确 score 与精确积分的理想条件下，两条反向过程共享各噪声时刻的边缘分布，但不共享逐样本轨迹。

这一区分也澄清了三个常见说法。其一，“一次生成 16 帧”描述的是 $K$；“采样 20 步”描述的是噪声轴离散点数。其二，frame-wise causal attention 限制的是 $k$ 方向的信息访问，并不规定 $\tau$ 方向采用 DDPM、ODE 还是少步蒸馏。其三，网络调用次数（number of function evaluations，NFE）减少，只说明噪声轴计算可能变少，不自动说明系统能在视频结束前持续发帧。

## 2. 从离散 DDPM 到 Score-SDE 与 PF-ODE

### 2.1 DDPM：前向链易采样，反向链需要学习

离散 DDPM 在噪声时间 $\tau=1,\ldots,T$ 上定义固定的前向马尔可夫链：

```math
q(X_\tau\mid X_{\tau-1})=
\mathcal N\!\left(\sqrt{1-\beta_\tau}X_{\tau-1},\beta_\tau I\right).
```

记 $\alpha_\tau=1-\beta_\tau$、$\bar\alpha_\tau=\prod_{s=1}^{\tau}\alpha_s$，则无需逐步执行前向链，就能直接构造任意噪声等级的训练样本：

```math
X_\tau=\sqrt{\bar\alpha_\tau}X_0+
\sqrt{1-\bar\alpha_\tau}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
```

原始 DDPM 学习的反向转移不是一个确定函数，而是高斯条件分布：

```math
p_\theta(X_{\tau-1}\mid X_\tau,c)=
\mathcal N\!\left(\mu_\theta(X_\tau,\tau,c),
\Sigma_\theta(X_\tau,\tau,c)\right).
```

因此，ancestral DDPM 的一次反向更新通常还要采样随机项。把反向过程写成 $X_{\tau-1}=g_\theta(X_\tau)$ 会漏掉这一点；只有在明确采用确定性 DDIM、PF-ODE 或其他确定性 sampler 时，这种简写才不误导。DDPM 原论文从变分界出发，并展示了它与 denoising score matching 的联系 [[1]](#ref-1)。

### 2.2 从噪声预测得到 score

用更一般的连续高斯扰动写法，令

```math
X_\tau=\alpha(\tau)X_0+\sigma(\tau)\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
```

给定干净样本时，扰动核的 score 是

```math
\nabla_{X_\tau}\log q(X_\tau\mid X_0)
=-\frac{X_\tau-\alpha(\tau)X_0}{\sigma(\tau)^2}
=-\frac{\epsilon}{\sigma(\tau)}.
```

若网络以均方误差预测噪声，其最优输出是条件均值 $\mathbb E[\epsilon\mid X_\tau,c]$。因此可将网络换算为扰动后条件分布 $p_\tau(X_\tau\mid c)$ 的边缘 score 估计：

```math
s_\theta(X_\tau,\tau,c)
\approx -\frac{\epsilon_\theta(X_\tau,\tau,c)}{\sigma(\tau)}
\approx \nabla_{X_\tau}\log p_\tau(X_\tau\mid c).
```

这里的关键不是“噪声模型变成了另一类 score 模型”，而是给定同一噪声 schedule 后，两者是同一个统计场的不同参数化。有限数据、有限容量与不同 loss weighting 会让训练结果不再完全等价，但分类时不应把它们列成互斥家族。

### 2.3 连续前向 SDE 与随机 reverse SDE

Score-SDE 把离散加噪链写成前向 Itô 随机微分方程 [[2]](#ref-2)：

```math
\mathrm dX=f(X,\tau)\,\mathrm d\tau+g(\tau)\,\mathrm dW_\tau,
\qquad \tau:0\rightarrow T.
```

生成时从 $T$ 向 $0$ 积分。对应的 reverse-time SDE 为

```math
\mathrm dX=\left[f(X,\tau)-g(\tau)^2
s_\theta(X,\tau,c)\right]\mathrm d\tau
+g(\tau)\,\mathrm d\bar W_\tau,
\qquad \tau:T\rightarrow0.
```

$\bar W_\tau$ 是反向时间的 Wiener 过程，所以这条生成路径仍是随机的。公式中的 $\mathrm d\tau$ 按 $T\rightarrow0$ 的方向积分；若改用正向增长的新时间变量，漂移项符号也要随变量变换调整，不能只抄公式而忽略方向约定。

### 2.4 PF-ODE：同一 score 的确定性生成路径

同一组前向 SDE 与 score 还定义 probability-flow ODE：

```math
\mathrm dX=\left[f(X,\tau)-\frac{1}{2}g(\tau)^2
s_\theta(X,\tau,c)\right]\mathrm d\tau,
\qquad \tau:T\rightarrow0.
```

PF-ODE 没有 Wiener 随机项，因此给定初始噪声和确定性求解器后，轨迹是确定的。在 score 精确、初始分布正确且连续方程被精确求解的理想条件下，reverse SDE 与 PF-ODE 在每个 $\tau$ 上具有相同边缘分布；这不表示它们对同一个初始噪声产生相同样本，也不表示粗离散、近似 score 下仍完全等价 [[2]](#ref-2)。由此可见，“diffusion 等于随机链、flow 等于确定 ODE”不是可靠分界：score-based diffusion 自身就同时拥有随机 SDE 和确定 ODE。

## 3. $\epsilon$、$X_0$、$v$、score 与训练权重

### 3.1 四种常见预测量怎样换算

同一个网络骨干可以输出不同目标。下表采用
$X_\tau=\alpha_\tau X_0+\sigma_\tau\epsilon$，并暂时省略条件 $c$。

| 参数化 | 网络目标 | 从输出恢复其他量 | 需要注意 |
|---|---|---|---|
| $\epsilon$ prediction | 预测加入的 $\epsilon$ | $\hat X_0=(X_\tau-\sigma_\tau\hat\epsilon)/\alpha_\tau$；$\hat s=-\hat\epsilon/\sigma_\tau$ | 接近 $\sigma=0$ 时换算需处理数值稳定性 |
| $X_0$ prediction | 预测干净样本 | $\hat\epsilon=(X_\tau-\alpha_\tau\hat X_0)/\sigma_\tau$ | 高噪声区直接恢复细节更难 |
| score prediction | 预测 $\nabla_X\log p_\tau(X)$ | $\hat\epsilon=-\sigma_\tau\hat s$ | score 的尺度随 $\sigma_\tau$ 显著变化 |
| $v$ prediction | $v=\alpha_\tau\epsilon-\sigma_\tau X_0$ | VP 归一化下，$\hat X_0=\alpha_\tau X_\tau-\sigma_\tau\hat v$，$\hat\epsilon=\sigma_\tau X_\tau+\alpha_\tau\hat v$ | 必须声明 $v$ 的符号与 schedule 约定 |

最后一行使用 variance-preserving（VP）约定 $\alpha_\tau^2+\sigma_\tau^2=1$。若不满足这一归一化，逆变换要除以 $\alpha_\tau^2+\sigma_\tau^2$。Progressive Distillation 系统讨论了有利于少步训练稳定性的参数化，Imagen Video 随后在视频级联系统中使用 $v$ prediction [[4]](#ref-4) [[10]](#ref-10)。因此，看到论文写“velocity prediction”不能据此判定它采用 rectified flow；diffusion 的 $v$、flow matching 的 vector field 与 rectified-flow velocity 是三个需要由 objective 和路径定义区分的概念。

### 3.2 Schedule、采样分布和 loss weighting 是三件事

一个统一的回归目标可写为

```math
\mathcal L=
\mathbb E_{X_0,c,\tau,\epsilon}
\left[w(\tau)\left\|y_\tau-
y_\theta(X_\tau,\tau,c)\right\|_2^2\right],
```

其中 $y_\tau$ 可以是 $\epsilon$、$X_0$、$v$ 或经过预条件的等价目标。训练时至少有三个独立选择：

1. **Noise schedule** $\alpha(\tau),\sigma(\tau)$ 决定每个噪声时刻的信噪比
   $\mathrm{SNR}(\tau)=\alpha(\tau)^2/\sigma(\tau)^2$，也决定数据怎样被扰动。
2. **Noise-time sampling** $p_{\mathrm{train}}(\tau)$ 决定优化器多频繁看到各噪声区间。
3. **Loss weighting** $w(\tau)$ 决定相同出现频率下，各区间的误差对参数更新贡献多大。

参数化可代数换算，不等于优化问题相同。例如由 $\epsilon$ 与 $X_0$ 的换算可得

```math
\|\epsilon-\hat\epsilon\|_2^2
=\frac{\alpha_\tau^2}{\sigma_\tau^2}
\|X_0-\hat X_0\|_2^2
=\mathrm{SNR}(\tau)\|X_0-\hat X_0\|_2^2.
```

所以“不加权的 $\epsilon$ MSE”在 $X_0$ 误差坐标中已经隐含 SNR 权重。EDM 将噪声尺度、网络预条件、训练权重、采样 schedule 与数值求解器拆成模块，说明高质量系统不能只报告“用了哪一种预测头” [[5]](#ref-5)。实践复现至少要记录 $\alpha/\sigma$ 定义、$\tau$ 的采样分布、loss weight、数据归一化和推理噪声网格。

## 4. 训练目标与 sampler 必须分开

### 4.1 相同网络可以接不同 sampler

训练目标回答“网络拟合什么统计量”，sampler 回答“推理时如何沿已学到的场从噪声走向数据”。二者相关，但不是同一层。

| 方法 | 推理对象 | 随机性 | 是否通常需要重训 | 正确归类 |
|---|---|---:|---:|---|
| DDPM ancestral sampling | 离散反向高斯条件链 | 是 | 否，使用原模型 | 随机 sampler |
| DDIM，$\eta=0$ | 与 DDPM 训练边缘相容的非马尔可夫生成过程 | 否 | 否 | 确定性 sampler [[3]](#ref-3) |
| Reverse-SDE solver | 连续 reverse-time SDE | 是 | 否 | 随机数值求解 |
| PF-ODE solver | diffusion 对应的 probability-flow ODE | 否 | 否 | 确定性数值求解 |
| DPM-Solver | diffusion ODE 的半线性结构 | 否 | 否 | training-free 专用高阶 solver [[6]](#ref-6) |

DDIM 与 PF-ODE 都能给出确定性路径，但来源不同：DDIM 从与 DDPM 训练目标相容的非马尔可夫过程构造采样，PF-ODE 从连续 SDE 的 Fokker–Planck 关系导出。将二者笼统写成“去掉随机项”会掩盖各自的方程与离散误差。

DPM-Solver 在不重新训练 score/denoiser 的前提下求解 diffusion ODE，并在其论文设置中用约 10–20 次网络调用获得高质量图像样本 [[6]](#ref-6)。这类数字不能直接外推到视频：视频网络单次前向更昂贵，guidance 可能增加前向次数，不同阶 solver 还可能在一个名义 step 内调用多次模型。比较速度时应同时报告 NFE、wall-clock、分辨率、帧数、batch、精度和硬件。

### 4.2 换 schedule 也不等于换 objective

训练 noise schedule 决定模型见过的扰动族；推理 schedule 则选择从 $T$ 到 $0$ 的离散节点。只要 sampler 与模型参数化兼容，可以在不重训的情况下改变节点数量和位置，但少步时数值误差、score 误差与 guidance 误差会共同放大。反过来，蒸馏后的少步 student 往往已经改变训练目标和参数，不能再称为“只是换了一个 solver”。

## 5. Pixel、latent 与 tokenizer 的真实上限

### 5.1 Pixel diffusion

Pixel diffusion 直接在 RGB 视频张量上学习 $p(X)$。它没有独立 codec 的信息瓶颈，但张量维度随帧数、空间分辨率和通道数增长；高分辨率、长视频的每次网络调用和中间激活都很昂贵。没有 codec 不代表没有架构瓶颈，也不表示逐像素生成一定更忠实，因为网络容量、训练数据与 sampler 误差仍会限制结果。

### 5.2 Latent video diffusion

Latent diffusion 先训练编码器 $E$ 与解码器 $D$：

```math
Z_0=E(X_0),\qquad X_{\mathrm{rec}}=D(Z_0),
```

再在低维连续表示 $Z_\tau$ 上执行扩散。Latent Diffusion Models 将这一思路系统化，Video LDM 则把预训练图像 LDM 加入时间层并用于高分辨率视频 [[8]](#ref-8) [[12]](#ref-12)。压缩降低了 denoiser 的时空成本，却把最终误差拆成两部分：生成器能否拟合 latent 分布，以及 decoder 能否把 latent 恢复成连续视频。

所谓“tokenizer 上限”应精确理解为**忠实可恢复信息的上限**。所有输出都落在 decoder 可表达的范围内；若 $E$ 已经丢掉小文字、细物体、快速运动或高频纹理，latent denoiser 无法从同一个编码中可靠恢复这些被丢弃的信息。它不是主观美学分数的数学上界，因为 decoder 仍可能生成看似锐利的新纹理；它约束的是输入信息能否被保真保留。因此，在比较 latent generator 之前，应先在目标分辨率、帧率和时长上报告 $D(E(X))$ 的重建质量与时间一致性，并冻结 latent 的 shape、dtype、时空网格及元素/token 预算。没有概率模型、熵编码器和可解码 bitstream 时，不报告 bpp 或 bitrate。

### 5.3 Representation 不等于 objective、factorization 或 backbone

连续 VAE/AE latent、VQ 离散 token 与 spacetime patch 是不同的 **representation** 接口；diffusion/score/flow 是 **objective 与采样路径**，autoregressive/masked/joint 是 **factorization**，DiT 则是 **backbone**。同一种连续 latent 可交给 diffusion 或 flow，同一种离散 token 可交给 AR、masked prediction 或 discrete diffusion。把连续 latent 切成 spacetime patches 只是给 Transformer 组织输入，并不会自动把它变成 VQ 离散 token。

本章只负责这些表示如何接入 diffusion。连续/离散/结构化类型、量化、预算口径、真实 codec 与 tokenizer 替换实验统一见[视频 Tokenizer 与生成式压缩](video-tokenizers.md)。Stable Video Diffusion 是连续 latent video diffusion 的公开技术报告实例 [[14]](#ref-14)；Sora 的机构技术报告则明确披露了视频压缩、spacetime patch 与 Transformer diffusion 的组合，但没有公开足以复现训练数据、完整注意力结构、参数量、sampler 和成本的细节 [[16]](#ref-16)。

## 6. 视频时空架构与条件控制

本节只说明 diffusion denoiser 的接口与历史代表；Video DiT 的 token 公式、full/factorized/window/sparse/linear/hybrid topology、position/fusion、MoE、并行和 cache 统一由[骨干扩展专章](video-dit-backbones.md)维护。

### 6.1 Backbone 只实现 denoiser，不定义概率家族

视频 denoiser 接收 $X_\tau$ 或 $Z_\tau$、噪声时间 $\tau$ 与条件 $c$。它必须在同一次或多次前向中处理空间结构、帧间运动和条件信息，但网络形态并不决定训练 objective。

| 架构模式 | 时空处理方式 | 代表证据 | 主要取舍 |
|---|---|---|---|
| 3D U-Net | 时空卷积与注意力直接处理固定片段 | Video Diffusion Models [[9]](#ref-9) | 局部时空建模直接，但长片段激活昂贵 |
| 图像 backbone + temporal layers | 复用图像权重，在层间加入时间卷积/注意力 | Make-A-Video、Video LDM、SVD [[11]](#ref-11) [[12]](#ref-12) [[14]](#ref-14) | 迁移图像先验容易，但时间模块仍需视频训练 |
| Space-Time U-Net | 一次覆盖完整短视频时间范围 | Lumiere [[15]](#ref-15) | 减少关键帧后插帧的分段设计，但仍受固定窗口限制 |
| DiT / spacetime Transformer | 把 latent patch 当作序列，由 full/factorized/window/sparse/linear/hybrid mixer 预测目标 | DiT、W.A.L.T.、CogVideoX、HunyuanVideo 等；详见[专章](video-dit-backbones.md) | 只有 global dense attention 的 score 主项随 token 数二次增长；其他 mixer 另有表达、state、selector 与 kernel 代价 |
| Recurrent / state-space backbone | 用递归/压缩状态传播历史 | 因果长视频与 hybrid 路线 | 状态容量、遗忘与暴露偏移需独立检验 |

DiT 是 backbone，不是与 diffusion、flow matching 并列的生成目标。原始 DiT 在图像 latent patches 上验证了 Transformer denoiser 的扩展性 [[13]](#ref-13)；视频工作把 patch 扩展到空间—时间维，但仍需另行说明 objective、attention topology/mask、表示空间与 sampler。Causal frame/chunk 属于 factorization/information mask，cascade 属于系统组合，rolling/inter-step cache 属于 execution；它们不再伪装成互斥 backbone 行。

### 6.2 条件怎样进入模型

文本通常先经文本编码器变成 token，再通过 cross-attention、联合 self-attention 或调制层注入 denoiser。首帧、参考图和源视频可以编码后与 noisy latent 拼接，作为额外 cross-attention memory，或经 adapter/ControlNet 类分支提供特征。深度、姿态、光流、相机轨迹与动作则可以栅格化、tokenize 或映射为调制信号。条件接口相同不代表模型学到相同能力：文本可控、几何轨迹可控、动作导致正确状态转移是逐级更强的声明。

Classifier-free guidance（CFG）在训练时随机丢弃条件，使一个网络同时学习 conditional 与 unconditional 预测；推理时按本章约定组合为 [[7]](#ref-7)：

```math
\epsilon_{\mathrm{cfg}}
=\epsilon_\theta(X_\tau,\tau,\varnothing)
+s\left[
\epsilon_\theta(X_\tau,\tau,c)
-\epsilon_\theta(X_\tau,\tau,\varnothing)
\right].
```

这里 $s=1$ 恢复 conditional 预测，$`s\gt1`$ 是向条件方向外推。另一种常见写法 $(1+w)\epsilon_c-w\epsilon_u$ 与本式对应 $s=1+w$；比较配置时不能混用两个 scale。CFG 提供质量/条件强度与覆盖度之间的可调折中，不会免费提升所有维度。对视频还要检查条件是否贯穿整个 $k=1\ldots K$，而不是只在首帧或少量关键帧上成立。

## 7. 训练难点与三类加速路线

### 7.1 视频训练不只是把图像 batch 增加一维

视频数据包含镜头切换、字幕、水印、重复片段、可变帧率、变速和弱文本描述。训练管线要显式记录镜头切分、去重、质量与运动过滤、caption 来源、时长/宽高比分桶及帧采样策略。Stable Video Diffusion 将图像预训练、视频预训练和高质量视频微调分成阶段，并把数据筛选作为核心变量 [[14]](#ref-14)。这类作者结论说明数据流程需要报告，但不能推出某一套阈值对所有数据域都最优。

图像—视频联合训练能复用静态语义并改善优化效率；Video Diffusion Models 在其设置中报告了联合训练降低 minibatch 梯度方差并加速优化 [[9]](#ref-9)。然而，静态样本无法提供真实运动监督，所以联合比例、帧重复方式与运动分布仍需消融。短 clip 训练也不会自动产生长视频状态保持；长时能力还依赖 data-time factorization、上下文长度、分段锚定、记忆和部署时的自生成历史。

### 7.2 三类加速不能合并成“少步扩散”

这里先按“是否训练新参数、主要对齐轨迹还是分布”做最小分叉；若要同时判断 DDPM/score、reverse SDE、PF-ODE、FM/RF、CM/DMD 与 causal streaming 分别处在哪一层，请对照 [Flow 专章的五层机制地图](flow-consistency-models.md#five-layer-map)。核心原则是：输出参数化不等于训练 objective，确定性动力学不等于 FM，不重训的 solver 不等于少步 student，少 NFE 也不等于视频能持续发帧。

![图 019：扩散生成三类加速路线](assets/imagegen-diagrams/019/diagram.png)
顺序化文字替代：第一，从已经训练好的 diffusion 或 score 场出发；第二，若不能重训，使用兼容的 DDIM、PF-ODE 或 DPM-Solver 等 sampler/solver；第三，若允许训练新模型，再判断监督主要对齐同一生成轨迹上的端点，还是对齐 student 与目标的生成分布；第四，前者进入 consistency/trajectory 路线，后者进入 DMD 或带对抗项的 DMD2 路线；第五，三条路线都必须在相同分辨率、时长、条件、NFE、硬件和精度下重新验收质量、覆盖与延迟。

#### Training-free sampler 与 solver

DDIM、PF-ODE solver 和 DPM-Solver 复用已经学到的场，不训练新 student。它们消除的是冗余或低阶的数值积分步骤，主要风险是低 NFE 下的离散误差。DPM-Solver 是明确面向 diffusion ODE 结构的高阶方法 [[6]](#ref-6)。这一路线最适合先建立“同一 checkpoint、不同 solver”的质量—NFE—延迟曲线。

#### Consistency trajectory

Consistency Model 学习函数 $F_\theta(X_\tau,\tau)$，使同一 PF-ODE 轨迹上不同噪声点映射到一致的数据端：

```math
F_\theta(X_\tau,\tau)\approx F_\theta(X_\rho,\rho),
\quad X_\tau,X_\rho\text{ 位于同一 PF-ODE 轨迹}.
```

它可以从预训练 diffusion 蒸馏，也可以 standalone training；一步是设计目标，多步 refinement 仍可换取质量 [[17]](#ref-17)。Progressive Distillation 则更早地把一个 $N$ 步确定性 teacher sampler 逐轮压缩为 $N/2$ 步 student [[4]](#ref-4)。二者都利用轨迹监督，但 loss 和训练程序不是同义词。

2026 年的 score-regularized continuous-time consistency model（rCM）把连续时间 consistency 扩展到大规模图像与视频 diffusion teacher，并加入 score distillation 作为长跳正则。论文在 Cosmos-Predict2、Wan2.1、最高 14B 参数和 5 秒视频的作者设置中报告 1–4 步以及 15–50 倍采样加速 [[22]](#ref-22)。这些是特定模型、数据、硬件和评测下的结果；“缓解 mode collapse”不应改写成“证明所有 consistency 都无覆盖损失”。

#### DMD 与 adversarial distillation

Distribution Matching Distillation（DMD）不要求 student 沿 teacher 的同一条样本轨迹前进，而是利用 target score 与 fake/student score 的差来更新生成器，使 student 分布接近目标分布；原始 DMD 还使用回归项稳定一步训练 [[18]](#ref-18)。DMD2 去掉对固定回归数据集的依赖，引入 two-time-scale fake critic 更新、GAN loss，并扩展到 multi-step/on-policy 输入 [[19]](#ref-19)。因此，“DMD”与“adversarial distillation”有交集但并非天然同义：对抗项是 DMD2 等具体实现新增的训练信号。

CausVid 把 DMD 扩展到视频，用双向 diffusion teacher 监督 4-step causal student，并通过 KV cache 流式生成 [[20]](#ref-20)。Self Forcing 则让 causal video diffusion 在训练时条件于自身已经生成的历史，以处理 teacher-forcing 与 rollout 之间的 exposure gap [[21]](#ref-21)。这两项工作同时改变了噪声轴上的步数和视频轴上的 factorization；不能把其 streaming 能力全部归因于 DMD，也不能把 causal mask 当作新的 diffusion objective。完整部署问题见[因果、流式与实时视频生成](causal-streaming-generation.md)。

## 8. 2022–2026：视频扩散的可证据化里程碑

下表选择改变表示、架构、训练或部署问题定义的节点，而不是做模型排行榜。A 表示正式同行评审 proceedings，B 表示作者论文或机构技术报告。不同工作使用的数据、提示词、分辨率、时长、采样预算和硬件不同，表中的任何数字都不能直接横向排名。

| 年份 | 工作与证据等级 | 可核验的技术转折 | 证据边界 |
|---:|---|---|---|
| 2022 | Video Diffusion Models，NeurIPS，A [[9]](#ref-9) | 以 3D U-Net 扩展图像 diffusion，覆盖生成、预测和插帧，并研究图像—视频联合训练与时空扩展 | 证明该架构在论文任务有效，不是所有长视频系统的统一方案 |
| 2022 | Imagen Video，作者技术报告，B [[10]](#ref-10) | 采用视频 diffusion cascade、空间/时间超分、$v$ prediction、CFG 与 progressive distillation | 未经正式会议同行评审；级联配置与速度结论只限其报告 |
| 2023 | Make-A-Video，ICLR，A [[11]](#ref-11) | 从文本—图像先验迁移语义，用无配对视频学习运动，并采用分解时空模块与多级生成 | 不使用成对文视频数据不等于不使用视频，也不证明数据问题已解决 |
| 2023 | Video LDM，CVPR，A [[12]](#ref-12) | 将图像 latent diffusion 加入时间层，联合时间一致的 decoder/upsampler，确立高分辨率 latent video diffusion 路线 | 生成上限仍受 codec 重建和各级超分约束 |
| 2023 | DiT，ICCV，A，图像侧桥梁 [[13]](#ref-13) | 证明 Transformer 可作为 latent diffusion denoiser，并以 latent patches 扩展 | 是图像证据；视频的时间 patch、attention 与成本需另证 |
| 2023 | Stable Video Diffusion，作者技术报告，B [[14]](#ref-14) | 明确图像预训练、视频预训练、高质量视频微调和数据筛选阶段，并发布模型与代码 | 技术报告不是正式 venue；其数据 recipe 不能无条件推广 |
| 2024 | Lumiere，SIGGRAPH Asia，A [[15]](#ref-15) | Space-Time U-Net 在一个模型中覆盖完整短片时间范围，避免先生成稀疏关键帧再插帧的主路径 | 只支持特定窗口内的 joint generation，不证明长时或流式优势 |
| 2024 | Sora，机构技术报告，B [[16]](#ref-16) | 披露压缩视频表示、spacetime patches、Transformer diffusion 与可变视频尺寸训练思路 | 未披露足够的模型、数据、attention、sampler、成本和复现细节 |
| 2025 | CausVid，CVPR，A [[20]](#ref-20) | 将 50-step 双向 video diffusion teacher 蒸馏为 4-step causal student；论文报告 1.3 秒初始延迟和单 GPU 约 9.4 FPS | 数字依赖论文模型、分辨率、硬件与计时口径；causal 仅指时间信息方向 |
| 2025 | Self Forcing，NeurIPS，A [[21]](#ref-21) | 训练时用自生成历史和 rolling KV cache，直接处理 causal rollout 的 exposure gap | 作者报告的单 GPU sub-second latency 不是跨系统通用保证 |
| 2026 | rCM，ICLR，A [[22]](#ref-22) | 把 JVP-based continuous-time consistency 扩展到最高 14B 和视频任务，以 score regularization 改善细节与覆盖 | 1–4 步、15–50 倍是作者设置结果；不是视频质量的全局结论 |
| 2026 | Separable Causal Diffusion，CVPR，A [[23]](#ref-23) | 将 once-per-frame 的因果时间推理与 multi-step frame-wise diffusion rendering 分开 | 论文中的 causality 明确指 temporal arrow-of-time，不是结构因果或物理理解 |
| 2026 | StreamDiffusionV2，MLSys，A [[24]](#ref-24) | 把 TTFF、逐帧 deadline、jitter、rolling KV、SLO-aware batching 和多 GPU pipeline 纳入同一流式系统 | 作者数字使用最多 4×H100；training-free pipeline 不表示没有基础模型和硬件成本 |

这条历史不是“U-Net 被 DiT 淘汰、diffusion 被少步模型淘汰”的直线。更准确的理解是：2022–2023 年解决视频化、压缩与时空骨干；2024 年扩大 joint spatiotemporal 建模规模；2025 年把高质量双向 teacher 转成因果少步 student；2026 年进一步分离时间推理与去噪计算，并把算法速度升级为带 TTFF、deadline 和 jitter 的服务目标。

## 9. 如何读结果：质量、速度与能力声明

扩散视频的实验至少应同时报告以下信息：

| 维度 | 最低记录项 | 避免的误读 |
|---|---|---|
| 任务与条件 | T2V、I2V、V2V、预测或动作条件；完整 prompt/参考/控制 | 把编辑或预测结果当开放生成结果 |
| 视频范围 | 分辨率、帧数、FPS、时长、是否分段拼接 | 把短 clip 质量外推到长视频 |
| 表示 | Pixel/continuous latent/VQ；shape、dtype、时空网格、元素/token 预算与 $D(E(X))$ 重建；仅有 bitstream 时报告 bpp/bitrate | 把 decoder 伪影归因于 denoiser，或把 tensor shape 当作码率 |
| 训练 | $\alpha/\sigma$ schedule、$p(\tau)$、target、weight、数据阶段 | 只写“使用 diffusion loss” |
| 采样 | sampler、节点、NFE、随机性、CFG scale、seed 数 | 把名义 step 当真实网络调用 |
| 系统 | GPU 型号与数量、精度、batch、编译/量化、是否含 VAE 解码 | 把多卡吞吐当单样本延迟 |
| 输出质量 | 外观、运动、条件、身份/对象持续性、物理、覆盖与失败率 | 用一张最佳帧代替整段评测 |
| 流式声明 | TTFF、逐帧延迟分布、jitter、deadline miss、峰值显存 | 用平均 FPS 证明实时 SLO |

“生成的视频很逼真”只支持感知质量；它不证明模型具有可干预、可规划的 world-model 能力。“Causal video diffusion”通常只表示第 $k$ 帧或块不读取未来输出；它不证明物理因果正确。“Four-step”只说明噪声轴 NFE 较低；它不证明视频轴能够增量生成。统一评测协议见[视频生成与世界模型评测](../evaluation.md)，因果与 SLO 的严格定义见[因果、流式与实时视频生成](causal-streaming-generation.md)。

最后，用五个问题可以快速判定一篇“视频 diffusion”论文真正改变了什么：它在哪个表示空间扩散？视频时间 $k$ 是 joint、causal 还是分块？网络预测 $\epsilon$、$X_0$、$v$ 还是 score？推理使用什么 sampler、多少 NFE？速度主张是否包含端到端硬件与 SLO？只有把五个答案补全，模型名称才不会替代技术事实。

## 参考文献

<a id="ref-1"></a>[1] [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html). Jonathan Ho, Ajay Jain, Pieter Abbeel. NeurIPS. 2020.

<a id="ref-2"></a>[2] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. ICLR Oral. 2021.

<a id="ref-3"></a>[3] [Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP). Jiaming Song, Chenlin Meng, Stefano Ermon. ICLR. 2021.

<a id="ref-4"></a>[4] [Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI). Tim Salimans, Jonathan Ho. ICLR. 2022.

<a id="ref-5"></a>[5] [Elucidating the Design Space of Diffusion-Based Generative Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html). Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. NeurIPS. 2022.

<a id="ref-6"></a>[6] [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html). Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu. NeurIPS. 2022.

<a id="ref-7"></a>[7] [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). Jonathan Ho, Tim Salimans. NeurIPS Workshop on Deep Generative Models and Downstream Applications. 2021; author manuscript revised 2022.

<a id="ref-8"></a>[8] [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html). Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. CVPR. 2022.

<a id="ref-9"></a>[9] [Video Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-10"></a>[10] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho, William Chan, Chitwan Saharia, et al. Author technical report. 2022.

<a id="ref-11"></a>[11] [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://openreview.net/forum?id=nJfylDvgzlq). Uriel Singer, Adam Polyak, Thomas Hayes, et al. ICLR. 2023.

<a id="ref-12"></a>[12] [Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.html). Andreas Blattmann, Robin Rombach, Huan Ling, et al. CVPR. 2023.

<a id="ref-13"></a>[13] [Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html). William Peebles, Saining Xie. ICCV. 2023.

<a id="ref-14"></a>[14] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, et al. Author technical report. 2023.

<a id="ref-15"></a>[15] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://doi.org/10.1145/3680528.3687614). Omer Bar-Tal, Hila Chefer, Omer Tov, et al. SIGGRAPH Asia Conference Papers. 2024.

<a id="ref-16"></a>[16] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Institution technical report. 2024.

<a id="ref-17"></a>[17] [Consistency Models](https://proceedings.mlr.press/v202/song23a.html). Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever. ICML. 2023.

<a id="ref-18"></a>[18] [One-Step Diffusion with Distribution Matching Distillation](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html). Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman. CVPR. 2024.

<a id="ref-19"></a>[19] [Improved Distribution Matching Distillation for Fast Image Synthesis](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html). Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman. NeurIPS. 2024.

<a id="ref-20"></a>[20] [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html). Tianwei Yin, Qiang Zhang, Richard Zhang, et al. CVPR. 2025.

<a id="ref-21"></a>[21] [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html). Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman. NeurIPS. 2025.

<a id="ref-22"></a>[22] [Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html). Kaiwen Zheng, Yuji Wang, Qianli Ma, et al. ICLR. 2026.

<a id="ref-23"></a>[23] [Causality in Video Diffusers is Separable from Denoising](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html). Xingjian Bai, Guande He, Zhengqi Li, et al. CVPR. 2026.

<a id="ref-24"></a>[24] [StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html). Tianrui Feng, Zhi Li, Shuo Yang, et al. MLSys. 2026.
