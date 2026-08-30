# 变分随机视频生成研究轨迹（2026-08-30）

> 范围：为 `docs/generative-models/variational-generation.md` 提供可复核证据。本文只把“训练时可看真实未来/新观测的 posterior、部署时不看未来的 generative prior（固定或只看历史/合法条件），以及二者之间的变分对齐”计入严格主线；普通 video VAE tokenizer、纯 diffusion/flow、确定性 latent、GAN 噪声和没有生成 prior 的 latent action 均单独分流。

## 1. 研究问题

本轮围绕十个问题展开：

1. 如何用统一信息合同区分 stochastic future latent、representation tokenizer、belief state、latent action 与 diffusion noise？
2. 从 fixed prior 到 learned conditional prior，真正改变了哪一个部署接口？
3. global、per-step、hierarchical、clockwork、object/module 与 particle latent 分别解决什么问题？
4. 模糊来自 likelihood 假设、容量不足、近似后验不足还是 rollout 漂移，如何分账？
5. posterior collapse、amortization gap 与 prior–posterior gap 如何用不同实验诊断？
6. 2015–2026 的严格里程碑、首次公开时间与正式发表状态是什么？
7. best-of-100、FVD、下游控制 return 分别能和不能证明什么？
8. 2023 后开放域生成迁移到 latent diffusion/flow，VAE 在系统中为何仍存在但角色已改变？
9. 2025–2026 哪些新系统是真正的变分随机动力学，哪些只是名字中带 latent/VAE？
10. 如何把这些结论变成一个有命名数据、等预算分叉、阈值与否证条件的最小实验？

## 2. 检索表面、日期与证据等级

专项核验于 **2026-08-30（Asia/Shanghai）** 执行。搜索结果只用于发现，最终事实回到正式 proceedings、出版社页面、作者 arXiv 稿或官方仓库/项目页。

| 检索表面 | 代表检索对象 | 用途 |
|---|---|---|
| arXiv | VRNN、SV2P、SVG-LP、SRVP、FitVid、RSP、LPWM | 核对 v1 首次公开时间、作者稿和 preprint 状态 |
| PMLR | SVG-LP、PlaNet、SRVP、VIM | 核对正式 venue、摘要、页码和作者实验合同 |
| CVF Open Access | Improved Conditional VRNN、S3VAE、GHVAE、SLAMP、NUQ、IV-VAE | 核对 accepted version、机制和表格协议 |
| NeurIPS proceedings | VRNN、Large Stochastic RNN、CW-VAE、STORM | 核对正式节点与时序/控制机制 |
| ICLR / OpenReview | SV2P、VPR、VP²、LPWM | 核对接收状态、posterior/prior 与 benchmark 合同 |
| Nature | DreamerV3 | 核对 2025 正式发表、RSSM 目标和控制证据边界 |
| ScienceDirect / PubMed | 2026 implicit hierarchical temporal–spatial residual model | 核对 journal 状态、DOI、摘要与数据口径 |
| 官方项目页 / GitHub | SAVP、FitVid、LPWM | 核对代码、checkpoint、数据入口；不把 README 的排名当独立复现 |

证据等级：

- **A**：正式 proceedings 或出版社论文；可支撑 venue、方法和论文内实验。
- **B**：作者 arXiv 稿；可支撑公开技术细节，不能冒充同行评审。
- **C**：作者/机构官方仓库、项目页；可支撑代码、权重和接口是否公开。
- **X**：聚合页、博客和搜索摘要；只用于发现，不支撑教材事实。

## 3. 严格纳入合同

### 3.1 核心判定

一个条目进入严格 variational future 主线，至少要有：

1. 显式随机潜变量或随机 latent state；
2. 训练 posterior $q_\phi(z\mid h,y,c)$ 能看待解释的真实未来/新观测；
3. 部署 prior $p_\psi(z\mid h,c)$ 固定，或只能看历史和合法条件；两种情况都不能看未来；
4. KL、ELBO 或等价 variational free-energy 路径对齐二者；
5. 部署时从 prior 产生未来或 latent rollout，而不是仍调用 posterior oracle。

顺序模型的统一目标可写为

$$
\mathcal L
=
\sum_t
\mathbb E_q\log p_\theta(x_t\mid x_{<t},z_{\le t},c)
-
\sum_t\beta_t
\mathbb E_{q_\phi(z_{<t}\mid x_{<t},c)}D_{\mathrm{KL}}
\left[q_\phi(z_t\mid z_{<t},x_{\le t},c)\Vert p_\psi(z_t\mid z_{<t},x_{<t},c)\right].
$$

只有在 $\beta_t=1$ 且其余项保持概率模型一致时，它才是原始模型对数似然的标准 ELBO；调低 $\beta$、free bits、额外 GAN/感知/对比目标都会改变 rate–distortion 取舍，不能再把目标值无条件解释成可比 likelihood。

### 3.2 近似但必须排除或分流的接口

| 系统/概念 | 为什么看起来相似 | 严格裁决 |
|---|---|---|
| video VAE tokenizer / IV-VAE | 有 encoder、Gaussian latent、KL 和 decoder | 表示已知完整视频；属于 tokenizer，不是未来不确定性 |
| latent diffusion / flow | 有随机噪声和概率路径 | 没有 learned semantic future posterior–prior 合同；属于另一 objective |
| VQ-VAE / AR token | token 是 latent，生成是随机的 | nearest-neighbor/VQ 与 categorical AR 不自动构成 variational inference |
| Wichers 2018 hierarchical prediction | 名字有 hierarchical，预测长视频 | 高层 feature predictor + adversarial feature loss；无 q/p/ELBO，移出严格主线 |
| PlaySlot 2025 | 有 latent action 和 inverse dynamics | 训练是 image/slot/VQ，部署 action 由人或 policy 给出；无 history-only generative prior 的 KL |
| ordinary Slot Attention / SlotFormer | 有 object latent | 多为 deterministic slots；“latent object”不等于“stochastic object” |
| GAN noise / dropout | 不同 seed 可生成不同结果 | 无 future-aware amortized posterior，不能证明 learned future distribution |
| PlaNet / Dreamer RSSM | 有 posterior、prior、KL 和 stochastic state | 数学上是变分支线；核心验收是 action/reward/return，不能冒充开放域视频生成排名 |
| LPWM 2026 | particle posterior/dynamics prior、inverse-action posterior/policy prior、ELBO | 纳入对象中心 world-model 前沿，但限定对象、相机运动和控制场景证据 |

## 4. 里程碑与正式状态账本

| 首次公开 | 正式状态 | 节点 | 新增机制 | 作者协议与证据边界 |
|---|---|---|---|---|
| 2015-06-07 | NeurIPS 2015 | VRNN | 每步 stochastic state、conditional prior/posterior | 通用序列前史；实验主要是语音/手写，不是视频 benchmark |
| 2017-10-30 | ICLR 2018 | SV2P | fixed $N(0,I)$ prior 下的全局/逐帧未来 latent | 100 samples 按最高 PSNR 选 oracle 样本并报告相应 PSNR/SSIM；不是校准 |
| 2018-02-21 | ICML 2018 | SVG-LP | 每步 learned history-conditioned prior 对齐 posterior | SM-MNIST/KTH/BAIR；AR 漂移和 best-of-(N) 仍存在 |
| 2018-04-04 | arXiv preprint | SAVP | posterior VAE 路径 + prior-sampled GAN 路径 | BAIR/KTH；best/avg/worst over 100；500-step 只是超出训练 horizon 的质性展示 |
| 2019-04-27 | ICCV 2019 | Improved Conditional VRNN | 多层 latent、dense connections、高容量 likelihood | 把模糊的一部分归因于 underfitting；层级与容量贡献仍缠绕 |
| 2019-06-19 | NeurIPS 2019 | [Struct-VRNN](https://arxiv.org/abs/1906.07889) | keypoint/object structure + stochastic dynamics | 结构 latent 支撑预测与下游任务；不等于自然视频对象已可识别 |
| 2019-11-05 | NeurIPS 2019 | Large Stochastic RNN | 减少手工 flow/mask 偏置，扩大 stochastic recurrent model | scale/capacity 节点；不能把算力混杂写成新不确定性语义 |
| 2018-11-12 | ICML 2019 | PlaNet | deterministic memory + stochastic RSSM、latent overshooting | 64×64 控制任务，优势主要是样本效率；不是开放域视频质量证明 |
| 2020-02-21 | ICML 2020 | SRVP | 全 latent residual dynamics，解耦 frame synthesis 与时序演化 | 每 context 100 futures；oracle frame metrics 与 5 组 FVD 需分开 |
| 2020-05-23 | CVPR 2020 | S3VAE | global/static 与 local/dynamic latent，自监督解耦 | 分类器/IS 证据不能替代开放域分布质量 |
| 2020 | ICML 2020 | G-SWM | object/global stochastic state，结构化阻断 attribute bypass | 合成球碰撞/CLEVR-like；对象证据不能直接外推真实视频 |
| 2020-10-05 | ICLR 2021 | DreamerV2 | categorical RSSM + KL balancing | Atari 55 tasks；discrete state 的关键节点，不应误归给 V3 |
| 2021-02-18 | NeurIPS 2021 | CW-VAE | 多层 latent chain 按固定慢时钟更新 | 最长 1000-frame rollout 主要来自 MineRL/Moving-MNIST 等受控域 |
| 2021-03-06 | CVPR 2021 | GHVAE | 层级 VAE 逐层贪心训练，缓解显存与联合优化困难 | 作者报告 4 数据集和机器人任务增益；容量/模块数是混杂 |
| 2021-06-24 | arXiv / CoRR | FitVid | 简单大容量 fixed-prior VAE，暴露 benchmark 过拟合 | best-of-100 与 FVD 协议不同；训练身份复现不等于未来建模 |
| 2021-08-05 | ICCV 2021 | SLAMP | appearance 与 motion 两条 stochastic branch + motion history | 驾驶数据有优势；命名分支不证明语义或因果解耦 |
| 2021-10-06 | ICCV 2021 | NUQ | hierarchical variational predictive uncertainty | latent uncertainty 未自动在事件空间校准；降权也可能容忍错误 |
| 2021-10-21 | ICLR 2022 | VPR | event-boundary-triggered hierarchical renewal | 主要是合成事件数据；不能外推开放视频 |
| 2022 | CLeaR 2022 | VIM | object slots + categorical mechanism selector + Gaussian residual | Random-Walk toy；主要结果使用 inference network 读当前目标观测，是 posterior-assisted reconstruction，不是 prior-only deployment generation |
| 2023 | ICLR 2023 | VP² | 固定 planner、11 类/310 任务实例的控制中心评测 | 证明感知指标可能不预测控制成功；不是一种新 VAE 架构 |
| 2023-06-09 | TMLR 2024 | DDLP | particle-tracking posterior + Transformer dynamics variational prior | LPWM 的直接粒子前身；对象化场景、遮挡和相机运动仍是边界 |
| 2024-06-11 | ICML 2024 | RSP | categorical future latent + masked-image objective 用于表示学习 | 作者明确只做单 future frame、图像质量不高；不是长视频突破 |
| 2023-01-10 | Nature 2025 | DreamerV3 | categorical RSSM、split KL、free bits、unimix | 150+ 控制任务与 5→45 action-conditioned rollout；return 不是校准 |
| 2026-02-17 / 07 | Neural Networks 2026 | Implicit hierarchical temporal–spatial residual model | prior–posterior spatial residual 与层级重建 | 3 数据集作者结果；正式延续不等于 VAE 重回开放域主流 |
| 2026-03-04（arXiv v1；OpenReview 首次可见日未冻结） | ICLR 2026 Oral | LPWM | object particles、stochastic dynamics、inverse-action posterior 与 policy prior | 对象中心、动作/语言/图像目标条件；仍受 particle identity、patch origin 和相机运动限制 |

## 5. 关键 paper review 证据

### 5.1 Fixed prior 到 learned prior

**SV2P** 把多未来问题显式交给 latent，但部署仍从固定标准高斯采样。训练 posterior 看完整未来，因而测试时绝不能使用 posterior 样本。论文图 6 从 100 次采样中按最高 PSNR 选 oracle 样本，并报告该样本的 PSNR/SSIM；这证明“样本集合中是否包含接近记录真值的一个”，不证明样本频率正确。

**SVG-LP** 令每步 prior 随历史变化，并与能看当前真实帧的 posterior 对齐。真正的里程碑不是“用了 VAE”，而是把部署分布写成 learned interface。它仍可能漏 mode、collapse 或在 open-loop 中累积误差。

**SAVP** 将 VAE coverage 路径与 GAN realism 路径组合。锐利度与视觉真实感可以改善，但 GAN mode collapse、prior gap 与未校准的 best-of-100 仍须单独报告。截止冻结日应保持 arXiv preprint 状态。

### 5.2 容量、层级与时间抽象

**Improved Conditional VRNNs** 使用多层 Gaussian latent、dense latent connections 与更高容量 likelihood。BAIR 的作者协议是 2 context + 10 train futures、测试延长到 28 futures；每个 context 100 samples，LPIPS/SSIM 取 best，FVD 则从 100 中随机取 1 并重复。不同采样聚合方式不能混成单一排行榜。论文的重要纠偏是：VAE 模糊并非只能归因于 MSE，underfitting 和受限 latent/likelihood 也可能是原因。

**GHVAE** 把层级模型拆成逐层贪心、模块化训练，绕开整网显存和双向依赖造成的优化困难。作者报告相对基线的 17–55% 视频指标提升及 35–40% 机器人成功率提升，但这些是论文内特定任务/预算，不能直接与后来的 FVD 或开放域生成比较。

**CW-VAE** 让第 $l$ 层只在固定慢时钟集合 $T_l$ 更新，非活跃时刻复制状态。它证明 temporal abstraction 可扩展 rollout，但“高层等于慢物体”是数据与 clock 设计共同产生的归纳偏置，不是自动语义发现。

**VPR** 用事件边界触发层级更新，是 fixed clock 的另一方向。其强证据集中在合成 3D Shapes Dynamics/Moving Ball；应称事件抽象机制证据，而非开放域视频结果。

### 5.3 把动力学从像素回灌中分离

**SRVP** 在 latent 中做一阶 residual update，再由独立 frame generator 渲染，避免每步把解码帧重新编码。它的贡献是 factorization：frame synthesis 与 temporal dynamics 可分别演进；一阶 residual 仍是结构限制，decoder 也可能掩盖 latent 失败。

**S3VAE** 把 static global $z_f$ 与 sequential dynamic $z_{1:T}$ 分开，并加入光流、landmark/audio 等自监督约束。分类器准确率和 inception-style score 只能说明给定 probe 下的可分性，不证明因果解耦或多未来校准。

**SLAMP** 分别建模 appearance/pixel 与 flow/motion 随机分支，再用 mask 融合渲染与 warp。对 KITTI/Cityscapes 的价值来自显式运动历史；遮挡、相机运动和 warping 错误仍是主要边界。KTH 上其 FVD 并未全面胜过 SRVP，因此不能写“所有数据 SOTA”。

### 5.4 对象、模块与 latent action

**G-SWM** 用对象级随机 state 和 global context，并通过先从 state 再生成 attributes 的结构阻断 observation-to-attribute shortcut。这比“给 slot 起语义名”更接近可检验的对象模型，但证据主要是合成场景。

**VIM** 为每个对象同时学习 categorical mechanism selector 与 Gaussian residual，并为二者设置 prior/posterior KL。其 OOD 组合证据来自 32×32、3 balls 的 Random-Walk toy；主要结果在测试时使用 inference network 读当前目标观测，因此属于 posterior-assisted reconstruction，不能当 prior-only deployment generation 证据。部分基线还 teacher-force 上一真值帧，数值不能无条件横排。

**DDLP** 用 particle-tracking posterior 对齐跨帧对象，并把 Transformer dynamics 作为只看过去粒子的 variational prior；它是 LPWM 的直接对象粒子前身。其 what-if 与紧凑生成证据建立在对象化数据上，不能自动解决对象 birth/death、严重遮挡或开放相机运动。

**LPWM** 是冻结日最新的正式前沿：对象粒子带位置、尺度、深度、透明度和 appearance 属性；inverse-action posterior 在训练看见转移，policy prior 在部署从历史/条件采样，particle encoder posterior 再与 dynamics prior 对齐。官方提供代码、数据和预训练模型。它把 variational dynamics、object-centric representation、latent action 和多条件接口接在一起，但仍未证明适用于开放相机运动和通用互联网视频。

### 5.5 World model 分支

**PlaNet** 的 RSSM 组合 deterministic memory 与 stochastic state，并用 latent overshooting 对齐多步 prior 与 informed posterior。其价值是样本高效控制；论文六个 64×64 Control Suite 任务中也并非全面超过 D4PG。

**DreamerV2** 在 2020 首次公开并于 ICLR 2021 正式发表，用 categorical RSSM 与 KL balancing 把 imagined behavior learning 扩到 Atari 55 tasks。**DreamerV3** 在此基础上加入 split/stop-gradient KL、1 nat free bits、unimix 和跨域稳定化；Nature 论文验证 150+ 控制任务，并展示 5 帧 context 加动作后预测 45 帧。主要结论来自 return、reward/continue 与行为学习，不是开放域视频 uncertainty calibration。

## 6. 评测协议与不可跨越的推断

### 6.1 必须分开的四种汇总

| 汇总 | 回答的问题 | 不能推出 |
|---|---|---|
| single sample | 一次典型部署可能得到什么 | 覆盖全部 mode |
| sample average / expected metric | 整个采样分布的平均质量 | 稀有 mode 已覆盖 |
| best-of-$K$ | $K$ 个样本中是否存在接近记录真值者 | 概率频率、校准或典型质量 |
| posterior oracle | 看见真实未来后 latent 能否解释它 | 部署 prior 能生成同样结果 |

所有 best-of-$K$ 必须披露 $K$；不同 $K$ 不能横比。best-of-100 可以奖励过宽甚至带噪的分布。

### 6.2 建议的证据组合

- **概率/latent 使用**：ELBO/NLL（仅同 likelihood/scaling）、逐时刻/逐层 KL、active units、$I_q(z;y\mid h,c)$ proxy、zero/shuffle/resample intervention。
- **分布**：event NLL/Brier/ECE、mode coverage、rare-mode recall、spurious-mode rate、sample-average 与 best-of-$K$。
- **视觉**：FVD/video precision–recall/LPIPS 与条件错误、identity/state tracking 并列；FVD 有 content bias，不能单独代表时间真实性。
- **长时**：按 horizon 报 prior/posterior gap、状态漂移、条件违规、diversity 和感知质量。
- **world model**：latent/reward/continue 误差、OOD/model exploitation、固定 planner 下 return；return 不替代分布校准。
- **object/latent action**：count/presence、tracking ID、interaction/collision、posterior–policy agreement、controllability 与下游 return。

每个数值必须附 context/predict horizon、分辨率/fps/crop、real/generated sample 数、sampler steps、backbone/checkpoint、$K$、seed/CI 和硬件。

## 7. `LatentFork-1` 预注册草案

### 7.1 机制数据 `Forking-Squares-v1`

- 64×64，8 帧历史、24 帧未来；同一 prefix 的重复未来共享前 8 帧。prefix 中可见 cue $g\in\{\text{cyan},\text{amber},\text{violet}\}$，对应 left/right/stop 真分布分别为 $(0.6,0.3,0.1)$、$(0.2,0.7,0.1)$、$(0.1,0.2,0.7)$；三种 cue 在 split 内平衡，模型不得读取未来随机数。
- 按初始状态分组切分 10,000/2,000/2,000，避免同一轨迹参数泄漏。
- 每个测试前缀由 simulator 固定生成 64 个真实未来，保存真实 mode label、概率、轨迹和 identity。
- 这是本教材提出的机制基准，不冒充已发表 benchmark；实验运行前须提交 generator、manifest 与 hash。

外部效用 smoke test 使用 VP² RoboDesk 的官方数据、任务、planner 和公共 SVG′ checkpoint 作锚点；所有比较分支仍须等预算重训，不能直接横比不同预训练模型。

### 7.2 分叉与预算

| Fork | 唯一变化 |
|---|---|
| A | 无随机 latent；将容量回填到 deterministic residual adapter |
| B | posterior + fixed $N(0,I)$ prior，保留 KL |
| C | posterior + history/action-conditioned learned prior |
| D | global $z_g$ + per-step $z_t$ hierarchical prior；缩减其他宽度以匹配预算 |
| Q | posterior-oracle，仅量化 posterior-assisted 上限与 deployment-prior gap；禁止作为部署结果 |

参数 ±1%、单 rollout active FLOPs ±5%、总训练 FLOPs ±2%；同数据顺序、优化步数、增强、decoder、forcing 和 5 个训练 seed。测试只允许 prefix、合法 action/condition 和 seed；隐藏未来变化不得改变 prior 输出。由于 B 的 history-conditioned decoder 理论上也能把固定 Gaussian 映射成 cue-dependent mode，C–B 只是**整系统优化/归纳偏置 ablation**，不能单独证明 learned prior 的表达能力必不可少。

### 7.3 报告与建议阈值

每个历史固定 $K=64$，同时报 single、sample-average、best-of-64、event Brier/NLL/ECE、coverage、rare recall、spurious rate、KL、active units、latent intervention、条件违规、轨迹/感知误差及 horizon 曲线。JS 分两项：主指标比较 prior event distribution 与 simulator truth；次指标把同一 $h$ 的 64 个真实未来分别送入 posterior 后聚合 event distribution，再与 prior samples 比较。禁止把单个真实未来对应的集中 posterior 直接对混合 prior。

下列是**教材建议的预注册保留阈值，不是已运行结果或领域公认阈值**：

- C 相对 B 的 Brier 改善至少 0.02，paired-bootstrap 95% CI 下界大于 0；
- 0.1 rare mode 的 recall@64 ≥ 0.80，spurious mode ≤1%；
- 条件违规相对 A/B 增幅 ≤1 个百分点，expected state/perceptual error 劣化 ≤3%；
- per-prefix $\operatorname{JS}(p_{\mathrm{prior}}(m\mid h),p_{\mathrm{sim}}(m\mid h))\le0.05$；聚合 posterior–prior JS 只作诊断，不对单样本 posterior 设混合分布阈值；
- 重采样 $z$ 至少在 30% 测试历史中改变持续事件分支；同 $(h,z)$ 重跑一致率 ≥99%；
- D 只有在 horizon-24 coverage 比 C 提高 ≥5 个百分点且质量/违规通过非劣界时，才保留“层级改善长期覆盖”；
- VP² 的“有助规划”声明要求成功率至少 +5 个百分点且 95% CI 排除 0。

一票否决：收益只存在于 Q 或 best-of-$K$；$z$ 只改纹理；覆盖来自非法 mode；预算匹配后收益消失；C 未改善 proper score 却被解释为 learned-prior 优势；控制收益只来自更多 planner rollout；把单 checkpoint 采样当 epistemic uncertainty。

## 8. 生成式教学图记录

### 8.1 学习目标

一图解释：训练 posterior 可见真实未来，部署 learned prior 只能见历史；KL 对齐两者；同一 decoder 产生 left/right/stop 多未来；验收依次检查 latent 是否被使用、prior gap、mode coverage 和 calibration。

### 8.2 精确生成提示词

```text
Create a clean, publication-quality scientific teaching infographic in 16:9 landscape, white background, flat vector style, high contrast, dark navy + cyan + amber accents. Audience: graduate students learning stochastic variational video prediction. Exact title: “VARIATIONAL VIDEO: TRAINING VS DEPLOYMENT”. Build a left-to-right two-lane diagram. Top lane labeled “TRAINING”: boxes “history h + observed future y” → “posterior q(z | h,y)” → “latent z”. Bottom lane labeled “DEPLOYMENT”: “history h only” → “learned prior p(z | h)” → three sampled latent dots. Draw a clearly labeled vertical dashed bridge “KL alignment” between posterior and learned prior. Both lanes feed one shared box “decoder p(y | h,z)”. From the decoder, show three coherent future branches of the same small moving square: “LEFT”, “RIGHT”, “STOP”; preserve the same object identity and scene while only the event branch changes. On the far right, show four compact evidence gates with check icons and exactly these labels: “LATENT USED?” with caption “z changes the event”; “PRIOR MATCH?” with caption “small prior–posterior gap”; “COVERAGE?” with caption “modes, not invalid futures”; “CALIBRATION?” with caption “frequency matches probability”. Add a bottom amber warning banner with the exact sentence: “A sharp best-of-100 sample is NOT a calibrated future distribution.” Keep all text horizontal, large, correctly spelled, uncluttered, and inside safe margins. Use arrows with unambiguous direction. No gradients, no photorealism, no decorative neural-network nodes, no logos, no citations, no extra text.
```

### 8.3 产物与当前 QA

- 文件：`assets/diagrams/variational-future-contract.png`
- 尺寸：1536×1024 RGB PNG；1,287,491 bytes。
- SHA-256：`ca309b30ebe193282a52a9d3d2c579ad0710552a32a084148d7cf234f6b52531`
- 原图人工检查：所有指定文字正确；箭头、训练/部署信息边界、三种 mode 和四道 evidence gate 可辨。
- 灰度检查：PASS；标题、两条信息通道、三种 mode、四道 evidence gate 与底部 warning 均保持可辨。
- 正文嵌入检查：相对路径存在，alt 与图注齐全；相关 Markdown 本地资源闭包通过。
- 比例偏差：提示词要求 16:9，工具实际返回 1536×1024（3:2）。本轮接受该偏差，因为内容完整且横向教材页可读；不裁切，以免破坏标题、右侧 evidence gate 或底部 warning。
- Mermaid 复核：本轮涉及的 11 个 Markdown 文件共 19 个图均由 mermaid-cli 11.16.0 + 系统 Chrome 成功渲染，且均含 `accTitle` 与 `accDescr`；新增/修改的 4 个图另做原色与灰度人工检查。

## 9. 本轮裁决

1. 2017–2022 的严格主线是 fixed → learned prior、per-step → hierarchy/temporal abstraction、pixel recurrence → latent dynamics、monolithic → object/module。
2. 2018–2025 的 RSSM/world-model 分支把 posterior/prior 工程化为 categorical state、KL balancing、free bits 和 action-conditioned imagination；控制证据不等于开放域视频校准。
3. 2023 后开放域高保真视频生成主流更多采用 VAE/VQ compression + diffusion/flow/AR；这里的 VAE 是表示接口，不是未来分布 posterior。
4. ICLR 2026 LPWM 是对象粒子 + latent action + 多条件变分动力学的正式前沿；Neural Networks 2026 的 implicit hierarchical residual model 是 direct long-term VAE 延续。两者都不能证明 VAE 已重新成为通用视频 foundation model 主干。
5. 当前最薄弱的证据不是“能否生成一张好看的未来”，而是 prior 是否匹配、mode 频率是否校准、长 horizon 是否保持条件、以及下游效用是否在固定预算下成立。
