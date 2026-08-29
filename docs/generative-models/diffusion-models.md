# 扩散模型：从噪声逐步还原视频

扩散模型把复杂的数据分布转化为一系列较容易学习的去噪问题。训练时逐步给真实视频加噪；生成时从高斯噪声出发，反向恢复结构、运动和细节。它成为现代视频生成主线，关键在于质量、覆盖度、条件控制和规模化稳定性的综合表现。

## 1. 前向加噪与反向生成

离散扩散定义前向马尔可夫链：

$$
q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

可直接采样任意时刻：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I)
$$

网络接收 $x_t$、时间步 $t$ 和条件 $c$，预测噪声 $\epsilon$、干净样本 $x_0$、速度参数 $v$ 或 score。最常见简化目标是：

$$
\mathcal{L}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2
$$

推理使用 DDPM、DDIM、DPM-Solver 等数值方法，从 $x_T$ 逐步走回 $x_0$。训练参数化与采样器是两个层面，不能把“换采样器”误写成“重新训练了扩散模型”。

## 2. 像素扩散与 latent diffusion

像素空间信息完整，但视频张量巨大。latent video diffusion 先用 VAE 压缩：

$$
z_0=E(x_0),\qquad \hat{x}_0=D(z_0)
$$

扩散过程发生在 $z$ 中，可显著降低时空分辨率和计算量。代价是 tokenizer 可能损失文字、小物体、高频纹理和快速运动。最终质量由 VAE 重建上限与 diffusion 建模能力共同决定。

## 3. 时间维如何进入网络

- **3D U-Net**：直接用时空卷积处理视频，局部一致性强但成本高。
- **2D 图像 backbone + temporal layer**：复用图像模型，在层间加入时间卷积或时间注意力，便于从图像权重迁移。
- **分解时空注意力**：分别计算空间和时间注意力，降低全时空注意力成本。
- **DiT/spacetime patch**：把 latent 切成时空 patch 交给 Transformer，适合扩大模型和处理不同分辨率、时长与宽高比。
- **级联模型**：低分辨率模型负责内容与运动，再用空间/时间超分辨率补细节。

Lumiere 的 Space-Time U-Net 尝试一次覆盖完整时间范围，减少先关键帧、后插帧造成的运动不一致 [[5]](#ref-5)。

## 4. 条件控制

文本通常经 cross-attention 注入；首帧、参考图、深度、姿态、光流、相机轨迹或已有视频可经拼接、额外编码器或 adapter 接入。Classifier-free guidance 用条件与无条件预测之差增强条件遵循：

$$
\hat\epsilon=(1+w)\epsilon_\theta(x_t,c)-w\epsilon_\theta(x_t,\varnothing)
$$

较大 guidance 不会免费提升质量：它可能降低多样性、过饱和，并放大运动和解剖错误。视频中还需判断条件在整个时间轴上是否持续生效。

## 5. 训练视频 diffusion 的关键难点

视频数据质量差异大，包含剪辑、字幕、水印、变速和弱文本。常见流程包括去重、镜头切分、质量/运动筛选、自动描述、宽高比和时长分桶。图像—视频联合训练能利用更多静态知识，但静态图像比例过高可能造成 motion collapse。

训练还要处理时间采样、帧率、噪声调度、loss weighting 和显存。短片段训练不自动带来长视频能力；长时一致性需要更长上下文、层级生成、记忆或分段锚定。

## 6. 采样速度与蒸馏

扩散的主要成本来自多步网络调用。可通过高阶求解器、减少步数、progressive distillation、consistency distillation 和 adversarial distillation 加速。评估加速时要同时检查细节、运动连续性、条件遵循和多样性；单张关键帧接近并不等于整段视频等价。

若还要在完整视频结束前持续发帧，则问题从“少做几步”扩展为双向 teacher 到 causal student 的迁移、自生成历史训练、KV cache 维护和在线 deadline 调度。完整路线见[因果、流式与实时视频生成](causal-streaming-generation.md)。

## 7. 代表工作与演进

DDPM 奠定现代训练框架 [[1]](#ref-1)；Video Diffusion Models 系统展示视频生成、预测和插帧 [[2]](#ref-2)；Imagen Video 与 Make-A-Video 探索文本图像先验和级联高分辨率视频 [[3]](#ref-3), [[4]](#ref-4)；Stable Video Diffusion 推动开放 latent video diffusion [[6]](#ref-6)；Sora 把 spacetime patch 与 Transformer diffusion 扩展到更大规模 [[7]](#ref-7)。

## 8. 优势、边界与评测

优势是训练相对稳定、样本覆盖好、多条件接口成熟，并能从图像模型迁移。限制是推理昂贵、长视频状态保持困难、输出缺乏可验证因果保证。评测应分开看画质、运动、条件遵循、身份/对象持续性、物理合理性和多样性，并明确是短视频、长视频还是分段拼接。

高质量 diffusion 视频可以学到大量世界规律，但仅凭演示不能证明它是可规划的 world model。动作条件、反事实正确性和闭环收益仍需独立验证。

## 参考文献

<a id="ref-1"></a>[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Jonathan Ho, Ajay Jain, Pieter Abbeel. NeurIPS. 2020.

<a id="ref-2"></a>[2] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-3"></a>[3] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. arXiv preprint. 2022.

<a id="ref-4"></a>[4] [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. ICLR. 2023.

<a id="ref-5"></a>[5] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. SIGGRAPH Asia. 2024.

<a id="ref-6"></a>[6] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-7"></a>[7] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. 2024.
