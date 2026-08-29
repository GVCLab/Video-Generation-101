# 生成模型路线：VAE、GAN、Diffusion 与 Flow

本章专门梳理“把视频当作数据分布来生成”的主线。它回答的问题是：给定噪声、文本、图像或历史帧，模型如何产生一个在视觉和时间上合理的视频样本。

这条线不同于决策型 world model。生成模型可以学到大量运动、物理和场景规律，但核心目标通常仍是建模 $p(x_{1:T}\mid c)$，而不是显式回答“采取动作 $a_t$ 后会发生什么”。

如果关注的是“文本如何约束视频内容”，请从[文本到视频](tasks/text-to-video.md)开始；本章只负责解释 T2V、I2V 和其他视频任务共同使用的 VAE、GAN、diffusion 与 flow 生成机制。

## 生成机制总览

| 生成机制 | 基本做法 | 主要优势 | 主要限制 |
|---|---|---|---|
| Recurrent prediction | 逐帧或逐 latent 预测未来 | 结构直观，适合在线 rollout | 自身误差会逐步累积 |
| Variational generation | 用随机 latent 表示多种可能未来 | 可建模不确定性 | 可能模糊或 posterior collapse |
| Adversarial generation | 让生成器与判别器对抗训练 | 容易得到锐利画面 | 训练不稳定，可能 mode collapse |
| Autoregressive generation | 按顺序预测下一个像素、token 或 latent | 天然支持变长生成 | 采样串行，长序列成本高 |
| Masked generation | 从 mask token 开始多轮并行填充 | 生成较快，也适合补全 | 需设计填充顺序和终止策略 |
| Diffusion | 从噪声出发，通过多步去噪得到视频 | 质量、覆盖度和条件控制的组合较强 | 多步推理成本高 |
| Flow / consistency | 学习从简单分布到数据分布的更直接路径 | 有机会减少采样步数 | 需验证加速后的细节和时间一致性 |
| Causal / streaming generation | 按帧或时间块生成，复用有界历史并持续发帧 | 支持未知时长、低延迟和条件在线更新 | 暴露偏移、长期漂移、缓存与 deadline 相互耦合 |

这些机制可以组合使用。例如，一个视频模型可先用 VAE 压缩视频，再用 Transformer 在 latent 中执行 diffusion 或 flow。“在什么空间生成”与“用什么机制生成”是两个不同问题。

## 分机制详细教程

本页负责建立全局路线；每种机制的数学目标、训练过程、视频结构、失败模式和实践选择见以下独立文档：

1. [递归预测：逐步生成未来视频](generative-models/recurrent-prediction.md)
2. [变分生成：用潜变量表达多种可能未来](generative-models/variational-generation.md)
3. [对抗生成：用判别器学习时空真实感](generative-models/adversarial-generation.md)
4. [自回归生成：把视频变成可预测的序列](generative-models/autoregressive-generation.md)
5. [掩码生成：从缺失 token 到并行视频合成](generative-models/masked-generation.md)
6. [扩散模型：从噪声逐步还原视频](generative-models/diffusion-models.md)
7. [Flow 与 Consistency：学习更直接的生成路径](generative-models/flow-consistency-models.md)
8. [因果、流式与实时视频生成：从 Diffusion Forcing 到可交互长视频](generative-models/causal-streaming-generation.md)

推荐初学者先读前七篇，再把第八篇当作综合章。递归与自回归解释“按什么顺序产生视频”，VAE/tokenizer 解释“在什么表示空间工作”，GAN、diffusion、flow 和 consistency 解释“如何把模型分布推向真实视频分布”；因果流式生成则把 factorization、少步蒸馏、长期记忆与在线系统组合起来。

## 1. 从视频预测到概率生成

早期深度视频预测通常直接优化下一帧或未来帧误差：

$$
\hat{x}_{t+1:T}=f_\theta(x_{1:t})
$$

这类方法直观，但未来具有多模态性。一个人可以向左走，也可以向右走；单一 MSE 目标容易把多个合理未来平均成模糊画面 [[1]](#ref-1)。后来的 VAE、GAN、diffusion 和 flow，本质上都在解决“如何表达多个可能未来”。

## 2. VAE 路线：用 latent 表达不确定性

VAE 将视频生成写成潜变量模型：

$$
p_\theta(x\mid z),\quad z \sim p(z)
$$

对于视频，它的关键价值是用随机 latent 表达未来的不确定性，并把内容、运动、速度或风格放进可采样空间。典型问题包括 posterior collapse、重建模糊，以及 latent 是否真的对应可解释运动。

代表工作：

- **Stochastic Video Generation with a Learned Prior [[3]](#ref-3)**：用 learned prior 生成多种可能未来，强调 stochastic prediction。
- **MoCoGAN [[2]](#ref-2)**：虽然是 GAN，但它的内容 latent 与运动 latent 解耦思想也常被放入 VAE/GAN 共同脉络中理解。

阅读时重点看三件事：latent 是全局还是逐帧采样，时间 dynamics 如何建模，评测是否覆盖多模态未来而不只是平均误差。

## 3. GAN 路线：用判别器追求锐利时空真实感

GAN 将生成器和判别器对抗训练：

$$
\min_G \max_D \mathbb{E}_{x\sim p_{data}}\log D(x)+\mathbb{E}_{z\sim p(z)}\log(1-D(G(z)))
$$

在视频中，判别器可以看单帧、短片段或完整时空体。它比 MSE 更容易产生锐利画面，也推动了内容/运动解耦、前景/背景分离和高分辨率视频生成。

代表工作：

- **Generating Videos with Scene Dynamics**：用时空卷积 GAN 生成短视频，并显式区分前景和背景。
- **MoCoGAN**：把内容和运动分离，形成早期视频生成的重要基线。
- **DVD-GAN [[4]](#ref-4)**：展示大型视频 GAN 在复杂数据集上的能力。

GAN 的核心短板也很清楚：训练不稳定、mode collapse、覆盖度难评估，以及很难像 diffusion 那样稳定吃下更大的数据和模型规模。

## 4. Diffusion 路线：从噪声逐步还原视频

Diffusion 将生成过程拆成多步去噪：

$$
x_T \sim \mathcal{N}(0,I), \quad x_{t-1}=g_\theta(x_t,t,c)
$$

视频 diffusion 的关键设计是如何处理时间维：

- 在像素或 latent 空间中联合去噪整个视频。
- 复用图像 diffusion backbone，再增加 temporal layer。
- 通过时空注意力、3D U-Net、patch Transformer 或级联超分辨率扩展到高分辨率和长时长。

代表工作：

- **Video Diffusion Models [[6]](#ref-6)**：系统展示 diffusion 可以用于视频生成、预测和插帧。
- **Make-A-Video [[7]](#ref-7) / Imagen Video [[8]](#ref-8)**：把大规模文本图像模型扩展到文本视频。
- **Align Your Latents [[9]](#ref-9) / Stable Video Diffusion [[11]](#ref-11)**：推动 latent video diffusion 与开放权重基线。
- **Lumiere [[12]](#ref-12)**：用 Space-Time U-Net 直接覆盖完整视频时间范围。
- **Sora**：用 spacetime patch 和 Transformer diffusion 扩展到更长、更复杂的视频生成。

Diffusion 成为主线的原因不是它最便宜，而是它在质量、覆盖、多条件控制和规模化稳定性上形成了很强的组合优势。

## 5. Flow 与 consistency：减少采样成本

Flow matching、rectified flow、consistency model 等方法试图把多步去噪变成更直接的分布变换。它们在视频中的动机尤其强：视频采样成本高，任何能减少步数、降低延迟、提高长时一致性的训练或采样方法都会非常重要。

阅读这一路线时，建议把问题拆开：

- 训练目标是否仍然学习从简单分布到数据分布的连续路径？
- 采样步数减少后，运动连续性和细节是否保持？
- 长视频生成是一次完成、分块生成，还是带记忆地递推？

当目标是边生成边播放、在 rollout 中接收新 prompt 或动作时，还要区分 causal、streaming、real-time、long video 与 interactive 五种主张。它们的训练—推理分布、KV memory、SLO 与公平评测见[因果、流式与实时视频生成专章](generative-models/causal-streaming-generation.md)。

## 6. 这条路线留下的核心问题

生成模型路线最成熟，也最容易被误解成“已经解决世界建模”。实际还需要继续问：

- 模型生成的是视觉上合理的相关性，还是可干预的因果 dynamics？
- 对象离开画面再回来，状态是否保持？
- 同一初始状态下改变动作，未来是否发生正确变化？
- 模型是否能被规划器使用，而不只是生成好看的 demo？

这些问题把本章自然连接到 [大模型路线](foundation-models.md)、[从视频生成到 World Model](world-models.md) 和 [相关应用](applications.md)。

## 最小阅读路径

1. **Beyond MSE**：理解为什么确定性像素预测会模糊。
2. **MoCoGAN**：理解内容和运动解耦。
3. **DVD-GAN**：理解视频 GAN 的扩展尝试。
4. **Video Diffusion Models**：理解 diffusion 如何进入视频。
5. **Stable Video Diffusion**：理解开放 latent video diffusion baseline。
6. **Diffusion Forcing → CausVid → Self Forcing**：进入[因果流式专章](generative-models/causal-streaming-generation.md)，理解少步、历史分布和长期记忆为何必须一起处理。
7. **Sora technical report [[13]](#ref-13)**：理解大规模视频生成与 world simulator 讨论的连接点。
本页主要参考工作：Denoising Diffusion Probabilistic Models [[5]](#ref-5)、AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning [[10]](#ref-10)。

## 参考文献

<a id="ref-1"></a>[1] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. ICLR. 2016.

<a id="ref-2"></a>[2] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. CVPR. 2018.

<a id="ref-3"></a>[3] [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). Remi Denton, Rob Fergus. ICML. 2018.

<a id="ref-4"></a>[4] [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571). Aidan Clark, Jeff Donahue, Karen Simonyan. arXiv preprint. 2019.

<a id="ref-5"></a>[5] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Jonathan Ho, Ajay Jain, Pieter Abbeel. NeurIPS. 2020.

<a id="ref-6"></a>[6] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-7"></a>[7] [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. ICLR. 2023.

<a id="ref-8"></a>[8] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. arXiv preprint. 2022.

<a id="ref-9"></a>[9] [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818). Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, et al. CVPR. 2023.

<a id="ref-10"></a>[10] [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. ICLR. 2024.

<a id="ref-11"></a>[11] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-12"></a>[12] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. SIGGRAPH Asia. 2024.

<a id="ref-13"></a>[13] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.
