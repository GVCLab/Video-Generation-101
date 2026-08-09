# 生成模型路线：VAE、GAN、Diffusion 与 Flow

本章专门梳理“把视频当作数据分布来生成”的主线。它回答的问题是：给定噪声、文本、图像或历史帧，模型如何产生一个在视觉和时间上合理的视频样本。

这条线不同于决策型 world model。生成模型可以学到大量运动、物理和场景规律，但核心目标通常仍是建模 $p(x_{1:T}\mid c)$，而不是显式回答“采取动作 \(a_t\) 后会发生什么”。

## 1. 从视频预测到概率生成

早期深度视频预测通常直接优化下一帧或未来帧误差：

$$
\hat{x}_{t+1:T}=f_\theta(x_{1:t})
$$

这类方法直观，但未来具有多模态性。一个人可以向左走，也可以向右走；单一 MSE 目标容易把多个合理未来平均成模糊画面。后来的 VAE、GAN、diffusion 和 flow，本质上都在解决“如何表达多个可能未来”。

## 2. VAE 路线：用 latent 表达不确定性

VAE 将视频生成写成潜变量模型：

\[
p_\theta(x\mid z),\quad z \sim p(z)
\]

对于视频，它的关键价值是用随机 latent 表达未来的不确定性，并把内容、运动、速度或风格放进可采样空间。典型问题包括 posterior collapse、重建模糊，以及 latent 是否真的对应可解释运动。

代表工作：

- **Stochastic Video Generation with a Learned Prior**：用 learned prior 生成多种可能未来，强调 stochastic prediction。
- **MoCoGAN**：虽然是 GAN，但它的内容 latent 与运动 latent 解耦思想也常被放入 VAE/GAN 共同脉络中理解。

阅读时重点看三件事：latent 是全局还是逐帧采样，时间 dynamics 如何建模，评测是否覆盖多模态未来而不只是平均误差。

## 3. GAN 路线：用判别器追求锐利时空真实感

GAN 将生成器和判别器对抗训练：

\[
\min_G \max_D \mathbb{E}_{x\sim p_{data}}\log D(x)+\mathbb{E}_{z\sim p(z)}\log(1-D(G(z)))
\]

在视频中，判别器可以看单帧、短片段或完整时空体。它比 MSE 更容易产生锐利画面，也推动了内容/运动解耦、前景/背景分离和高分辨率视频生成。

代表工作：

- **Generating Videos with Scene Dynamics**：用时空卷积 GAN 生成短视频，并显式区分前景和背景。
- **MoCoGAN**：把内容和运动分离，形成早期视频生成的重要基线。
- **DVD-GAN**：展示大型视频 GAN 在复杂数据集上的能力。

GAN 的核心短板也很清楚：训练不稳定、mode collapse、覆盖度难评估，以及很难像 diffusion 那样稳定吃下更大的数据和模型规模。

## 4. Diffusion 路线：从噪声逐步还原视频

Diffusion 将生成过程拆成多步去噪：

\[
x_T \sim \mathcal{N}(0,I), \quad x_{t-1}=g_\theta(x_t,t,c)
\]

视频 diffusion 的关键设计是如何处理时间维：

- 在像素或 latent 空间中联合去噪整个视频。
- 复用图像 diffusion backbone，再增加 temporal layer。
- 通过时空注意力、3D U-Net、patch Transformer 或级联超分辨率扩展到高分辨率和长时长。

代表工作：

- **Video Diffusion Models**：系统展示 diffusion 可以用于视频生成、预测和插帧。
- **Make-A-Video / Imagen Video**：把大规模文本图像模型扩展到文本视频。
- **Align Your Latents / Stable Video Diffusion**：推动 latent video diffusion 与开放权重基线。
- **Lumiere**：用 Space-Time U-Net 直接覆盖完整视频时间范围。
- **Sora**：用 spacetime patch 和 Transformer diffusion 扩展到更长、更复杂的视频生成。

Diffusion 成为主线的原因不是它最便宜，而是它在质量、覆盖、多条件控制和规模化稳定性上形成了很强的组合优势。

## 5. Flow 与 consistency：减少采样成本

Flow matching、rectified flow、consistency model 等方法试图把多步去噪变成更直接的分布变换。它们在视频中的动机尤其强：视频采样成本高，任何能减少步数、降低延迟、提高长时一致性的训练或采样方法都会非常重要。

阅读这一路线时，建议把问题拆开：

- 训练目标是否仍然学习从简单分布到数据分布的连续路径？
- 采样步数减少后，运动连续性和细节是否保持？
- 长视频生成是一次完成、分块生成，还是带记忆地递推？

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
6. **Sora technical report**：理解大规模视频生成与 world simulator 讨论的连接点。
