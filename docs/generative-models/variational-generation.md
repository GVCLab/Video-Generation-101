# 变分生成：用潜变量表达多种可能未来

变分生成以潜变量模型描述视频：观测视频由不可直接观测的随机变量 $z$ 产生。它的重要价值不只是压缩，而是把“未来可能有多种合理结果”显式写进模型。

## 1. 从潜变量到 VAE

VAE 把生成模型定义为 [[1]](#ref-1)：

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz
$$

真实后验 $p(z\mid x)$ 通常难以计算，VAE 引入编码器 $q_\phi(z\mid x)$ 近似后验，并最大化证据下界（ELBO）：

$$
\log p_\theta(x)\ge
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{KL}(q_\phi(z\mid x)\Vert p(z))
$$

第一项要求 latent 能重建视频，第二项让后验接近可采样先验。重参数化 $z=\mu+\sigma\odot\epsilon$ 使随机采样可反向传播。

## 2. 视频 VAE 有两种不同角色

**概率生成器**使用随机 latent 描述未来的不确定性，例如给定相同历史帧采样出不同运动。**视频 tokenizer/autoencoder**则把视频压缩到更小的连续或离散空间，供 diffusion、flow 或 Transformer 使用。现代所谓 video VAE 很多主要承担后者；它未必独立负责完整的生成分布。

必须区分：空间 VAE 逐帧压缩，容易保留图像质量但不主动保证时间一致；时空 VAE 用 3D 卷积或时空注意力联合压缩，能利用时间冗余，但可能损失快速运动或引入时间边界伪影。

## 3. 如何组织视频 latent

- **全局 latent**：整段视频共享，适合身份、背景、风格和总体动作意图。
- **逐帧 latent**：每一时刻一个变量，适合局部随机变化，但需时间先验避免抖动。
- **分层 latent**：高层变量描述长期计划，低层变量描述局部运动和纹理。
- **内容—运动解耦**：内容变量在时间上稳定，运动变量随时间演化；解耦通常依赖结构偏置或监督，不能仅凭 latent 维度自动出现。

视频预测常使用 learned prior：训练时后验能看到真实未来，而推理时先验只能看历史。二者通过 KL 项对齐：

$$
q_\phi(z_t\mid x_{\le T})\approx p_\theta(z_t\mid x_{\le t})
$$

## 4. 常见失败模式

**重建模糊**来自简单高斯似然或强压缩；**posterior collapse** 指解码器忽略 $z$，使后验退化到先验；**先验—后验缺口**会导致训练重建好、采样差；**时间不一致**常源于独立逐帧编码；**不可解释 latent** 则说明“内容/运动解耦”只是愿望而非证据。

常见改进包括 KL warm-up、free bits、减弱过强解码器、分层先验、时序相关先验、感知/对抗损失和更好的时空 tokenizer。提高重建清晰度时也要警惕对抗损失产生看似真实但不忠于输入的细节。

## 5. 训练与评测

训练通常同时关注重建、KL 正则、感知质量和时间一致性。作为 tokenizer 时，应报告压缩率、重建 PSNR/SSIM/LPIPS、时间感知指标、运动边缘和下游生成质量。作为随机预测器时，仅报告最佳样本会夸大能力，应同时评估样本多样性、分布覆盖、校准程度，以及多个样本是否都符合历史条件。

## 6. 代表路线

- SVG-LP 用 learned prior 对齐训练时后验和测试时先验，形成经典随机视频预测框架 [[2]](#ref-2)。
- VQ-VAE 将连续 latent 量化为离散码，为 VideoGPT 等自回归模型提供 token [[3]](#ref-3)。
- Latent Diffusion 证明高质量生成可在 autoencoder latent 中完成，显著降低后续去噪成本 [[4]](#ref-4)。
- 现代 latent video diffusion 使用时空 autoencoder，但生成质量仍受其压缩瓶颈支配 [[5]](#ref-5)。

## 7. 何时选择这条机制

需要多未来预测、紧凑可采样表示或为大型生成 backbone 降低成本时，VAE 很合适。若首要目标是极锐利的单次样本，纯 VAE 往往不如 GAN/diffusion；若要求精确 likelihood，ELBO 只是下界；若要可解释控制，还需明确监督或结构设计。

## 8. 与其他机制的组合

VAE 经常是底座而非终点：VAE + GAN 改善感知锐度；VAE + 自回归在离散码上建模；VAE + diffusion/flow 在连续 latent 中生成。判断一个系统时，应分别问 tokenizer 重建了什么，以及生成模型学到了什么。

## 参考文献

<a id="ref-1"></a>[1] [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Diederik P. Kingma, Max Welling. ICLR. 2014.

<a id="ref-2"></a>[2] [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). Remi Denton, Rob Fergus. ICML. 2018.

<a id="ref-3"></a>[3] [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. NeurIPS. 2017.

<a id="ref-4"></a>[4] [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. CVPR. 2022.

<a id="ref-5"></a>[5] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.
