# 对抗生成：用判别器学习时空真实感

生成对抗网络（GAN）让生成器 $G$ 与判别器 $D$ 进行博弈：生成器制造视频，判别器判断真假。与逐像素回归相比，对抗损失不要求生成结果与某个真值逐像素一致，因此更容易产生锐利纹理。

## 1. 基本目标

GAN 的经典 minimax 目标为 [[1]](#ref-1)：

$$
\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}\log D(x)
+\mathbb{E}_{z\sim p(z)}\log(1-D(G(z)))
$$

实践中常改用 non-saturating loss、hinge loss 或 Wasserstein 目标，并配合谱归一化、梯度惩罚和判别器正则化。理论上对抗训练在比较真实分布与生成分布；实践中训练是非平稳双优化，平衡非常敏感。

## 2. 视频判别器看什么

- **图像判别器**逐帧检查纹理与物体外观，却可能忽略闪烁。
- **时序判别器**查看连续帧或 3D 特征，检查运动是否连贯。
- **多尺度判别器**分别处理低分辨率全局结构和高分辨率细节。
- **条件判别器**同时看文本、类别、首帧或动作，判断视频不仅真实而且符合条件。

DVD-GAN 使用空间与时空判别器分担高分辨率视频的巨大成本 [[3]](#ref-3)。这体现了一个普遍原则：空间真实感和时间真实感相关，但不能假设一个判别头会自动兼顾两者。

## 3. 生成器结构

早期方法从噪声直接生成固定长度时空体；两流结构分别生成静态背景与运动前景 [[4]](#ref-4)；MoCoGAN 把内容 latent 固定在整段视频中，让运动 latent 通过 RNN 演化 [[2]](#ref-2)。条件 GAN 还可接入文本、类别、姿态、语义图或历史帧。

生成器可以一次性输出整段视频，也可以递归生成。前者并行且全局一致，但长度固定、内存昂贵；后者支持变长，却会累积误差。

## 4. 为什么 GAN 锐利，又为什么不稳定

像素损失惩罚与真值的偏差，容易平均多个未来；判别器学习“像不像真实视频”的可适应损失，更关注自然图像流形，因此输出更锐利。但判别器若太强，生成器梯度可能无用；若太弱，又无法约束质量。

典型失败包括 mode collapse、训练振荡、对超参数敏感、伪纹理、时间循环和条件忽略。mode collapse 尤其危险：少数样本看起来很好，并不代表模型覆盖了真实视频的多样性。

## 5. 稳定训练的常用方法

常用工具包括 hinge/Wasserstein 损失、谱归一化、R1 或梯度惩罚、生成器指数滑动平均、判别器更新比控制、数据增强、分辨率渐进训练和大批量训练，其中多数经验来自图像侧的大规模 GAN 实践 [[5]](#ref-5)。视频中还要控制片段采样策略：判别器只看很短窗口，模型可能学不会长期一致性；窗口太长则训练成本和不稳定性上升。

## 6. 如何正确评测

不要只展示精选样本。应同时检查：单帧质量、运动连贯、样本覆盖、条件一致性、重复模式和长序列漂移。FVD 常用于分布比较，但会受到特征提取器、样本数和实现细节影响；人工评测也应随机化、盲测，并把画质、运动、条件遵循分开提问。

## 7. 历史地位与适用场景

GAN 奠定了现代视频生成中的内容—运动解耦、多判别器和感知质量训练。它在低延迟、单步生成、特定域和可用强监督的场景仍有价值。其主流地位后来被 diffusion 削弱，原因不是 GAN 无法产生高质量结果，而是 diffusion 通常更稳定、更易扩大数据和模型规模，也更容易兼顾覆盖度与条件控制。

## 8. 与其他机制的组合

VAE-GAN 用编码器获得 latent、用判别器改善重建；自回归或 diffusion 模型可加入对抗蒸馏以减少采样步数；视频 tokenizer 也常用感知和对抗损失提升解码锐度。此时要区分：GAN 是完整生成机制，还是只作为训练损失的一部分。

## 参考文献

<a id="ref-1"></a>[1] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, et al. NeurIPS. 2014.

<a id="ref-2"></a>[2] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. CVPR. 2018.

<a id="ref-3"></a>[3] [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571). Aidan Clark, Jeff Donahue, Karen Simonyan. arXiv preprint. 2019.

<a id="ref-4"></a>[4] [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612). Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. NeurIPS. 2016.

<a id="ref-5"></a>[5] [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948). Tero Karras, Samuli Laine, Timo Aila. CVPR. 2019.
