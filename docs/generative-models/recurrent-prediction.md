# 递归预测：逐步生成未来视频

递归预测（recurrent prediction）把视频看成一个随时间推进的序列：模型读取历史帧或历史 latent，预测下一步，再把自己的输出作为下一次输入。它是视频预测、动作条件世界模型和在线交互生成最直观的基础机制。

## 1. 核心直觉与概率分解

对视频 $x_{1:T}$，链式法则给出：

$$
p(x_{1:T}\mid c)=\prod_{t=1}^{T}p(x_t\mid x_{<t},c)
$$

递归模型用隐藏状态压缩历史：

$$
h_t=F_\theta(h_{t-1},x_t,c),\qquad \hat{x}_{t+1}=G_\theta(h_t)
$$

这里的“递归”描述的是时间推进方式，不限定网络必须是 RNN。ConvLSTM、状态空间模型、递归 Transformer、逐帧 diffusion，甚至带固定记忆的 next-frame DiT 都可以采用递归 rollout。

## 2. 视频中究竟预测什么

- **像素**：直接输出 RGB，概念简单但计算重，容易用平均值解释不确定未来。
- **运动变换**：预测光流、卷积核、仿射变换或遮挡 mask，再 warp 已有像素；适合短期、可见运动。
- **连续 latent**：先用 autoencoder 压缩视频，再预测潜在状态；成本低，但上限受 tokenizer 约束。
- **离散 token**：预测下一个视觉 token，可使用分类损失和 Transformer。
- **结构化状态**：预测对象、深度、相机、动作结果等，更适合规划，但难覆盖全部视觉细节。

## 3. 训练流程

最常见训练方式是 teacher forcing：第 $t$ 步输入真实帧 $x_t$，优化下一帧损失。

$$
\mathcal{L}=\sum_t \ell(\hat{x}_{t+1},x_{t+1})
$$

问题在于推理时输入变成模型自己的预测，训练与推理分布不一致，形成 exposure bias。微小错误进入下一步后会被放大，表现为纹理漂移、身份变化、物体消失和运动失控。

常见缓解方法包括 scheduled sampling、多步联合训练、在预测样本上继续训练、噪声扰动历史、周期性重新锚定关键帧，以及维护有限但高质量的上下文记忆。它们能减轻漂移，却不能从根本上消除闭环分布偏移。

## 4. 从确定性预测到随机递归

若仅用 MSE 训练，模型面对“向左或向右都合理”的未来时，容易输出两个未来的平均，造成模糊 [[4]](#ref-4)。概率递归模型因此引入随机变量：

$$
z_t\sim p_\theta(z_t\mid x_{\le t}),\qquad
x_{t+1}\sim p_\theta(x_{t+1}\mid x_{\le t},z_{\le t})
$$

随机变量可以是每段共享的全局 latent，也可以逐帧采样。前者更利于保持全局意图，后者更灵活但可能产生时间抖动。实际系统常把二者结合：全局变量控制身份、场景和总体动作，局部变量表达细节变化。

## 5. 典型结构谱系

1. **RNN/LSTM**：用隐藏状态建模时间，但全连接结构不保留空间布局。
2. **ConvLSTM**：把门控运算换成卷积，隐藏状态仍是空间特征图 [[1]](#ref-1)。
3. **变换式预测**：DNA、CDNA、STP 预测像素移动方式，减少重新合成整帧的负担 [[2]](#ref-2)。
4. **随机视频预测**：SVG 用 learned prior 表达多种未来 [[3]](#ref-3)。
5. **递归 latent/DiT**：在压缩空间预测下一段，并用记忆选择或上下文打包控制长期漂移 [[5]](#ref-5)。

## 6. 优势、局限与适用场景

递归预测的优势是因果、可变长、适合流式运行，并能在每一步接收动作或新观测。它特别适合天气临近预报、机器人控制、游戏模拟和在线 world model。

主要限制有三类：误差累积；串行生成限制并行度；长历史必须被压进有限状态，容易遗忘。它也容易把“视觉上延续得像”误当成“动力学正确”。对动作条件系统，必须用反事实动作、长期状态保持和闭环任务收益验证。

## 7. 实践检查清单

- 训练时是否包含多步 rollout，而不只优化一步预测？
- 推理时的上下文来自真值还是模型输出？
- latent 压缩是否丢失小物体和高速运动？
- 随机性在时间上是否一致，还是逐帧闪烁？
- 长序列评测是否报告随时间增长的误差曲线？
- 对动作条件模型，换一个动作是否得到正确而不同的结果？

## 8. 与其他生成机制的关系

递归预测规定“按时间逐步生成”，而 VAE、GAN、diffusion、flow 规定“每一步怎样建模分布”。因此可以有递归 VAE、递归 GAN 或逐帧 diffusion。它也与[自回归生成](autoregressive-generation.md)重叠：前者强调状态递推，后者强调联合分布的顺序分解。

## 参考文献

<a id="ref-1"></a>[1] [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo. NeurIPS. 2015.

<a id="ref-2"></a>[2] [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). Chelsea Finn, Ian Goodfellow, Sergey Levine. NeurIPS. 2016.

<a id="ref-3"></a>[3] [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). Remi Denton, Rob Fergus. ICML. 2018.

<a id="ref-4"></a>[4] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. ICLR. 2016.

<a id="ref-5"></a>[5] [Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models (FramePack)](https://arxiv.org/abs/2504.12626). Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, Maneesh Agrawala. arXiv preprint. 2025.
