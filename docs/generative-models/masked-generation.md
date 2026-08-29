# 掩码生成：从缺失 token 到并行视频合成

掩码生成（masked generation）先把部分或全部视频 token 替换为 `[MASK]`，再根据已知条件预测缺失内容。模型通常在多轮迭代中保留高置信度结果、重新预测低置信度位置，因此比逐 token 自回归采样更容易并行。

## 1. 训练目标

掩码建模在视觉表征学习中已被证明可扩展 [[1]](#ref-1)。给定离散视频 token $y$，随机选择 mask 集合 $M$，模型学习：

$$
\mathcal{L}_{mask}=-\sum_{i\in M}\log p_\theta(y_i\mid y_{\bar M},c)
$$

训练时 mask 比例可以从少量缺失到接近全部缺失。模型因此同时学会补洞、插帧、条件生成和从粗上下文恢复细节。与 BERT 不同，生成时没有完整真值上下文，需要迭代构造自己的上下文。

## 2. 典型采样过程

迭代并行解码的基本流程沿用自图像侧的 MaskGIT [[4]](#ref-4)：

1. 用 mask token 初始化未知区域。
2. 一次并行预测所有未知位置的 token 分布。
3. 按置信度或调度函数接受一部分预测。
4. 保留已接受 token，对其余位置重新 mask。
5. 重复到没有 mask，或达到固定轮数。

早期轮次确定场景、布局和主要运动，后期轮次补充纹理与局部细节。余弦等 mask schedule 控制每轮解锁多少 token。一次接受太多会锁定早期错误；太少则失去速度优势。

## 3. 视频中的 mask 设计

- **随机 token mask**：通用，但可能过度依赖局部邻居。
- **tube mask**：遮住跨时间的同一空间块，迫使模型推断运动和对象持续性。
- **整帧 mask**：对应插帧或未来预测。
- **时空块 mask**：接近视频修复和局部编辑。
- **因果 mask**：只允许访问过去，可用于预测和交互。
- **条件区域 mask**：保留首帧、关键帧、深度或轨迹作为锚点。

mask 分布应与实际任务匹配。只用独立随机 mask 训练，却声称能可靠生成完整长视频，往往存在训练—推理缺口。

## 4. MAGVIT 路线

MAGVIT 将多个视频任务统一为不同掩码模式：无条件生成、帧预测、插帧和补全都可由同一个 masked token model 处理 [[2]](#ref-2)。它说明掩码不是单一任务，而是一种统一接口；Phenaki 则把同一思路用于变长文本条件视频 [[5]](#ref-5)。MAGVIT-v2 进一步强调 tokenizer 对视觉生成质量和语言模型式生成的重要性 [[3]](#ref-3)。

## 5. 优势

相比 AR，掩码生成每轮能并行预测大量 token，采样轮数可远小于序列长度；双向上下文也适合插帧和局部补全。相比连续 diffusion，它在离散 token 空间可直接使用分类目标，并能按置信度选择重采样区域。

## 6. 局限和失败模式

tokenizer 失真仍是硬上限。独立位置预测还可能产生彼此冲突的 token，表现为物体边界断裂、时间闪烁或动作不连续。置信度不一定等于正确度：模型可能高置信地锁定错误结构。生成轮数、mask 调度、温度和重采样策略都会显著影响质量，方法比较时必须统一这些设置。

对于长视频，模型还可能不断局部修补，却缺少全局故事或运动计划。分层 token、关键帧锚定、全局规划 token 和跨段记忆可以改善这一点，但也增加系统复杂度。

## 7. 何时使用

需要一个模型统一生成、补全、插帧和预测，或希望获得比 token-by-token AR 更低延迟时，masked generation 很有吸引力。若输出是连续 latent，可使用离散化或改用 mask-based continuous modeling；若任务严格要求在线因果响应，则必须限制双向信息，速度优势也可能缩小。

## 8. 实践检查清单

- tokenizer 的时空压缩率和重建质量是多少？
- 训练 mask 是否覆盖真实推理场景？
- 置信度如何校准，错误 token 能否被重新打开？
- 采样轮数变化时质量、速度如何变化？
- 时空邻接 token 是否在每轮形成一致结构？
- 长视频是否有跨片段的身份和状态记忆？

## 参考文献

<a id="ref-1"></a>[1] [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. CVPR. 2022.

<a id="ref-2"></a>[2] [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. CVPR. 2023.

<a id="ref-3"></a>[3] [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. ICLR. 2024.

<a id="ref-4"></a>[4] [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200). Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman. CVPR. 2022.

<a id="ref-5"></a>[5] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://arxiv.org/abs/2210.02399). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.
