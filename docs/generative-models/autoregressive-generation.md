# 自回归生成：把视频变成可预测的序列

自回归生成（autoregressive generation, AR）把联合分布严格分解为一系列条件分布，依次预测下一个像素、token、patch、帧或 latent。它与语言模型的建模方式最接近。

## 1. 概率基础

自回归分解最早在像素级别被系统验证 [[1]](#ref-1)。将视频编码为序列 $y_{1:N}$ 后：

$$
p(y_{1:N}\mid c)=\prod_{i=1}^{N}p(y_i\mid y_{<i},c)
$$

训练时可并行计算每个位置的交叉熵：

$$
\mathcal{L}_{AR}=-\sum_i\log p_\theta(y_i\mid y_{<i},c)
$$

采样时却必须等待前一个 token，形成天然串行瓶颈。AR 的“顺序”不一定等于时间顺序：可以逐像素扫描、先空间后时间、按块生成，或先低分辨率再细化。

## 2. 为什么通常先做视频 tokenizer

原始视频的像素序列极长。常见做法是用 VQ-VAE [[4]](#ref-4)、VQGAN 或时空 tokenizer 把视频变成离散码，再由 Transformer 预测码序列，最后解码回像素。tokenizer 在这里决定三件事：序列长度、重建上限和模型需要学习多少局部视觉细节。

压缩太弱会令上下文爆炸；压缩太强会永久丢失文字、手部和高速运动。改进 backbone 无法恢复 tokenizer 已经删除的信息，因此 tokenizer 必须单独验收。

## 3. 序列化策略

- **帧内优先**：完成一帧所有空间 token 再到下一帧，因果直观但相邻时间 token 距离远。
- **时空交错**：让同一空间位置的时间 token 更接近，利于运动建模。
- **分块/多尺度**：先生成全局低频结构，再生成局部细节。
- **层级序列**：高层 token 负责场景与故事，低层 token 负责帧级实现。

位置编码必须同时表达时间、水平、垂直位置和可能变化的宽高比。长视频还需滑动窗口、稀疏注意力、记忆压缩或分段条件。

## 4. 训练与采样

teacher forcing 让训练稳定高效，但推理使用模型自己的历史，仍存在 exposure bias。采样策略包括 temperature、top-k、top-p、classifier-free guidance 式条件引导及重复惩罚。温度低会提高确定性但减少多样性；温度高会放大错误。视频中某个错误 token 可能持续影响后续身份和几何。

KV cache 能避免重复计算历史注意力，却不能消除串行步数和随序列增长的缓存占用。并行预测多个 token 会加速，但通常已向 blockwise 或 masked generation 过渡。

## 5. 代表路线

VideoGPT 将 VQ-VAE 视频码与 Transformer 结合，展示了离散视频 token 的 AR 生成 [[2]](#ref-2)。Phenaki 使用因果生成支持可变长度视频，并通过时间变化的文本条件构建长视频 [[3]](#ref-3)。后续视觉语言模型进一步统一文本、图像、视频与动作 token [[5]](#ref-5)，但“统一词表”不等于这些模态已经具有相同误差代价或时间尺度。

## 6. 优势与局限

优势是 likelihood 目标清晰、训练稳定、天然支持变长和前缀条件，并可复用语言模型的 scaling 与推理基础设施。局限是采样串行、视频 token 极多、错误会向后传播，且 token 顺序本身引入人为归纳偏置。

AR 很适合离散动作—视频联合模型、交互环境、长叙事规划和需要逐步接收条件的系统。若重点是一次性生成高分辨率短视频，diffusion、flow 或 masked 方法往往更易并行。

## 7. 评测重点

- tokenizer 重建质量与生成模型质量要分开报告。
- 比较不同长度下身份、场景和运动是否持续稳定。
- 检查采样温度改变后质量—多样性的完整曲线。
- 报告真实端到端延迟，而不只算单 token 吞吐。
- 对交互模型测试动作响应与反事实，而不只播放固定轨迹。

## 8. 与递归和掩码生成的关系

AR 是概率分解方式；递归预测是状态推进方式，两者常同时出现。Masked generation 则删除固定左到右顺序，通过多轮并行补全 token，通常更快，但需要额外设计置信度、mask 调度和停止条件。

## 参考文献

<a id="ref-1"></a>[1] [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759). Aäron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu. ICML. 2016.

<a id="ref-2"></a>[2] [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas. arXiv preprint. 2021.

<a id="ref-3"></a>[3] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://arxiv.org/abs/2210.02399). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.

<a id="ref-4"></a>[4] [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. NeurIPS. 2017.

<a id="ref-5"></a>[5] [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. ICLR. 2024.
