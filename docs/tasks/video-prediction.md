# 视频预测

## 任务定义

Video prediction 输入历史帧，预测未来帧或未来 latent。它比无条件生成多了时间上下文，比 text-to-video 更接近物理与状态建模，也是 world model 和机器人规划的重要前身。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要瓶颈 |
|---|---|---|---|
| 光流与运动补偿 | Lucas-Kanade [[1]](#ref-1)、Horn-Schunck [[2]](#ref-2)、block matching | 显式估计运动并 warp | 处理不了复杂非刚体与新区域 |
| 状态空间 | Dynamic Textures [[3]](#ref-3)、Kalman / LDS | latent state + linear dynamics | 表达能力有限 |
| CNN/RNN | LSTM [[4]](#ref-4)、ConvLSTM [[5]](#ref-5)、PredNet [[6]](#ref-6) | recurrent temporal modeling | MSE 模糊、误差累积 |
| 变换预测 | DNA、CDNA、STP [[7]](#ref-7) | 预测像素移动核或空间变换 | 长程和遮挡仍困难 |
| 随机预测 | SVG [[8]](#ref-8)、VRNN 类方法 | latent stochastic future | 多样性评测困难 |
| Diffusion / Transformer | Video Diffusion Models [[9]](#ref-9)、FramePack、next-frame DiT | 条件去噪或固定上下文记忆 | 推理成本、长上下文 |
| World model | Dreamer [[10]](#ref-10)、V-JEPA 2-AC [[11]](#ref-11)、WAM | action-conditioned latent rollout | 反事实和真实任务验证 |

## 技术演化逻辑

视频预测最早是估计运动 [[1]](#ref-1), [[2]](#ref-2)。深度学习把它变成端到端的自监督任务 [[4]](#ref-4), [[5]](#ref-5)，但确定性预测很快遇到多未来问题：平均损失会鼓励模糊 [[12]](#ref-12)。后来方法引入随机 latent、对抗损失、flow/warp 结构和 latent dynamics [[7]](#ref-7), [[8]](#ref-8)。现在的重点转向两个方向：用 diffusion/DiT 生成高质量未来 [[9]](#ref-9)，或只预测对行动有用的 latent state [[10]](#ref-10), [[11]](#ref-11)。

## 最新趋势

- 用大型 I2V/T2V diffusion backbone 做 future frame generation。
- 用 fixed-size context memory 处理更长历史，例如 FramePack 类结构。
- 在 latent space 中预测 future representation，减少像素重建负担。
- 将 prediction 与 action、reward、policy 或 planning 结合，进入 world model。

## 关键评测

- 多种合理未来是否都能表达。
- 预测是否保持对象永久性和遮挡后的状态。
- 长 rollout 是否逐步崩溃。
- 预测误差是否对应真实任务收益。
- 对历史帧中小事件的因果后果是否敏感。

## 开放问题

1. 对未来不确定性应该输出样本、分布、latent 还是可规划状态？
2. 高质量像素预测和决策有用性是否一致？
3. 如何避免 closed-loop rollout 中的 exposure bias？
4. 是否能从被动视频中学到可干预 dynamics？
本页主要参考工作：Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models [[13]](#ref-13)。

## 参考文献

<a id="ref-1"></a>[1] [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Bruce D. Lucas and Takeo Kanade. Proceedings of IJCAI. 1981.

<a id="ref-2"></a>[2] [Determining optical flow](https://doi.org/10.1016/0004-3702(81)90024-2). Berthold K. P. Horn and Brian G. Schunck. Artificial Intelligence. 1981.

<a id="ref-3"></a>[3] [Dynamic Textures](https://doi.org/10.1023/A:1021669406132). Gianfranco Doretto, Alessandro Chiuso, Ying Nian Wu, Stefano Soatto. International Journal of Computer Vision. 2003.

<a id="ref-4"></a>[4] [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681). Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov. arXiv preprint. 2015.

<a id="ref-5"></a>[5] [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo. arXiv preprint. 2015.

<a id="ref-6"></a>[6] [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104). William Lotter, Gabriel Kreiman, David Cox. arXiv preprint. 2016.

<a id="ref-7"></a>[7] [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). Chelsea Finn, Ian Goodfellow, Sergey Levine. arXiv preprint. 2016.

<a id="ref-8"></a>[8] [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687). Remi Denton and Rob Fergus. arXiv preprint. 2018.

<a id="ref-9"></a>[9] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. arXiv preprint. 2022.

<a id="ref-10"></a>[10] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. arXiv preprint. 2019.

<a id="ref-11"></a>[11] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. arXiv preprint. 2025.

<a id="ref-12"></a>[12] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. arXiv preprint. 2015.

<a id="ref-13"></a>[13] [Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models](https://arxiv.org/abs/2504.12626). Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, Maneesh Agrawala. arXiv preprint. 2025.
