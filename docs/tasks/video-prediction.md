# 视频预测

## 任务定义

Video prediction 输入历史帧，预测未来帧或未来 latent。它比无条件生成多了时间上下文，比 text-to-video 更接近物理与状态建模，也是 world model 和机器人规划的重要前身。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要瓶颈 |
|---|---|---|---|
| 光流与运动补偿 | Lucas-Kanade、Horn-Schunck、block matching | 显式估计运动并 warp | 处理不了复杂非刚体与新区域 |
| 状态空间 | Dynamic Textures、Kalman / LDS | latent state + linear dynamics | 表达能力有限 |
| CNN/RNN | LSTM、ConvLSTM、PredNet | recurrent temporal modeling | MSE 模糊、误差累积 |
| 变换预测 | DNA、CDNA、STP | 预测像素移动核或空间变换 | 长程和遮挡仍困难 |
| 随机预测 | SVG、VRNN 类方法 | latent stochastic future | 多样性评测困难 |
| Diffusion / Transformer | video diffusion prediction、FramePack、next-frame DiT | 条件去噪或固定上下文记忆 | 推理成本、长上下文 |
| World model | Dreamer、V-JEPA 2-AC、WAM | action-conditioned latent rollout | 反事实和真实任务验证 |

## 技术演化逻辑

视频预测最早是估计运动。深度学习把它变成端到端的自监督任务，但确定性预测很快遇到多未来问题：平均损失会鼓励模糊。后来方法引入随机 latent、对抗损失、flow/warp 结构和 latent dynamics。现在的重点转向两个方向：用 diffusion/DiT 生成高质量未来，或只预测对行动有用的 latent state。

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

## 推荐阅读

- Beyond MSE、ConvLSTM、PredNet：深度视频预测基础。
- CDNA / DNA / STP：动作与变换预测。
- Stochastic Video Generation with a Learned Prior：随机未来。
- Video Diffusion Models、FramePack：现代生成式预测。
- Dreamer、V-JEPA 2-AC、World Action Models：面向行动的预测。
