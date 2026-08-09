# 任务与方法分类

视频生成领域容易混乱，通常不是因为模型太多，而是不同工作使用了同一个词，却在解决不同的问题。本页提供一套适合阅读论文和维护时间线的统一分类。

## 一、按生成任务分类

| 任务 | 条件 | 输出 | 主要难点 |
|---|---|---|---|
| [Unconditional video generation](tasks/unconditional-video-generation.md) | 无或类别标签 | 新视频 | 多样性与时空真实性 |
| [Text-to-video](tasks/text-to-video.md) | 文本 | 与描述匹配的视频 | 语义、动作、镜头和叙事遵循 |
| [Image-to-video](tasks/image-to-video.md) | 首帧或参考图 | 动态视频 | 保持身份和布局，同时产生合理运动 |
| [Video prediction](tasks/video-prediction.md) | 历史帧 | 未来帧 | 多模态未来、长期误差和不确定性 |
| [Frame interpolation](tasks/frame-interpolation.md) | 前后关键帧 | 中间帧 | 大运动、遮挡和运动边界 |
| [Video-to-video](tasks/video-to-video.md) | 视频与编辑指令 | 编辑后视频 | 结构保持与跨帧一致性 |
| [Video inpainting](tasks/video-inpainting.md) | 视频与 mask | 补全区域 | 遮挡、运动传播和背景一致性 |
| [Story / multi-shot generation](tasks/story-multishot.md) | 剧本、分镜、参考素材 | 多镜头序列 | 角色、场景、因果和镜头连续性 |
| [Action-conditioned prediction](tasks/action-conditioned-prediction.md) | 观测与动作 | 动作后的未来 | 动作可控性和反事实准确性 |
| [Interactive world generation](tasks/interactive-world-generation.md) | 初始世界与实时动作 | 持续生成的环境 | 低延迟、记忆、3D 和闭环稳定性 |

每个任务的完整技术演化见 [生成任务深度路线图](tasks/index.md)。

## 二、按表示空间分类

### Pixel space

直接生成 RGB 像素。信息完整，但计算和存储成本高；模型还要花大量容量重建对决策无关的细节。

### Motion space

预测光流、仿射变换、深度、轨迹或 deformation，再通过 warping 得到视频。控制性强，但在新出现区域和复杂非刚体运动上容易失败。

### Continuous latent space

先用 autoencoder 将视频压缩，再在 latent 中建模。这是 latent diffusion 和许多现代视频基础模型的常见选择。

### Discrete token space

使用 VQ-VAE、VQGAN 或视频 tokenizer 将视频转换为离散 token，再用 autoregressive 或 masked Transformer 生成。优点是能借用语言模型的建模方式，缺点是 token 数量巨大且 tokenizer 会引入不可逆失真。

### Structured / object-centric state

使用对象、深度、相机、3D 几何、场景图或可学习 slot 表示世界。它更适合物理推理和规划，但很难覆盖开放世界的全部视觉细节。

## 三、按生成机制分类

### Recurrent prediction

逐帧或逐 latent 预测未来。结构直观，适合在线 rollout；但训练时看到真实历史、推理时看到自身输出，会产生 exposure bias。

### Variational generation

通过隐变量表示未来的不确定性。适合多模态预测，但 ELBO 的优化目标容易造成模糊或 posterior collapse。

### Adversarial generation

判别器推动生成器产生锐利、真实的视频。GAN 曾是高质量视频生成的主线，但训练不稳定、mode collapse 和大规模扩展困难。

### Autoregressive generation

把视频表示成序列并建模：

$$
p(z_{1:N})=\prod_i p(z_i\mid z_{<i})
$$

它具有明确似然并天然支持变长输出，但采样串行，长期误差可能逐步积累。

### Masked generation

从全部或部分 mask 的 token 开始，多轮并行填充。通常比逐 token 自回归更快，也适合视频补全和多任务条件生成。

### Diffusion and flow

从噪声或简单分布出发，通过迭代去噪或连续概率流到达数据分布。它们在画质、覆盖度和条件控制上表现突出，但需要解决多步推理和超长时序一致性。

## 四、按条件和控制信号分类

- **语义条件**：文本、类别、剧本。
- **视觉条件**：首帧、参考图、参考视频、首尾帧。
- **结构条件**：姿态、边缘、深度、分割、关键点。
- **运动条件**：轨迹、光流、相机路径、动作序列。
- **音频条件**：语音、音乐、环境声。
- **智能体动作**：离散按键、控制量、机器人关节或末端执行器动作。

最后一类是区分创作型视频模型和决策型 world model 的重要信号。

## 五、时间建模的四个尺度

1. **帧内空间结构**：物体形状、纹理、文字和局部几何。
2. **短期运动**：速度、光流、碰撞、姿态和相机运动。
3. **场景状态**：对象永久性、遮挡后的记忆、人物身份和环境变化。
4. **长程叙事或任务**：多镜头因果、目标进展和智能体策略。

只解决第 1、2 层的模型可以生成漂亮短片，却可能完全没有第 3、4 层所需的状态记忆。

## 六、三个经常被混用的“可控性”

### Prompt steerability

修改文字能否改变输出。它衡量语义遵循，不等于精确控制。

### Trajectory controllability

给定相机路径、物体轨迹或动作，输出是否按指定轨迹演化。

### Closed-loop interactivity

用户或智能体在生成过程中连续提供动作，模型必须低延迟响应，并保持之前世界状态的一致性。

## 七、判断一项工作属于哪一类

阅读论文时，建议记录以下字段：

```yaml
task: text-to-video | prediction | editing | action-conditioned | interactive
representation: pixel | continuous-latent | discrete-token | structured-state
generator: recurrent | vae | gan | autoregressive | masked | diffusion | flow
conditions: [text, image, video, audio, camera, action]
temporal_horizon: frames | seconds | minutes
interaction: open-loop | chunked-control | closed-loop
evaluation: [quality, alignment, physics, memory, action, planning]
availability: paper | code | weights | api
```

这套记录方式比单纯按公司或发布时间整理更容易发现真正的技术演化。
