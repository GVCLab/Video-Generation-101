# 技术时间线

本时间线选择改变了“视频如何表示、生成或被用于决策”的工作。年份以论文首次公开或官方发布为主，不代表同年所有工作具有相同成熟度。

## 1990s–2003：显式运动与统计动态

### 关键思想

- 运动补偿、光流、morphing 和 image-based rendering。
- 将视频看作可重组的帧或可辨识的动态系统。
- 显式模拟器通过几何、材质和物理状态生成画面。

### 代表节点

- **1981 — Lucas–Kanade**：用局部最小二乘估计光流。
- **1981 — Horn–Schunck**：用全局平滑正则化估计密集光流。
- **2000 — Video Textures**：通过寻找相似帧和跳转点，将短视频重组为连续动态纹理。
- **2003 — Dynamic Textures**：以线性动力系统建模烟、火、水面和树叶等随机视觉过程。

### 留下的遗产

现代模型中的 flow-guided generation、warping、显式相机控制、state-space model 和数字孪生，都能在这一时期找到祖先。

## 2014–2016：深度视频预测

### 关键思想

- 使用 CNN 编码空间内容，用 RNN/LSTM 建模时间。
- 将自监督下一帧预测作为学习视觉表示的方法。
- 从直接预测像素转向预测变换、运动核和多尺度残差。

### 代表节点

- **2014 — Video sequence modeling with RNNs**：探索用递归网络预测视频。
- **2015 — ConvLSTM**：把 LSTM 的全连接运算替换为卷积，保留空间结构。
- **2015 — Beyond MSE**：指出 MSE 会产生平均化和模糊，引入多尺度与对抗损失。
- **2016 — CDNA / DNA / STP**：通过预测像素变换和运动核学习物理交互。
- **2016 — PredNet**：以预测编码思想组织深层视频预测网络。

### 核心矛盾

未来通常不是唯一的。确定性网络在多个合理未来之间求平均，因此“数值误差较低”可能对应“视觉结果更模糊”。

## 2016–2019：VAE 与 GAN 视频生成

### 关键思想

- 使用随机 latent 表达不确定未来。
- 用时空判别器替代逐像素损失。
- 解耦静态内容和动态运动。

### 代表节点

- **2016 — Generating Videos with Scene Dynamics**：使用时空卷积 GAN，并分离前景和背景。
- **2017 — MoCoGAN**：把内容 latent 和运动 latent 分开建模。
- **2018 — Stochastic Video Generation**：进一步研究可控、随机和长程的视频生成。
- **2019 — DVD-GAN**：展示大型 GAN 在高分辨率视频上的扩展能力。

### 核心矛盾

GAN 能产生锐利画面，但大规模训练不稳定、模式覆盖难以评估，也缺少像 likelihood 那样统一的训练目标。

## 2017–2023：视觉 Token 与 Transformer

### 关键思想

- 先压缩视频，再对离散 token 建模。
- 使用自回归、masked token prediction 和时空注意力。
- 将文本、图像和视频放入更统一的序列建模框架。

### 代表节点

- **2017 — VQ-VAE**：学习离散视觉 codebook。
- **2021 — VideoGPT**：VQ-VAE 视频 tokenizer + GPT 式自回归 Transformer。
- **2022 — Phenaki**：用 causal video tokenizer 和 masked Transformer 生成可变长度视频。
- **2022/2023 — MAGVIT**：用 3D tokenizer 和 masked generation 统一多种视频生成任务。
- **2023 — MAGVIT-v2**：继续提升视觉 tokenizer 和语言模型兼容性。

### 核心矛盾

压缩减少计算，却可能丢失运动细节；自回归生成具有统一概率建模形式，但视频 token 数量和串行采样成本非常高。

## 2020–2024：Diffusion 成为主线

### 关键思想

- 从噪声逐步还原视频分布。
- 复用大规模图像生成模型，再增加 temporal layer 或时空注意力。
- 在 latent space 训练，并通过级联超分辨率提高输出质量。

### 代表节点

- **2020 — DDPM**：奠定现代 diffusion 生成框架。
- **2022 — Video Diffusion Models**：系统展示时空 diffusion 在视频生成和预测中的能力。
- **2022 — Make-A-Video / Imagen Video**：大规模文本视频生成与级联时空超分辨率。
- **2022/2023 — Latent Video Diffusion**：将生成过程移动到压缩 latent。
- **2023 — AnimateDiff**：向图像 diffusion 注入可复用 motion module。
- **2023 — Stable Video Diffusion**：开放权重图像到视频模型推动社区复现。
- **2024 — Lumiere**：以 Space-Time U-Net 直接生成完整视频时间范围。
- **2024 — Sora**：把不同时长、尺寸和宽高比的视频表示为空时 patch，并以 Transformer diffusion 扩展规模。

### 核心矛盾

短视频画质快速提高，但对象永久性、因果、复杂交互、超长状态一致性和推理成本仍是主要问题。

## 2018–2023：决策型 World Model 的并行谱系

这一谱系并不是从文本视频生成自然“升级”而来，而是来自控制、强化学习和规划。

### 代表节点

- **2018 — World Models**：用 VAE 表示观测、RNN 建模 dynamics，并在想象环境中训练 controller。
- **2019 — PlaNet**：在 latent dynamics 中进行在线规划。
- **2020 — Dreamer**：在学习到的 latent world model 中优化行为。
- **2020 — MuZero**：不重建完整观测，学习足以预测价值、策略和奖励的模型。
- **2023 — DreamerV3**：展示一套 world model 方法跨多种任务工作的可能性。
- **2023 — GAIA-1**：探索面向自动驾驶的生成式 world model。

### 核心矛盾

面向决策的 latent state 不一定需要生成漂亮像素；面向视频的生成器也不一定包含对规划有用的、因果一致的状态。

## 2024：生成模型与交互世界汇合

- **V-JEPA**：在 representation space 预测缺失的时空信息，而非重建全部像素。
- **Genie 1**：从无动作标签的互联网游戏视频中发现 latent action，并生成可控制环境。
- **GameNGen**：使用神经生成模型模拟可玩的游戏环境。
- **Sora**：提出大规模视频生成可能通向通用物理世界模拟器，同时公开展示模型的物理失败。

这一年之后，“视频生成模型是否就是 world model”成为领域的核心争论之一。

## 2025：动作、物理与实时交互

- **Cosmos 1**：提供面向 Physical AI 的 world foundation model、tokenizer 和数据处理平台。
- **V-JEPA 2**：将视频自监督表示、物理预测和少量机器人数据后训练连接到 zero-shot planning。
- **Veo 3**：除原生音视频生成外，研究显示其生成预训练中出现了分割、深度、物理属性和 affordance 等零样本能力。
- **Genie 3**：从文本生成可实时导航、维持数分钟的交互环境。
- **Sora 2**：加强物理、可控性和同步音频，同时更加突出肖像、来源与安全问题。
- **GWM-1**：将可探索世界、实时 avatar 和机器人动作条件 rollout 放在一个 general world model 家族下。

## 2026：Omnimodal Physical AI

- **Cosmos 3**：尝试用统一模型覆盖语言、图像、视频、音频、动作、forward dynamics、inverse dynamics 和机器人策略。
- **V-JEPA 2.1**：通过 dense predictive loss、deep self-supervision 和图像—视频输入路径强化时空 dense feature。
- **LeWorldModel / EB-JEPA**：一条路线研究从像素端到端学习可规划 latent dynamics，另一条路线提供单卡可复现的 JEPA 教学与研究组件。
- **Seedance 2.0、Kling 3.0 等创作模型**：进一步强化原生音视频、多模态参考、多镜头叙事和生成—编辑一体化。
- 研究重心从“生成一段看起来真实的视频”继续转向“能否记忆、交互、预测动作结果，并为真实策略带来收益”。

## 如何阅读这条时间线

建议不要只记模型名，而是为每个时代回答四个问题：

1. 视频被压缩或表示成了什么？
2. 时间变化是通过什么机制建模的？
3. 模型接受何种控制信号？
4. 成功是通过画质、预测误差，还是闭环任务完成率证明的？

JEPA 从图像表征、视频预测到动作条件规划的独立演化见 [JEPA 路线专章](jepa.md)。
