# 从视频生成到 World Model

“World model”是当前视频生成领域最有吸引力、也最容易被滥用的术语。本页给出一个操作性定义，并将不同技术路线放在同一坐标系中比较。

## 1. 最小定义

一个面向智能体的 world model，应当让系统根据当前状态和动作预测环境如何变化：

$$
p(s_{t+1}, o_{t+1}, r_{t+1}\mid s_t,a_t)
$$

- $s_t$：模型内部状态，可以是显式变量或 latent。
- $o_t$：智能体可以观察到的图像、声音或传感器数据。
- $a_t$：键盘输入、相机控制、机器人动作或其他干预。
- $r_t$：任务相关的奖励、成本或成功信号，并非所有 world model 都显式预测它。

普通视频生成模型更常见的目标是：

$$
p(x_{1:T}\mid \text{text, image, audio, video})
$$

它可以学习大量物理和视觉规律，但如果没有动作条件、反事实验证和闭环使用证据，称其为“可用于决策的 world model”仍需谨慎。

## 2. 两条历史路线

### 路线 A：生成式视觉模型

```text
视频预测 → GAN / VAE → 视频 Token → Diffusion / Flow → 视频基础模型
```

优化重点：画质、多样性、语义遵循、运动、音视频同步和编辑能力。

### 路线 B：控制与强化学习

```text
系统辨识 → 状态空间模型 → Model-based RL → Latent imagination → Planning
```

优化重点：动作结果、奖励、价值、长期 rollout、样本效率和任务成功率。

现代 world foundation model 正试图把两条路线合并：既拥有开放世界的视觉知识，又可以接受动作并支持规划。

## 3. World model 能力阶梯

| 等级 | 能力 | 判定问题 |
|---|---|---|
| L0：视觉生成器 | 生成合理视频 | 视频看起来真实吗？ |
| L1：时间预测器 | 根据历史预测未来 | 后续运动与历史一致吗？ |
| L2：状态保持器 | 记住对象、位置和变化 | 离开后再返回，世界还一样吗？ |
| L3：动作条件模型 | 响应外部动作 | 不同动作是否产生正确的不同未来？ |
| L4：闭环模拟器 | 实时接受连续动作 | 智能体能否稳定交互数分钟？ |
| L5：决策世界模型 | 支持规划和策略学习 | 使用模型是否提高真实任务成功率？ |

一个模型可以在视觉上达到 L0/L1，却在决策上仍未达到 L3。

## 4. 五种常见技术形态

### Pixel world model

直接预测未来图像。容易被人检查，适合生成训练数据，但会把大量容量消耗在纹理和光照细节上。

### Latent dynamics model

在压缩状态中预测未来，只重建必要观测或直接预测价值。PlaNet [[2]](#ref-2)、Dreamer [[3]](#ref-3) 和许多机器人模型属于这一路线。

### Latent predictive representation

在语义 representation 中预测未来，不要求逐像素还原。优点是更聚焦抽象结构，难点是如何证明 latent 包含规划需要的全部信息。JEPA 属于这一路线的代表，但在本仓库中作为参考阅读收录，见 [JEPA 参考阅读](jepa.md)。

### Action-conditioned video model

使用大规模视频模型作为视觉先验，再用带动作的视频后训练。它兼顾视觉开放性与机器人动作，但动作数据分布和 embodiment 差异仍是瓶颈。

### Interactive neural simulator

逐帧或逐块生成环境，实时响应用户动作。它需要同时解决低延迟、空间记忆、状态持久性和错误恢复。

## 5. 代表案例应该怎样理解

### Sora [[5]](#ref-5) / Veo 等视频基础模型

重要性在于规模化生成训练可能带来隐式的 3D、运动、物理属性和视觉推理能力。它们为 world model 提供强大的视觉先验，但“涌现视觉能力”不自动等于“拥有可验证的动作动力学”。

### Genie 系列 [[6]](#ref-6)

代表“从被动互联网视频中学习潜在动作，再生成可交互环境”的路线。Genie 3 [[8]](#ref-8) 进一步强调实时导航、较长的一致性和 promptable world events。

### GWM-1 [[9]](#ref-9)

把可探索环境、实时角色和机器人动作条件视频作为不同 post-training 方向，强调相机、语音和机器人命令等动作接口。

### Cosmos 系列 [[10]](#ref-10)

面向 Physical AI，将数据治理、视频 tokenizer、生成、物理推理、动作预测和机器人策略放入同一平台。Cosmos 3 [[11]](#ref-11) 进一步尝试统一语言、视觉、声音和动作。

## 6. World model 的关键评测

漂亮 demo 不能回答以下问题：

### Action sensitivity

改变动作而保持其他条件不变，未来是否发生合理变化？

### Counterfactual consistency

对同一状态测试多种动作，模型是否产生互相一致、符合因果结构的分支？

### State persistence

对象被移动、打开、破坏或遮挡后，相关状态能否在长时间后保持？

### Spatial consistency

相机绕行、离开再返回时，几何布局是否保持一致？

### Calibration

面对不可预测或训练分布外情况，模型是否表达不确定性，而不是自信地产生错误未来？

### Planning utility

通过模型选择动作，是否比无模型策略、真实数据 baseline 或简单 simulator 获得更高任务成功率？

## 7. 当前最重要的开放问题

- 视频数据中的相关性是否足以学习因果干预？
- 2D 生成器能否形成稳定、可复用的 3D 空间记忆？
- 如何避免长 rollout 中的小错误积累成全局崩溃？
- 如何统一不同机器人、相机和控制频率下的动作空间？
- 模型应该生成像素，还是只预测对决策有用的 latent？
- 如何验证罕见危险事件，而不被逼真的常见场景平均分掩盖？
- 智能体是否会利用 learned simulator 的错误，在想象世界中获得虚假高奖励？

## 8. 一个谨慎的表述模板

讨论新模型时，可以使用以下表述：

> 该模型在大规模视频生成中表现出若干世界建模能力，包括 X 和 Y；但其动作条件、反事实准确性、长期状态持久性及闭环规划价值仍需通过 Z 类实验验证。

这种写法既不会低估生成预训练的潜力，也不会把视觉真实性误当成完整的物理或因果理解。

## 参考文献

<a id="ref-1"></a>[1] David Ha, and Jürgen Schmidhuber. [World Models](https://arxiv.org/abs/1803.10122). Advances in Neural Information Processing Systems 31, 2018.

<a id="ref-2"></a>[2] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). arXiv preprint, 2018.

<a id="ref-3"></a>[3] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). arXiv preprint, 2019.

<a id="ref-4"></a>[4] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). arXiv preprint, 2023.

<a id="ref-5"></a>[5] OpenAI. [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). Technical report, 2024.

<a id="ref-6"></a>[6] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). arXiv preprint, 2024.

<a id="ref-7"></a>[7] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). arXiv preprint, 2024.

<a id="ref-8"></a>[8] Google DeepMind. [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Project report, 2025.

<a id="ref-9"></a>[9] Runway. [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Project report, 2025.

<a id="ref-10"></a>[10] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-11"></a>[11] NVIDIA. [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). arXiv preprint, 2026.

<a id="ref-12"></a>[12] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). arXiv preprint, 2025.
