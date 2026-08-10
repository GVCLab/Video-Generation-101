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

## 6. Memory：Persistent World Modeling

对长时段生成而言，记忆不是简单地增加上下文长度，而是让模型持续维护一个可写入、可检索、可更新的世界状态。相比

$$
\text{Observation} + \text{Action} \rightarrow \text{Future}
$$

更完整的形式是：

$$
M_{t+1}=\operatorname{Update}(M_t,o_t,a_t),\qquad
\hat{o}_{t+1}=G(o_t,a_t,\operatorname{Retrieve}(M_t,q_t))
$$

其中 $M_t$ 不应被理解成单一的历史帧缓存，而应包含对未来预测有用的空间、对象、事件和动态状态。一个完整的 Memory 系统至少需要处理：

```text
Write → Store → Retrieve → Update → Consolidate / Forget
```

### Working memory：短期时序记忆

最近帧、滑动窗口、KV cache 和压缩上下文负责局部运动、姿态、外观及帧间连续性。它们能缓解上下文预算随视频长度增长的问题，但“压缩了历史”并不等于“形成了持久的世界状态”：过早压缩可能丢失身份、位置或事件因果关系。

### Episodic memory：经历与事件记忆

模型可以把关键帧、片段或 latent 作为可检索的经历，在视角回访或生成需要时取回。检索键可以来自视觉相似度、相机位姿、时间戳、文本描述或当前查询。与只保留最近窗口相比，这一路线允许模型回忆很久以前发生过的内容，但也引入了检索误差和注意力分散问题。

### Recurrent / compressed memory：压缩状态记忆

另一条路线把长历史递归地写入固定大小的 latent 或 state：

$$
h_t=f(h_{t-1},x_t)
$$

它的存储成本不随视频长度线性增长，适合长 rollout；代价是状态必须在有限容量内保留真正影响未来的变量。较合理的架构通常是“局部 attention + 持久 recurrent state”，分别承担细节连续性和全局身份、场景与动态。

### Spatial memory：空间可寻址的世界记忆

World model 需要记住的不只是“以前看见过什么”，还包括“什么东西在世界的什么位置”。3D point map、surfel、历史视角和 latent 3D cache 都可以作为空间地址，使相机离开场景后再返回时恢复几何布局、遮挡关系和外观。探索阶段可更多依赖时间记忆，回访阶段则应路由到空间记忆。

### Entity memory：对象与身份记忆

人物、物体和场景可以拥有独立的可寻址 slot，分别保存身份、外观、位置、关系和当前状态。例如，角色换了视角或经历多个 shot 后，模型仍应知道“谁是谁”；道具被拿起、移动或损坏后，状态也应写回相应对象，而不是只留在某一帧的纹理中。

### Dynamic / event memory：状态、事件与因果记忆

静态背景可以被归档，动态对象则需要在离开视野后继续被跟踪或预测。因此，Memory 应保存速度、方向、潜在状态和可能的未观测轨迹，而不只是 appearance。更进一步，事件记忆应记录“手推杯子 → 杯子掉落 → 杯子破碎”这类状态转移，使模型在重新看到杯子时仍能保持 `broken(cup)=True`。

这也说明 Memory 与 dynamics 并非两个完全独立的模块：

$$
M_{t+k}=F(M_t,a_{t:t+k})
$$

对象离开视野期间，记忆本身也应按照世界动力学演化。

### 研究判断

现有工作已经较好地覆盖了短期连续性、历史检索和空间回访，但以下问题仍适合作为 World Model 的独立研究主线：

- **Unified World Memory**：统一表示几何、实体、动态和事件，而不是为每种信息维护互不沟通的缓存。
- **Memory consolidation**：将 raw frames 逐步压缩为 episodes、objects、events 和可复用的 world state。
- **Learned forgetting**：长期保留重要事件，合并重复背景，覆盖过时状态，删除与当前任务无关的细节。
- **Addressability**：区分“存得下”“没有丢”和“之后找得到”。长期上下文中的位置编码、检索键和状态漂移都可能让已保存的记忆失效。

因此，本章将 Memory 定义为：

> Maintaining a persistent, updateable, and addressable internal state of the world during long-horizon generation and interaction.

它是连接视频生成、空间建模、动力学预测和规划的核心能力，而不是单纯的长视频工程优化。

## 7. World model 的关键评测

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

## 8. 当前最重要的开放问题

- 视频数据中的相关性是否足以学习因果干预？
- 2D 生成器能否形成稳定、可复用的 3D 空间记忆？
- 如何避免长 rollout 中的小错误积累成全局崩溃？
- 如何统一不同机器人、相机和控制频率下的动作空间？
- 模型应该生成像素，还是只预测对决策有用的 latent？
- 如何验证罕见危险事件，而不被逼真的常见场景平均分掩盖？
- 智能体是否会利用 learned simulator 的错误，在想象世界中获得虚假高奖励？

## 9. 一个谨慎的表述模板

讨论新模型时，可以使用以下表述：

> 该模型在大规模视频生成中表现出若干世界建模能力，包括 X 和 Y；但其动作条件、反事实准确性、长期状态持久性及闭环规划价值仍需通过 Z 类实验验证。

这种写法既不会低估生成预训练的潜力，也不会把视觉真实性误当成完整的物理或因果理解。
本页主要参考工作：World Models [[1]](#ref-1)、Mastering Diverse Domains through World Models [[4]](#ref-4)、Diffusion Models Are Real-Time Game Engines [[7]](#ref-7)、V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning [[12]](#ref-12)。

## 参考文献

<a id="ref-1"></a>[1] [World Models](https://arxiv.org/abs/1803.10122). David Ha and Jürgen Schmidhuber. Advances in Neural Information Processing Systems 31. 2018.

<a id="ref-2"></a>[2] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. arXiv preprint. 2018.

<a id="ref-3"></a>[3] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. arXiv preprint. 2019.

<a id="ref-4"></a>[4] [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. arXiv preprint. 2023.

<a id="ref-5"></a>[5] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-6"></a>[6] [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. arXiv preprint. 2024.

<a id="ref-7"></a>[7] [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter. arXiv preprint. 2024.

<a id="ref-8"></a>[8] [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Google DeepMind. Project report. 2025.

<a id="ref-9"></a>[9] [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Runway. Project report. 2025.

<a id="ref-10"></a>[10] [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. arXiv preprint. 2025.

<a id="ref-11"></a>[11] [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). NVIDIA. arXiv preprint. 2026.

<a id="ref-12"></a>[12] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. arXiv preprint. 2025.
