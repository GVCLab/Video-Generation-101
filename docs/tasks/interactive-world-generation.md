# 交互式世界生成

## 任务定义

Interactive world generation 输入初始世界、文本或视频上下文，并在生成过程中实时接受用户或智能体动作，持续输出可交互环境。它是视频生成、world model、游戏引擎和机器人模拟的交叉点。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统引擎 | 物理引擎、游戏引擎、仿真器 | 显式状态、规则和渲染 | 制作成本高，开放世界弱 |
| 学习模拟器 | World Models [[1]](#ref-1)、PlaNet [[2]](#ref-2)、Dreamer [[3]](#ref-3) | 学习状态转移供规划 | 视觉真实感有限 |
| 被动视频生成 | T2V / I2V foundation model [[4]](#ref-4), [[5]](#ref-5) | 强视觉先验 | 无实时动作闭环 |
| latent action | Genie [[6]](#ref-6) | 从无动作标签视频学习可控 action space | 动作含义隐式 |
| 神经游戏引擎 | GameNGen [[7]](#ref-7) | 生成模型实时模拟游戏帧 | 场景范围受限 |
| 世界基础模型 | Genie 3 [[8]](#ref-8)、GWM-1 [[9]](#ref-9)、Cosmos [[10]](#ref-10), [[11]](#ref-11) | promptable world events、动作接口、Physical AI 平台 | 状态持久性和可靠性仍待验证 |
| 多智能体世界 | MultiWorld 类方向 | 多主体、多视角、动作条件 rollout | 复杂交互与评测困难 |

## 技术演化逻辑

传统游戏引擎强在规则明确和状态可靠，弱在开放内容制作成本。早期 learned world model 通过 latent imagination 支持控制和规划 [[1]](#ref-1), [[3]](#ref-3)。视频生成模型强在开放视觉先验，弱在可控状态和因果 [[4]](#ref-4)。交互式世界生成试图合并两者：用神经模型生成开放视觉世界，同时像引擎一样响应动作、保持状态、支持探索和规划 [[6]](#ref-6), [[7]](#ref-7), [[8]](#ref-8)。

## 最新趋势

- 从单次视频生成转向实时导航和连续动作输入。
- 从视觉一致性转向空间记忆、对象状态、可回访性和 promptable events。
- 从游戏和驾驶扩展到机器人、Physical AI 和多智能体场景。
- 从纯像素输出转向状态、动作、声音、语言和策略的统一模型。

## 关键评测

- 延迟是否足够低，是否能实时交互。
- 玩家离开再返回时，世界状态是否保持。
- 动作是否有稳定、可学习、可规划的后果。
- 模型是否能持续运行数分钟而不崩溃。
- 智能体在模型中规划是否能迁移到真实或标准环境。

## 开放问题

1. 神经世界模型能否达到传统引擎的状态可靠性？
2. 如何把隐式视频 latent 转成可编辑、可验证的世界状态？
3. 长时间交互中的 memory 应该如何写入和读取？
4. 模型如何表达不确定性，避免给 planner 错误自信？
5. 开放世界生成和安全约束如何同时成立？

## 参考文献

<a id="ref-1"></a>[1] David Ha, and Jürgen Schmidhuber. [World Models](https://arxiv.org/abs/1803.10122). Advances in Neural Information Processing Systems 31, 2018.

<a id="ref-2"></a>[2] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). arXiv preprint, 2018.

<a id="ref-3"></a>[3] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). arXiv preprint, 2019.

<a id="ref-4"></a>[4] OpenAI. [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). Technical report, 2024.

<a id="ref-5"></a>[5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). arXiv preprint, 2023.

<a id="ref-6"></a>[6] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). arXiv preprint, 2024.

<a id="ref-7"></a>[7] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). arXiv preprint, 2024.

<a id="ref-8"></a>[8] Google DeepMind. [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Project report, 2025.

<a id="ref-9"></a>[9] Runway. [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Project report, 2025.

<a id="ref-10"></a>[10] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-11"></a>[11] NVIDIA. [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). arXiv preprint, 2026.

<a id="ref-12"></a>[12] [MultiWorld](https://arxiv.org/html/2604.18564v1). 多智能体多视角交互世界.

<a id="ref-13"></a>[13] [From World Models to World Action Models](https://arxiv.org/html/2607.00836v1). world action model 教程.
