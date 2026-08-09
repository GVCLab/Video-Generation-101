# 动作条件预测

## 任务定义

Action-conditioned prediction 输入观测历史和动作，预测动作执行后的未来观测、状态或 latent。它是区分普通视频预测和 world model 的关键任务：模型必须回答“如果采取这个动作，会发生什么”。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要问题 |
|---|---|---|---|
| 控制与系统辨识 | dynamics model、Kalman filter、MPC、World Models [[1]](#ref-1) | 显式状态与动作转移 | 开放视觉场景能力弱 |
| 机器人视频预测 | CDNA、DNA、STP [[2]](#ref-2) | 动作条件像素变换 | 长程和多物体交互有限 |
| Model-based RL | PlaNet [[3]](#ref-3)、Dreamer [[4]](#ref-4)、DreamerV3 [[5]](#ref-5) | latent dynamics + imagination | 视觉真实度不是主要目标 |
| Game / embodied | Genie [[6]](#ref-6)、GameNGen [[7]](#ref-7)、GAIA-1 [[8]](#ref-8) | latent action 或 action-conditioned generation | 动作语义和真实控制映射困难 |
| JEPA / latent | V-JEPA 2-AC [[9]](#ref-9)、LeWorldModel [[10]](#ref-10) | 预测动作后的 future representation | latent 是否足够可规划 |
| World Action Model | WAM、GigaWorld-Policy、A2World | 联合视频与动作建模 | 跨机器人、跨视角、跨任务迁移 |
| 多智能体 | MultiWorld | 多主体、多视角动作条件视频 | 交互复杂度高 |

## 技术演化逻辑

动作条件预测最早来自控制：状态和动作决定下一状态。视觉世界里，难点在于状态不可直接观测，动作空间又因机器人、游戏、驾驶和相机而异。深度视频预测先尝试用动作驱动像素变换 [[2]](#ref-2)；RL world model 转向 latent imagination [[1]](#ref-1), [[3]](#ref-3), [[4]](#ref-4)；现代视频基础模型则把强视觉先验与少量动作数据结合，希望获得更开放的 action-conditioned dynamics [[6]](#ref-6), [[9]](#ref-9), [[11]](#ref-11)。

## 最新趋势

- 用大规模视频预训练作为视觉先验，再用机器人或交互数据后训练动作模块。
- 从单一 embodiment 转向跨机器人、跨视角和跨任务的 world action model。
- 同时预测 future video 和 future action，让视觉 dynamics 约束 policy。
- 从单智能体转向多智能体、多视角、多对象交互建模。

## 关键评测

- 同一状态下不同动作是否产生不同且正确的未来。
- 预测是否能帮助 MPC 或 policy 选择动作。
- 模型是否只学到数据相关性，还是能处理干预。
- rollout 误差是否会导致 planner 利用模型漏洞。
- 在真实环境中是否提升任务成功率。

## 开放问题

1. 被动互联网视频能提供多少动作因果知识？
2. 不同机器人动作空间如何统一？
3. future video 与 future latent 哪个更适合规划？
4. action-conditioned model 是否需要显式 3D 或物理变量？

## 参考文献

<a id="ref-1"></a>[1] David Ha, and Jürgen Schmidhuber. [World Models](https://arxiv.org/abs/1803.10122). Advances in Neural Information Processing Systems 31, 2018.

<a id="ref-2"></a>[2] Chelsea Finn, Ian Goodfellow, and Sergey Levine. [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). arXiv preprint, 2016.

<a id="ref-3"></a>[3] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). arXiv preprint, 2018.

<a id="ref-4"></a>[4] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). arXiv preprint, 2019.

<a id="ref-5"></a>[5] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). arXiv preprint, 2023.

<a id="ref-6"></a>[6] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). arXiv preprint, 2024.

<a id="ref-7"></a>[7] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). arXiv preprint, 2024.

<a id="ref-8"></a>[8] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, et al. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080). arXiv preprint, 2023.

<a id="ref-9"></a>[9] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). arXiv preprint, 2025.

<a id="ref-10"></a>[10] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312). arXiv preprint, 2026.

<a id="ref-11"></a>[11] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-12"></a>[12] [World Action Models are Zero-shot Policies](https://arxiv.org/html/2602.15922v1). WAM 路线.

<a id="ref-13"></a>[13] [GigaWorld-Policy](https://arxiv.org/html/2603.17240v2). 视频生成辅助策略学习.

<a id="ref-14"></a>[14] [A2World](https://arxiv.org/html/2606.29501v1). transferable action-conditioned dynamics priors.

<a id="ref-15"></a>[15] [MultiWorld](https://arxiv.org/html/2604.18564v1). 多智能体多视角 world model.
