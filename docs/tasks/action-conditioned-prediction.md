# 动作条件预测

## 任务定义

Action-conditioned prediction 输入观测历史和动作，预测动作执行后的未来观测、状态或 latent。它是区分普通视频预测和 world model 的关键任务：模型必须回答“如果采取这个动作，会发生什么”。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要问题 |
|---|---|---|---|
| 控制与系统辨识 | dynamics model、Kalman filter、MPC、World Models [@ha2018worldmodels] | 显式状态与动作转移 | 开放视觉场景能力弱 |
| 机器人视频预测 | CDNA、DNA、STP [@finn2016unsupervised] | 动作条件像素变换 | 长程和多物体交互有限 |
| Model-based RL | PlaNet [@hafner2018learning]、Dreamer [@hafner2019dream]、DreamerV3 [@hafner2023mastering] | latent dynamics + imagination | 视觉真实度不是主要目标 |
| Game / embodied | Genie [@bruce2024genie]、GameNGen [@valevski2024diffusion]、GAIA-1 [@hu2023gaia] | latent action 或 action-conditioned generation | 动作语义和真实控制映射困难 |
| JEPA / latent | V-JEPA 2-AC [@assran2025vjepa2]、LeWorldModel [@maes2026leworldmodel] | 预测动作后的 future representation | latent 是否足够可规划 |
| World Action Model | WAM、GigaWorld-Policy、A2World | 联合视频与动作建模 | 跨机器人、跨视角、跨任务迁移 |
| 多智能体 | MultiWorld | 多主体、多视角动作条件视频 | 交互复杂度高 |

## 技术演化逻辑

动作条件预测最早来自控制：状态和动作决定下一状态。视觉世界里，难点在于状态不可直接观测，动作空间又因机器人、游戏、驾驶和相机而异。深度视频预测先尝试用动作驱动像素变换 [@finn2016unsupervised]；RL world model 转向 latent imagination [@ha2018worldmodels; @hafner2018learning; @hafner2019dream]；现代视频基础模型则把强视觉先验与少量动作数据结合，希望获得更开放的 action-conditioned dynamics [@bruce2024genie; @assran2025vjepa2; @nvidia2025cosmos]。

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

- [@finn2016unsupervised] CDNA / DNA / STP：动作条件视频预测起点。
- [@ha2018worldmodels] World Models：latent imagination 经典路线。
- [@hafner2018learning] PlaNet：latent dynamics planning。
- [@hafner2019dream] Dreamer：latent imagination control。
- [@hafner2023mastering] DreamerV3：跨领域 world model。
- [@hu2023gaia] GAIA-1：自动驾驶生成式 world model。
- [@bruce2024genie] Genie：latent action 与交互环境。
- [@valevski2024diffusion] GameNGen：diffusion game engine。
- [@assran2025vjepa2] V-JEPA 2 / 2-AC：latent predictive planning。
- [@maes2026leworldmodel] LeWorldModel：JEPA-style action-conditioned latent dynamics。
- [World Action Models are Zero-shot Policies](https://arxiv.org/html/2602.15922v1)：WAM 路线。
- [GigaWorld-Policy](https://arxiv.org/html/2603.17240v2)：视频生成辅助策略学习。
- [A2World](https://arxiv.org/html/2606.29501v1)：transferable action-conditioned dynamics priors。
- [MultiWorld](https://arxiv.org/html/2604.18564v1)：多智能体多视角 world model。
