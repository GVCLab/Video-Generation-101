# 交互式世界生成

## 任务定义

Interactive world generation 输入初始世界、文本或视频上下文，并在生成过程中实时接受用户或智能体动作，持续输出可交互环境。它是视频生成、world model、游戏引擎和机器人模拟的交叉点。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统引擎 | 物理引擎、游戏引擎、仿真器 | 显式状态、规则和渲染 | 制作成本高，开放世界弱 |
| 学习模拟器 | World Models [@ha2018worldmodels]、PlaNet [@hafner2018learning]、Dreamer [@hafner2019dream] | 学习状态转移供规划 | 视觉真实感有限 |
| 被动视频生成 | T2V / I2V foundation model [@openai2024sora; @blattmann2023stable] | 强视觉先验 | 无实时动作闭环 |
| latent action | Genie [@bruce2024genie] | 从无动作标签视频学习可控 action space | 动作含义隐式 |
| 神经游戏引擎 | GameNGen [@valevski2024diffusion] | 生成模型实时模拟游戏帧 | 场景范围受限 |
| 世界基础模型 | Genie 3 [@deepmind2025genie3]、GWM-1 [@runway2025gwm1]、Cosmos [@nvidia2025cosmos; @nvidia2026cosmos3] | promptable world events、动作接口、Physical AI 平台 | 状态持久性和可靠性仍待验证 |
| 多智能体世界 | MultiWorld 类方向 | 多主体、多视角、动作条件 rollout | 复杂交互与评测困难 |

## 技术演化逻辑

传统游戏引擎强在规则明确和状态可靠，弱在开放内容制作成本。早期 learned world model 通过 latent imagination 支持控制和规划 [@ha2018worldmodels; @hafner2019dream]。视频生成模型强在开放视觉先验，弱在可控状态和因果 [@openai2024sora]。交互式世界生成试图合并两者：用神经模型生成开放视觉世界，同时像引擎一样响应动作、保持状态、支持探索和规划 [@bruce2024genie; @valevski2024diffusion; @deepmind2025genie3]。

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

- [@ha2018worldmodels] World Models：latent imagination 根源。
- [@hafner2018learning] PlaNet：planning from pixels。
- [@hafner2019dream] Dreamer：latent imagination control。
- [@openai2024sora] Sora technical report：视频生成作为 world simulator 的讨论。
- [@blattmann2023stable] Stable Video Diffusion：I2V foundation prior。
- [@bruce2024genie] Genie：从互联网视频学习 latent action。
- [@valevski2024diffusion] GameNGen：diffusion 作为实时游戏引擎。
- [@deepmind2025genie3] Genie 3：实时导航和 promptable world events。
- [@runway2025gwm1] GWM-1：general world model 产品研究线。
- [@nvidia2025cosmos] Cosmos：Physical AI world foundation model。
- [@nvidia2026cosmos3] Cosmos 3：omnimodal world model。
- [MultiWorld](https://arxiv.org/html/2604.18564v1)：多智能体多视角交互世界。
- [From World Models to World Action Models](https://arxiv.org/html/2607.00836v1)：world action model 教程。
