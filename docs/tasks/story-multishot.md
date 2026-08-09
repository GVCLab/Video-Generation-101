# 故事与多镜头生成

## 任务定义

Story / multi-shot generation 输入剧本、故事、分镜或多段 prompt，输出由多个镜头组成的视频。它比单段 T2V 更接近影视创作，核心难点是叙事、角色、场景、镜头语言和状态的长程一致性。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统流程 | storyboard、animatic、template rendering | 人工分镜 + 渲染或剪辑 | 自动化弱 |
| 图像故事 | story visualization、Make-A-Story、Phenaki [@villegas2022phenaki] | 逐图生成 + visual memory / 变长视频 token | 视频运动不足或一致性有限 |
| 分步 pipeline | LLM 分解故事、生成关键帧、I2V 动画 [@blattmann2023stable] | 脚本到镜头计划 | 误差累积，风格漂移 |
| 多镜头 T2V | VideoGen-of-Thought、Text-to-Multi-Shot Video | 镜头级条件与全局 attention | 成本高、长度有限 |
| 记忆增强 | StoryMem、adaptive memory | 显式 visual memory 连接镜头 | 记忆选择和冲突管理 |
| 流式生成 | ShotStream | 自回归多镜头，交互式 storytelling | 实时性与质量权衡 |
| 数据与 benchmark | MuSS、ConStoryBoard、AnimeShooter | 电影级镜头标注与评测 | 版权、标注和客观指标困难 |

## 技术演化逻辑

单镜头 T2V 主要回答“这一段画面是什么”，早期变长视频生成如 Phenaki 已经开始处理长 prompt 和连续事件 [@villegas2022phenaki]。多镜头生成要回答“前后镜头如何构成一个故事”。因此模型需要比普通视频生成更多结构：角色设定、场景地图、镜头计划、故事进度、视觉记忆和状态更新。许多最新方法不再一次性生成整片，而是先规划，再逐镜头生成，并用 memory 或 reference 保持一致。

## 最新趋势

- 从单 prompt 变成 LLM-assisted script decomposition。
- 从逐镜头独立生成变成 memory-conditioned shot synthesis。
- 从固定长度短片变成 streaming multi-shot generation。
- 数据集开始标注 shot scale、camera angle、camera movement、story progress 和角色身份。

## 关键评测

- 角色身份是否跨镜头保持。
- 场景、服装、物体状态是否连贯。
- 镜头之间是否有叙事因果，而不是随机拼接。
- shot length、camera angle、transition 是否符合用户要求。
- 长视频中是否出现重复、遗忘或身份混淆。

## 开放问题

1. 多镜头生成应一次性全局 attention，还是逐镜头带记忆生成？
2. 视觉记忆应该保存参考图、latent token、scene graph 还是剧本状态？
3. 如何客观评估叙事质量和电影语言？
4. 长视频训练数据的版权与标注如何解决？

## 参考文献

- [@villegas2022phenaki] Phenaki：变长文本视频生成。
- [@blattmann2023stable] Stable Video Diffusion：多镜头 pipeline 中常用的 I2V 先验。
- [@openai2024sora] Sora technical report：长视频与复杂时空场景讨论。
- [VideoGen-of-Thought](https://arxiv.org/abs/2412.02259)：step-by-step multi-shot synthesis。
- [Text-to-Multi-Shot Video Generation](https://arxiv.org/html/2505.07652v1)：全帧 attention 的多镜头生成。
- [StoryMem](https://arxiv.org/html/2512.19539v1)：显式 visual memory。
- [ShotStream](https://arxiv.org/html/2603.25746v1)：流式多镜头生成。
- [MuSS](https://arxiv.org/html/2604.23789v1)：电影级多镜头数据与 benchmark。
- [Storyboard-Anchored Generation](https://arxiv.org/html/2512.12372v2)：结构化 storyboard 标注与生成。
