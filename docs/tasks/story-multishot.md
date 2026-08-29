# 故事与多镜头生成

## 任务定义

Story / multi-shot generation 输入剧本、故事、分镜或多段 prompt，输出由多个镜头组成的视频。它比单段 T2V 更接近影视创作，核心难点是叙事、角色、场景、镜头语言和状态的长程一致性。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统流程 | storyboard、animatic、template rendering | 人工分镜 + 渲染或剪辑 | 自动化弱 |
| 图像故事 | story visualization、Make-A-Story、Phenaki [[1]](#ref-1) | 逐图生成 + visual memory / 变长视频 token | 视频运动不足或一致性有限 |
| 分步 pipeline | LLM 分解故事、生成关键帧、I2V 动画 [[2]](#ref-2) | 脚本到镜头计划 | 误差累积，风格漂移 |
| 多镜头 T2V | VideoGen-of-Thought [[4]](#ref-4)、Text-to-Multi-Shot Video | 镜头级条件与全局 attention | 成本高、长度有限 |
| 记忆增强 | StoryMem [[6]](#ref-6)、adaptive memory | 显式 visual memory 连接镜头 | 记忆选择和冲突管理 |
| 流式生成 | ShotStream [[7]](#ref-7) | 自回归多镜头，交互式 storytelling | 实时性与质量权衡 |
| 数据与 benchmark | MuSS [[8]](#ref-8)、ConStoryBoard、AnimeShooter | 电影级镜头标注与评测 | 版权、标注和客观指标困难 |

## 技术演化逻辑

单镜头 T2V 主要回答“这一段画面是什么”，早期变长视频生成如 Phenaki 已经开始处理长 prompt 和连续事件 [[1]](#ref-1)。多镜头生成要回答“前后镜头如何构成一个故事”。因此模型需要比普通视频生成更多结构：角色设定、场景地图、镜头计划、故事进度、视觉记忆和状态更新。许多最新方法不再一次性生成整片，而是先规划，再逐镜头生成，并用 memory 或 reference 保持一致。

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
本页主要参考工作：Video Generation Models as World Simulators [[3]](#ref-3)、ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models [[5]](#ref-5)、STAGE: Storyboard-Anchored Generation for Cinematic Multi-shot Narrative [[9]](#ref-9)。

## 参考文献

<a id="ref-1"></a>[1] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://arxiv.org/abs/2210.02399). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.

<a id="ref-2"></a>[2] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-3"></a>[3] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-4"></a>[4] [VideoGen-of-Thought: Step-by-step generating multi-shot video with minimal manual intervention](https://arxiv.org/abs/2412.02259). Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, et al. arXiv preprint. 2024.

<a id="ref-5"></a>[5] [ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](https://arxiv.org/abs/2505.07652). Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M. Rehg, Tobias Hinz. arXiv preprint. 2025.

<a id="ref-6"></a>[6] [StoryMem: Multi-shot Long Video Storytelling with Memory](https://arxiv.org/abs/2512.19539). Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, et al. arXiv preprint. 2025.

<a id="ref-7"></a>[7] [ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling](https://arxiv.org/abs/2603.25746). Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, et al. arXiv preprint. 2026.

<a id="ref-8"></a>[8] [MuSS: A Large-Scale Dataset and Cinematic Narrative Benchmark for Multi-Shot Subject-to-Video Generation](https://arxiv.org/abs/2604.23789). Haojie Zhang, Di Wu, Bingyan Liu, Linjie Zhong, Yuancheng Wei, Xingsong Ye, et al. arXiv preprint. 2026.

<a id="ref-9"></a>[9] [STAGE: Storyboard-Anchored Generation for Cinematic Multi-shot Narrative](https://arxiv.org/abs/2512.12372). Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, Boxin Shi. arXiv preprint. 2025.
