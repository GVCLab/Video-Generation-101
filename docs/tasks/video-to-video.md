# 视频到视频

## 任务定义

Video-to-video（V2V）输入已有视频和编辑条件，输出结构相关但内容变化的视频。条件可以是文本指令、风格、参考图、mask、深度、姿态、光流、相机或多轮编辑历史。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要挑战 |
|---|---|---|---|
| 传统视频处理 | color transfer、style transfer、optical flow propagation [[1]](#ref-1), [[2]](#ref-2) | 单帧处理 + 时间平滑 | 语义编辑弱 |
| 条件翻译 | vid2vid、pix2pixHD video variants | semantic map / pose / edge 到视频 | 需要结构标注 |
| 扩散编辑 | Tune-A-Video、TokenFlow、FateZero、AnimateDiff [[3]](#ref-3) | inversion、attention control、one-shot tuning | 保持运动和身份困难 |
| I2V-based editing | first-frame guided editing、flow-driven I2V [[4]](#ref-4) | 用首帧编辑结果驱动整段视频 | 复杂局部编辑易漂移 |
| DiT / foundation editing | TV-LiVE [[7]](#ref-7)、V2Edit [[8]](#ref-8)、Ditto [[9]](#ref-9) 类数据扩展 | training-free 或大规模 instruction video editing | 多轮一致性和高质量数据 |
| 多轮记忆 | memory-augmented V2V | 保存前几轮编辑状态 | 避免已编辑区域被覆盖 |

## 技术演化逻辑

V2V 最初是视频风格化和结构翻译问题，通常依赖光流传播或结构条件来维持时间一致性 [[1]](#ref-1), [[2]](#ref-2)。Diffusion 出现后，它变成“如何在强生成先验下不破坏原视频” [[5]](#ref-5), [[3]](#ref-3)。这使得 inversion、attention sharing、flow propagation、mask control 和 memory 成为核心技术。现代方法越来越像视频版 Photoshop：用户希望局部、可逆、多轮、可追踪。

## 最新趋势

- 从单次编辑转向 instruction-based multi-turn editing。
- 从 one-shot tuning 转向 training-free DiT editing 或大规模合成编辑数据。
- 用 flow、depth、mask 和 first-frame condition 保持结构。
- 从视频编辑扩展到 3D scene editing、多视角和 4D video-to-video translation。

## 关键评测

- 编辑目标是否完成。
- 未编辑区域是否保持。
- 时间一致性是否稳定。
- 人物身份、物体纹理和运动轨迹是否漂移。
- 多轮编辑是否遗忘之前改动。

## 开放问题

1. 真实视频 inversion 是否仍是必要步骤？
2. 如何构造高质量 instruction-video editing 数据？
3. 多轮编辑中的“记忆”应保存像素、latent、mask 还是编辑图层？
4. V2V 与 inpainting、I2V、3D editing 的边界如何统一？
本页主要参考工作：Consistent Video Editing as Flow-Driven Image-to-Video Generation [[6]](#ref-6)、Memory-V2V: Augmenting Video-to-Video Diffusion Models with Memory [[10]](#ref-10)。

## 参考文献

<a id="ref-1"></a>[1] [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Bruce D. Lucas and Takeo Kanade. Proceedings of IJCAI. 1981.

<a id="ref-2"></a>[2] [Determining optical flow](https://doi.org/10.1016/0004-3702(81)90024-2). Berthold K. P. Horn and Brian G. Schunck. Artificial Intelligence. 1981.

<a id="ref-3"></a>[3] [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. arXiv preprint. 2023.

<a id="ref-4"></a>[4] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-5"></a>[5] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. arXiv preprint. 2022.

<a id="ref-6"></a>[6] [Consistent Video Editing as Flow-Driven Image-to-Video Generation](https://arxiv.org/abs/2506.07713). Ge Wang, Songlin Fan, Hangxu Liu, Quanjian Song, Hewei Wang, Jinfeng Xu. arXiv preprint. 2025.

<a id="ref-7"></a>[7] [TV-LiVE: Training-Free, Text-Guided Video Editing via Layer Informed Vitality Exploitation](https://arxiv.org/abs/2506.07205). Min-Jung Kim, Dongjin Kim, Seokju Yun, Jaegul Choo. arXiv preprint. 2025.

<a id="ref-8"></a>[8] [V2Edit: Versatile Video Diffusion Editor for Videos and 3D Scenes](https://arxiv.org/abs/2503.10634). Yanming Zhang, Jun-Kun Chen, Jipeng Lyu, Yu-Xiong Wang. arXiv preprint. 2025.

<a id="ref-9"></a>[9] [Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](https://arxiv.org/abs/2510.15742). Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, et al. arXiv preprint. 2025.

<a id="ref-10"></a>[10] [Memory-V2V: Augmenting Video-to-Video Diffusion Models with Memory](https://arxiv.org/abs/2601.16296). Dohun Lee, Chun-Hao Paul Huang, Xuelin Chen, Jong Chul Ye, Duygu Ceylan, Hyeonho Jeong. arXiv preprint. 2026.
