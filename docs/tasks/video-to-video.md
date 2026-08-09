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
| DiT / foundation editing | TV-LiVE、V2Edit、Ditto 类数据扩展 | training-free 或大规模 instruction video editing | 多轮一致性和高质量数据 |
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

## 参考文献

<a id="ref-1"></a>[1] Bruce D. Lucas, and Takeo Kanade. [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI), 1981.

<a id="ref-2"></a>[2] Berthold K.P. Horn, and Brian G. Schunck. [Determining optical flow](https://doi.org/10.1016/0004-3702(81)90024-2). Artificial Intelligence, 1981.

<a id="ref-3"></a>[3] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). arXiv preprint, 2023.

<a id="ref-4"></a>[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). arXiv preprint, 2023.

<a id="ref-5"></a>[5] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. [Video Diffusion Models](https://arxiv.org/abs/2204.03458). arXiv preprint, 2022.

<a id="ref-6"></a>[6] [Consistent Video Editing as Flow-Driven Image-to-Video Generation](https://arxiv.org/html/2506.07713v1). flow + I2V editing.

<a id="ref-7"></a>[7] [TV-LiVE](https://arxiv.org/html/2506.07205v1). training-free DiT video editing.

<a id="ref-8"></a>[8] [V2Edit](https://arxiv.org/html/2503.10634v1). instruction-guided video and 3D scene editing.

<a id="ref-9"></a>[9] [Ditto](https://arxiv.org/html/2510.15742v1). instruction-based video editing 数据扩展.

<a id="ref-10"></a>[10] [Memory-Augmented Video-to-Video Diffusion](https://arxiv.org/html/2601.16296v2). 多轮编辑记忆.
