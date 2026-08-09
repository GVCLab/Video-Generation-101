# 帧插值

## 任务定义

Frame interpolation 输入前后关键帧，生成中间帧。它常用于升帧、慢动作、视频修复、动画补帧和生成式关键帧过渡。与 video prediction 不同，它同时知道起点和终点。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统运动补偿 | block matching、optical flow [[1]](#ref-1), [[2]](#ref-2) | 估计双向运动并 warp | 遮挡、大运动、透明物体困难 |
| 深度 flow | Super SloMo、DAIN、RIFE | 学习光流、深度和遮挡 mask | 依赖运动估计质量 |
| Kernel / hybrid | SepConv、AdaCoF、softmax splatting | 学习局部卷积核或 splatting | 复杂变形和语义补全有限 |
| Transformer | IFRNet、VFIformer 等 | 全局匹配和注意力 | 成本较高 |
| 生成式插值 | diffusion / I2V adaptation [[3]](#ref-3), [[4]](#ref-4) | 用视频生成模型补全中间过程 | 可能偏离端点或生成不真实运动 |
| 最新方向 | auto-regressive DiT、generative VFI | 联合建模整段过渡 | 端点约束与创造性之间权衡 |

## 技术演化逻辑

传统 VFI 是低层视觉任务：估计运动、处理遮挡、融合像素，光流是其中最重要的基础之一 [[1]](#ref-1), [[2]](#ref-2)。深度学习以后，模型把 flow、depth、occlusion 和 refinement 网络联合训练。生成模型进入后，插值不再只是像素对齐，而是对“从 A 到 B 的合理动态路径”建模，尤其适合大运动、非刚体和关键帧动画 [[3]](#ref-3), [[4]](#ref-4)。

## 最新趋势

- 将 I2V diffusion 改造成大运动插值模型。
- 用 autoregressive diffusion transformer 联合建模长过渡。
- 从单一中间帧转向任意时间、多帧、可控速度曲线。
- 在动画、电影和生成式编辑中与 keyframe control 融合。

## 关键评测

- 中间帧是否严格连接两个端点。
- 遮挡区域是否合理出现或消失。
- 大运动是否自然，是否出现重影。
- 任意时间采样是否时间一致。
- 生成式插值是否引入不应该出现的新内容。

## 开放问题

1. 插值应更像物理运动估计，还是更像条件视频生成？
2. 大模型是否会牺牲端点忠实度来追求视觉合理性？
3. 如何评估动画场景中不存在唯一真实中间帧的问题？
4. 任意长度插值如何避免局部速度不均和语义漂移？
本页主要参考工作：AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation [[5]](#ref-5)、Adapting Image-to-Video Diffusion Models for Large-Motion Frame Interpolation [[6]](#ref-6)、Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers [[7]](#ref-7)。

## 参考文献

<a id="ref-1"></a>[1] [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). Bruce D. Lucas and Takeo Kanade. Proceedings of IJCAI. 1981.

<a id="ref-2"></a>[2] [Determining optical flow](https://doi.org/10.1016/0004-3702(81)90024-2). Berthold K. P. Horn and Brian G. Schunck. Artificial Intelligence. 1981.

<a id="ref-3"></a>[3] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. arXiv preprint. 2022.

<a id="ref-4"></a>[4] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-5"></a>[5] [AceVFI: A Comprehensive Survey of Advances in Video Frame Interpolation](https://arxiv.org/abs/2506.01061). Dahyeon Kye, Changhyun Roh, Sukhun Ko, Chanho Eom, Jihyong Oh. arXiv preprint. 2025.

<a id="ref-6"></a>[6] [Adapting Image-to-Video Diffusion Models for Large-Motion Frame Interpolation](https://arxiv.org/abs/2412.17042). Luoxu Jin and Hiroshi Watanabe. arXiv preprint. 2024.

<a id="ref-7"></a>[7] [Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers](https://arxiv.org/abs/2601.14959). Xinyu Peng, Han Li, Yuyang Huang, Ziyang Zheng, Yaoming Wang, Xin Chen, et al. arXiv preprint. 2026.
