# 帧插值

## 任务定义

Frame interpolation 输入前后关键帧，生成中间帧。它常用于升帧、慢动作、视频修复、动画补帧和生成式关键帧过渡。与 video prediction 不同，它同时知道起点和终点。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统运动补偿 | block matching、optical flow | 估计双向运动并 warp | 遮挡、大运动、透明物体困难 |
| 深度 flow | Super SloMo、DAIN、RIFE | 学习光流、深度和遮挡 mask | 依赖运动估计质量 |
| Kernel / hybrid | SepConv、AdaCoF、softmax splatting | 学习局部卷积核或 splatting | 复杂变形和语义补全有限 |
| Transformer | IFRNet、VFIformer 等 | 全局匹配和注意力 | 成本较高 |
| 生成式插值 | diffusion / I2V adaptation | 用视频生成模型补全中间过程 | 可能偏离端点或生成不真实运动 |
| 最新方向 | auto-regressive DiT、generative VFI | 联合建模整段过渡 | 端点约束与创造性之间权衡 |

## 技术演化逻辑

传统 VFI 是低层视觉任务：估计运动、处理遮挡、融合像素。深度学习以后，模型把 flow、depth、occlusion 和 refinement 网络联合训练。生成模型进入后，插值不再只是像素对齐，而是对“从 A 到 B 的合理动态路径”建模，尤其适合大运动、非刚体和关键帧动画。

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

## 推荐阅读

- Super SloMo、DAIN、RIFE：深度 VFI 代表。
- 2025 VFI survey：从 motion compensation 到 diffusion 的系统综述。
- Adapting I2V diffusion for large-motion interpolation：生成式 VFI 方向。
- Auto-regressive diffusion transformer for VFI：长程插值趋势。
