# 文本到视频

## 任务定义

Text-to-video（T2V）输入自然语言描述，输出符合语义、动作、场景、风格和镜头要求的视频。它是当前最受关注的视频生成任务，也是从“生成一段视频”走向“可创作系统”的核心入口。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要变化 |
|---|---|---|---|
| 模板与检索 | 视频检索、模板动画、procedural animation | 用文本匹配已有片段或参数化动画 | 生成性弱，可控但不开放 |
| 文本条件 GAN | TGAN / text-conditioned video GAN 变体 | 文本 embedding + 时空生成器 | 能生成短片，但质量和语义有限 |
| Transformer token | Phenaki、VideoPoet、MAGVIT 系列 | video tokenizer + masked / autoregressive generation | 支持变长、组合语义和多任务条件 |
| Diffusion | Make-A-Video、Imagen Video、ModelScope、AnimateDiff、Lumiere | 文本编码器 + temporal U-Net / DiT | 画质和 prompt following 快速提升 |
| 大规模基础模型 | Sora、Veo、Kling、Seedance、Cosmos | spacetime patch、长上下文、多模态训练 | 强化长视频、物理、镜头和音视频同步 |

## 技术演化逻辑

T2V 的难点不是“把文字变成几帧图像”，而是把语言中的动作、时间顺序、物体关系和镜头语言转成连续视觉事件。早期方法通常只能生成短、粗糙、语义较弱的视频。Diffusion 以后，模型可以复用图像生成的语义能力，再通过 temporal layer 或时空注意力学习运动。基础模型阶段则进一步把不同长度、分辨率和宽高比的视频统一成时空 token 或 patch。

## 最新趋势

- 从单镜头短片转向多镜头叙事、原生音视频和长时一致性。
- 从纯 prompt 转向 prompt + reference image + camera path + motion trajectory 的复合条件。
- 从封闭产品 demo 转向开放权重模型、低成本 LoRA、推理加速和移动端部署。
- 从视觉偏好评测转向文本遵循、物理一致性、角色一致性、来源标记和安全治理。

## 关键评测

- 文本实体是否出现，属性是否正确。
- 动作是否按时间顺序发生，而不是只在单帧中暗示。
- 镜头运动和主体运动是否可分离。
- 多对象关系是否稳定，尤其是左右、前后、接触、遮挡和数量。
- 长视频中角色、服装、场景状态是否保持。

## 开放问题

1. 语言模型能否可靠分解复杂动态 prompt？
2. 长视频是否需要显式脚本、分镜和记忆，而不是一次性采样？
3. 物理合理性来自数据规模、架构偏置还是外部模拟器约束？
4. 安全、版权、肖像和水印如何与生成质量一起设计？

## 推荐阅读

- Make-A-Video、Imagen Video：大规模 T2V diffusion 的早期代表。
- Phenaki、MAGVIT、VideoPoet：token / Transformer 路线。
- Lumiere：完整时间范围的 Space-Time U-Net。
- Sora technical report：大规模视频生成与 world simulator 讨论。
- 2024-2026 T2V survey：用于补充产品和开放模型谱系。
