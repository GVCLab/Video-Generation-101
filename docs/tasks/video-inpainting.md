# 视频修复与补全

## 任务定义

Video inpainting 输入视频和 mask，补全被遮挡、删除或缺失的区域。它覆盖对象移除、遮挡恢复、局部重绘、视频 outpainting 和长视频修复。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统补全 | exemplar-based inpainting、patch match | 从邻域或其他帧复制纹理 | 语义生成弱 |
| Flow-guided | optical flow propagation、FGVC | 传播可见区域再补洞 | 依赖 flow 与可见参考 |
| 深度修复 | 3D CNN、attention、transformer inpainting | 学习时空上下文 | 大 mask 和长视频困难 |
| Diffusion inpainting | text-guided video inpainting、DiffuEraser | 强生成先验 + mask 条件 | 计算贵、时间一致性问题 |
| DiT inpainting | VideoPainter、DiTPainter、EraserDiT | diffusion transformer 与长程注意力 | 长视频 memory 与效率 |
| 统一长视频 | long video inpainting/outpainting | overlapping windows、memory、global consistency | 边界融合和状态漂移 |

## 技术演化逻辑

传统 inpainting 假设缺失区域可以从附近复制；深度模型开始学习语义补全；diffusion 则把“补洞”变成条件生成。视频比图像更难，因为补全区域必须在时间上稳定，还要处理被删除对象后的背景、光照、遮挡和运动连续性。

## 最新趋势

- 使用 DiT 获得更长程时空一致性。
- 通过 optical flow 或 feature propagation 降低闪烁。
- 支持 text-guided object removal、局部编辑和任意长度视频。
- 将 inpainting 与 outpainting、V2V editing、multi-shot generation 统一。

## 关键评测

- mask 内视觉质量是否自然。
- mask 外内容是否保持。
- 补全背景是否跨帧一致。
- 大面积 mask、长期遮挡和新出现区域是否可靠。
- 文本指令是否只影响指定区域。

## 开放问题

1. 长视频 inpainting 是否需要全局记忆或场景表示？
2. 扩散模型如何避免每个窗口生成不同背景？
3. 对象删除后，被遮挡过的区域是否应该遵循物理和几何？
4. 如何评估没有 ground truth 的创作型补全？

## 推荐阅读

- PatchMatch / exemplar inpainting：传统补全思想。
- Flow-guided video completion：显式运动传播。
- VideoPainter、DiTPainter、EraserDiT：DiT video inpainting。
- Unified long video inpainting/outpainting：长视频统一修复趋势。
