# 图像到视频

## 任务定义

Image-to-video（I2V）输入一张或多张参考图，生成保持主体、身份、布局和风格的视频。它常用于角色动画、产品展示、照片动效、首帧驱动创作和参考图一致性控制。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要难点 |
|---|---|---|---|
| 传统动画 | morphing、optical flow、image warping | 手工或估计运动场 | 大运动和新出现区域困难 |
| 运动迁移 | pose / keypoint / flow conditioned animation | 姿态、关键点、轨迹驱动 | 依赖特定类别或模板 |
| GAN / VAE | image animation GAN、motion-content disentanglement | 静态内容 + 动态 latent | 身份保持和复杂运动不足 |
| Diffusion I2V | Stable Video Diffusion、DynamiCrafter、I2VGen-XL | 首帧条件 + temporal diffusion | 运动可控性与细节保持冲突 |
| Foundation I2V | 多参考、轨迹、相机和音频条件模型 | 统一视觉条件接口 | 多条件冲突、长时漂移 |
| 轻量与实时 | MobileI2V、蒸馏/剪枝/少步采样 | 小模型和低延迟生成 | 质量、速度和分辨率三角权衡 |

## 技术演化逻辑

I2V 的核心矛盾是“保留静态信息”和“创造动态变化”。保留太强，视频只是轻微抖动；运动太强，主体身份、纹理和布局会漂移。早期 warping 方法可控但不擅长 hallucinate 新区域；diffusion 模型能补全不可见区域，但容易改变原图身份。现代 I2V 因此越来越强调 motion control、camera control、reference preservation 和 temporal consistency。

## 最新趋势

- 从“首帧动起来”转向多参考图、多主体、多镜头一致性。
- 从随机运动转向轨迹、相机、姿态、深度和 optical flow 控制。
- 从云端大模型转向移动端 I2V、少步采样和轻量 VAE。
- 从单任务 I2V 转向 text-image-to-video（TI2V）统一条件接口。

## 关键评测

- 首帧主体身份、颜色、纹理和局部结构是否保持。
- 运动是否自然，而不是全局纹理漂移或镜头假动。
- 新出现区域是否合理。
- 长视频中主体是否逐步变形。
- 用户给定轨迹或相机控制是否被遵循。

## 开放问题

1. 如何在保持身份的同时产生大幅、非刚体、可控运动？
2. 多参考图之间冲突时，模型应如何选择？
3. 是否需要显式 3D 表示才能解决相机绕行和遮挡？
4. I2V 与 V2V、inpainting、story generation 的边界会不会逐渐消失？

## 推荐阅读

- Latent Motion Diffusion：把 I2V 视作 motion latent 生成。
- Stable Video Diffusion：开放 I2V baseline。
- DynamiCrafter / I2VGen-XL：开放域图像动画。
- AnyI2V、MobileI2V：运动控制与轻量化方向。
