# 无条件视频生成

## 任务定义

无条件视频生成学习视频数据本身的分布，通常只给随机噪声或类别标签，输出一段新视频。它是研究视频分布建模的最纯粹形式，也是后来文本、图像、编辑和交互条件生成的底层能力来源。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 局限 |
|---|---|---|---|
| 传统合成 | Video Textures、Dynamic Textures | 重组相似帧或学习线性动态系统 | 适合循环纹理，不适合开放语义 |
| 早期深度生成 | Video GAN、MoCoGAN | 时空卷积、内容/运动 latent 解耦 | 分辨率低、训练不稳定 |
| 大型 GAN | DVD-GAN、DIGAN | 多尺度判别器、隐式动态建模 | mode collapse、难扩展到开放世界 |
| Token / AR | VideoGPT、MAGVIT | VQ-VAE / video tokenizer + Transformer | token 数量大，采样慢 |
| Diffusion / DiT | Video Diffusion Models、latent video diffusion | 从噪声迭代还原视频分布 | 推理成本高，长程一致性难 |
| Foundation model | Sora、Veo、Cosmos 类系统 | 大规模视频数据、spacetime patch、多任务后训练 | 数据治理、评测和可解释性复杂 |

## 技术演化逻辑

早期无条件生成主要关心“视频是否像真实数据”。GAN 时代通过判别器获得锐利样本，但覆盖度与稳定性较弱。Tokenizer 与 Transformer 把视频转为序列，使似然建模和大规模自回归成为可能。Diffusion 则把训练稳定性、样本质量和多样性推到新的水平，成为现代视频基础模型的主要底座。

今天无条件生成通常不再作为产品入口出现，因为纯随机视频很难控制；但它仍然是理解生成模型本体能力的关键：模型能否学到运动先验、场景统计、对象持久性和物理规律。

## 最新趋势

- 从像素级 GAN 转向 latent diffusion、DiT 和 flow matching。
- 从短视频样本转向长视频、可变宽高比和多分辨率时空 patch。
- 从单一生成目标转向可复用 backbone：同一个模型通过条件接口支持 T2V、I2V、V2V、inpainting 和 world modeling。
- 评测从 FVD/IS 扩展到物体持久性、物理常识、长程一致性和人工偏好。

## 关键问题

1. 无条件模型学到的是表面视频统计，还是可迁移的动态结构？
2. 如何评估 rare event 与长尾运动，而不是只看平均视觉质量？
3. 视频 tokenizer 的压缩损失会不会限制后续任务上限？
4. diffusion / flow 能否在保持质量的同时大幅降低采样成本？

## 推荐阅读

- Video Textures、Dynamic Textures：传统数据驱动视频合成。
- Video GAN、MoCoGAN、DVD-GAN：GAN 视频生成路线。
- VideoGPT、MAGVIT：tokenizer + Transformer 路线。
- Video Diffusion Models、Stable Video Diffusion、Lumiere、Sora：diffusion 与基础模型路线。
