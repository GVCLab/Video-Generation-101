# 文本到视频

## 任务定义

Text-to-video（T2V）输入自然语言描述，输出符合语义、动作、场景、风格和镜头要求的视频。它是当前最受关注的视频生成任务，也是从“生成一段视频”走向“可创作系统”的核心入口。

## 从原始方法到现代方法

| 阶段 | 代表方法 | 技术核心 | 主要变化 |
|---|---|---|---|
| 模板与检索 | 视频检索、模板动画、procedural animation | 用文本匹配已有片段或参数化动画 | 生成性弱，可控但不开放 |
| 文本条件 GAN | TGAN / text-conditioned video GAN 变体、Video GAN [[1]](#ref-1) | 文本 embedding + 时空生成器 | 能生成短片，但质量和语义有限 |
| Transformer token | Phenaki [[2]](#ref-2)、MAGVIT 系列 [[3]](#ref-3), [[4]](#ref-4) | video tokenizer + masked / autoregressive generation | 支持变长、组合语义和多任务条件 |
| Diffusion | Make-A-Video [[5]](#ref-5)、Imagen Video [[6]](#ref-6)、AnimateDiff [[7]](#ref-7)、Lumiere [[8]](#ref-8) | 文本编码器 + temporal U-Net / DiT | 画质和 prompt following 快速提升 |
| 大规模基础模型 | Sora [[9]](#ref-9)、Cosmos [[10]](#ref-10) | spacetime patch、长上下文、多模态训练 | 强化长视频、物理、镜头和音视频同步 |

## 技术演化逻辑

T2V 的难点不是“把文字变成几帧图像”，而是把语言中的动作、时间顺序、物体关系和镜头语言转成连续视觉事件。早期方法通常只能生成短、粗糙、语义较弱的视频 [[1]](#ref-1)。Phenaki 和 MAGVIT 将视频压缩为 token，使变长和 masked generation 成为主线之一 [[2]](#ref-2), [[3]](#ref-3)。Diffusion 以后，模型可以复用图像生成的语义能力，再通过 temporal layer 或时空注意力学习运动 [[5]](#ref-5), [[6]](#ref-6), [[8]](#ref-8)。基础模型阶段则进一步把不同长度、分辨率和宽高比的视频统一成时空 token 或 patch [[9]](#ref-9), [[10]](#ref-10)。

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

## 参考文献

<a id="ref-1"></a>[1] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612). arXiv preprint, 2016.

<a id="ref-2"></a>[2] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399). arXiv preprint, 2022.

<a id="ref-3"></a>[3] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). arXiv preprint, 2022.

<a id="ref-4"></a>[4] Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). arXiv preprint, 2023.

<a id="ref-5"></a>[5] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). arXiv preprint, 2022.

<a id="ref-6"></a>[6] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). arXiv preprint, 2022.

<a id="ref-7"></a>[7] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). arXiv preprint, 2023.

<a id="ref-8"></a>[8] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). arXiv preprint, 2024.

<a id="ref-9"></a>[9] OpenAI. [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). Technical report, 2024.

<a id="ref-10"></a>[10] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-11"></a>[11] [Sora as a World Model? A Complete Survey on Text-to-Video Generation](https://arxiv.org/html/2403.05131v3). 2024-2026 T2V 与 world model survey.
