# 文本到视频

## 任务定义

Text-to-video（T2V）输入自然语言，输出符合描述的视频。它同时约束内容、时间、空间和镜头：人物与物体是否出现，属性是否正确，动作是否按顺序发生，物体关系是否保持，以及景别、视角、相机运动和叙事节奏是否符合要求。

因此 T2V 不是“逐帧生成图片再拼起来”，而是学习：

$$
p(x_{1:T}\mid y,r,a)
$$

其中 $x_{1:T}$ 是视频序列，$y$ 是文本，$r$ 是可选的参考图像、视频或结构条件，$a$ 是动作、相机轨迹等控制信号。本页只梳理**文本条件带来的特殊问题和路线**；通用生成机制、视频基础模型和评测方法分别见：

| 想了解的问题 | 统一入口 |
|---|---|
| VAE、GAN、Diffusion、Flow 如何演化 | [生成模型路线](../generative-models.md) |
| Tokenizer、Transformer 和视频基础模型 | [大模型路线](../foundation-models.md) |
| 视频生成的任务边界、输入与输出 | [视频生成的下游与相关任务](../taxonomy.md) |
| T2V、I2V、V2V、预测和交互任务的边界 | [视频生成的下游与相关任务](../taxonomy.md) |
| 组合性、物理、安全和人类偏好评测 | [评测指南](../evaluation.md) |
| 多镜头、长视频和故事生成 | [故事与多镜头生成](story-multishot.md) |
| 参考图像驱动的视频生成 | [图像到视频](image-to-video.md) |
| 动作条件和闭环世界模拟 | [动作条件预测](action-conditioned-prediction.md)、[交互式世界生成](interactive-world-generation.md)、[World Model](../world-models.md) |

## T2V 在总技术路线中的位置

~~~mermaid
flowchart LR
    accTitle: Text-to-video position
    accDescr: Text-to-video adds language conditions to the general video generation stack and connects to image conditioning, long-form storytelling, and action-conditioned world models.

    base[通用视频生成 backbone] --> t2v[T2V：文本约束时空内容]
    t2v --> i2v[I2V：参考图像约束身份与布局]
    t2v --> story[多镜头：脚本、分镜与记忆]
    t2v --> action[动作条件：反事实与闭环]
    base --> token[Tokenizer / latent / patch]
    base --> generator[GAN / Transformer / diffusion / flow]
    token -. "通用表示" .-> t2v
    generator -. "通用生成机制" .-> t2v

    classDef base_style fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef task_style fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef extension_style fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class base,token,generator base_style
    class t2v task_style
    class i2v,story,action extension_style
~~~

T2V 的核心贡献不是重新发明一套视频生成器，而是提供一个开放、组合、可表达的条件接口。它最先暴露出语言与时空世界之间的错位：文本描述是离散的，视频事件是连续的；文本可以指定关系和顺序，但训练数据往往只提供粗粒度 caption。

## 从原始方法到现代 T2V

| 阶段 | 代表方法 | T2V 特有推进 | 仍然遗留的问题 |
|---|---|---|---|
| 模板、检索与程序动画 | 视频检索、template animation、procedural animation | 将文本映射到有限动作或已有片段 | 开放域生成能力弱 |
| 文本条件 GAN | Video GAN、文本条件视频 GAN 变体 [[1]](#ref-1) | 将文本 embedding 注入时空生成器 | 语义弱、训练不稳定、模式坍塌 |
| 内容—运动解耦 | MoCoGAN [[2]](#ref-2) | 分离主体/背景内容与运动 latent | 复杂交互和语言组合能力有限 |
| 视频 token + Transformer | VideoGPT [[3]](#ref-3)、Phenaki [[4]](#ref-4)、MAGVIT [[5]](#ref-5) | 通过 token 统一文本、图像和视频序列，支持变长与 masked generation | token 数量和长程建模成本高 |
| 文本—视频 diffusion | Video Diffusion Models [[6]](#ref-6)、Make-A-Video [[7]](#ref-7)、Imagen Video [[8]](#ref-8) | 复用图像文本先验，并学习时空去噪 | 多步采样、运动和长程一致性 |
| 图像模型到视频 | AnimateDiff [[9]](#ref-9)、Stable Video Diffusion [[10]](#ref-10) | 用 temporal module 或视频 latent 继承图像模型能力 | 图像先验不等于真实运动先验 |
| 原生时空基础模型 | Lumiere [[11]](#ref-11)、Sora [[12]](#ref-12) | 统一不同长度、分辨率和宽高比，扩大时空上下文 | 数据治理、可控性、评测和成本 |
| 多模态创作模型 | Veo、Sora 2、Seedance、Kling 等路线 | 文本、参考素材、音频、镜头和编辑统一输入 | 身份、来源、安全和长时状态 |

上述阶段中的表示空间和生成机制不再在本页重复展开，分别参见[生成模型路线](../generative-models.md)和[大模型路线](../foundation-models.md)。

## T2V 的五个特有技术问题

### 1. 文本如何变成时空事件

T2V 需要把 prompt 从“物体清单”转换成事件结构：

| 文本成分 | 视频中的对应对象 | 典型失败 |
|---|---|---|
| 实体 | 人、物、背景和角色 | 漏掉实体、对象数量错误 |
| 属性 | 颜色、材质、服装和风格 | 属性绑定到错误对象 |
| 空间关系 | 左右、前后、内外、接触和遮挡 | 关系被平均化或只在单帧成立 |
| 动作 | 方向、速度、姿态和交互 | 动作静态化、方向错误 |
| 时间关系 | 先后、持续、同时和最终状态 | 事件顺序错乱 |
| 镜头语言 | 景别、视角、运动和转场 | 主体运动与相机运动混淆 |

因此，prompt following 不能只问“视频是否像这段文字”，还需要检查实体、属性、关系、动作和状态转移是否正确。这也是 T2V 区别于无条件视频生成的首要地方。

### 2. 文本和视频数据如何对齐

图像—文本数据提供外观和构图先验，视频—文本数据提供动作和时间结构；字幕、ASR、OCR、标题和自动 caption 能扩大规模，但不一定精确描述动作、关系和镜头。Make-A-Video 代表了利用图像—视频迁移减少成对文本—视频数据依赖的路线 [[7]](#ref-7)。

训练数据的关键不是简单增加视频数量，而是提高以下信息的可用性：

- 镜头边界、时间戳和动作阶段
- 主体、物体、属性和空间关系
- 相机运动与主体运动的区分
- 不同长度、帧率、分辨率和宽高比
- 手部、文字、接触、碰撞和多对象交互
- 数据去重、版权、肖像、来源和安全过滤

更完整的数据与训练阶段说明见[大模型路线](../foundation-models.md)和[数据集索引](../../resources/datasets.md)。

### 3. 语言条件如何注入生成器

文本条件通常通过 cross-attention、条件 token、adapter、T2I 先验、prompt expansion 或结构化控制注入生成器。工程上可以把控制信号分成三层：

| 层级 | 作用 | 例子 |
|---|---|---|
| 语义条件 | 决定生成什么 | 文本、剧本、风格、镜头描述 |
| 视觉与结构条件 | 决定主体和布局 | 首帧、参考图、姿态、深度、分割 |
| 运动与交互条件 | 决定如何变化 | 轨迹、相机路径、动作、关键帧、音频 |

首帧或参考图驱动的身份和布局保持属于 I2V 的专门问题，见[图像到视频](image-to-video.md)；动作和实时输入则进入 world model 范畴，见[动作条件预测](action-conditioned-prediction.md)。

### 4. 如何从单镜头扩展到长视频

单镜头 T2V 主要回答“这一段画面是什么”。长视频还要维护角色、场景、道具、事件顺序和叙事状态。常见路线包括：

- 全视频联合建模：一次处理完整时间范围，减少片段边界
- 分块生成：用重叠窗口、首尾帧或 latent 连接短片段
- 递推生成：用历史视频继续预测下一段
- 记忆增强：保存角色、场景、状态和事件摘要
- 分层生成：先生成脚本与分镜，再逐镜头生成

这些方法的取舍和代表工作已独立整理在[故事与多镜头生成](story-multishot.md)；本页只保留判断标准：越长的视频，越需要显式 memory、状态更新和跨镜头评测，而不能只依赖一次性采样。

### 5. 如何判断“理解”而不是“看起来像”

T2V 的评测应从整体质量转向可诊断能力：

- 主体与属性是否绑定正确
- 多对象的数量和空间关系是否正确
- 动作方向、事件顺序和最终状态是否完成
- 角色、背景、文字和光照是否跨帧稳定
- 相机运动是否符合描述
- 接触、碰撞、重力和材料行为是否合理
- 长视频和多镜头是否保持状态
- 安全拒绝、肖像/版权风险和来源信息是否可控

FETV、VBench、EvalCrafter、VideoScore、TC-Bench、T2V-CompBench、VideoPhy 和 T2VSafetyBench 已分别覆盖这些维度；指标定义、实验协议和常见误区统一见[评测指南](../evaluation.md)，本页不再重复列出评测论文清单。

一个实用的 prompt 评测记录可以写成：

~~~yaml
entities: [red_ball, blue_box]
attributes: [ball_is_red, box_is_blue]
initial_relation: red_ball_left_of_blue_box
event: red_ball_rolls_right_into_box
final_state: red_ball_inside_blue_box
must_persist: [object_identity, color, camera_consistency]
~~~

## 与相邻任务的边界

| 任务 | 输入 | 与 T2V 的关系 | 应转到的文档 |
|---|---|---|---|
| Unconditional video generation | 无或类别条件 | T2V 去掉文本条件后的生成基础 | [无条件视频生成](unconditional-video-generation.md) |
| Image-to-video | 首帧或参考图 | T2V 增加视觉条件，重点变为身份和布局保持 | [图像到视频](image-to-video.md) |
| Video-to-video / inpainting | 视频、mask、编辑指令 | T2V backbone 被用于编辑，但要保持输入结构 | [视频到视频](video-to-video.md)、[视频修复与补全](video-inpainting.md) |
| Story / multi-shot | 剧本、分镜、多段 prompt | T2V 增加叙事规划、镜头切分和记忆 | [故事与多镜头生成](story-multishot.md) |
| Video prediction | 历史帧 | 从文本创作转向根据观测预测未来 | [视频预测](video-prediction.md) |
| Action-conditioned prediction | 观测与动作 | 从语言可控转向动作可干预和反事实 | [动作条件预测](action-conditioned-prediction.md) |
| Interactive world generation | 世界状态与连续动作 | 从开放式创作转向实时闭环环境 | [交互式世界生成](interactive-world-generation.md) |

一个简单判断规则是：如果主要问题是“描述的内容是否出现”，它属于 T2V；如果主要问题是“给定动作后会发生什么”，它更接近 action-conditioned world model。

## 最新趋势与开放问题

- 从单镜头短片转向脚本、分镜、记忆和多镜头叙事
- 从纯文本转向文本、参考图、视频、轨迹、相机和音频的复合条件
- 从离线多步采样转向 flow、蒸馏、分块缓存和流式生成
- 从整体语义相似度转向组合关系、状态转移和物理专项评测
- 从封闭产品转向 HunyuanVideo、Wan、CogVideoX、LTX-Video、Open-Sora 等开放生态，入口见[开放模型与代码](../../resources/open-models.md)
- 从生成漂亮画面转向记忆、反事实、动作结果和闭环可靠性

最重要的开放问题仍然是：模型能否绑定复杂实体关系，能否在长视频中保持状态，能否根据动作产生可验证的反事实未来，以及这些能力是否足以支持规划，而不仅是提升视觉观感。

## 最小阅读路径

1. 先读[视频生成的下游与相关任务](../taxonomy.md)，建立任务边界，明确每类任务的输入与输出
2. 读[生成模型路线](../generative-models.md)，理解 VAE、GAN、Diffusion 和 Flow
3. 读[大模型路线](../foundation-models.md)，理解 tokenizer、Transformer 和基础模型
4. 回到本页，重点阅读“文本如何变成时空事件”和“语言条件如何注入生成器”
5. 根据研究目标继续读[故事与多镜头生成](story-multishot.md)、[评测指南](../evaluation.md)或[World Model](../world-models.md)

## 参考文献

<a id="ref-1"></a>[1] [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612). Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. NeurIPS. 2016.

<a id="ref-2"></a>[2] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. CVPR. 2018.

<a id="ref-3"></a>[3] [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas. arXiv preprint. 2021.

<a id="ref-4"></a>[4] [Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions](https://arxiv.org/abs/2210.02399). Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. ICLR. 2023.

<a id="ref-5"></a>[5] [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. CVPR. 2023.

<a id="ref-6"></a>[6] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-7"></a>[7] [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792). Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. ICLR. 2023.

<a id="ref-8"></a>[8] [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303). Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, et al. arXiv preprint. 2022.

<a id="ref-9"></a>[9] [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725). Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, et al. ICLR. 2024.

<a id="ref-10"></a>[10] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-11"></a>[11] [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945). Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, et al. SIGGRAPH Asia. 2024.

<a id="ref-12"></a>[12] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.
