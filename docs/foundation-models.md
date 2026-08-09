# 大模型路线：Video Foundation Model 与 World Foundation Model

本章讨论的是视频生成进入“大模型阶段”之后发生的变化：模型不再只是某个任务的生成器，而开始成为多任务、多条件、多模态、可迁移的基础模型。

这里的“大模型”不只指参数量。更重要的是数据规模、统一表示、多条件接口、跨任务迁移，以及是否能承载下游编辑、理解、预测、交互和规划。

## 1. 从单任务模型到基础模型

早期视频模型通常服务于单一任务：预测未来帧、无条件生成短视频、文本生成视频或图像生成视频。基础模型阶段的变化是：

- 训练数据从小型标注集扩展到互联网级视频、图像、文本、音频和合成数据。
- 条件从单一文本扩展到图像、视频、音频、轨迹、相机、动作和多参考素材。
- 输出从几秒短片扩展到多镜头、长时长、可编辑和可交互内容。
- 模型从单一生成器变成 tokenizer、backbone、decoder、adapter、安全模块和评测系统的组合。

## 2. Tokenizer 是视频大模型的地基

视频比文本和图像更贵。一个基础问题是：如何把巨大的时空信号压成模型能处理的 token 或 latent。

常见选择：

- **连续 latent**：autoencoder 压缩后在 latent 中 diffusion 或 flow。
- **离散 token**：VQ-VAE [[1]](#ref-1)、VQGAN、MAGVIT [[4]](#ref-4) 类 tokenizer，把视频变成离散序列。
- **spacetime patch**：把不同时长、分辨率和宽高比的视频统一成时空 patch。
- **多尺度表示**：低分辨率负责全局运动，高分辨率负责纹理和细节。

Tokenizer 的质量 [[5]](#ref-5)会直接限制大模型上限：压缩太强会损失运动细节，token 太多又会让训练和采样成本爆炸。

## 3. Transformer 化：把视频放进统一序列建模框架

VideoGPT [[2]](#ref-2)、Phenaki [[3]](#ref-3)、MAGVIT 和 Sora [[6]](#ref-6) 等路线都体现出一个趋势：把视频转成序列或 patch，再用 Transformer 建模长程依赖。

Transformer 的优势在于统一：

- 文本、图像、视频和动作都可以变成 token。
- 多条件控制可以通过 cross-attention、prefix、mask 或 adapter 接入。
- scaling law 思维可以迁移到视频生成。

它的问题也同样直接：视频 token 数量巨大，注意力成本高，长程一致性和可控性仍然昂贵。

## 4. 多模态条件：从 prompt 到完整创作接口

现代视频大模型往往不只接受一句 prompt。更实用的接口包括：

- 文本到视频：语义、动作、风格、镜头描述。
- 图像到视频：让静态参考图动起来。
- 视频到视频：风格化、补全、扩展、局部编辑。
- 音频到视频：口型、音乐节奏、环境声同步。
- 多参考生成：角色、场景、物体、姿态和风格保持。
- 镜头与运动控制：相机轨迹、关键帧、物体路径和深度结构。

这也是“大模型路线”和“相关应用”强相关的地方：能力是否真实可用，往往取决于条件接口和编辑闭环，而不是单次生成质量。

## 5. World foundation model：从内容生成走向物理与动作

当模型开始接受动作、相机、机器人控制或交互输入时，它就进入 world foundation model 的讨论范围。

代表方向：

- **Sora / Veo 类视频基础模型**：强调大规模视频生成中出现的空间、运动和物理规律。
- **Genie 系列 [[7]](#ref-7)**：从被动视频中学习 latent action，生成可交互环境。
- **GameNGen [[8]](#ref-8)**：用生成模型模拟可玩的游戏环境。
- **Cosmos [[9]](#ref-9)**：把视频 tokenizer、生成、物理推理、动作预测和机器人数据管线放入 Physical AI 平台。
- **GWM-1 [[11]](#ref-11)**：探索可探索世界、实时角色和机器人动作条件 rollout。

这里要保持一个清醒边界：video foundation model 可以成为 world model 的视觉先验，但只有在动作条件、状态持久性、反事实和闭环任务中有效，才更接近决策型 world model。

## 6. 大模型路线的评估维度

基础模型不能只看 FID 或人工偏好。更合适的评估维度包括：

- 文本、图像、音频和视频条件是否被同时遵循。
- 角色、物体和场景状态能否跨镜头保持。
- 物理事件是否在常识和几何上自洽。
- 编辑是否局部生效，而不是破坏全局结构。
- 长视频是否有可追踪的叙事、因果和镜头连续性。
- 动作条件模型是否能通过真实或高保真环境中的闭环任务验证。

## 7. 这条路线的核心矛盾

大模型让视频生成拥有更强的开放世界知识，但也带来新的不透明性：

- 数据来源、版权、人物肖像和安全边界更复杂。
- 模型可能在 demo 中显得理解物理，却在反事实测试中失败。
- 越是统一的模型，越需要分清它在生成、理解、编辑、预测和控制中分别被验证到了什么程度。

因此，本仓库把“大模型路线”放在生成模型之后、world model 之前：它是两者之间的桥，但不是自动完成的终点。

## 参考文献

<a id="ref-1"></a>[1] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). arXiv preprint, 2017.

<a id="ref-2"></a>[2] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/abs/2104.10157). arXiv preprint, 2021.

<a id="ref-3"></a>[3] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, et al. [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399). arXiv preprint, 2022.

<a id="ref-4"></a>[4] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, et al. [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199). arXiv preprint, 2022.

<a id="ref-5"></a>[5] Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, et al. [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737). arXiv preprint, 2023.

<a id="ref-6"></a>[6] OpenAI. [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). Technical report, 2024.

<a id="ref-7"></a>[7] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). arXiv preprint, 2024.

<a id="ref-8"></a>[8] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837). arXiv preprint, 2024.

<a id="ref-9"></a>[9] NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). arXiv preprint, 2025.

<a id="ref-10"></a>[10] Google DeepMind. [Genie 3: A New Frontier for World Models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/). Project report, 2025.

<a id="ref-11"></a>[11] Runway. [Introducing Runway GWM-1](https://runway.com/research/introducing-runway-gwm-1). Project report, 2025.
