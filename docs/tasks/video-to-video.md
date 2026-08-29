# 视频编辑：从时空传播到视频基础模型

Video editing 不是视频生成的附属功能，而是它走向真实创作工作流的关键能力。纯生成回答“能否从无到有做出视频”；编辑回答的是更难落地的问题：**能否只改变用户指定的内容，同时保留原视频中不该改变的一切。**

本页将 video-to-video（V2V）、局部编辑、运动编辑和多轮指令编辑放在同一条历史主线上，并解释它们与当前 T2I、I2V、T2V、Diffusion / Flow、DiT、3D/4D 表示和多模态基础模型的关系。

## 1. 什么是视频编辑

给定源视频 $x$、编辑指令 $e$，以及可选的 mask、参考图、轨迹、深度或姿态等条件 $c$，视频编辑模型生成新视频：

$$
y \sim p_\theta(y\mid x,e,c).
$$

它同时追求两个方向相反的目标：

$$
\underbrace{\operatorname{Edit}(y,e)}_{\text{该改的改对}}
\quad + \quad
\lambda\underbrace{\operatorname{Preserve}(y,x,\bar m)}_{\text{不该改的保持不变}}.
$$

$m$ 表示编辑区域，$\bar m$ 表示未编辑区域。编辑强度越大，通常越容易破坏源视频；保留约束越强，又越可能导致指令没有真正执行。这一 **editability–fidelity trade-off** 是贯穿所有方法的核心矛盾。

### 视频编辑不只是一种任务

| 编辑类型 | 例子 | 最需要保持什么 |
|---|---|---|
| 修复与恢复 | 去噪、超分、去模糊、旧片修复 | 原始内容与运动 |
| 局部补全 | 去物体、换背景、扩画幅 | mask 外内容、遮挡后的背景 |
| 外观编辑 | 换颜色、材质、服装、天气、风格 | 几何、身份和原运动 |
| 对象编辑 | 增加、删除或替换主体 | 空间关系、遮挡和接触 |
| 运动编辑 | 改动作、轨迹、速度或局部形变 | 身份、场景和物理连续性 |
| 镜头编辑 | 改视角、运镜或构图 | 3D 场景和对象状态 |
| 结构重绘 | 草图/深度/姿态/语义图到新视频 | 输入结构与时间对应 |
| 多轮指令编辑 | 连续换装、改背景、再调动作 | 前几轮已经接受的改动 |

因此，“video editing”比狭义的风格迁移更广，也不能完全等同于 video-to-video translation 或 video inpainting。

## 2. 它与当前视频生成方法是什么关系

### 2.1 编辑本质上是带强保留约束的条件生成

T2V 从文字和随机噪声出发，允许模型重新决定大部分画面；V2V 则把源视频作为高带宽条件。二者可以共用 VAE、视频 tokenizer、Diffusion / Flow backbone、DiT、文本编码器和 decoder，差别主要在于条件接口与保留机制。

```text
文本到视频：文字 ───────────────→ 视频生成 backbone → 新视频

视频编辑：  源视频 → 编码 / inversion ┐
            指令 → 文本编码          ├→ 同一个或相近的 backbone → 编辑视频
       mask / 深度 / 轨迹 / 参考图 ┘             ↑
                                     保留与一致性约束
```

这解释了为什么视频基础模型增强后，编辑通常也会变强；但它也解释了为什么“会生成”不等于“会精确编辑”：生成模型可以忽略源视频重新画一段合理结果，而编辑必须知道哪些像素、对象、运动和事件属于不可改变的约束。

### 2.2 与图像编辑（T2I editing）的关系

图像模型通常拥有更好的局部细节、指令理解和成熟编辑工具。2022–2024 年大量视频编辑工作因此采用：

```text
逐帧或关键帧图像编辑
        +
光流 / attention / diffusion feature 跨帧传播
        =
时间一致的视频编辑
```

FateZero、Pix2Video 和 TokenFlow 分别通过 attention 融合、锚帧传播和 diffusion feature correspondence 把图像扩散能力搬到视频 [[8]](#ref-8), [[9]](#ref-9), [[10]](#ref-10)。优势是复用强大的图像先验；局限是每帧单独“画得对”后，仍要额外解决遮挡、快速运动和长期漂移。

### 2.3 与 I2V 的关系

I2V 已逐渐成为视频编辑的重要执行器：先把源视频的首帧或关键帧编辑好，再让 I2V 模型沿源视频的运动和结构向后生成。AnyV2V 将这一思路概括为“任意图像编辑器 + I2V + temporal feature injection” [[11]](#ref-11)；后续 flow-driven 方法进一步从源视频提取运动，用编辑首帧驱动整段结果 [[19]](#ref-19)。

这条路线的意义是模块化：图像编辑负责“改成什么”，I2V 负责“怎样随时间动”。但首帧没有出现的对象、后续新遮挡和复杂局部运动，不能只靠首帧条件解决。

### 2.4 与 T2V、Diffusion / Flow 和 DiT 的关系

早期方法通常在预训练 T2I 模型上增加 temporal attention，并依赖 DDIM inversion 或逐视频优化。当前 T2V / I2V DiT 已经具有更强的时空先验，视频编辑因而出现三种变化：

1. **从额外拼接 temporal module，转向直接利用原生视频 backbone；**
2. **从必须 inversion，转向 flow/rectified-flow 下的 inversion-free 或近似反演；**
3. **从每个任务一个网络，转向把 source、mask、reference 和 instruction 统一成条件 token。**

VACE 把 reference-to-video、V2V 和 masked V2V 组织为统一的 Video Condition Unit [[13]](#ref-13)；FiVE-Bench 则展示了将 rectified-flow T2V 模型直接适配为编辑器的潜力 [[14]](#ref-14)。这说明当前关系正在从“借用生成模型做编辑”转向“生成与编辑共享同一个视频基础模型接口”。

### 2.5 与 inpainting 的关系

Masked V2V 是视频 inpainting 的上位接口：

- mask 内允许生成或修改；
- mask 外要求尽量保持；
- 时间维度上还要恢复被遮挡的背景和对象状态。

因此，去物体、局部替换、扩画幅和局部重绘可以由同一个条件形式表达。区别在于，传统 inpainting 更强调填补缺失区域，而 instruction editing 还要理解“删掉谁、换成什么、怎样与场景互动”。专题见[视频修复与补全](video-inpainting.md)。

### 2.6 与个性化、多参考和角色一致性的关系

编辑天然需要“保留这个人、这件衣服、这个产品或这套风格”。因此 subject-driven generation、reference conditioning 和 identity preservation 与 V2V 越来越难分开。当前系统常同时接收源视频、主体参考图、风格参考和文字指令；真正的难点不是让所有条件都出现，而是决定 **每个参考约束什么、作用在哪些帧、条件冲突时听谁的**。

### 2.7 与 3D / 4D 场景编辑的关系

2D 视频编辑容易出现新视角不一致：正面改好了，镜头转到侧面时修改消失。Neural atlas、NeRF、3D Gaussian Splatting 和其他 4D 表示尝试把多帧映射到同一个规范空间，在那里做一次编辑，再渲染回所有帧。Layered Neural Atlases 是这条线的重要节点 [[4]](#ref-4)。

它能改善跨帧与多视角一致性，但需要场景分解、对应关系或几何重建；动态拓扑、强遮挡、反射和非刚体仍很困难。

### 2.8 与 world model 的关系

运动、相机和对象编辑看起来像在问“如果改变一个条件，世界会怎样”。这与反事实 world modeling 相邻，但不能直接画等号：

- 创作型编辑只需输出符合用户意图的合理视频；
- 决策型 world model 必须在固定初始状态下，对不同动作给出动力学正确的不同后果。

把汽车转弯的视频编辑得很自然，并不能证明模型正确预测了方向盘动作。两者的边界见 [World Model 专章](../world-models.md) 与 [物理一致性](../physical-consistency.md)。

## 3. 当前方法的七种核心机制

| 方法族 | 怎样保留源视频 | 怎样执行编辑 | 优势 | 主要失败 |
|---|---|---|---|---|
| 光流与显式传播 | 像素/特征对应 | 编辑一帧后传播 | 直观、局部保真 | 遮挡和大形变易错 |
| Atlas / canonical space | 多帧映射到统一图层或场 | 在规范表示中编辑一次 | 时间和多视角一致 | 每视频优化慢，分解假设强 |
| Inversion + latent reuse | 把源视频反演到噪声/latent | 改 prompt 后重新去噪 | 与 diffusion 接口自然 | 反演误差，编辑强度难平衡 |
| Attention / feature injection | 注入源 attention 或 feature | 用目标 prompt 改语义 | training-free、复用图像模型 | 注入过强会压制编辑 |
| Keyframe / I2V propagation | 锚帧与源运动作为条件 | 图像编辑后向时间生成 | 模块化、细节好 | 首帧外事件和长视频漂移 |
| Native V2V DiT / Flow | 直接编码 source、mask、reference | 基座模型端到端生成 | 任务统一、可规模化 | 成对编辑数据稀缺、算力高 |
| Memory / agentic editing | 保存历史视频、mask、token 或图层 | 多轮理解、检索并修改 | 支持真实迭代工作流 | 旧编辑被覆盖、记忆成本增长 |

这些方法并非互斥。一个现代编辑器可以同时使用 V2V DiT、mask condition、首帧传播、feature injection 和外部 memory。

## 4. 我们建议怎样定义 milestone

本页不把“分辨率更高”或“demo 更漂亮”单独算作里程碑。一个节点至少应改变以下一项：

1. **可编辑对象**：从局部像素扩展到对象、运动、镜头或多轮状态；
2. **控制接口**：从人工 mask/结构图扩展到文字、参考图和自然语言指令；
3. **时间一致机制**：从光流传播扩展到统一时空表示或生成 backbone；
4. **训练范式**：从每视频优化扩展到 zero-shot、training-free 或大规模预训练；
5. **系统角色**：从单点工具扩展到统一视频基础模型或可迭代创作系统。

按照这个标准，下面是一条 **建议性的 video editing milestones**。它强调技术转折，不声称是唯一历史划分。

## 5. Video editing milestones

| 时间 | 建议里程碑 | 关键变化 | 为什么重要 | 当时仍未解决 |
|---|---|---|---|---|
| 1997 | Video Rewrite [[1]](#ref-1) | 根据新音频重排并拼接嘴部视频片段 | 很早展示“只改局部语义、保留其余表演”的自动视频重写 | 只适用于受限人脸与语音 |
| 2004 | Space-Time Video Completion [[2]](#ref-2) | 把视频视为时空体，用 patch 全局优化补齐缺失区域 | 奠定去物体、修复和时空补全的经典问题形式 | 缺少高层语义与生成先验 |
| 2018 | vid2vid [[3]](#ref-3) | 从语义图、姿态等结构序列学习高分辨率条件视频映射 | 将 image-to-image translation 正式扩展为可学习的 video-to-video synthesis | 依赖成对结构标注，开放式编辑弱 |
| 2021 | Layered Neural Atlases [[4]](#ref-4) | 把前景与背景展开为可编辑的统一 2D atlas | 一次编辑可一致传播到变形、遮挡、阴影和反射中的多帧 | 每个视频都要优化，主要擅长外观编辑 |
| 2022 | Text2LIVE [[5]](#ref-5) | 用文字生成可合成的编辑图层 | 文字第一次更自然地进入分层、非破坏式图像/视频编辑 | 复杂形状和运动编辑受限 |
| 2022–2023 | Tune-A-Video / Dreamix [[6]](#ref-6), [[7]](#ref-7) | 将预训练图像/视频 diffusion 适配到单视频与通用视频编辑 | 编辑从固定映射进入开放文本、外观与运动生成 | 逐视频微调慢，源视频保真与编辑幅度冲突 |
| 2023 | FateZero / Pix2Video [[8]](#ref-8), [[9]](#ref-9) | 用 inversion、attention 融合和锚帧传播保留结构与运动 | 建立“复用强 T2I + 显式跨帧一致”的主流范式 | 快速运动、遮挡和大语义改变仍漂移 |
| 2023–2024 | TokenFlow [[10]](#ref-10) | 在 diffusion feature space 建立帧间对应并传播编辑 | 将 training-free feature correspondence 推成代表路线 | 对底层图像编辑器和对应质量敏感 |
| 2024 | AnyV2V [[11]](#ref-11) | 任意图像编辑器编辑首帧，I2V 模型传播到整段视频 | 明确提出模块化的 image editor + I2V 编辑范式 | 首帧条件不足以覆盖全部时间事件 |
| 2024 | Movie Gen [[12]](#ref-12) | 在大规模媒体基础模型家族中统一生成、个性化与精准编辑 | Video editing 从外挂算法升级为 foundation-model 核心能力 | 闭源模型家族，不是一个万能 checkpoint |
| 2025 | VACE [[13]](#ref-13) | 用统一条件单元覆盖生成、V2V 与 masked V2V | 代表原生 DiT 把多种视频创作任务合成一个接口 | 统一接口不保证每个子任务都达到专用模型上限 |
| 2025 | FiVE-Bench / instruction data scaling [[14]](#ref-14), [[15]](#ref-15) | 同时测编辑成功、背景保持、时间一致和质量；规模化合成指令编辑数据 | 领域开始从精选 demo 转向数据和可诊断评测 | 自动指标仍会漏掉对象级和短暂时间错误 |
| 2026 | Memory-V2V [[16]](#ref-16) | 显式检索和压缩前几轮编辑状态 | 把单次 V2V 推向多轮、可持续的创作过程 | 长期记忆冲突与版本回滚仍未成熟 |
| 2026 | EgoEdit / FFP-300K [[17]](#ref-17), [[18]](#ref-18) | 实时流式编辑与大规模高保真首帧传播数据 | 速度、长序列、真实交互和数据规模成为下一阶段主轴 | 结论仍需跨场景、跨模型与独立复现 |

### 怎样读这张表

这不是“新方法淘汰旧方法”的直线：

- 光流和 mask 仍是现代 diffusion 编辑器的重要条件；
- atlas / 3D 表示仍比纯生成模型更容易保证多视角一致；
- T2I 编辑器仍常负责关键帧的精细改动；
- I2V / T2V backbone 提供开放世界外观和运动先验；
- 大规模 V2V 模型正在吸收上述模块，但尚未替代专业时间线、图层、跟踪和人工修正工具。

## 6. 2025–2026 的真正转折

### 6.1 从“测试时技巧”转向“原生编辑模型”

2023 年的核心问题是怎样不训练新视频模型，直接把 T2I diffusion 变成编辑器；当前问题则变成怎样用视频 DiT / Flow 在预训练或后训练中直接学习 source-to-target 映射。inversion、attention injection 不会消失，但不再是唯一中心。

### 6.2 从文本 prompt 转向 instruction editing

Prompt 通常只描述目标画面；instruction 明确描述“把什么改成什么，同时保留什么”。这要求训练数据包含 source video、编辑指令和 target video 三元组。真实成对数据昂贵，因此当前方法大量使用图像编辑对提升、专家模型合成、伪视频变换和自动过滤 [[15]](#ref-15)。

### 6.3 从单次输出转向多轮非破坏式编辑

真实用户会说：“先换衣服，再把背景改成雪天，但保留上一轮的人物和动作。”这需要：

```text
编辑历史 → 选择有效版本 → 定位本轮变化 → 保留既有修改 → 生成新版本
```

Memory-V2V 表明外部视频 cache 与 token compression 是一种实现 [[16]](#ref-16)，但更完整的系统还需要图层、版本、撤销、冲突解析和局部重算。

### 6.4 从短片离线处理转向长视频与实时流式编辑

当前研究开始把时长、分辨率、速度和显存放到与画质同等重要的位置。流式第一视角编辑还必须处理持续相机运动、手—物交互和新内容不断进入画面的情况 [[17]](#ref-17)。这比把一个固定短 clip 整段加载后编辑更接近真实应用。

### 6.5 从 2D 视觉合理转向场景与物理约束

对象删除要补出真实被遮挡背景；换材质要保持光照和形变；改运动不能破坏接触关系；改相机需要场景在新视角仍成立。下一代编辑器会更紧密地结合 tracking、depth、3D/4D reconstruction、physics prior 与视频生成，而不是只依赖文本相似度。

## 7. 如何评测 video editing

编辑不能只报 FVD、CLIP 或总体美学分。至少需要把以下维度分开：

| 维度 | 要问的问题 | 典型测法 |
|---|---|---|
| 编辑成功 | 指令要求的变化真的发生了吗？ | 对象/属性/关系专项判断，人工或 VLM 校准 |
| 未编辑区域保持 | 背景、身份、构图和细节是否被误改？ | mask 外 LPIPS/DINO、tracking、人工对照 |
| 运动保持或正确改变 | 原运动是否保留，指定新动作是否实现？ | 轨迹、光流、姿态与 motion fidelity |
| 时间一致 | 是否闪烁、漂移、突然变形或遗忘？ | 长短期 feature consistency + 人评 |
| 视频质量 | 是否清晰、自然、无压缩和生成伪影？ | VQA、逐帧诊断和人评 |
| 多轮一致 | 前几轮已接受的修改是否持续？ | 跨轮 source/target 对照和版本测试 |
| 效率 | 用户是否能交互式使用？ | 延迟、峰值显存、吞吐和重算范围 |

VE-Bench、FiVE-Bench 与 IVEBench 等工作共同把评测从“文字相似 + 帧间相似”扩展为画质、指令遵循、源视频保真和细粒度编辑成功 [[14]](#ref-14), [[20]](#ref-20), [[21]](#ref-21)。但 VLM 裁判可能忽略短暂闪烁、接触错误和小对象消失，因此必须用人工评测校准。

### 最小公平协议

1. 同一组源视频、指令、mask、参考素材和输出分辨率；
2. 同一生成次数与挑选规则，不能只展示 best-of-many；
3. 同时包含局部和全局、外观和运动、短视频和长视频；
4. 分别报告编辑成功与保留质量，不把两者平均后隐藏 trade-off；
5. 展示失败案例，并说明模型是否使用逐视频训练、人工关键帧或外部编辑器；
6. 多轮编辑要固定历史，并检查撤销、冲突和遗忘。

## 8. 仍未解决的问题

1. **精确局部控制**：怎样只改一个对象的一部分，又不污染邻近区域？
2. **大幅运动编辑**：怎样改变动作和轨迹，同时保持身份、接触和遮挡关系？
3. **长时状态**：对象离开画面再返回时，编辑是否仍然存在？
4. **多轮与版本管理**：模型应记住像素、latent、对象、图层还是操作历史？
5. **3D/4D 一致性**：编辑在新视角和动态形变中是否仍然成立？
6. **数据**：如何获得真实、高分辨率、长时、多样的 source–instruction–target 三元组？
7. **可逆与可追踪**：生成式编辑怎样支持图层、撤销、局部重算和来源记录？
8. **评测**：如何测到短暂错误、小对象、物理交互和跨轮冲突？
9. **安全与权利**：人物、商标、版权内容和误导性编辑怎样授权、披露和追踪？

## 9. 建议阅读路径

### 只想建立历史直觉

```text
Space-Time Completion
    → vid2vid
    → Layered Neural Atlases / Text2LIVE
    → Dreamix / FateZero / TokenFlow
    → AnyV2V
    → Movie Gen / VACE
    → instruction data / memory / streaming
```

### 想做当前研究

1. 用 FateZero、Pix2Video、TokenFlow 理解 inversion、attention 和 feature propagation；
2. 用 AnyV2V 与 flow-driven editing 理解图像编辑 + I2V 的模块化路线；
3. 用 Movie Gen、VACE、FiVE 理解 foundation video model / Flow / DiT 如何改变编辑；
4. 用 Ditto、Memory-V2V、EgoEdit、FFP-300K 理解数据、多轮、长视频和实时方向；
5. 同时阅读[视频修复与补全](video-inpainting.md)、[视频基础模型](../foundation-models.md)与[评测指南](../evaluation.md)。

## 参考文献

<a id="ref-1"></a>[1] [Video Rewrite: driving visual speech with audio](https://doi.org/10.1145/258734.258880). Christoph Bregler, Michele Covell, Malcolm Slaney. SIGGRAPH. 1997.

<a id="ref-2"></a>[2] [Space-Time Video Completion](https://graphics.stanford.edu/courses/cs448a-06-winter/wexler-completion-cvpr04.pdf). Yonatan Wexler, Eli Shechtman, Michal Irani. CVPR. 2004.

<a id="ref-3"></a>[3] [Video-to-Video Synthesis](https://arxiv.org/abs/1808.06601). Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, et al. NeurIPS. 2018.

<a id="ref-4"></a>[4] [Layered Neural Atlases for Consistent Video Editing](https://arxiv.org/abs/2109.11418). Yoni Kasten, Dolev Ofri, Oliver Wang, Tali Dekel. ACM TOG (SIGGRAPH Asia). 2021.

<a id="ref-5"></a>[5] [Text2LIVE: Text-Driven Layered Image and Video Editing](https://text2live.github.io/). Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, Tali Dekel. ECCV. 2022.

<a id="ref-6"></a>[6] [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.html). Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, et al. ICCV. 2023.

<a id="ref-7"></a>[7] [Dreamix: Video Diffusion Models are General Video Editors](https://arxiv.org/abs/2302.01329). Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, et al. arXiv preprint. 2023.

<a id="ref-8"></a>[8] [FateZero: Fusing Attentions for Zero-shot Text-based Video Editing](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html). Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, et al. ICCV. 2023.

<a id="ref-9"></a>[9] [Pix2Video: Video Editing using Image Diffusion](https://openaccess.thecvf.com/content/ICCV2023/html/Ceylan_Pix2Video_Video_Editing_using_Image_Diffusion_ICCV_2023_paper.html). Duygu Ceylan, Chun-Hao P. Huang, Niloy J. Mitra. ICCV. 2023.

<a id="ref-10"></a>[10] [TokenFlow: Consistent Diffusion Features for Consistent Video Editing](https://arxiv.org/abs/2307.10373). Michal Geyer, Omer Bar-Tal, Shai Bagon, Tali Dekel. ICLR. 2024.

<a id="ref-11"></a>[11] [AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks](https://arxiv.org/abs/2403.14468). Max Ku, Cong Wei, Weiming Ren, Harry Yang, Wenhu Chen. TMLR. 2024.

<a id="ref-12"></a>[12] [Movie Gen: A Cast of Media Foundation Models](https://arxiv.org/abs/2410.13720). Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, et al. arXiv preprint. 2024.

<a id="ref-13"></a>[13] [VACE: All-in-One Video Creation and Editing](https://arxiv.org/abs/2503.07598). Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, Yu Liu. ICCV. 2025.

<a id="ref-14"></a>[14] [FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html). Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, Mengyu Wang. ICCV. 2025.

<a id="ref-15"></a>[15] [Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](https://arxiv.org/abs/2510.15742). Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, et al. CVPR. 2026.

<a id="ref-16"></a>[16] [Memory-V2V: Memory-Augmented Video-to-Video Diffusion for Consistent Multi-Turn Editing](https://arxiv.org/abs/2601.16296). Dohun Lee, Chun-Hao Paul Huang, Xuelin Chen, Jong Chul Ye, Duygu Ceylan, Hyeonho Jeong. arXiv preprint. 2026.

<a id="ref-17"></a>[17] [EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing](https://openaccess.thecvf.com/content/CVPR2026/html/Li_EgoEdit_Dataset_Real-Time_Streaming_Model_and_Benchmark_for_Egocentric_Video_CVPR_2026_paper.html). Runjia Li, Moayed Haji-Ali, Ashkan Mirzaei, Chaoyang Wang, Arpit Sahni, Ivan Skorokhodov, et al. CVPR. 2026.

<a id="ref-18"></a>[18] [FFP-300K: Scaling First-Frame Propagation for Generalizable Video Editing](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html). Xijie Huang, Chengming Xu, Donghao Luo, Xiaobin Hu, Peng Tang, Xu Peng, et al. CVPR. 2026.

<a id="ref-19"></a>[19] [Consistent Video Editing as Flow-Driven Image-to-Video Generation](https://arxiv.org/abs/2506.07713). Ge Wang, Songlin Fan, Hangxu Liu, Quanjian Song, Hewei Wang, Jinfeng Xu. arXiv preprint. 2025.

<a id="ref-20"></a>[20] [VE-Bench: Subjective-Aligned Benchmark Suite for Text-Driven Video Editing Quality Assessment](https://arxiv.org/abs/2408.11481). Shangkun Sun, Xiaoyu Liang, Songlin Fan, Wenxu Gao, Wei Gao. AAAI. 2025.

<a id="ref-21"></a>[21] [IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment](https://arxiv.org/abs/2510.11647). Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, et al. ICLR. 2026.
