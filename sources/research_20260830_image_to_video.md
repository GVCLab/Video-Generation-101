# 图像到视频（I2V）研究记录：检索、证据、图片与验收

> 检索冻结日：**2026-08-30（Asia/Shanghai）**
> 对应章节：[`docs/tasks/image-to-video.md`](../docs/tasks/image-to-video.md)
> 记录目的：区分真正的 I2V 任务里程碑与通用视频祖先/邻接任务，固定首发与正式年份，并让 tensor 合同、论文结论、生成图片和最终验证可以逐项复核。

## 1. 研究问题与范围

本轮围绕八个可判定问题展开：

1. 输入图何时是输出时间锚点，何时只提供身份、外观或风格？
2. I2V 与 reference-to-video、角色动画、image-conditioned editing、camera-conditioned T2V、VFI 的边界是什么？
3. 图像通过帧替换、通道拼接、cross-attention、control residual、噪声初始化或显式 warp 进入模型时，分别能保证什么？
4. 隐式 video prior、显式 flow/trajectory、latent motion、camera/3D control、keyframe 和 long-rollout 路线各解决哪个变量？
5. 为什么 image fidelity 与 motion amplitude/correctness 必须联合报告，而不能被一个总分替代？
6. 2016 原型、2018 命名任务、2023–2024 diffusion I2V、2025–2026 控制/蒸馏前沿的首发和正式年份如何裁决？
7. 长视频、移动端和 native audio-video 何时属于 I2V 子协议，何时只是相邻能力？
8. 如何从静态偏置、身份漂移、相机/物体纠缠、首帧误差和 chunk seam 反推失败路径？

纳入范围为：单图/稀疏图作为输出时间锚点的视频生成，以及直接改善该合同的运动、相机、身份、关键帧、长视频、蒸馏和端侧方法。角色动画、视频编辑、纯 camera-conditioned T2V 与 native AV 只作为边界样本。章节不汇总跨论文 SOTA 分数。

## 2. 检索设计

### 2.1 三类互补的一手来源

| 来源 | 作用 | 本轮使用 | 局限 |
|---|---|---|---|
| arXiv API / PDF | 冻结 v1 日期、标题、作者、版本；回读公式/张量/训练协议 | 题名/摘要检索与目标 ID 批量回读 | comment 中的 accepted 声明不单独等于正式出版 |
| OpenAlex API | 检查题名覆盖与遗漏方向 | `title.search` 多组词检索 | 连字符、stemming 和 tokenization 造成极大噪声 |
| 正式 proceedings / 期刊 | 核验题名、作者、venue、页码、正式年份 | NeurIPS、CVF、ECVA、OpenReview/TMLR、ACM、AAAI、IJCV、ICML/PMLR 论文页或正式 PDF | 常不记录最早预印本日期 |
| 作者/机构项目与代码 | 核查模型接口、checkpoint、输入条件与实物 | NVIDIA Research、作者 project/GitHub 与论文身份互证 | README 的速度/SOTA 仍是作者协议，不视为独立复现 |

搜索引擎摘要、二手综述和排行榜只用于发现候选；正文结论均回到论文全文、正式出版页或作者原始 artifact。

### 2.2 精确查询、日期与原始结果数

以下查询执行于 2026-08-30。计数是 API 当时返回的原始数量；不同查询高度重叠，**不能相加**。

| 通道 | 精确检索式 | 原始结果数 | 处理 |
|---|---|---:|---|
| arXiv API | `ti:"image-to-video"` | 435 | 主发现池；题名 tokenization 仍会收进邻接工作 |
| arXiv API | `ti:"image conditional video generation"` | 3 | 补无连字符写法 |
| arXiv API | `ti:"image-conditioned video generation"` | 3 | 补条件式命名 |
| arXiv API | `ti:"text-image to video"` | 6 | 补 TI2V 与非配对图文路线 |
| arXiv API | `ti:"image to video generation"` | 96 | 题名级宽检索 |
| arXiv API | `abs:"image-to-video generation"` | 366 | 摘要级补漏；不作纳入分母 |
| OpenAlex | `filter=title.search:image-to-video` | 20,622 | 发现连字符被拆词后极度过宽 |
| OpenAlex | `filter=title.search:image conditioned video` | 47 | 条件式题名核对 |
| OpenAlex | `filter=title.search:text image to video` | 449 | TI2V 题名核对 |
| OpenAlex | `filter=title.search:image-to-video camera` | 503 | camera 子路线发现池 |

探索性查询 `all:"image-to-video"` 在 arXiv 返回 34,709；该结果把普通的 image/video 共现大量纳入，判定为**无效宽检索**，只记录噪声，不参与筛选。OpenAlex 的 20,622/449/503 也只用来检查可能遗漏的题名，不被解释为 I2V 论文规模。

### 2.3 定向元数据与正式页核验

目标 arXiv ID 主要包括：

`1609.02612`、`1807.09951`、`2105.04551`、`2112.02815`、`2303.13744`、`2304.11603`、`2310.12190`、`2311.04145`、`2311.15127`、`2312.16693`、`2401.15977`、`2402.04324`、`2406.02509`、`2409.11367`、`2412.07730`、`2502.10059`、`2503.11251`、`2504.12626`、`2505.20629`、`2507.02857`、`2510.01284`、`2511.21475`、`2601.03233`、`2605.19398`、`2605.30895`、`2608.13205`。

此外，SEINE、ConsistI2V、Motion-I2V、ReasonDiff、V-PAE、DynamiCrafter、RealCam-I2V、AnyI2V、VACE、CameraCtrl 等分别回到 OpenReview、ACM DOI、CVF、AAAI、ECVA 正式页/PDF核验。CameraCtrl 的正式题名为 *CameraCtrl: Enabling Camera Control for Video Diffusion Models*，正式 venue 是 ICLR 2025；arXiv v1 题名中的 `Text-to-Video` 不被误写成 ECCV 论文。ReasonDiff 未找到可一手确认的同名 arXiv v1，故首发栏记为 `—`，不猜 ID。

## 3. 纳入、排除、去重与证据等级

### 3.1 纳入条件

- 明确把输入图放在输出时间轴中，或直接改善该 I2V 条件合同；
- 能从正式全文、arXiv 全文或作者 artifact 核验至少一项机制、tensor、训练/推理协议或日期；
- 2026 新工作可只有预印本，但正文必须显式写“预印本”；
- 邻接工作只有在帮助划定 animation/editing/camera/native-AV 边界时纳入；
- 同一工作合并 arXiv 与正式版：首次公开取 arXiv v1，正式年份取 proceedings/期刊。

### 3.2 排除条件

- 标准任务没有外部静态输入图的通用视频生成/预测工作；
- 只有二手博客、模型聚合站或无法与作者对应的复刻仓库；
- 只在 related work 提到 I2V，方法本身不提供 I2V 条件或评测；
- 只有精选 demo、没有明确输入/输出/锚点合同的产品宣传；
- 截止冻结日之后的版本主张；若 v1 在冻结日前、v2 在冻结日后，只采用冻结日前可见证据。

### 3.3 筛选裁决

最终正文证据集为 **34 个工作实体**：21 个已有正式出版的一手来源，13 个技术报告/预印本；其中 7 个仅用于原型或邻接任务边界。没有把 435 个 arXiv 或 20,622 个 OpenAlex 结果描述成“已阅读全文”。重点 PDF/正式全文回读覆盖条件路径、训练/推理合同和限制；其余只支持题名、venue 或边界定义。

典型排除/降级案例：

| 候选 | 裁决 | 原因 |
|---|---|---|
| MoCoGAN | 不列 I2V 里程碑 | 内容 latent 不是测试时外部参考图 |
| SVG / learned-prior video prediction | 不列 I2V 里程碑 | 标准合同是给过去 context 预测未来，不是任意静态图锚点 |
| Vondrick 2016 | 作为原型，不称现代 I2V foundation model | 从单图生成短未来，但目标、分辨率和开放域控制合同不同 |
| Eulerian motion fields | 纳入窄域 animation 分支 | 高质量循环自然动态，不代表任意物体大动作 |
| Animate Anyone | 只用于角色动画边界 | 参考人物图 + pose sequence，不要求参考图成为像素首帧 |
| CameraCtrl | 只用于 camera-conditioned T2V 边界 | 无锚点图时不是 I2V |
| VACE | 只用于 editing/统一接口边界 | 有源视频时主任务是编辑；仅无源视频 + 时间锚点子协议属于 I2V |
| Ovi | 只用于 native AV 前沿 | 原生联合音视频本身不自动提供 I2V 锚点 |
| LTX-2 | 只用于 native AV 前沿 | 联合音视频技术报告；仍须另有图像时间锚点才构成严格 I2V 子协议 |

### 3.4 证据等级

| 等级 | 定义 | 可支持的表述 |
|---|---|---|
| **E1** | 正式会议/期刊全文或正式 PDF，关键机制/协议已回读 | 正式 venue、机制、公式、协议和作者实验边界 |
| **E2** | arXiv/技术报告全文与元数据 | v1 日期、预印本方法、作者报告；不冒充同行评审结论 |
| **E3** | 作者/机构项目页、代码、模型仓库或产品页，与论文身份互证 | 只支持该页面明确提供的发布面；性能仍需复测 |
| **E4** | OpenAlex/搜索摘要/二手索引 | 只用于发现线索，不单独支持正文结论 |

作者自报的“最佳”“实时”“100×”“SOTA”均不跨协议升级为事实。正文只保留机制与条件化限制；MobileI2V 的端侧时延、Step-Video 的规模/帧数、V-PAE 的加速等均绑定作者设备、实现和任务。

发布面另用独立字段记录，不能由 E3 合并推断：

| 字段 | 含义 | 不可替代 |
|---|---|---|
| `paper` | 可读方法与作者实验 | 不等于代码可运行 |
| `code` | 指定实现/配置可审计 | 不等于论文权重已发布 |
| `weights` | 指定 checkpoint 可下载 | 不等于在线 demo 使用同一权重 |
| `demo_or_api` | 冻结日可访问的交互入口 | 不等于可离线复现或长期可用 |
| `access_date` | 实际回读日期 | 不等于未来仍保持相同开放状态 |

## 4. 条件合同与机制回读

### 4.1 锚点与图像注入

| 工作 | 原始证据 | 已核验的条件路径 | 证据 |
|---|---|---|---|
| SVD | arXiv `2311.15127` 全文 | 条件图加 noise augmentation，经图像 latent 沿时间复制，再与视频 latent 通道拼接；frame rate/motion score 条件；14/25 帧 I2V 微调 | E2 |
| DynamiCrafter | ECCV 2024 PDF/补充 | query transformer 产生 text-aligned rich image context 进 cross-attention；完整图像 latent 再与噪声拼接 | E1 |
| Step-Video-TI2V | arXiv `2503.11251` PDF | 单帧 `Zc` 时间补零，与 `Zv` 通道拼接为 `f' x 2c x h' x w'` DiT 输入 | E2 |
| ConsistI2V | TMLR 2024 PDF | first-frame spatiotemporal attention + 参考图低频噪声初始化 | E1 |
| STIV | ICCV 2025 PDF | variable image condition/known-frame replacement，image dropout，joint image-text CFG | E1 |
| I2V-Adapter | arXiv `2312.16693` | adapter 将图像信息接入已有 diffusion，而非重训完整基座 | E2 |

“通道拼接”本身不是硬锚点；“latent replacement”也只保证编码空间中的条件，除非最后把 RGB 首帧直接复制回输出，否则不能承诺逐像素相等。

### 4.2 motion prior 与显式运动

| 工作 | 表示/训练 | 优势 | 不能越界的结论 | 证据 |
|---|---|---|---|---|
| Forecast & Refine | structure sequence + residual motion refinement | 早期明确 I2V translation | 单物体/早期域不代表开放域 | E1 |
| cINN I2V | 条件可逆网络采样随机未来 | 同图多种未来 | 多样性受训练域限制 | E1 |
| LFDM | flow AE + 3D U-Net latent-flow diffusion + warp | motion 可解释、可分离 | warp 不能自动补新显露纹理 | E1 |
| LaMD | motion-decomposed video AE + latent motion diffusion | diffusion 目标更低维 | 正式 IJCV 2025，不应只写 2023 | E1 |
| DynamiCrafter/SVD | 隐式 latent video prior | 可直接 hallucinate 新区域 | appearance/motion 共网导致难定位 | E1/E2 |
| Motion-I2V | flow/trajectory diffusion + motion-augmented temporal attention | 可显式轨迹控制和传播首帧 feature | 错 flow 会系统传播；遮挡仍需生成 | E1 |
| DyMoS | 早期 denoising 调 reference-key attention logits | training-free motion slider | 2026 预印本干预，不是所有架构的因果定律 | E2 |

### 4.3 camera/3D、keyframe、long rollout 与 native AV

| 路线 | 已回读事实 | 边界 | 证据 |
|---|---|---|---|
| CamCo | camera trajectory 条件与 3D consistency 目标 | 2024 预印本 | E2 |
| RealCam-I2V | 单目 metric depth 构建可交互 3D 场景；训练 metric-scale pose alignment；推理 scene-constrained noise shaping | depth/SfM 错误仍会传入生成 | E1 |
| CamGeo | 稀疏 camera keyframe；VGGT pose/depth 只在训练期蒸馏；三阶段 coarse-to-fine curriculum | ICML 2026；不是显式 4D 输出 | E1 |
| SEINE | random-mask diffusion 统一 prediction/transition/completion | 首尾 transition 多解，不等于严格 VFI | E1 |
| FramePack | 几何重要性打包历史；endpoint planning/inverted sampling 抑制 drift | 反向规划使用未来端点，非严格在线 | E2 |
| Ovi | 对称 audio/video DiT，每 block 双向跨模态 fusion、scaled RoPE 对齐时间 | native AV 只有带锚点子协议才是 I2V | E2 |
| LTX-2 | 14B video / 5B audio 非对称双流，以双向 cross-attention、temporal embeddings 和 cross-modality AdaLN 联合生成 | 论文公开音视频模型；仍须声明图像时间锚点，才是严格 I2V 子协议 | E2+E3 |

### 4.4 正文 34 条引用的一手证据清单

下表与正文 `ref-1` 至 `ref-34` 一一对应。`E1` 的 venue 来自正式页/全文；`E2` 只写预印本或技术报告；`E3` 只补 artifact，不提升论文结论等级。

| Ref | 工作 | 一手核验入口 | 等级 | 正文角色 |
|---:|---|---|---|---|
| 1 | Generating Videos with Scene Dynamics | NeurIPS 2016 proceedings | E1 | 单图未来原型，不称现代 TI2V |
| 2 | Forecast & Refine Residual Motion | ECCV 2018/CVF | E1 | 命名 I2V translation |
| 3 | Stochastic I2V using cINNs | CVPR 2021/CVF | E1 | 随机未来 |
| 4 | Eulerian Motion Fields | CVPR 2021/CVF | E1 | 窄域循环 animation |
| 5 | Interactive Object Dynamics | CVPR 2021/CVF | E1 | poke/稀疏运动控制 |
| 6 | Make It Move | CVPR 2022/CVF | E1 | 文本控制 TI2V |
| 7 | Latent Flow Diffusion | CVPR 2023/CVF | E1 | flow diffusion + warp |
| 8 | LaMD | IJCV DOI/full text | E1 | latent motion 分解 |
| 9 | DynamiCrafter | ECCV 2024/ECVA | E1 | 双图像注入路径 |
| 10 | I2VGen-XL | arXiv `2311.04145`/project | E2+E3 | 级联技术报告 |
| 11 | Stable Video Diffusion | arXiv `2311.15127` | E2 | 大规模 latent I2V 报告 |
| 12 | SEINE | ICLR 2024/OpenReview | E1 | 首尾 transition/掩码条件 |
| 13 | ConsistI2V | TMLR 2024/OpenReview | E1 | identity/layout consistency |
| 14 | Motion-I2V | NVIDIA Research + ACM DOI | E1+E3 | 显式 motion field |
| 15 | CamCo | arXiv `2406.02509` | E2 | camera/3D 预印本 |
| 16 | STIV | ICCV 2025/CVF | E1 | text/image condition 与 rollout |
| 17 | OSV | CVPR 2025/CVF | E1 | 一步 I2V |
| 18 | RealCam-I2V | ICCV 2025/CVF | E1 | metric camera control |
| 19 | AnyI2V | ICCV 2025/CVF | E1 | 多模态条件图 + trajectory |
| 20 | Step-Video-TI2V | arXiv `2503.11251`/作者代码 | E2+E3 | 明确 `2C` 通道合同 |
| 21 | FramePack | arXiv `2504.12626`/作者代码 | E2+E3 | 长 rollout/context packing |
| 22 | MobileI2V | arXiv `2511.21475`/作者代码 | E2+E3 | 端侧 I2V |
| 23 | ReasonDiff | CVPR 2026/CVF | E1 | unpaired/OOD 锚点推理 |
| 24 | V-PAE | AAAI 2026 official proceedings | E1 | 单步蒸馏/条件坍塌 |
| 25 | CamGeo | ICML 2026/PMLR 306 PDF | E1 | 稀疏相机 3D 蒸馏 |
| 26 | DyMoS | arXiv `2605.19398`/project | E2+E3 | reference dominance 干预 |
| 27 | HPSD | arXiv `2608.13205`/project | E2+E3 | condition-state 蒸馏 |
| 28 | Ovi | arXiv `2510.01284`/project | E2+E3 | native AV 邻接前沿 |
| 29 | Animate Anyone | arXiv `2311.17117`/project | E2+E3 | 角色 animation 边界 |
| 30 | VACE | ICCV 2025/CVF | E1 | video editing/统一接口边界 |
| 31 | CameraCtrl | ICLR 2025/OpenReview | E1 | camera-conditioned T2V 边界 |
| 32 | Unified Text-Image-to-Video | arXiv `2505.20629` | E2 | training-free 多时刻视觉条件 |
| 33 | I2V-Adapter | arXiv `2312.16693` | E2 | adapter 路线 |
| 34 | LTX-2 | arXiv `2601.03233` / official `Lightricks/LTX-2` repository | E2+E3 | native AV 邻接前沿；冻结日推荐 LTX-2.5、LTX-2.3 为 legacy；仅带时间锚点的子协议才属于严格 I2V |

## 5. 首发日期与正式年份裁决

| 工作 | 首次公开 | 正式发表 | 裁决 |
|---|---|---|---|
| Scene Dynamics | 2016-09-08 | NeurIPS 2016 | 原型/思想祖先 |
| Forecast & Refine | 2018-07-26 | ECCV 2018 | 明确命名 I2V translation |
| cINNs | 2021-05-10 | CVPR 2021 | 随机 I2V |
| Make It Move | 2021-12-06 | CVPR 2022 | 文本控制 TI2V |
| LFDM | 2023-03-24 | CVPR 2023 | latent flow diffusion |
| LaMD | 2023-04-23 | IJCV 2025 | DOI `10.1007/s11263-025-02386-7` |
| DynamiCrafter | 2023-10-18 | ECCV 2024 | 开放域 animation/I2V |
| I2VGen-XL | 2023-11-07 | — | 技术报告 |
| SVD | 2023-11-25 | — | 技术报告 |
| Motion-I2V | 2024-01-29 | SIGGRAPH 2024 | DOI 正式核验 |
| ConsistI2V | 2024-02-06 | TMLR 2024-07 | OpenReview/TMLR 正式核验 |
| CamCo | 2024-06-04 | — | 预印本 |
| OSV | 2024-09-17 | CVPR 2025 | 一步 I2V |
| STIV | 2024-12-10 | ICCV 2025 | scalable text/image condition |
| RealCam-I2V | 2025-02-14 | ICCV 2025 | metric camera control |
| Step-Video-TI2V | 2025-03-14 | — | 技术报告 |
| FramePack | 2025-04-17 | — | 预印本 |
| AnyI2V | 2025-07-03 | ICCV 2025 | training-free 多模态条件图 |
| Ovi | 2025-09-30 | — | native AV 预印本 |
| MobileI2V | 2025-11-26 | — | 端侧预印本 |
| LTX-2 | 2026-01-06 | — | native AV 技术报告与开放 artifact |
| ReasonDiff | — | CVPR 2026 | 未猜测 arXiv v1 |
| V-PAE | — | AAAI 2026 | 正式页 2026-03-14 |
| DyMoS | 2026-05-19 | — | 预印本 |
| CamGeo | 2026-05-29 | ICML 2026 | 正式 PDF 标 PMLR 306 |
| HPSD | 2026-08-13 | — | 冻结日前最新训练策略预印本 |

## 6. 2026 前沿逐条证据边界

### 6.1 ReasonDiff：先推断时间锚点

CVPR 2026 正式页题名为 *Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation*。其目标不是普通 paired first-frame I2V：输入图可能与文本事件未配对，也不一定属于第 0 帧。VisionNarrator 产生逐帧 narrative 并推断 anchor position，AlignFormer 再用多阶段 temporal anchor attention 形成帧级 reasoning-enhanced latent。章节只把它视为“锚点推理”前沿，不与固定首帧模型横排。

### 6.2 CamGeo：训练期 3D 教师

ICML 2026/PMLR 306 论文首页和 arXiv `2605.30895` 均写明：稀疏 camera pose 下，用 VGGT 提供 keyframe trajectory cycle-consistency、跨帧 camera/depth consistency；教师只在训练时存在，推理不增加该教师开销。三阶段 curriculum 从基本连贯到全局 pose，再到细粒度 depth。该方法改善的是 sparse camera-conditioned I2V，不等于恢复可编辑完整 4D 场景。

### 6.3 DyMoS：reference-frame dominance 干预

arXiv `2605.19398` 把后续帧 query 对 reference-frame key 的过高 self-attention 称为 reference-frame dominance，并在早期去噪对对应 attention logits 加 scalar bias。其优势是 training-free/model-agnostic 的作者实验；章节避免转录跨 backbone 分数，也不把该机制外推到没有显式 reference-frame token 分区的所有网络。

### 6.4 HPSD：蒸馏中的 condition-state mismatch

arXiv `2608.13205` 的 teacher 使用 TI2V clean fixed first frame，而 T2V student 状态不同。hybrid policy 让 teacher 轨迹先行，将锚点重新加噪到 student-compatible state，走 student subtrajectory，再重施 clean first frame 回到 teacher 条件并监督后续帧。它是训练策略，不是一个新的 I2V 输出任务。

## 7. 评测协议裁决

### 7.1 不同协议不得横排

| 协议差异 | 为什么改变问题 |
|---|---|
| paired GT first frame vs unpaired/OOD image-text | 前者图像与真实视频天然匹配，后者要先推断事件和锚点 |
| 像素 hard anchor vs latent replacement vs soft condition | 首帧误差的下限和意义不同 |
| 14/16/25/32/102 帧、不同 fps | motion、FVD、时延和漂移不在同一时间尺度 |
| 单 clip vs autoregressive/chunk rollout | 后者引入历史压缩和 seam 误差 |
| 随机运动 vs 指定 point/camera/pose | “自然”与“遵循”是不同目标 |
| silent I2V vs audio-driven vs native AV | 后两者需要同步、声源和音色维度 |
| GPU report vs mobile/Core ML | 编解码、精度、内存和能耗边界不同 |

### 7.2 最低评测卡

正文采用五轴协议：anchor/identity、motion、temporal quality、condition adherence、long-horizon/system。每次比较必须冻结数据清单、分辨率、帧数、fps、锚点、prompt、控制坐标、sampler/steps/guidance/seeds、模型/VAE 版本、设备/精度和是否包含 I/O。主图应为 reference fidelity–motion Pareto，而非把静帧复制奖励成高总分。

## 8. AI 科学示意图生成与审计

### 8.1 技能与 fallback

本轮完整读取并遵循：

- `literature-review/SKILL.md` 及直接要求的 `references/database_strategies.md`；
- `scientific-schematics/SKILL.md` 及 `references/best_practices.md`；
- 内置 `imagegen/SKILL.md` 及 `references/prompting.md`、`references/sample-prompts.md`。

`scientific-schematics/references/diagram_types.md` 在技能目录中不存在；按技能指引记录缺失，并回退到 `best_practices.md` 的数据流/分层框图原则：关系优先、少文字、颜色不是唯一编码、图注与顺序化替代文本补全语义。

### 8.2 初始提示词与独立审计

> Create a publication-quality scientific infographic diagram on a clean white 16:9 canvas, landscape orientation, titled exactly “I2V CONDITION CONTRACT”. Flat vector style, crisp high-contrast lines, generous whitespace, no logos, no model names, no benchmark scores, no watermark, no tiny decorative text. Use a colorblind-safe Okabe–Ito-inspired palette that remains distinguishable in grayscale. Left column: five compact input cards with simple icons and exact labels: “REFERENCE FRAME”, “TEXT”, “MOTION”, “CAMERA”, “AUDIO”. Make REFERENCE FRAME visually mandatory with a solid arrow; the other four optional with dashed arrows. Center: a clear three-stage vertical processing stack with exact labels “ANCHOR”, “MOTION PRIOR”, “VIDEO DENOISER”. Show three distinct injection routes: the reference frame locks the first film frame and feeds ANCHOR; motion and camera feed MOTION PRIOR; text and audio feed VIDEO DENOISER. Use simple arrows, not equations. Right: a horizontal filmstrip of five frames showing the same neutral geometric subject (an orange paper bird or abstract origami bird) moving naturally across a minimal scene. Frame 1 has a small padlock icon and must visibly match the reference; later frames change pose and reveal plausible new regions while preserving subject color and identity. Bottom: four verification gates in a single row with exact labels “PRESERVE”, “MOVE”, “OBEY”, “ROLL OUT”, each with a small check-shaped icon but no numeric scores. Connect the generated filmstrip down to these gates. Ensure every label is spelled exactly, large and readable. Avoid photorealistic people, brands, dense prose, pseudo-code, charts with fake numbers, 3D rendering, gradients that reduce grayscale contrast, and visual clutter.

独立审计没有只检查文字美观，而是检查每条箭头：初版项目图把 `MOTION PRIOR` 直接连到 filmstrip、`VIDEO DENOISER` 没有输出，并残留一条无来源蓝色虚线，因此判为科学数据流 P1，不能靠图注豁免。最终修订要求形成 `ANCHOR + CONTROL → MOTION PRIOR → VIDEO DENOISER → filmstrip → acceptance gates`，已知音频作为可选输入进入 denoiser。

### 8.3 生成迭代、尺寸与文件完整性

- 生成器：OpenAI 内置 image generation tool，2026-08-30。
- 初版 artifact：`generated_images/01a04ef9-a290-72c0-8a4e-9ec87d5e30cf/exec-eafdc882-e96b-46fc-a52d-164334ba305f.png`；因上述三处箭头关系被独立审计拒绝。
- 中间修订 `exec-702470a6...`、`exec-02e612b6...`、`exec-5bc4f4d6...`、`exec-37b07bda...` 分别暴露无输出箭头、错误直连、缺失 reference 路径或条件汇合标签不准确，均未作为项目资产。
- 接受版 artifact：`generated_images/01a04c93-4978-7ad2-9956-339854046832/exec-b193843d-2f2c-4d78-8378-12f38991c21a.png`。
- 项目文件：[`assets/diagrams/image-to-video-conditioning-contract.png`](../assets/diagrams/image-to-video-conditioning-contract.png)。
- 最终尺寸/格式：1672 × 941（约 16:9），PNG，8-bit RGB，1,293,769 bytes；不声称嵌入 ICC/sRGB profile。
- SHA-256：`ade7d4ea1f9983eff37e0412fbd98a7e20065a6175a4034303813c7e5ed67889`。

### 8.4 原图与灰度检查

最终 PNG 已用原始分辨率回读，并另生成临时 grayscale 版本检查：

- 标题与 12 个短标签均拼写正确、无截断、无伪造数值、品牌或水印；
- reference frame 用实线进入 `ANCHOR + CONTROL`，可选文本/运动/相机用虚线汇合，已知音频单独进入 denoiser；灰度下仍可辨；
- 因果主链为 `ANCHOR + CONTROL → MOTION PRIOR → VIDEO DENOISER → filmstrip`，没有 denoiser 旁路或无来源箭头；
- 五帧纸鸟保持颜色/轮廓身份并产生姿态和背景变化；
- `PRESERVE / MOVE / OBEY / ROLL OUT` 四门在彩色与灰度中都清楚；
- `ANCHOR + CONTROL` 是合同层汇合符号而非声称存在统一模块；正文 Mermaid 保留具体的 concat、cross-attention、control residual、warp 与 audio 路径。

## 9. 综合判断与剩余不确定性

### 9.1 可高置信陈述

- Vondrick 2016 是单图未来合成原型；Zhao 2018 才是更合适的命名 I2V 任务里程碑。
- 图像“作为条件”不自动等于首帧像素硬锚定；必须区分 pixel/latent/soft 三种合同。
- DynamiCrafter 的双图像路径、SVD/Step-Video 的 channel concat、ConsistI2V 的 first-frame attention/低频初始化、Motion-I2V 的显式 flow 是不同注入/运动机制。
- reference fidelity 与 motion amplitude/correctness 存在结构性冲突，静态复制不能因高相似度被评为成功。
- RealCam-I2V 与 CamGeo 改善 camera/geometry 条件，但都不能据此宣称通用 4D 世界一致性已解决。
- FramePack 支持受限上下文的长 rollout；endpoint planning 使用未来信息，不能称严格在线。
- Ovi/LTX-2 的 native AV、Animate Anyone 的角色动画、VACE 的编辑和 CameraCtrl 的纯相机 T2V 都只在满足锚点子协议时属于 I2V。

### 9.2 保留边界

- 2026 预印本 DyMoS、HPSD 和 2025 MobileI2V/Ovi 尚需正式版与独立复现。
- 自动身份、动态度和视频质量指标都依赖特征模型、clip 长度、crop 和样本量；本章未创建新的跨论文总分。
- 公开大规模 I2V 数据的训练近重复审计仍不足；高 reference similarity 可能部分来自记忆。
- native AV + strict I2V、长时对象永久性、多参考冲突和可编辑 4D 仍没有统一 benchmark。

## 10. 交付前验证记录

2026-08-30 冻结候选已完成以下实际检查：

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Markdown lint | `npx --yes markdownlint-cli2` 对正文和本日志执行 | **PASS**，0 issue |
| 引用锚点 | 脚本核对 `href=#ref-*`、`<a id=...>`、重复/缺失/未使用 | **PASS**，34/34 定义均被使用；0 缺失、0 重复、0 未使用 |
| 相对链接 | 解析两个 Markdown 的本地 links，按各自目录 resolve | **PASS**，0 broken relative link |
| 外部一手入口 | 对正文 36 个唯一参考 URL 检查，跟随 redirect；GitHub 页面同时用 API/Web 入口复核 | **PASS**，原 35 个入口均返回 2xx；新增 LTX-2 官方仓库经 GitHub API/Web 确认存在（直接页面请求出现一次瞬时 TLS 失败）；0 missing |
| Mermaid 可访问性 | 统计 block、`accTitle`、`accDescr` 与顺序化替代文本 | **PASS**，2/2 block 均有 title/description；正文另有逐步替代 |
| Mermaid 真渲染 | Mermaid CLI 11.16.0 + 本机 Chrome，从 stdin 分别渲染 SVG，并以 1600 px PNG 回读 | **PASS**，2/2 非空 SVG（28,702 / 41,640 bytes）；边界树与条件注入图无裁切、标签可读 |
| PNG 结构 | `file`、尺寸/模式/格式、SHA-256 | **PASS**，1672×941、RGB、PNG、1,293,769 bytes，SHA 与记录一致 |
| PNG 灰度 | 原图与 grayscale 原尺寸目视 | **PASS**，标题、所有标签、实/虚线、因果箭头和四道门均可辨 |
| whitespace | 仓库级与三交付物限定 `git diff --check` | **PASS**，0 whitespace error |

Mermaid 临时 SVG 与 grayscale PNG 位于系统临时目录，不属于交付物。日志回填后已再次运行 lint 与 `git diff --check`，结果仍为 0 issue / 0 whitespace error。
