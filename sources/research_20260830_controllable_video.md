# 细粒度可控视频生成研究记录：检索、证据、发布面与图片审计

> 检索冻结日：**2026-08-30（Asia/Shanghai）**
>
> 对应章节：[`docs/tasks/controllable-video-generation.md`](../docs/tasks/controllable-video-generation.md)
>
> 记录目的：把控制信号、注入位置、任务边界、发表状态、开放 artifact、作者自报结果和最终图片分别留痕，避免将“有条件”“有 demo”“有论文”和“可复现”混成一个结论。

## 1. 研究问题与范围

本轮围绕七个可判定问题展开：

1. 哪些显式信号足以构成细粒度控制，而不只是文本或首帧条件？
2. camera、object trajectory、结构序列、identity/reference 和 multi-control 分别需要什么坐标、时间、可见性和冲突合同？
3. 条件经 training-time adapter/ControlNet、attention/feature injection、latent optimization 或 inference guidance 进入模型时，各自可支持什么强度的主张？
4. 可控生成与 I2V、V2V editing、story/multishot、action-conditioned prediction/world model 的输入时间轴和保真对象如何区分？
5. 2025–2026 的 2D→3D、camera/world-time 解耦、few-step 和 online 4D frontier，哪些已正式发表，哪些仍是预印本作者协议？
6. 控制遵循、参考保真、运动自然性和条件多样性为何必须分开评测？
7. 遮挡、身份交换、控制冲突、漂移和新视角失败如何被一个可复现协议捕获？

纳入范围是：以可观察、可改变、可单独测量的时空信号控制视频生成或重生成的工作。只有文本条件的通用 T2V 不纳入；I2V、V2V、角色动画和 world model 只在显式控制合同或任务边界处出现。正文不汇总跨协议排行榜，也不把产品宣传数字改写成独立事实。

## 2. 检索设计

### 2.1 互补的一手来源

| 来源类型 | 本轮用途 | 能支持什么 | 不能单独支持什么 |
|---|---|---|---|
| arXiv API/摘要/PDF | 冻结首发日期、版本、预印本方法和作者实验 | v1 时间、作者、题名、预印本协议 | 正式接收、独立复现、长期可用性 |
| CVF、PMLR、NeurIPS、ECVA、OpenReview、ACM、AAAI 正式页/全文 | 核验 venue、正式年份与关键机制 | 同行评审发表状态、论文明确写出的模型/实验 | 代码和权重真的可下载 |
| 作者/机构官方 GitHub 与项目页 | 核查代码、checkpoint、数据/benchmark、运行入口 | 冻结日的 release surface | 第三方复现、未来维护状态 |
| Adobe、Kling、Runway 官方帮助页 | 核验真实产品输入合同 | 冻结日产品入口和输入限制 | 底层用了哪篇论文、学术可复现性 |
| OpenAlex | 题名覆盖和漏项检查 | 宽检索发现候选 | 方法结论、唯一论文数 |

搜索引擎摘要、二手综述和聚合榜只用于发现候选。章节中的强断言均回到上述一手来源；仓库 README 的速度或质量数字仍标作作者协议。

### 2.2 精确检索式、日期与原始计数

以下 API 查询执行于 2026-08-30。计数是冻结日原始返回量，查询高度重叠，**不能相加，也不是 PRISMA 纳入数**。

| 通道 | 精确检索式 | 原始结果数 | 用途/裁决 |
|---|---|---:|---|
| arXiv API | `all:"controllable video generation"` | 156 | 主发现池；含宽泛“可控”用法 |
| arXiv API | `(all:"video generation") AND (all:trajectory OR all:"camera control" OR all:pose OR all:depth OR all:flow)` | 1,246 | 控制信号补漏；噪声很高 |
| arXiv API | `all:"camera-controllable video generation"` | 26 | 相机子路线核对 |
| OpenAlex | `title.search=controllable video generation`, `from_publication_date=2022-01-01`, `to_publication_date=2026-08-30` | 417 | 题名覆盖检查 |
| OpenAlex | 同日期范围，`title.search=trajectory controllable video generation` | 25 | 轨迹子路线检查 |
| OpenAlex | 同日期范围，`title.search=camera controllable video generation` | 75 | 相机子路线检查 |

定向核验又覆盖以下词组：`ControlNet video`、`point trajectory video generation`、`box motion video generation`、`mask-guided video generation`、`pose-guided video generation`、`camera pose video generation`、`3D trajectory video generation`、`multi-control video generation`、`few-step trajectory`、`online 4D control`。2026 work 逐篇回到 CVPR 2026 正式页或 arXiv v1，而不是从搜索结果标题推断发表状态。

### 2.3 纳入、排除与去重

纳入条件：

- 方法输入至少包含一类可在输出上单独测量的显式时空控制；
- 能从正式全文、预印本全文或作者 artifact 核验信号表示、注入位置、训练/推理协议或 release surface；
- 2026 新工作可以只有预印本，但正文必须显式写“预印本”和“作者报告”；
- 邻接任务只在帮助划分 I2V/V2V/story/action/world-model 边界时保留；
- 同一工作合并 arXiv 与正式版：首发取 arXiv v1，正式状态取 proceedings。

排除条件：

- 只有文本 prompt、没有可单测的空间/时间/外观控制；
- 只有精选 demo 或二手博客，不能确认输入合同；
- 只生成单图，或只在 related work 中提到视频控制；
- 产品页面没有明确说明输入类型；
- 截止冻结日之后才出现的版本、代码或权重。

最终章节引用 **45 条一手记录**：32 个正式发表论文入口、3 个预印本入口、3 个官方产品文档和 7 个官方 release surface。后 7 条与论文实体重合，因此不能说成 45 篇论文；论文实体是 35 个，其中 32 个已正式发表、3 个在冻结日仅为预印本。

## 3. 证据等级与陈述规则

| 等级 | 定义 | 正文允许的写法 |
|---|---|---|
| **E1** | 正式 proceedings/期刊全文或官方论文页 | 正式 venue、明确机制、作者实验及其协议边界 |
| **E2** | arXiv/技术报告全文 | 预印本方法与作者报告；不可冒充同行评审或独立复现 |
| **E3** | 作者/机构项目页、官方代码/权重/数据仓库 | 冻结日页面明确提供的 artifact |
| **E4** | 官方产品帮助页 | 冻结日产品输入合同；不推断底层架构 |
| **D** | OpenAlex/搜索摘要 | 只发现候选，不独立进入强断言 |

发布面拆成五个字段：`paper`、`code`、`weights`、`data_or_benchmark`、`product_or_demo`。任何一个字段为 `yes` 都不能推断其他字段。作者自报的 SOTA、速度、显存和“实时”必须和分辨率、帧数、硬件、采样步数、是否含 VAE/I/O 绑定。

## 4. 任务边界裁决

| 邻接任务 | 本轮判据 | 纳入/排除示例 |
|---|---|---|
| I2V | 图像位于输出时间轴或充当身份/外观锚点；额外相机/轨迹/姿态才构成细粒度控制 | Animate Anyone、MotionPro、FlexTraj 作为受控 I2V；首帧相似不能证明轨迹有效 |
| V2V editing | 完整源视频已给定，需守住未编辑区域和原时序 | ReCapture、Generative Video Motion Editing with 3D Point Tracks 按 V2V 守恒合同验收 |
| story/multishot | 多镜头状态、角色/道具关系与切镜是主合同 | 单镜头 camera/trajectory 成功不能证明跨镜头一致 |
| action-conditioned prediction | 历史观测 + 环境可执行 action，输出代表动作后果 | EgoControl 给 3D pose，不等同机器人/游戏 action |
| interactive world model | 在线 action–observation–memory 闭环和持久状态 | 4DStreamCtrl 的流式作者报告不自动证明可玩闭环或反事实 dynamics |

## 5. 控制信号与注入位置的证据矩阵

### 5.1 控制信号

| 信号路线 | 代表一手来源 | 已核验转折 | 必须保留的边界 |
|---|---|---|---|
| camera extrinsics/intrinsics | MotionCtrl、CameraCtrl、GEN3C、UCPE、FaceCam、BulletTime、WorldStereo | pose adapter；显式 3D cache；相对/绝对方位编码；人像尺度；world-time 解耦；几何记忆 | `pan/zoom` 文本不等于 metric path；必须声明坐标系、尺度与内参 |
| point/drag/box/mask | DragAnything、Boximator、Peekaboo、Motion Prompting、MotionPro、Image Conductor、IM-Zero | entity drag；hard/soft box；masked attention；通用 motion prompts；实例级零样本控制 | 点/框不含完整 3D、可见性和实例连续性 |
| 3D trajectory | 3DTrajMaster、LAMP、4DStreamCtrl | 多对象 6DoF；语言→显式 3D program；统一 camera/object/depth 的 3D point tracks | 合成监督、深度误差和作者协议分别记录 |
| pose/depth/normal/edge/flow | Control-A-Video、ControlVideo、MOFA-Video、Animate Anyone、PoseAnything | video ControlNet；复用 image ControlNet；motion field adapters；reference/pose 分离；part-aware pose coherence | 各结构丢失的信息不同，不能合成一个“structure score” |
| identity/appearance | Animate Anyone、MOFA-Video、FaceCam | ReferenceNet/外观路径与姿态路径拆分；多 adapter；人像尺度控制 | 全帧 CLIP 高可来自静帧复制或背景复制 |
| multi-control | VideoComposer、MotionCtrl、MOFA-Video、VACE、LAMP | 统一编码、分支组合、adapter 组合、统一任务接口、DSL 编译 | “同时接收”不证明冲突时同时遵循 |

### 5.2 注入位置

| 注入族 | 代表来源 | 证据要点 | 复现时的最小消融 |
|---|---|---|---|
| training-time adapter / ControlNet residual | ControlNet、Control-A-Video、MagicMotion、FlashMotion | 冻结主干；可训 residual/adapter；few-step backbone 需重新对齐 | 去 adapter；扫 residual scale；固定 backbone 比 slow/fast adapter |
| attention / feature injection | Tora、3DTrajMaster、VideoComposer、MOFA-Video | trajectory tokens/patches、gated self-attention、STC features、motion field adapters | 交换 object ID/KV；打乱 mask；查运动泄漏 |
| latent/noise optimization | Text2Video-Zero、FreeTraj | 改初始 latent/noise 和跨帧 attention；无需训练新视频主干 | 固定 seed 比原噪声/导引噪声；同时画质量和多样性 |
| inference guidance / CFG | Peekaboo、PoseAnything、WorldForge | masked attention 或解耦 CFG anchors；零样本 camera manipulation | 完整 guidance-scale 曲线；测试过强时锐化/静态化/坍塌 |

## 6. 发表状态与里程碑核验

### 6.1 正式发表与预印本

| 工作 | 首次公开/正式状态（冻结日） | 证据入口 | 裁决 |
|---|---|---|---|
| ControlNet | ICCV 2023 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | 图像控制祖先，不能直接证明视频时序 |
| Text2Video-Zero | ICCV 2023 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Khachatryan_Text2Video-Zero_Text-to-Image_Diffusion_Models_are_Zero-Shot_Video_Generators_ICCV_2023_paper.html) | training-free latent/attention 转折 |
| VideoComposer | NeurIPS 2023 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html) | 多条件统一接口 |
| Control-A-Video | arXiv `2305.13840`，冻结日只按预印本 | [arXiv](https://arxiv.org/abs/2305.13840) | 不写成正式会议论文 |
| MotionCtrl | SIGGRAPH 2024 | [ACM DOI](https://doi.org/10.1145/3641519.3657518) | camera/object 解耦里程碑 |
| Boximator | ICML 2024 | [PMLR](https://proceedings.mlr.press/v235/wang24cr.html) | hard/soft box |
| Peekaboo | CVPR 2024 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jain_PEEKABOO_Interactive_Video_Generation_via_Masked-Diffusion_CVPR_2024_paper.html) | training-free masked attention |
| DragAnything、MOFA-Video | ECCV 2024 | [ECCV poster](https://eccv.ecva.net/virtual/2024/poster/1397)、[ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02842.pdf) | entity drag / motion-field adapter |
| CameraCtrl、3DTrajMaster | ICLR 2025 | [OpenReview CameraCtrl](https://openreview.net/forum?id=Z4evOUYrk7)、[OpenReview 3DTrajMaster](https://openreview.net/forum?id=Gx04TnVjee) | pose adapter / 多对象 6DoF |
| Tora、Motion Prompting、MotionPro、GEN3C、ReCapture、IM-Zero | CVPR 2025 | 各自 [CVF 2025 proceedings](https://openaccess.thecvf.com/CVPR2025?day=all) 正式页 | 轨迹表示、相机几何与 V2V 边界分别记录 |
| MagicMotion、VACE | ICCV 2025 | 各自 [CVF 2025 proceedings](https://openaccess.thecvf.com/ICCV2025?day=all) 正式页 | dense-to-sparse curriculum / creation-editing interface |
| FlexTraj、FlashMotion、LAMP、BulletTime、UCPE、WorldStereo、PoseAnything、FaceCam、EgoControl、3D Point Tracks editing、WorldForge | CVPR 2026 | 各自章节参考文献中的 CVF 正式页 | 均按正式发表；不由年份推断 artifact 已开放 |
| FreeTraj | arXiv `2406.16863` | [arXiv](https://arxiv.org/abs/2406.16863) | tuning-free 方法按预印本 |
| 4DStreamCtrl | arXiv v1 `2608.25479`，2026-08-26 | [arXiv](https://arxiv.org/abs/2608.25479) | 最新 frontier，但冻结日不写成正式发表或已复现 |

### 6.2 Release surface 冻结快照

| 工作 | 代码 | 权重 | 数据/评测 | 冻结日边界 |
|---|---|---|---|---|
| MotionCtrl | yes | yes，多主干入口 | 配置/示例 | [官方仓库](https://github.com/TencentARC/MotionCtrl)；复现必须锁具体 backbone/commit |
| Tora | yes | yes | 示例/评测入口 | [官方仓库](https://github.com/alibaba/Tora) |
| BulletTime | yes | yes | dataset/tool 入口 | [官方仓库](https://github.com/19reborn/BulletTime) 提供 camera/time editing tools |
| FlexTraj | yes | checkpoint 下载入口 | FlexBench 入口 | [官方仓库](https://github.com/bestzzhang/FlexTraj_code) |
| FlashMotion | training/inference/eval | model weights 入口 | benchmark 入口 | [官方仓库](https://github.com/quanhaol/FlashMotion)，发布说明含 2026-03-13 更新 |
| WorldStereo | WorldStereo 2.0 code | yes | data preprocessing 仍有 TODO | [官方仓库](https://github.com/FuchengSu/WorldStereo)，不能写成全链路一键复现 |
| LAMP | local setup/Gradio | model downloads | — | [官方仓库](https://github.com/mbkizil/LAMP/) 的 client inference 仍标 coming soon |
| 4DStreamCtrl | coming soon | not exposed | not exposed | [项目页](https://4dstreamctrl.github.io/) 在冻结日仍标 paper/code coming soon；arXiv 摘要另已公开 |

### 6.3 作者自报与独立事实分栏

- **独立可核验：** CVPR/ICLR/ICML/ICCV/SIGGRAPH 的正式论文入口；仓库中是否存在训练、推理、checkpoint 或数据链接；产品帮助页是否明确描述输入。
- **作者实验但未独立复现：** 4DStreamCtrl 报告 480p、20 FPS、单高端 GPU 和数百帧一致；FlashMotion/其他 few-step 工作的速度、显存和质量结果；任何仓库 README 的 SOTA 声明。
- **本章不作的推断：** NFE 少等于端到端实时；代码按钮存在等于完整训练链路可复现；在线 demo 等于公开权重；流式视频等于持久 world model。

## 7. 产品控制接口快照

| 产品 | 冻结日官方合同 | 一手来源 | 不推断的内容 |
|---|---|---|---|
| Adobe Firefly | 5–10 秒 motion-reference video，提取 pan/zoom/tilt/path 等相机运动 | [官方帮助页](https://helpx.adobe.com/firefly/web/work-with-audio-and-video/work-with-video/match-camera-motion-to-reference-video.html)，更新 2026-06-09 | 不推断内部使用 pose adapter 或 3D cache |
| Kling VIDEO 3.0 Motion Control | 角色参考图 + 驱动视频/动作库，主要复制动作和表情 | [官方指南](https://kling.ai/quickstart/motion-control-user-guide)，发布 2026-03-05 | 不泛化成任意对象 6DoF 控制 |
| Runway Gen-4 | I2V 入口，文本描述主体、场景与相机运动；官方指南说明 5/10 秒生成 | [官方指南](https://help.runwayml.com/hc/en-us/articles/39789879462419-Gen-4-Video-Prompting-Guide) | 不把文本 camera words 当 metric camera trajectory |

产品状态会改变；以上只证明 2026-08-30 回读到的输入入口，不证明之后仍开放，也不作为学术 benchmark。

## 8. 评测与复现裁决

### 8.1 四个目标不可合并

1. `control adherence`：按 camera/track/box/mask/pose/depth/flow/identity 各自定义误差；
2. `fidelity`：参考身份、外观、未编辑区域和历史状态；
3. `motion naturalness`：速度/加速度、接触、形变、穿透、时间稳定；
4. `conditional diversity`：只在同一条件且先通过控制阈值的有效样本内计算。

美学总分、CLIP-I、FVD 或单一 tracking error 都不能同时代替上述四项。

### 8.2 必须单列的困难子集

- **遮挡/re-entry：** visible ADE/FDE、身份恢复、出画重返；
- **新视角：** pose error、loop closure、geometry consistency、disocclusion；
- **冲突：** camera/object 相抵、两个对象争同一空间、pose/mask 不一致、相互矛盾的 reference；
- **长时漂移：** 每帧曲线和 chunk seam，不只看首末帧平均；
- **少步/流式：** latency 拆分 encoder、denoising、VAE、I/O，记录 warm-up 和 batch。

### 8.3 最小复现字段

`model/commit/checkpoint`、`VAE/encoders`、`F×H×W/fps`、坐标系/尺度/内参、轨迹插值/visibility、sampler/steps/guidance schedules、seed list、是否重采样/精选、GPU/精度/显存/计时边界、控制来源及估计器版本。

## 9. 图片生成与视觉审计

### 9.1 语义边界

最终图只表达五步合同：`SIGNALS → CONTRACT → INJECTION → GENERATOR → EVALUATE`。四类信号不得绕过合同；虚线反馈只表示修改控制/测试规格，不表示自动回传训练。`CONTROL`、`FIDELITY`、`MOTION`、`DIVERSITY` 分开，不画单一总分。正文提供等价 Mermaid、`accTitle`、`accDescr` 和六步顺序化文字替代。

### 9.2 生成与修订 prompt

最终构图的基准生成 prompt（原样记录）：

```text
Use case: scientific-educational.
Asset type: one clean 16:9 landscape PNG infographic for an advanced textbook chapter on fine-grained controllable video generation.

Create exactly one left-to-right five-column scientific diagram on an opaque white background. The full diagram, including every arrow and feedback line, must fit within a generous SAFE AREA: leave at least 8 percent blank white margin on the left and right and at least 7 percent blank white margin on the top and bottom. No colored stroke, arrowhead, loop, card, or text may touch or cross the safe-area boundary.

Five column headings, in order, using the exact uppercase labels only:
1 SIGNALS
2 CONTRACT
3 INJECTION
4 GENERATOR
5 EVALUATE

Under SIGNALS, show four vertically stacked cards with exact labels: CAMERA, TRACKS, STRUCTURE, IDENTITY. Use simple icons: camera axes/path; visible and occluded point trajectories; pose/depth/edge/flow; multi-view person/appearance swatches.

Every one of the four signal cards must connect fully and clearly to CONTRACT. CONTRACT uses icons, not extra text, to convey shared coordinates, timestamps, masks, confidence, and conflict resolution. No signal may bypass CONTRACT.

CONTRACT connects to three vertically stacked paths under INJECTION, with exact labels: ADAPTER, ATTENTION, GUIDANCE. Then all three connect to GENERATOR, depicted as a video filmstrip. GENERATOR connects to four separate evaluation cards under EVALUATE, with exact labels: CONTROL, FIDELITY, MOTION, DIVERSITY. Use no single overall score.

Add one dashed feedback arrow returning from EVALUATE to CONTRACT along the bottom, completely inside the safe area with substantial white space below it. This feedback means revise the control specification, not model training.

Text constraints: render only these exact words and no others: SIGNALS, CONTRACT, INJECTION, GENERATOR, EVALUATE, CAMERA, TRACKS, STRUCTURE, IDENTITY, ADAPTER, ATTENTION, GUIDANCE, CONTROL, FIDELITY, MOTION, DIVERSITY.

Visual style: crisp flat vector-like scientific diagram, restrained Okabe-Ito-inspired colors, high contrast, no gradients, no 3D gloss, no photorealism, no decorative flourishes. Accessible and grayscale-readable through redundant shapes, line styles, and direct labels. Large readable type, unambiguous arrowheads, no crossing arrows, no clipped labels, no logos, no watermark, no figure number. Keep the layout airy and compact enough that all content is comfortably inside the margins.
```

基准图出现 `1–5` 标号与 `STRUCTURA` 后，使用以下仅文字修订 prompt（原样记录）：

```text
Make only these precise text corrections to this existing five-column scientific infographic. Preserve the entire composition, safe white margins, arrows, icons, color palette, and dashed bottom feedback line exactly as they are.

1. Remove the numeral and following space before each of the five column headings. The headings must read exactly: SIGNALS, CONTRACT, INJECTION, GENERATOR, EVALUATE.
2. Correct the misspelled card label STRUCTURA to exactly STRUCTURE.
3. Keep all other labels exactly as currently shown: CAMERA, TRACKS, IDENTITY, ADAPTER, ATTENTION, GUIDANCE, CONTROL, FIDELITY, MOTION, DIVERSITY.
4. Do not add any text, numbers, logos, watermark, title, caption, or figure number.
5. Keep at least 8 percent blank margin left/right and 7 percent top/bottom. Every line and arrowhead must remain fully inside the safe area. Do not crop anything.

This is a text-only correction; do not redesign the diagram.
```

### 9.3 版本、拒绝理由与最终校验

| 版本 | 尺寸 | SHA-256 | 裁决 |
|---|---:|---|---|
| v0 原始五栏图 | 1672×941 | `c4ee21144ca4f83f3f58908dbef7bf7e220421243ceaea521f63199879aca565` | 拒绝：左/右箭头及底部反馈线贴边或被裁，透明边缘显截断 |
| v1 路径编辑尝试 | 1672×941 | `9f59677d6fdb3843a7b47a09841169e914a24eb381f7be0961e267c4ab81b7b6` | 拒绝：错误生成成 audio–video contract，语义完全偏离 |
| v2 安全留白重绘 | 1672×941 | `8cddb68d1c876f149dd174b232d9fba6260636769ebde1ff36ef88a11200bdfd` | 拒绝：结构/留白合格，但多出编号且 `STRUCTURE` 拼成 `STRUCTURA` |
| **v3 最终** | **1672×941** | **`1eafab73c4f0bb63a752e815215bdc4c7b496afc6fbd3d12c194947f2f802785`** | 接受：标签正确、所有箭头在安全区、无越界/水印/交叉箭头 |

最终资产：[`assets/diagrams/controllable-video-contract.png`](../assets/diagrams/controllable-video-contract.png)。

- **原图检查：** 1672×941、sRGB、16:9 近似比例；五栏顺序清楚；左侧四条输入、右侧四个验收箭头和底部虚线反馈均完整入画；无贴边、无裁切、无错拼、无多余文字。
- **灰度检查：** 用 ImageMagick `-colorspace Gray` 生成临时副本并以 original detail 回读；边框深浅、虚实线、形状和直接标签均可区分，语义不依赖红/绿颜色。
- **可访问替代：** Markdown `alt` 给出一句话语义；Mermaid 含 `accTitle`/`accDescr`；正文另有六步有序文字替代。

## 10. 技能与过程边界

本轮按顺序使用并完整阅读 `literature-review`、`scientific-schematics` 与系统 `imagegen` 技能；读取了 literature-review 的数据库策略、scientific-schematics 的最佳实践，以及 imagegen 的 prompting/sample-prompts。scientific-schematics 指向的 `references/diagram_types.md` 在本机技能目录不存在，因此没有伪称已读取；改以其余出版/可访问性规则、PNG + Mermaid + 顺序化替代完成等价约束。

## 11. 限制

1. 原始 API 计数是检索宽度，不是唯一论文数，也没有声称阅读全文 1,246 个结果。
2. 正式论文的作者实验仍可能依赖私有数据、闭源 base model 或未公开预处理；formal publication 不等于全链路复现。
3. GitHub 和产品页是 2026-08-30 快照；后续 release 需重新回读。
4. 4DStreamCtrl 是冻结日前四天发布的 v1；所有性能数字严格停留在作者协议层。
5. 本章提出的 `ControlContract-120` 是可证伪复现设计，不是已经跑完的新 benchmark。

## 12. 最终验收记录

冻结版本的实际检查结果如下：

- `npx --yes markdownlint-cli2 docs/tasks/controllable-video-generation.md sources/research_20260830_controllable_video.md`：markdownlint-cli2 `v0.23.2`，**0 issues**。
- 引用闭环脚本：正文共有 98 次 callout、45 个唯一编号；文末 45 条唯一 reference，编号配对正确，**0 missing、0 orphan**。
- 本地链接检查：两份文档合计 9 个相对链接，目标均存在。
- 外链检查：共 48 个唯一 HTTPS URL。并发 GET 中 38 个返回 HTTP 200；ACM DOI 与 Runway 因站点策略返回 403，7 个 GitHub 与 Adobe 在该 shell 通道超时。随后用浏览器读取层复核 7 个 GitHub、Adobe 和 Runway 均能打开；MotionCtrl DOI 又由 Crossref 返回同题名、ACM publisher 和同一 DOI。未发现确定性 404。
- Mermaid：从正文提取代码块，使用 `@mermaid-js/mermaid-cli` 和本机 Chrome 实际渲染出 37,343-byte SVG；SVG 含 `<title>`、`<desc>`、`aria-labelledby` 与 `aria-describedby`，语法和可访问描述均通过。
- 图片：最终 PNG 为 1672×941、sRGB，SHA-256 为 `1eafab73c4f0bb63a752e815215bdc4c7b496afc6fbd3d12c194947f2f802785`；原图与灰度 original-detail 视觉检查均通过。
- whitespace/diff：`git diff --check --no-index /dev/null <file>` 对两个新增 Markdown 文件均无诊断；限定三条目标路径的 `git diff --check` 无诊断。
