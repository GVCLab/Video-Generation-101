# 视频修复与补全：截至 2026-08-30 的检索、证据与图像审计

> 冻结时间：2026-08-30（Asia/Shanghai）
> 对应正文：[`docs/tasks/video-inpainting.md`](../docs/tasks/video-inpainting.md)
> 对应图像：[`assets/diagrams/video-inpainting-evidence-pipeline.png`](../assets/diagrams/video-inpainting-evidence-pipeline.png)
> 研究类型：机制导向的 scoping literature review；不是穷尽式 meta-analysis，也不是跨协议排行榜

## 📋 1. 改写对象与研究问题

### 1.1 改写前快照

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `docs/tasks/video-inpainting.md` | 61 | `c1a2458d4b81a6de4e1037df4edc9de35515510bed88ecf472c1a3293c14c84b` |

改写前只有 8 条参考文献、0 张图，直接从 1981 optical flow 跳到 2024–2025 diffusion / DiT；未解释 STTN、FuseFormer、E²FGVI、ProPainter，也没有 mask 外保护、数据/mask 协议和 evidence-aware milestone。覆盖审计把该页列为 P0、depth 1。

### 1.2 本次研究问题

1. video inpainting、object removal、completion / outpainting、V2V editing 和 generative fill 的验收合同分别是什么？
2. valid-pixel propagation、residual-hole synthesis、global temporal consistency 与 outside-mask protection 如何组成可检验的 tensor pipeline？
3. patch、flow、alignment、non-local attention、STTN、FuseFormer、E²FGVI、ProPainter、diffusion 与 Video DiT 的技术转折是什么？
4. 2024–2026 的 direct-task 前沿如何与通用视频基础模型、产品能力和邻近 outpainting 分开？
5. 对象移除怎样从轮廓内补全扩展到影子、反射、光、透明效应和物理交互后果？
6. DAVIS / YouTube-VOS、DEVIL、VPBench、WIPER / ROSE / PROVE 等证据面怎样组合，哪些指标会系统性误导？
7. 长视频的 overlap、memory、identity、scene cut、codec round-trip 和成本应怎样报告？

### 1.3 邻章所有权

- [`video-to-video.md`](../docs/tasks/video-to-video.md) 拥有更一般的 instruction editing、global/local V2V、source fidelity 与多轮编辑；
- [`frame-interpolation.md`](../docs/tasks/frame-interpolation.md) 拥有已知前后端点之间的时间插帧；
- [`foundation-models.md`](../docs/foundation-models.md) 拥有模型家族、产品/checkpoint、开放面与完整现代系统栈；
- 本页只负责带 hole/edit support 的时空补全、对象移除、扩画幅邻接边界及其专用证据协议。

## 🔎 2. 检索表面、日期与实际返回量

### 2.1 表面 A：arXiv API

2026-08-30 使用 `export.arxiv.org/api/query`。以下数字来自 `opensearch:totalResults`；查询彼此重叠，不可相加为 PRISMA 总数。

| 检索式 | 返回量 | 用途 |
|---|---:|---|
| `all:"video inpainting"` | 156 | 全历史候选与同义词发现 |
| `ti:"video inpainting"` | 74 | 提高 direct-task 精度 |
| `all:"video inpainting" AND submittedDate:[202401010000 TO 202608302359]` | 88 | 2024–2026 前沿扫描 |
| `all:"video object removal" AND submittedDate:[202401010000 TO 202608302359]` | 16 | 对象、副作用与反事实分支 |
| `all:"video outpainting" AND submittedDate:[202401010000 TO 202608302359]` | 8 | 与内部 hole restoration 分界 |

实际按 `submittedDate desc` 拉取 `all:"video inpainting"` 前 100 条；本次对终端可见的前 60 条逐题名/摘要筛选，并对 2024–2026 direct candidates 做 exact-ID 回查。代表性 exact-ID 请求：

~~~text
https://export.arxiv.org/api/query?
id_list=2412.00857,2501.10018,2501.12267,2503.05639,
2504.15661,2506.12853,2510.08555,2511.03272,
2601.06391,2603.09283,2603.19224,2604.02296,
2604.14648,2605.14534,2608.05565
~~~

API 用于核对题名、作者、首次提交、更新版本、abstract、comment 与作者项目链接。arXiv comment 不是正式 venue 的充分证据；能找到 proceedings 时必须回查会场页。

### 2.2 表面 B：OpenAlex

使用 `api.openalex.org/works` 的 `title.search`、日期过滤和 `cited_by_count` 排序。数字是索引命中，不代表都与核心任务相关。

| OpenAlex filter | 返回量 | 处置 |
|---|---:|---|
| `title.search:video inpainting`, 2004-01-01 至冻结日 | 523 | 候选发现、同题/版本去重 |
| 同上，2024-01-01 至冻结日 | 155 | 新近候选交叉检查 |
| `title.search:video object removal`, 2024-01-01 至冻结日 | 33 | effect-aware removal 候选 |
| `title.search:video outpainting`, 2024-01-01 至冻结日 | 22 | outpainting 边界候选 |

对前 30 条 `video inpainting` title match 按 cited-by 排序的冻结日快照中，核心节点包括：STTN 275、Newson 272、Deep Flow-Guided 267、Free-Form 205、FuseFormer 158、E²FGVI 144、ProPainter 89。数值只用于发现/优先级，随索引更新而漂移，不进入正文性能主张。

### 2.3 表面 C：正式 proceedings / DOI

使用标题精确式回查：

~~~text
site:openaccess.thecvf.com "Deep Flow-Guided Video Inpainting"
site:openaccess.thecvf.com "Copy-and-Paste Networks for Deep Video Inpainting"
site:ecva.net "Learning Joint Spatial-Temporal Transformations for Video Inpainting"
site:ecva.net "Flow-edge Guided Video Completion"
site:openaccess.thecvf.com FuseFormer E2FGVI ProPainter video inpainting
site:openaccess.thecvf.com AVID "Keyframe-Guided Creative Video Inpainting"
site:openaccess.thecvf.com VACE video creation editing
site:openaccess.thecvf.com Object-WIPER CVPR 2026
~~~

用于正文的正式页包括 CVPR / ICCV / ECCV 的 DFVI、VINet、CPNet、FVI、STTN、FGVC、FuseFormer、E²FGVI、DEVIL、ProPainter、AVID、language-driven inpainting、VideoRepainter、VACE、Unboxed、Object-WIPER，以及 VideoPainter 的 ACM DOI。正式页支持 venue、作者、机制与论文设置内结果；不支持跨硬件、跨数据和跨协议的普适最优。

### 2.4 表面 D：作者项目、代码与 benchmark artifact

| 工作 | 官方 artifact | 核对项 |
|---|---|---|
| Space-Time Completion | [作者项目页](https://www.wisdom.weizmann.ac.il/~vision/VideoCompletion.html) | CVPR 2004 / TPAMI 2007 与全局时空 patch 目标 |
| FGVC | [作者项目页](https://www.chengao.vision/FGVC/) | motion-edge、non-local flow、ECCV 2020 |
| E²FGVI | [MCG-NKU/E2FGVI](https://github.com/MCG-NKU/E2FGVI) | 三阶段端到端结构、YouTube-VOS / DAVIS split 与 mask 文件 |
| ProPainter | [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | dual-domain propagation、模型/推理发布面 |
| FloED | [作者项目页](https://nevsnev.github.io/FloED/) | flow branch、adapter、cache；未从项目页升级 venue |
| DiffuEraser | [lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) | prior 初始化、长序列推理 artifact |
| VideoPainter | [作者项目页](https://yxbian23.github.io/project/video-painter) | VPData / VPBench、dual stream、ID resampling |
| VACE | [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | MV2V / V2V / R2V、代码/权重与任务边界 |
| Object-WIPER | [sakshamsingh1/object_wiper](https://github.com/sakshamsingh1/object_wiper) | WIPER-Bench、TokSim；官方 repo 明示第三方实现不是官方支持 |
| SVOR | [作者项目页](https://xiaomi-research.github.io/svor/) | MUSE、DA-Seg、两阶段训练 |
| EffectErase | [作者项目页](https://henghuiding.com/EffectErase/) | CVPR 2026、VOR paired data、inverse insertion task |
| PROVE | [xiaomi-research/prove](https://github.com/xiaomi-research/prove) | RC-S / RC-T、PROVE-M/H、开源数据与论文数据存在合规差异 |
| EffectLearner | [作者项目页](https://morleyolsen.github.io/EffectLearner/) | 2026-08 最新 preprint 与 EffectWorld 发布面 |

Artifact 只支持“作者公开了什么、代码怎样配置、数据怎样声明”。没有本次独立复现实验时，不把 README demo、速度或 benchmark 排名写成普适结论。

## 🧾 3. 纳入、排除、去重与证据等级

### 3.1 纳入标准

- 直接研究视频空间/时空洞补全、对象移除、相邻 outpainting，或其专用评测；
- 改变了可见像素传播、缺失区生成、全局时间一致、mask grounding、known-region protection 或任务/评测定义；
- 机制事实能回到原论文、正式 proceedings 或作者技术报告；
- 2025–2026 preprint 只有在直接改变任务分支时纳入，并明确证据弱于正式会场；
- 通用基础模型只有在提供 masked V2V / inpainting direct interface 时作为邻近系统证据，不自动成为专用里程碑。

### 3.2 排除、降级或转交

| 候选 | 处置 | 原因 |
|---|---|---|
| 通用 image inpainting / object removal | 排除主线 | 不建模视频时间一致；例如 ObjectClear 是 image model |
| 仅 T2V / I2V 生成模型 | 排除直接 milestone | 没有 mask contract、outside preservation 或 direct-task evaluation |
| virtual try-on、stereo、endoscopy、driving insertion | 不进通用主线 | arXiv phrase 命中，但属于特定领域条件；可在专章扩展 |
| VACE | 邻近系统证据 | direct masked V2V 存在，但核心贡献是 all-in-one interface |
| DiTPainter / EraserDiT | 前沿观察 | 截至冻结日为 arXiv preprint / technical report，不与正式会场同级 |
| VideoCanvas / unified long-video co-denoising | 前沿观察 | 新近预印本；保留 task definition / algorithm claim，限制 venue 与普适效果 |
| Unboxed / Seen-to-Scene | 纳入边界，不倒写 | direct outpainting，目标在原画布之外，与内部洞恢复不同 |
| 产品按钮、商业 API、精选 demo | 排除论文证据 | 产品名不等于单 checkpoint；缺模型、数据、seed 与协议 |
| OpenAlex broad-query 噪声 | title/abstract 排除 | 包含 survey、tensor completion、medical/3D 应用和仅引用 video inpainting 的论文 |

正文最终注册 **38 个去重 primary works**：R01–R34 是方法/benchmark，R35–R38 是数据与基础指标。arXiv、proceedings、project/code 是同一工作的不同证据面，不重复计数。由于检索式高度重叠且本次是机制导向 scoping review，不虚构一个“全库去重后总文献数”。

### 3.3 证据等级

| 等级 | 来源组合 | 可支持 | 不能自动支持 |
|---|---|---|---|
| **E1** | 正式 proceedings / DOI + 原论文 | 作者、venue、机制、论文设置内结果 | 跨协议最优、商业可用、独立复现 |
| **E2** | 作者 arXiv technical report + 官方项目 | 方法、作者报告数据/结果、版本与发布面 | 同行评审共识或普适泛化 |
| **E3** | 官方代码、dataset、project、conference index | 配置、权重、数据许可/差异、venue 交叉核验 | 论文全结论已复现 |
| **E4** | OpenAlex / 搜索索引 | 候选发现、重复/年份/DOI 提示 | 最终机制、作者或 venue 裁决 |
| **S** | 本次跨来源综合 | 术语边界、tensor contract、最小评测协议 | 新的实证结果 |

## 📚 4. Primary evidence registry

### 4.1 Patch、flow、attention 与 hybrid

| ID | 工作与一手页 | 等级 | 正文使用与边界 |
|---|---|---|---|
| R01 | [Wexler et al., Space-Time Video Completion](https://www.wisdom.weizmann.ac.il/~vision/VideoCompletion.html), CVPR 2004 / TPAMI 2007 | E1 + E3 | 全局时空 patch 一致；不写成语义生成 |
| R02 | [Newson et al., Video Inpainting of Complex Scenes](https://arxiv.org/abs/1503.05528), SIAM 2014 | E1 | 全局 patch functional、动态纹理与移动背景 |
| R03 | [Xu et al., Deep Flow-Guided Video Inpainting](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Deep_Flow-Guided_Video_Inpainting_CVPR_2019_paper.html), CVPR 2019 | E1 | completed flow → pixel propagation；flow 误差边界 |
| R04 | [Kim et al., Deep Video Inpainting](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Deep_Video_Inpainting_CVPR_2019_paper.html), CVPR 2019 | E1 | 时间结构 + 空间细节；VINet 历史定位 |
| R05 | [Lee et al., Copy-and-Paste Networks](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Copy-and-Paste_Networks_for_Deep_Video_Inpainting_ICCV_2019_paper.html), ICCV 2019 | E1 | learned alignment 与 reference copying |
| R06 | [Chang et al., Free-Form Video Inpainting](https://openaccess.thecvf.com/content_ICCV_2019/html/Chang_Free-Form_Video_Inpainting_With_3D_Gated_Convolution_and_Temporal_PatchGAN_ICCV_2019_paper.html), ICCV 2019 | E1 | 3D gated conv、temporal PatchGAN、free-form mask data |
| R07 | [Zeng et al., STTN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2590_ECCV_2020_paper.php), ECCV 2020 | E1 | joint spatial-temporal attention、同步补多帧 |
| R08 | [Gao et al., FGVC](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570698.pdf), ECCV 2020 | E1 | motion-edge first、non-local flow connection |
| R09 | [Liu et al., FuseFormer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_FuseFormer_Fusing_Fine-Grained_Information_in_Transformers_for_Video_Inpainting_ICCV_2021_paper.html), ICCV 2021 | E1 | Soft Split / Composition 与细粒度 token 融合 |
| R10 | [Li et al., E²FGVI](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Towards_an_End-to-End_Framework_for_Flow-Guided_Video_Inpainting_CVPR_2022_paper.html), CVPR 2022 | E1 + E3 | flow completion、feature propagation、content hallucination 端到端 |
| R11 | [Szeto & Corso, DEVIL](https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html), CVPR 2022 | E1 + E3 | 1,250 clips、相机/背景与 mask 属性 slice；不覆盖所有真实失败 |
| R12 | [Zhou et al., ProPainter](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_ProPainter_Improving_Propagation_and_Transformer_for_Video_Inpainting_ICCV_2023_paper.html), ICCV 2023 | E1 + E3 | dual-domain propagation、sparse Transformer；论文内强基线，不写普适最优 |
| R13 | [Zhang et al., FGT++](https://arxiv.org/abs/2301.10048), TPAMI 2024 | E1 | flow-guided feature / attention 互补；journal metadata 由 DOI/题录交叉核验 |

### 4.2 Diffusion、Video DiT、长视频与 outpainting

| ID | 工作与一手页 | 等级 | 正文使用与边界 |
|---|---|---|---|
| R14 | [Zhang et al., AVID](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_AVID_Any-Length_Video_Inpainting_with_Diffusion_Model_CVPR_2024_paper.html), CVPR 2024 | E1 | motion module、structure guidance、Temporal MultiDiffusion；any-length 不等于无漂移 |
| R15 | [Wu et al., Language-Driven Video Inpainting](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Towards_Language-Driven_Video_Inpainting_via_Multimodal_Large_Language_Models_CVPR_2024_paper.html), CVPR 2024 | E1 | 从人工 mask 向语言 grounding 扩展；仍需定位误差审计 |
| R16 | [Gu et al., FloED](https://arxiv.org/abs/2412.00857), v3 | E2 | flow branch、adapter、latent interpolation、cache；仅写 preprint |
| R17 | [Li et al., DiffuEraser](https://arxiv.org/abs/2501.10018) | E2 + E3 | 传统 prior 初始化/弱条件、长时感受野；不宣称正式 venue |
| R18 | [Xie et al., VipDiff](https://arxiv.org/abs/2501.12267), WACV 2025 | E1 | training-free reverse diffusion 约束与多样候选 |
| R19 | [Guo et al., VideoRepainter](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Keyframe-Guided_Creative_Video_Inpainting_CVPR_2025_paper.html), CVPR 2025 | E1 | keyframe image editing + I2V symmetric condition；creative target 不等同恢复 GT |
| R20 | [Bian et al., VideoPainter](https://doi.org/10.1145/3721238.3730673), SIGGRAPH 2025 | E1 + E3 | context encoder 为 backbone 参数 6%（作者设置）、ID resampling、VPData/VPBench >390K |
| R21 | [Jiang et al., VACE](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf), ICCV 2025 | E1 + E3 | VCU / Context Adapter、masked V2V；邻近 unified interface |
| R22 | [Wu & Liu, DiTPainter](https://arxiv.org/abs/2504.15661) | E2 | 从头训练的高效 DiT 探索；preprint，不升级 venue |
| R23 | [Liu & Hui, EraserDiT](https://arxiv.org/abs/2506.12853) | E2 | Circular Position-Shift 与作者速度设置；technical report |
| R24 | [Cai et al., VideoCanvas](https://arxiv.org/abs/2510.08555), v2 | E2 | arbitrary spatiotemporal patches / in-context condition；不宣称所有子任务已解决 |
| R25 | [Lyu et al., Unified Long Video In/Outpainting](https://arxiv.org/abs/2511.03272) | E2 | overlap-and-blend high-order co-denoising；“arbitrarily long”限制为算法/作者主张 |
| R26 | [Yu et al., Unboxed](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Unboxed_Geometrically_and_Temporally_Consistent_Video_Outpainting_CVPR_2025_paper.html), CVPR 2025 | E1 | 3DGS 静态区 + 动态内容生成；仅作 outpainting 边界 |
| R27 | [Jeon et al., Seen-to-Scene](https://arxiv.org/abs/2604.14648) | E2 | propagation + generation 的 outpainting；venue 只按作者 comment 写 CVPR 2026 Findings |

### 4.3 对象效应、反事实与 2026 评测

| ID | 工作与一手页 | 等级 | 正文使用与边界 |
|---|---|---|---|
| R28 | [Miao et al., ROSE](https://arxiv.org/abs/2508.18633) | E2 | 五类对象副作用、3D synthetic paired data、ROSE-Bench；真实域泛化需验证 |
| R29 | [Kushwaha et al., Object-WIPER](https://openaccess.thecvf.com/content/CVPR2026/papers/Kushwaha_Object-WIPER_Training-Free_Object_and_Associated_Effect_Removal_in_Videos_CVPR_2026_paper.pdf), CVPR 2026 | E1 + E3 | training-free effect localization、background token copy、WIPER-Bench / TokSim |
| R30 | [Hu et al., SVOR](https://arxiv.org/abs/2603.09283), v2 | E2 + E3 | MUSE、DA-Seg、mask degradation、two-stage curriculum；preprint |
| R31 | [Fu et al., EffectErase](https://henghuiding.com/EffectErase/), CVPR 2026 | E1 + E3 | 作者报告 60K paired VOR、insertion–removal reciprocal learning |
| R32 | [Motamed et al., VOID](https://arxiv.org/abs/2604.02296) | E2 | object-interaction counterfactual pairs、VLM affected-region guidance；不外推普适因果 |
| R33 | [Li et al., PROVE](https://arxiv.org/abs/2605.14534), ACM MM 2026 | E1 + E3 | RC-S / RC-T、PROVE-M 80 / PROVE-H 100；开源数据与论文数据存在官方声明差异 |
| R34 | [Wu et al., EffectLearner](https://arxiv.org/abs/2608.05565) | E2 | VLM Object-Effect Reasoner + DiT eraser；2026-08 新近 preprint |

### 4.4 数据与基础指标

| ID | 工作 | 等级 | 用法 |
|---|---|---|---|
| R35 | [DAVIS](https://openaccess.thecvf.com/content_cvpr_2016/html/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.html), CVPR 2016 | E1 | VOS 数据被二次用作 clean RGB / mask source；不是原生 inpainting dataset |
| R36 | [YouTube-VOS](https://arxiv.org/abs/1809.00461), ECCV 2018 | E1 | 同上；具体 split/mask 文件必须公开 |
| R37 | [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html), CVPR 2018 | E1 | masked / outside perceptual distance；不替代时间一致 |
| R38 | [FVD](https://arxiv.org/abs/1812.01717) | E2 | 视频 feature distribution 基础；VFID/FVD 实现须写 checkpoint/clip/sample |

## ⚖️ 5. 关键裁决与不升级规则

### 5.1 任务名称

- 狭义 video inpainting：原画布内 hole；通常有 known-region preservation contract。
- object removal：hole 支持集可能必须扩到 shadow / reflection / light / interaction。
- outpainting：未知区在原画布外；2D flow、3D scene 与生成模型的权重不同。
- masked V2V：可能执行局部重绘，但 target 不必是恢复被遮挡的原内容。
- generative fill：产品/workflow 名称，不能单独确定模型、checkpoint、数据或评测协议。

### 5.2 Venue / 版本

1. VideoPainter 使用 ACM DOI `10.1145/3721238.3730673` 与 CUHK publication metadata，确定 SIGGRAPH Conference Papers 2025。
2. VACE 使用 ICCV 2025 CVF 正式论文；官方 repo 的 Wan/LTX checkpoint 属发布面，不倒写成论文只使用某一 checkpoint。
3. FloED、DiffuEraser、DiTPainter、VideoCanvas、Unified Long、ROSE、SVOR、VOID、EffectLearner 按 arXiv technical/preprint 状态写；即使有项目页也不自动升级 venue。
4. EraserDiT 的 arXiv comment 明示 `technical report`，正文不把作者速度写成独立 benchmark。
5. Seen-to-Scene 的 “CVPR 2026 Findings” 来自作者 arXiv comment；在正式 proceedings 可定位前，不提升为 E1。
6. Object-WIPER 使用 CVPR 2026 CVF 正式 PDF；其 GitHub 明确标注 third-party implementation 非官方维护。
7. EffectErase 的 CVPR 2026 由作者项目页与 arXiv comment 交叉核验；机制仍引用原论文/项目。
8. PROVE 的 ACM MM 2026 由作者 project/GitHub acceptance update 交叉核验；repo 明示开源数据因 compliance 与论文版本略有差异。

### 5.3 “最新”不等于“最强”

- 2026-08 的 EffectLearner 只作为冻结日 frontier observation；不因提交晚就排序第一。
- 传播模型在“别帧真实可见 + mask 外严格保护”条件下可能更符合任务合同；生成模型在“全时不可见大洞”条件下更有先验优势。
- 任何 SOTA 数字都必须绑定数据 split、mask、分辨率、帧数、seed、硬件与输出选择；正文因此不复制跨论文排行榜。

## 🧠 6. 正文综合是怎样形成的

### 6.1 Copy-first / generate-residual 合同

跨 R03、R08、R10、R12、R16–R20 形成 S 级综合：

1. 双向 flow / correspondence 只为通过 visibility、occlusion、out-of-bound 与 consistency 的 source 位置建立证据；
2. propagated coverage 之外的 residual holes 才交给 synthesis prior；
3. Transformer / diffusion 的任务不是重画所有已知内容，而是融合可靠 anchor 并生成不可观测部分；
4. 最终 decoded RGB hard composite 与 outside-mask audit 是系统不变量，不由模型 loss 自动保证。

正文公式是统一记号，不声称任何单篇论文完整实现了所有置信度、loss 与保护步骤。

### 6.2 长视频协议

跨 R14、R20、R24、R25、R27 形成：窗口长 $L$、stride $S$、RGB blend、latent co-denoise、noise sharing、ID memory、source re-anchor 与 scene-cut reset 必须分开。接口可继续运行只支持 open-ended capability；质量不漂移要由随窗口数的 curve 和最长测量点证明。

### 6.3 评测协议

跨 R11、R29、R33、R35–R38 形成：

- 有 GT 时，洞内 reconstruction、mask 外 preservation、局部 temporal consistency、全局 distribution 与成本分栏；
- 一对多 removal / fill 时，不以到唯一 GT 的 PSNR 单独裁决；
- global metric 容易被未编辑背景稀释，必须加 masked / ring-local 指标；
- flow-based warp metric 受 flow 误差影响，稳定的模糊/常数输出也可取得好 temporal score；
- 自动指标必须用随机双盲人评校准，报告 ties、置信区间和选择策略。

### 6.4 反事实边界

R28–R31 支持外观副作用，R32 进一步处理交互后果；两者不是同一证据等级。消除影子/反射不证明物理状态正确；模拟 counterfactual 和人类偏好也不自动证明真实世界的普适干预因果。

## 🎨 7. AI 科学示意图生成与视觉审计

### 7.1 技能与执行方式

- 使用 `literature-review`、`scientific-schematics` 与系统 `imagegen` 技能；
- `scientific-schematics/references/diagram_types.md` 在本机安装中缺失，已按技能规则使用 `best_practices.md` 与 `imagegen/references/prompting.md`、`sample-prompts.md` 回退；
- 使用内置 `image_gen`，不是 CLI / 外部 API；
- use-case：`scientific-educational`；
- 原始生成 artifact：`generated_images/01a04ecd-f44e-7d80-b18b-0e8677454405/exec-4c3b76db-89f6-454f-84f2-5491848e50cb.png`；
- 项目内最终文件：`assets/diagrams/video-inpainting-evidence-pipeline.png`。

### 7.2 最终 prompt

~~~text
Use case: scientific-educational
Asset type: publication-quality textbook schematic for a technical video generation chapter
Primary request: Create a clean landscape scientific diagram titled exactly "VIDEO INPAINTING: EVIDENCE PIPELINE". Show one left-to-right pipeline with five large numbered stages and explicit arrow flow.
Stage 1 exact label: "1  INPUT + MASK". Show a strip of three video frames with the same moving object-shaped red hatched hole. Label the unmasked area exactly "KNOWN PIXELS" and the red hatched area exactly "HOLE".
Stage 2 exact label: "2  VALID PROPAGATION". Show completed optical-flow arrows from neighboring frames, a small confidence shield and occlusion symbol. Use blue only for pixels copied from reliable visible references. Include exact small labels "FLOW", "CONFIDENCE", "OCCLUSION".
Stage 3 exact label: "3  SYNTHESIZE RESIDUAL". Show that only the remaining unresolved region, not the whole frame, enters a compact block labeled exactly "TRANSFORMER / VIDEO DiT". Use purple for generated residual content. Label remaining holes exactly "RESIDUAL HOLES".
Stage 4 exact label: "4  GLOBAL TEMPORAL CHECK". Show overlapping temporal windows, a memory ribbon, a two-way consistency arrow, and a hard reset gate at a scene cut. Include exact labels "OVERLAP", "MEMORY", "SCENE-CUT RESET".
Stage 5 exact label: "5  COMPOSITE + EVALUATE". Show a green lock over the known region and the final coherent frames. Include four compact metric chips with exact text: "MASK QUALITY", "OUTSIDE-MASK ERROR", "WARP ERROR", "TIME + MEMORY".
Add a bottom rule with exact text: "COPY WHEN VISIBLE · GENERATE WHEN HIDDEN · PROTECT KNOWN PIXELS".
Style/medium: flat vector-like scientific infographic, white background, dark charcoal sans-serif text, crisp thin lines, generous whitespace, 16:9 landscape, readable at textbook width. Use a colorblind-safe palette with redundant patterns: green plus lock means protected, blue plus arrows means propagated, purple plus sparkle means synthesized, red hatch means missing. Provide a small legend with the exact labels "PROTECTED", "PROPAGATED", "GENERATED", "MISSING".
Scientific constraints: propagation must occur before synthesis; synthesis must target residual holes only; global temporal checking must occur before final compositing; known pixels remain unchanged in the final output. No equations, no quantitative claims, no logos, no trademarks, no watermark, no decorative characters. All text must be rendered verbatim once, with no misspellings, no duplicated stages, no extra text, no cropped labels, no overlapping arrows or elements.
~~~

### 7.3 文件与视觉检查

| 项 | 结果 |
|---|---|
| 尺寸 | 1693 × 929 px |
| 格式 | PNG，8-bit RGB，non-interlaced，无 alpha |
| 文件大小 | 1,453,618 bytes |
| SHA-256 | `bc70ca30031d104fb94af328bb9977acead75333d6c9532b7858ee7b77287676` |
| 原图检查 | 使用 `view_image(detail=original)`；标题、五阶段、三帧时序、箭头、legend、四指标和底部规则均完整 |
| 文本检查 | `VIDEO INPAINTING`、`TRANSFORMER / VIDEO DiT`、`SCENE-CUT RESET`、`OUTSIDE-MASK ERROR` 等无错拼、重复或裁切 |
| 布局检查 | 无标签/箭头重叠；五栏顺序明确；末栏绿色 lock 与已知区保护一致 |
| 科学检查 | propagation 先于 residual synthesis；temporal check 先于 composite；`MASK QUALITY` 是输入诊断，`OUTSIDE-MASK ERROR` 才是已知区保护证据；没有虚构数字/benchmark |
| 可访问性 | 红色缺失区另有斜线；蓝色传播另有箭头；紫色生成另有闪光；绿色保护另有锁；正文提供完整 alt、图注与顺序化文字替代 |
| 灰度检查 | ImageMagick 转换后以原分辨率目视；斜线、箭头、闪光、锁和文字仍能区分四类状态 |

首版即通过文字、布局与科学语义检查，因此没有为了“多迭代”而无目的重生成。灰度临时文件位于 `/tmp/video-inpainting-evidence-pipeline-gray.png`，未加入仓库。

## 🚧 8. 局限与冻结日后的更新规则

- arXiv / OpenAlex 返回量会变；以后更新必须写新冻结日，不能覆盖本记录的历史数字。
- OpenAlex citation count 是索引信号，不是质量分或纳入标准。
- 2025–2026 preprint 的 venue、作者顺序、数据和实验可能更新；必须重查 arXiv version、proceedings 与官方 artifact。
- VPData、VOR、EffectWorld、PROVE 等作者数据的许可、开放范围和论文/开源差异需逐版本审计；“有 GitHub”不等于数据/权重/训练 recipe 全开放。
- 本次没有运行各模型 checkpoint，因此正文只陈述论文机制、作者报告和 S 级协议，不提供新的 SOTA 排名或速度复现。
- 用户研究、RC/TokSim、VLM judge 与传统 GT 指标都可能偏置；未来如有更大规模独立 human-alignment study，应更新评测章节而非简单追加模型名。

## ✅ 9. 写作验收映射

| 要求 | 正文位置 | 证据 |
|---|---|---|
| 六类任务边界 | §2 + Mermaid 1 | 公式、对照表、顺序化文字替代 |
| flow / propagation / attention / hybrid | §4–5 | R01–R13 + tensor/error chain |
| STTN / FuseFormer / E²FGVI / ProPainter | §4–5、§8 | 正式 proceedings 与 milestone |
| diffusion / DiT / VACE | §5.5–5.6 | R14–R24，直接/邻近证据分栏 |
| 2024–2026 frontier | §5–8 | venue/preprint 边界与 2026-08 冻结 |
| 长视频、overlap、scene cut | §6 | 窗口表 + report contract |
| 对象副作用、物理交互 | §7 | R28–R34，外观/反事实分层 |
| 数据、mask、指标与成本 | §9 | DAVIS / YouTube-VOS / DEVIL / PROVE + 8 类 mask |
| 失败模式 | §10 | 13 行根因–定位–修复矩阵 |
| 可访问视觉 | §2–4 | 2 Mermaid + 1 AI PNG，各有 alt/acc/文字替代 |
| 最详细 paper review / milestone | §5、§8、本记录 §4–5 | 38-work evidence registry 与不升级规则 |

最终静态检查、引用锚点、相对链接和 Mermaid 实际渲染结果将在完成后追加于下一节。

## ✅ 10. 冻结日最终验证

验证均在仓库根目录、冻结日 2026-08-30 执行：

| 检查 | 命令或方法 | 精确结果 |
|---|---|---|
| Markdown 规范 | `npx --yes markdownlint-cli2 docs/tasks/video-inpainting.md sources/research_20260830_video_inpainting.md` | 2 个文件，`0 issues` |
| 引用锚点 | 脚本核对正文引用编号与 `<a id="ref-N">` | 38 个定义、84 次引用、38 个唯一引用；缺失 0、未使用 0、重复 0、编号错配 0、顺序正确 |
| 相对链接 | 逐个从所在 Markdown 目录解析本地链接 | 检查 8 个，失效 0 |
| Mermaid 可访问性 | 统计 fenced block、`accTitle`、`accDescr` 与顺序化文字替代 | 2 个 Mermaid；2 个 `accTitle`、2 个 `accDescr`；正文共 3 处顺序化文字替代（含 AI 图） |
| Mermaid 实际渲染 | `@mermaid-js/mermaid-cli` + 本机 Google Chrome，2× scale 输出 PNG | 两图均退出码 0；尺寸分别为 1568×1742、1568×200；原图目视无裁切、节点/连线重叠或错误分支 |
| AI 图资产 | PNG 属性、SHA-256 与原图/灰度目视检查 | 1693×929、RGB、1,453,618 bytes；SHA-256 `bc70ca30031d104fb94af328bb9977acead75333d6c9532b7858ee7b77287676`；文字、布局、色弱冗余与科学语义通过 |
| 补丁空白 | `git diff --check -- docs/tasks/video-inpainting.md sources/research_20260830_video_inpainting.md` | 退出码 0，无输出 |

本记录与正文通过以上检查后冻结；没有运行模型 checkpoint，因而不把文献报告的质量、速度或开放状态写成独立复现结论。
