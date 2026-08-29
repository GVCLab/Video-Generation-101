# 视频帧插值（VFI）研究记录：检索、证据与图像审计

> 检索冻结日：**2026-08-30（Asia/Shanghai）**
> 对应章节：[`docs/tasks/frame-interpolation.md`](../docs/tasks/frame-interpolation.md)
> 记录目的：让方法归类、首发年份、数据协议、生成图片和验收结果可以被逐项复核，而不是把作者摘要或排行榜直接当成结论。

## 1. 研究问题与范围

本轮围绕六个可判定问题展开：

1. VFI 与缺帧修复、未来外推、时空超分、生成式关键帧过渡的输入和输出边界是什么？
2. flow/warping、SoftSplat、depth/occlusion、kernel/deformable、轻量 CNN、Transformer、SSM、diffusion/DiT 各自解决哪个变量？
3. RIFE、IFRNet、VFIformer、AMT 等名称是否造成错误归类？
4. midpoint、显式 arbitrary-time、递归 midpoint 和联合多帧是否有可区分的训练与评测证据？
5. Vimeo90K、Middlebury、UCF101、Xiph、SNU-FILM、DAVIS、X4K 的同名分数何时不可直接比较？
6. 生成路线在允许中间路径多解时，怎样仍把两个观测端点当作硬条件，并检查身份、几何和时序守恒？

纳入范围为双端点条件下的帧间合成及其直接评测；只在定义边界时涉及视频外推、空间超分和无条件/文本视频生成。章节不汇总未经统一协议复测的 SOTA 数值。

## 2. 检索设计

### 2.1 三类互补的一手来源

| 来源类型 | 作用 | 本轮使用方式 | 证据局限 |
|---|---|---|---|
| arXiv API / 论文 PDF | 冻结首个公开版本、读公式和实验协议 | 主题检索、18 个目标 ID 批量回读、重点论文全文搜索 | arXiv comment 中的接收声明不等于正式出版页 |
| OpenAlex API | 检查主题覆盖和题名规模 | broad search 与 `title.search` 双查询 | broad search 会把正文提及 VFI 的论文也计入，结果数不能视为核心文献数 |
| 官方会议/期刊页 | 核验正式题名、作者、venue、年份 | CVF Open Access、ECVA、AAAI、Middlebury 官方页逐条打开 | 正式页通常不记录首个预印本日期 |
| 作者/机构项目与官方代码 | 核验数据、checkpoint、接口和可复现实物 | GitHub API 搜索后回到作者仓库/项目页 | README 中速度和 SOTA 仍是作者自报，需要统一环境复测 |

检索和结论均优先采用论文原文、官方 proceedings、数据集主页与作者代码；搜索引擎摘要只用于定位，不作为章节证据。

### 2.2 查询、日期与结果数

所有查询执行于 2026-08-30；arXiv 时间窗截止冻结日 23:59。

| 通道 | 精确检索式或调用 | 原始结果数 | 本轮用途 |
|---|---|---:|---|
| arXiv API | `all:"video frame interpolation" AND submittedDate:[201601010000 TO 202608302359]` | 190 | 观察 2016–2026 主题跨度，并读取最新 5 条 |
| arXiv API | `all:"video frame interpolation" AND (all:diffusion OR all:transformer OR all:"optical flow") AND submittedDate:[201601010000 TO 202608302359]` | 107 | 对 flow、attention、diffusion 三类交集做初筛 |
| arXiv API | `id_list=1708.01692,...,2608.22861` | 18/18 返回 | 冻结关键论文 v1 日期、标题和版本状态 |
| arXiv API | `id_list=2205.07230` | 1/1 返回 | 区分 VFIformer 与 VFIT；前者不是 `2111.13817` |
| OpenAlex API | `search=video frame interpolation`, `from_publication_date=2016-01-01`, `to_publication_date=2026-08-30` | 39,525 | 发现 broad full-text/metadata search 过宽，不作为纳入分母 |
| OpenAlex API | `filter=title.search:video frame interpolation`，同一日期窗 | 490 | 题名级覆盖检查和去重辅助 |
| GitHub Search API | `"video frame interpolation" in:name,description,readme` | 1,326 | 探索性检索；噪声高，排除 fork/聚合镜像后改用定向仓库 |
| GitHub Search API | VFI 题名/关键词窄检索 | 59 | 定位作者或机构仓库候选，再与论文作者/项目页互证 |
| 官方 venue | 14 个 CVF/ECVA/AAAI 目标 URL | 14/14 HTTP 200 | 核验 SepConv、DAIN、SoftSplat、XVFI、IFRNet、VFIformer、AMT、VIDIM、BiM-VFI、EDEN、LDF-VFI、RIFE、LDMVFI 等正式页 |

目标 arXiv ID 集合为：`1708.01692`、`1712.00080`、`1904.00830`、`1907.10244`、`2003.05534`、`2011.06294`、`2111.13817`、`2202.04901`、`2205.14620`、`2303.09508`、`2304.09790`、`2404.01203`、`2412.11365`、`2503.15831`、`2601.14959`、`2607.15585`、`2608.13460`、`2608.22861`；VFIformer 另查 `2205.07230`。

### 2.3 纳入、排除与去重

纳入条件：

- 明确定义双端点 VFI，或直接定义其数据/指标；
- 能从论文 PDF、官方出版页、项目页或作者仓库核验至少一项机制、协议或年份；
- 对 2026 前沿工作，允许只有预印本，但必须显式标为“预印本/作者 venue 声明”；
- 经典 flow 和质量指标仅在解释假设与评价边界时纳入。

排除条件：

- 只有二手博客、模型聚合页或无作者对应关系的复刻仓库；
- 只做 future prediction、纯空间超分、视频压缩或文本视频生成而未提供 VFI 证据；
- 没有固定数据版本、帧间隔或指标实现，却只报告一个不可追溯分数；
- 重复的 arXiv/会议版本保留一个工作实体，首发日期取 arXiv v1，正式年份取官方 proceedings。

初筛标题与摘要后，章节核心纳入 32 个引用实体：21 个方法/前沿工作、6 个数据或 benchmark/质量研究、5 个经典 flow/指标来源。没有把 190 或 490 当成“已阅读全文”的数量。

## 3. 证据等级

| 等级 | 定义 | 可支持的表述 |
|---|---|---|
| **E1** | 正式论文全文或官方数据/评测页，且关键公式/协议已回读 | 机制、数据划分、正式 venue、评测规则 |
| **E2** | arXiv 全文与元数据；正式页缺失或尚未独立出现 | 首发日期、预印本机制、作者报告结果 |
| **E3** | 作者/机构代码、项目页、checkpoint 或 README，与论文身份互证 | 可用实现、输入接口、数据下载、工程配置 |
| **E4** | 二手索引、搜索摘要、作者 comment 或未复测 benchmark 表 | 只用于发现线索，不单独支撑正文结论 |

同一结论尽量由互补来源闭环。例如“RIFE 首发 2020、正式 ECCV 2022、代码可运行”分别由 arXiv 元数据、ECVA 页和作者仓库支持；“实时”仍只记为作者特定硬件下的报告，不升级为跨设备事实。

## 4. 关键机制的原文回读

### 4.1 对应、投影与遮挡

| 工作 | 已核对的原始证据 | 结论 | 等级 |
|---|---|---|---|
| SepConv | ICCV 2017 正式页、arXiv `1708.01692` | 每像素预测可分离局部核，降低完整二维动态核的内存代价 | E1 |
| DAIN | CVPR 2019 PDF、arXiv `1904.00830` | 用 inverse depth 加权 flow projection 的碰撞候选，近表面优先；不是“有 depth 就解决所有遮挡” | E1 |
| AdaCoF | CVPR 2020 正式页、arXiv `1907.10244` | 同时学习局部样本的权重和 offset，连接 kernel 与 flow 参数化 | E1 |
| SoftSplat | CVPR 2020 PDF、arXiv `2003.05534` | `exp(Z)` 归一化 forward splat；对 $Z$ 的常数平移不变，但不自动修正错误 flow 或空洞 | E1 |

SoftSplat 章节公式来自论文定义：多个源像素碰撞到同一目标像素时，先按双线性 footprint 和指数重要性累加，再除以权重和。它处理的是可微碰撞聚合，不是完整的 occlusion reasoning。

### 4.2 高效卷积路线与纠错

| 工作 | 原始来源 | 核验结论 | 等级 |
|---|---|---|---|
| RIFE | arXiv `2011.06294` PDF、ECCV 2022 页、`hzwer/ECCV2022-RIFE` | IFNet 直接估计 intermediate flow；基础设置是 midpoint，`RIFE_m` 才用 septuplet 随机时间训练 | E1+E3 |
| IFRNet | CVPR 2022 PDF、arXiv `2205.14620`、`ltkong218/IFRNet` | 共享 encoder + 四级 CNN decoder，联合更新双边 flow 与中间特征；**不是 Transformer** | E1+E3 |
| FILM | arXiv `2202.04901`、`google-research/frame-interpolation` | 跨尺度共享卷积权重、双向匹配与 GridNet synthesis 面向大位移 | E1+E3 |
| AMT | CVPR 2023 PDF、arXiv `2304.09790`、`MCG-NKU/AMT` | 双向 all-pairs correlation + multi-field flow；摘要明确称 convolution-based，名称的 Transforms **不是 Transformer** | E1+E3 |

IFRNet 的 `T` 是时间图，表示架构能接收查询时间。正文主表仍以中点为主；因此章节区分“接口支持”与“训练/测试已覆盖 arbitrary-time”。

### 4.3 Transformer、SSM 与生成路线

| 工作 | 原始来源 | 核验结论 | 等级 |
|---|---|---|---|
| VFIformer | CVPR 2022 PDF、arXiv `2205.07230`、`JIA-Lab-research/VFIformer` | cross-scale window attention；原生只生成 midpoint，多帧靠递归 | E1+E3 |
| VFIT | CVPR 2022 页、arXiv `2111.13817` | content-aware self-attention aggregation；与 VFIformer 是两篇不同论文 | E1 |
| LDMVFI | AAAI 2024 页、arXiv `2303.09508`、作者代码 | latent diffusion 生成单个条件中间帧 | E1+E3 |
| VIDIM | CVPR 2024 页、arXiv `2404.01203` | 低分辨率联合视频 diffusion，再级联空间超分；输出粒度是整段 | E1 |
| BiM-VFI | CVPR 2025 PDF、arXiv `2412.11365`、作者代码 | bidirectional motion field 表达非匀速下的位置和方向，区分 fixed/arbitrary 协议 | E1+E3 |
| EDEN | CVPR 2025 PDF、arXiv `2503.15831`、作者代码 | Transformer tokenizer + temporal DiT + start/end difference；多分辨率与多间隔共同训练 | E1+E3 |
| LDF-VFI | CVPR 2026 PDF、arXiv `2601.14959`、作者代码 | 自回归 DiT、Local Diffusion Forcing、局部注意、tiled VAE；面向 2×–16× 长序列 | E1+E3 |
| SPEED | arXiv `2607.15585` | 一步 pixel diffusion；作者标注 ACM MM 2026，正式页未在本轮独立核验 | E2 |
| SNM-VFI | arXiv `2608.13460` | 预训练 flow 与 video diffusion 的 training-free 组合；“free”只指组合阶段 | E2 |
| MGMVFI | arXiv `2608.22861` | flow-guided serialization + selective SSM/Mamba；属于重建/状态空间路线，不是 diffusion | E2 |

生成式模型学习的是条件分布，不等于取消约束。两张观测端点在 VFI 中仍是硬条件：候选路径可以多解，但任何样本都要保持端点身份、几何、文字和物体数量，并满足时间顺序。若模型允许改变端点语义，任务已经滑向 keyframe transition generation，不能继续只用 VFI 的单参考 PSNR 命名。

## 5. 首发年份与正式年份裁决

| 工作 | arXiv v1 | 正式出版 | 裁决说明 |
|---|---:|---:|---|
| SepConv | 2017-08 | ICCV 2017 | 同年首发/正式 |
| Super SloMo | 2017-11 | CVPR 2018 | 任意时间组合与多帧证据来自正式论文 |
| DAIN | 2019-04 | CVPR 2019 | depth-aware projection |
| AdaCoF | 2019-07 | CVPR 2020 | 首发与正式跨年 |
| SoftSplat | 2020-03 | CVPR 2020 | softmax splatting 首发 |
| RIFE | 2020-11 | ECCV 2022 | 不写成“2022 才提出” |
| FILM | 2022-02 | ECCV 2022 | 大运动特征匹配 |
| VFIformer | 2022-05 | CVPR 2022 | arXiv ID `2205.07230` |
| IFRNet | 2022-05 | CVPR 2022 | arXiv ID `2205.14620`；CNN 路线 |
| LDMVFI | 2023-03 | AAAI 2024 | 不写成 AAAI 2023 |
| AMT | 2023-04 | CVPR 2023 | 卷积/相关体路线 |
| VIDIM | 2024-04 | CVPR 2024 | 整段 diffusion |
| BiM-VFI | 2024-12 | CVPR 2025 | 非匀速运动 |
| EDEN | 2025-03 | CVPR 2025 | DiT VFI |
| LDF-VFI | 2026-01 | CVPR 2026 | 长序列自回归 DiT |
| SPEED | 2026-07 | 作者标注 ACM MM 2026 | 本轮仅按预印本证据写作 |
| SNM-VFI | 2026-08 | 作者标注 ECCV Workshop 2026 | 本轮仅按预印本证据写作 |
| MGMVFI | 2026-08 | 作者标注 ECCV 2026 | 本轮仅按预印本证据写作 |

## 6. 数据集与指标的协议裁决

### 6.1 数据集

| 名称 | 已核对事实 | 不能省略的协议字段 | 证据 |
|---|---|---|---|
| Vimeo90K | 项目页总计 73,171 triplets；常用 split 为 51,312/3,782；septuplet 共 91,701 序列 | triplet/septuplet、抽帧规则、训练去重、中心帧或多 $\tau$ | 官方 TOFlow 项目页，E1 |
| Middlebury | `OTHER` 可本地有 GT；official evaluation 使用隐藏 GT，官网不设唯一默认排名 | OTHER/eval、IE/NIE、crop、上传版本 | 官方 evaluation，E1 |
| UCF101 triplets | VFI 论文常用 DVF 选取的 379 个 256×256 triplets | 文件清单、是否从完整 UCF101 派生 | DVF/后续论文交叉回读，E1 |
| Xiph 2K/4K | 开放编码序列常被抽帧或裁剪作高分辨率测试 | 片段名、frame indices、RGB/Y、crop、边界 | 官方序列 + 各论文协议，E1/E3 |
| SNU-FILM | 1,240 triplets，Easy/Medium/Hard/Extreme | fixed/arbitrary、gap、resize、四档汇报方式 | CAIN 官方代码与后续全文，E1/E3 |
| DAVIS | 原生任务是视频对象分割，VFI 由论文派生 pairs/triplets | 2016/2017、split、gap、resize、是否只选连续镜头 | DAVIS 官方报告 + 方法协议，E1 |
| X4K1000FPS | XVFI 提供 4K/1000-fps 与大位移测试 | XTest4K/crop/entire、倍率、原生或 tiled | XVFI 正式页与项目，E1/E3 |

结论：数据集名字不是协议。尤其 `SNU-FILM-entire` 与 1,240 个 triplet、`X4K-entire` 与中心 crop、Middlebury `OTHER` 与 hidden evaluation 都不能混为一张表。

### 6.2 指标与系统成本

| 指标 | 回答的问题 | 主要盲区 |
|---|---|---|
| PSNR | 与唯一 GT 的均方误差有多小 | 对亚像素错位敏感，多解时奖励条件均值 |
| SSIM | 局部亮度、对比度和结构是否相近 | 仍是单帧/局部参考指标 |
| LPIPS | 预训练特征中的感知距离 | 不保证端点、物理运动或时间一致 |
| FloLPIPS | 感知误差是否集中于不一致运动区域 | 依赖 flow estimator 与实现，仍需公开版本 |
| FVD/视频指标 | 生成序列分布与时间特征是否接近 | 样本量、特征网络和 clip 长度显著影响结果 |
| latency | 固定硬件/实现上的单次或吞吐成本 | 未同步 GPU、含不含 I/O、warm-up、精度不同会失真 |
| peak memory | 给定分辨率/帧数是否可运行 | allocated/reserved、decoder、缓存与 tiling 定义不同 |
| parameter count | 静态权重规模 | 不含 correlation、activation、sampler NFE 和 memory traffic |

BVI-VFI 与 FloLPIPS 说明动态纹理和时间伪影会改变方法排序；2024 benchmark 论文进一步指出 test set、边界裁剪和误差实现不统一。因此正文只描述比较条件，不转录跨论文排行榜。

## 7. 官方实现登记

以下仓库通过论文作者、机构或项目页互证后纳入，不以 GitHub 星数作为质量证据：

- SoftSplat：<https://github.com/sniklaus/softmax-splatting>
- DAIN：<https://github.com/baowenbo/DAIN>
- RIFE：<https://github.com/hzwer/ECCV2022-RIFE>
- IFRNet：<https://github.com/ltkong218/IFRNet>
- FILM：<https://github.com/google-research/frame-interpolation>
- VFIformer：<https://github.com/JIA-Lab-research/VFIformer>
- AMT：<https://github.com/MCG-NKU/AMT>
- EDEN：<https://github.com/MCG-NJU/EDEN>
- BiM-VFI：<https://github.com/KAIST-VICLab/BiM-VFI>
- LDF-VFI：<https://github.com/Xinyu-Peng/LDF-VFI>
- LDMVFI：<https://github.com/danier97/LDMVFI>
- XVFI：<https://github.com/JihyongOh/XVFI>

仓库只用于核查实物与接口；性能数字仍须按正文的系统账本在目标设备复测。

## 8. AI 科学示意图生成与审计

### 8.1 生成目标

图文件：[`assets/diagrams/video-frame-interpolation-dual-route.png`](../assets/diagrams/video-frame-interpolation-dual-route.png)

图要表达一个不能只靠方法列表传达的关系：VFI 有对应/重建与生成/diffusion 两条路线；生成路线允许中间路径多解，但两个观测端点仍是硬条件，两路最终进入同一验证门，并共享遮挡、大运动、重复纹理与曝光变化四类失败源。

### 8.2 初始提示词

> Create a publication-quality scientific schematic for a Chinese technical textbook, but use concise ENGLISH labels. Use a clean 16:9 white canvas and colorblind-safe blue/orange/teal accents. Title: “Video Frame Interpolation: Two Routes, One Verification Gate”. At left show Start frame I0, End frame I1, and t in (0,1). Top route A “Correspondence / Reconstruction”: Feature pyramid -> Flow / correlation / kernel -> Warp or splat -> Occlusion + visibility -> Blend + refine. Lower route B “Generative / Diffusion”: Encode endpoints -> Condition tokens -> Noisy latent or pixels -> DiT denoising -> Decode candidate path. Merge both into one “Verification gate” containing Endpoint fidelity, Motion + temporal coherence, Perceptual detail, Latency + memory. Add warnings: Occlusion, Large motion, Repeated texture, Exposure change. Use flat vector shapes, clear arrows, generous spacing, readable typography, no fake metrics, no logos, no paper names, no decorative imagery.

### 8.3 迭代记录

| 版本 | 修改 | 目视结果 | 处置 |
|---|---|---|---|
| v1 | 初始提示词 | 路线和验证门清楚，但四个 warning 的布局更像只属于生成路线 | 未采用 |
| v2 | 把上路说明改为 `Best when correspondences are reliable; reconstruction target`；加入 `Shared failure cases`；要求 warning 同时适用于两路 | 文字正确、两路和统一验证门清晰；warning 标题明确为共享 | **采用** |
| v3 | 进一步要求 warning 用连线显式指向两路 | 一条虚线连接在视觉上悬空，反而降低关系清晰度 | 拒绝，回退 v2 |

### 8.4 采用文件与完整性

- 生成器：OpenAI image generation tool，2026-08-30。
- 原始生成 artifact：`generated_images/01a04ecd-c660-7292-9a41-f8e598150cd5/exec-99e25358-e33d-4366-8ab5-284fec040efa.png`。
- 项目文件：`assets/diagrams/video-frame-interpolation-dual-route.png`。
- 尺寸：**1672 × 941 px**。
- 格式：PNG，RGB 8-bit，non-interlaced，无 alpha。
- 文件大小：约 1.2 MiB。
- SHA-256：`d7854f1a9354e12e148c2531cad551516f7f250e4317116c89c407884d60768e`。

### 8.5 原图视觉检查

采用文件已按原分辨率回读，不以缩略图代替验收：

- 标题、`I0`、`I1`、`t in (0,1)`、两条路线、统一 verification gate 和四类 failure cases 均可辨认；
- 英文拼写正确，没有伪造数字、论文名、logo 或不可核验的性能主张；
- 箭头顺序清楚，框体无重叠和截断，颜色在灰度下仍由实线/虚线和空间位置提供冗余；
- v2 的 warning 虚线视觉上靠近下路，因此正文 alt、图注和六步文字替代明确说明它们是两路共享失败源；
- 图注同时写明：生成路线可以输出多个合理中间路径，但端点忠实度仍是硬门槛，不允许以“生成自由度”掩盖身份或几何漂移。

## 9. 综合判断与剩余不确定性

### 9.1 已可高置信陈述

- IFRNet 是 encoder–decoder CNN；AMT 是 correlation-driven convolutional model，二者都不应归入 Transformer。
- VFIformer 与 VFIT 是 CVPR 2022 的两篇独立论文；VFIformer 本身只做 midpoint，递归才得到多帧。
- SoftSplat 的核心是可微的、重要性归一化 forward collision handling；DAIN 把 inverse depth 写入 flow projection。
- RIFE、IFRNet、BiM-VFI 的时间接口/训练协议不同，不能只看是否有 $\tau$ 输入就判定 arbitrary-time 证据等价。
- diffusion/DiT 路线改变的是从点估计到条件分布的建模方式，不改变端点是已观测硬条件这一事实。
- triplet 单帧分数与 entire-sequence、4K、递归高倍率指标不能直接横排。

### 9.2 仍需保留的边界

- SPEED、SNM-VFI、MGMVFI 在冻结日较新；正式会场、最终代码和独立复现可能继续变化。
- Xiph、DAVIS 在 VFI 中没有一个跨论文完全统一的官方 protocol；必须跟随每篇论文公开文件清单。
- FLOP、时延和峰值显存未在本轮统一硬件复测，章节不宣称任何一个模型具有跨平台绝对最快地位。
- 生成式 VFI 的“合理多解”尚缺统一的人类偏好、端点守恒与多样性联合协议；单参考 LPIPS 只能覆盖其中一部分。

## 10. 交付前验证记录

2026-08-30 冻结候选已完成以下实际检查：

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Markdown lint | `npx --yes markdownlint-cli` 对正文和本日志执行 | **PASS**，0 error |
| 引用锚点 | 脚本核对 `href=#ref-*`、`<a id=...>`、重复/缺失/未使用 | **PASS**，32 个定义唯一，32 个均被引用 |
| 相对链接 | 解析两个 Markdown 的本地 Markdown links 并按各自目录 resolve | **PASS**，0 broken relative link |
| Mermaid | 抽取正文 3 个 Mermaid block，使用 Mermaid CLI 与本机 Chrome 逐个渲染 SVG | **PASS**，3/3 生成非空 SVG |
| 图片 | `file`、原图回读、`shasum -a 256` | **PASS**，1672×941 RGB PNG，SHA 与本记录一致 |
| whitespace | `git diff --check` 限定本章、本日志和新图 | **PASS**，0 whitespace error |

临时 SVG 仅用于渲染验收，位于系统临时目录，不属于项目交付物。最后一次日志回填后再次执行 lint 与 `git diff --check`，结果仍需保持为 0 error。
