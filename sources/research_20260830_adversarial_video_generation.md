# 视频对抗生成检索、筛选与证据记录

> 检索与证据冻结：2026-08-30（Asia/Shanghai）。本文是 [视频对抗生成章](../docs/generative-models/adversarial-generation.md) 的可复核工作台，不是独立的统计 meta-analysis。

## 1. 研究问题和判定标准

本轮不用“新模型是否超过旧模型”组织证据，而围绕五个可判定问题：

1. 完整视频 GAN 的 generator 与 discriminator 如何从固定短 clip 演化到连续时间、高分辨率与长视频？
2. frame、spatiotemporal、multi-scale 和 conditional discriminator 各自看到什么，各自漏掉什么？
3. 对抗目标在 video tokenizer/decoder 中是主生成目标，还是重建子系统的感知约束？
4. ADD、DMD、DMD2、SDXL-Lightning、T2V-Turbo、AnimateDiff-Lightning、OSV、CausVid、SnapGen-V、Seaweed-APT、ADM、ASD、V-PAE 和 AAD-1 是否真有显式对抗目标，是否有直接视频证据？
5. mode coverage、FVD 协议、训练稳定与长视频失败应如何分开验收？

本章的**对抗目标操作性判据**是：存在可学习的 critic/判别器，它在对抗分类或评分目标下，区分 reference/positive distribution（真实数据或 teacher/self-teacher 输出）与 student/generated distribution；student 反向更新以迎合该 critic。每个方法必须记录 positive 的真实来源，不能默认它总是真实数据。下列信号不单独满足判据：

* 固定的可微 reward model；
* 只通过 denoising score matching 训练的 fake-score estimator；
* consistency target；
* 方法名称中的 distribution matching、one-step 或 distillation。

## 2. 检索表面、日期和结果数

### 2.1 OpenAlex API：宽口径书目发现

所有查询均限定发表日期 `2014-01-01..2026-08-30`，`per-page=50`。结果数来自 API `meta.count`，“审查数”是本轮实际下载的首页记录数。

| ID | `search` 字符串 | API 总数 | 审查数 | 用途 |
|---|---|---:|---:|---|
| OA1 | `"video generative adversarial network"` | 159 | 50 | 历史视频 GAN、长视频、评测与应用候选 |
| OA2 | `"adversarial diffusion distillation"` | 496 | 50 | 图像/视频蒸馏与后训练候选 |
| OA3 | `"one step video generation adversarial"` | 0 | 0 | 测试精确短语索引的召回边界 |
| OA4 | `"video tokenizer adversarial loss"` | 0 | 0 | 测试 tokenizer 相关精确短语召回 |

OA3/OA4 的 0 是精确短语与索引的失配，不是“没有相关工作”的证据。因此随后改用 arXiv 字段查询和已知标题的正式论文页。

### 2.2 arXiv API：预印本与标题/摘要发现

| ID | API 查询 | 总结果 | 审查数 | 用途 |
|---|---|---:|---:|---|
| AX1 | `all:"video generation" AND all:"generative adversarial"` | 74 | 50 | 视频 GAN 历史与后续变体 |
| AX2 | `all:"adversarial diffusion distillation"` | 6 | 6 | ADD、Lightning 与直接蒸馏候选 |
| AX3 | `all:"one-step video generation"` | 4 | 4 | 一步视频生成的直接候选 |
| AX4 | `all:"video tokenizer" AND all:"adversarial"` | 1 | 1 | 对抗重建在 tokenizer 中的证据线索 |

### 2.3 Semantic Scholar API：失败也记录

2026-08-30 对下列四个查询调用 Semantic Scholar Graph API：

1. `video generative adversarial network`
2. `adversarial diffusion distillation video`
3. `one-step video adversarial generation`
4. `video tokenizer adversarial perceptual loss`

四次均返回 HTTP 429，因此本轮**没有**将 Semantic Scholar 结果数或排名当作证据。为了不把限流误写成零结果，书目补充转用 OpenAlex，最终科学主张仍回到正式 proceedings/论文原文。

### 2.4 正式论文页与作者官方项目/代码

定向标题核对使用了 NeurIPS Proceedings、CVF Open Access、ECVA、PMLR、ICLR Proceedings/OpenReview、AAAI OJS 和 arXiv 原文页。这些页面用于核对标题、作者、venue、年份、方法图与优化式。

官方实现/项目检查用于确认释放状态和实现边界，不单独支撑性能主张：

| 工作 | 官方实现/项目 | 本轮使用方式 |
|---|---|---|
| MoCoGAN | [sergeytulyakov/mocogan](https://github.com/sergeytulyakov/mocogan) | 检查 content/motion 代码谱系，性能回到 CVPR 论文 |
| TGANv2 | [pfnet-research/tgan2](https://github.com/pfnet-research/tgan2) | 核对 train sparsely/generate densely 实现归属 |
| MoCoGAN-HD | [snap-research/MoCoGAN-HD](https://github.com/snap-research/MoCoGAN-HD) | 核对预训练图像 generator 与 latent trajectory 路线 |
| DIGAN | [sihyun-yu/digan](https://github.com/sihyun-yu/digan) | 核对 ICLR 2022 工作身份，不误标 venue |
| StyleGAN-V | [universome/stylegan-v](https://github.com/universome/stylegan-v) | 核对 CVPR 2022 和连续时间设定 |
| LongVideoGAN | [NVlabs/long-video-gan](https://github.com/NVlabs/long-video-gan) | 核对层级长视频生成路线 |
| MAGVIT | [google-research/magvit](https://github.com/google-research/magvit) | 核对 tokenizer 与 masked generator 分阶段训练 |
| T2V-Turbo | [Ji4chenLi/t2v-turbo](https://github.com/Ji4chenLi/t2v-turbo) | 确认 reward-guided consistency 路线，不误归为 GAN |
| AnimateDiff-Lightning | [ByteDance 官方模型页](https://huggingface.co/ByteDance/AnimateDiff-Lightning) | 确认公开权重与应用范围，论文状态仍是 arXiv |
| CausVid | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | 确认为 causal video diffusion 实现，对抗分类仍以 method/loss 为准 |
| SnapGen-V | [SnapGen-V 官方项目](https://snap-research.github.io/snapgen-v/) | 核对移动端部署设定，速度仅作者协议 |
| AAD-1 | [AAD-1 官方项目](https://aad-1.github.io/) | 核对 DMD warm-up 与 holistic video discriminator；venue 另查 ICML 官方 program |

上述至少构成四种互补检索表面：书目索引、预印本 API、正式出版页和官方代码/项目。

## 3. 筛选、去重与排除

### 3.1 纳入规则

* 完整 GAN 路线：必须在视频上训练/评测，并对 generator 时间表示、critic 视野、分辨率扩展或长视频问题有可复用的设计变化。
* tokenizer/decoder 路线：必须是视频 tokenizer 或视频重建证据，不仅是图像 VQGAN 的二次转述。
* 蒸馏/后训练路线：同时标注“是否显式对抗”和“是否直接视频”；图像方法仅作机制前驱，不作视频性能证据。
* 评测：纳入 FVD 原始定义、FVD 时间敏感性的受控研究和覆盖度分解工具。
* 主张优先使用 A/B 级一手来源；C 级只证明实现或发布状态；D 级只用于发现。

### 3.2 排除或降级规则

| 情形 | 处理 | 代表例 |
|---|---|---|
| survey、新闻、榜单、搜索摘要代替原文 | 不进入最终科学引用 | 所有性能与方法主张回到 proceedings/arXiv 原文 |
| 只有图像实验 | 可作机制前驱，“直接视频证据”标为否 | ADD、DMD、DMD2、SDXL-Lightning |
| 方法名含 distribution matching/consistency 但无真假 critic | 不归为对抗 | DMD、T2V-Turbo、CausVid |
| 会议 submission 无法确认接收 | 按预印本/技术报告引用 | DVD-GAN 按 arXiv 2019，不写 ICLR 2020 接收 |
| 预印本投稿页无接收证据 | 保留预印本状态 | SAVP 按 arXiv 2018 |
| 会议 proceedings 尚未索引 | 只在官方 program 独立确认后写 venue，并保留说明 | AAD-1 由 [ICML 2026 official program](https://icml.cc/Downloads/2026) 确认；截止日 proceedings 页待建 |
| 作者速度/质量描述 | 保留硬件、帧数、分辨率和实现限定，不改写为普遍事实 | SnapGen-V、Seaweed-APT、OSV、AAD-1 |

### 3.3 结果流

* OpenAlex 实际下载 100 个结果槽位，arXiv 实际审查 61 个结果槽位，合计 161。这是跨表面的返回槽位数，不是去重后独立论文数。
* 对标题/摘要去重并按上述规则筛选；再用历史必要标题、正式 proceedings 和官方项目做 citation chasing。
* 最终正文使用 32 个核心一手学术来源：其中图像侧前驱仍在分类表中显式标为“无直接视频证据”。
* 因为定向 citation chasing 会引入 API 首页之外的历史工作，且两个 API 表面彼此重叠，本记录不将 161 冒充为 PRISMA 意义上的唯一记录总数，也不伪造精确的排除篇数。

## 4. 证据等级

| 等级 | 定义 | 允许支撑的主张 | 不允许的用法 |
|---|---|---|---|
| A | 同行评审正式 proceedings/期刊一手论文 | 方法、实验协议、作者限定的结果 | 不把作者协议结果写成跨数据/硬件定理 |
| B | 作者 arXiv 预印本/技术报告 | 方法定义和作者报告实验 | 不误写为已接收 venue；性能保留未独立复现边界 |
| C | 作者/机构官方项目、代码、模型页或会议官方 program | 发布、实现入口、checkpoint、venue program 状态 | 不独立支撑方法有效性或排名 |
| D | OpenAlex/Semantic Scholar/通用搜索结果 | 候选发现与 citation chasing | 不作最终科学主张的唯一证据 |

正文的方法与性能主张使用 A/B 级；C 级用于公开实现或 AAD-1 的 ICML program 状态；D 级不单独落入正文参考文献。

## 5. 核心一手来源 registry

ID 与正文 `ref-N` 一一对应。作者全名、venue 与年份均从所列一手页面核对，正文参考文献未用 `et al.` 省略作者。

| ID | 一手来源 | 状态/等级 | 支撑的主张 | 必须保留的边界 |
|---|---|---|---|---|
| P01 | [Generative Adversarial Nets](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html) | NeurIPS 2014，A | minimax GAN 概念目标 | 图像基础，未定义视频 critic |
| P02 | [Generating Videos with Scene Dynamics](https://proceedings.neurips.cc/paper_files/paper/2016/hash/04025959b191f8f9de3f924f0940515f-Abstract.html) | NeurIPS 2016，A | 静态背景 + 前景/mask 的 VGAN | 短、低分辨率证据 |
| P03 | [Temporal Generative Adversarial Nets with Singular Value Clipping](https://openaccess.thecvf.com/content_iccv_2017/html/Saito_Temporal_Generative_Adversarial_ICCV_2017_paper.html) | ICCV 2017，A | temporal generator 到 frame latent，再由 image generator 生帧 | 不写成 NeurIPS/ICLR |
| P04 | [MoCoGAN](https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html) | CVPR 2018，A | 固定 content + RNN motion + image/video critics | 解耦是归纳偏置，非因果可识别证明 |
| P05 | [Stochastic Adversarial Video Prediction](https://arxiv.org/abs/1804.01523) | arXiv 2018，B | VAE + GAN 的随机条件视频预测 | 未确认正式接收；不是通用文生视频 |
| P06 | [Adversarial Video Generation on Complex Datasets](https://arxiv.org/abs/1907.06571) | 2019 技术报告，B | DVD-GAN 的高清稀疏空间 critic + 低清全时长 critic | 不标为已接收 ICLR 2020 |
| P07 | [Train Sparsely, Generate Densely](https://doi.org/10.1007/s11263-020-01333-y) | IJCV 2020，A | TGANv2 的稀疏训练/密集生成 | 密集时刻仍需独立评测 |
| P08 | [A Good Image Generator Is What You Need for High-Resolution Video Synthesis](https://openreview.net/forum?id=6puCSjH3hwA) | ICLR 2021，A | MoCoGAN-HD 复用预训练 image GAN，学 latent trajectory | 不是通用长视频解法 |
| P09 | [Generating Videos with Dynamics-aware Implicit GANs](https://openreview.net/forum?id=Czsdv-S4-w9) | ICLR 2022，A | DIGAN 坐标型连续视频与 dynamics-aware critic | venue 是 ICLR 2022，不是 CVPR |
| P10 | [StyleGAN-V](https://openaccess.thecvf.com/content/CVPR2022/html/Skorokhodov_StyleGAN-V_A_Continuous_Video_Generator_With_the_Price_Image_Quality_CVPR_2022_paper.html) | CVPR 2022，A | 连续时间运动表示 + holistic discriminator | venue 是 CVPR 2022；任意时间查询不证明无限长一致 |
| P11 | [Generating Long Videos of Dynamic Scenes](https://papers.nips.cc/paper_files/paper/2022/hash/ce208d95d020b023cba9e64031db2584-Abstract-Conference.html) | NeurIPS 2022，A | LongVideoGAN 低清长序列 + 按帧超分 | 长视频仍需状态与因果评测 |
| P12 | [PV3D](https://openreview.net/forum?id=o3yygm3lnzS) | ICLR 2023，A | 3D-aware portrait video GAN 的垂直域继续演化 | 头像偏置不外推到通用生成 |
| P13 | [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5950_ECCV_2022_paper.php) | ECCV 2022，A | TATS 的 3D-VQGAN tokenizer + 长序列 prior | tokenizer 与 prior 的贡献需拆分 |
| P14 | [MAGVIT](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html) | CVPR 2023，A | 视频 tokenizer 中的感知/对抗重建，再训 masked generator | codec 用 critic 不等于上层 generator 是 GAN |
| P15 | [Which Training Methods for GANs Do Actually Converge?](https://proceedings.mlr.press/v80/mescheder18a.html) | ICML 2018，A | R1 类梯度正则与局部收敛分析 | 理论条件不自动覆盖大视频 Transformer critic |
| P16 | [Towards Accurate Generative Models of Video](https://arxiv.org/abs/1812.01717) | arXiv 2018，B | FVD 定义与初始验证 | FVD 依赖特征与协议 |
| P17 | [On the Content Bias in Fréchet Video Distance](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html) | CVPR 2024，A | FVD 可被空间内容主导，时间敏感性不足 | 不意味 FVD 毫无用处，而是需多指标 |
| P18 | [Adversarial Diffusion Distillation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11557_ECCV_2024_paper.php) | ECCV 2024，A | ADD 的 score distillation + adversarial loss | 图像证据，非视频 |
| P19 | [One-step Diffusion with DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html) | CVPR 2024，A | 真/fake score 差和 regression loss | 原始 DMD 没有二元 GAN loss；图像证据 |
| P20 | [Improved Distribution Matching Distillation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) | NeurIPS 2024，A | DMD2 加入 GAN loss 和 two-time-scale update | 图像证据，不是视频结果 |
| P21 | [SDXL-Lightning](https://arxiv.org/abs/2402.13929) | arXiv 2024，B | progressive adversarial diffusion distillation | 图像预印本证据 |
| P22 | [T2V-Turbo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a57aa8e8b57e64a42e95f7dceb0adb9-Abstract-Conference.html) | NeurIPS 2024，A | reward-guided consistency distillation 的直接视频证据 | 无显式对抗目标 |
| P23 | [AnimateDiff-Lightning](https://arxiv.org/abs/2403.12706) | arXiv 2024，B | 对抗 diffusion distillation 的直接视频证据 | 技术报告，基于 AnimateDiff 分支 |
| P24 | [OSV](https://openaccess.thecvf.com/content/CVPR2025/html/Mao_OSV_One_Step_is_Enough_for_High-Quality_Image_to_Video_CVPR_2025_paper.html) | CVPR 2025，A | 第一阶段 LGP（GAN + Huber）；第二阶段 ACD（teacher-generated positive 对 student negative 的可训练 latent discriminator，adversarial term + consistency distance） | 两阶段均含对抗项；顺序、positive 来源与损失从 camera-ready Fig. 1、Fig. 4 和 Eq. 9–10 核对 |
| P25 | [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) | CVPR 2025，A | CausVid 的 asymmetric DMD 和直接视频实验 | fake score 由 DSM 训练；方法无显式二元对抗损失 |
| P26 | [SnapGen-V](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_SnapGen-V_Generating_a_Five-Second_Video_within_Five_Seconds_on_a_CVPR_2025_paper.html) | CVPR 2025，A | 紧凑架构搜索 + 4-step 对抗微调 | “5 秒视频/5 秒/iPhone 16 Pro Max”是作者设置 |
| P27 | [Diffusion Adversarial Post-Training for One-Step Video Generation](https://proceedings.mlr.press/v267/lin25m.html) | ICML 2025，A | Seaweed-APT 从预训练 diffusion 权重对真实视频做对抗后训练 | 作者实时/720p 主张需绑定硬件协议 |
| P28 | [Adversarial Distribution Matching for Diffusion Distillation](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Adversarial_Distribution_Matching_for_Diffusion_Distillation_Towards_Efficient_Image_and_ICCV_2025_paper.html) | ICCV 2025，A | diffusion-based discriminator；CogVideoX 2B/5B 直接视频实验 | 视频证据是多步，不外推图像一步结论 |
| P29 | [Adversarial Self-Distillation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html) | ICLR 2026，A | $n$ 步学生对 $(n+1)$ 步自教师的一步 causal 对抗自蒸馏 | one-step causal 不等于无限长 streaming |
| P30 | [Phased One-Step Adversarial Equilibrium](https://ojs.aaai.org/index.php/AAAI/article/view/37318) | AAAI 2026，A | V-PAE 的稳定性预热 + self-adversarial equilibrium | 一手页支持 Wan2.1-I2V-14B 上的 single-step I2V，不支持 causal/AR 主张，也不是通用收敛证明 |
| P31 | [AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](https://arxiv.org/abs/2606.03972) / [官方项目](https://aad-1.github.io/) | arXiv 2026，B；ICML program，C | DMD warm-up + causal generator 对 bidirectional holistic video discriminator | ICML 2026 已由官方 program 确认；冻结日时 proceedings 页待索引 |
| P32 | [Assessing Generative Models via Precision and Recall](https://papers.nips.cc/paper_files/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html) | NeurIPS 2018，A | 将保真度和分布覆盖分开的评测思路 | 原始主要是图像特征，视频应选合适特征并说明协议 |

## 6. 容易出错的分类决策

| 工作 | 是否显式对抗 | 是否直接视频 | 决策依据 |
|---|---:|---:|---|
| ADD | 是 | 否 | 论文有 adversarial loss，主实验为图像 |
| DMD | 否 | 否 | score-difference + regression，fake-score 不是二元 critic |
| DMD2 | 是 | 否 | 明确在 DMD 上新增 GAN loss；图像实验 |
| SDXL-Lightning | 是 | 否 | progressive adversarial diffusion distillation；图像预印本 |
| T2V-Turbo | 否 | 是 | consistency + fixed differentiable rewards，无同步真假 critic |
| AnimateDiff-Lightning | 是 | 是 | 视频 motion module 的 progressive adversarial diffusion distillation |
| OSV | 是 | 是 | 第一阶段 LGP（GAN + Huber）；第二阶段 ACD（teacher-generated positive 对 student negative 的可训练 latent discriminator，adversarial term + consistency distance） |
| CausVid | 否 | 是 | asymmetric DMD，fake score 用 DSM；引用 DMD2 更新比不等于使用 GAN loss |
| SnapGen-V | 是 | 是 | 为四步移动端模型设计 dedicated adversarial fine-tuning |
| Seaweed-APT | 是 | 是 | 对预训练视频 diffusion 做真实数据对抗后训练 |
| ADM | 是 | 是 | diffusion-based discriminator，CogVideoX 2B/5B 多步视频证据 |
| ASD | 是 | 是 | adversarial self-distillation 的 causal video student |
| V-PAE | 是 | 是 | phased self-adversarial equilibrium；是 single-step I2V 证据，不是 causal/AR 证据 |
| AAD-1 | 是 | 是 | DMD warm-up 后对 holistic video discriminator 做对抗精修 |

关键反例是 CausVid：论文 related work 中出现 `adversarial distillation`，并从 DMD2 借用两时标更新比，但 method 的 student 由 asymmetric DMD 更新，fake score 网络由 denoising score matching 更新。所以本章标为“直接视频，无显式二元对抗目标”，而不是因为名称相似就自动归类。

## 7. 判别器、损失与正则的证据拆分

| 设计轴 | 一手实例 | 主要作用 | 盲区/必报协议 |
|---|---|---|---|
| Frame/spatial critic | MoCoGAN image D，DVD-GAN spatial D | 纹理、边界、单帧形状 | 可忽略闪烁；报抽帧数与分辨率 |
| Spatiotemporal critic | MoCoGAN video D，MAGVIT 3D D | 局部运动、节奏、闪烁 | 短 clip 不覆盖长期状态；报帧数/FPS/stride |
| Multi-scale critic | DVD-GAN，MoCoGAN-HD | 将高清细节和较长时空视野的成本拆开 | 报每个 critic 权重、输入尺度和更新比 |
| Sparse/holistic temporal critic | StyleGAN-V，AAD-1 | 在稀疏时间点上看整体轨迹 | 稀疏未抽中闪烁和超长事件仍可漏过 |
| Conditional critic | 条件 GAN/I2V 方法 | 同时检查真实与条件匹配 | 报条件注入点、错配负样本和条件遵循指标 |

损失层面分开记录 non-saturating softplus、hinge、score/distillation 与各 critic 加权；正则层面分开 R1（真样本梯度）、插值 gradient penalty、spectral normalization、时空一致数据增强、EMA 与 update ratio。只写“用了 GAN loss + R1”不足以复现，还需每项权重、频率、optimizer 和视频抽样协议。

## 8. FVD、覆盖度与长时验收

### 8.1 FVD 最小可比账本

1. 特征提取器名称、checkpoint、输入归一化和实现版本；
2. 真实/生成样本数，真实统计是否复用，随机种子与重复次数；
3. 分辨率、FPS、clip 帧数、stride、offset、crop/resize 和颜色空间；
4. 长视频如何切窗，是否把同一视频的相关窗当成独立样本；
5. 均值、方差/置信区间，以及是否从多 checkpoint 挑最优。

`FVD_16` 与 `FVD_128` 是不同时间视野的协议，不应直接排名。P17 表明 FVD 有显著 content bias，所以本章不把单一 FVD 当作时间一致的充分证据。

### 8.2 覆盖度与长时失败

除 FVD 外至少需：

* 保真度/precision 与覆盖度/recall 或等价分解；
* 同条件多次采样与分层类别/动作/镜头覆盖；
* nearest-neighbor 和训练集记忆检查；
* 短时闪烁、运动幅度、光流/轨迹与条件遵循；
* 随 rollout 时长递增的身份、物体数量、几何、事件完成、循环与 chunk 边界检查。

连续时间坐标、任意 FPS 查询或 causal attention 都不是长期一致的充分条件。长视频主张必须同时报告训练 critic 视野、测试总时长与上述长程指标。

## 9. 里程碑纳入判据和未解决项

| 类型 | 纳入为里程碑的判据 | 未解决项 |
|---|---|---|
| 完整 GAN | 引入可复用的时间表示、critic 视野或长视频分层，而不只是一个数据集排名 | 高分辨率、覆盖度、训练稳定与长期因果的联合成本 |
| Tokenizer/decoder | 在固定上层 prior 时证明感知重建改善，并不牺牲输入忠实与时间一致 | adversarial hallucination；codec 收益与 latent prior 收益难分 |
| 直接视频蒸馏 | 生成对象、critic 输入和主实验都是视频 | 相同 teacher/data/NFE/decoder/hardware 下对抗项的净贡献 |
| 一步 causal student | 信息访问真的 causal，并在长于训练 clip 的 rollout 上验收 | exposure bias；bidirectional critic 如何稳定指导 causal generator |
| 部署级低延迟 | 报真实 NFE、CFG 额外前向、decoder、I/O、batch、precision、硬件和首块延迟 | 不同论文速度协议不可比；整段快不等于流式首帧快 |

这套判据防止两个过度简化：一是“diffusion 取代 GAN”，二是“少步 diffusion 都是对抗蒸馏”。更符合证据的描述是：对抗学习的作用域发生了迁移，完整 GAN、codec reconstruction critic 与 diffusion/flow student critic 在 2026 年并存。

## 10. 更新触发器

出现以下任一情形时，应重新运行检索并更新正文：

1. AAD-1 的 ICML 2026 proceedings 正式页建立，或作者/标题/页码与 program 不一致；
2. 新视频方法宣称 adversarial/DMD/consistency，但优化式对 critic 更新定义不清；
3. FVD 特征或实现的新受控研究改变现有 content-bias 结论；
4. 有独立复现在相同 teacher/data/hardware 下分离对抗损失、架构压缩和蒸馏目标的净贡献；
5. 长视频 benchmark 能将局部质感、运动、身份、几何、事件进展与覆盖度拆开，并公开可复现协议。
