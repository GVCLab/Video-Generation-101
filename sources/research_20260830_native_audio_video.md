# 原生音视频联合生成研究日志（2026-08-30）

## 1. 范围与冻结点

- **目标章节：** `docs/tasks/native-audio-video-generation.md`
- **图片资产：** `assets/diagrams/native-audio-video-generation-contract.png`
- **检索/核验日：** 2026-08-30（Asia/Shanghai）
- **核心问题：** 哪些系统真正同时建模未知音频与未知视频，哪些只是 V2A、A2V、共同条件、级联或产品工作流？
- **冻结规则：** 论文、仓库、权重、模型卡和产品能力分别按当日页面记录；后续版本不得倒灌。

本日志用于复核检索、取舍、断言与制图，不替代全库 bibliography。

## 2. 检索问题与策略

### 2.1 研究问题

1. “原生联合”可由哪些结构或采样证据操作性判定？
2. 双塔、共享 attention、joint token/latent 与产品输出之间是什么包含关系？
3. 同步来自真实时间编码、cross-attention、显式 prior、辅助任务、CFG、codec 还是搜索？
4. 2025–2026 前沿在架构、表示、流式与 inference-time scaling 上分别解决什么问题？
5. 哪些能力可从代码/权重独立运行，哪些仍只是作者论文或官方发布声明？

### 2.2 查询模板

2026-08-30 使用下列组合做发现性检索，再回到正式 proceedings、arXiv 原文、官方项目/仓库/模型卡逐项回读：

```text
"joint audio video generation" diffusion transformer
"native audio visual" generation joint denoising
"audio video" single stream packed tokens generation
"audio video generation" cross-modal recurrent memory streaming
"inference-time scaling" "joint audio-video"
site:openaccess.thecvf.com audio video generation CVPR ICCV
site:openreview.net "audio-video generation"
site:github.com <exact model name> official
site:huggingface.co <exact model name> model card
```

补充定点检索：`MM-Diffusion`、`Seeing and Hearing`、`Movie Gen Audio`、`MMAudio`、`AV-Link`、`JointDiT`、`JavisDiT`、`Ovi`、`LTX-2`、`Harmony`、`NAVA`、`OmniVAE`、`Ripple`、`VABench`、`MiniMax H3`。

### 2.3 来源优先级

1. 正式会议/期刊正文；
2. 作者 arXiv 原稿；
3. 官方项目、代码仓库与模型卡；
4. 机构产品发布，只支持提供方披露；
5. 搜索摘要、媒体、聚合站仅用于导航，不承担章节断言。

本次至少使用三类互补一手来源：CVPR/ICCV/ICLR/TMLR 正式论文；作者 arXiv 预印本；官方代码/模型卡/产品发布。

## 3. 纳入、排除与判别规则

### 3.1 纳入

- 能定义 V2A、A2V、共同条件、级联、耦合推理或联合生成的明确边界；
- 方法改变生成器内 AV 交互、codec latent、同步训练、流式 memory 或推理时搜索；
- 2025–2026 frontier 有可核验论文或官方发布面；
- benchmark 能把视频、音频、语义、时序、声源/空间分开评价。

### 3.2 排除/降级

- 只有“视频带声音”宣传，没有结构或采样披露：保留为产品能力，架构记未知；
- 只生成一侧：归入 V2A/A2V 对照，不写原生联合；
- 同一 prompt 分别调用两个模型：归共同条件；
- 先视频后音频：归 staged，即使最终同步很好；
- 共用词表/模型但任务是单向条件：只记表示统一；
- 论文列出代码但冻结日入口未公开：发布面记未核验；
- 作者基准、速度与“SOTA”只写作者报告，不转写为独立结论。

### 3.3 证据等级

| 级别 | 条件 | 章节语气 |
|---|---|---|
| A | 正式 proceedings/期刊 | “论文提出/报告；发表于……” |
| B | 作者预印本 | “作者提出/报告；截至冻结日为预印本” |
| C | 机构技术报告 | “报告披露” |
| D | 官方代码、模型卡或产品发布 | “提供方发布/声明” |
| S | 本章综合合同与复现规范 | “本章要求/定义” |

## 4. 证据台账

### 4.1 边界对照

| 工作 | 一手入口 | 状态 | 本章采用的事实 | 边界 |
|---|---|---|---|---|
| TPoS | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Jeong_The_Power_of_Sound_TPoS_Audio_Reactive_Video_Generation_with_ICCV_2023_paper.html) | A | 音频语义/幅度条件化视频 | A2V，音频不是输出 |
| Movie Gen | [技术报告](https://arxiv.org/abs/2410.13720) | C | 30B video 与 13B Audio 是模型家族；Audio 读视频特征 | staged/V2A，不是单 checkpoint 联合 |
| MMAudio | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_MMAudio_Taming_Multimodal_Joint_Training_for_High-Quality_Video-to-Audio_Synthesis_CVPR_2025_paper.html)、[代码](https://github.com/hkchengrex/MMAudio) | A+D | 视频/文本→音频；多类数据 joint training | joint training 不等于 joint output |
| AV-Link | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Haji-Ali_AV-Link_Temporally-Aligned_Diffusion_Features_for_Cross-Modal_Audio-Video_Generation_ICCV_2025_paper.html) | A | 同框架支持 A2V/V2A、时间对齐融合 | 每次仍有固定条件侧 |
| Seeing and Hearing | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.html)、[代码](https://github.com/yzxing87/Seeing-and-Hearing) | A+D | 冻结生成器在推理期由 ImageBind aligner 耦合 | 仓库当前只有 V2A 主路径；非联合训练 backbone |
| VideoPoet | [ICML 2024](https://proceedings.mlr.press/v235/kondratyuk24a.html) | A | 统一离散 token 与多任务自回归接口 | 统一 token 不自动证明同时联合采样 |

### 4.2 联合 backbone 主线

| 工作 | 一手入口 | 状态 | 架构/同步证据 | 发布面 |
|---|---|---|---|---|
| MM-Diffusion | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ruan_MM-Diffusion_Learning_Multi-Modal_Diffusion_Models_for_Joint_Audio_and_Video_CVPR_2023_paper.html)、[代码](https://github.com/researchmm/MM-Diffusion) | A+D | coupled U-Net、random-shift AV attention、双输出扩散 | 训练/推理/checkpoint 开放 |
| JointDiT | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Animate_and_Sound_an_Image_CVPR_2025_paper.html)、[项目](https://anonymoushub4ai.github.io/JointDiT/) | A+D | I2SV；预训练 AV experts + joint block + JointCFG | 项目样例；未核验正式代码/权重 |
| JavisDiT | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e11afe29671baa67c9f15fa77fc97357-Abstract-Conference.html)、[代码](https://github.com/JavisVerse/JavisDiT) | A+D | 双分支、ST-prior 与双向交互 | 训练/推理/模型/数据开放；主线已升级 ++ |
| Ovi | [作者稿](https://arxiv.org/abs/2510.01284)、[代码](https://github.com/character-ai/Ovi) | B+D | twin backbone、逐块双向 cross-attention、RoPE 速率缩放 | 5s/10s checkpoint 与推理开放；训练脚本 TODO |
| LTX-2 | [作者稿](https://arxiv.org/abs/2601.03233)、[代码](https://github.com/Lightricks/LTX-2) | B+D | 14B/5B 非对称双流、双向 AV cross-attention、时间 RoPE/AdaLN | 当前仓库已演进到后续 checkpoint；含 trainer |
| Harmony | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hu_Harmony_Harmonizing_Audio_and_Video_Generation_through_Cross-Task_Synergy_CVPR_2026_paper.pdf) | A | joint+A2V+V2A cross-task training、local/global interaction、SyncCFG | 本章未把项目样例等同完整复现 |
| NAVA | [作者稿](https://arxiv.org/abs/2605.30073)、[代码](https://github.com/ernie-research/NAVA)、[模型卡](https://huggingface.co/baidu/NAVA) | B+D | 前 10 层对齐、后 20 层共享；context 外部注入 | 训练/推理/权重/Gradio 开放 |
| MiniMax H3 | [发布](https://www.minimax.io/blog/minimax-h3)、[模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md)、[代码](https://github.com/MiniMax-AI/MiniMax-H3) | D | 33B dense single stream，packed multimodal sequence，联合预测独立 AV latent | 768p Base 开；Context-IR/2K/稀疏 attention 未开 |

### 4.3 表示、流式、搜索与评测

| 工作 | 一手入口 | 状态 | 本章采用的事实 | 边界 |
|---|---|---|---|---|
| OmniVAE | [作者稿](https://arxiv.org/abs/2607.23855)、[代码](https://github.com/OpenMOSS/OmniVAE) | B+D | segment InfoNCE + semantic distillation；独立 AV encoder/decoder | 对齐的双 latent，不是单一 generator |
| ITS-AVGen | [TMLR](https://openreview.net/forum?id=MHNFjjm5nO)、[代码](https://github.com/kaistmm/ITS-AVGen) | A+D | Best-of-N/EvoSearch、多 verifier、ARW；讨论 verifier hacking | 不改 base 权重，付出多候选成本 |
| Ripple | [作者稿](https://arxiv.org/abs/2607.26818) | B | fixed window + recurrent AV memory + distill/RL；作者报 480p 约 28 FPS | 未核验代码/权重；速度仅作者协议 |
| VABench | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_VABench_A_Comprehensive_Benchmark_for_Audio-Video_Generation_CVPR_2026_paper.html) | A | T2AV/I2AV/stereo、15 维、7 类内容 | 自动指标与小规模人工验证仍有边界 |

### 4.4 产品能力与架构边界

| 产品/发布 | 一手入口 | 允许写 | 不允许写 |
|---|---|---|---|
| Veo | [官方模型页](https://deepmind.google/models/veo/) | 官方提供同步音频能力 | 未披露的联合 latent/backbone |
| Sora 2 | [官方发布](https://openai.com/index/sora-2/) | 发布时提供同步对白/音效能力 | 由样片反推训练因子化 |
| Seedance 2.0 | [官方预印本](https://arxiv.org/abs/2604.14148) | 官方团队披露的系统能力 | 把服务链自动等同单 checkpoint |
| MiniMax H3 | [官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) | 已披露 Base 架构和本地 768p 面 | 把 hosted Context-IR/2K 说成完全开放 |

## 5. 关键裁决记录

1. **MMAudio：** 标为 V2A。题名中的“joint training”指 AVT 与 AT 数据联合训练，不改变输出只有音频的事实。
2. **AV-Link：** 标为双向条件框架。支持 A2V 与 V2A 不等于一次联合采样。
3. **Seeing and Hearing：** 标为推理期耦合。两套冻结 LDM 由外部 latent aligner 连接，不写原生联合训练。
4. **VideoPoet：** 标为统一 token/多任务祖先。不由统一词表推导 T2AV 同步联合轨迹。
5. **Ovi：** 原论文 5s/720p 与仓库 Ovi 1.1 10s/960 作为两个 release surface。
6. **LTX-2：** 论文 checkpoint 与当前仓库默认后续版本分开；repo 能支持训练/微调，不倒灌新能力。
7. **NAVA：** 标为预印本 + 完整官方 release；架构/训练量来自作者披露，benchmark 不改写为独立 SOTA。
8. **OmniVAE：** “jointly trained AV VAE”仍保留独立 AV encoder/decoder，deployment 不交互；称对齐双 latent，不称单一联合 latent。
9. **Ripple：** “约 28 FPS/480p/单 H100”始终加作者协议限定；没有 TTFF/p95/full-pipeline 独立测量。
10. **MiniMax H3：** 官方模型卡足以支持 single-stream packed AV 与 release surface；没有正式论文，证据仍为 D。

## 6. 强断言—来源映射

| 强断言 | 直接一手证据 | 章节处理 |
|---|---|---|
| MM-Diffusion 是早期明确联合 AV diffusion | CVPR 2023 正文 | 限定为论文框架起点，不泛化现代质量 |
| MMAudio 只生成音频 | CVPR 2025 任务定义与官方 README | 用作命名误读反例 |
| Movie Gen 是模型家族/音频侧读视频 | 技术报告架构章节 | 归 staged V2A |
| Ovi/LTX-2 逐层双向交换 | 两篇作者稿 method | 归双支路原生联合 |
| NAVA align-then-fuse 10+20 层、6.3B | 作者稿 + 官方 repo/card | 写作者披露并保留预印本级别 |
| H3 33B dense single stream、独立 AV VAE | 官方模型卡 architecture | 写官方披露，不称同行评议事实 |
| Ripple 480p ~28 FPS | arXiv 正文及硬件条件 | 写作者在单 H100 协议报告 |
| ITS 单 verifier 有不对称/投机风险 | TMLR 正文 | 只支持其 JavisDiT/MMDisCo 实验范围 |
| VABench 15 维/7 类/T2AV-I2AV-stereo | CVPR 2026 正文 | 用作评测框架，不冻结排行榜 |

## 7. 图片生成与审计

### 7.1 目标

用一张低文字 16:9 教学图直接排除五种常见混淆：V2A、A2V、共同条件、级联与原生联合。只在 `JOINT` 行显示同一时间轴上的重复双向交换；颜色只辅助区分视频/音频，不承担语义。

### 7.2 最终 prompt（逐字记录）

```text
Create a publication-quality scientific educational schematic, exact 16:9 landscape PNG, clean warm-white background, flat vector style, crisp lines, generous whitespace, color-blind-safe Okabe-Ito palette, no gradients, no decorative 3D, no photorealism. Title: “AUDIO–VIDEO GENERATION CONTRACT”. Organize five horizontal rows, each with a small video-frame icon (blue) and audio-waveform icon (orange), and only these short row labels in large sans-serif text: “V→A”, “A→V”, “SHARED C”, “STAGED”, “JOINT”. Row 1: a solid one-way arrow from a fixed video frame to a generated audio waveform. Row 2: a solid one-way arrow from a fixed audio waveform to generated video frames. Row 3: a neutral condition node C splits into two independent branches to video and audio, with no bridge between branches. Row 4: condition C generates video first, then video generates audio as a sequential chain. Row 5: highlight with a pale green rounded rectangle; show noisy video-latent tokens and noisy audio-latent tokens evolving together along one left-to-right time ruler, with repeated bidirectional cross-modal arrows at several denoising stages, then decoded video and audio outputs. Add a compact right-side verification gate with four icon-only checks: semantic match, event timing, source binding, unimodal quality. Add one small bottom warning badge with exact text “SOUND ≠ JOINT”. Scientifically precise: do not draw bidirectional arrows in any row except JOINT; do not imply shared condition or staged generation is native joint modeling. Keep all text exactly as specified, spell it correctly, do not add model names, equations, citations, numbers, legends, logos, watermarks, or extra labels.
```

### 7.3 工件记录

- 生成器：OpenAI 内置 image generation；2026-08-30 生成。
- 项目路径：`assets/diagrams/native-audio-video-generation-contract.png`
- 原始尺寸：**1672 × 941 px**，宽高比约 1.777（16:9）。
- SHA256：`9f59677d6fdb3843a7b47a09841169e914a24eb381f7be0961e267c4ab81b7b6`
- 拒绝版本：**无**；首版满足科学边界、文字和构图要求，直接接收。
- 生成器额外加入 `noisy video-latents` / `noisy audio-latents` 两个解释性标签；它们准确且未造成文字拥挤，保留。

### 7.4 视觉检查

- **原图：** 五行标签均清楚；V2A/A2V 单向，`SHARED C` 无支路桥，`STAGED` 为先视频后音频，只有 `JOINT` 有四处双向箭头；底部警示文字正确，无模型名、logo 或水印。
- **灰度：** 使用 ImageMagick 转为 8-bit grayscale 临时图后按原始分辨率检查；箭头、行分隔、latent 深浅序列、绿色框在灰度中仍由轮廓和布局可辨。灰度派生图只用于审计，未纳入仓库。
- **边界：** 图片是本章综合示意，不代表任一具体论文架构；`JOINT` 行只表达判别合同，不承诺双向 attention 一定被模型有效使用。

### 7.5 Mermaid 等价性

章节 Mermaid 与 PNG 表达同一判据，但 Mermaid 增加了“推理期耦合/产品未知”分支。两者都有 `accTitle`、`accDescr` 和顺序化文字替代；图片负责快速识别，Mermaid 负责无歧义决策路径。

## 8. 局限与更新触发器

- NAVA、Ovi、LTX-2、OmniVAE、Ripple 的冻结状态含预印本；正式版出现后需重核页码、方法和限制。
- Ovi、LTX、JavisDiT 和 H3 仓库持续更新；release surface 应以 commit/checkpoint hash 重冻。
- 商业服务可能变更名称、可用性、模型版本与后处理；产品页面只能支持访问日声明。
- 论文的同步器、reward 与 benchmark 可能共享训练域，不能当独立真值。
- Ripple 的 streaming、NAVA 的规模和 H3 的产品性能尚未在本任务中做本地运行或独立复现。

触发更新的事件包括：正式 proceedings 替代预印本；Ripple 发布官方代码/权重；H3 发布技术报告、Context-IR/2K/稀疏 attention；独立复现推翻作者速度/同步结论；VABench evaluator 或 API 版本发生实质变化。

## 9. 验证记录

以下均为文件定稿后的实际执行结果，不以计划代替执行。

- 引用锚点闭合：通过；正文引用 33 个唯一编号、参考文献 33 个唯一锚点，缺失与孤立锚点均为 0。
- 仓库内相对链接：通过；章节到研究日志与 PNG 的 2 个链接均存在。
- 外部一手链接：通过；34 个唯一 URL 中 23 个经 `curl -L` 返回 HTTP 200，10 个因本机 LibreSSL/超时未取得状态码、1 个返回 HTTP 403，后 11 个均逐一经浏览器成功打开。
- Markdown lint：通过；`markdownlint-cli` 0 条错误。
- Mermaid 实际渲染：通过；`@mermaid-js/mermaid-cli` 11.12.0 使用本机 Chrome 生成有效 SVG，且产物含 `aria-labelledby` 与 `aria-describedby`。
- PNG 尺寸/SHA256：已验证，见 7.3。
- 原图与灰度视觉检查：已完成，见 7.4。
- `git diff --check`：通过，无空白错误。
