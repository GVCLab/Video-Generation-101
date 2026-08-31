# 数字人视频生成研究日志（冻结于 2026-08-30）

对应正文：[数字人视频生成：任务边界、条件契约与可审计评测](../docs/tasks/digital-human.md)

## 1. 审计目标与边界

本轮不是声称覆盖全部数字人论文的系统综述，而是一次**有边界、可追溯的深度审计**：

- 修正 Hallo、SadTalker、VASA-1、OmniHuman-1、OmniAvatar、TalkCuts 的首次预印本、正式 venue 与公开资产；
- 将 lip-sync、talking portrait、portrait animation、full-body audio-driven、motion-driven human animation、native audio-video、multi-shot human video 拆成七种条件合同；
- 追踪到 2026-08-30 已能由正式 proceedings、arXiv 版本页、官方项目或官方仓库支持的进展；
- 不以二手榜单、媒体稿、搜索摘要、仓库 star 或项目 demo 代替论文与 release surface 证据；
- 不把作者自称“accepted”、摘要中的“code available”或论文中“will release”自动升级为正式 proceedings 或可下载权重。

冻结规则：网页和仓库状态按 2026-08-30 读取；论文“首次公开”以 arXiv version history 的 v1 日期为准；“正式发表”要求可定位到 CVF、NeurIPS、ICLR/OpenReview、ACM DOI 等正式记录。

## 2. 证据分级

| 等级 | 可支持的断言 | 本轮接受的来源 |
|---|---|---|
| **E1** | 正式 venue、论文题名、正式年份 | CVF Open Access、NeurIPS/ICLR proceedings、OpenReview 正式 venue 页面、ACM DOI |
| **E2** | 首次预印本日期、作者声明、演示与实际发布资产 | arXiv version history、作者官方项目页、组织/作者官方 GitHub、模型托管页 |
| **E3** | 候选发现、交叉核对题名/作者 | arXiv API、DBLP、Semantic Scholar、OpenAlex、Crossref；不能单独支持核心结论 |
| **S** | 本文的任务边界、合同、实验与指标建议 | 基于 E1/E2 的分析性综合，明确写成建议而非论文原结论 |

release surface 分为：paper、project/demo、inference code、training code、checkpoint/adapter、dataset、hosted API。正文逐项写明，不用一个模糊的“开源”覆盖全部资产。

## 3. 检索记录

### 3.1 数据库检索

检索在 2026-08-30 完成。命中数用于描述检索宽度，不代表互不重复的纳入论文数。

| 数据库 | 查询或动作 | 返回/状态 | 如何使用 |
|---|---|---|---|
| arXiv API | `all:"audio-driven" AND (all:avatar OR all:portrait OR all:"talking face" OR all:"human animation")` | 188 | 宽检索；按日期检查近期候选 |
| arXiv API | `ti:"talking face"` | 119 | 补全 talking-face 术语 |
| arXiv API | `all:"human animation" AND all:audio` | 22 | 补全全身音频驱动 |
| arXiv API | `all:"joint audio-video" AND (all:avatar OR all:human)` | 14 | 检查原生音视频边界 |
| arXiv API | `all:"multi-shot" AND all:"human speech video"` | 1 | 定位 TalkCuts |
| arXiv API | 上述宽查询按 submittedDate 降序取 30 项 | 30 条人工题名/摘要筛查 | 发现 EfficientSync、DynaForcing、Omni-LiveAvatar 等冻结日前候选 |
| DBLP | `audio driven avatar` | 33 | venue/题名交叉发现 |
| DBLP | `talking face generation` | 208 | 宽检索，噪声较高 |
| DBLP | `audio driven human animation` | 8 | 全身任务补检 |
| DBLP | 第四组宽查询 | 429 | 只作候选池，不把命中数当证据 |
| Semantic Scholar | `talking face generation` | 157,307 | 说明词组过宽；后续请求触发 rate limit |
| OpenAlex | 题名/概念组合检索 | HTTP 429 | 未用其结果支持断言 |
| Crossref | 对候选题名/DOI 做定点核验 | 宽 token 查询噪声大 | 仅保留精确 DOI/题名核验 |

arXiv 五个主查询共返回 344 个“query slots”，存在大量重叠，不能写成 344 篇独立论文。数据库 API 的超宽命中、429 与题名歧义均如实保留；核心事实回到 E1/E2 来源。

### 3.2 官方来源核验路线

1. 从 arXiv abs/version history 读取 v1 日期；不使用 PDF 页脚推断首次公开时间。
2. 在 CVF、NeurIPS、ICLR proceedings 或 OpenReview venue 页面查正式记录。
3. 从论文/项目页反向定位官方组织仓库，读取 README、LICENSE、release/checklist、下载链接和提交历史中的日期声明。
4. 将 code、training、weights、dataset、API 分列；README 内部矛盾采用保守结论。
5. 对 2026 年作者标注的未来/近期 venue，若冻结日无正式 proceedings，则保持 E2，不升为 E1。

## 4. 关键工作逐条证据

### 4.1 需要纠正的六项

| 工作 | v1 首次公开 | 正式记录（冻结日） | release surface 读法 | 证据 |
|---|---:|---|---|---|
| **SadTalker** | 2022-11-22 | CVPR 2023 | 官方仓库含推理、256/512 检查点及训练相关资源；Apache-2.0 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_Image_CVPR_2023_paper.html), [arXiv](https://arxiv.org/abs/2211.12194), [repo](https://github.com/OpenTalker/SadTalker) |
| **VASA-1** | 2024-04-16 | NeurIPS 2024 Main, Oral | 微软项目/论文有演示；未定位到官方研究代码、权重或 API | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/014fe398da515cd552fa6e1f33e0565e-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2404.10667), [project](https://www.microsoft.com/en-us/research/project/vasa-1/) |
| **Hallo v1** | 2024-06-13 | 未定位到 Hallo v1 的正式 proceedings | README 记录 2024-06-15 权重/推理、2024-06-28 训练代码；MIT；训练数据局限写为 English-only | [arXiv](https://arxiv.org/abs/2406.08801), [repo](https://github.com/fudan-generative-vision/hallo) |
| **OmniHuman-1** | 2025-02-03 | ICCV 2025 | 正式论文、项目页与 demos；项目页未链接研究代码或权重 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_OmniHuman-1_Rethinking_the_Scaling-Up_of_One-Stage_Conditioned_Human_Animation_Models_ICCV_2025_paper.html), [arXiv](https://arxiv.org/abs/2502.01061), [project](https://omnihuman-lab.github.io/) |
| **OmniAvatar** | 2025-06-23 | 未定位到正式 venue；按预印本 | 官方仓库含代码、Wan2.1 14B/1.3B LoRA/音频相关权重；Apache-2.0 | [arXiv](https://arxiv.org/abs/2506.18866), [repo](https://github.com/Omni-Avatar/OmniAvatar) |
| **TalkCuts** | 2025-10-08 | NeurIPS 2025 Datasets & Benchmarks | 官方仓库 2025-12-14 宣布 data/code；数据表单访问，研究/非商业许可，不得再分发 | [OpenReview](https://openreview.net/forum?id=4a0w7AkrY7), [arXiv](https://arxiv.org/abs/2510.07249), [repo](https://github.com/UMass-Embodied-AGI/TalkCuts) |

重要纠错：Hallo2 的正式 venue 是 [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e4be7e9867ef163563f4a5e90cec478f-Abstract-Conference.html)，Hallo3 是 [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Hallo3_Highly_Dynamic_and_Realistic_Portrait_Image_Animation_with_Video_CVPR_2025_paper.html)。它们是后续独立论文，不能把正式 venue 追溯赋给 Hallo v1。

### 4.2 2026 正式记录

| 工作 | 正式记录 | 本文采用的能力边界 |
|---|---|---|
| OmniHuman-1.5 / *Instilling an Active Mind in Avatars via Cognitive Simulation* | [ICLR 2026 Oral](https://proceedings.iclr.cc/paper_files/paper/2026/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html) | 音频/语义驱动人体与环境交互；不等同于原生音视频 |
| StreamAvatar | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_StreamAvatar_Streaming_Diffusion_Models_for_Real-Time_Interactive_Human_Avatars_CVPR_2026_paper.html) | 流式、交互式 human avatar；实时声明仍需协议化延迟 |
| InfinityHuman | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_InfinityHuman_Towards_Long-Term_Audio-Driven_Human_Animation_CVPR_2026_paper.html) | 长时音频驱动人体动画 |
| AudioAvatar | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_AudioAvatar_Personalized_Audio-driven_Whole-body_Talking_Avatars_CVPR_2026_paper.html) | 个性化音频驱动全身 talking avatar |
| SpeakerVid-5M | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bf7dbac50ed7f6e12ad529c5b9396bc4-Abstract-Conference.html) | 数据与 benchmark：独白、倾听、双人/多轮；不是生成器 |

### 4.3 冻结日前预印本与资产核验

| 工作 | 首次公开 | 冻结日结论 | 证据 |
|---|---:|---|---|
| Hallo-Live | 2026-04-26 | 预印本；仓库含 inference/training、合成 prompts、阶段权重；仓库自称 ACM MM 2026 accepted，但未以正式 proceedings 升级 | [arXiv](https://arxiv.org/abs/2604.23632), [repo](https://github.com/fudan-generative-vision/hallo-live) |
| LongCat-Video-Avatar 1.5 | 2026 | 预印本；官方仓库可见 code、weights 与少步配置，MIT | [arXiv](https://arxiv.org/abs/2605.26486), [repo](https://github.com/meituan-longcat/LongCat-Video-Avatar) |
| AptAvatar | 2026-07-27 | 预印本；README checklist 与正文矛盾、命令有占位符，保守记为代码可见但完整权重不可确认 | [arXiv](https://arxiv.org/abs/2607.24013), [repo](https://github.com/TaoLiveAIGC/AptAvatar) |
| Omni-LiveAvatar | 2026-08-07 | 摘要称 code available，但仓库 code/checkpoints 仍为 TODO；按 paper/project only | [arXiv](https://arxiv.org/abs/2608.13602), [repo](https://github.com/Omni-LiveAvatar/Omni-LiveAvatar) |
| OmniMate | 2026-07 | 临近冻结日预印本，只作候选，不把摘要结果升级为稳定主线 | [arXiv](https://arxiv.org/abs/2607.23023) |
| DynaForcing | 2026-08 | 临近冻结日预印本，只作候选 | [arXiv](https://arxiv.org/abs/2608.17707) |
| EfficientSync | 2026-08 | 临近冻结日预印本，只作候选 | [arXiv](https://arxiv.org/abs/2608.18832) |

## 5. 任务边界的判定规则

同一模型可能跨多类，但每个实验必须按实际输入/输出重新分类：

1. 输入已有视频且主要改口腔：lip-sync。
2. 输入静态身份图和音频、输出头肩：talking portrait。
3. 主要控制来自驱动视频/关键点/3D motion：portrait animation 或 motion-driven human animation；有音轨不自动变成 audio-driven。
4. 音频是主要运动条件且输出包括手势、躯干或腿部：full-body audio-driven。
5. 音轨由模型和画面共同生成，而非先给定 driver：native audio-video。
6. 模型显式规划/生成多个镜头并维护跨镜头状态：multi-shot human video；数据含 cut 或把长视频剪段不够。

对于混合条件模型，正文要求写明：哪个条件决定口型、哪个决定姿态、哪个决定身份、冲突时谁优先。否则“audio-driven”可能只是“音频存在”。

## 6. 方法、指标与协议的证据选择

### 6.1 方法主线

- 2D warp/keypoint：[FOMM, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html)。
- 3DMM/motion coefficient：[SadTalker, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_Image_CVPR_2023_paper.html)。
- NeRF avatar：[AD-NeRF, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_AD-NeRF_Audio_Driven_Neural_Radiance_Fields_for_Talking_Head_Synthesis_ICCV_2021_paper.html)。
- GAN/expert sync：[Wav2Lip, ACM MM 2020](https://doi.org/10.1145/3394171.3413532)。
- diffusion：[DiffTalk, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_DiffTalk_Crafting_Diffusion_Models_for_Generalized_Audio-Driven_Portraits_Animation_CVPR_2023_paper.html)、[Hallo](https://arxiv.org/abs/2406.08801)、[VASA-1](https://proceedings.neurips.cc/paper_files/paper/2024/hash/014fe398da515cd552fa6e1f33e0565e-Abstract-Conference.html)。
- Video DiT/规模化混合条件：[OmniHuman-1, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_OmniHuman-1_Rethinking_the_Scaling-Up_of_One-Stage_Conditioned_Human_Animation_Models_ICCV_2025_paper.html)。

正文中的“目标冲突”“反事实”“最小指标集”属于 S 级方法学建议：它们不是任何单篇论文已经证明的统一结论，而是用于避免同步、身份、画质和动作被一个分数混淆。

### 6.2 数据与指标原始来源

| 项目 | 一手来源 | 本文只采用什么 |
|---|---|---|
| MEAD | [官方项目页](https://wywu.github.io/projects/MEAD/MEAD.html) | 受控情绪表演数据；不把标签当真实心理状态 |
| LRS3 | [论文](https://arxiv.org/abs/1809.00496) | 视听语音数据代表；提醒来源/许可/语言差异 |
| VoxCeleb | [VGG 官方页](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 身份数据代表；提醒身份和节目来源重叠 |
| HDTF | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.html) | 高分辨率 talking-face 数据代表 |
| SyncNet | [论文](https://arxiv.org/abs/1603.04433) | 声画同步代理；不当作完整口腔质量或跨语言真理 |
| ArcFace | [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) | 人脸 embedding 代理；不扩展到服装/身体/授权 |
| FVD | [OpenReview](https://openreview.net/forum?id=r1E9O1Obg) | 分布视频质量代理；要求公开实现与样本数 |

## 7. 安全与来源证据

- [C2PA Technical Specification 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)（2026-04）：用于内容凭证、声明绑定和完整性。正文明确限制：签名/manifest 不能证明内容真实或当事人同意。
- [NIST AI 100-4](https://doi.org/10.6028/NIST.AI.100-4)：用于生成式 AI 风险管理结构。它不是法律意见，也不代替身份/声音授权合同。

授权要求由系统用途决定，不能从开源许可证推导。例如代码 Apache-2.0 或 MIT，不代表训练数据、人物肖像、声音、模型输出或数据集都可商用；TalkCuts 的数据许可必须与代码许可分开读取。

## 8. 原图中文重绘与正文引用审计

- 被要求检查的完整路径：`docs/tasks/figures/digital-human-video-generation-overview.png`；2026-08-31 已在原路径原位替换为中文重绘版。
- 2026-08-30 原图记录：尺寸 1672 × 941，SHA-256 为 `735fa1795dc64505ca35cd8a0435d92d0100cc55c6dafc8fb8f231d4ee4255de`。原图包含 `Identity 0.92`、`Temporal Consistency 0.90` 等无协议支撑的示意数值，不能当作实验结果。
- 2026-08-31 中文重绘版记录：尺寸 1672 × 941，RGB PNG，文件大小 1,287,961 bytes，SHA-256 为 `4e550f7c7646fe6906df485e0be9bf4fc3b9d3f137e2f5252c527fc02f7bd971`。主体文字已改为简体中文，仅保留 `Transformer`、`Token`、`×L` 等必要技术术语；原数值已删除，三条评测均标为“示意值”，并明确注明“评测条仅为流程示意，不代表实验结果”。
- 处理：原图已在原路径中文重绘；正文仍不引用这张总览图，无示意分数的条件与同步合同图继续作为正文主图；同时保留两幅教学型 Mermaid，分别说明任务决策树和授权—生成—评测—溯源链。
- Mermaid 只表达分类/流程，不声称 benchmark 分数或 SOTA 排名。

## 9. 纳入、排除与残余不确定性

### 9.1 纳入标准

- 能改变任务定义、技术路线、评测或安全合同的代表工作；
- 冻结日前存在可核验 E1 或 E2 来源；
- 2026 前沿候选只在其与流式、长时、全身或 release surface 审计直接相关时纳入。

### 9.2 排除或降级

- 只在聚合博客、社交媒体或模型榜单出现的工作；
- 正式 venue 无法核实却仅由仓库 README 声称 accepted 的条目；
- 只有项目 demo、没有可下载资产的系统，不写成开源；
- 题名命中但实际属于 voice conversion、纯 TTS、纯 ASR、静态 3D avatar 或一般视频生成的论文；
- 预印本中的作者自报指标不用于跨论文排名。

### 9.3 残余不确定性

- 仓库可在冻结日后补发/撤回权重，故 surface 结论是时间戳快照。
- 大模型预训练集通常不完全公开，严格的“测试未见”无法仅靠 benchmark split 证明。
- 公开网页可证明链接存在，不能证明第三方能在所有地区、所有时间成功下载大文件。
- 论文的 fps、延迟、身份与同步指标高度依赖硬件、裁切、分辨率和 evaluator；本轮不做不同协议下的数值横排。

## 10. 交付前验证记录

2026-08-30 的交付前验证结果：

- `markdownlint-cli2 0.20.0`：正文与本日志共 2 个文件，0 errors。
- 引用闭环脚本：45 个正文引用编号、45 个唯一参考锚点；missing 0、unused 0、duplicate 0。
- 本地链接：正文指向本日志与新教学 PNG 的相对路径均存在；旧 PNG 仍保留，但正文不引用。
- Mermaid CLI 11.16.0：从正文提取 2 个 Mermaid block，在临时目录实际生成 2 个非空 SVG；没有把渲染产物写入仓库。
- `git diff --check`：正文与新日志均无空白错误。
- 外链语法共解析出 50 个唯一 HTTPS 目标。批量 HTTP 探测受当前环境对 arXiv/GitHub 的超时及 ACM DOI 403 限制，不能把网络失败当作坏链；探测发现的 SadTalker 与 Hallo3 两个 CVF 路径错误已回到 CVF 官方记录逐项修正。
- 新增教学 PNG；本条记录对应 2026-08-30 当日状态，未执行 commit/push。

2026-08-31 中文重绘补充验证：

- `docs/tasks/figures/digital-human-video-generation-overview.png` 已在原路径完成中文重绘，尺寸仍为 1672 × 941，主体文字为简体中文。
- 已删除原有 `0.92`、`0.90` 数字，并以“示意值”和“不代表实验结果”的可见文字约束证据边界。
- 视觉回读确认四栏流程、条件、表征、生成器、输出与评测结构均完整，无边缘裁切；正文仍继续使用无示意分数的合同图。

## 11. 图像资产记录

- 项目文件：[`assets/diagrams/digital-human-condition-sync-contract.png`](../assets/diagrams/digital-human-condition-sync-contract.png)
- 生成提示核心：白底、16:9、无论文名与分数；身份参考、驱动音频、姿态表情和场景镜头必须先汇入“授权与用途检查”，通过后再进入“时间轴对齐 → 条件融合 → 视频与音频生成 → 分轴验收”，不通过则停止处理。
- 像素尺寸：1672 × 941，RGB PNG。
- SHA-256：`69c11a02b0e27f038096c50cff97b7ddf0b035dee61be6c26dfcc2c4f9818ad5`。
- 视觉回读：四类输入均先经过授权门，且没有绕过路径；通过与不通过出口、四步生成链和口型/身份/动作/音画四个验收轴均清晰。正文同时保留 Mermaid 与文字合同。

补充资产记录（2026-08-31）：

- 项目文件：[`docs/tasks/figures/digital-human-video-generation-overview.png`](../docs/tasks/figures/digital-human-video-generation-overview.png)
- 重绘方式：以内置 Imagegen 对原图做中文定向重绘，并在独立视觉复核后删除无协议支撑的具体分数。
- 像素尺寸：1672 × 941，RGB PNG；文件大小 1,287,961 bytes。
- SHA-256：`4e550f7c7646fe6906df485e0be9bf4fc3b9d3f137e2f5252c527fc02f7bd971`。
- 视觉回读：四栏结构与主要流程保持完整；三条评测均标“示意值”，底部明确“不代表实验结果”；该图仍不作为数值证据引用。
