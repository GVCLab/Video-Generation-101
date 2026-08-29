# 技术时间线证据状态与前沿节点复核

> 检索与核验日期：2026-08-30（Asia/Shanghai）
>
> 对应正文：`docs/timeline.md`
>
> 范围：区分首次公开、正式发表、研究里程碑和截止日产品/工件可用性；重点复核 2025–2026 节点。

## 1. 研究问题

1. 时间线中的年份是论文首次公开、正式发表，还是机构首次官方发布？
2. “研究里程碑”如何与学术录用、产品发布和开源可复现性分开？
3. 截止 2026-08-30，2025–2026 节点的代码、权重、模型卡、API 和托管产品实际处于什么状态？
4. 哪些能力只由作者自评或产品宣传支持，必须保留“作者/官方报告”归属？

## 2. 检索策略

### 2.1 数据库与入口

- 首次公开与版本：arXiv abstract/version page 和 export API；
- 正式发表：期刊官方页、会议 proceedings、OpenReview 会议/工作坊记录；
- 机制和实验：论文正文、机构技术报告、官方项目页与模型卡；
- 开放工件：作者/机构官方 GitHub 与 Hugging Face 组织页；
- 产品可用性：机构发布说明、系统卡、官方模型页、文档与官方下线通知。

### 2.2 实际使用的检索式

```text
arxiv API id_list=2501.03575,2503.20314,2506.09985,2508.13009,
2509.20328,2602.03604,2603.14482,2603.19312,2604.14148,2606.02800

site:openreview.net OR site:proceedings.mlr.press OR
site:openaccess.thecvf.com "<exact paper title or arXiv id>"

"<exact paper title>" proceedings
"<model name>" official release model card code weights API
"<model name>" official available limited research preview discontinued
site:github.com/<official organization> "<model name>"
site:huggingface.co/<official organization> "<model name>"
```

对每个近期节点还在官方页内检查 `available`、`early access`、`research preview`、`model card`、`code`、`weights`、`API`、`discontinued` 和局限性字段。搜索结果页只用于定位，不作为最终证据。

### 2.3 纳入标准

- 论文、技术报告、官方项目页、官方模型/系统卡或官方产品发布页能直接支持节点的日期、机制、工件或访问状态；
- 作者自报的速度、时长、成功率和基准可纳入，但正文必须明确归属给“作者/官方报告”；
- 正式发表状态只由正式 proceedings/期刊记录支持；workshop 单独标注，不与主会 archival proceedings 合并；
- 当前可用性只记录截止日能从官方入口确认的状态，并与研究贡献分轴。

### 2.4 排除标准

- 不用媒体、榜单站、社交转述、第三方模型页或搜索摘要支持技术断言；
- 不把官方宣传中的“领先”、“物理正确”、“世界理解”等定性词升格为独立验证的能力事实；
- 不因为有 arXiv 记录就写成“论文已发表”，也不因为有官方 demo 就推出代码、权重或训练配方可得；
- “未找到正式发表记录”只是本次检索截止日结论，不写成永久不存在；
- 对没有新机制证据的产品更新，不为了“最新”而无限扩张时间线。

## 3. 证据分级

| 等级 | 证据类型 | 可支持 | 不可单独支持 |
|---|---|---|---|
| **E1** | 期刊或主会官方 proceedings | 正式发表状态，及论文定义的机制/实验 | 开源、独立复现或当前产品可用 |
| **E1w** | 官方 workshop 页或 camera-ready | 具体工作坊的接收/版本状态 | 主会 archival 录用或稳定学术共识 |
| **E2** | arXiv 预印本、技术报告或模型卡 | 首次公开日期，作者定义的机制、实验与限制 | 已同行评议或已独立复现 |
| **E3** | 机构官方发布、项目页或系统卡 | 机构发布事实、产品规格、自述能力与局限 | 学术录用、独立基准或通用能力 |
| **A** | 官方代码、权重、API、托管入口或下线通知 | 截止日的工件/产品访问快照 | 论文机制的完整实现或未来仍可用 |
| **M** | 本页的谱系编辑判断 | 为什么一个节点值得放入技术历史 | 论文录用、产品成功或能力通用性 |

`A` 和 `M` 是正交维度，不是比 `E1`–`E3` 更高或更低的证据等级。

## 4. 2025–2026 节点证据台账

| 节点 | 首次公开/学术状态 | 本次采用的关键断言 | 截止日可用性 | 主要一手来源 |
|---|---|---|---|---|
| Wan 2.1 | 2025-03-26 arXiv v1；E2 | 视频 DiT 家族、新 VAE、1.3B/14B 和多任务开放；速度/显存为作者报告 | 代码、权重、官方 demo 可访问 | [arXiv](https://arxiv.org/abs/2503.20314) · [GitHub](https://github.com/Wan-Video/Wan2.1) · [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| NVIDIA Cosmos | 2025-01-07 arXiv v1；E2 | 平台由 tokenizer、AR/diffusion world foundation models、curation、guardrail 和后训练工具组成；不是单一“Cosmos 1”checkpoint | 官方代码与权重可得 | [technical report](https://arxiv.org/abs/2501.03575) · [project](https://research.nvidia.com/labs/cosmos-lab/) · [GitHub](https://github.com/NVIDIA/Cosmos) |
| Cosmos Predict2 | 2025-06-11 官方技术发布；E3 | 2B/14B 世界状态生成、多条件与 Physical AI 后训练；“物理准确”只作 NVIDIA 自述 | 代码、权重、PyPI 和 demo 可访问 | [technical release](https://developer.nvidia.com/blog/?p=101575) · [project](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2/) · [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2) |
| V-JEPA 2 | 2025-06-11 arXiv v1；E2 | 大规模无动作视频表征 + 小量机器人动作数据后训练；`zero-shot` 不等于从未使用机器人数据 | 代码与 checkpoints 可访问 | [arXiv](https://arxiv.org/abs/2506.09985) · [official project](https://ai.meta.com/research/vjepa/) · [GitHub](https://github.com/facebookresearch/vjepa2) |
| Veo 3 / 3.1 | 2025-05-20 官方产品发布；黑盒探测于 2025-09-24 arXiv v1 | 原生音视频是产品发布；视觉/物理探测来自后续预印本，不等于显式 perception head 或机器人闭环 | Gemini、Flow、AI Studio 和 API 入口可见；无公开权重 | [launch](https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/) · [model page/card](https://deepmind.google/models/veo/) · [probing preprint](https://arxiv.org/abs/2509.20328) |
| Genie 3 | 2025-08-05 官方研究发布；E3 | 720p、24 FPS、数分钟和可插入事件均是官方报告；动作空间和持续性等限制同时保留 | Project Genie 是可访问的实验性原型，受访问条件限制；无代码/权重 | [release](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) · [current model page](https://deepmind.google/models/genie/) · [Project Genie](https://labs.google/fx/projectgenie/) |
| Matrix-Game 2.0 | 2025-08-18 arXiv v1；E2 | few-step causal diffusion、25 FPS、约 1,200 小时数据和分钟级交互为作者报告 | 2.0 代码/权重入口仍在；官方仓库主线已纳入 3.0 | [arXiv](https://arxiv.org/abs/2508.13009) · [project](https://matrix-game-v2.github.io/) · [GitHub](https://github.com/SkyworkAI/Matrix-Game) · [weights](https://huggingface.co/Skywork/Matrix-Game-2.0) |
| Marble | 2025-11-12 官方产品发布；E3 | 文本/图像/视频/3D 条件到可编辑显式 3D 世界；动态交互是后续方向而非当前能力 | 官方称可普遍使用，需账号；无公开代码/权重 | [release](https://www.worldlabs.ai/blog/marble-world-model) · [docs](https://docs.worldlabs.ai/) · [product](https://marble.worldlabs.ai/) |
| Sora 2 | 2025-09-30 官方发布/系统卡；E3 | 音视频、多镜头和物理改进均保留官方自述归属；系统卡的物理和安全限制同时纳入 | 官方确认 Sora 产品自 2026-04-26 起下线 | [release](https://openai.com/index/sora-2/) · [system card](https://openai.com/index/sora-2-system-card/) |
| GWM-1 | 2025-12-11 官方研究/产品发布；E3 | Gen-4.5 共享底座上分别后训练 Worlds、Characters 和 Robotics；不是一个通用 checkpoint | Worlds/Robotics 申请式 early access；Characters 有 Web/API；无公开权重 | [official page](https://runway.com/research/introducing-runway-gwm-1) |
| Cosmos 3 | 2026-06-01 arXiv v1；E2 technical report | AR reasoner + diffusion generator 的 omnimodal 模型家族；各后训练分支不等于单 checkpoint 通吃全任务 | 代码、模型卡/权重和托管 demo 可访问 | [technical report](https://arxiv.org/abs/2606.02800) · [project](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) · [GitHub](https://github.com/NVIDIA/cosmos) · [model collection](https://huggingface.co/collections/nvidia/cosmos3) |
| V-JEPA 2.1 | 2026-03-15 arXiv v1；E2 | dense predictive loss 和跨层自监督；抓取提升为作者设置下的结果 | 官方代码与预训练 checkpoints 可访问 | [arXiv](https://arxiv.org/abs/2603.14482) · [GitHub](https://github.com/facebookresearch/vjepa2) |
| LeWorldModel | 2026-03-13 arXiv v1；E2 | next-embedding prediction + Gaussian latent regularization 的小型动作条件规划实验；不外推为 foundation-scale 结论 | 官方代码、数据与 checkpoints 可访问 | [arXiv](https://arxiv.org/abs/2603.19312) · [project](https://le-wm.github.io/) · [GitHub](https://github.com/lucas-maes/le-wm) |
| EB-JEPA | 2026-02-03 arXiv v1；E1w workshop camera-ready | 单卡教学/研究组件；ICLR 2026 World Models Workshop，不写成 ICLR main track | 官方代码可访问；未另行发布权重 | [arXiv](https://arxiv.org/abs/2602.03604) · [OpenReview](https://openreview.net/forum?id=ZVAMdXGCUC) · [GitHub](https://github.com/facebookresearch/eb_jepa) |
| Kling Video 3 / 3 Omni | 2026 官方产品发布；E3 | 多参考、多镜头和原生音频均为官方产品规格；无闭环或可规划状态证据 | 官方 Web 产品可访问，受账号/地区限制；无代码/权重 | [release note](https://kling.ai/release-note/release-notes/whbvu8hsip) · [product guide](https://kling.ai/quickstart/klingai-video-3-model-user-guide) |
| Seedance 2.0 | 2026-02-12 官方产品发布；2026-04-15 arXiv model card | 统一多模态音视频架构与规格为官方自述；模型卡不等于同行评议论文 | 托管产品可访问，受账号/地区限制；无代码/权重 | [release](https://seed.bytedance.com/en/blog/seedance-2-0-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83) · [model card](https://arxiv.org/abs/2604.14148) · [project](https://seed.bytedance.com/seedance2_0) |
| MiniMax H3 | 2026-07-31 官方发布；2026-08-03 开放 Base | 官方架构/规格与开放边界；完整托管 2K 系统不能笼统称为全部开源 | 两个 H3-Base checkpoint 与代码可得，本地基础工作流为 768p；Context-IR/2K regeneration 仍托管 | [release](https://www.minimax.io/blog/minimax-h3) · [open release](https://www.minimax.io/news/minimax-h3-open-source) · [GitHub](https://github.com/MiniMax-AI/MiniMax-H3) · [weights](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Seedance 2.5 | 2026-07-31 官方产品发布；E3 | 30 秒、多轮延展、30 图/10 视频/10 音频与时间戳编辑为官方规格；物理/多主体限制保留 | 官方称在即梦/Doubao Pro 上线；发布页仍称 API 即将提供；无权重 | [release](https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5) · [project](https://seed.bytedance.com/seedance2_5) |

## 5. 关键断言与写作决策

### 5.1 四条时间轴必须分开

1. **首次公开轴：**arXiv v1 或机构首次官方发布决定时间线年份。
2. **学术状态轴：**预印本、技术报告、model card、workshop camera-ready 和主会/期刊 proceedings 分别标注。
3. **谱系判断轴：**“研究里程碑”是本页根据机制差异、实验证据与后续影响做出的编辑判断，不是录用结果。
4. **可用性轴：**代码、权重、托管产品和 API 会随时间变化；本页只给出 2026-08-30 快照。

因此，新增 Mermaid 使用四条并行泳道，并用“不自动推出”的虚线显式阻断跨轴误推。

### 5.2 2025–2026 统一标为 frontier observation

- 该标签表示“截止日可见的前沿观察”，不表示预印本已同行评议或产品能力已独立确认；
- 一律将企业规格、速度、时长、成功率和基准改写为“官方/作者报告”；
- 对预印本资源使用 `Preprint`、`Technical Report`、`Model Card` 或 `Workshop Paper`，不用模糊的 `Paper` 暗示已正式发表。

### 5.3 当前产品状态不改写历史

- Sora 2 产品下线后仍可作为 2025 的历史技术/治理节点，但资源栏必须标记“已下线”；
- Genie 3 初始为有限研究预览，截止日可通过 Project Genie 访问实验性托管原型，两个时点均保留；
- Matrix-Game 官方仓库已继续到 3.0，但不用新版覆盖 2.0 在 2025 的首次里程碑；
- MiniMax H3 区分托管完整系统与开放的 Base checkpoints，不把 2K 托管能力归给本地 768p Base 工件。

## 6. 未纳入或仅保留为边界的内容

- 未将新闻报道、社交热度、第三方排行或未公开协议的传闻纳入时间线。
- 未将 Matrix-Game 3.0 新增为卡片；它只用来解释 2.0 工件的当前仓库语境，避免本次更新无限扩展范围。
- 未将 Genie 3、Marble、GWM-1、Kling 3 的官方演示转写成论文级机制或独立能力事实。
- 未将 Veo 黑盒探测的任务表现扩张为显式深度估计、机器人策略或通用物理模型。
- 未将 EB-JEPA 的 workshop camera-ready 写成 ICLR 2026 主会录用。

## 7. 限制与后续复核点

- 产品可用性与官方仓库主线是高漂移信息；任何后续版本应重新核查，不复用本次快照当作当前事实。
- 对“未找到正式发表记录”的节点，本次已交叉检索题名、arXiv ID、OpenReview 和主要 proceedings；仍不能将缺失证据表述成全球穷尽证明。
- 多个 2025–2026 节点仍只有作者基准或产品演示；本页不提供新的独立复现、实时延迟测试或闭环机器人验收。
- 本次只调整时间线的证据语义和截止日状态，不更改 bibliography/registry，也不将产品发布补成学术引用。

## 8. 文档验收记录

- `markdownlint-cli2 0.20.0`：`docs/timeline.md` 与本文件共 0 个错误；
- Mermaid CLI `11.12.0`：2 个 Mermaid 块均成功渲染为临时 SVG（64,091 与 34,586 bytes），未向仓库生成 PNG/SVG；
- 可访问性：2 幅图均含 1 个 `accTitle` 和 1 个 `accDescr`；新图还有顺序化文字替代；
- 图像：75 个既有 HTML 图像引用全部保留，75 个 `alt` 均非空；
- 闭合：`a` 318/318、`table` 75/75、`tr` 75/75、`td` 150/150、`strong` 375/375、`code` 113/113；
- 本地链接：正文检出 85 个唯一本地路径，缺失 0；
- 空白与 diff：指定文件与全部已跟踪工作树的 `git diff --check` 均通过；本新文件的 `--no-index --check` 无空白错误。
