# 掩码视频生成调研记录（截至 2026-08-30）

本文是 `docs/generative-models/masked-generation.md` 的证据账本，记录检索表面、筛选规则、来源状态、争议和写作边界。它不是另一篇面向读者的综述，也不把检索结果条数当作纳入文献数。

## 1. 调研问题与冻结范围

本轮回答六个问题：

1. masked modeling、MaskGIT 式采样与 absorbing-state discrete diffusion 在哪一层等价？
2. confidence ranking、schedule、remasking 是否定义了可复现的 sampler？
3. masked AR、next-set AR、帧间 AR + 帧内 mask 是同义词吗？
4. token、tube、frame、block 与 causal chunk 各改变什么学习任务？
5. MAGVIT、Phenaki、MAGVIT-v2、MAGI、Lumos-1 等路线的正式 venue 和模块边界是什么？
6. 哪些结论来自文本/图像理论，只能用于机制解释而不能外推视频质量？

冻结日期为 **2026-08-30（Asia/Shanghai）**。改写前目标文档 SHA-256 为 `6f54af4aee50c57f230504181b0d83d968759acfc992f33100753eb2490606d1`。本轮只纳入一手论文、正式会议页、arXiv 原文和 Crossref/OpenAlex 元数据；二手综述只用于发现，不作为正文事实的最终依据。

## 2. 至少三种检索表面

### 2.1 OpenAlex：宽召回发现

检索日期：2026-08-30。时间过滤统一为 2021-01-01 至 2026-08-30，每式取前 25 条，只用于发现候选。

| 检索式 | API 快照 | 返回总量 | 观察 |
|---|---|---:|---|
| `masked video generation` | [OpenAlex API](https://api.openalex.org/works?search=masked+video+generation&filter=from_publication_date%3A2021-01-01%2Cto_publication_date%3A2026-08-30&per-page=25&select=id%2Cdisplay_name%2Cpublication_year%2Cdoi%2Cprimary_location) | 35,204 | 大量分割、理解、医学与编辑噪声；可找到 MAGVIT、VideoMAE |
| `masked autoregressive video generation` | [OpenAlex API](https://api.openalex.org/works?search=masked+autoregressive+video+generation&filter=from_publication_date%3A2021-01-01%2Cto_publication_date%3A2026-08-30&per-page=25&select=id%2Cdisplay_name%2Cpublication_year%2Cdoi%2Cprimary_location) | 7,105 | 可发现 MAGI、MarDini；术语边界仍混杂 |
| `discrete diffusion video generation` | [OpenAlex API](https://api.openalex.org/works?search=discrete+diffusion+video+generation&filter=from_publication_date%3A2021-01-01%2Cto_publication_date%3A2026-08-30&per-page=25&select=id%2Cdisplay_name%2Cpublication_year%2Cdoi%2Cprimary_location) | 24,135 | 召回广但 precision 低，必须用标题和全文复筛 |
| `MaskGIT video MAGVIT Phenaki MAGI Lumos` | [OpenAlex API](https://api.openalex.org/works?search=MaskGIT+video+MAGVIT+Phenaki+MAGI+Lumos&filter=from_publication_date%3A2021-01-01%2Cto_publication_date%3A2026-08-30&per-page=25&select=id%2Cdisplay_name%2Cpublication_year%2Cdoi%2Cprimary_location) | 0 | 证明长实体串对该索引过于脆弱，不能以零结果判定无文献 |

这些数字是索引在检索日的动态计数，不是 PRISMA flow，也不代表逐条阅读全文。宽检索的价值是发现术语变体，最终 venue 和机制必须回到一手页面。

### 2.2 arXiv API / 原文：精确标题与全文机制

检索日期：2026-08-30。统一模板：

`https://export.arxiv.org/api/query?search_query=<query>&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending`

| query | 命中 | 用途 |
|---|---:|---|
| `all:"masked video generation"` | 1 | 显示精确短语召回很窄，不足以覆盖路线 |
| `ti:"Masked Generative Video Transformer"` | 1 | MAGVIT 标题与 arXiv 版本 |
| `ti:"Phenaki"` | 1 | 核对 2210.02399，并回查 OpenReview 与 ICLR 2023 官方议程 |
| `ti:"Masked Autoregressive Video Generation"` | 1 | MAGI 标题候选，再回 CVF proceedings 定 venue |
| `ti:"discrete diffusion" AND all:"video generation"` | 5 | 发现 Lumos-1、MotionAura、URSA 等 |
| `ti:"Lumos-1"` | 1 | 核对标题与 arXiv 条目，再回 ICLR proceedings |
| `ti:"MaskFlow"` | 3 | 锁定 2502.11234 与预印本状态 |
| `ti:"MarDini"` | 1 | 锁定 2410.20280，并回查 TMLR / OpenReview 正式版本 |
| `ti:"MotionAura"` | 1 | 标题核验，再回 ICLR proceedings |
| `ti:"Uniform Discrete Diffusion"` | 2 | 锁定 URSA，再回 ICLR proceedings |
| `all:"maskgit" AND all:"video generation"` | 2 | 说明仅靠方法名共现会漏掉大量后继工作 |

随后阅读 arXiv HTML 的 method、algorithm、training/inference 和 limitation 部分，而非只读摘要。重点核对：MaskGIT 的已提交 token 置信度、cosine schedule；Phenaki 的 C-ViViT 时间因果与 12–48 步；MAGVIT COMMIT；MaskFlow 的 frame-level masking 与 MGM/FM sampling；MarDini 的连续 VAE latent 和 diffusion renderer。

### 2.3 正式 proceedings：venue 与最终题名

检索日期：2026-08-30。使用 CVF、NeurIPS、ICLR、ECVA 的正式页面做 venue 主证据：

- CVPR：MaskGIT 2022、MAGVIT 2023、MAGI 2025；
- ECCV：Token-Critic 2022；
- NeurIPS：D3PM 2021、VideoMAE 2022、MDLM / simplified masked diffusion / MAR 2024；
- ICLR：MAGVIT-v2 2024，MotionAura / RADD / time-agnostic MDM 2025，Lumos-1 / URSA 2026。

若 CVF 页面被自动访问策略返回 403，则使用其搜索索引摘要、公开 PDF 和 BibTeX 交叉核验；这不改变页面本身作为正式 proceedings 的地位。

### 2.4 Crossref：DOI 与容器名交叉核验

检索日期：2026-08-30。示例模板：

`https://api.crossref.org/works?query.bibliographic=<title>&rows=5&select=DOI,title,published,container-title,type,URL`

- MAGVIT 命中 `10.1109/CVPR52729.2023.01008`，容器为 CVPR 2023；
- MAGI 命中 `10.1109/CVPR52734.2025.00691`，容器为 CVPR 2025；
- Token-Critic 命中 `10.1007/978-3-031-20050-2_5`，容器为 ECCV 2022；
- Phenaki 的 Crossref 结果未给出正式会议记录，但 ICLR 2023 官方 poster 页面与其链接的 OpenReview `vOEXS39nOF` 确认收录；正式会议页优先于聚合元数据缺失；
- Lumos-1、MaskFlow 的 Crossref 结果不足以定 venue，分别回 ICLR proceedings 或 arXiv；MarDini 由 TMLR / OpenReview 正式版本确认。

### 2.5 前向 / 后向引文链

从 D3PM 的 absorbing state 追到 NeurIPS 2024 两篇 simplified masked diffusion，再到 ICLR 2025 的 RADD 与 time-agnostic 分析；从 MaskGIT 追到 Token-Critic、Phenaki、MAGVIT，再追到 MAGI、MaskFlow 和 Lumos-1。引文链只负责发现，所有写入正文的机制均回查论文 method/abstract 或正式 proceedings。

## 3. 筛选标准

### 3.1 纳入

- 直接定义 masked objective、absorbing corruption、iterative unmasking 或集合生成顺序；
- 直接提出视频 masked generation、masked AR 或 discrete diffusion 系统；
- 能澄清 mask unit、confidence/remasking、serial depth 或 train–test mismatch；
- 截止日已有正式 proceedings，或虽为预印本但对路线边界不可替代并明确标注状态；
- 理论论文即使主要在文本/图像验证，只要用于限定“等价到哪一层”，可以纳入但必须标明外推边界。

### 3.2 排除

- 只把 mask 用作数据增强、分割标签或局部编辑控件，却不涉及生成建模机制；
- 只讨论连续 Gaussian video diffusion 且没有 masked planning / discrete state 边界价值；
- 二手博客、新闻、未能回到原论文的 benchmark 数字；
- 名称含 `mask` 或 `AR`，但全文定义与本章概念无关；
- 截止日之后的更新和无法确认来源状态的传闻。

### 3.3 最终证据集

最终登记 18 个一手来源：17 个正式发表工作、1 个用于必要路线说明的 arXiv 预印本。其中 MaskFlow 在正文明确写成预印本；Phenaki 由 ICLR 2023 官方页面确认收录，MarDini 由 TMLR / OpenReview 正式版本确认；MaskGIT 的 arXiv HTML 仅用于读算法，venue 仍由 CVPR 页面确定。没有用宽检索返回总量冒充筛选数量。

## 4. 证据分级

| 级别 | 含义 | 正文允许的用法 |
|---|---|---|
| A | 正式 proceedings + 论文 method/algorithm | 定 venue、定义机制、报告带协议限定的作者实验 |
| B | arXiv 原文，截止日未确认正式 venue | 描述方法与预印本结果，必须写“预印本” |
| C | Crossref / OpenAlex 元数据 | 交叉核题名、DOI、日期；不单独支撑机制 |
| S | 本轮基于多篇一手来源的综合推断 | 必须显式写成边界、建议或“可作某种视角”，不冒充原文定理 |

## 5. 核心事实登记

| ID | 来源与状态 | 支撑的事实 | 使用边界 |
|---|---|---|---|
| R01 | [D3PM](https://proceedings.neurips.cc/paper_files/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)，NeurIPS 2021，A | 离散 transition matrix 可为 uniform、邻接或 absorbing；absorbing state 连接 mask-based / AR | 不把所有 discrete diffusion 称作 masked diffusion |
| R02 | [MaskGIT](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html)，CVPR 2022，A | 随机 mask 训练；全 mask 初始化；并行预测、采样、按置信度与 schedule 重掩；cosine 为论文经验选择 | 图像证据；不宣称其 heuristic 是精确 reverse diffusion |
| R03 | [Token-Critic](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2901_ECCV_2022_paper.php)，ECCV 2022，A | 辅助 critic 区分真实/采样 token，指导接受、拒绝和重采样 | 图像侧；不能替代视频 calibration 实验 |
| R04 | [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html)，NeurIPS 2022，A | 90%–95% tube masking 可用于自监督视频表征 | 不是生成 sampler 或生成 schedule 证据 |
| R05 | [Phenaki](https://iclr.cc/virtual/2023/poster/12256)，ICLR 2023，A | 时间因果 C-ViViT tokenizer；双向 masked video token generator；12–48 采样步；动态文本/变长路线 | 会议页确认 venue；机制细节同时回查论文正文 |
| R06 | [MAGVIT](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html)，CVPR 2023，A | 3D tokenizer、COMMIT condition embedding、统一多任务、约 12 轮示例 | 不简化为“Video MaskGIT”而遗漏 condition refinement |
| R07 | [MAGVIT-v2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html)，ICLR 2024，A | 高质量图像/视频共享 tokenizer 使 LM 式视觉生成更可行 | 不是 MAGVIT masked sampler 的第二版 |
| R08 | [MDLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html)，NeurIPS 2024，A | Rao–Blackwellized objective 是经典 MLM losses 的混合；可 semi-AR 采样 | 主要语言实验，不外推视频质量 |
| R09 | [Simplified and Generalized Masked Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html)，NeurIPS 2024，A | continuous-time absorbing variational objective 是加权 CE 积分；可状态相关 schedule | 支撑目标层桥，不支撑 MaskGIT sampler 等价 |
| R10 | [MAR](https://proceedings.neurips.cc/paper_files/paper/2024/hash/66e226469f20625aaebddbe47f0ca997-Abstract-Conference.html)，NeurIPS 2024，A | generalized masked AR 可用于连续 token，并以 per-token diffusion loss 建模 | 图像侧，说明 AR 与离散 CE 不绑定 |
| R11 | [MarDini](https://openreview.net/forum?id=fuOHI59rUW)，TMLR 2025，A | 低分辨率连续 latent 的 masked frame planner + 高分辨率 continuous diffusion renderer | 不是 absorbing discrete diffusion；正式版本页眉标注 05/2025 |
| R12 | [MAGI](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Taming_Teacher_Forcing_for_Masked_Autoregressive_Video_Generation_CVPR_2025_paper.html)，CVPR 2025，A | 帧内 masked、帧间 causal；CTF 用完整观察帧；论文协议下 FVD +23%、16 帧训练到 100+ 帧 | 作者设置，不作普遍性能保证；CTF 不消灭全部 exposure bias |
| R13 | [MotionAura](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ad996b5c45130de2bc00b60d8607904-Abstract-Conference.html)，ICLR 2025，A | 3D-MBQ-VAE 的 full-frame mask 训练；VQ discrete diffusion + spectral denoiser | codec mask 与 generator diffusion 是不同模块 |
| R14 | [RADD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a365e37c18fb91af547a2f0012a89e98-Abstract-Conference.html)，ICLR 2025，A | absorbing concrete score = clean conditional × analytic time scalar；NLL bound 与 any-order AR 关联 | 主要语言理论；是界/目标解释，不是逐轨迹等价 |
| R15 | [Time-Agnostic MDM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html)，ICLR 2025，A | 所研究 MDM 与 first-hitting sampler 可去时间变量；低温 categorical 数值误差降低多样性 | 不推广到所有 masked visual sampler；视频需重新验证 |
| R16 | [MaskFlow](https://arxiv.org/abs/2502.11234)，arXiv 2025，B | discrete flow matching；逐帧独立 mask ratio；MGM/FM sampling；full-sequence 到 frame/chunk AR 可切换 | 不是自动等价于 absorbing D3PM；预印本状态 |
| R17 | [Lumos-1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59ad89d72559dd4ce557d56f36313724-Abstract-Conference.html)，ICLR 2026，A | LLM 内 parallel mask discrete diffusion；帧内双向、帧间 causal；temporal tube policy 处理 frame-wise loss imbalance | 不等于逐 token AR，也不等于整段双向 masked model |
| R18 | [URSA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/daf8364f0715a41a469c677c0adc4754-Abstract-Conference.html)，ICLR 2026，A | uniform categorical diffusion、metric path、全局离散时空 token refinement | 作为“离散 diffusion 不必 masked/absorbing”的反例 |

## 6. 争议与裁决

### 6.1 “Masked modeling 就是 masked diffusion”

裁决：**不成立**。masked CE 只定义条件预测任务。只有给定 absorbing 前向过程、时间/速率、参数化和正确权重后，特定 variational objective 才能化成加权 CE。正文把这称为“目标层桥”，不称“全算法等价”。

证据：R01、R08、R09、R14、R15。

### 6.2 “MaskGIT 是 absorbing diffusion 的 sampler”

裁决：**不能默认成立**。MaskGIT 采用 confidence-ranked commit 和经验 schedule；概率化 reverse process 需要由转移率或 transition kernel 定义。两者可以共享 masked denoiser，却没有因此共享逐步概率。

证据：R01、R02、R09、R15。正文不使用“MaskGIT diffusion steps”这种混称。

### 6.3 “Remasking 能纠正所有早期错误”

裁决：**原版 MaskGIT 不支持这个强说法**。论文给已 unmasked 位置置信度 1，后续主要重掩当前候选中的低置信位置。Token-Critic 是另一套 learned accept/reject 机制，可以更积极质疑生成 token。

证据：R02、R03。正文明确“remask 的作用域”。

### 6.4 “Masked AR 与 next-set AR 完全等价”

裁决：**只有在定义了有序集合分区和选择策略后可作视角转换**。同一集合内的并行位置通常还包含条件独立近似；若集合由本轮样本置信度动态决定，policy 也是联合生成过程的一部分。RADD 给出的 any-order 联系是在 bound/objective 层。

证据：R10、R14；其余为 S 级概率分解审计。

### 6.5 “Masked AR 一定是离散 token”

裁决：**不成立**。MAR 在连续 token 上用 diffusion loss；MarDini 也是连续 VAE latent 的 masked frame planner + diffusion renderer。`mask` 描述缺失结构，`AR` 描述因果分解，`discrete` 描述表示/状态空间。

证据：R10、R11。

### 6.6 “Tube mask 已被证明普遍提升视频生成”

裁决：**证据需分模块**。VideoMAE 支撑 tube mask 的表征预训练；Lumos-1 支撑其特定生成训练与 frame-wise loss imbalance 设计。两篇的任务、目标和指标不同。

证据：R04、R17。

### 6.7 “MAGVIT-v2 是 MAGVIT sampler v2”

裁决：**错误**。正式题名和摘要把贡献定位为视觉 tokenizer，并展示 LM 式视觉生成。正文只将其列为 tokenizer milestone。

证据：R07。

### 6.8 “Phenaki 是 ICLR 2023”

裁决：**成立**。arXiv 首次提交于 2022-10-05，但 ICLR 2023 官方 poster 页面明确标注该论文为主会 poster accept，并链接 OpenReview `vOEXS39nOF`。Crossref 没有对应条目属于聚合元数据缺失，不能据此否定会议收录。

证据：R05 + ICLR 2023 官方议程与 OpenReview 标识；Crossref 仅作缺失案例记录。

## 7. 图表设计审计

正文使用三张 Mermaid，不生成 PNG：

1. 四层关系图：强调 objective、sampler、set order、video causality 分层；
2. MaskGIT 单轮状态机：明确候选重掩与已提交冻结；
3. 帧间因果 + 帧内 masked：显示串行深度约为 $T\times J$ 和 CTF/MTF 分叉。

每图必须包含 `accTitle`、单行 `accDescr`，并紧随顺序化文字替代。节点 ID 使用 snake_case；颜色只辅助分组，文字本身能够传达语义。最终以 Mermaid CLI 实际渲染 SVG 到临时目录，不把渲染产物写入仓库。

## 8. 写作证据边界

- 对正式论文数字使用“论文在其协议下报告”，不写成跨数据集保证；
- 对 MaskFlow 始终保留 preprint 标签；Phenaki 按 ICLR 2023、MarDini 按 TMLR 2025 正式收录记录；
- MDLM、RADD、time-agnostic 等主要是文本/离散理论，只用于目标与 sampler 边界；
- MAR 与 Token-Critic 是图像侧证据，只用于反例或机制，不外推视频性能；
- VideoMAE 是表示学习证据，不用于证明生成质量；
- 任何 calibration、分布漂移和复现实验建议，若非原论文直接结论，均作为 S 级审计建议表述。

## 9. 维护触发条件

发生以下任一情况需更新正文与本账本：

- MaskFlow 获得可核正式 venue；
- 新论文证明 MaskGIT confidence sampler 与某个 absorbing reverse process 的严格等价；
- 视频 masked diffusion 出现公开 calibration、数值 categorical sampling 或 exact likelihood 对照；
- Lumos-1 / URSA 的正式版本题名、作者或 proceedings 发生更正；
- 仓库改变引用格式、Mermaid 可访问性规范或章节分类轴。

## 10. 验证记录

验证日期：2026-08-30。

- markdownlint：`markdownlint-cli2` 检查两个目标文件，0 issue；
- 引用锚点：正文 18 个引用编号全部有唯一 `ref-N` 定义，无缺失集合；
- 内部链接：4 个相对 Markdown 路径均存在；
- Mermaid CLI 实际渲染：3/3 个代码块成功渲染为临时 SVG，均有 `accTitle`、`accDescr` 与顺序化文字替代，未把产物写入仓库；
- `git diff --check`：两个目标文件通过；
- 改动范围：本子任务只改 `docs/generative-models/masked-generation.md` 与新建 `sources/research_20260830_masked_video.md`；工作区其他变更来自并行任务，未触碰。
