# 视频基础模型系统综述：检索、筛选与 release-surface 证据账本

## 1. 任务、范围与冻结规则

- 检索与核验日期：2026-08-30（Asia/Shanghai）。
- 目标章节：docs/foundation-models.md。
- 核心问题：视频基础模型怎样从数据、过滤/去重、caption、tokenizer 和 generator 延伸到后训练、偏好/RL、蒸馏、SR/插帧、音频、安全/provenance 与 API；每项能力应归因给论文机制、单 checkpoint、模型家族还是产品系统。
- 时间范围：用 2017–2025 的正式论文建立机制基础，重点 fresh-check 2024–2026 的系统与发布状态。
- 冻结规则：论文状态以 arXiv/正式 proceedings 在冻结日可见记录为准；开放状态以官方仓库、模型卡、权重清单、变更日志与许可证为准；产品能力以官方项目/API 文档为准。
- 不做的事：不把作者 benchmark 排成统一榜单；不把社区 fork 当官方发布；不从产品 demo 反推单 checkpoint；不把 repository license 自动外推到所有权重或托管服务。

本日志遵循 literature-review 方法，但它不是穷尽式系统综述。宽检索用于发现 2026 前沿，指定模型用题名和官方 artifact 做定向核验。动态页面可能在冻结日后变化，正文的日期限定是结论的一部分。

## 2. 研究对象与纳排标准

### 2.1 纳入

满足以下任一条件：

1. 为视频 foundation model 的表示、训练、后训练、因果/长视频、联合音视频或部署提供可复用机制；
2. 披露至少两个系统层，例如数据管线与训练、generator 与蒸馏、checkpoint 与服务；
3. 属于任务指定 fresh-check：LTX-Video/LTX-2、Mochi、HunyuanVideo/1.5、Wan/Wan2.2、Step-Video、Open-Sora 2.0、SkyReels-V2/V3、MAGI-1、Ovi、Movie Gen、Cosmos，以及 2026 frontier；
4. 为正式 proceedings、原始预印本/技术报告、官方代码/模型卡/许可证或官方产品/API。

### 2.2 排除

- 只做视频理解、检索、压缩、超分或编辑且不影响基础模型系统链的工作；
- 只面向 avatar、医学、机器人单任务等窄域且没有可迁移系统机制的工作；
- 二手榜单、媒体稿、聚合模型页、非官方 fork；
- 只有营销性结果而没有足以界定模型、版本或服务面的页面；
- 名称相近但题名、作者或官方组织不匹配的条目；
- World Model 仅因名称出现而纳入；必须另有动作、状态、反事实或闭环证据。

窄域工作若能证明关键机制仍可作为机制证据，但不会被列为通用 foundation checkpoint。SkyReels-V3 的 TalkingAvatar 之所以保留，是为说明同一官方仓库包含多个任务权重，而非把 avatar 能力提升为通用模型能力。

## 3. 检索表面一：正式 proceedings 与 arXiv

### 3.1 arXiv 宽检索

API 共同参数：

- endpoint：<https://export.arxiv.org/api/query>
- start=0
- max_results=50
- sortBy=submittedDate
- sortOrder=descending

精确检索式与冻结日结果：

| 检索式 | arXiv totalResults | 实际检查 | 用途 |
|---|---:|---:|---|
| all:"video generation" AND all:"foundation model" | 165 | 最新 50 条题名；候选再看摘要 | 通用 foundation 与表示前沿 |
| all:"video generation" AND (all:"preference optimization" OR all:"reinforcement learning") | 152 | 最新 50 条题名；候选再看摘要 | preference、DPO/RL 与 post-train |
| (all:"audio-video generation" OR all:"audio visual generation") AND (all:joint OR all:synchronized) | 75 | 最新 50 条题名；候选再看摘要 | 联合 A/V 与同步 |

三个返回切片共 150 条记录，按 arXiv ID 去重后为 142 条；三对检索的交集分别为 4、1、3 条。它们是“最新 50 条”发现样本，不是三个查询全部 392 条结果的穷尽纳入。宽检索最终进入正文的新前沿是 V-RAE、VideoRAE 与系统化 post-train；其余多数因窄域、理解/评测而非生成系统、重复或证据过弱而排除。

### 3.2 指定题名与 arXiv ID 核验

使用 ti:"完整题名" 或 id_list 交叉核对题名、作者、首次提交时间与版本：

| 实体 | arXiv ID | 冻结日状态 |
|---|---|---|
| LTX-Video | 2501.00103 | 预印本；首次提交 2024-12-30 |
| LTX-2 | 2601.03233 | 预印本；首次提交 2026-01-06 |
| HunyuanVideo | 2412.03603 | 预印本 |
| HunyuanVideo 1.5 | 2511.18870 | 技术报告预印本；首次提交 2025-11-24 |
| Wan | 2503.20314 | 预印本；对应 Wan 2.1，不是独立 Wan2.2 论文 |
| Step-Video-T2V | 2502.10248 | 技术报告预印本 |
| Open-Sora 2.0 | 2503.09642 | 预印本 |
| SkyReels-V2 | 2504.13074 | 预印本 |
| MAGI-1 | 2505.13211 | 预印本 |
| Ovi | 2510.01284 | 预印本 |
| Movie Gen | 2410.13720 | 预印本 |
| Cosmos v1 | 2501.03575 | 预印本 |
| SkyReels-V3 | 2601.17323 | 技术报告预印本 |
| Seedance 2.0 | 2604.14148 | 预印本 |
| Cosmos 3 | 2606.02800 | 预印本 |

Mochi 1 preview 与 MiniMax H3 没有在指定题名核验中得到可作为其正式论文的 arXiv 记录；正文只用官方仓库/模型卡支持发布事实。不得用第三方解读补成“论文机制”。

### 3.3 正式 proceedings 核验

正式 venue 只在出版方或会议入口确认：

| 工作 | 正式入口 | 结论用途 |
|---|---|---|
| VQ-VAE | [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | 离散视觉表示基础 |
| MAGVIT | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.html) | masked 多任务视频 token |
| VideoPoet | [ICML 2024](https://proceedings.mlr.press/v235/kondratyuk24a.html) | 多模态 token 自回归 |
| Diffusion Forcing | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html) | per-token noise 与 causal rollout |
| CogVideoX | [ICLR 2025](https://openreview.net/forum?id=LQzN6TRFg9) | 公开的大型视频 DiT |
| CausVid | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) | 双向 teacher 到少步 causal student |

指定的 2025–2026 系统中，大多数截至冻结日仍是预印本/技术报告。正文明确使用“作者报告”“官方发布”而不是把它们写成正式 venue 结论。

## 4. 检索表面二：OpenAlex 书目交叉核验

### 4.1 宽查询

endpoint：<https://api.openalex.org/works>

共同过滤：

- from_publication_date:2024-01-01
- to_publication_date:2026-08-30

| search 参数 | count | 判定 |
|---|---:|---|
| video generation foundation model | 178,213 | 全文语义过宽，医学/LLM 等噪声高 |
| video generation preference reinforcement learning | 13,998 | 理解、机器人与通用 RL 混入 |
| video generation distillation | 10,743 | 图像、压缩和知识蒸馏混入 |
| joint synchronized audio video generation | 34,183 | avatar、理解与检索混入 |

这些 count 只记录检索行为，不用于趋势或规模论证。OpenAlex 的 citation count 是动态值，未写入正文。

### 4.2 精确题名交叉核验

请求模板：

<https://api.openalex.org/works?filter=title.search:TITLE&per-page=1&select=id,display_name,publication_year,publication_date,type,cited_by_count,doi,primary_location>

交叉核验题名：LTX-Video、LTX-2、HunyuanVideo、HunyuanVideo 1.5、Wan、Step-Video-T2V、Open-Sora 2.0、SkyReels-V2、MAGI-1、Ovi、Movie Gen、Cosmos World Foundation Model Platform for Physical AI。

结果：

- 12 个目标均能以题名和 DOI/arXiv ID 对齐；
- OpenAlex 将这些目标的主记录标为 preprint，而不是正式 proceedings；
- LTX-2、HunyuanVideo 1.5 等查询返回不止一个近似记录时，以题名、DOI 和 arXiv ID 三者共同裁决；
- OpenAlex 仅作书目交叉核验，不作为架构、发布状态或许可证的一手证据。

## 5. 检索表面三：官方代码、模型卡、技术报告与服务

### 5.1 核验方法

对每个官方组织执行以下只读检查：

1. GitHub repository metadata：owner、created/updated/pushed、default branch、license、archived 状态；
2. README/model card：checkpoint 名称、任务、参数量、依赖模块、下载与推理命令；
3. release/news/changelog：首次发布与后续资产是否真的落盘；
4. LICENSE 与模型条款：代码、权重和商业使用是否同一许可；
5. tree/config：训练、推理、LoRA、评测、SR/插帧、audio、serving 是否存在；
6. 官方项目/API：托管能力是否依赖未开放的前后处理。

GitHub 元数据通过 gh api repos/OWNER/REPO 与 gh api repos/OWNER/REPO/contents/PATH 获取；论文内容回到 arXiv HTML/PDF，避免用 README 的营销摘要替代论文。

核验的官方仓库：

- <https://github.com/Lightricks/LTX-Video>
- <https://github.com/Lightricks/LTX-2>
- <https://github.com/genmoai/mochi>
- <https://github.com/Tencent-Hunyuan/HunyuanVideo>
- <https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5>
- <https://github.com/Wan-Video/Wan2.1>
- <https://github.com/Wan-Video/Wan2.2>
- <https://github.com/stepfun-ai/Step-Video-T2V>
- <https://github.com/hpcaitech/Open-Sora>
- <https://github.com/SkyworkAI/SkyReels-V2>
- <https://github.com/SkyworkAI/SkyReels-V3>
- <https://github.com/SandAI-org/MAGI-1>
- <https://github.com/character-ai/Ovi>
- <https://github.com/NVIDIA/Cosmos>
- <https://github.com/MiniMax-AI/MiniMax-H3>

### 5.2 证据等级

| 等级 | 来源 | 可支持 | 主要限制 |
|---|---|---|---|
| E1 | 正式 proceedings、标准正文 | 论文机制、正式 venue、标准语义 | 不支持当前代码/服务状态 |
| E2 | 原始预印本、技术报告 | 作者披露的结构、训练和实验 | 尚未或未必同行评审；benchmark 为作者协议 |
| E3 | 官方代码、权重、模型卡、变更日志、许可证 | release surface、可运行资产、动态状态 | 资产存在不证明论文全复现；README 可含营销表述 |
| E4 | 官方产品页/API/system card | 某日服务能力、政策与接口 | 后端不透明；不能归因给单 checkpoint |
| E5 | OpenAlex/搜索索引、二手综述 | 发现与元数据交叉核验 | 不单独支撑技术或开放声明 |

冲突裁决顺序：

1. 开放状态：实际仓库 tree/权重清单/许可证优先于论文中的未来计划；
2. venue/作者/年份：正式 proceedings 或 arXiv metadata 优先于 README；
3. 系统能力：官方 API 文档可证明服务存在，但不可证明基础 checkpoint 单独提供；
4. benchmark：保留原作者口径，不与其他论文的数字直接排序；
5. 许可证：使用最具体的 artifact 条款；不以 GitHub 自动识别的 SPDX 代替模型许可证正文。

## 6. release surface 矩阵

符号：有=冻结日发现官方可用资产；部分=仅部分阶段/任务；无=官方明确未提供或未发现；托管=只能通过官方/合作服务；逐项=不同 artifact 条款不同。

| 实体 | 论文状态 | 代码 | 权重 | 训练/配方 | 数据/标注 | 许可边界 | API/产品 |
|---|---|---|---|---|---|---|---|
| LTX-Video | E2 预印本 | 有 | 有，多个尺寸/蒸馏变体 | 部分，LoRA/训练工具 | 无原始预训练数据 | 仓库 Apache-2.0 | 旧家族；主开发迁往 LTX-2 |
| LTX-2/2.5 | E2 预印本 | 有 | 有，主干与独立 VAE/upscaler/LoRA | 有 LoRA/全参/IC-LoRA 工作流，非原始数据复现 | 无 | 2.5 自 2026-08-11 起为自定义社区许可；年收入不低于 1000 万美元的 Commercial Entity 仅在商业使用时需付费条款，非商业用途适用第 2.2 节 | 官方 API；DFR 为多模块管线 |
| Mochi 1 preview | 未发现对应正式论文 | 有 | 有 | LoRA 有；完整预训练无 | 无 | 仓库 Apache-2.0 | 本地 Python pipeline，不等于托管 API |
| HunyuanVideo | E2 预印本 | 有 | 有 | 部分 | 无 | 逐仓库/模型卡 | 官方产品与开源 artifact 分开 |
| HunyuanVideo-1.5 | E2 技术报告 | 有 | 部分；T2V/I2V 与指定蒸馏资产 | 2025-12-05 发布训练+LoRA | 无 | 逐仓库/权重卡 | 未把 roadmap 未勾选项算作发布 |
| Wan2.2 | 没有独立正式论文；Wan 论文对应 2.1 | 有，推理 | 有，按 T2V/I2V/TI2V/S2V/Animate 分权重 | 未发现完整预训练 | 无 | 代码仓库 Apache-2.0；权重逐卡核对 | 不把家族任务合并为一权重 |
| Step-Video-T2V | E2 技术报告 | 有，推理/评测 | 有，含 Turbo 资产 | 未发现完整预训练发布 | 无 | 代码仓库 MIT | README 的 caption/VAE API 是内部推理服务 |
| Open-Sora 2.0 | E2 预印本 | 有 | 有 | 有训练代码与配置 | 数据处理有，原始训练集无 | 仓库 Apache-2.0 | 高质量 T2V 管线含外部 FLUX T2I 前级 |
| SkyReels-V2 | E2 预印本 | 有，推理 | 部分；含 captioner | 完整预训练未发现 | captioner 权重有，原始数据无 | 逐仓库/模型卡 | “infinite”不代表无漂移 |
| SkyReels-V3 | E2 技术报告 | 有，推理 | 有，三个任务家族 | 无完整训练 | 无 | 逐仓库/权重卡 | 合作 API；不是通用单 checkpoint |
| MAGI-1/1.1 | E2 预印本 | 有，推理 | 有，base/distill/quant 按尺寸 | 无完整预训练 | 无 | 仓库 Apache-2.0 | 无法从名称推断 world-model 闭环 |
| Ovi | E2 预印本 | 有 | 有 | 无完整预训练 | 无 | 仓库 Apache-2.0 | 开放 checkpoint，非正式 venue |
| Movie Gen | E2 预印本 | 无完整官方实现 | 无 | 无 | 无 | 不适用 | 项目 demo；Video/Audio/Edit/Personalized 为家族 |
| Cosmos/Cosmos 3 | E2 预印本 | 有，分组件 | 有，分家族/任务 | 部分 post-train/recipe | curator 工具有，原始训练集无 | 逐资产 | vLLM-Omni/SGLang/NIM 等任务面不同 |
| Seedance 2.0 | E2 预印本 | 无 | 无 | 无 | 无 | 服务条款 | 官方项目与托管 API |
| MiniMax H3 | 官方仓库/发布；无完整正式论文 | 有，H3-Base 推理 | 有，FL2VA/Ref2VA | 无完整训练；初始 sparse attention 未开 | 无 | MiniMax H3 Community License | Context-IR 与 Regenerate-2K 为托管组件 |

## 7. 关键 fresh-check 判定

### 7.1 LTX-Video、LTX-2 与 LTX-2.5

- LTX-Video 官方 README 已说明主要开发迁往 LTX-2，旧页面不能代表当前家族。
- LTX-2 README 在冻结日默认描述 LTX-2.5：22B distilled/development transformer、Gemma 4 12B projection、独立视频/音频 VAE、latent upscaler、LoRA/IC-LoRA 与 duration 组件；最终 pipeline 不是单一权重。
- [CHANGELOG 1.2.0](https://github.com/Lightricks/LTX-2/blob/main/CHANGELOG.md)（2026-08-11）加入 2.5、Gemma 4、checkpoint-driven architecture、diffusion VAE decode、auto-duration 与 2.5 training workflow。
- 同一 changelog 的 1.3.0（2026-08-25）加入 DFR 4K tiled spatial epilogue、多 GPU DFR、keyframe-aware diffusion-VAE decode 等。4K 应写成 pipeline/release-note 能力，不写成 base-checkpoint benchmark。
- [LICENSE-2_x](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) 对 LTX-2.5 采用自定义社区协议；依第 2.2 节，年收入不低于 1000 万美元的 Commercial Entity 仍可在协议下作非商业用途，但进行商业使用前须另行取得付费 Commercial Use Agreement。故使用“开放权重/公开代码和工作流”，不写“宽松开源”。

### 7.2 HunyuanVideo-1.5

- 官方 repo 标明 8.3B、T2V/I2V、3D causal VAE、SSTA 和 SR。
- 2025-11-20：推理代码与模型权重。
- 2025-12-05：训练代码+LoRA、480p I2V step-distilled checkpoint；该 checkpoint 是具体 artifact，不代表所有表格变体。
- roadmap 中“all model weights (Sparse attention, distill model, and SR models)”仍未全勾选；配置文件行或 “coming soon” 不计为发布。

### 7.3 Wan2.2

- 官方 repo 于 2025-07-28 发布推理代码与权重。
- T2V-A14B、I2V-A14B、TI2V-5B、S2V-14B、Animate-14B 是不同任务 artifact；后两项在后续更新增加。
- A14B 的 MoE 沿去噪时间阶段分专家；TI2V-5B 使用高压缩 VAE 并支持官方配置中的 720p/24fps。两项不能合并为一个 checkpoint 的属性。
- arXiv 2503.20314 是 Wan/2.1 论文，不虚构独立 Wan2.2 正式论文。

### 7.4 Open-Sora 2.0

- 官方 repo 记录 2025-03-12 发布，11B、checkpoint 与训练代码，Apache-2.0。
- 单模型支持 256/768px T2V/I2V，但 README 同时说明其偏向 I2V；高质量 T2V 使用 FLUX text-to-image 后再 image-to-video。
- 因而“Open-Sora 2.0 T2V 系统”可能包含外部图像模型；约 20 万美元训练成本和效果对齐均按作者 claim 转述。

### 7.5 SkyReels-V2 与 V3

- V2 论文披露 raw video→切镜/caption→多级质量/安全/去重→裁切平衡→渐进预训练→概念 SFT→运动 DPO/reward→Diffusion Forcing→HQ-SFT。
- V2 repo 有推理权重和 SkyCaptioner-V1，但 5B、camera director、部分 step/guidance-distill 项仍列 TODO。
- V3 不能被 V2 覆盖：官方 repo 于 2026-01-29 发布 inference code 和 task-specific weights，分别为 Reference2Video 14B-720P、Video-Extension 14B-720P、TalkingAvatar 19B-720P，并列合作 API。

### 7.6 MAGI-1 与 Ovi

- MAGI-1 的官方源是 SandAI-org/MAGI-1，不使用第三方 fork；chunk-wise causal generation、shortcut/CFG distillation、24B/4.5B base/distill/quant 按 artifact 分开。
- Ovi 的官方源是 character-ai/Ovi 和 arXiv 2510.01284；twin backbone 做跨模态 A/V 融合，视频分支继承 Wan2.2、音频 VAE 继承 MMAudio。官方有代码与权重，但论文仍是预印本。

### 7.7 Movie Gen、Cosmos、Seedance 2.0 与 MiniMax H3

- Movie Gen 是 “cast of models”：Video 30B、Audio 13B、Personalized 与 Edit 分支；Audio 是视频到音频，不是同一 latent 中联合生成。无完整官方代码/权重。
- Cosmos v1 是 curator、tokenizer、AR/diffusion、post-train 与 guardrail 平台；Cosmos 3 又是 AR reasoner、diffusion generator、不同尺寸/任务和 serving 的家族。平台范围不等于每个 checkpoint 有动作条件或闭环证据。
- Seedance 2.0 有预印本、官方项目与托管 API，但未发现完整公开代码/权重；联合 A/V 是产品/报告能力。
- H3-Base 是 33B 单流联合视频/立体声音频模型，开放的是 768p base 与 FL2VA/Ref2VA 两类 CFG-distilled 权重；完整 H3 还包括未开放的 Context-IR 与 Regenerate-2K，初始 sparse-attention 实现也未开。2K 只归因给托管系统。

## 8. 系统层 claim-to-source 映射

| 系统层 | 主证据 | 等级 | 正文使用边界 |
|---|---|---|---|
| 数据过滤/去重/caption | SkyReels-V2、Open-Sora 2.0、Cosmos 报告 | E2 | 作者披露的 pipeline，不假定原始数据公开 |
| 离散/连续 tokenizer | VQ-VAE、MAGVIT、VideoPoet、CogVideoX | E1 | 正式机制基础；不代表当前 release |
| 表示前沿 | V-RAE、VideoRAE | E2 | 2026 假设，标为预印本 |
| post-train/preference | Step-Video、SkyReels-V2、Systematic Post-Train | E2 | 只支持作者协议与阶段分解 |
| causal/few-step distillation | Diffusion Forcing、CausVid、MAGI-1 | E1+E2/E3 | factorization、objective 与部署分别归因 |
| SR/DFR | LTX-2 changelog、Hunyuan1.5 repo | E3 | pipeline/module 能力，不归给 base |
| 联合 A/V | LTX-2、Ovi、H3、Seedance 2.0 | E2/E3/E4 | 区分单流、双 backbone、托管产品 |
| safety/provenance | Cosmos repo、C2PA 2.4 | E3+E1 标准 | provenance 证明来源声明绑定，不证明事实真伪 |
| API/serving | 官方 README/API 文档 | E3/E4 | endpoint 任务面，不反推权重或训练 |
| 开放状态/许可 | 官方 tree、model card、LICENSE | E3 | 截止日快照，逐 artifact 解释 |

## 9. 排除与负面发现

- 未发现 Mochi 1 preview 的正式论文；保留为官方开放 checkpoint，而不是补写不存在的 venue。
- 未发现独立 Wan2.2 正式论文；Wan 论文只支持 2.1 的机制背景。
- 未发现 MiniMax H3 完整技术报告；只从官方 repo 支持明确的架构和 release boundary。
- 未发现 Movie Gen、Seedance 2.0 的完整官方开放权重/训练代码。
- 不把 SkyReels-V2 “infinite-length”、Open-Sora 2.0 “commercial-level/$200k”、HunyuanVideo-1.5 “SOTA”等作者词汇写成独立事实。
- 不把 Cosmos 的 “world foundation model” 或 MAGI 的产品命名当作动作条件、反事实和闭环控制证据。
- 不使用官方仓库之外的 Mochi、MAGI、Ovi fork 判断发布状态。
- 不纳入冻结日后更新；若 README 未来更改，必须以 commit/tag 或新审计日期重新冻结。

## 10. 可复核性与遗留风险

### 10.1 已执行

- arXiv 三个宽检索表面与 15 个指定 ID/题名核验；
- OpenAlex 四个宽检索和 12 个精确题名交叉核验；
- 15 个官方 GitHub 仓库及相关 README/changelog/license/model card fresh-check；
- 正式 proceedings 与预印本分栏；
- 动态开放状态标注冻结日；
- 机制、checkpoint、family、product、artifact、API 六种对象分栏；
- 章节中的 2 张 Mermaid 均要求实际渲染，另有一张系统栈 PNG 与独立图示研究日志。

### 10.2 遗留风险

1. README、模型文件与许可证可在同一默认分支无版本更新提示地变化；高风险部署应保存 commit SHA 与权重 hash。
2. 部分权重托管在 Hugging Face，repository license、model card 与单文件条款可能不一致；商用前需逐 artifact 法务复核。
3. 训练数据通常不开放，无法独立复核权利、污染、去重和 caption 质量。
4. 作者 benchmark 跨系统不可比，本章没有运行这些巨型模型，因而不声称独立复现能力或速度。
5. 托管 API 的前后处理和路由不可见；产品输出只能归因给版本化系统。
6. C2PA 能提供签名 provenance，不提供语义真实性判定；部署仍需内容审核与事件响应。

## 11. 章节验收命令与实际结果

2026-08-30 执行结果：

1. markdownlint：npx --yes markdownlint-cli2 docs/foundation-models.md sources/research_20260830_video_foundation_models.md；markdownlint-cli2 0.23.2 / markdownlint 0.41.1，0 issue。
2. ref-N：提取正文引用标签、目标和 id；63 次引用、42 个唯一锚点，无 label/target 不一致、缺失、重复或孤立，编号连续。
3. 相对链接：解析两文件 Markdown link/image target；13 个相对目标全部存在，包括系统栈 PNG、图示日志和本研究日志。
4. 外链：对正文 42 个唯一 HTTP(S) 引用执行 HEAD，失败时以 range GET 复核；42/42 可达。该检查证明链接当时可访问，不证明页面以后不变。
5. Mermaid：从正文提取 2 个 .mmd，以 @mermaid-js/mermaid-cli 11.16.0 实际渲染为 SVG；两图均成功，文件非空，并在 SVG 中保留 title/desc。首次调用因 CLI 默认 chrome-headless-shell 未安装而失败；显式使用系统 Google Chrome 后重跑通过。
6. diff：git diff --check 检查正文，git diff --no-index --check /dev/null 检查新研究日志；均无 whitespace error。
7. scope：本子任务只写 docs/foundation-models.md 与 sources/research_20260830_video_foundation_models.md；共享 PNG 与其图示日志由根任务生成，本子任务只读取并引用。未提交、未 push。
