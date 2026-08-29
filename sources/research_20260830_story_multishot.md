# 故事与多镜头视频生成研究记录 — 2026-08-30

本文件记录 `docs/tasks/story-multishot.md` 重写所用的检索入口、纳排规则、证据等级、逐条事实、发布面与验收结果。它不是第二份教程。所有会随时间变化的“已公开/未公开”判断冻结于 **2026-08-30（Asia/Shanghai）**。

## 1. 审查问题

1. 长单镜头、续写、V2V、story visualization、storyboard 与 multi-shot 的最小边界是什么？
2. 输入/输出 tensor 与跨镜头状态必须公开哪些字段，才能复现？
3. 规划、bible、生成、memory、冲突和 rollback 分别由什么机制承担？
4. 2025–2026 年哪些工作改变了可证伪能力，哪些只是发布了样片？
5. OneStory、MultiShotMaster、VGoT、STAGE/ConStoryBoard、StoryMem、ShotStream 与 MuSS 到底公开了什么？
6. 如何避免把 SVD 当成 LLM storyboard 证据，或把静态故事图当视频？

## 2. 纳排与措辞协议

### 2.1 纳入

- 直接提出故事/多场景/多事件/多镜头视频生成机制的一手论文；
- 正式 proceedings/OpenReview 页面，用于核验 venue、年份和题名；
- arXiv 原始记录，用于尚未正式发表的 2025–2026 frontier；
- 作者/机构官方项目页、GitHub 和 Hugging Face，用于核验代码、权重、数据、许可证及 TODO；
- story visualization、storyboard、I2V、continuation 只作为任务边界或技术邻域。

### 2.2 排除或降级

- 综述、媒体稿、SEO 排行榜、转载仓库与第三方模型镜像不支持技术事实；
- 项目精选样片不支持总体质量、稳定性或复现主张；
- “will release / coming soon” 不记作已发布；
- 同一工作重复 arXiv 记录只保留作者确认的规范记录；
- 论文自报数值统一写为“论文/作者报告”，不改写成独立确认；
- 只输出静态图的工作不列为 multi-shot video 方法。

### 2.3 证据等级

| 等级 | 一手证据 | 允许的写法 |
|---|---|---|
| A | 正式 proceedings/OpenReview + 论文 | 可确认 venue 与“论文提出/报告” |
| B | arXiv/技术报告，可能附官方资产 | 只能写“作者提出/报告”，不称独立 SOTA |
| C | 官方 repo/project/model/data card | 只支持发布面与维护者自己的说明 |
| S | 本综述综合出的合同、状态机或实验 | 必须明确不是某篇论文已实现的完整功能 |

代码、权重和数据不是论文等级：A 级论文可能没有资产，B 级预印本也可能完整开源。

## 3. 检索日志

检索均在 2026-08-30 执行。计数仅用于发现与复核检索式，不代表领域总体规模。

| 入口 | query / 路径 | 返回/核验结果 | 用途与限制 |
|---|---|---:|---|
| arXiv API | `all:"multi-shot video generation"` | 20 | 精确短语发现；会漏掉 multi-scene/multi-event 命名 |
| arXiv API | `all:"multi-shot" AND (all:"storytelling" OR all:"cinematic narrative")` | 19 | 聚焦故事与电影叙事 |
| arXiv API | `(all:"story visualization" OR all:"storyboard") AND all:"generation"` | 113 | 查静态邻域与 storyboard；噪声较高 |
| arXiv API | exact IDs `2505.07652`, `2412.02259`, `2512.12372`, `2512.03041`, `2512.07802`, `2512.19539`, `2603.25746`, `2604.23789` | 逐条返回 | 核验题名、版本、日期和 author comment |
| OpenAlex API | `search=ShotAdapter` | 正式 work DOI `10.1109/CVPR52734.2025.02645` + arXiv work | 核验 ShotAdapter venue/DOI；技术事实仍回到论文 |
| OpenAlex API | exact-title search：OneStory、STAGE、MultiShotMaster、VGoT、StoryGAN | 均找到对应 work | 交叉核验题名/年份；不把引用数当质量证据 |
| CVF Open Access | CVPR 2025/2026 exact-title pages | 核验 ShotAdapter、HoloCine、OneStory、MultiShotMaster、STAGE、ShotDirector、DreamShot | 正式 venue 一手入口 |
| ECCV/ECVA | exact-title pages/PDF | 核验 StoryDALL-E、VideoStudio、MEVG | 正式 venue 一手入口 |
| NeurIPS | proceedings exact title；2025 virtual workshop page | 核验 EchoShot 主会与 VGoT NextVid workshop oral | workshop 不改写为主会 |
| 官方 GitHub | exact repository + README/tree/license/release | 核验 VGoT、MultiShotMaster、STAGE、StoryMem、ShotStream、MuSS、UnityShots 等 | README 声明与实际文件树分开记录 |
| Hugging Face API/页面 | exact organization/repo | 核验 MultiShotMaster、STAGE、ConStoryBoard、StoryMem、ShotStream、SEAM-Bench | 公开可访问不等于论文全表可复现 |

### 3.1 目标化 query families

- `multi-shot video generation story memory consistency transition`
- `storyboard anchored generation start end frame multi-shot`
- `streaming causal multi-shot video generation cache distillation`
- `multi-shot benchmark entity recurrence narrative cinematic`
- `site:openaccess.thecvf.com ShotAdapter OneStory MultiShotMaster STAGE`
- `site:github.com <exact title>` 与 `site:huggingface.co <exact title>`
- 候选的 exact title、arXiv ID、作者项目域名与正式 venue 页。

## 4. 逐条证据表

| 工作 | 主记录/venue | 本章使用的技术事实 | 冻结日发布面 | 限制或纠错 |
|---|---|---|---|---|
| StoryGAN | CVPR 2019 | 多句故事→静态图序列 | 正式论文 | 不是视频生成 |
| StoryDALL-E / Make-A-Story | ECCV 2022 / CVPR 2023 | 源图续故事、visual memory 的静态故事图 | 正式论文 | 没有镜头内运动 |
| Phenaki | ICLR 2023 OpenReview | 连续文本、可变长度视频 token | 正式论文 | 长视频不等于显式切镜 |
| SVD | arXiv:2311.15127 | I2V 数据/模型缩放 | 论文及其既有官方生态 | **不支持** LLM 剧本分解、storyboard 或跨镜头 memory |
| VideoStudio | ECCV 2024 | LLM 拆多场景脚本、共享实体参考、逐场景生成 | 正式论文 | pipeline 误差会跨模块传播 |
| VGoT | arXiv:2412.02259；NextVid workshop oral | storyline→镜头属性→关键帧→I2V→边界处理 | 官方代码/脚本/eval | 不是 NeurIPS 主会；2503.15138 为作者标注误重复提交 |
| ShotAdapter | **CVPR 2025** | transition token、local attention mask、逐镜头 prompt/长度 | 正式论文、项目页 | 旧稿把它只写作 arXiv；已用 DOI/正式页纠正 |
| OneStory | CVPR 2026 | Frame Selection + Adaptive Conditioner + I2V 自回归 | 正式论文、项目页 | 项目仍写 model/data will be released；未核验到官方权重/数据 |
| MultiShotMaster | CVPR 2026 | narrative RoPE、position-aware RoPE、reference token/mask、镜头控制 | 训练/推理代码；HF 1.3B/14B 权重 | 最完整公开面之一；仍需遵守其输入格式/基座条件 |
| STAGE | CVPR 2026 | STEP² 首尾帧对、memory pack、双编码、外部视频生成器动画化 | repo + HF STEP² 模型 | repo 的完整多镜头/训练/DPO 仍为 TODO；不能写成全系统已开源 |
| ConStoryBoard | STAGE/CVPR 2026 | 约 100K 电影片段、故事进度/电影属性、偏好子集 | HF dataset 可访问；API 报 96,642 files | file count 不是 sample count；不据此反推样本数 |
| StoryMem | arXiv:2512.19539 | 关键帧选择/过滤、latent 拼接、负 RoPE shift、MI2V/MM2V | 官方代码、HF LoRA；ST-Bench prompts | ST-Bench 仅 30 个合成故事/300 prompts |
| ShotStream | arXiv:2603.25746；官方称 ECCV 2026 accepted | 双向 teacher→因果 student、global/local cache、self-forcing | 代码、训练/推理配置、Wan2.1-1.3B 权重 | proceedings 在冻结日尚未发布；论文原模型含内部数据，开源实现非完全复现 |
| MuSS | arXiv:2604.23789 | 3,000+ 影片、30K+ 片段、1,000+ 小时；主体/电影叙事双轨指标 | repo 仅数据构建代码 | 数据、benchmark 实现、权重未随 repo 发布，许可证仍待定 |
| CausalCine | arXiv:2605.12496 | context-aware historical KV routing、少步因果生成 | 论文/项目页 | 未核验到官方代码/权重 |
| CineWeaver | arXiv:2607.26529 | 训练免位置编码/attention 操纵、shot-routed reference、anchor memory | 论文/项目页 | 未核验到官方代码/权重 |
| HoloCine | CVPR 2026 | Window Cross-Attention、Sparse Inter-Shot Self-Attention | 推理代码与 14B 模型 | 部分更小/更长/keyframe/audio 变体仍待发布 |
| ShotDirector | CVPR 2026 | 6-DoF/内参相机、剪辑 pattern、shot-aware mask | 正式论文 | 本章不据论文页推断额外权重发布 |
| DreamShot | CVPR 2026 | 视频 prior 辅助个性化 storyboard | 正式论文 | 输出是静态 storyboard，不是视频 |
| UnityShots | arXiv:2606.21661 | opening-shot 长期 + preceding-tail 短期 audio-video memory | 官方 repo/项目 | 训练代码、权重、agent 系统仍列 soon |
| MSVBench | arXiv:2602.23969 | 层级脚本、参考图、LMM+专家指标 | 预印本 | 94.4% Spearman 为作者报告，需新模型外部校准 |
| EntityBench | arXiv:2605.15199 | 140 episodes/2,491 shots、回归间隔至 48、fidelity gate | 官方 repo | 聚焦实体，不包办电影叙事质量 |
| PersonaShot | arXiv:2608.16717 | 约千段、16 个物理/情感/电影语法指标 | 预印本 | 冻结日前未核验官方代码/数据 |
| LogiShot | arXiv:2608.08820 | context video + 多条件逻辑生成 | 预印本 | 论文称将发布；冻结日不计为已发布 |
| SEAM | arXiv:2608.22725 | prompt 层 episodic memory graph、检索/过滤/回写 | SEAM-Bench HF | 主要是 storyboarding/prompt 系统，不是完整视频 generator |

## 5. 关键发布面快照

| 项目 | Paper | Project | Code | Weights | Data/benchmark | 本章表述 |
|---|---:|---:|---:|---:|---:|---|
| OneStory | A | 是 | 未核验 | 未核验 | 未核验 | 正式论文 + 项目，不称开源模型 |
| MultiShotMaster | A | 是 | 是 | 是 | 示例/脚本 | 公开训练/推理及 1.3B/14B 权重 |
| VGoT | B + workshop | 是 | 是 | 依赖外部基座 | eval/assets | 开源 pipeline，不称主会 |
| STAGE/ConStoryBoard | A | 是 | 部分 | STEP² 是 | 是 | storyboard 层可复现，完整 pipeline 未发布 |
| StoryMem | B | 是 | 是 | 是 | ST-Bench 是 | 可运行 memory LoRA；benchmark 较小且合成 |
| ShotStream | B + accepted 声明 | 是 | 是 | 是 | prompts/config | 可运行公开实现；原论文内部数据缺口保留 |
| MuSS | B | 是 | 仅 construction | 否 | 数据/实现未核验 | 不称数据集已完整发布 |

## 6. 明确排除项与负面核验

1. **SVD 作为 LLM storyboard 证据：排除。** 它只支持 I2V 基座事实。
2. **arXiv:2503.15138 作为第二篇 VGoT：排除。** 其 author comment 说明是 2412.02259 精修版误作新论文提交。
3. **把 VGoT 写成 NeurIPS 主会：排除。** 核验到的是 2025 NextVid workshop oral。
4. **把 StoryGAN、StoryDALL-E、Make-A-Story、DreamShot 直接列成视频方法：排除。** 它们输出静态故事图/storyboard。
5. **把 Phenaki 的变长视频当多镜头结构证据：排除。** 只作长单流邻域。
6. **把 ConStoryBoard 仓库文件数当样本数：排除。** 只记录 API 文件面和论文规模表述。
7. **把 OneStory 项目页的 will release 当成已发布：排除。**
8. **把 STAGE repo 当完整 end-to-end release：排除。** README 中完整 pipeline、训练与 DPO 仍未完成。
9. **把 MuSS repo 当数据/模型已发布：排除。** 冻结日仅见 construction code。
10. **第三方博客、模型聚合站与转载 repo：全部排除。**

## 7. 本章综合判断的来源边界

“规划→bible→镜头生成→验证→原子 memory update→局部重试/依赖回滚”状态机是本综述对 VideoStudio/VGoT 的规划、OneStory/StoryMem 的选择性记忆、STAGE 的关键状态锚定、ShotStream/CausalCine 的因果生成路线做出的工程综合（等级 S）。没有任何被引论文被描述为已经完整实现事务提交、冲突检测和依赖级 rollback。

同理，正文的 12 故事 × 8 镜头 × 回归间隔实验是预注册建议，不是已执行结果。它刻意把 direct multi-shot、iterative memory、streaming、pipeline 和 storyboard 子任务分轨，避免把不兼容的输入/输出合同压成虚假总榜。

## 8. 验收记录

最终验收于 2026-08-30 执行：

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Markdown lint | `npx --yes markdownlint-cli2 docs/tasks/story-multishot.md sources/research_20260830_story_multishot.md` | `markdownlint-cli2 v0.23.2` / `markdownlint v0.41.1`；2 文件，0 issues |
| 引用闭环 | 脚本比较正文 `[[N]](#ref-N)` 与 `<a id="ref-N"></a>[N]` 集合 | 34 个被引编号、34 个定义；dangling 0、unused 0、编号错配 0 |
| 本地链接 | 解析两个 Markdown 的相对链接并检查目标存在 | 正文到研究记录、正文/日志到教学图的相对路径均存在；0 missing |
| 外链 | 对两个文件所有唯一 HTTP(S) URL 并发 HEAD，受限状态回退 ranged GET | 54 个唯一 URL，54 个 HTTP 200；失败 0 |
| Mermaid | 提取正文两个 Mermaid block，用 `@mermaid-js/mermaid-cli 11.16.0` + 系统 Google Chrome 实际渲染临时 SVG | 2/2 成功；30,243 B 与 32,718 B；均含 1 个 `<title>` 和 1 个 `<desc>`；未把临时渲染产物纳入仓库 |

## 9. 变更约束

- 修改 `docs/tasks/story-multishot.md`，新建本日志与一张教学总图；
- 未修改 `sources/coverage_audit_20260829.md`；
- 未 commit、未 push。

## 10. 图像资产记录

- 项目文件：[`assets/diagrams/story-multishot-memory-conflict.png`](../assets/diagrams/story-multishot-memory-conflict.png)
- 生成提示核心：白底、16:9、无模型名；六步循环为“故事规划 → 镜头合同 → 镜头生成 → 状态提取 → 记忆更新 → 冲突检查”，用绿色通过路径进入下一镜头，用红色冲突路径回滚重生成，并限定身份、场景、时间、因果四类检查。
- 像素尺寸：1672 × 941，RGB PNG。
- SHA-256：`e7a19feb776fd0f386d1fb39698f7a9a48904b7eb4dd71fa6c2763abc961dacf`。
- 视觉回读：六个主步骤、两条闭环和四类冲突标签均清晰；没有论文名、模型名、分数或超出正文的能力结论。正文同时保留 Mermaid 与顺序化文字替代。
