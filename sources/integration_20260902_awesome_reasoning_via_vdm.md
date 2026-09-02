# Awesome-Reasoning-via-VDM 融合审计 — 2026-09-02

## 范围与结论

- **来源仓库：** [GVCLab/Awesome-Reasoning-via-VDM](https://github.com/GVCLab/Awesome-Reasoning-via-VDM)
- **冻结提交：** [`2159815271df5621f3f27fa589a622e0e6bf676f`](https://github.com/GVCLab/Awesome-Reasoning-via-VDM/commit/2159815271df5621f3f27fa589a622e0e6bf676f)
- **核验日期：** 2026-09-02，Asia/Shanghai
- **目标：** 将旧清单的发现入口并入 [`docs/video-reasoning.md`](../docs/video-reasoning.md)，同时保留最新正式 venue、任务边界和可验证证据口径。

冻结提交的树中只有一个 2,075 字节的 `README.md`，包含 9 个条目；没有代码、数据、图片、CI、引用文件或 `LICENSE`。因此本次采用**事实性书目信息重核 + 独立改写 + 来源致谢**，不做 Git subtree/submodule 合并，也不复制旧 README 的原文或版式。

## 9/9 条目迁移映射

| 旧清单条目 | 融合状态 | Video Generation 101 去向 | 核验或边界 |
|---|---|---|---|
| Video models are zero-shot learners and reasoners | 已覆盖 | [Video Reasoning ref 3](../docs/video-reasoning.md#ref-3)，并作为全章叙事原点 | 保留零样本、黑盒 prompt rewriter 与 pass@$k$ 边界 |
| MME-CoF | 已覆盖 | [Video Reasoning ref 5](../docs/video-reasoning.md#ref-5)；[评测章节](../docs/evaluation.md)亦有独立条目 | 使用 Findings of CVPR 2026 正式页，不回退到早期 arXiv 标签 |
| TiViBench | 已覆盖 | [Video Reasoning ref 7](../docs/video-reasoning.md#ref-7) | 区分基础模型能力与 VideoTPO 的测试时选择收益 |
| VMEvalKit | 别名归并 | [VBVR / ref 19](../docs/video-reasoning.md#ref-19) | 旧 GitHub 地址现重定向到 `Video-Reason/VBVR-EvalKit`，不是独立 benchmark |
| Gen-ViRe | 已覆盖 | [Video Reasoning ref 8](../docs/video-reasoning.md#ref-8) | 保留自动 judge 需要独立校准的边界 |
| Reasoning via Video / VR-Bench | 已覆盖 | [Video Reasoning ref 9](../docs/video-reasoning.md#ref-9) | 作为程序化迷宫训练与评测路线，不外推为开放世界推理 |
| Thinking with Video | 已覆盖 | [Video Reasoning ref 6](../docs/video-reasoning.md#ref-6) | 使用 CVPR 2026 正式信息，并与 2026 年的 *Thinking in Video* 区分 |
| MiniVeo3-Reasoner | 新增 | [Video Reasoning §8.1 与 ref 59](../docs/video-reasoning.md#ref-59) | GitHub 工程发布；无技术报告；项目方结果与独立复现分开 |
| DiffThinker | 新增为相邻路线 | [Video Reasoning §3.1 与 ref 60](../docs/video-reasoning.md#ref-60) | 原生 image-to-image diffusion reasoning，不计作视频模型的 Chain-of-Frames 证据 |

映射计数：**6 项既有正式条目 + 1 项历史别名归并 + 2 项新增或补界 = 9/9**。没有为同一工作新建重复 benchmark 行。

## 增量一手来源核验

| 项目 | 一手来源 | 本次采用的信息 | 不采用或降级的说法 |
|---|---|---|---|
| VMEvalKit / VBVR-EvalKit | [当前仓库](https://github.com/Video-Reason/VBVR-EvalKit)、[VBVR 项目页](https://video-reason.com/bench/) | 旧地址已重定向；当前仓库是 VBVR 的规则化评测工具，覆盖 100 个任务、50 ID + 50 OOD、共 500 个评测视频 | 不把旧名称当作一项额外 benchmark |
| MiniVeo3-Reasoner | [官方仓库](https://github.com/thuml/MiniVeo3-Reasoner)、[LoRA 权重](https://huggingface.co/thuml/MiniVeo3-Reasoner-Maze-5B) | Wan2.2-TI2V-5B + LoRA；程序化迷宫；公开训练、推理与 EM/PR 评测流程；仓库明确写明暂无技术报告 | 不把 README 的项目方分数写成独立复现或同行评审结论 |
| DiffThinker | [DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models](https://arxiv.org/abs/2512.24165)、[项目页](https://diffthinker-project.github.io/)、[官方仓库](https://github.com/lcqysl/DiffThinker) | 将多模态推理表述为 image-to-image diffusion；代码覆盖 Maze、FrozenLake、TSP、Sudoku 与 Jigsaw；ICML 2026 | 旧清单的 `Arxiv 26.01` 标签不准确：arXiv v1 日期是 2025-12-30；不把图像推理升级成视频推理 |

## 许可与归属边界

1. 来源仓库没有 `LICENSE`。公开可读不等于可把其表达性文字纳入本仓库的 MIT 许可。
2. 本次只迁移论文标题、标识符、日期、URL 和仓库重定向等事实，并根据论文、项目页和官方代码独立撰写解释。
3. 未复制来源仓库或所列项目的代码、图片、视频、数据、README 段落和排版。
4. MiniVeo3-Reasoner 与 DiffThinker 的第三方代码、基础模型和权重许可仍由各自上游决定；本仓库只提供链接和证据边界。
5. 是否把旧仓库改成迁移提示或归档，属于另一个外部写操作，不在本次融合范围内。

## 验收标准

| 要求 | 验收方式 |
|---|---|
| 旧清单 9 个条目全部有唯一去向 | 对照上表，计数为 9/9 |
| 不制造 VMEvalKit / VBVR 重复项 | 旧名称只作为 ref 19 的历史别名出现 |
| MiniVeo3 证据等级准确 | 明示工程发布、无技术报告、分数为项目方报告 |
| DiffThinker 分类准确 | 明示 image-to-image 相邻路线，不计入 VGM benchmark |
| 不引入无许可资产 | Git diff 中只有独立撰写的 Markdown，无上游代码或素材 |
| 引用与网站可用 | 运行引用闭合、相对链接、Markdown、站点严格构建和外链检查 |

## 验收结果

| 检查 | 结果 |
|---|---|
| 条目闭合 | 旧清单 9 项全部映射：6 项既有条目、1 项别名归并、2 项新增或补界 |
| 章节引用闭合 | `docs/video-reasoning.md` 的 60 个引用目标与 60 个锚点一一对应；无缺失或孤立锚点 |
| 针对性在线引用核验 | 本章与本审计共 54 个 arXiv ID，标题、年份及 URL venue 规则为 0 findings |
| Markdown | `markdownlint-cli2` 0.23.0 / markdownlint 0.41.0 检查两份文件：0 errors |
| 本地链接 | 两份文件中的相对链接全部解析到现有目标 |
| 外部来源 | 9 个唯一来源 URL 均可访问；8 个直接返回 HTTP 200，Hugging Face 模型卡由网页读取复核 |
| 严格网站构建 | Python 3.12、MkDocs 1.6.1、Material 9.7.6 下 `--strict` 构建成功 |
| 生成页面 | Video Reasoning 页、融合审计页和全文搜索索引均包含 MiniVeo3-Reasoner、DiffThinker 与迁移说明 |
| 变更边界 | 本次只新增独立撰写的审计 Markdown，并修改 Video Reasoning Markdown；未复制上游代码或素材，未触碰工作区原有未跟踪文件 |

全库在线引用扫描还会报告其他既有页面使用短名、别名或非标题链接文字的问题；本次只修正新增内容，并以针对本章和融合审计的 0-findings 结果作为验收，未扩大到无关章节。
