# Agentic Video Generation 调研与证据记录

- 检索与页面核验日期：2026-09-20（Asia/Shanghai）。
- 交付章节：[Agentic Video Generation](../docs/agentic-video-generation.md)。
- 范围：具有明确规划、工具编排或反馈机制的视频制作系统；兼收具身视频计划作为任务边界对照。不是全面系统性综述。
- 检索式：`agentic video generation VideoAgent MovieAgent`、`agentic video generation automated filmmaking`、`ViMax Agentic Video Generation arxiv`、`VISTA Test-Time Self-Improving Video Generation Agent`、`FilmAgent Automating Virtual Film Production`。
- 纳入标准：原始论文或作者项目可访问，且能辨认输入输出、智能体操作位置与任务目标。排除二手文章作为机制证据，以及仅做视频问答却被名称误判为视频生成的工作。
- 未开展：生成实验、端到端代码复现、许可证逐项审计、完整发表状态审计或跨论文指标排名。

## 来源与论断

| 来源 | 实际核验范围 | 用于支持的论断 | 限制 |
|---|---|---|---|
| [VideoAgent](https://arxiv.org/abs/2410.10076) / [项目](https://video-as-agent.github.io/) | 摘要、v1/v3 日期和任务描述 | 外部反馈、视频计划细化与环境数据 | 具身控制证据，不外推影视效果 |
| [VideoGen-of-Thought](https://arxiv.org/abs/2412.02259) | 当前摘要页与检索返回版本 | 脚本、关键帧、镜头和衔接的模块化路线 | 检索摘要仍出现早期标题；章末采用当前页面标题 |
| [FilmAgent](https://arxiv.org/abs/2501.12909) / [作者仓库](https://github.com/UnitSeeker/FilmAgent) | 摘要、项目和 README | 虚拟 3D 场景中的多角色制作、反馈修订 | 脚本验证不等于开放域像素级修复 |
| [MovieAgent](https://arxiv.org/abs/2503.07314) / [作者仓库](https://github.com/showlab/MovieAgent) | 摘要、作者链接和仓库页面 | 剧本与角色库输入，层级场景/镜头规划 | 不声称已复现；不从角色数推断增益 |
| [Scientific VideoAgent](https://arxiv.org/abs/2509.11253) | 摘要与元数据 | 论文素材库、个性化编排、SciVidEval / Video-Quiz | 未独立验证作者效果宣称 |
| [VISTA](https://arxiv.org/abs/2510.15831) / [Google Research](https://research.google/pubs/vista-towards-test-time-self-improving-video-generation-agent/) | 摘要与官方研究页 | 计划、候选比较、三类批评、提示重写 | 两页人评比例不同；正文不摘录或混合结果数字 |
| [ViMax v2](https://arxiv.org/html/2606.07649v2) / [作者仓库](https://github.com/HKUDS/ViMax) | 摘要、全文方法 2.1–2.3 与仓库页面 | 层级叙事、RAG、视觉依赖、best-of-k 与过渡视频 | 将候选选择与任意局部修复/回滚分开 |

## 关键编辑判断

1. 新章位于 `docs/` 系统层，与任务地图、多镜头和 TTS 互链；现有站点没有显式 `nav`，由 MkDocs 自动发现页面，无需改动构建脚本。
2. 表格按 arXiv 首次提交时间，不把提交年份自动当作会议发表年份。VISTA 的官方研究页列 CVPR 2026，但此次直接访问对应 CVF 页面未成功，故章节以已读预印本描述机制。
3. 工作定义、参考架构、状态表、诊断动作、停止规则与最小实验为编辑者综合分析，正文明确其性质，不声称由某篇论文完整实现或已验证。
4. 本次新增论文在章末保留局部引用。中央 `bibliography/registry.json` 的现有 scope 默认排除只在单专题提供局部证据的条目；不为本章批量重生成中央文献库。
5. README 仅增加章节入口和目录项；全库证据快照未整体更新。
