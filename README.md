# Video Generation 101：视频生成技术手册

_Created by Codex and <a href="https://vinthony.github.io/">Xiaodong Cun (Corresponding Author)</a>, from <a href="https://gvclab.github.io/">GVC Lab, Great Bay University</a>_

[在线阅读与搜索](https://gvclab.github.io/Video-Generation-101/) · [手册使用说明](docs/manual-guide.md) · [贡献指南](CONTRIBUTING.md)

本手册介绍视频生成的基本原理、模型结构、训练与推理流程、任务接口、数据处理和评测方法，面向学习、研究和系统开发。技术章节说明方法的输入输出、工作机制、适用条件、配置要求与常见故障；资源章节提供论文、模型和数据的查阅入口。

手册结构修订日期：**2026-10-15**。各章资料覆盖时间及动态资源状态见条目说明。本次编辑不代表所有模型均已复现或所有资源均已重新下载测试。

## 使用方法

1. 初次阅读从[视频生成入门](docs/getting-started.md)和[方法分类](docs/generative-models.md)开始。
2. 实现具体任务时，先在[任务分类](docs/taxonomy.md)确定输入、输出与约束，再进入相应专题。
3. 选择[模型](resources/open-models.md)与[数据](resources/datasets.md)后，按[评测方法](docs/evaluation.md)固定配置、对照和验收条件。
4. 按学习目标安排阅读顺序，可使用[读者指南](docs/reader-guides.md)与[论文阅读清单](docs/reading-list.md)。

## 内容目录

| 板块 | 先读 | 分组与后续入口 |
|---|---|---|
| 使用与索引 | [使用说明与章节分工](docs/manual-guide.md) | [入门](docs/getting-started.md) · [阅读路线](docs/reader-guides.md) · [任务接口](docs/taxonomy.md) · [能力地图](docs/foundation-model-capabilities.md) |
| 生成原理 | [五轴方法总览](docs/generative-models.md) | 表示与骨干（[Tokenizer](docs/generative-models/video-tokenizers.md)、[DiT](docs/generative-models/video-dit-backbones.md)）；分解与递推（[自回归](docs/generative-models/autoregressive-generation.md)、[掩码](docs/generative-models/masked-generation.md)、[递归](docs/generative-models/recurrent-prediction.md)）；目标与过程（[变分](docs/generative-models/variational-generation.md)、[对抗](docs/generative-models/adversarial-generation.md)、[扩散](docs/generative-models/diffusion-models.md)、[Flow/Consistency](docs/generative-models/flow-consistency-models.md)） |
| 训练、推理与系统 | [系统组成](docs/foundation-models.md) | [后训练](docs/generative-models/video-post-training-alignment.md) · [长视频](docs/generative-models/long-video-generation.md) · [因果与流式](docs/generative-models/causal-streaming-generation.md) · [推理预算](docs/generative-models/test-time-scaling.md) · [执行优化](docs/generative-models/inference-acceleration.md) · [智能体制作](docs/agentic-video-generation.md) |
| 任务与应用 | [任务分类](docs/taxonomy.md) | 创作输入与输出；控制、主体与叙事；编辑、恢复与补全；预测与交互接口。按输入输出进入专章，再到[应用设计](docs/applications.md)组合系统 |
| 世界建模、行动与推理 | [世界模型基础](docs/world-models.md) | 表示与动力学（[JEPA](docs/jepa.md)、[Coding World Model](docs/coding-world-models.md)）；行动耦合（[WAM](docs/world-action-model.md)）；约束与求解（[物理一致性](docs/physical-consistency.md)、[视觉推理](docs/video-reasoning.md)） |
| 评测与资源 | [评测总览](docs/evaluation.md) | [世界模型协议](docs/evaluation/world-models.md) · [数据](resources/datasets.md) · [开放模型](resources/open-models.md) · [时间线](docs/timeline.md) · [论文课程](docs/reading-list.md) · [文献索引](docs/bibliography.md) |

## 框架与章节分工

沿“任务接口 → 能力要求 → 建模机制 → 系统实现 → 验证证据”追踪同一个问题。导航子组标明正在回答的分类轴；同一模型可以同时使用连续表示、自回归分解、扩散目标和流式部署。长视频描述时间跨度，DiT 描述骨干，后训练描述训练阶段，不能把它们当作互斥的算法类别。

共用定义、公式推导和通用协议在[主责章节表](docs/manual-guide.md#chapter-ownership)指定的页面维护。相邻章节保留任务差异、实验条件、故障和必要的简短说明，通过链接进入完整技术内容。

## 技术说明与实验示例

正文区分作者报告、本仓库复现、方法解释和建议实验。未经运行的实验设计均作为示例使用，不能当作已发布基准或实测结果。比较结果时，需同时记录模型版本、数据、硬件、采样预算和指标实现。符号、术语及记录模板见[手册使用说明](docs/manual-guide.md)。

## 仓库文件

| 路径 | 内容 |
|---|---|
| `docs/` | 原理、任务、系统与评测章节 |
| `resources/` | 模型和数据资源 |
| `bibliography/` | 结构化文献元数据 |
| `assets/` | 图表与网页资源 |
| `sources/` | 来源记录、核验笔记与历史修订资料 |
| `scripts/` | 构建与检查工具 |

贡献前请阅读[贡献指南](CONTRIBUTING.md)。

## 引用

如果本仓库对您的研究、教学或项目有帮助，欢迎引用：

```bibtex
@software{video_generation_101_2026,
  title = {Video Generation 101: From Pixel Animation to World Model},
  author = {Cun, Xiaodong and Video Generation 101 contributors},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/GVCLab/Video-Generation-101}
}
```

完整引用元数据请参见 [CITATION.cff](CITATION.cff)。

## 许可

本仓库以 [MIT License](LICENSE) 发布。论文、数据集、模型及第三方材料仍遵循各自的许可证和使用条款。
