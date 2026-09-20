# Video Generation 101：视频生成技术手册

_Created by Codex and <a href="https://vinthony.github.io/">Xiaodong Cun (Corresponding Author)</a>, from <a href="https://gvclab.github.io/">GVC Lab, Great Bay University</a>_

[在线阅读与搜索](https://gvclab.github.io/Video-Generation-101/) · [手册使用说明](docs/manual-guide.md) · [贡献指南](CONTRIBUTING.md)

本手册介绍视频生成的基本原理、模型结构、训练与推理流程、任务接口、数据处理和评测方法，面向学习、研究和系统开发。技术章节说明方法的输入输出、工作机制、适用条件、配置要求与常见故障；资源章节提供论文、模型和数据的查阅入口。

手册结构修订日期：**2026-09-20**。各章资料覆盖时间及动态资源状态见条目说明。本次编辑不代表所有模型均已复现或所有资源均已重新下载测试。

## 使用方法

1. 初次阅读从[视频生成入门](docs/getting-started.md)和[方法分类](docs/generative-models.md)开始。
2. 实现具体任务时，先在[任务分类](docs/taxonomy.md)确定输入、输出与约束，再进入相应专题。
3. 选择[模型](resources/open-models.md)与[数据](resources/datasets.md)后，按[评测方法](docs/evaluation.md)固定配置、对照和验收条件。
4. 按学习目标安排阅读顺序，可使用[读者指南](docs/reader-guides.md)与[论文阅读清单](docs/reading-list.md)。

## 内容目录

| 板块 | 核心入口 | 延伸专题 |
|---|---|---|
| 生成原理 | [生成模型路线](docs/generative-models.md) | [视频 Tokenizer](docs/generative-models/video-tokenizers.md) · [Video DiT](docs/generative-models/video-dit-backbones.md) · [推理加速](docs/generative-models/inference-acceleration.md)（[蒸馏](docs/generative-models/inference-acceleration/distillation.md) · [量化](docs/generative-models/inference-acceleration/quantization.md) · [剪枝](docs/generative-models/inference-acceleration/pruning.md) · [稀疏](docs/generative-models/inference-acceleration/sparsity.md) · [缓存](docs/generative-models/inference-acceleration/caching.md) · [系统](docs/generative-models/inference-acceleration/systems.md)） · [Test-Time Scaling](docs/generative-models/test-time-scaling.md) · [长视频生成](docs/generative-models/long-video-generation.md) · [后训练与对齐](docs/generative-models/video-post-training-alignment.md) · [因果与流式生成](docs/generative-models/causal-streaming-generation.md) |
| 基础模型能力 | [能力地图](docs/foundation-model-capabilities.md) · [基础模型系统](docs/foundation-models.md) | [个性化](docs/tasks/personalized-video-generation.md) · [细粒度控制](docs/tasks/controllable-video-generation.md) · [多视角/4D](docs/tasks/multiview-4d-generation.md) · [原生音视频](docs/tasks/native-audio-video-generation.md) |
| 智能体与制作系统 | [Agentic Video Generation](docs/agentic-video-generation.md) | 规划与工具调用 · 视觉反馈 · 记忆与局部返工 · 预算与评测 |
| 编辑与时序任务 | [视频编辑](docs/tasks/video-to-video.md) | [视频补全](docs/tasks/video-inpainting.md) · [视频修复](docs/tasks/video-restoration.md) · [视频虚拟试衣](docs/tasks/video-virtual-try-on.md) · [故事与多镜头](docs/tasks/story-multishot.md) |
| 推理与世界模型 | [Video Reasoning](docs/video-reasoning.md) · [World Model](docs/world-models.md) · [World Action Model](docs/world-action-model.md) | [物理一致性](docs/physical-consistency.md) · [动作条件预测](docs/tasks/action-conditioned-prediction.md) · [交互式世界生成](docs/tasks/interactive-world-generation.md) |
| 应用与研究资源 | [应用系统](docs/applications.md) · [评测指南](docs/evaluation.md) | [精选阅读](docs/reading-list.md) · [开放模型](resources/open-models.md) · [数据集](resources/datasets.md) |

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
