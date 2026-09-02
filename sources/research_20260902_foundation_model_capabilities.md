# 视频基础模型能力地图：检索、分类与证据记录

## 1. 任务与范围

- 检索与核验日期：2026-09-02（Asia/Shanghai）。
- 目标页面：`docs/foundation-model-capabilities.md`。
- 核心问题：物理一致性在视频基础模型能力体系中的位置；还有哪些可跨任务复用的能力；后训练应算能力、机制还是系统阶段。
- 输出目标：建立“能力 / 任务接口 / 获得方式 / 系统属性”四轴分类，并与仓库现有专题建立链接。
- 不做的事：不制作模型排行榜；不把单个 benchmark 的维度当成唯一标准；不把 2026 新预印本写成社区共识；不从产品 demo 反推 base checkpoint。

本记录是面向知识地图的定向叙事综述，不是穷尽式系统综述。正式 proceedings、原始预印本和官方研究页面作为一手证据；二手综述仅用于发现，不用于支撑关键分类。

## 2. 检索问题与纳排标准

### 2.1 检索问题

1. 现有视频生成 benchmark 如何拆分视觉、时间、语义、控制、物理和常识？
2. 统一视频 foundation model 是否还覆盖理解、编辑、重建和生成式推理？
3. 长时、多视角、音视频和动作条件系统各自新增了什么能力合同？
4. SFT、偏好优化、RL、蒸馏和测试时引导分别更新什么对象？
5. 何时可把结果归因给 base checkpoint，何时只能归因给 post-trained checkpoint 或完整系统？

### 2.2 纳入

- 为至少一项能力提供明确、可测试定义的 benchmark 或正式论文；
- 将多个任务统一到 foundation model 视角的工作；
- 能区分 base、后训练、推理期或系统能力的训练/部署论文；
- 仓库中已有完整专题且证据已逐项核验的物理、推理、音视频、4D 与 World Model 工作。

### 2.3 排除或降级

- 只有产品宣传词、精选样例或动态排行榜；
- 把分辨率、FPS、NFE、API 文件上传能力直接写成语义能力；
- 只因模型名称包含 “world” 就推断物理、因果或闭环能力；
- 把 reward model 的评价能力误写成 generator 的生成能力；
- 把 training reward 提升当成独立终评证据。

## 3. 关键一手证据

| 来源 | 证据等级 | 用于本页的内容 | 主要边界 |
|---|---|---|---|
| [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) | 原始报告 | 广泛预训练后可跨任务适配的定义 | 不是视频专属 taxonomy |
| [VBench, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) | 正式论文 | 视觉质量、时间一致、条件一致的基础维度 | benchmark 维度不是内部机制 |
| [VBench-2.0](https://arxiv.org/abs/2503.21755) | 2025 预印本 | Human Fidelity、Controllability、Creativity、Physics、Commonsense | 仍依赖多种自动 evaluator；不是社区唯一标准 |
| [UniVBench, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_UniVBench_Towards_Unified_Evaluation_for_Video_Foundation_Models_CVPR_2026_paper.html) | 正式论文 | 理解、生成、编辑、重建的统一任务表面 | 属于任务接口视角，不是原子能力全集 |
| [T2V-CompBench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.html) | 正式论文 | 数量、属性、动作、关系和交互的组合诊断 | 不能单独验证物理与因果 |
| [SV4D 2.0, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_SV4D_2.0_Enhancing_Spatio-Temporal_Consistency_in_Multi-View_Video_Diffusion_for_ICCV_2025_paper.html) | 正式论文 | view–time 联合一致性 | 作者协议不能直接跨数据排序 |
| [VABench, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_VABench_A_Comprehensive_Benchmark_for_Audio-Video_Generation_CVPR_2026_paper.html) | 正式论文 | 跨模态语义、事件同步、口型和立体声维度 | evaluator 与用户研究仍有域边界 |
| [PhyGenBench, ICML 2025](https://proceedings.mlr.press/v267/meng25c.html) | 正式论文 | 27 条物理规律的可见诊断 | 不提供精确状态或动作反事实 |
| [VideoPhy-2, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c02f6a1d5c55e16db50d339dad905b4d-Abstract-Conference.html) | 正式论文 | 更困难的 action-centric 物理常识 | 可见结果仍不等于闭环规划能力 |
| [WorldModelBench, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4ec03ed08a3fcb59e1c815b5598beff1-Abstract-Datasets_and_Benchmarks_Track.html) | 正式论文 | 指令、常识与可见物理违规 | 主要是生成视频诊断，不测试策略收益 |
| [Video models are zero-shot learners and reasoners](https://arxiv.org/abs/2509.20328) | 2025 预印本 | 零样本感知、操作和生成式推理行为 | 闭源模型、prompt rewriter 与 pass@$k$ 归因 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 2025 预印本 + 官方 artifact | 理解、预测、动作条件 predictor 与规划的模块边界 | action-free encoder 本身不是动作模型 |
| [VideoDPO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html) | 正式论文 | 多维偏好数据更新 generator | pair 与 evaluator 覆盖决定能力范围 |
| [VideoAlign, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76227feb18ea0ee40bd15cf02c33e18e-Abstract-Conference.html) | 正式论文 | 训练期 Flow-DPO/RWR 与推理期 Flow-NRG 的区别 | reward hacking 与循环评价仍需独立审计 |
| [T2V-Turbo, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a57aa8e8b57e64a42e95f7dceb0adb9-Abstract-Conference.html) | 正式论文 | reward 与 consistency distillation 的双重合同 | 偏好改善与少步加速必须分别消融 |

## 4. 分类决策

最终采用九类能力，而不直接照抄任一 benchmark：

1. C1 生成质量与分布建模；
2. C2 语义与组合遵循；
3. C3 时间动力学与长期状态；
4. C4 空间、几何与多视角一致性；
5. C5 控制、编辑与个性化；
6. C6 多模态音视频协同；
7. C7 常识、物理与反事实一致性；
8. C8 视觉理解与生成式推理；
9. C9 动作条件世界建模与决策。

归类原则：每一类都必须能写成可证伪行为，并能跨至少两个任务接口复用。安全、鲁棒性、校准、多样性和泛化作为横向属性；速度、流式、显存、能耗、provenance 与 API 作为系统属性。

“可适配性 / 可对齐性”作为 foundation model 的元能力：它衡量固定 base 在受控数据、参数和计算预算下获得新行为且保留旧能力的效率。CPT、SFT、DPO、RL、蒸馏与 test-time adaptation 是实现或测量这项元能力的路线，不是与 C1–C9 并列的行为名称。

## 5. 图像生成与检查

- 文件：`assets/diagrams/video-foundation-capability-map.png`
- 用途：说明三层能力、形成机制与证据升级之间的多对多关系。
- 生成方式：内置图像生成工具；本地 scientific-schematics 脚本因缺少 `OPENROUTER_API_KEY` 未运行。
- 尺寸：1672 × 941，RGB PNG。
- SHA-256：`8581ba9b7c40021e48da879fe1c222747189e5a1a7f04df8a0d4a22a2a9114fa`
- 人工检查：标题、三层能力、五类形成机制、证据阶梯与页脚均可读；无 logo、水印或模型品牌；箭头较密，仅表示多对多影响，正文 caption 已明确不是固定流水线或因果保证。

最终生成提示保留的核心文本：

```text
视频基础模型：能力 × 形成机制 × 验证证据
基础生成能力：视觉质量、时间一致、条件遵循
结构化创作能力：身份与状态、控制与编辑、空间与 4D、音视频协同
世界与行动能力：常识与物理、视觉推理、动作响应、闭环决策
形成机制（不是能力）：预训练、SFT、偏好优化 / RL、蒸馏、测试时引导
证据升级：样本、多随机种子、反事实、闭环
```

## 6. 与现有章节的分工

- `docs/foundation-model-capabilities.md`：能力是什么。
- `docs/foundation-models.md`：基础模型系统怎样从数据、checkpoint 走到服务。
- `docs/taxonomy.md`：任务输入、输出、保持约束与运行协议。
- `docs/generative-models.md`：表示、factorization、objective、backbone 与 deployment。
- `docs/generative-models/video-post-training-alignment.md`：后训练更新谁、使用什么反馈、增加什么成本。
- `docs/evaluation.md`：能力声明应由什么实验、指标和证据支持。
- `docs/physical-consistency.md`、`docs/video-reasoning.md`、`docs/world-models.md`：C7–C9 的深入证据边界。

## 7. 更新触发条件

以下情况出现时，应重新审阅能力地图：

1. 出现被广泛采用的统一 video foundation model capability benchmark；
2. 正式论文取代本页使用的 VBench-2.0 或零样本 reasoning 预印本；
3. 交互 world model 在独立环境中给出可复现的反事实与策略收益证据；
4. 后训练研究建立跨 C1–C9 的统一迁移、遗忘和 reward-hacking 协议；
5. 原生联合音视频、长时多镜头或 4D 系统改变当前任务与能力边界。
