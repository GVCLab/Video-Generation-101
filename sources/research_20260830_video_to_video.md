# Video-to-video 编辑研究日志（冻结于 2026-08-30）

## 1. 任务、产物与冻结规则

- **目标页：** `docs/tasks/video-to-video.md`
- **检索与核验日期：** 2026-08-30（Asia/Shanghai）
- **技术事实冻结：** 只使用冻结日可访问的一手来源；项目页写“accepted”与正式 proceedings 页面分开记录。
- **允许来源：** 正式会议 / 期刊页面与论文、arXiv 原始论文、作者 / 机构项目页、官方仓库 / 模型卡。
- **未使用为技术依据：** 博客转载、媒体报道、聚合榜单、搜索摘要、二手综述和厂商营销摘要。
- **本次复现层级：** R0 文档与 release-surface 审计；未下载全部模型、未执行 GPU 推理、未复跑作者速度。

研究问题：

1. 严格 V2V 的源视频角色、输入 / 输出、编辑区与保留区合同是什么？
2. 如何避免与 I2V、视频修复、video translation、novel-view generation、prediction 混淆？
3. propagation / warp、GAN、atlas、test-time injection、inversion、native DiT / flow、3D / 4D、memory / streaming 各依赖什么守恒假设？
4. 2025–2026 的能力转折、正式 / 预印本状态和实际 release surface 是什么？
5. 如何同时评价 preservation、edit success、temporal、identity、geometry、physics、latency 与 undo？

## 2. 邻接任务先冻结，避免重复写作

开始检索前逐页阅读：

- `docs/tasks/image-to-video.md`：I2V 的图像是输出时间轴中的已知锚点；源视频不是被编辑对象。
- `docs/tasks/video-inpainting.md`：known pixels 是证据，目标是恢复 / 补全未知区；instruction-based counterfactual editing 只在任务目标改变时与其相交。
- `docs/taxonomy.md`：V2V 是“源视频 + 指令 / 参考 / 轨迹 → 改变指定内容并保留未编辑区域”。
- 原 `docs/tasks/video-to-video.md`：保留历史脉络，但重写为严格合同、主题机制、冻结状态、评测与复现矩阵。

边界决定：

| 条件 | 归类 |
|---|---|
| 完整源 RGB 视频定义既有时间轴，输出是其反事实 | 严格 V2V |
| 单图 / 首帧是时间锚点 | I2V |
| 视频是历史前缀，输出未知未来 | prediction / continuation |
| 视频仅作为 pose / motion 驱动 | animation / conditional synthesis |
| 目标是恢复同一语义的缺失 / 退化像素 | inpainting / restoration |
| 同一动态场景换相机 / 视角 | novel-view V2V 交叉任务 |
| 仅文本 + 相机轨迹生成新场景 | camera-conditioned generation，非 V2V |

## 3. 检索式、数据库结果与局限

### 3.1 arXiv API 精确题名 / 摘要检索

使用 `export.arxiv.org/api/query`，冻结日返回：

| 查询 | 命中数 |
|---|---:|
| `ti:"video editing"` | 246 |
| `ti:"video-to-video"` | 67 |
| `abs:"instruction-based video editing"` | 24 |
| `abs:"streaming video editing"` | 2 |
| `abs:"video motion editing"` | 5 |

补充精确题名 / 关键词检索：

```text
ti:"video editing" AND (all:diffusion OR all:flow OR all:DiT)
all:"multi-turn video editing"
all:"real-time streaming video editing"
all:"video motion editing" AND (all:track OR all:trajectory)
all:"camera-controlled generative rendering" OR all:"intrinsic properties"
all:"video editing benchmark"
```

命中数只用于说明候选池规模，不是 PRISMA 流程，也不声称穷尽所有工作。arXiv 默认版本可能在冻结日后更新，因此正文中的时间线使用 v1 日期，release surface 使用冻结日快照。

### 3.2 正式会议 / 期刊站点回查

候选论文按标题回查 NeurIPS proceedings、CVF Open Access（ICCV / CVPR）、AAAI / ICLR 官方状态。正式年份只在有 proceedings / 会议官方记录时写入；仅 arXiv 或作者页的工作继续标为预印本。重点回查：

```text
site:openaccess.thecvf.com video editing ICCV 2025 VACE FiVE MotionFollower
site:openaccess.thecvf.com video editing CVPR 2026 EasyV2V MotionV2V EgoEdit
site:openreview.net video editing ICLR 2026 EditVerse UNIC IVEBench
```

### 3.3 官方项目、仓库与机构页面回查

对入选工作逐项核对作者项目页、机构研究页、GitHub 默认分支、README、LICENSE、release / checkpoint / dataset 链接。GitHub 状态通过官方仓库 API / 页面读取，不用 star 数判断成熟度。

OpenAlex 标题检索在本轮连续返回 HTTP 429；因此未把它的结果数或元数据写入正文，也没有用二手索引补写缺失事实。

## 4. 纳入、排除与证据等级

### 4.1 纳入条件

- 源视频在推理时参与并定义待修改时间轴；或该工作对 V2V 的历史机制、评测、数据 / release surface 有直接作用；
- 能定位一手论文或官方资源；
- 至少贡献一个可核验的任务接口、机制、数据、评测、实时 / 多轮协议或公开资产事实；
- 2025–2026 工作必须额外核验正式 / 预印本状态，避免把未来 proceedings 或作者“accepted”声明写成已出版。

### 4.2 排除条件

- 仅有 T2V / I2V 生成，没有源视频编辑合同；
- 只用 driving video 控制另一个角色，而不编辑 driving video 本身；
- 只有视频预测、插帧、超分或补全，且没有反事实编辑目标；
- 只有搜索摘要、复刻仓库、第三方 demo 或无法定位的一手材料；
- 论文标题相关但没有可分离的 preservation / edit-success 证据；
- release surface 由 issue、fork 或非官方镜像声称，官方 README 未确认。

### 4.3 证据等级

| 等级 | 定义 | 本页用途 |
|---|---|---|
| E1 | 正式 proceedings / 期刊原文 | 正式发表状态、方法与实验事实 |
| E2 | arXiv 原始论文及版本元数据 | 预印本状态、v1 时间、正式页暂缺时的方法事实 |
| E3 | 作者 / 机构项目页、官方仓库、官方模型卡 | code / weights / data / demo / license / accepted 声明和冻结快照 |
| E4 | 搜索引擎、聚合索引、二手页面 | 仅发现候选；不得单独支撑正文技术结论 |

冲突规则：正式论文状态优先 E1；代码和权重状态优先冻结日 E3；论文作者速度是论文设置下的作者报告，不升级为本地复现事实。

## 5. 版本时间线核验

以下 arXiv v1 日期从官方 API 读取：

| 工作 | arXiv ID | v1 日期 |
|---|---|---|
| vid2vid | 1808.06601 | 2018-08-20 |
| Layered Neural Atlases | 2109.11418 | 2021-09-23 |
| Dreamix | 2302.01329 | 2023-02-02 |
| TokenFlow | 2307.10373 | 2023-07-19 |
| AnyV2V | 2403.14468 | 2024-03-21 |
| Movie Gen | 2410.13720 | 2024-10-17 |
| StableV2V | 2411.11045 | 2024-11-17 |
| VACE | 2503.07598 | 2025-03-10 |
| V2Edit | 2503.10634 | 2025-03-13 |
| FlowV2V | 2506.07713 | 2025-06-09 |
| Ditto | 2510.15742 | 2025-10-17 |
| MotionV2V | 2511.20640 | 2025-11-25 |
| EgoEdit | 2512.06065 | 2025-12-05 |
| EasyV2V | 2512.16920 | 2025-12-18 |
| Memory-V2V | 2601.16296 | 2026-01-22 |
| LiveEdit | 2606.26740 | 2026-06-25 |
| JoyAI-Video-Edit | 2608.03974 | 2026-08-04 |
| EditStream | 2608.21424 | 2026-08-16 |

Memory-V2V 在冻结日已到 v3（2026-08-26 更新）。正文仍用 v1 表示首次公开、用 v3 / 仓库表示冻结状态，避免把最新版本日期误作首发日期。

## 6. 逐项来源账本（与正文参考文献编号一致）

### 6.1 历史、传播与测试时编辑

| Ref | 等级 | 一手来源 | 正文使用的事实 |
|---:|---|---|---|
| 1 | E1 | [Video Rewrite DOI](https://doi.org/10.1145/258734.258880) | 1997 局部口型 / 音频驱动历史起点 |
| 2 | E1 | [Space-Time Video Completion PDF](https://graphics.stanford.edu/courses/cs448a-06-winter/wexler-completion-cvpr04.pdf) | 时空补全与局部证据传播历史 |
| 3 | E1 | [NeurIPS vid2vid](https://proceedings.neurips.cc/paper/2018/hash/d86ea612dec96096c5e0fcc8dd42ab6d-Abstract.html) | paired conditional translation、时间约束 |
| 4 | E2 / E1 | [Layered Neural Atlases](https://arxiv.org/abs/2109.11418) | 前景 / 背景 atlas 与可合成编辑 |
| 5 | E3 / E1 | [Text2LIVE project](https://text2live.github.io/) | 文本驱动 edit layer 与原视频合成 |
| 6 | E2 | [Dreamix](https://arxiv.org/abs/2302.01329) | diffusion video editor、混合微调和多接口 |
| 7 | E1 | [FateZero at ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html) | inversion 轨迹与 attention fusion |
| 8 | E1 | [Pix2Video at ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Ceylan_Pix2Video_Video_Editing_using_Image_Diffusion_ICCV_2023_paper.html) | 图像扩散特征上的视频编辑 |
| 9 | E2 / E1 | [TokenFlow](https://arxiv.org/abs/2307.10373) | diffusion feature correspondence 与 token propagation |
| 10 | E2 / E3 | [AnyV2V paper](https://arxiv.org/abs/2403.14468) | 图像编辑与视频传播模块化 |
| 11 | E2 / E3 | [StableV2V](https://arxiv.org/abs/2411.11045) | shape consistency 与 DAVIS-Edit 资源 |
| 12 | E2 | [FlowV2V](https://arxiv.org/abs/2506.07713) | flow-driven I2V 重写一致编辑 |
| 13 | E1 | [FFP-300K at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html) | 300K、720p、81-frame 配对和首帧传播 |

### 6.2 原生编辑、控制、数据与 benchmark

| Ref | 等级 | 一手来源 | 正文使用的事实 |
|---:|---|---|---|
| 14 | E2 | [Movie Gen](https://arxiv.org/abs/2410.13720) | 媒体基础模型中的视频编辑接口 |
| 15 | E1 / E3 | [VACE at ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.html) | 统一生成 / 参考 / 编辑条件 |
| 16 | E2 / E1 | [EditVerse](https://arxiv.org/abs/2509.20360) | 232K 编辑样本、统一 token 序列 / in-context 学习 |
| 17 | E2 / E1 | [UNIC](https://arxiv.org/abs/2506.04216) | source、noisy target、多模态条件共同 token 化 |
| 18 | E1 / E3 | [EasyV2V at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_EasyV2V_A_High-quality_Instruction-based_Video_Editing_Framework_CVPR_2026_paper.html)；[project](https://snap-research.github.io/easyv2v/) | sequence concat、LoRA、时空 mask、可选参考 |
| 19 | E1 | [Ditto at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Scaling_Instruction-Based_Video_Editing_with_a_High-Quality_Synthetic_Dataset_CVPR_2026_paper.html) | Ditto-1M 与大规模原生指令编辑训练 |
| 20 | E1 | [VIVA at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Cong_VIVA_VLM-Guided_Instruction-Based_Video_Editing_with_Reward_Optimization_CVPR_2026_paper.html) | VLM instructor 与 reward optimization |
| 21 | E1 | [CoT-Edit at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_CoT-Edit_Let_CoT_Guide_Instruction_Video_Editing_CVPR_2026_paper.html) | plan → box → mask → editor 的显式规划 |
| 22 | E1 | [MotionFollower at ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Tu_MotionFollower_Editing_Video_Motion_via_Score-Guided_Diffusion_ICCV_2025_paper.html) | pose / appearance controllers、score guidance 与一致性正则 |
| 23 | E1 | [MotionV2V at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Burgert_MotionV2V_Editing_Motion_in_a_Video_CVPR_2026_paper.html) | motion counterfactual 与稀疏轨迹 |
| 24 | E1 | [3D Point Tracks at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Generative_Video_Motion_Editing_with_3D_Point_Tracks_CVPR_2026_paper.html) | 源 / 目标三维点轨迹、深度和遮挡 |
| 25 | E1 | [TrajectoryCrafter at ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_TrajectoryCrafter_Redirecting_Camera_Trajectory_for_Monocular_Videos_via_Diffusion_Models_ICCV_2025_paper.html) | 单目视频相机轨迹重定向 |
| 26 | E1 | [ReCamMaster at ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Bai_ReCamMaster_Camera-Controlled_Generative_Rendering_from_A_Single_Video_ICCV_2025_paper.html) | 单视频相机可控生成式渲染 |
| 27 | E1 | [V-RGBX at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_V-RGBX_Video_Editing_with_Accurate_Controls_over_Intrinsic_Properties_CVPR_2026_paper.html) | albedo / normal / material / irradiance 内禀控制 |
| 28 | E2 | [V2Edit](https://arxiv.org/abs/2503.10634) | 视频与三维场景的通用编辑 |
| 29 | E2 / E3 | [Memory-V2V](https://arxiv.org/abs/2601.16296)；[Adobe Research record](https://research.adobe.com/publication/memory-v2v-memory-augmented-video-to-video-diffusion-for-consistent-multi-turn-editing/) | 外部 memory、retrieval、dynamic tokenization / compression；ECCV 标注与冻结日后的 publication date |
| 30 | E1 / E3 | [EgoEdit at CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Li_EgoEdit_Dataset_Real-Time_Streaming_Model_and_Benchmark_for_Egocentric_Video_CVPR_2026_paper.html) | 第一视角数据、streaming model 与 benchmark |
| 31 | E2 / E3 | [LiveEdit](https://arxiv.org/abs/2606.26740) | 双向到因果蒸馏、自回归 mask cache、作者速度 |
| 32 | E2 / E3 | [JoyAI-Video-Edit](https://arxiv.org/abs/2608.03974) | 16B MMDiT、causal VAE、bounded KV cache、作者速度 |
| 33 | E2 / E3 | [EditStream](https://arxiv.org/abs/2608.21424) | 四步自回归交互编辑；实时声明与论文限制 |
| 34 | E2 / E1 | [VE-Bench](https://arxiv.org/abs/2408.11481) | 8 模型、24 标注者、28,080 主观分数 |
| 35 | E1 | [FiVE-Bench at ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html) | 74 真实 + 26 生成源、6 类编辑、420 prompt / mask |
| 36 | E2 / E1 | [IVEBench](https://arxiv.org/abs/2510.11647) | 600 源视频、32–1024 帧、8 主类 / 35 子类 |

### 6.3 官方 release surface

| Ref | 等级 | 官方来源 | 冻结日核验 |
|---:|---|---|---|
| 37 | E3 | [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | SHA `48eb44f1c4be87cc65a98bff985a26976841e9f3`；Apache-2.0；inference / preprocess / Gradio、weights、benchmark；未见完整预训练管线 |
| 38 | E3 | [TIGER-AI-Lab/AnyV2V](https://github.com/TIGER-AI-Lab/AnyV2V) | SHA `bc540befacafddb9689ee86a396e7738bfed0e4f`；MIT；模块化代码，依赖外部组件 |
| 39 | E3 | [AlonzoLeeeooo/StableV2V](https://github.com/AlonzoLeeeooo/StableV2V) | SHA `68aff43cd01aed58159f57fee626e5f1bb72aef3`；MIT；code / weights / DAVIS-Edit |
| 40 | E3 | [EzioBy/Ditto](https://github.com/EzioBy/Ditto) | SHA `4b1c86eee134d1f1e8c3db35a4eef750cec13f16`；CC BY-NC-SA；inference / train / data / weights |
| 41 | E3 | [DoHunLee1/Memory-V2V](https://github.com/DoHunLee1/Memory-V2V) | SHA `a9cb5430ba8d05669a5cbafc8967cff2858d5d0a`；README 写 Code coming soon |
| 42 | E3 | [snap-research/EgoEdit](https://github.com/snap-research/EgoEdit) | SHA `8e90b19e6dfe46fa06f2c4915ebdadb5d3de5d72`；dataset / benchmark；非商业条款；未核验到论文模型完整 checkpoint |
| 43 | E3 | [jd-opensource/JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) | SHA `114a1f605acacbf494b46c383fab49df6240268c`；Apache-2.0；deploy code / weights；full training / data 为 TODO |
| 44 | E3 | [RyannDaGreat/MotionV2V](https://github.com/RyannDaGreat/MotionV2V) | SHA `f2efa3d3ce641f34ae9b177c2df84d39e688e82c`；项目资产；未见 code / weights |
| 45 | E2 / E3 | [EditCtrl paper](https://arxiv.org/abs/2602.15031)；[public repo](https://github.com/yehonathanlitman/EditCtrl) | SHA `e0a31e6abcc8684dd319772eabf916b5d9dc616a`；Apache-2.0；README 说明是公开重实现，并非内部精确 checkpoint |
| 46 | E3 | [cp-cp/LiveEdit](https://github.com/cp-cp/LiveEdit) | 官方项目所链仓库；inference / train / checkpoint；README 标注 ECCV 2026 accepted |
| 47 | E3 | [EditStream project](https://real-time-video-research.github.io/editstream/) | 冻结日页面写 Code soon、Data & Model soon |

## 7. 关键机制事实复核

为避免只根据摘要写综述，下载并解析 EasyV2V、Ditto、MotionV2V、Generative Video Motion Editing with 3D Point Tracks、EditCtrl、VIVA、CoT-Edit 与 V-RGBX 的一手 PDF；抽查方法、实验和限制段。未把 PDF 下载目录加入仓库。

| 工作 | PDF / 官方材料中核到的机制 | 正文中的限定 |
|---|---|---|
| EasyV2V | sequence concatenation 优于 channel concatenation；LoRA；统一 spatial-temporal mask；可选 reference | 不把 mask condition 等同于解码后像素硬守恒 |
| Ditto | 高质量合成编辑数据；Ditto-1M；论文报告大规模 GPU 预算 | 训练成本是作者报告，未复跑 |
| MotionV2V | motion counterfactual 与 sparse tracks；强调通用对象运动 | 轨迹遵循不能证明接触 / 碰撞物理正确 |
| 3D Point Tracks | 完整源视频 + source / target 3D point tracks；显式深度 / 遮挡 | 深度估计误差须与编辑器误差分离 |
| EditCtrl | masked tokens 主计算 + low-resolution global context；算力随编辑 mask 缩放 | 公共仓库是重实现；不冒充内部 checkpoint |
| VIVA | VLM instructor 与 Edit-GRPO / reward optimization | VLM reward 不能替代局部性和几何实测 |
| CoT-Edit | 计划 → boxes → mask → editor | planner 与 renderer 分开验收 |
| V-RGBX | inverse rendering 到 albedo / normal / material / irradiance | “intrinsic-aware”不自动证明完整 4D scene representation |

流式材料补查：

- LiveEdit 官方论文 / 仓库：三阶段 bidirectional-to-causal distillation、自回归 mask cache；论文作者报告 12.66 FPS。
- JoyAI 官方论文 / README：16B MMDiT、causal VAE、bounded KV；README 曾报告 720×1248 约 30 FPS，并在 2026-08-24 增补 RTX 5090、840×480、24 FPS 设置。正文不固定宣传数字，只要求逐设置报告。
- EditStream 论文：四步、作者报告 16 FPS / 720p / 单 GPU；限制段同时说明未使用专门 long-memory，且高分辨率实时仍有优化空间。

## 8. 正式状态与冻结日后的时间陷阱

### 8.1 已核对正式页面

- ICCV 2025：VACE、MotionFollower、TrajectoryCrafter、ReCamMaster、FiVE-Bench。
- AAAI 2025：VE-Bench。
- ICLR 2026：EditVerse、UNIC、IVEBench。
- CVPR 2026：EasyV2V、Ditto、MotionV2V、3D Point Tracks、EditCtrl、VIVA、CoT-Edit、V-RGBX、EgoEdit、FFP-300K。

### 8.2 accepted 与 published 分离

- Memory-V2V 的 Adobe Research 页面标注 ECCV 2026，但其机构页面 publication date 为 2026-09-12，晚于冻结日；正文只写冻结日可核的机构 / accepted 状态，不声称当时已有正式 proceedings 页面。
- LiveEdit 官方仓库标注 accepted to ECCV 2026；冻结日以 arXiv + 官方仓库为证据，不将其升级为已出版 proceedings。
- JoyAI 和 EditStream 在冻结日是 2026-08 arXiv 预印本，不能因系统已放 demo 就写成正式发表。

## 9. 负面核验清单

| 主张 | 检查位置 | 冻结日结论 |
|---|---|---|
| VACE 是否开放完整预训练代码 | 官方 README、目录、模型 / benchmark 链接 | 有推理 / preprocess / demo / weights；未见完整 pretraining pipeline |
| Memory-V2V 是否开源 | 官方 README | `Code coming soon` |
| MotionV2V 是否开源模型 | 官方默认分支 | 有项目资产；未见 inference code / checkpoint |
| EgoEdit 是否有论文模型完整权重 | 官方 README、release / issue 指向 | 核到 dataset / benchmark；未核到完整对应 checkpoint |
| JoyAI 是否有完整训练与数据 | 官方 README checklist | deploy / weights 已有；full training / data 仍 TODO |
| EditCtrl repo 是否等于论文内部实现 | 官方 README | 明示 public reimplementation，不是 exact internal checkpoint |
| EditStream 是否已放 code / data / model | 官方项目按钮 / 文案 | Code soon；Data & Model soon |
| “streaming”是否都严格因果 | 论文输入可见性与 cache | 各系统 look-ahead / cache 不同，必须实验分层 |
| “real-time”是否可横向比较 | 论文 / README 的硬件、分辨率、模块计时 | 不可直接横比；本页未本地复测 |
| OpenAlex 能否提供第二索引计数 | 标题 API 查询 | HTTP 429，未使用其计数 |

负面核验措辞统一为“冻结日在官方核验位置未见”，不推断作者私有资产，也不把 GitHub issue 中的承诺当正式发布。

## 10. 评测协议的来源到设计映射

| 设计项 | 来源动机 | 本页扩展 |
|---|---|---|
| edit success 与 source preservation 分开 | VE-Bench / FiVE / IVEBench 均显示单一质量分不足 | 加入 mask 外误差、轨迹 / reference task metric |
| 真实与生成源分层 | FiVE 同时含 real / generated sources | 不把生成源上的一致性外推到真实视频 |
| 长度分层 | IVEBench 覆盖 32–1024 帧 | 画出随帧数漂移，而非只给短 clip 平均 |
| 运动 / 相机显式控制 | MotionV2V、3D tracks、ReCamMaster | 加 geometry、occlusion、camera 与 physics 轴 |
| mask 计算与局部性 | EasyV2V、EditCtrl | 分别测 token 不更新、解码后像素泄漏和边界环 |
| 多轮 memory | Memory-V2V | 加顺序交换、撤销循环、状态压缩曲线 |
| 因果 streaming | EgoEdit、LiveEdit、JoyAI、EditStream | 统一 look-ahead，报告 capture-to-display 全延迟 |

benchmark 数字只作为数据集设计事实，不把作者指标榜单抄入正文，避免模型版本和评测脚本漂移。

## 11. 图示设计与可访问性

正文保留两个原生 Mermaid，并加入一张生成式教学 PNG：

1. **任务边界图：** source video → 是否编辑既有时间轴 → restoration / local / global / motion / camera 分流；
2. **方法选择器：** 硬 mask、训练 / 测试时条件、编辑幅度、2D / 3D 控制、多轮 / 流式依次分流。

两幅 Mermaid 都有 `accTitle` 与 `accDescr`，并在图后给顺序化文字替代。生成 PNG 用于快速比较四类输出关系，Mermaid 继续承担边界判定和机制路线的可编辑精确逻辑。

生成式教学图记录：

- **文件：** `assets/diagrams/video-to-video-method-selector.png`
- **工具与日期：** built-in OpenAI image generation，2026-08-30
- **尺寸：** 1672 × 941 RGB PNG
- **SHA-256：** `87cafa676cb2761d1b0ce3795fcc97bdccf916303b43b46f39dfa5b5bd464fcf`
- **灰度统计：** mean 0.864148，standard deviation 0.275993，min 0，max 1

Prompt 要求从 source video 与可选 instruction、mask/track、target domain、reference、audio/pose 出发，经“what may change?”分成 restore/complete、translate appearance、edit semantics 与 retime/restructure；每路显示 before/after 和 preservation tag；右侧独立检查 edit success、source fidelity、locality、temporal consistency、identity/motion preservation，并明确 future-only prediction 与 first-frame animation 是不同合同。禁止模型名、benchmark 数字和架构 SOTA 排名。

原尺寸视觉回读确认：源视频四帧时间顺序清楚；四条 before/after 路线和 preservation 卡片没有交叉；右侧五个 gauge 标签可读；底部规则完整。正文图注特别说明 restoration 是邻接分支，不把它误计为严格语义 V2V；PNG 另配 alt、图注和六步顺序化文字替代。

## 12. 验证记录

命令只作用于两份目标 Markdown 或临时目录；主线整合阶段另新增并回读上述教学 PNG。

- **Markdown lint：** `markdownlint-cli2 v0.23.2`（markdownlint v0.41.1）检查两文件，0 issues。
- **引用闭合：** 正文 119 次引用、47 个唯一编号；47 个 anchor 连续为 1–47；0 缺失、0 孤儿、0 重复、0 非标准引用标签。
- **本地链接：** 主线加入研究记录和教学图链接后，正文 7 个相对链接均存在；研究日志无相对链接。
- **外部链接：** 两文件去重后 50 个 URL；49 个由 GET 返回 HTTP 2xx。`doi.org/10.1145/258734.258880` 正常重定向到 ACM，但自动客户端收到 HTTP 403；保留 DOI 作为正式、持久标识，并明确这是访问限制而非 404。LiveEdit arXiv 在首轮 curl 超时，随后用独立 GET 返回 200。
- **Mermaid 实际渲染：** 提取 2 个代码块到 `/tmp`；首次因 Mermaid CLI 默认的 `chrome-headless-shell` 未安装而失败，随后显式指定本机 Google Chrome，2/2 成功渲染为临时 SVG；两个块均含 `accTitle`、`accDescr`，临时渲染产物未写入仓库。
- **主线 PNG：** 1672 × 941 RGB、hash 与灰度统计均已记录；original-detail 回读通过，正文含 alt、图注与文字替代。
- **diff 检查：** 全工作区 `git diff --check` 与目标路径 `git diff --check -- docs/tasks/video-to-video.md sources/research_20260830_video_to_video.md` 均无输出。
- **变更范围：** 审稿子任务只修改正文并新增本研究日志；主线随后增加上述 PNG、正文引用与图示记录。未改其他章节，未 commit 或 push。

## 13. 已知局限

- 没有运行任何编辑模型，不能把论文图、作者速度或仓库存在视为推理复现成功。
- 没有下载 / 校验所有 checkpoint 与训练数据哈希；release surface 是页面级 R0 快照。
- arXiv 题名 / 摘要检索会漏掉标题不用 editing / V2V 的相关工作；用正式会议页与引用链补充，但不声称穷尽。
- OpenAlex 429 使跨索引去重受限；没有用不稳定的搜索结果总数替代。
- 2026 工作的仓库和 proceedings 状态可能在冻结日后改变；页面显式标注冻结日与 SHA，后续更新应重新核验。
- benchmark 的 human score、VLM score 和作者速度都受版本、硬件与协议影响；正文只提取设计事实，不建立跨论文排行榜。
