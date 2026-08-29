# 全仓库引用核验与批量修正日志

执行日期：**2026-08-29**

## 范围与方法

- 覆盖 README、`docs/`（含 `docs/tasks/`、`docs/generative-models/`）、`resources/` 共 30 个 Markdown 文件的 **440 条参考文献**，去重后为 **230 个唯一来源**。
- 每个来源回一手页面核验：arXiv abs、CVF openaccess、ACL Anthology、PMLR、NeurIPS proceedings、IJCAI proceedings、ACM DL / Crossref、Nature、OpenReview、ITU、厂商官方发布页。
- 每一批核验都跑了负对照（不存在的 arXiv ID / GitHub 仓库），确认工具会返回 404 而不是编造条目。
- arXiv 页面存在偶发 "PDF is empty" 假阴性，凡疑似"不存在"的条目均用 alphaXiv、HuggingFace Papers、dblp 或出版方页面二次复核后才下结论。

## 统一后的引用格式

```
<a id="ref-N"></a>[N] 规范标题（URL）. 作者1, 作者2, ..., 作者6, et al. Venue. 年份.
```

- **作者**：完整姓名，最多列前 6 位，超出以 `et al.` 收尾。此前存在的三种混用格式（`Surname, X., et al. [Title]`、`[Title]. Author et al. Year.`、无作者）已全部归一。
- **Venue**：有正式发表版本的一律标注该 venue（CVPR / ICCV / ECCV / ICLR / ICML / NeurIPS / SIGGRAPH / AAAI / IJCAI / EMNLP / ACL / TMLR / Nature / ACM TOG / IEEE TIP…），确无正式收录的写 `arXiv preprint`，不臆造。
- **年份**：跟随 venue 年份（例如 MoCoGAN 由 `arXiv preprint. 2017.` 改为 `CVPR. 2018.`）。`docs/timeline.md` 的节点年份仍沿用"首次公开"规则，两者定义不同、互不冲突。

## 修正总量

| 项目 | 数量 |
|---|---:|
| 重写的参考文献行 | 371 |
| 补齐/修正作者列表 | 约 260 条 |
| 补上此前缺失的正式 venue | 约 90 条 |
| 修正的错误 arXiv ID | 2 |
| 修正的失效 URL | 4 |
| 修正的错误标题 | 12 |
| 修正的错配正文引用标记 | 5 |
| 补上的孤儿引用标记 | 14 |
| 修正后编号/锚点/内部死链问题 | 0 |

## 一、严重错误（错误 ID、错误论文、错误作者）

### `docs/tasks/digital-human.md`（17 条中 14 条有问题，已全部重建）

| ref | 原问题 | 已改为 |
|---|---|---|
| [1] | 作者伪造 | MakeItTalk：Yang Zhou, Xintong Han, Eli Shechtman, Jose Echevarria, Evangelos Kalogerakis, Dingzeyu Li. ACM TOG (SIGGRAPH Asia). 2020 |
| [2] | 漏第 4 作者 | 补 C. V. Jawahar；标题 In The Wild 大小写按 ACM 版 |
| [3] | **arXiv 2011.03727 指向一篇量子光力学论文**，标题 "Audio-Driven Talking Face Video Generation with Dynamic-NeRF" 不存在 | 改为 AD-NeRF，arXiv **2103.11078**，Yudong Guo 等，ICCV 2021 |
| [4] | 作者伪造（本仓库作者本人的论文） | SadTalker：Wenxuan Zhang, **Xiaodong Cun**, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, et al. CVPR. 2023 |
| [5][6] | 作者截断未标 et al. | 补全并标注 venue（VASA-1 → NeurIPS 2024） |
| [7] | 缺 venue、作者不全 | OmniHuman-1：补 Yuan Zhang 等，ICCV 2025 |
| [8] | **张冠李戴**，写的是 X2Face 的作者 | Deep Video Portraits：Hyeongwoo Kim, Pablo Garrido, Ayush Tewari, Weipeng Xu, Justus Thies, Matthias Nießner, et al. ACM TOG (SIGGRAPH). 2018 |
| [9] | **arXiv 1907.07837 指向一篇符号图论数学论文**；作者伪造；年份错 | 改为 arXiv **1807.07860**，Hang Zhou, Yu Liu, Ziwei Liu, Ping Luo, Xiaogang Wang. AAAI. **2019** |
| [10][12] | 两篇不同论文被安上同一串伪造作者 | [10] Ran Yi 等；[12] Audio2Head：Suzhen Wang, Lincheng Li, Yu Ding, Changjie Fan, Xin Yu. IJCAI. 2021 |
| [11] | CVF URL 前缀 `Zhang_` 为 404；作者伪造 | 改为 `Zhou_...`；Hang Zhou, Yasheng Sun, Wayne Wu, Chen Change Loy, Xiaogang Wang, Ziwei Liu. CVPR. 2021 |
| [13] | 作者伪造（本仓库作者本人的论文） | VideoReTalking：Kun Cheng, **Xiaodong Cun**, Yong Zhang, Menghan Xia, Fei Yin, Mingrui Zhu, et al. SIGGRAPH Asia. 2022 |
| [14] | 作者伪造 | StyleTalker：Dongchan Min, Minyoung Song, Eunji Ko, Sung Ju Hwang |
| [15] | CVF URL 前缀 `Ren_` 为 404；作者伪造 | 改为 `Shen_...`；Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, et al. CVPR. 2023 |
| [20] | VGG 页面疑似下线 | 改指 arXiv 1809.00496（LRS3-TED），Triantafyllos Afouras 等 |
| [22] | 只写"数据集仓库" | 补论文：Flow-Guided One-Shot Talking Face Generation with a High-Resolution Audio-Visual Dataset (HDTF)，Zhimeng Zhang 等，CVPR 2021 |

### 其他文件的错配引用

| 文件 | 原问题 | 处理 |
|---|---|---|
| `resources/datasets.md` | UCF-101 挂 `[[4]]`（实为 MoCoGAN）、nuScenes 挂 `[[5]]`（实为 GAIA-1）、二级标题挂 `[[6]]`（实为 Genie） | 新增 UCF101（Soomro 等 2012）与 nuScenes（Caesar 等 CVPR 2020）两条正确引用；删除标题上的引用标记与文末无意义的 DreamerV3 兜底句 |
| `docs/applications.md` | "视频编辑 `[[5]]`" 指向 AnimateDiff（并非视频编辑工作）；文末兜底句把 Imagen Video 硬塞进来 | 标题去掉引用标记，把 AnimateDiff 移到"局部风格化/动作迁移"处；Imagen Video 并入 §1 文本到视频；删除兜底句 |
| `docs/generative-models/flow-consistency-models.md` | 5 条参考文献正文一条都没引用 | 在 §2–§5、§7 补上对应引用标记 |
| `adversarial-/autoregressive-/masked-/recurrent-/variational-generation.md` | 共 11 条孤儿引用 | 在相应正文位置补上引用标记 |

## 二、venue / 年份 / 标题订正（部分要点）

- **PhyGenBench（2410.05363）**：`ICLR 2025` → **ICML 2025**（PMLR v267）。
- **VideoPhy（2406.03520）** → ICLR 2025；**VideoPhy-2（2503.06800）** → **ICLR 2026**。
- **GenAI-Bench（2406.13743）**：第一作者不是 Zhiqiu Lin，而是 **Baiqi Li**；venue 为 CVPR 2024 SynData4CV Workshop。
- **Beyond FVD（2410.05203）**：第一作者全名 **Ge Ya Luo**；ICLR 2025 发表版标题为 "Beyond FVD: **An** Enhanced Evaluation Metrics for Video Generation **Distribution** Quality"。
- **TC-Bench（2406.08656）**：ACL 发表版标题为 "…Temporal Compositionality in **Conditional Video Generation**"，venue 为 Findings of ACL 2025。
- **WorldModelBench** → NeurIPS 2025 D&B；**VideoScore** → EMNLP 2024；**Content Bias in FVD** → CVPR 2024。
- **决策中心评测（2606.15032）**：标题补回被漏掉的 "**for Embodied Decision-Making**"。
- **Thinking in Frames（2601.21037）**：原副标题 "How Video Generation Models Exhibit, Scale, and Fail at Visual Reasoning" 是错的，真实副标题为 "**How Visual Context and Test-Time Scaling Empower Video Reasoning**"（语义方向相反）。
- **World Models（1803.10122）**：原写 "Advances in Neural Information Processing Systems 31"，但 NeurIPS 31 收录版本题名是 "Recurrent World Models Facilitate Policy Evolution"。已改为 arXiv preprint 并在 BibTeX 中注明改题。
- **Video Pixel Networks**、**FVD**、**DVD-GAN**、**VideoGPT**、**Imagen Video**、**Movie Gen**、**HunyuanVideo**、**SVD**、**Cosmos**、**Wan**、**V-JEPA**、**MC-JEPA**、**GAIA-1** 等经核验确无正式会议收录，统一标 `arXiv preprint`，未臆造 venue。
- **VideoPainter** → SIGGRAPH 2025；**VACE** → ICCV 2025；**AnyV2V** → TMLR 2024；**VE-Bench** → AAAI 2025；**IVEBench** → ICLR 2026；**Ditto/Scaling Instruction-Based Video Editing** → CVPR 2026；**TD-MPC2** → ICLR 2024；**UniSim** → ICLR 2024；**TD-JEPA** → ICLR 2026；**Video Generation Models are General-Purpose Vision Learners** → ECCV 2026；**Articulated Object Reconstruction** → ECCV 2026。
- **Memory-V2V（2601.16296）**：原用 v1 题名，已更新为当前版本 "Memory-Augmented Video-to-Video Diffusion for Consistent Multi-Turn Editing"。
- **FramePack（2504.12626）**：现题确为 "Frame Context Packing and Drift Prevention…"（v3 改题），标题后加注 (FramePack) 便于辨认。
- **Horn–Schunck DOI**：原始链接 `…0004-3702(81)90024-2` 中的圆括号会截断部分 Markdown 解析，统一改为百分号编码形式。
- **Lucas–Kanade PDF**：改为 `publications.ri.cmu.edu` 的实际跳转地址。

## 三、bibliography 数据源

- `bibliography/registry.json` 按 section A→H 重新排序，修复 `docs/bibliography.md` 中 `…F G H G G H G G G H H…` 的乱序（此前是渲染顺序直接跟随 registry 写入顺序）。
- `oh2015actionconditional` 标题补回 "in Atari Games"；`lucas1981iterative` 更新 URL；`ha2018worldmodels` 改为 `@misc` 并注明 NeurIPS 版改题。
- `docs/bibliography.md` 表头措辞修正：原文"覆盖仓库正文登记的 N 篇"会让读者以为这就是全部文献，实际 registry 只收 58 篇核心条目、正文引用的唯一来源有 230 个。现已说明二者关系。
- 已用 `python scripts/update_bibliography.py --offline` 重新生成 `references.bib` 与 `docs/bibliography.md`。

## 四、刻意未改动的条目

厂商发布页与标准文档保持原样（Genie 2/3、Runway GWM-1、Marble、Cosmos Predict-2、Seedance 2.5、MiniMax H3、OpenAI Sora 报告、ITU-T P.910、C2PA、VMAF）。其中两条建议后续单独处理：

- **ITU-T P.910** 已有 **(07/2026)** 新版，仓库引用的是 (10/2023)；现已在条目中显式标注版本号。
- **C2PA** 规范当前最新为 **2.4**，仓库锁定在 2.2；条目已改为 "Version 2.2" 明示。

## 五、修正后的机器校验结果

- 参考文献行 **440** 条，格式异常 **0**。
- 正文 `[[n]]` 与文末 `<a id="ref-n">` 一一对应：孤儿引用 **0**、缺定义 **0**、编号与锚点不符 **0**、断号 **0**。
- 内部相对链接与图片路径死链 **0**。
- `references.bib` 58 条，花括号配平。

## 六、并发写入提示

修正期间检测到另一个会话正在改写 `docs/video-reasoning.md`、`docs/world-models.md`、`docs/foundation-models.md`、`docs/tasks/video-to-video.md`、`docs/tasks/text-to-video.md`、`docs/timeline.md`。本次修正基于 08:13 的工作区状态。若这些文件之后又新增了参考文献条目，需按本文档第二节的格式规范补齐 venue 与完整作者。
