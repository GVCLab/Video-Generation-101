# 全仓库严格审阅日志

审阅日期：**2026-08-29**（快照时间 06:30–07:00，期间 `docs/video-reasoning.md`、`docs/foundation-models.md` 有并发写入）

审阅范围：README、`docs/` 全部章节、`docs/tasks/`、`docs/generative-models/`、`resources/`、`bibliography/`、`.github/`、`CITATION.cff`、`CONTRIBUTING.md`。

> **状态更新（2026-08-29 08:15）**：P0-1 与 P1-4 ~ P1-12 已全部修复，全仓库 440 条引用已重新核验并统一格式，详见 [引用核验与批量修正日志](citation_fix_20260829.md)。本文件保留原始审阅记录，未勾选的项（P0-2 assets 未入库、P0-3 提交前链接检查、P1-1 Sora 2 表述、P1-3 VPN 年份、P2 全部、P3 全部）仍待处理。

## 审阅方法

- 机器检查：内部相对链接与图片路径、正文 `[[n]]` 与文末 `<a id="ref-n">` 的一一对应、编号断号、锚点错位、孤儿引用、git 跟踪状态。
- 人工 + 联网复核：抽取全部 2026 年 arXiv 条目、全部 CVF/WACV openaccess URL、全部厂商发布声明、`docs/tasks/digital-human.md` 的 17 条引用，逐条回一手来源核对标题、作者、arXiv ID、venue 与数字。
- 每个核查子任务都跑了负对照（用不存在的 arXiv ID / GitHub 仓库），确认工具不会对不存在的条目编造内容。

## 总体判断

先说结论：**时间线、评测、World Model、JEPA、physical-consistency、video-reasoning 六章的事实密度和证据纪律是同类中文资料里罕见的水平**，抽查的 40 余个 2025–2026 arXiv 条目、7 个 CVF/WACV 链接、8 项厂商声明全部为真，`sources/timeline_review_20260829.md` 记录的核查流程也确实执行到位。

问题集中在三处：**（1）2023 年以前的人像/数字人文献是靠记忆写的，作者大面积失真；（2）自动引用注入脚本产生了错配与兜底句；（3）工程层面（git 跟踪、CI、索引一致性）有几个会直接损坏线上呈现的漏洞。**

---

## P0 — 必须先修，否则损害仓库可信度

### ~~P0-1 已修复~~（见 citation_fix_20260829.md）

#### 原始记录：`docs/tasks/digital-human.md` 参考文献大面积作者伪造

17 条引用中只有 3 条完全正确（[7] OmniHuman-1、[16] OmniAvatar、[19] 综述）。逐条核实结果：

| ref | 问题 | 正确信息 |
|---|---|---|
| [1] L195 | 作者伪造 | MakeItTalk 真实作者：Yang Zhou, Xintong Han, Eli Shechtman, Jose Echevarria, Evangelos Kalogerakis, Dingzeyu Li（TOG/SIGGRAPH Asia 2020） |
| [2] L197 | 作者漏第 4 位 | 应补 **C. V. Jawahar** |
| [3] L199 | **arXiv ID 指向无关论文 + 标题不存在** | `2011.03727` 是一篇量子光力学论文。若指 AD-NeRF：`arXiv 2103.11078`，Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, Juyong Zhang，ICCV 2021。另有一篇易混淆的真实论文 *Dynamic Convolution Kernels*（2201.05986, TMM 2022），勿混用 |
| [4] L201 | **作者伪造（本仓库作者自己的论文）** | SadTalker 真实作者：Wenxuan Zhang, **Xiaodong Cun**, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, Fei Wang；venue 应写 **CVPR 2023**, pp. 8652–8661 |
| [5] L203 | 作者截断未标 et al. | 全表 10 人，末位 Yao Yao, Siyu Zhu |
| [6] L205 | 作者截断 + venue 缺失 | 应标 **NeurIPS 2024 (Oral)** |
| [7] L207 | venue 可补 | **ICCV 2025** |
| [8] L209 | **张冠李戴** | 现列的 Wiles/Koepke/Zisserman 是 **X2Face（ECCV 2018）** 的作者。Deep Video Portraits 真实作者：Hyeongwoo Kim, Pablo Garrido, Ayush Tewari, Weipeng Xu, Justus Thies, Matthias Nießner, Patrick Pérez, Christian Richardt, Michael Zollhöfer, Christian Theobalt |
| [9] L211 | **arXiv ID 指向无关论文 + 作者伪造 + 年份错** | `1907.07837` 是一篇符号图论数学论文。DAVS 正确：`arXiv 1807.07860`，Hang Zhou, Yu Liu, Ziwei Liu, Ping Luo, Xiaogang Wang，**AAAI 2019**（非 2020） |
| [10] L213 | 作者伪造 | 真实：Ran Yi, Zipeng Ye, Juyong Zhang, Hujun Bao, Yong-Jin Liu |
| [11] L215 | **CVF URL 404 + 作者伪造** | URL 前缀应为 `Zhou_` 而非 `Zhang_`；真实作者：Hang Zhou, Yasheng Sun, Wayne Wu, Chen Change Loy, Xiaogang Wang, Ziwei Liu |
| [12] L217 | 作者伪造（与 [10] 撞了同一串伪造名） | Audio2Head 真实：Suzhen Wang, Lincheng Li, Yu Ding, Changjie Fan, Xin Yu |
| [13] L219 | **作者伪造（本仓库作者自己的论文）** | VideoReTalking 真实：Kun Cheng, **Xiaodong Cun**, Yong Zhang, Menghan Xia, Fei Yin, Mingrui Zhu, Xuan Wang, Jue Wang, Nannan Wang |
| [14] L221 | 作者伪造 | StyleTalker 真实：Dongchan Min, Minyoung Song, Eunji Ko, Sung Ju Hwang |
| [15] L223 | **CVF URL 404 + 作者伪造** | URL 前缀应为 `Shen_` 而非 `Ren_`；真实作者：Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, Jiwen Lu |

同一串伪造作者名 "Xinya Ji, Hang Zhou, Keke Li" 被同时安到 [10] 和 [12] 两篇不同论文上，是典型的"凭记忆补作者"痕迹。建议整段参考文献推翻重建，全部回 arXiv/CVF/ACM DL 抓取。

### P0-2 `assets/` 91 个文件未纳入 git，但被 84 处引用

`git status` 显示 `assets/` 整目录未跟踪（20 MB / 92 个文件），而 `docs/timeline.md`（75 处）、`docs/getting-started.md`（4 处）、`README.md`（3 处）、`docs/video-reasoning.md`（1 处）都在引用它们。当前 `HEAD` 版 `timeline.md` 对 `assets/timeline` 的引用数为 0，说明这批图是工作区新增的。

若只 `git add` `.md` 就提交，GitHub 上全部时间线插图、入门图、演化图会 404，`links.yml` 也会失败。提交前请显式 `git add assets/`，并确认 20 MB 二进制入库是可接受的（否则考虑图床或 Git LFS）。

### P0-3 提交前重跑一次链接检查

审阅期间 `docs/video-reasoning.md` 一度不存在（06:26–06:42），而 README 有两处链接 + 结构树三处引用它。文件现已恢复，但说明有并发写入，`sources/timeline_review_20260829.md` 里"所有本地交叉引用都能解析"的结论只对当时快照成立。建议在 pre-commit 或 CI 里加一个纯本地的相对链接检查（不依赖网络），避免这类窗口期漏网。

---

## P1 — 事实与表述需要修正

> P1-4 ~ P1-12 已于 2026-08-29 修复。P1-1（Sora 2 API 时间线）、P1-2（已修）、P1-3（VPN 年份）见下表。

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| P1-1 | `docs/timeline.md` L623 | Sora 2 条目写"官方页面注明 Sora 产品自 2026-04-26 起不再提供"。实际：**Web/App 于 2026-04-26 停服，Sora API 到 2026-09-24 才停**，截至今天仍可用 | 改为"Sora Web/App 自 2026-04-26 停止提供，API 计划 2026-09-24 停止" |
| P1-2 | `docs/evaluation.md` L570 | ref [35] 标题漏字 | 正确标题：*How Should World Models Be Evaluated **for Embodied Decision-Making**? A Decision-Making-Centric Position* |
| P1-3 | `docs/timeline.md` L190 | Video Pixel Networks 标 2017，但本页开头明确规定"年份采用论文首次公开时间"，VPN 首次公开是 `arXiv 1610.00527`（2016-10） | 改为 2016，或在卡片里注明"2016 预印本 / ICML 2017"，与 MAGVIT、PlaNet 的处理保持一致 |
| P1-4 | `resources/datasets.md` L12 | UCF-101 挂 `[[4]]`，而 ref-4 是 **MoCoGAN** | 换成 Soomro et al. 2012 的 UCF-101 报告，或直接去掉编号 |
| P1-5 | `resources/datasets.md` L55 | nuScenes 挂 `[[5]]`，而 ref-5 是 **GAIA-1** | 换成 Caesar et al. nuScenes 论文 |
| P1-6 | `resources/datasets.md` L52 | 二级标题"驾驶与具身环境"后面挂了 `[[6]]`（Genie） | 删除；标题不应带引用 |
| P1-7 | `docs/applications.md` L13 | "多镜头生成 `[[3]]`"指向 **Sora 技术报告**，与多镜头无直接关系 | 改引 Phenaki / ShotAdapter / StoryMem |
| P1-8 | `docs/applications.md` §2 标题 | "视频编辑 `[[5]]`"指向 **AnimateDiff**，AnimateDiff 不是视频编辑工作 | 改引 VideoPainter / 指令式编辑工作，或去掉 |
| P1-9 | `docs/bibliography.md` | 章节列顺序错乱：A B C D E F **H G H G H** | `scripts/update_bibliography.py` L369 附近渲染表格时按 `entry['section']` 排序 |
| P1-10 | `docs/bibliography.md` L5 | 声称"覆盖仓库正文登记的 **49 篇**"，但全仓库实际出现 **132 个不同 arXiv ID**。多个章节写"标准引用见引用与代码索引"，读者去查会落空 | 要么改写为"覆盖核心 49 篇"，要么把 evaluation / physical-consistency / video-reasoning / tasks 的条目纳入 registry |
| P1-11 | 6 处 vs 1 处 | 同一作者两种写法：`generative-models.md` / `reading-list.md` / `tasks/image-to-video.md` / `tasks/video-prediction.md` / `generative-models/*.md` 写 "Remi Denton"，`evaluation.md` L510 写 "Denton, E." | 全仓库统一一种写法即可（沿用作者当前姓名或论文原始 byline，二选一，不要混用） |
| P1-12 | `docs/generative-models/*.md` 等 7 个文件 | 孤儿引用：`flow-consistency-models.md` 5 条参考文献**正文一条都没引**；`adversarial-generation.md`/`autoregressive-generation.md`/`masked-generation.md` 各 3 条；`recurrent-prediction.md`/`variational-generation.md` 各 1 条；`tasks/digital-human.md` 3 条 | 要么在正文补上引用点，要么把未引条目移到"延伸阅读" |

---

## P2 — 结构与一致性

### P2-1 三套互不兼容的 "L 等级"

- `docs/world-models.md` §3：L0 视觉生成器 → **L5 决策世界模型**
- `docs/evaluation.md` §8.2：L0 渲染质量 → **L7 现实效用**（且 §10.1 的模板要求填 "L0-L7"）
- `docs/video-reasoning.md` §12：另一套 0–6 级视频推理证据阶梯

三套都用 `Lk` 记号但语义不同：world-models 的 L5 是"决策世界模型"，evaluation 的 L5 却只是"闭环 rollout"。跨章节阅读必然串。建议：保留 evaluation 的 L0–L7 作为唯一的 world-model 证据阶梯，world-models §3 改成命名式（"生成器 / 预测器 / 状态保持 / 动作条件 / 交互 / 决策"）并注明映射关系，video-reasoning 的阶梯改用 `R0–R6` 之类独立前缀。

### P2-2 自动引用注入脚本留下的痕迹

`本页主要参考工作：…` 这句兜底话出现在 **13 个文件**（datasets、applications、generative-models、world-models、reading-list、tasks/ 下 8 个）。它的作用是给正文没引用到的参考文献找个落脚点，读起来像自动生成残留——尤其 `resources/datasets.md` 末尾"本页主要参考工作：Mastering Diverse Domains through World Models"（DreamerV3 出现在数据集页）几乎无意义。

同源问题还有 `docs/reading-list.md`：引用标记被插进标题中间——

- "**Unsupervised Learning for Physical Interaction `[[3]]` through Video Prediction**"
- "**VideoGPT `[[4]]`: Video Generation using VQ-VAE and Transformers**"
- "**Genie `[[8]]`: Generative Interactive Environments**"
- "**Neural Discrete Representation Learning `[[17]]` / VQ-VAE**"
- "**Adversarial Video Generation on Complex Datasets `[[16]]` / DVD-GAN**"

标记应放在整个条目末尾。这是"精选阅读列表"的门面页，值得手工修一遍。

### P2-3 时间线的收录标准前后矛盾

`docs/timeline.md` L458 "同期闭源产品如何放入时间线"说 Veo、Kling、Runway Gen-3 等因公开材料不足"视为产业背景，不用产品展示替代可核验的技术节点"。但后文 Veo 3、Sora 2、Genie 2/3、Marble、GWM-1、Kling 3、Seedance 2.0/2.5、MiniMax H3 全部作为独立节点收录。

标准本身没问题（这些新条目确实代表新能力），但这段说明需要重写为"2024 年的闭源产品迭代不单列；2025 年起，当官方发布明确给出新的技术能力维度（原生音视频、实时交互、多参考）时按官方发布收录，并标注证据等级"。

### P2-4 `.markdownlint.json` 与实际内容冲突

配置 `MD033.allowed_elements: ["a"]`，但 `timeline.md` 使用了 75 个 `<table>` 以及 `<tr>/<td>/<img>/<strong>/<br>/<code>`。CI 里没有 lint job，所以这个配置从未被执行；一旦有人跑 markdownlint 会瞬间几千条报错。建议要么补全 allowed_elements，要么删掉这个配置文件。

### P2-5 CI 链接检查偏松

`.github/workflows/links.yml` 用 `--accept 200,204,206,403,429`。403 被接受意味着 Cloudflare/反爬挡住的死链会被判为通过——本次人工核查中 openaccess.thecvf.com 就在连续请求后全站返回 403。同时只在 `push`/`pull_request` 触发，没有 `schedule`，资源腐化（本仓库大量链接标注了"已归档/已失效/当前不可用"）不会被自动发现。建议加每周定时任务，并对 403 单列白名单而不是全局接受。

### P2-6 其他一致性

- `CITATION.cff`：`authors` 用实体形式 `- name: "Video Generation 101 contributors"`，与 README 的 "Created by Codex and Xiaodong Cun (Corresponding Author)" 不一致；引用工具会渲染成机构而非个人。`date-released: 2026-08-09` 已落后于正文的 2026-08-29。建议改为 `family-names: Cun / given-names: Xiaodong` + ORCID，并同步日期与 version。
- README 仓库结构树：`sources/` 只列了 1 个文件（实际 17 个），没列 `assets/`、`.github/`、`docs/tasks/figures/`。
- `resources/open-models.md`：CogVideo 用 `THUDM/CogVideo`，`timeline.md` 用 `zai-org/CogVideo`（同仓库改名后的新地址），统一到 `zai-org`；Genie 列在"开放模型与代码"里但无开放代码，与本页"适合阅读、运行或二次开发"的定位冲突。
- `docs/generative-models.md` §2 标题是"VAE 路线"，代表工作却放了 MoCoGAN（GAN），且缺 SV2P、Stochastic Adversarial Video Prediction 等真正的 VAE 视频工作。
- `docs/generative-models.md` 记号冲突：§1 的 $x_t$ 指第 $t$ 帧，§4 的 $x_t$ 指扩散第 $t$ 步；同一页两种含义，对"101"定位的读者是硬伤。另外 §4 把反向步写成确定性映射 $x_{t-1}=g_\theta(x_t,t,c)$，DDPM 的反向步是随机的，建议写成 $x_{t-1}\sim p_\theta(x_{t-1}\mid x_t,c)$。

---

## P3 — 覆盖度：当前最明显的五个内容缺口

时间线和 foundation-models 对"规模化 + 条件扩展"这条线覆盖很完整，但下面五条 2024–2026 的主线基本缺席。按对本仓库定位的重要性排序：

1. **少步 / 因果化实时生成**：Diffusion Forcing、CausVid、Self-Forcing、few-step 蒸馏。全仓库 0 次提及。这是 Genie 3、Matrix-Game 2、GWM-1 能做到实时逐帧交互的直接技术前提，现在时间线里这些系统"为什么能实时"是断的，`flow-consistency-models.md` 只抽象提了一句"少步"。**这是最该补的一条。**
2. **可控性方法谱系**：相机控制（MotionCtrl / CameraCtrl / ReCamMaster）、轨迹控制、主体/参考一致性。`taxonomy.md` 有对应任务、`foundation-models.md` 有"多条件"维度，但没有任何一个代表方法。全仓库 0 次提及 ControlNet 类控制接口。
3. **偏好 / RL 后训练**：视频侧的 DPO、GRPO、reward model 训练。目前只在 `video-reasoning.md` 里以 RLVR 形式出现，`foundation-models.md` §4 的"后训练"只列了名词。2025–2026 最活跃的方向之一。
4. **数字人主线（你的主场，反而最薄）**：`digital-human.md` 的谱系止于 OmniHuman-1，缺 EMO、Hallo2/3、EchoMimic、Sonic、HunyuanVideo-Avatar、Wan-Animate、OmniHuman-1.5，以及 GVC Lab 自己的 PersonaLive。修 P0-1 时建议一并把 2024–2026 段补齐。
5. **开源模型索引滞后**：`resources/open-models.md` 声称"状态截至 2026-08"，但只有 Wan 2.1，缺 Wan 2.2（当前事实标准的开源 baseline）、Mochi、Step-Video-T2V、LTX-2、SkyReels、LongCat-Video 等；`FramePack` 只在 `tasks/video-prediction.md` 出现，没进时间线也没进开放模型页。

另外可考虑：`evaluation.md` 的 benchmark 谱系停在 WorldModelBench（2025）+ 一个泛指的"2026 决策中心评测框架"，可以补 VBench-2.0 一类的具体条目。

---

## 已复核为正确、不需要改动的部分（供后续审阅省时）

- 全部 2026 年 arXiv 条目真实存在：Cosmos 3 `2606.02800`、V-JEPA 2.1 `2603.14482`、LeWorldModel `2603.19312`、EB-JEPA `2602.03604`、Seedance 2.0 `2604.14148`、决策中心评测 `2606.15032`。
- `physical-consistency.md` 全部 10 条 CVF/WACV 链接真实可访问，venue 与第一作者均正确（含 CVPR 2026 的 PHANTOM、PhyCo 与 WACV 2026 的 Physics-IQ）。
- `docs/tasks/*.md` 中 16 个 2025–2026 新条目全部真实，标题与作者一致。特别说明：`tasks/video-prediction.md` ref [13]（`2504.12626`）标题看似不是 FramePack，实际是该论文 v3 的改题版本，**当前写法正确**，可加注 "(FramePack)" 帮助读者识别。
- `docs/video-reasoning.md` 抽查的 15 个 arXiv 条目 + 3 个非 arXiv 链接全部真实。仅两处微调：ref [19] 官方标题不含 "(VBVR)"、ref [15] 不含 "(VIPER)"，缩写是 benchmark 名而非标题的一部分。
- 厂商声明全部准确：Runway GWM-1（2025-12-11，Gen-4.5 底座，2 分钟 / 720p）、Genie 3（720p / 24 FPS / 数分钟）、Kling 3.0（2026-02-05，15 秒）、Seedance 2.5（2026-07-31，30 秒 / 30 图 / 10 视频 / 10 音频 / 时间戳编辑）、MiniMax H3（33B dense、2026-08-03 开源两个 CFG-distilled Base checkpoint、768p / 4–15 秒 / 24 FPS / 32 kHz）、Marble（2025-11-12，动态交互列为后续方向）、Cosmos Predict-2 链接有效。
  - 唯一可优化：Kling 官方命名是 **Kling AI 3.0**（Video 3.0 / Video 3.0 Omni），多镜头 storyboard 是 **Omni** 变体的能力；`developer.nvidia.com/blog/?p=101575` 建议替换为解析后的正式 slug。
- `docs/video-reasoning.md` 的 `lucas-maes/le-wm` 4,269 stars、`galilai-group/lejepa`（由 `rbalestr-lab/lejepa` 重定向）等仓库归属均核实无误。

---

## 建议的修复顺序

1. 重建 `docs/tasks/digital-human.md` 参考文献（P0-1），顺带补 2024–2026 数字人主线（P3-4）。
2. `git add assets/` 并跑一次本地链接与图片检查（P0-2、P0-3）。
3. 批量修 P1 的 12 条（多数是一行改动）。
4. 统一 L 等级记号（P2-1），清理引用注入残留（P2-2），重写时间线收录标准说明（P2-3）。
5. 按需补 P3 的内容缺口，优先"少步 / 因果化实时生成"。
