# 动作条件预测研究记录（冻结于 2026-08-30）

## 1. 任务与边界

- 对应正文：`docs/tasks/action-conditioned-prediction.md`
- 时间冻结：2026-08-30，Asia/Shanghai。
- 目标：把动作条件视频预测写成可复核的 causal/control contract，而不是新论文清单。
- 仅使用一手技术来源支撑技术主张：正式 proceedings/journal、arXiv 原文、作者官方项目页、官方仓库或官方模型卡。
- 二手搜索结果只作 discovery，不进入正文证据。
- 正文相邻边界：passive past-only prediction 留给 `docs/tasks/video-prediction.md`；持续实时交互系统留给 `docs/tasks/interactive-world-generation.md`；更广义的 model-based control 留给 `docs/world-models.md`。

本轮必须分别回答：

1. action-conditioned observation model 是否只学到行为策略相关性；
2. 动作命令、实际执行和观测帧如何按时间对齐；
3. fixed open-loop、policy-in-model、MPC 与真实 closed-loop 的观测来源是否不同；
4. inverse dynamics、reward/termination 和 joint action head 是否被越界解释；
5. pixel、latent、representation、token/diffusion 路线各自接受什么证据；
6. 2025–2026 论文状态与 release surface 是否可独立核验；
7. 是否给出 model exploitation、uncertainty、多模态和最小证伪实验。

## 2. 证据等级

| 等级 | 定义 | 可支持的表述 | 不足之处 |
|---|---|---|---|
| **A** | 正式 proceedings 或 journal 的原论文页/论文 | 正式发表、论文内方法和实验结果 | 不保证代码、权重或部署栈可得 |
| **B** | arXiv 原文，且有作者官方项目/仓库/模型卡交叉核对 | 截止版本的预印本方法、作者报告结果和 release 入口 | 未经正式同行评审；仓库可继续变化 |
| **C** | 作者官方项目页、博客、仓库或模型卡 | demo/release surface、硬件或版本说明 | 不能替代论文方法与完整实验；动态页面易漂移 |
| **D** | 搜索索引、聚合元数据或搜索摘要 | 发现候选、检查是否可能存在正式版本 | 不进入正文技术结论 |

规则：正文主张的强度不高于支持它的最高证据等级；“正式发表”“有项目演示”“有代码”“有权重”“可端到端复现”逐项记录，不能合并成一个“已发布”。

## 3. 检索记录

### 3.1 arXiv API 定量检索

同日用 arXiv API 执行以下 title/all-field 查询；计数只用于说明候选池，不用作科学趋势的统计结论：

| 查询式 | 结果数 | 用途 |
|---|---:|---|
| `ti:"action conditioned" AND all:video` | 28 | 直接动作条件视频候选 |
| `ti:"world action model"` | 131 | 2025–2026 WAM 命名候选；噪声较高 |
| `all:"action-conditioned world model"` | 56 | 补充不以该短语为标题的工作 |
| `ti:"video prediction" AND all:robot` | 29 | 机器人视频预测前史与评测 |
| `ti:"world model" AND all:robot` | 442 | 高召回发现；仅凭该结果不纳入 |

### 3.2 精确搜索式

按方法、证据面和负面核验分别搜索：

```text
"action-conditioned video prediction" Atari NeurIPS proceedings
"physical interaction through video prediction" NeurIPS proceedings
PlaNet latent dynamics planning pixels PMLR
Dreamer Nature 2025 reward continue observation model
MuZero Nature learned model reward value policy
TD-MPC2 OpenReview decoder-free latent world model
DINO-WM ICML 2025 official repository
Genie generative interactive environments PMLR latent action
DIAMOND diffusion world modeling NeurIPS official repository
GameNGen ICLR 2025 OpenReview
V-JEPA 2 action conditioned planning official repository
DreamGen CoRL 2025 official project inverse dynamics synthetic video
ViPRA ICLR 2026 action chunk Hz project repository
"World Action Models are Zero-shot Policies" arXiv project code latency
GigaWorld-Policy 0.5 official repository model weights runtime
A2World official repository ECCV 2026 proceedings
MultiWorld multi-agent multi-view official repository
LeWorldModel official repository checkpoint
WorldGym ICLR 2026 policy evaluation project
RoboWM-Bench CVPR 2026 workshop proceedings
MiraBench action following optimism bias official code
RLVR-World NeurIPS 2025 proceedings
Genie 3 paper checkpoint code official
```

### 3.3 正式 venue 与 release 核验

- PMLR 原始页面核验 PlaNet、Genie、DINO-WM、DreamGen。
- NeurIPS 原始页面核验 Oh 2015、Finn 2016、DIAMOND 2024、RLVR-World 2025。
- OpenReview 原始 forum 核验 Dreamer、TD-MPC2、GameNGen、ViPRA、WorldGym 的 venue/status。
- Nature 原始页面核验 MuZero 与 DreamerV3 journal 版本。
- CVF Open Access 原始页面核验 RoboWM-Bench 的 CVPR 2026 Workshop 身份。
- arXiv abs 页面核验 V-JEPA 2、DreamZero、GigaWorld-Policy、GigaWorld-Policy-0.5、A2World、MultiWorld、LeWorldModel、MiraBench 的 ID、标题和版本面。
- GitHub REST API 核验官方仓库默认分支、HEAD、license 字段和更新时间；仓库 README/目录只用于 release surface。
- Hugging Face 仅在作者仓库直接链接时作为官方模型卡使用。

### 3.4 失败与降级记录

- OpenAlex 查询返回 HTTP 429，错误指出当日余额不足；没有使用 OpenAlex count，也没有用搜索摘要补数。
- 一次 `git ls-remote` 因 LibreSSL `SSL_ERROR_SYSCALL` 失败；改用 GitHub REST API 核验 commit SHA。
- 没有把搜索引擎 snippet 当作论文状态、实验数字或 release 证据。
- 外链最终状态统一在验证阶段逐项记录；403/429 不自动等于来源不存在，要与浏览器可读性和来源类型分开。

## 4. 纳入与排除

### 4.1 纳入标准

至少满足一项能力转折，并有可定位的一手来源：

- 动作第一次进入像素/观测转移；
- latent dynamics 真正进入 online planning 或 imagined policy learning；
- 放弃像素重建而保留任务充分 latent；
- 从 action-free 视频学习 latent action 或 representation，再映射到控制；
- 生成环境被 policy/agent 消费，而不是只展示视频；
- 世界模型生成视频/伪动作并改善下游策略；
- joint video-action 模型在 fresh-sensor loop 中直接输出动作；
- benchmark 把动作跟随、物理执行、optimism bias 或 policy ranking 单独测量。

### 4.2 排除或降权

- 综述、媒体报道、聚合榜单和未指向原始来源的项目列表：排除为正文证据。
- 只做 text-to-video、camera control 或 passive continuation，且没有可执行 action contract：转到相邻章节。
- 只有精选 demo、没有论文/代码/模型卡：最多 C 级，不能进可复现路线。
- 只报告 inverse dynamics accuracy，没有 forward prediction 或 policy utility：不能支持 world-model 主张。
- 只用同一 logged action schedule 的视频指标，没有 action shuffle/paired branch：只视为 associational open-loop 证据。
- 没有真实 sensor 回灌的 simulator rollout、retargeting 或 VLM judge：不写成真实 closed-loop。
- GAIA-1、Cosmos 等较广平台本轮不作为主线：不是否定其价值，而是已有来源足以覆盖本章的能力转折，且需控制篇幅与动作合同差异。
- UniSim、iVideoGPT 等相邻交互模拟工作未进入主表：正文已有 Genie/DIAMOND/GameNGen 与机器人 WAM 两条代表链；避免把论文数量当综述深度。

## 5. 逐项证据账本

| 工作 | 主来源 | 等级 | 本章采用的事实 | 明确不推出 |
|---|---|---|---|---|
| Oh et al. 2015 | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2015/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html) | A | Atari 动作进入卷积视觉转移，生成长动作条件帧序列 | 通用物理、真实机器人闭环 |
| Finn et al. 2016 | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2016/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html) | A | DNA/CDNA/STP、真实 pushing interaction、action-conditioned pixel motion | 长程多物体/多峰问题已解决 |
| PlaNet | [PMLR](https://proceedings.mlr.press/v97/hafner19a.html) | A | stochastic + deterministic latent transition，online latent planning | 逼真视频生成能力 |
| Dreamer | [OpenReview](https://openreview.net/forum?id=S1lOTC4tDS) | A | latent imagination 中学习行为 | 无模型偏差或无需真实交互 |
| DreamerV3 | [Nature](https://www.nature.com/articles/s41586-025-08744-2) | A | recurrent latent transition 以动作条件化，并有 observation/reward/continue heads；固定配置跨多任务 | 所有视觉世界模型统一最优 |
| MuZero | [Nature](https://www.nature.com/articles/s41586-020-03051-4) | A | task-relevant latent、reward/value/policy 可用于 planning，无需重建完整观测 | latent 足以迁移到任意新任务 |
| TD-MPC2 | [OpenReview](https://openreview.net/forum?id=Oxh5CstDJU) | A | decoder-free latent model 与 MPC/价值结合用于连续控制 | 观察视频的像素级正确性 |
| DINO-WM | [PMLR](https://proceedings.mlr.press/v267/zhou25a.html), [official repo](https://github.com/gaoyuezhou/dino_wm) | A+C | frozen DINO features、离线 action dynamics、测试时动作优化；论文覆盖六个环境 | 通用真实机器人像素模拟 |
| Genie | [PMLR](https://proceedings.mlr.press/v235/bruce24a.html) | A | tokenizer + autoregressive dynamics + 从无标签视频学习 latent action | latent action 已标定为真实 actuator command |
| DIAMOND | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html), [official repo](https://github.com/eloialonso/diamond) | A+C | diffusion environment model 被 agent 训练/交互消费 | 游戏结果直接外推真实物理控制 |
| GameNGen | [OpenReview](https://openreview.net/forum?id=P8pqeEkn1H) | A | action-conditioned diffusion 实时模拟 DOOM 的作者实验 | 端到端机器人 latency 或开放世界 |
| Genie 3 | [official research preview](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | C | 官方演示口径：720p、24 FPS、数分钟、有限研究预览与有限直接动作空间 | 有公开论文/checkpoint、可独立复现 |
| V-JEPA 2 / 2-AC | [arXiv](https://arxiv.org/abs/2506.09985), [official repo](https://github.com/facebookresearch/vjepa2) | B | action-free 大规模视觉预训练；另以少量 DROID 动作视频后训练 AC predictor 做机器人规划 | base encoder 本身就是 action-conditioned policy |
| DreamGen | [PMLR](https://proceedings.mlr.press/v305/jang25a.html), [official project](https://research.nvidia.com/labs/gear/dreamgen/) | A+C | video WM 生成 synthetic video，经 latent action/IDM 伪标动作后训练 policy | 在线 MPC 或模型在真实闭环中生成下一观测 |
| ViPRA | [OpenReview](https://openreview.net/forum?id=w3Ik8HUyTT), [project](https://vipra-project.github.io/), [repo](https://github.com/sroutray/vipra) | A+C | joint future observation + motion latent；少量 demo 训练 flow action decoder；action chunk release | 所有 Hz 数字属于同一测量合同 |
| DreamZero | [arXiv](https://arxiv.org/abs/2602.15922), [project](https://dreamzero0.github.io/), [repo](https://github.com/dreamzero0/dreamzero) | B | 14B joint video-action diffusion；作者报告真实 closed-loop、真实观测刷新缓存与约 7 Hz | 任意任务/机器人/动作空间上的绝对 zero-shot |
| GigaWorld-Policy | [arXiv](https://arxiv.org/abs/2603.17240) | B | action token 不读取 future-video token；视频生成可在部署省略 | causal attention 已证明环境因果可识别 |
| GigaWorld-Policy-0.5 | [arXiv](https://arxiv.org/abs/2607.13960), [official repo](https://github.com/open-gigaai/giga-world-policy) | B | 独立 revision、weights/runtime/release 声明 | 可与原论文数字直接拼成一个版本 |
| A2World | [arXiv](https://arxiv.org/abs/2606.29501), [repo](https://github.com/LogosRoboticsGroup/A2World), [model card](https://huggingface.co/Fleurrr/A2World-World-Model) | B | 多 embodiment、多视角 action-conditioned pretraining，world-model/policy release 面；arXiv comments 与仓库标注 ECCV 2026 accepted | 截止冻结日已有可独立定位的 ECCV proceedings 页面 |
| MultiWorld | [arXiv](https://arxiv.org/abs/2604.18564), [official repo](https://github.com/CIntellifusion/MultiWorld) | B | multi-agent、多视角 action-conditioned video 与 code/data/weight 入口 | 多视角一致自动解决多主体因果归因 |
| LeWorldModel | [arXiv](https://arxiv.org/abs/2603.19312), [official repo](https://github.com/lucas-maes/le-wm) | B | 15M 参数量级 end-to-end JEPA 与 latent MPC 的作者报告 | 正式同行评审结论 |
| WorldGym | [OpenReview](https://openreview.net/forum?id=hidBHy1CAw), [project](https://world-model-eval.github.io/) | A+C | action-conditioned video rollout + Monte Carlo + VLM reward 可用于作者协议的 policy ranking | absolute simulator fidelity 或 safety certification |
| RoboWM-Bench | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.html) | A | 生成行为经 inverse dynamics/retargeting 后在物理模拟执行，暴露接触/空间错误 | 原始模型直接控制真实机器人 |
| MiraBench | [arXiv](https://arxiv.org/abs/2605.29360) | B- | physics、action following、optimism bias 分层；作者报告视觉质量不是动作正确性的可靠代理 | 已有核验过的官方代码/项目 release |
| RLVR-World | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html) | A | 以 decoded verifiable rewards 后训练 tokenized world models | 任意代理 reward 都不会被 exploit |

## 6. 关键负面核验

| 核验问题 | 结论 | 正文处理 |
|---|---|---|
| Genie 3 是否有公开论文和 checkpoint？ | 截止冻结日只定位到官方研究预览；未定位到论文/checkpoint release | 列 C 级 demo，不列可复现论文 |
| V-JEPA 2 是否整体 action-conditioned？ | 大规模 base pretraining 是 action-free；V-JEPA 2-AC 是后训练 predictor | 两部分分开写 |
| DreamGen 是否 online world-model control？ | 主线是生成视频、恢复伪动作、训练下游 policy 的 offline pipeline | 不写成 MPC/真实观测闭环 |
| DreamZero 的 zero-shot 是否无限定？ | 仍受预训练数据、任务描述、embodiment、相机和动作表示支持范围约束 | 只写 scoped zero-shot |
| ViPRA 的 20/22/7/3.5 Hz 是否同一指标？ | 论文摘要、项目动作 chunk 与有/无 KV cache 的视频生成表面不同 | 不合并，正文要求完整 latency contract |
| GigaWorld-Policy 与 0.5 是否同一版本？ | arXiv ID 和 release 面不同 | 分列并冻结版本 |
| A2World 的 ECCV 2026 是否有官方 proceedings？ | arXiv comments 与仓库均标注 accepted；冻结日未独立定位到正式 proceedings 页面 | 正文以 arXiv 为技术来源，并把会议状态限定为作者元数据 |
| MultiWorld license 是否单一？ | 仓库元数据为 NOASSERTION；README 区分代码 Apache-2.0 与数据/权重 CC BY-NC 4.0 | 不笼统写“全项目 Apache” |
| WorldGym 的 policy ranking 是否等于真实物理精度？ | 否；作者也报告现实对象交互不足 | 只支持 scoped ranking correlation |
| RoboWM-Bench 是否真实 robot closed-loop？ | 主要链路含 inverse dynamics/retargeting 和 physics simulation | 明确不是原模型真实闭环 |
| MiraBench 是否有已核验官方 repo？ | 本轮未找到可由论文/作者页面闭合的官方代码仓库 | 只引用 arXiv，不称代码已发布 |

## 7. 仓库版本冻结

以下 SHA 由 GitHub REST API 在 2026-08-30 读取默认分支 HEAD；它们只冻结本轮观察面，不保证仓库所有外部权重仍可下载：

| 仓库 | HEAD SHA | API license 字段 / 备注 |
|---|---|---|
| `facebookresearch/vjepa2` | `204698b45b3712590f06245fbfba32d3be539812` | MIT |
| `gaoyuezhou/dino_wm` | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` | MIT |
| `eloialonso/diamond` | `5bcd1599755b4f2fae8e5e079e02f0728e174965` | MIT |
| `NVIDIA/GR00T-Dreams` | `ec3881d44545016871997f8e17dd15f1d792e91d` | Apache-2.0 |
| `sroutray/vipra` | `216c5da91a929ece664a888121e45f6c216a14bb` | Apache-2.0 |
| `dreamzero0/dreamzero` | `ab790c198fbce33503358efbbd4187ce9a89adf3` | Apache-2.0 |
| `open-gigaai/giga-world-policy` | `f3d5a8864895aa57377095ceddf21e9419c0b7e6` | API 未给出确定 license；按仓库文件单独核验 |
| `LogosRoboticsGroup/A2World` | `bab12ba61f67993083e5ef61ee13c95d150bc17c` | Apache-2.0 |
| `CIntellifusion/MultiWorld` | `6e0a1c3414a97dd5528d94f67ba3c48cf6d36ab5` | API 为 NOASSERTION；README 区分 code 与 data/weights |

LeWorldModel 仓库用于确认 release 入口，但本轮没有把未记录的 commit 猜成冻结 SHA。

## 8. 正文设计决策

### 8.1 为什么不按论文逐篇介绍

正文按能力门组织：动作进入转移、latent imagination 接 planning、任务充分 latent、无标签 latent action、生成环境被 agent 消费、视觉基础模型后训练、offline synthetic data、joint video-action closed loop。这样可见每次真正新增了什么证据，也可防止“模型更大/年份更新”被误写成能力跃迁。

### 8.2 为什么加入 reward 与 termination

policy 不只消费画面。若 reward 被高估或 continue 永远为 1，planner 可偏好视觉合理但任务失败的轨迹。正文因此把 observation、reward 和 continue/done 作为独立 heads/metrics，并要求 Brier/ECE 与终止时序误差。

### 8.3 为什么把 latency 写进 tensor contract

动作命令时刻、执行时刻和观测曝光时刻决定条件的因果顺序；数组索引对齐不足。action chunk 还决定每次盲飞时长和真实观测回灌频率，因此“帧率”不能单独代表可控实时性。

### 8.4 为什么要画 model-exploitation curve

一般 open-loop metric 平均了大量普通动作；planner 会主动搜索模型偏差。把候选数从 16 扩到 1024，并同时画预测/真实 return，可直接观察搜索能力增强是否反而降低真实表现。

## 9. 主线生成图记录

- **路径：** `assets/diagrams/action-conditioned-world-model-loop.png`
- **工具与日期：** built-in OpenAI image generation，2026-08-30
- **尺寸：** 1672 × 941 RGB PNG
- **SHA-256：** `211afd3695c9f6bceeb7b1d834199afa821f266870cbf9924d713c5ef2cef7ef`
- **灰度统计：** mean 0.907275，standard deviation 0.231726，min 0，max 1

Prompt 要求 16:9 白底科学示意图，四栏分别展示 observation history 与 hidden state、action schedule 与 latency、同一 belief 下的 `do(a1/a2/a3)` 随机反事实、goal/reward → planner → action → model rollout → real observation → belief update 的决策回路；底部把 pixel plausibility、action sensitivity、intervention accuracy 与 closed-loop utility 分层。禁止模型名、benchmark 数字、营销式写实机器人与含糊交叉箭头。

原尺寸回读确认：三条反事实从同一 `s(t)` 分叉；action timeline 与 $\Delta$ 清楚；闭环从 update belief 返回 planner；证据阶梯没有把像素合理性升级成控制效用。图中没有塞入 inverse dynamics、reward/done calibration 或 exploitation curve，以免信息过载；正文图注明确把这些精确关系留给 Mermaid 与第 7–9 节。正文同时提供 alt、图注和五步顺序化文字替代。

## 10. 验证结果

验证于 2026-08-30 在仓库根目录执行。

| 项目 | 命令/方法 | 结果 |
|---|---|---|
| Markdown | 主线最终复跑 `npx --yes markdownlint-cli2` 检查正文与本记录 | **PASS**：markdownlint-cli2 0.23.2 / markdownlint 0.41.1，2 files，0 errors |
| 引用闭环 | Ruby 解析 `[[N]](#ref-N)` 与 `<a id="ref-N"></a>[N]` | **PASS**：67 次正文引用；33 个唯一编号；33 个定义；范围 1–33；无缺失、孤儿、重复或编号错配 |
| 本地链接 | 逐文件按相对目录解析非 HTTP Markdown 链接 | **PASS**：主线插图后 10 个本地链接全部存在 |
| 外链 | 取两文件唯一 URL，`curl -L` 并发 GET；失败项再用网页读取 | **PASS with transport notes**：39 个唯一外链；29 个由 `curl` 返回 HTTP 200；9 个 GitHub 页面和 1 个 Hugging Face 模型卡因本机 LibreSSL `SSL_ERROR_SYSCALL` 返回 000，随后网页读取逐项成功解析页面标题与正文 |
| 403/429 | 单独记录限流/拒绝，不把它写成来源不存在 | GitHub REST 定向复查在额度耗尽后返回 **403**；本轮外链页面无 429；发现阶段 OpenAlex 返回 **429**，故未采用其结果 |
| Mermaid 数量与无障碍字段 | 提取 `mermaid` fenced blocks，检查 `accTitle`/`accDescr` | **PASS**：2 blocks，二者均含标题与描述 |
| Mermaid 实渲染 | `@mermaid-js/mermaid-cli@11.9.0` + 本机 Google Chrome，输出到 `/tmp` SVG | **PASS**：2/2；临时 SVG 分别为 38,511 与 31,040 bytes；临时渲染产物未写入仓库 |
| 主线 PNG | 尺寸、hash、灰度统计与 original-detail 回读 | **PASS**：四栏、反事实分叉、闭环回箭头和四级证据均清晰；正文含 alt、图注、文字替代 |
| Diff whitespace | `git diff --check`；新文件另用 `git diff --no-index --check /dev/null ...` 查看 whitespace diagnostics | **PASS**：tracked diff 无错误；新文件无 trailing whitespace、space-before-tab 或 EOF blank-line 错误 |

Mermaid CLI 首次安装因 Puppeteer 尝试下载浏览器而退出；复查 npm log 后设置 `PUPPETEER_SKIP_DOWNLOAD=true`，并以临时 `puppeteer.json` 指向 `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`，随后两图均实际生成非空 SVG。该过程只写 `/tmp/action-conditioned-mermaid.*`，未修改仓库资产。

最终 `git status --short` 显示工作区还有其他协作者的章节和图片改动；审稿子任务只负责两份 Markdown，主线随后增加上述 PNG 与图文记录。未修改 coverage audit，未 commit 或 push。
