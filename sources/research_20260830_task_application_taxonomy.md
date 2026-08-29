# 任务分类与应用验收框架：检索、证据和生成图记录

> 检索日期：**2026-08-30（Asia/Shanghai）**
> 对应正文：`docs/taxonomy.md`、`docs/applications.md`
> 性质：任务定义与应用证据的 scoping review；不是产品排行榜，也不把厂商演示当独立复现

## 1. 改写目标与冻结快照

覆盖审计指出两个 P0：旧任务页把任务排成一条隐含成熟度链；旧应用页只有场景清单，缺少“能力 → 系统要求 → 验收 → 风险门 → 回滚”的证据链。本轮目标是：

1. 用条件来源、源内容关系和交互时域三个可组合轴替代线性任务链；
2. 为仓库每个任务给出输入、输出、不变量、允许变化和一票否决式错误；
3. 把论文、checkpoint、产品、工作流和部署证据分开；
4. 给创作、数字人、交互世界和 Physical AI 建立不同硬门槛；
5. 生成两张教材图，并保留可访问 Mermaid 与顺序化文字替代。

改写前快照：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `docs/taxonomy.md` | 76 | `50b0c049456eff1aafb2cb381cbfbd47faeed43ac338fd01317e4a33dffe2802` |
| `docs/applications.md` | 118 | `d5ef41d32c776bcca952221b96596262246004afcfdaf76ae4461d97fc840240` |

## 2. 可证伪问题

### 2.1 任务分类

1. 什么属性属于任务定义，什么属性只是模型实现？
2. T2V、I2V、V2V、inpainting、插帧、预测、动作预测、数字人、多镜头和交互世界的最近邻边界是什么？
3. 统一模型支持多个任务时，哪些任务合同仍不能合并？
4. 什么技术节点真正改变了条件接口、保留合同或交互时域？
5. 每条任务轴对应的最小证据和反事实测试是什么？

### 2.2 应用与部署

1. 一篇论文或一个 demo 到底能证明哪一层应用能力？
2. 创作、数字人、交互世界和 Physical AI 为什么不能共享一个总分？
3. 怎样把生成、编辑、人工修订、版本、SLO、来源和回滚串成可执行流程？
4. C2PA、NIST AI RMF 与 ITU-T P.910 能支撑什么，不能替代什么？
5. 2024–2026 的统一编辑、语义数字人、交互世界和 omnimodal Physical AI 应如何标注证据等级？

## 3. 检索策略与结果

### 3.1 数据源

采用三层互补来源：

1. **arXiv API**：发现最新预印本和核对首次公开日期、标题、作者；
2. **OpenAlex API**：宽召回和题名交叉检索；
3. **正式 proceedings / 标准 / 机构原页**：CVF、PMLR、NeurIPS、OpenReview、ITU、NIST、C2PA 与机构系统页作为最终证据。

Semantic Scholar Graph API 也执行了两条 discovery query，但在检索时返回 HTTP 429；没有把失败请求写成检索结果，也没有因此降低一手来源门槛。

### 3.2 机器检索快照

| 数据源 | 查询 | 返回量 / 状态 | 用途与限制 |
|---|---|---:|---|
| arXiv API | `all:"video generation"`，按 submittedDate 倒序 | 3,493 | 宽召回；包含 world model、理解与相邻任务，不能直接当纳入数 |
| arXiv API | `all:"video editing"`，按 submittedDate 倒序 | 569 | 检查统一编辑与 2026 frontier；标题/摘要筛选后再回原文 |
| OpenAlex | `title.search:video generation`，2015-01-01 至 2026-08-30 | 5,344 | title search 是 discovery，不等于精确主题分类 |
| OpenAlex | `title.search:video editing`，同日期范围 | 1,580 | 同上；可能匹配超分、分析或工具论文 |
| OpenAlex | `title.search:video world model`，2018-01-01 至 2026-08-30 | 379 | 宽词匹配会纳入题名中分散出现的词 |
| OpenAlex | `title.search:human animation`，2015-01-01 至 2026-08-30 | 451 | 包含传统动画，不直接等于生成式数字人 |
| Semantic Scholar | `video generation`；`video generation applications editing world model` | HTTP 429 | 记录为检索失败，不用于结果计数或证据 |

arXiv API 快照更新时间为 `2026-08-29T18:40:39Z`。最新 `video generation` 宽查询首条为 CLAP（`2608.27406`），最新 `video editing` 宽查询首条为 EditaLive（`2608.27123`）；二者用于确认 2026-08-27 仍有新分支出现，没有在本轮导航/应用总览中以未经充分筛选的名字堆砌。

### 3.3 手工精确检索

对候选逐一用题名和官方页面核验：

- `Video Diffusion Models`
- `Stable Video Diffusion`
- `Action-Conditional Video Prediction using Deep Networks in Atari Games`
- `Genie: Generative Interactive Environments`
- `Learning Interactive Real-World Simulators`
- `VACE: All-in-One Video Creation and Editing`
- `OmniHuman-1.5`
- `DreamGen`
- `Cosmos World Foundation Model Platform`
- `Cosmos 3`
- `C2PA Technical Specification 2.4`
- `NIST AI 600-1`
- `ITU-T P.910 07/2026`
- `MJ-Video`

## 4. 纳入、排除与证据等级

### 4.1 纳入

- 首次提出任务接口或能改变边界判断的原始论文；
- 正式会议论文页或作者技术报告；
- 官方系统页，仅用于当前公开规格和产品/机构主张；
- 官方标准与风险框架；
- 能把视频指标连接到下游或部署结果的原始研究。

### 4.2 排除

- 二手综述和媒体报道作为最终事实来源；
- 聚合排行榜、无协议产品比较、仅展示精选样例的页面；
- 只因使用通用视频生成器就被称为任务里程碑的工作；
- 只有平均 FPS、没有首帧/尾延迟/deadline 的“实时”主张；
- 只有画质、没有动作干预或闭环结果的“world model 可用于决策”主张；
- 只因存在水印或 Content Credentials 就声称内容为真的表述。

### 4.3 证据等级

| 等级 | 来源 | 可支撑范围 |
|---|---|---|
| A | 正式会议 proceedings / 标准规范 | 论文方法、正式 venue、规范范围与版本 |
| B | arXiv / 机构技术报告 | 作者披露的方法、设置与结果；必须标作者报告 |
| C | 官方系统、代码、模型或产品页 | 当前公开规格、发布面与入口；不能单独支撑科学共识 |
| S | 本文综合 | 分类轴、任务合同、应用门槛；必须与 A/B/C 事实分开 |

## 5. 关键证据矩阵

| 来源 | 等级 | 本轮使用的最小事实 | 明确没有外推的结论 |
|---|---|---|---|
| [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | B | 同一研究框架报告无条件、文本条件、预测和视频扩展 | 一个模型覆盖多任务不等于任务协议相同 |
| [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | B | 视频预训练、I2V 与相机 motion LoRA 适配 | 技术报告能力不等于当前产品或所有 checkpoint 能力 |
| [Action-Conditional Video Prediction](https://proceedings.neurips.cc/paper_files/paper/2015/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html) | A | 把动作作为控制变量进入高维未来帧预测 | Atari 设置不等于现实交互世界或机器人部署 |
| [Genie](https://proceedings.mlr.press/v235/bruce24a.html) | A | tokenizer + autoregressive dynamics + latent action，支持逐帧动作控制 | 不把该论文与 Genie 3 当前官方规格混写 |
| [Genie 3](https://deepmind.google/models/genie/) | C | 官方页面声称 720p、20–24 FPS 和持续交互 | 没写成公开 checkpoint、独立 SLO 或论文复现 |
| [UniSim](https://openreview.net/forum?id=sFyTZEqmUY) | A | action-in/video-out 统一接口，多域数据用于交互模拟 | 原文已限定 universal 的含义，不扩成模拟一切 |
| [VACE](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.html) | A | VCU 统一 reference、editing 与 mask，作者构建 12 任务评测 | 统一接口不取消每个任务的硬约束 |
| [OmniHuman-1.5](https://arxiv.org/abs/2508.19209) | B | 结构化语义条件与 Multimodal DiT 用于语义表演 | 不据此证明同意治理、长期身份或独立领先 |
| [DreamGen](https://arxiv.org/abs/2505.12705) | B | 生成 neural trajectories，latent action / IDM 恢复伪动作，连接 policy training | 下游收益不外推到其他机器人、数据或真实环境 |
| [Cosmos WFM](https://arxiv.org/abs/2501.03575) | B/C | 平台包含数据管线、tokenizer、预训练 WFM 与 post-training 示例 | 平台完整不等于每个下游系统已经安全 |
| [Cosmos 3](https://arxiv.org/abs/2606.02800) | B/C | omnimodal 技术报告处理/生成 language、image、video、audio、action；披露开放发布面 | 作者榜单不写成独立确认或同行评审共识 |
| [C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) | A | 2026-04 版本；可验证来源声明、绑定和防篡改 | 规范明确不判断内容好坏或事实真假 |
| [NIST AI 600-1](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) | A | 生成式 AI 生命周期风险管理 profile | 通用框架不替代视频任务指标或法律意见 |
| [ITU-T P.910 07/2026](https://www.itu.int/rec/T-REC-P.910-202607-P/en) | A | 2026-07-29 批准、当前 in-force 的主观视频质量建议 | 不覆盖全部生成指令、身份、动作与闭环证据 |
| [MJ-Video](https://proceedings.neurips.cc/paper_files/paper/2025/hash/71ad539a57b1fd49b19e5c80070cb8b9-Abstract-Conference.html) | A | 偏好拆为 alignment、safety、fineness、coherence/consistency、bias/fairness | 作者 reward benchmark 不是全应用唯一真值 |

## 6. 分类决策与负分类

### 6.1 为什么采用三个轴

旧页按“从生成到交互”排列任务，容易暗示后者是前者的升级版。实质上：

- 条件来源决定模型可利用什么证据；
- 源内容关系决定保持与允许变化的合同；
- 交互时域决定提交、记忆、延迟和闭环协议。

三者可组合，但不是任意无约束的笛卡尔积。数字人可用图像、音频、文字和姿态；V2V 可附文字、mask、reference 与相机；交互世界还可能接受语言与连续控制。因此正文把图中卡片称为“主坐标”，并用集合 $C$ 表示多条件。

### 6.2 刻意拒绝的分类

| 候选表述 | 处理 | 原因 |
|---|---|---|
| “I2V 是 T2V 的下一阶段” | 拒绝 | 两者条件与参考保持合同不同，不是成熟度顺序 |
| “插帧就是短期预测” | 拒绝 | 插值知道两侧端点；预测通常只知过去且未来多模态 |
| “inpainting 是一种 V2V，所以只需编辑分” | 拒绝 | 已知区是硬约束，必须报告 outside-mask error |
| “action-conditioned 就是 interactive” | 拒绝 | 离线动作序列不要求 deadline、状态反馈和错误恢复 |
| “数字人是 I2V 加音频” | 拒绝 | 增加身份、同步、人体、语义表演与同意/冒用风险 |
| “统一模型让任务边界过时” | 拒绝 | VACE 统一接口，但 reference/edit/mask 的验收仍不同 |
| “平均 FPS 达标即可部署” | 拒绝 | 必须报告 TTFF、p95/p99、deadline miss、并发和恢复 |
| “有 C2PA 就证明视频真实” | 拒绝 | 规范证明声明与资产的绑定/完整性，不判断事实真伪 |

## 7. 里程碑准则

里程碑至少满足以下一项：

1. 引入新的、可操作的条件接口；
2. 建立新的源内容保持或局部修改合同；
3. 将 open-loop 输出推进到长期记忆或 closed-loop；
4. 将生成视频连接到可重复的下游/部署证据；
5. 提供影响领域评测方式的新协议。

参数量、演示热度、分辨率和单一分数不是单独的里程碑准则。正文因此选取 Action-Conditional Prediction、VDM、SVD、Genie、UniSim、VACE、OmniHuman-1.5、DreamGen 和 Cosmos 3，并在表中同时写未解决问题。

## 8. 应用门槛的综合逻辑

应用 PASS 被写成硬门槛合取：

$$
G_q \land G_c \land G_p \land G_{\mathrm{SLO}}
\land G_s \land G_g.
$$

这是 S 级框架，不是某篇论文的统一公式。设计理由是：

- 数字人的高画质不能抵消未授权身份；
- 视频修复的感知质量不能抵消 mask 外证据变化；
- 交互世界的平均 FPS 不能抵消动作无效或 deadline miss；
- Physical AI 的真实感不能抵消闭环失败；
- provenance 完整也不能抵消事实错误。

## 9. AI 生成图记录

两图均使用 Codex 内置 `image_gen`，不是 CLI；原始输出保留在 Codex generated-images 目录，最终选中版本复制到仓库。所有图均为 1672 × 941 PNG，并在原始分辨率回读。

### 9.1 Video Generation Task Map

#### 学习目标

用一张图让读者看到：任务在“条件来源 × 源内容关系”矩阵中有主坐标，交互时域是独立第三轴；这些轴可以组合，不是成熟度阶梯。

#### 生成规格摘要

- 16:9 白底科学教育信息图；
- 纵轴：None、Text / Semantics、Image / Video / Audio、State + Action；
- 横轴：Create New、Reference-Constrained、Transform / Edit、Repair / Complete；
- 独立顶部轴：Open-Loop Clip、Long-Horizon Memory、Closed-Loop Control；
- 十个代表任务卡；
- 禁止模型名、分数、品牌、logo、水印和含混斜线。

#### 迭代与拒绝理由

| 版本 | 原始路径 | SHA-256 | 结论 |
|---|---|---|---|
| v1 | `.../exec-a3bf3659-a602-4669-9759-5b6c8ef3e51c.png` | `4ef29ef1ad4b4ff73f5f4286fe0e25c9100514dcfbb03a89c0783d0ea9b7199f` | 拒绝：Frame Interpolation 被放在 None 行；Video Prediction 和 Digital Human 坐标不准确 |
| v2 | `.../exec-a77b7aa0-22bd-43a3-95a4-634406915869.png` | `5e770c76fee8fc068c3ac17818bb775331edde4a60c9d9e10da49335734b8ddf` | 拒绝：任务坐标已修，但虚线把时域与错误卡片相连，可能暗示固定映射 |
| v3 | `.../exec-0f17cf1c-feda-4c99-8ee5-58ac00d4b370.png` | `54c3cc8abf27f7365f4ff1841ac745b73871e7ecbac44e118669240f8fc16907` | 采用：删除全部误导连接，保留独立第三轴 |

仓库文件：`assets/diagrams/video-generation-task-map.png`
仓库 SHA-256：`54c3cc8abf27f7365f4ff1841ac745b73871e7ecbac44e118669240f8fc16907`

#### 原图视觉验收

- 标题、三轴、十个任务卡和 footer 均完整；
- 无乱码、裁切、重叠、伪造数字、模型品牌或水印；
- Unconditional、T2V、I2V、Prediction、Digital Human、V2V、Inpainting、Interpolation、Action-Conditioned 和 Interactive World 主坐标与正文一致；
- 交互时域无指向具体任务的误导连线；
- 图注明确卡片不是穷尽映射，Mermaid 和文字替代承载精确定义。
- 使用 ImageMagick 转为灰度后原尺寸回读，轴、卡片、图标和所有文字仍可区分。

### 9.2 From Capability Claim to Deployment Evidence

#### 学习目标

显示从 use-case contract 到 input/control、model workflow、human-in-the-loop、acceptance、deploy/monitor 的硬门槛流程，并把领域证据分成创意媒体、数字人、交互世界和 Physical AI。

#### 生成规格摘要

- 六个编号阶段与 `PASS / STOP` gate；
- 四条领域证据 lane；
- footer：`A demo proves possibility. Deployment requires repeatability, risk controls, and rollback.`；
- 禁止模型/公司名、伪造数字、排行榜、logo 和水印。

原始路径：`.../exec-7bd666f9-969c-4ac8-968f-f150a98170ad.png`
仓库文件：`assets/diagrams/capability-to-deployment-evidence.png`
SHA-256：`39faec4d17c67b5c84ebf31ed99ee9d3c77102bb6ef8d5a25ada58ba956962a7`

#### 原图视觉验收

- 六阶段顺序、PASS / STOP、四条 evidence lane 和 footer 均可读；
- 无截断、乱码、伪造性能数字、logo 或水印；
- 四领域文字与正文验收重点一致；
- 图中反馈/回滚箭头与正文 Mermaid 的确定性流程一致；
- 颜色不是唯一编码：编号、标题、卡片形状和图标共同表达层级。
- 去除 alpha 并转为灰度后原尺寸回读，六阶段、PASS / STOP 和四条 evidence lane 仍可辨认。

## 10. 最终机器验证

在 2026-08-30 全批次文件冻结后执行：

- Markdownlint：17 个改动或新增 Markdown，0 issues；
- 引用锚点：221 个定义、398 次使用；missing、unused、duplicate 均为 0；
- 相对文件与图片链接：79 条，失效 0；
- Mermaid：用 Mermaid CLI 11.16.0 与本机 Chrome 实际渲染 17/17，均生成非空 SVG；各图另由对应改写线做原图/布局检查；
- PNG：6/6 为可解码 PNG，单图 1.2–1.6 MB；任务图、部署图和其他专题图的尺寸与 SHA 已分别记录；
- 一手来源回归：纠正 UniSim 的 ICLR OpenReview ID；独立审计核对 87 个 arXiv ID/标题及 DOI、会议和项目外链，并修正 WAV、WorldGym、MiraBench、RoboWM-Bench、Infinite-World、ViPRA、Deep Voxel Flow 与 NeurIPS 页面错配；
- 安全与仓库卫生：未发现凭证、密钥、临时文件或超过 10 MB 的新增文件；
- `git diff --check`：通过；
- 独立只读 pre-push 审计：已发现的问题均回归修复，无未解决 P0/P1。
