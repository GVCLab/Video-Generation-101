# 世界模型报告与评测协议

将来源、运行条件和行为证据分别记录，用于动作条件预测、可交互生成环境与 WAM 的比较。

**前置知识：** [评测总览](../evaluation.md)、[世界模型](../world-models.md)。

**使用步骤：** 固定声明与测试单位 → 记录来源和运行规格 → 选择相应 L0–L7 实验 → 在独立环境核查决策收益。

本页负责详细报告字段与指标。能力约定 L0–L7 的唯一释义见[评测总览](../evaluation.md#52)；WM 编号用于兼容已有报告，不构成另一套能力排名。场景特有的动作延迟和回访协议仍由[动作条件预测](../tasks/action-conditioned-prediction.md)与[交互式世界生成](../tasks/interactive-world-generation.md)负责。

## 1. 报告对象与能力证据

单个总分会掩盖“画面好但动作错”或“短时准但规划会利用漏洞”。发布内容、运行规格与行为能力分别报告；字段完整不等于能力通过。这里保留 **WM0–WM6** 作为世界模型报告字段编号，避免与[评测指南](../evaluation.md)中面向能力主张的全局 **L0–L7 报告约定**混淆：WM0–WM1 是发布与系统前提，不是能力等级；WM2–WM6 分别对应所需的全局能力证据。

| 报告字段（兼容 WM 编号） | 对应的能力证据 L0–L7 | 解释 |
|---|---|---|
| WM0 发布内容 | 不对应能力等级 | 只说明结论来自哪里、artifact 是否可得 |
| WM1 系统规格 | 不对应能力等级 | 只说明速度、延迟、动作频率与内存边界 |
| WM2 open-loop 感知 | L0–L2 | 从渲染质量到时间、语义与物理诊断 |
| WM3 状态与动作 | L3 | 给定动作后的状态转移是否正确 |
| WM4 反事实与不确定性 | L4 | 换动作时是否产生正确且校准的分支 |
| WM5 规划与决策 | L5–L6 | 先验证闭环 rollout，再验证策略排序与决策收益 |
| WM6 现实迁移 | L7 | 是否在独立真实系统中改善成功率、安全或数据效率 |

### WM0：发布内容与证据来源

对每一个结论记录：

- paper 版本、venue 状态、项目页更新日期；
- code、weights、data、evaluation script 是否公开；
- 论文 checkpoint 是否等同公开 checkpoint，公开小模型是否等同在线大 demo；
- 数字来自 peer-reviewed paper、arXiv、作者项目、代码 README 还是厂商发布。

### WM1：系统与运行规格

| 字段 | 统一报告方式 | 常见误读 |
|---|---|---|
| FPS | 生成帧数 / wall-clock second；注明硬件、batch、采样步数 | 把离线吞吐当交互速度 |
| 分辨率 | 模型实际输出，不把后处理上采样混入 | 分辨率更高就更物理 |
| Rollout horizon | 同时报 frames、seconds、decision steps 与首次失效点 | 只报最好样片长度 |
| Action rate | 每秒接受/执行多少次新 action；注明 action chunk | 与 video FPS 混为一谈 |
| Memory | context frames/tokens、KV 大小、landmark/entity store、增长率 | “支持长视频”等于有 persistent state |
| Latency | p50/p95 sense-to-act、first-frame、replan latency | 用平均 FPS 掩盖长尾卡顿 |

### WM2：open-loop perceptual quality

在 teacher-forced 或固定动作序列下报告 FVD/LPIPS/SSIM、human preference、temporal artifacts，并把 camera motion、对象运动、接触和遮挡分层。此层只回答“未来看起来怎样”，不能回答“动作是否正确”。

### WM3：state 与 action fidelity

- 从生成视频提取对象 pose、速度、接触、关节状态和任务事件，与真值轨迹比较；
- 计算 action-conditioned transition error、action recognizability、inverse-dynamics consistency；
- 对 no-op、相反动作、无效动作和边界动作分别报告；
- 检查 visual action、proprioception 与真实控制命令是否时序对齐。

### WM4：counterfactual 与不确定性

- 同初态、多动作、多随机种子成对比较；
- 测 causal effect error、branch coverage、calibration error、negative log-likelihood 或 risk–coverage curve；
- 设计 OOD 动作组合、未见对象关系和接触条件；
- 对“高价值但高 uncertainty”的候选，验证 conservative scoring 是否降低灾难失败。

### WM5：planning / decision utility

世界模型必须在独立真值环境中接受考试：

- policy ranking correlation：模型中的 policy 排名与真实环境排名是否一致；
- planning regret：world-model planner 与 oracle/真实 rollout planner 的收益差；
- model exploitation gap：在模型内高分、真实环境低分的差距；
- data efficiency：加入世界模型或合成数据后，达到同一成功率需要多少真实样本；
- safety utility：碰撞、违规、失败恢复和 constraint violation 是否改善。

WorldGym 把 action-conditioned video model 当作 policy evaluation environment，用 Monte Carlo rollouts 与视觉语言模型 reward 评估 VLA policy；作者报告模型内成功率与真机成功率高度相关，并能保留不同版本、规模和 checkpoint 的相对排序 [[28]](#ref-28)。论文同时明确指出真实物体交互仍难生成，因此“ranking 可用”与“绝对转移可信”仍是两个问题。

### WM6：real-world transfer

最后报告冻结/微调设置、真实机器人与相机、任务数、重复次数、成功率置信区间、干预次数、故障类别和 sim-to-real gap。若只在生成器自身 rollout 中评估 planner，不应记为通过 WM5，更不能据此支持全局 L7 的现实效用主张。

## 2. 评测集与最小报告矩阵

| 工作 | 主要补充 | 仍需搭配 |
|---|---|---|
| WorldEval [[27]](#ref-27) | 用 Policy2Vec 构造 latent-action world simulator，在线比较 policy 与 checkpoint 排名 | 独立环境的绝对成功率、广泛失败类别 |
| WorldGym [[28]](#ref-28) | 从真实初始帧做 Monte Carlo rollout，以 VLM reward 评估 policy 排名和新任务泛化 | 更可靠的物体交互、绝对校准和安全 |
| MiraBench [[29]](#ref-29) | physical adherence、action-following fidelity 与 optimism bias | 真实闭环 transfer |
| RoboWM-Bench [[30]](#ref-30) | 将生成的人手/机器人操作视频恢复为 embodied action，并在物理约束仿真中执行，暴露空间、接触与非物理形变错误 | 真机 transfer、更广任务、长期 memory 与成本报告 |

推荐最小报告矩阵：

| 层级 | 指标 | 短 horizon | 长 horizon | In-distribution | OOD |
|---|---|---:|---:|---:|---:|
| WM1 系统 | FPS、action Hz、p95 latency、memory | ✓ | ✓ | ✓ | ✓ |
| WM2 感知 | perceptual/temporal quality | ✓ | ✓ | ✓ | ✓ |
| WM3 状态动作 | pose/contact/event/action fidelity | ✓ | ✓ | ✓ | ✓ |
| WM4 反事实 | paired intervention、calibration | ✓ | ✓ | ✓ | ✓ |
| WM5 决策 | ranking、regret、exploitation gap | ✓ | ✓ | ✓ | ✓ |
| WM6 现实 | success、safety、transfer gap | ✓ | ✓ | ✓ | ✓ |


## 资料版本

结构调整：2026-10-15；正文由世界模型原报告章节迁入，保留 2026-08-30 资料范围。未刷新 benchmark、代码或权重状态；未运行模型。原始出处见[研究日志](../../sources/research_20260830_video_world_action_models.md)。

## 参考文献

<a id="ref-27"></a>[27] [WorldEval: World Model as Real-World Robot Policies Evaluator](https://arxiv.org/abs/2505.19017). arXiv:2505.19017. 2025.

<a id="ref-28"></a>[28] [WorldGym: World Model as An Environment for Policy Evaluation](https://arxiv.org/abs/2506.00613). arXiv:2506.00613, v3. 2025.

<a id="ref-29"></a>[29] [MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models](https://arxiv.org/abs/2605.29360). arXiv:2605.29360. 2026.

<a id="ref-30"></a>[30] [RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation](https://arxiv.org/abs/2604.19092). arXiv:2604.19092, v2. 2026.
