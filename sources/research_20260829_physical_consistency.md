# 视频生成物理一致性：一手来源审计与研究注册表

- **审计日期：** 2026-08-29
- **覆盖范围：** 2024-01-01 至 2026-08-29 首次公开或正式发表的物理视频生成、物理评测、3D/4D 动力学、动作条件世界模型工作；少量更早工作只作为定义参照
- **用途：** 为 `docs/physical-consistency.md`、`docs/world-models.md`、动作条件与交互式世界任务页提供可核查的概念、方法、benchmark、指标和证据边界
- **来源纪律：** 仅采用 arXiv、CVF、PMLR、NeurIPS Proceedings、OpenReview、作者项目页或官方代码仓；厂商演示和二手综述不用于支撑论文数字
- **版本纪律：** “预印本”“正式会议论文”“官方代码当前协议”分别标注；动态 leaderboard 不当作永久论文结论

## 1. 结论先行

1. **看起来合理不等于物理忠实。** 物理合理性回答“这个结果可能发生吗”；物理忠实度回答“在给定初态、控制、物性和边界条件下，结果是否在测量误差内接近指定现实系统”。一条形状正确但重力加速度错误的抛物线，可以合理却不忠实。
2. **物理一致性是中间概念，不应包办所有主张。** 它要求时序状态不自相矛盾并满足声明的约束，但模型仍可能在错误参数、错误尺度或错误系统边界下保持自洽。
3. **2024--2025 年主流 benchmark 以人类或 VLM 判断物理常识为主；2025--2026 年出现三次关键升级：** 真实实验续写与程序测量、逐定律和逐时刻诊断、机器人/驾驶中的可执行与决策证据。
4. **VLM judge 与程序指标互补，不能互相冒充。** VLM 擅长开放域语义和显眼违规，程序测量擅长标定轨迹、接触与参数；二者分别受语言先验和感知提取器误差影响。
5. **物理引擎参与不自动产生真实物理。** 几何、物性、外力、接触模型和边界条件估计错误时，求解器只会给出“精确但错误”的轨迹；生成渲染还可能破坏已经正确的模拟状态。
6. **动作条件是从视频生成到 world model 的必要但非充分条件。** 相机控制、latent action、语言事件和标定机器人动作是不同证据；只有反事实分支、闭环执行、策略排序或真实任务收益才能支撑决策型 world model 主张。
7. **截至审计日，最接近“physical fidelity”定义的公开评测是 GAUGE。** 它使用标定真实轨迹、物理元数据、不确定性和任务特定观测量，直接显示“方程形式看起来正确但加速度、动量传递或振荡时序错误”的失败。

## 2. 检索、纳入与证据分级

### 2.1 检索路径

本次审计采用以下路径，而不是把一个聚合站点当作完整索引：

1. 从现有章节中的 VideoPhy、PhyGenBench、VideoPhy-2、Physics-IQ、PhysGen、PhysGen3D、PhyT2V、VLIPP、PHANTOM、PhyCo 逐条回到论文最终页。
2. 沿 benchmark 的相关工作和官方项目页向前追踪 WorldModelBench、Physics-IQ Verified、PAI-Bench、PhysInOne、Physion-Eval、PhyGround、GAUGE、RoboWM-Bench、Apple-PI 和 YoCausal。
3. 沿方法机制追踪四条链：显式仿真与 3D/4D 表示、轨迹/物性条件、latent dynamics、训练后与推理时对齐。
4. 沿应用证据追踪动作条件机器人和驾驶工作，检查是否真正报告闭环执行，而不是只报告 FVD、VBench 或生成样例。
5. 对所有 2026 条目核对 arXiv 首次提交日期和正式会议页；没有正式 proceedings 的统一标为预印本。

### 2.2 纳入标准

- 直接研究生成视频的物理合理性、物理一致性或物理忠实度；
- 提供可复用 benchmark、数据、评估器或程序化协议；
- 通过动作条件预测、闭环机器人或驾驶评测把物理视频模型连接到决策；
- 公开一手论文或官方材料，且截至 2026-08-29 可核查。

### 2.3 不纳入或降级的证据

- 只有产品 demo、无论文/代码/独立协议的“world model”宣传，只能作为能力声明，不能支撑物理忠实度；
- 单个精选视频、作者挑选的成功案例和未冻结 judge 的动态网页分数不作为稳定结论；
- FVD、审美、清晰度、普通 temporal consistency 不单独视为物理证据；
- 仅证明 latent 可被线性 probe 读出，不等于模型因果地使用该变量；
- 模拟器中的 task success 不直接外推为真实世界成功。

## 3. 概念层级与形式化边界

### 3.1 六个必须分开的层次

| 层次 | 操作性问题 | 最低证据 | 仍不能声称什么 |
|---|---|---|---|
| 视觉真实感 `visual fidelity` | 单帧或局部外观是否接近照片/参考 | 人评、图像质量、参考相似度 | 时序正确、动力学正确 |
| 时间连贯性 `temporal coherence` | 身份、纹理、几何是否连续，是否闪烁/瞬移 | 长时跟踪、时序指标、人评 | 接触、守恒和参数正确 |
| 物理合理性 `physical plausibility` | 一个观察者是否认为“这件事可能发生” | 人类/VLM rubric、逐规则违规 | 唯一动力学、参数恢复、可控响应 |
| 物理一致性 `physical consistency` | 生成状态是否内部自洽并满足已声明约束 | 对象持续、接触、约束残差、事件顺序 | 与某个指定现实系统数值一致 |
| 物理忠实度 `physical fidelity` | 给定初态、动作、物性和边界，预测是否匹配标定参照及其不确定性 | 轨迹/接触/参数/场量误差，成组反事实 | 对规划一定有用、真实部署一定安全 |
| 决策忠实度 `decision fidelity` | 模型是否保持动作后果、策略排序、回报和风险 | 闭环 rollout、regret、policy ranking、真实成功率 | 开放世界全域的物理完备性 |

### 3.2 最小形式化

令真实或高可信参照系统满足：

```math
s_{t+1}=F(s_t,u_t,\theta,b,\xi_t),
\qquad
y_t=H(s_t,v)+\epsilon_t,
```

其中：

- $s_t$ 是位置、速度、姿态、形变、温度或场变量等状态；
- $u_t$ 是力、机器人控制、车辆控制或环境干预；
- $\theta$ 是质量、摩擦、恢复系数、刚度、黏度等物性；
- $b$ 是支撑、容器、接触、光照和其他边界条件；
- $v$ 是相机和成像条件；
- $\xi_t$ 与 $\epsilon_t$ 分别表示过程随机性和测量噪声。

视频模型产生 $\hat x_{1:T}$ 后，评测器通过测量算子 $M_k$ 提取可观察量。物理忠实度不是简单要求像素相等，而是检查：

```math
E_k=
\frac{d_k\!\left(M_k(\hat x_{1:T}),y^{*}_{k,1:T}\right)}
{\sigma_k+\varepsilon},
```

其中 $y^*$ 是标定真实或高可信模拟参照，$\sigma_k$ 是参照与测量链的不确定性。一个结果只有在报告了单位、时间尺度、相机标定、测量覆盖率和误差条带时，才接近“fidelity”而不是“plausibility”。

反事实忠实度还要求模型响应干预：

```math
\Delta^{\text{model}}_{u\rightarrow u'}
=M(\hat x\mid u')-M(\hat x\mid u)
\approx
\Delta^{\text{ref}}_{u\rightarrow u'}.
```

只在一个默认参数点产生看似正确的结果，无法排除模板匹配或训练分布记忆。

### 3.3 约束不是同一种东西

| 约束类型 | 示例 | 合适的检查 | 常见误用 |
|---|---|---|---|
| 硬等式 | 刚体长度、不可压缩体积、无源连续性 | 几何/体积/PDE 残差 | 把像素身份稳定当质量守恒 |
| 不等式与接触互补 | 不穿透、法向力非负、接触/分离互斥 | 接触时刻、signed distance、冲量 | 只看慢动作是否平滑 |
| 动量与能量账 | 外力冲量、碰撞、功、热与耗散 | 系统边界内的残差 | 声称机械能在有摩擦时也必须守恒 |
| 随机或统计规律 | 飞溅、湍流、材料微结构 | 分布、覆盖率、校准 | 要求随机系统逐像素复现唯一未来 |
| 学习先验 | 人体动作、复杂材料响应 | OOD、干预、失败覆盖 | 将高似然直接解释为定律 |
| 决策约束 | 动作可执行、避免碰撞、完成任务 | 闭环成功、regret、安全指标 | 用 FVD 代替控制效果 |

### 3.4 守恒陈述必须声明系统边界

- 开放系统或存在外力时，总动量可以改变；应比较动量变化与外部冲量，而不是机械地要求零残差。
- 有摩擦、塑性或阻尼时，机械能不守恒；完整能量账应包括热、不可逆形变和外界做功。
- RGB 中“外观未变”只能支持对象持续性，不能直接证明质量未变。
- 流体体积近似守恒需要相机/深度/遮挡处理；单视角 2D 面积不是体积。

## 4. 证据阶梯

![图 092：L0 · 视觉真实感单帧外观与参考相似到禁止压成一个不透明总分的流程](assets/imagegen-diagrams/092/diagram.png)
使用规则：

- claim 不得高于实际通过的最高证据层；L2 高分不能写成 L3，L3 也不能自动写成 L6。
- 高层级不意味着每篇论文都必须做到；研究问题可以只针对 L2 或 L3，但必须准确命名。
- 每次升级都要过一个额外 gate：L2 到 L3 需要 **judge 校准与程序测量**，L3 到 L4 需要 **成组干预**，L4 到 L5 需要 **独立执行环境**。
- 推荐配色：L0--L1 灰、L2 蓝、L3--L4 青、L5--L6 橙、L7 深绿；不能只靠红/绿表达好坏。

## 5. Benchmark 名称与版本审计

### 5.1 容易混淆的名称

- **VideoPhy** 是论文题名；官方仓库后来用 **VideoPhy-1** 与 VideoPhy-2 区分。引用论文时不应把题名改成 VideoPhy-1。
- **PhyGenBench** 是准确拼写，自动评估器叫 **PhyGenEval**。未核验到名为 “PhysGenBench” 的对应正式 benchmark；该写法通常是把 **PhysGen** 方法与 **PhyGenBench** 混在一起。
- **PhysGen** 是刚体物理引导的 I2V 方法；**PhysGen3D** 是单图构建可交互 3D 小世界的方法，均不是 PhyGenBench。
- **Physics-IQ** 使用连字符和大写 IQ；2026 年新增的审计版叫 **Physics-IQ Verified**，两个版本的样本、prompt 和聚合协议不能混用。
- **WorldModelBench** 是一个完整专名；不要与泛称 “world model benchmark” 或同名 workshop 混写。
- **PhysVideoBench** 是 NS-Diff 论文引入的方法特定评测，不等于 PhyGenBench，也不宜在尚无广泛复现时称为社区标准。
- **PISA Experiments** 的 free-fall 诊断常被称作 PisaBench；“PISA”还与无关的稀疏注意力方法撞名，引用时必须带完整题名或 arXiv:2503.09595。
- **PhyWorld** 至少指两个不同工作：arXiv:2411.02385 的受控 scaling 研究，以及 arXiv:2605.19242 的两阶段后训练视频模型；必须带年份和题名。

### 5.2 2024--2025 benchmark 注册表

| Benchmark / 工作 | 版本与状态 | 原生规模与协议 | 主要指标/裁判 | 可支持的结论 | 关键边界 |
|---|---|---|---|---|---|
| [VideoPhy](https://arxiv.org/abs/2406.03520) | 2024 arXiv；ICLR 2025 | 688 captions、138 unique actions、12 个 T2V 模型、11,330 个生成视频、36,500 条人类标注；三类材料交互 | semantic adherence、physical commonsense；人评和 VideoCon-Physics | 开放域材料交互中的可见常识违规 | 主要是二元/分类判断；无法恢复重力、摩擦或质量等参数；官方仓库排行榜会更新 |
| [PhyGenBench](https://proceedings.mlr.press/v267/meng25c.html) | ICML 2025 | 160 prompts、27 条物理定律、4 个领域；论文实验为 8 模型、1,280 视频 | 分层 PhyGenEval，结合 VLM/LLM 与规则问题 | 逐定律、逐场景诊断物理常识 | 仍依赖视觉语言裁判；prompt 规模小于开放世界；不是精密动力学测量 |
| [How Far Is Video Generation from World Model: A Physical Law Perspective](https://arxiv.org/abs/2411.02385) | 2024 预印本；受控研究 | Box2D 物理任务；数据约 30K 到 3M，DiT 约 22M 到 310M，区分 ID、组合和 OOD | 受控状态/轨迹误差与 scaling 对比 | 规模和覆盖可改善 ID/组合泛化，但物理 OOD 不会自动闭合 | 简单 2D 合成世界；不是开放域 T2V benchmark |
| [Physics-IQ](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html) | 2025 arXiv；WACV 2026 最终版 | 66 个真实实验场景 × 3 视角 × 2 次拍摄 = 396 个参考视频；198 个 take-1 模型条件；原始视频 4K/30 FPS；3 秒条件、5 秒预测 | Spatial IoU、Spatiotemporal IoU、Weighted Spatial IoU、MSE；按两次真实实验的物理随机性归一化 | 真实实验续写中的运动位置、时刻、幅度和变化形态 | 静态相机和像素变化提取；聚合分数受版本影响；最终论文最佳原始基线为 29.5%，不要与后续 leaderboard 混用 |
| [WorldModelBench](https://arxiv.org/abs/2502.20694) | NeurIPS 2025 Datasets & Benchmarks | 7 domains、56 subdomains、350 图文条件；14 个前沿模型；67K human labels | instruction following 0--3；5 项物理违规；commonsense；2B human-aligned judge | 应用域中的指令遵循、常识与物理违规分解 | 仍评生成视频，不测试动作规划；“质量变化”多是外观/几何代理；judge 可能漂移 |
| [VideoPhy-2](https://arxiv.org/abs/2503.06800) | ICLR 2026；2025 首次公开 | 200 个动作；ICLR 接受版写 4,000 个详细 prompts；人类 1--5 评分和逐规则标注 | semantic adherence、physical commonsense、rule grounding；VideoPhy-AutoEval | 动作中心、较困难且细粒度的常识评测 | 200 是动作数，不是 prompt 数；发布数据过滤后的行数应随 tag 报告；最佳模型 hard subset 联合表现仅 22% 是该论文版本结论 |
| [PISA Experiments / free-fall diagnostic](https://proceedings.mlr.press/v267/li25bu.html) | ICML 2025 | 单一基础但可控的自由落体任务；含模拟后训练与诊断 benchmark | 轨迹、下落行为、生成质量和奖励模型 | 检查后训练能否诱导基本重力行为 | 任务极窄；自由落体改善不能外推到碰撞、流体或开放世界 |

### 5.3 2026 benchmark 与验证升级

| Benchmark / 工作 | 截止日状态 | 准确规模或协议 | 证据升级 | 关键边界 |
|---|---|---|---|---|
| [Physics-IQ Verified](https://arxiv.org/abs/2606.18943) | 预印本；代码已并入官方仓库 | 修订 57.6% 样本，改善 34.8% prompts；6 个 I2V 模型重测后排名 Kendall $\tau=0.46$；官方协议建议 4 runs 报均值和标准差 | 审计 ground truth、prompt、样本权重和聚合方式 | 直接证明 benchmark 版本会改变模型排序；不能把旧榜和 Verified 榜混合 |
| [PAI-Bench](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PAI-Bench_A_Comprehensive_Benchmark_For_Physical_AI_CVPR_2026_paper.html) | CVPR 2026 | 2,808 个真实世界 cases，覆盖视频生成、条件视频生成和视频理解 | 把生成与理解纳入统一 Physical AI 诊断 | 主要仍是 perception/prediction，不等同于真实机器人闭环 |
| [PhysInOne](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PhysInOne_Visual_Physics_Learning_and_Reasoning_in_One_Suite_CVPR_2026_paper.html) | CVPR 2026 | 2M synthetic videos、153,810 dynamic 3D scenes、71 个基础现象；含 3D 几何、语义、运动、物性和文本标注 | 同一套数据支持生成、长/短期预测、物性估计、motion transfer | 合成数据和新视角泛化仍有 sim-to-real；规模不代表每种复杂动力学都被充分覆盖 |
| [Physion-Eval](https://arxiv.org/abs/2603.19607) | 预印本 | 5 个生成模型；10,990 条专家 reasoning traces；22 个细粒度类别；ego/exo；每个生成视频有真实参考和时间定位 glitch | 从粗分数升级到逐时刻、逐原因的人类诊断 | 专家解释仍是观察性判断；论文报告 83.3% exocentric、93.5% egocentric 视频至少有一个可识别 glitch，不等于全部帧错误 |
| [PhyGround](https://arxiv.org/abs/2605.10806) | 预印本 | 250 prompts、13 条定律、8 个模型；459 annotators、5,796 份完整标注、超过 37.4K 细粒度 labels | 逐定律 observable sub-question；发布 PhyJudge-9B | VLM judge 仍是裁判；论文报告 split-half 排名 Spearman $`\rho\gt0.90`$，是该采样设计的可靠性，不是绝对正确率 |
| [GAUGE](https://arxiv.org/abs/2608.05948) | 2026-08 新预印本 | 22 个受控任务族，覆盖刚体、柔性缆绳、织物和体积可变形体；3 个物理引擎评 14 族；6 个 I2V 模型评 5 个刚体任务 | 标定真实轨迹、物理元数据、不确定性、任务特定观测、generalized trajectory error 和参数稳定性 | 最新预印本，尚待独立复现；视频模型部分只覆盖 5 个刚体任务，但它最清楚地区分 plausibility 与 fidelity |
| [RoboWM-Bench](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.html) | CVPR 2026 Workshop | 将人手或机器人生成视频经 inverse dynamics / pose retargeting 转成动作，在 real-to-sim 重建环境执行 | 从“看起来可行”升级到 step executability 和 task success | 逆动力学、retargeting 与重建仿真器是额外瓶颈；证明仿真可执行性，不自动证明现实可执行性 |
| [Apple-PI](https://arxiv.org/abs/2607.16401) | 预印本 | Orchard 含 400 个视频、10 个经典力学任务；11 个模型；Perception--Formulation--Deduction 三阶段 | 混合 MLLM 主观分和定律客观测量，定位推理阶段 | 把生成视频当“可见思维轨迹”是评测假设，不等于内部因果机制；最佳视频模型 0.473 为当前论文快照 |
| [YoCausal](https://arxiv.org/abs/2605.30346) | 预印本 | 用真实视频时间反转构造可扩展 counterfactual；评 13 个 VDM | RSI 测时间箭头，CCI 借 VLM 分层因果/非因果数据 | 反向异常不等于因果理解；CCI 仍受 VLM 分层误差影响 |
| [PhysVideoBench in NS-Diff](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_NS-Diff_Fluid_Navier-Stokes_Guided_Video_Diffusion_via_Reinforcement_Learning_CVPR_2026_paper.html) | CVPR 2026；方法内 benchmark | curated 刚体/流体序列和 ground-truth motion proxy | 为 jerk 与流体 divergence 提供程序测量 | 方法特定、覆盖窄；minimum jerk 与低 divergence 分别不是牛顿动力学和完整 Navier--Stokes 的充分条件 |

## 6. 2024--2026 方法注册表

### 6.1 隐式学习、语言规划与事件链

| 工作 | 状态 | 机制 | 实际里程碑 | 失效边界 |
|---|---|---|---|---|
| [PhyWorld scaling study](https://arxiv.org/abs/2411.02385) | 2024 预印本 | 在受控 2D 物理中系统改变数据量、模型量与覆盖 | 把 ID、组合泛化、物理 OOD 分开，显示单纯 scaling 不够 | 结论来自简单合成任务，不能直接量化开放域基础模型 |
| [PhyT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_PhyT2V_LLM-Guided_Iterative_Self-Refinement_for_Physics-Grounded_Text-to-Video_Generation_CVPR_2025_paper.html) | CVPR 2025 | LLM 生成物理规则，迭代检查并重写生成条件 | 展示无需重训完整基础模型的轻量对齐路线 | 语言规则不能强制扩散模型逐帧执行；易受同一 VLM 自评闭环影响 |
| [VLIPP](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VLIPP_Towards_Physically_Plausible_Video_Generation_with_Vision_and_Language_ICCV_2025_paper.html) | ICCV 2025 | VLM 作粗粒度 motion planner，再以轨迹/变化引导 VDM | 把“先规划事件/运动、再渲染”明确分工 | 粗轨迹不表达连续接触、流体场或真实物性 |
| [CoECT](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Chain_of_Event-Centric_Causal_Thought_for_Physically_Plausible_Video_Generation_CVPR_2026_paper.html) | CVPR 2026 | 公式辅助的 event-centric causal chain 与 transition-aware prompt | 将物理提示从孤立规则推进到事件链 | 事件顺序正确仍不保证速度、冲量和材料响应数值正确 |

### 6.2 显式模拟器、系统辨识与 3D/4D 表示

| 工作 | 状态 | 几何/物理接口 | 实际里程碑 | 失效边界 |
|---|---|---|---|---|
| [PhysGaussian](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html) | CVPR 2024 | 3D Gaussians 与 Material Point Method 结合 | 让可渲染 3D 表示直接参与物理仿真 | 依赖场景重建、材质类别与 MPM 假设；不是开放域 RGB-only 世界模型 |
| [PhysDreamer](https://arxiv.org/abs/2404.13026) | ECCV 2024 | 从单图/视频先验估计物性并驱动 3D Gaussian 动力学 | 把视频生成先验用于动态 3D 资产和物性反演 | 单目系统辨识不适定；漂亮运动不证明物性唯一或正确 |
| [DreamPhysics](https://arxiv.org/abs/2406.01476) | AAAI 2025 | 视频生成先验指导动态 3D Gaussian 运动 | 强化单图动态资产生成的视觉覆盖 | 生成先验可能把常见运动风格当物理；需真实参数验证 |
| [Physics3D](https://arxiv.org/abs/2406.04338) | 2024 预印本 | 借视频扩散先验学习 3D Gaussian 物性 | 探索从视觉先验到 3D 物理属性的桥接 | 物性识别的可辨识性和真实标定不足 |
| [PhysGen](https://arxiv.org/abs/2409.18964) | ECCV 2024 | 图像理解、2D 刚体模拟、视频扩散细化 | 明确“模拟状态负责运动，生成模型负责外观” | 主要是 2D 刚体；深度、接触和材质估计错误会传递到底层轨迹 |
| [PhysGen3D / MiniTwin](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_PhysGen3D_Crafting_a_Miniature_Interactive_World_from_a_Single_Image_CVPR_2025_paper.html) | CVPR 2025 | 单图重建可交互 3D 小世界，设置速度和材质后模拟 | 从一次性视频推进到可编辑的 3D 状态 | 单图几何/材质不适定；miniature world 的可控性不等于真实系统标定 |
| [PhysFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Unleashing_the_Potential_of_Multi-modal_Foundation_Models_and_Video_Diffusion_CVPR_2025_paper.html) | CVPR 2025 | 多模态模型初始化材质，3DGS 表示，可微 MPM 和 optical-flow guidance 反演 | 形成“材质初始化--可微求解--视频先验校正”的系统辨识闭环 | 光流和视频扩散都是代理；材料类型与求解器覆盖有限 |
| [WonderPlay](https://openaccess.thecvf.com/content/ICCV2025/html/Li_WonderPlay_Dynamic_3D_Scene_Generation_from_a_Single_Image_and_ICCV_2025_paper.html) | ICCV 2025 | 单图与动作输入；物理求解、视频生成和 3D 场景更新循环 | 将 action-conditioned 视频与动态 3D 世界联结 | 循环误差会累积；solver、视频模型和重建器的责任需分开消融 |
| [TRACE](https://openaccess.thecvf.com/content/ICCV2025/html/Li_TRACE_Learning_3D_Gaussian_Physical_Dynamics_from_Multi-view_Videos_ICCV_2025_paper.html) | ICCV 2025 | 从动态多视角视频学习 3D Gaussian translation/rotation dynamics 与物理参数 | 强化从多视角观测学习未来 3D 动力学 | 需要多视角；参数聚类可分对象不等于物性已真实恢复 |
| [PSIVG](https://openaccess.thecvf.com/content/CVPR2026/html/Foo_Physical_Simulator_In-the-Loop_Video_Generation_CVPR_2026_paper.html) | CVPR 2026 | template video → 4D/mesh 重建 → 物理模拟 → trajectory guidance；TTCO 保纹理 | 把模拟器真正放入生成测试时循环，而非只用于造数据 | 模板视频和 4D 重建先验可能错误；TTCO 修纹理不修动力学 |

这一类方法的共同审计链应固定为：

```math
\text{geometry}
\rightarrow \text{system identification}
\rightarrow \text{solver}
\rightarrow \text{render/generate}
\rightarrow \text{measurement}.
```

每一箭头都必须有独立误差或消融。只报告最终视频偏好，无法知道改进来自更正确的物理、较好的纹理，还是更有利的相机视角。

### 6.3 轨迹、力、物性与跨视图条件

| 工作 | 状态 | 条件接口与规模 | 实际里程碑 | 失效边界 |
|---|---|---|---|---|
| [PhysCtrl](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f53fd88a4340063ecd258c0ae9948b40-Abstract-Conference.html) | NeurIPS 2025 | 3D point trajectories；550K 物理模拟动画；四类材料；物理参数与外力条件 | 将可解释连续物性/外力转为可驱动 I2V 的 3D 运动控制 | 合成材质与真实物体存在域差；展示可控不等于参数响应已校准 |
| [PhyCo](https://openaccess.thecvf.com/content/CVPR2026/html/Narayanan_PhyCo_Learning_Controllable_Physical_Priors_for_Generative_Motion_CVPR_2026_paper.html) | CVPR 2026 | 超过 100K photorealistic simulations；摩擦、恢复、形变、外力；pixel-aligned property maps + ControlNet + VLM reward | 无需推理时模拟器也能提供连续物理控制 | VLM reward 可能投机；应测摩擦/恢复/力扫描的单调性和数值误差 |
| [PhysVideo](https://arxiv.org/abs/2603.18639) | 预印本 | PhysMV：40K scenes、四个正交视角、160K sequences；cross-view geometry guidance | 用多视角几何减少单视角运动歧义 | 合成正交视角不是现实常见输入；项目页仍为匿名托管，需等待稳定代码版本 |
| [Goal Force](https://openaccess.thecvf.com/content/CVPR2026/html/Gillman_Goal_Force_Teaching_Video_Models_To_Accomplish_Physics-Conditioned_Goals_CVPR_2026_paper.html) | CVPR 2026 | 显式 force vector 和中间动力学；合成 causal primitives | 将语言目标细化为可解释的力条件 | 简单合成力传播到真实工具操作是作者报告的 zero-shot 结果；仍需标定力--运动曲线和闭环验证 |
| [GenieDrive](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_GenieDrive_Towards_Physics-Aware_Driving_World_Model_with_4D_Occupancy_Guided_CVPR_2026_paper.html) | CVPR 2026 | 控制 → 4D occupancy → 多视角视频；41 FPS；occupancy 模块 3.47M 参数 | 用显式 4D 状态瓶颈约束驾驶视频 | 论文报告 forecasting mIoU 提升 7.2%、FVD 降低 20.7%；这些不是闭环驾驶安全或规划收益 |

### 6.4 Latent dynamics 与表示解释

| 工作 | 状态 | 机制 | 实际里程碑 | 失效边界 |
|---|---|---|---|---|
| [PHANTOM](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_PHANTOM_Physics-Infused_Video_Generation_via_Joint_Modeling_of_Visual_and_CVPR_2026_paper.html) | CVPR 2026 | 共同预测视觉内容与 physics-aware latent dynamics | 把物理状态预测直接嵌入视频未来生成 | latent 语义未完全可辨；benchmark 提升不能证明 latent 是正确物理变量 |
| [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) | 预印本 | 对 V-JEPA 2、VideoMAE-v2 做逐层 probe、子空间、patch decoding、attention ablation | 发现中间层 Physics Emergence Zone；变量呈高维分布式编码 | probe 可读出不等于生成/决策时因果使用；线性可分也不等于完整状态 |
| [LDR](https://arxiv.org/abs/2608.09926) | 2026-08 新预印本 | 在 structured latent 中显式做低阶运动学积分，只回归三阶及以上 residual | 五个受控任务上测动力学 OOD；论文报告 ID--OOD error gap 比扩散基线小超过 20×、参数少 26×、速度快 143× | 仅 uniform/parabola/collision/bouncing/looming 等简单模拟任务，分辨率最高 256²；不能直接外推开放世界 |

### 6.5 训练后、奖励与推理时对齐

| 工作 | 状态 | 优化信号 | 实际里程碑 | 失效边界 |
|---|---|---|---|---|
| [PISA Experiments](https://proceedings.mlr.press/v267/li25bu.html) | ICML 2025 | 少量模拟自由落体 SFT + reward modeling | 证明窄物理任务可通过后训练显著诱导 | generalization 和 distribution modeling 仍有限；单一重力原语不能代表通用物理 |
| [NewtonRewards](https://arxiv.org/abs/2512.00425) | 预印本 | optical flow 代理速度；高层外观特征代理质量；恒加速度和“质量保持”奖励；NewtonBench-60K，5 个运动原语 | 把程序代理变为可优化的 verifiable rewards | optical flow 不是速度，appearance 不是质量；五个原语只支持牛顿运动的窄结论 |
| [Inference-time Physics Alignment](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Inference-time_Physics_Alignment_of_Video_Generative_Models_with_Latent_World_CVPR_2026_paper.html) | CVPR 2026 | V-JEPA 2 latent reward 搜索/引导多条 denoising trajectory；WMReward | 显示 test-time compute 和 latent world model 可改善物理合理性 | reward 仍是学习代理；论文的 62.64% PhysicsIQ Challenge 分数对应特定 challenge/protocol，不应与 WACV 原始 29.5% 直接比较 |
| [NS-Diff](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_NS-Diff_Fluid_Navier-Stokes_Guided_Video_Diffusion_via_Reinforcement_Learning_CVPR_2026_paper.html) | CVPR 2026 | noisy-latent motion detector；速度场/形变梯度/材料 mask；minimum jerk 和简化流体 divergence RL | 论文报告 jerk error 降 43%、fluid divergence 降 33%、FVD 降 22.7% | 这些是代理改善；作者方法并未求解完整 Navier--Stokes，也未证明真实流体参数忠实 |
| [DynamicsBoost](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.html) | CVPR 2026 | 参考帧更多的 continuation 作为偏好优序；Asymmetrical DPO 排除共享 prefix | 无人工/VLM 标签构造动态偏好 | “更多真实 prefix 的 continuation 更好”是数据构造假设；偏好顺序不等于逐定律正确 |
| [PhyWorld: Physics-Faithful World Model](https://arxiv.org/abs/2605.19242) | 预印本 | flow-matching continuation fine-tune + physics preference DPO | 将续写稳定性和物理偏好分两阶段训练 | 论文 VBench 0.769 与 per-law 3.09 是作者报告；DPO 目标主要支持 plausibility，不自动达到测量型 fidelity |

### 6.6 从生成视频到动作条件与闭环

| 工作 | 状态 | 动作与环境接口 | 最高证据 | 边界 |
|---|---|---|---|---|
| [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) | 2025 预印本/官方发布 | 超过 1M 小时互联网视频预训练；少于 62 小时 DROID 无标签机器人视频后训练 latent action-conditioned model | 两个实验室 Franka image-goal pick-and-place 零样本规划 | 任务和 embodiment 有限；latent 预测质量与物理忠实度仍应分开测 |
| [DreamZero / World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) | 预印本 | 14B 自回归视频扩散联合建模视频与动作；作者报告 7 Hz 闭环 | 真实机器人实验；作者报告对新任务/环境相对强 VLA 超过 2×，少量跨 embodiment 数据迁移 | 尚属作者预印本证据；系统优化、动作头和视频模型贡献需独立复现 |
| [GenieDrive](https://arxiv.org/abs/2512.12751) | CVPR 2026 | 标定驾驶控制，经 4D occupancy 再生成多视角未来 | 状态预测和动作可控视频 | 没有用 occupancy/FVD 结果证明闭环 planning、安全或 policy ranking |
| [RoboWM-Bench](https://arxiv.org/abs/2604.19092) | CVPR 2026 Workshop | 生成行为 → inverse dynamics / retargeting → 重建仿真执行 | step executability 与 task success | 结果是整个动作提取--仿真管线的联合表现；需报告每个转换步骤误差 |

## 7. 训练、后训练与验证路线

![图 093：成组数据真实标定 + 模拟干预 + action logs到按证据层分别报告的流程](assets/imagegen-diagrams/093/diagram.png)
推荐的工程顺序：

1. **先定义 claim。** 如果目标只是减少明显穿透，就不应把工作描述成“学习真实世界定律”；如果目标是机器人 planning，就必须预先定义决策指标。
2. **建立成组干预数据。** 固定场景、相机和文本，只系统改变 $u$、$\theta$ 或 $b$；保存 RGB、深度、mask、轨迹、接触和测量不确定性。
3. **选择状态瓶颈。** 刚体可用 3D pose/trajectory，软体需 mesh/point/strain，流体需速度/体积分数等场；不能用一个稀疏点轨迹覆盖所有材料。
4. **选择物理注入点。** 训练时损失适合大规模学习；后训练适合修基础模型；推理时 simulator/verifier 适合无需改权重但计算更高的控制。
5. **冻结评测器再训练。** judge checkpoint、prompt、frame sampler、感知提取器和 benchmark commit 必须版本化，避免训练目标与最终评测悄然同源。
6. **四路验证。** 人/VLM、程序测量、反事实、闭环分别报告，不以加权平均掩盖某一路完全失败。

## 8. 指标、协议与失效边界

### 8.1 VLM 与人类裁判

适合：

- 对象、动作和事件是否出现；
- 宏观事件顺序是否合理；
- 明显穿透、悬浮、增生、融化、错误反弹；
- 开放域、难以预先写程序规则的失败发现。

不适合单独判断：

- 瞬时接触、短暂拓扑错误和高频振荡；
- 隐藏质量、摩擦、力、温度和实际单位；
- 相机运动与物体运动的精确分离；
- 同样“看起来合理”的不同参数系统；
- 已针对同一 judge 优化后的模型是否 reward hacking。

最低复现字段：

- judge 模型、checkpoint 或 API 日期；
- system/user prompt、rubric 与输出解析；
- frame sampling、视频压缩、最大时长；
- 是否允许 `abstain / cannot determine`；
- 每类样本的人类校准、混淆矩阵和 inter-rater reliability；
- 对已知正确/已知违规 counterfactual 的敏感度；
- absolute score 与 pairwise preference 同时报；
- judge 失败和无法解析样本占比。

### 8.2 程序化测量

| 目标 | 推荐观测量 | 示例指标 | 必须报告的提取器风险 |
|---|---|---|---|
| 刚体运动 | 3D/2D 位置、姿态、速度、加速度 | trajectory RMSE、generalized trajectory error、拟合 $g$ | tracker 漂移、相机标定、深度尺度 |
| 接触与碰撞 | 接触时刻/法向、碰撞前后速度 | contact-time error、coefficient of restitution、momentum residual | 遮挡、帧率、接触点不可见 |
| 对象持续 | mask、体积、拓扑、身份 | count error、volume drift、identity survival | segmentation 错误、单视角体积歧义 |
| 软体/布料 | mesh/point trajectory、strain、self-contact | vertex/Chamfer error、strain error、penetration rate | 重建误差和拓扑对应 |
| 流体 | 速度场、体积分数、边界通量 | divergence、volume error、flow-field error | optical flow 不是流体速度；遮挡/透明材质 |
| 光学/热学/磁学 | 光路、温度代理、磁体位姿 | task-specific residual、event timing | RGB 可观察性弱，必须有专门传感或标定 |
| 反事实 | 干预前后同一测量量 | monotonicity、effect-size error、branch locality | prompt/seed 变化混入干预效应 |
| 决策 | 动作、回报、碰撞、任务结果 | policy ranking、regret、success、safety violations | planner 利用模型漏洞、sim-to-real |

三个常见“代理偷换”必须明确写出：

1. **低 jerk 只表示运动平滑，不能证明受力正确。** 匀速穿透可以有很低 jerk。
2. **低 divergence 只对应不可压缩流的一部分条件，不能证明完整 Navier--Stokes、边界条件或黏度正确。**
3. **外观特征稳定只支持身份/形状保持，不是质量守恒的直接测量。**

### 8.3 成组反事实协议

以“球落地反弹”为最小单元：

1. 固定纹理、背景、相机、文本和随机种子集合；
2. 分别扫描释放高度、重力、恢复系数和地面摩擦，每次只改变一个因素；
3. 每个条件至少生成多个种子，报告均值、标准差和失败覆盖率；
4. 测量球心轨迹、首次接触时刻、反弹高度、水平滑移和对象完整性；
5. 检查参数响应的方向、近似函数关系和置信区间，而非只检查“有没有反弹”；
6. 在未见材质、背景、尺度、相机和参数范围做 OOD；
7. 把 tracker 失败与 generator 失败分开，不把缺失测量样本静默删除。

### 8.4 闭环评测的额外要求

- 对同一初态提供 no-op 与至少两个动作，检查动作分支是否正确且只改变应改变的局部后果；
- 比较模型内策略排序与真实/高可信环境排序，而不仅是单条 rollout 像素误差；
- 测量 planner 是否利用 model exploitation 获得虚假高回报；
- 分开报告 world model、inverse dynamics、action retargeting、planner 和执行器误差；
- 报告 off-policy、OOD、长时误差累积、不确定性校准和失败恢复；
- 真实部署至少报告任务成功、碰撞/损坏、安全约束和置信区间。

## 9. 关键分歧与当前最安全的解释

### 9.1 “规模会自然产生物理”还是“必须显式注入物理”

- VideoPhy、PhyGenBench 和 VideoPhy-2 说明强视频模型在视觉进步后仍有显著物理常识失败。
- PhyWorld scaling study 在受控任务中显示 ID/组合泛化可随规模和覆盖改善，但物理 OOD 不自动闭合。
- 最安全结论：规模提供强外观与常见运动先验，显式状态、干预数据、约束或验证器仍是可测物理泛化的重要补充；现有证据不足以证明二者谁在无限规模下最终必要。

### 9.2 “生成视频是 world model”还是“只是视频预测器”

- WorldModelBench 等 L2 benchmark 能诊断 next-frame 是否可行，但不提供动作干预和决策收益。
- V-JEPA 2-AC、DreamZero、RoboWM-Bench 才开始进入 L5 及以上证据。
- 最安全结论：视频生成器可以是 world model 的视觉动力学组件；只有状态、动作、反事实、长期记忆和闭环用途被验证时，才应称决策型 world model。

### 9.3 “latent 中有物理变量”是否等于“模型懂物理”

- Interpreting Physics in Video World Models 显示物理信息在中间层可读，且呈高维分布式结构。
- 但 probe 可能读取相关特征；需要 activation intervention、属性干预和输出后果共同证明因果使用。
- 最安全结论：可读性是 representation evidence，不是 dynamics fidelity evidence。

### 9.4 “模拟器给真值”还是“真实实验给真值”

- 模拟器提供精确 state 和成组干预，适合训练和单元测试。
- GAUGE 显示不同物理引擎本身都可能偏离现实，尤其冲击接触、快速织物和体积形变。
- 最安全结论：模拟 ground truth 是对某个 solver 的真值，不自动是现实真值；核心结论应以真实标定、引擎交叉验证和不确定性共同支撑。

### 9.5 “VLM judge 已与人类一致”是否足以替代人评

- WorldModelBench、PhyGenBench、PhyGround 都展示不同程度的人类对齐。
- Physics-IQ Verified 的排名变化说明数据、prompt 与聚合细节足以改变结论；judge 还会随 API 和训练数据更新。
- 最安全结论：VLM judge 可以扩展筛查和排序，但必须冻结版本、做人类校准、允许弃权，并由程序测量/闭环证据补足。

## 10. 对仓库正文的直接写作约束

1. 首次使用时明确写出 plausibility、consistency、fidelity、decision fidelity 四级定义；不要把“物理真实感”作为未定义总称。
2. 把“质量、动量和能量不能消失”改为带系统边界的表述，区分机械能与总能量。
3. Benchmark 按证据类型组织，而非只按年份：
   - prompt/common-sense；
   - 真实视频续写；
   - 标定程序测量；
   - 定律推理；
   - 具身执行与决策。
4. 每个结果附 `benchmark version + evaluator version + prompt + frame sampler + seeds`。
5. 任何 2026 arXiv-only 工作明确写“预印本”；CVPR Workshop 与 CVPR 主会分开。
6. VLM、人评、程序指标和闭环任务分别成列，不计算一个掩盖失败的总平均。
7. 3D/4D 和 simulator 路线按 geometry → system identification → solver → renderer → measurement 拆解。
8. Action-conditioned 页区分 latent action、camera/ego control、语言事件和标定连续动作。
9. 交互世界页区分系统实时性与世界正确性：TTFF/FPS/deadline 不等于 action fidelity、state persistence 或 decision utility。
10. 对最新预印本保留日期快照，不把作者自报性能写成独立确认。

## 11. 一手 URL 注册表

### 11.1 Benchmark、数据与官方代码

- VideoPhy：[论文](https://arxiv.org/abs/2406.03520)；[官方代码与当前版本说明](https://github.com/Hritikbansal/videophy)
- PhyGenBench：[ICML/PMLR](https://proceedings.mlr.press/v267/meng25c.html)；[官方代码](https://github.com/OpenGVLab/PhyGenBench)；[项目页](https://phygenbench123.github.io/)
- VideoPhy-2：[论文](https://arxiv.org/abs/2503.06800)；[项目页](https://videophy2.github.io/)
- WorldModelBench：[论文](https://arxiv.org/abs/2502.20694)；[项目页](https://worldmodelbench-team.github.io/)；[OpenReview](https://openreview.net/forum?id=a3hafrDzuA)
- Physics-IQ：[WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html)；[官方代码](https://github.com/google-deepmind/physics-IQ-benchmark)
- Physics-IQ Verified：[论文](https://arxiv.org/abs/2606.18943)；代码沿用并更新于 [Physics-IQ 官方仓库](https://github.com/google-deepmind/physics-IQ-benchmark)
- PAI-Bench：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PAI-Bench_A_Comprehensive_Benchmark_For_Physical_AI_CVPR_2026_paper.html)
- PhysInOne：[论文](https://arxiv.org/abs/2604.09415)；[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PhysInOne_Visual_Physics_Learning_and_Reasoning_in_One_Suite_CVPR_2026_paper.html)
- Physion-Eval：[论文](https://arxiv.org/abs/2603.19607)；[官方数据](https://huggingface.co/datasets/PhysionLabs/Physion-Eval)
- PhyGround：[论文](https://arxiv.org/abs/2605.10806)；[项目页](https://phyground.github.io/)
- GAUGE：[论文](https://arxiv.org/abs/2608.05948)
- RoboWM-Bench：[论文](https://arxiv.org/abs/2604.19092)；[CVPR 2026 Workshop](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.html)；[项目页](https://robowm-bench.github.io/RoboWM-Bench/)
- Apple-PI：[论文](https://arxiv.org/abs/2607.16401)
- YoCausal：[论文](https://arxiv.org/abs/2605.30346)

### 11.2 方法、表示与闭环系统

- PhysGaussian：[论文](https://arxiv.org/abs/2311.12198)；[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html)
- PhysDreamer：[论文](https://arxiv.org/abs/2404.13026)
- DreamPhysics：[论文](https://arxiv.org/abs/2406.01476)
- Physics3D：[论文](https://arxiv.org/abs/2406.04338)
- PhysGen：[论文](https://arxiv.org/abs/2409.18964)；[项目页](https://stevenlsw.github.io/physgen/)
- PhyWorld scaling study：[论文](https://arxiv.org/abs/2411.02385)
- PhyT2V：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_PhyT2V_LLM-Guided_Iterative_Self-Refinement_for_Physics-Grounded_Text-to-Video_Generation_CVPR_2025_paper.html)
- PhysGen3D：[论文](https://arxiv.org/abs/2503.20746)；[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_PhysGen3D_Crafting_a_Miniature_Interactive_World_from_a_Single_Image_CVPR_2025_paper.html)
- PISA Experiments：[ICML/PMLR](https://proceedings.mlr.press/v267/li25bu.html)；[官方代码](https://github.com/vision-x-nyu/pisa-experiments)
- PhysFlow：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Unleashing_the_Potential_of_Multi-modal_Foundation_Models_and_Video_Diffusion_CVPR_2025_paper.html)
- V-JEPA 2：[论文](https://arxiv.org/abs/2506.09985)
- VLIPP：[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VLIPP_Towards_Physically_Plausible_Video_Generation_with_Vision_and_Language_ICCV_2025_paper.html)
- WonderPlay：[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_WonderPlay_Dynamic_3D_Scene_Generation_from_a_Single_Image_and_ICCV_2025_paper.html)
- TRACE：[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_TRACE_Learning_3D_Gaussian_Physical_Dynamics_from_Multi-view_Videos_ICCV_2025_paper.html)
- PhysCtrl：[论文](https://arxiv.org/abs/2509.20358)；[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f53fd88a4340063ecd258c0ae9948b40-Abstract-Conference.html)；[项目页](https://cwchenwang.github.io/physctrl/)
- NewtonRewards：[论文](https://arxiv.org/abs/2512.00425)
- GenieDrive：[论文](https://arxiv.org/abs/2512.12751)；[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_GenieDrive_Towards_Physics-Aware_Driving_World_Model_with_4D_Occupancy_Guided_CVPR_2026_paper.html)
- Interpreting Physics in Video World Models：[论文](https://arxiv.org/abs/2602.07050)
- DreamZero：[论文](https://arxiv.org/abs/2602.15922)
- CoECT：[论文](https://arxiv.org/abs/2603.09094)；[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Chain_of_Event-Centric_Causal_Thought_for_Physically_Plausible_Video_Generation_CVPR_2026_paper.html)
- PSIVG：[论文](https://arxiv.org/abs/2603.06408)；[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Foo_Physical_Simulator_In-the-Loop_Video_Generation_CVPR_2026_paper.html)；[项目页](https://vcai.mpi-inf.mpg.de/projects/PSIVG/)
- PHANTOM：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_PHANTOM_Physics-Infused_Video_Generation_via_Joint_Modeling_of_Visual_and_CVPR_2026_paper.html)
- PhysVideo：[论文](https://arxiv.org/abs/2603.18639)
- PhyCo：[论文](https://arxiv.org/abs/2604.28169)；[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Narayanan_PhyCo_Learning_Controllable_Physical_Priors_for_Generative_Motion_CVPR_2026_paper.html)
- Inference-time Physics Alignment：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Inference-time_Physics_Alignment_of_Video_Generative_Models_with_Latent_World_CVPR_2026_paper.html)
- Goal Force：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gillman_Goal_Force_Teaching_Video_Models_To_Accomplish_Physics-Conditioned_Goals_CVPR_2026_paper.html)
- DynamicsBoost：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.html)
- NS-Diff：[CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_NS-Diff_Fluid_Navier-Stokes_Guided_Video_Diffusion_via_Reinforcement_Learning_CVPR_2026_paper.html)
- PhyWorld 2026 model：[论文](https://arxiv.org/abs/2605.19242)
- LDR：[论文](https://arxiv.org/abs/2608.09926)；[项目页](https://lat-dyn-reason.github.io/)

## 12. 审计结论的有效期

本文件是 **2026-08-29 快照**。下列信息最容易漂移，后续引用前应重新检查：

- arXiv-only 工作是否已被会议接收、题名和数字是否随版本变化；
- VideoPhy、Physics-IQ/Verified 等官方 leaderboard 与代码默认协议；
- judge checkpoint、API 模型版本和数据发布过滤后的实际行数；
- 新增独立复现是否改变 DreamZero、GAUGE、LDR、PhyGround、Physion-Eval、Apple-PI 等预印本的解释。

最稳定的写法不是记住一个排行榜数字，而是同时记录 **论文/数据版本、输入条件、评价器版本、样本数、随机种子、测量不确定性和最高证据层**。
