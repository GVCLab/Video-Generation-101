# 视频生成评测综述：从参考保真到因果与决策证据

> **综述范围。** 本章覆盖视频生成、视频编辑、multi-shot/story video generation、Video Reasoning 与 World Model 的评价方法。主体文献检索冻结于 **2026-08-29**；多镜头叙事补充、正文引用锚点与正式 proceedings 元数据于 **2026-09-02** 复核。
>
> **方法与证据边界。** 本章属于面向方法谱系的 targeted/scoping review，而非穷尽式系统综述。时间线原则上按论文首次公开版本排序，参考文献则优先记录最终正式版本；两者年份不同时在正文或条目中显式说明。同行评审论文用于支撑相对成熟的实证结论，预印本仅按当前版本陈述，标准与官方文档用于规范性要求，行业方法学只作为操作参考。

## 摘要

视频生成没有类似分类准确率的单一充分统计量。一段视频可以逐帧清晰却几乎不动，可以符合文本主题却绑定错对象、动作或事件顺序，也可以在视觉上逼真却无法预测动作干预后的真实后果。因而，评测对象已从早期的“预测帧—参考帧”逐步扩展为“生成分布—真实分布”“prompt—多次采样—能力分解”，并进一步发展到多镜头叙事中的“故事—镜头计划—跨镜头状态”与 World Model 场景中的“初始状态—动作—后果—决策收益”。本文沿四条相互关联但并不等价的时间线回顾这一演化，归纳参考保真、无参考质量、分布距离、条件遵循、人类/学习式 judge 及任务与决策指标六类方法，比较开放生成、组合性、物理、多镜头叙事、编辑、推理和交互世界模型 benchmark 的证据边界，并提出一套可复现的统一评测原则。核心结论是：新指标并未简单淘汰旧指标；可靠评测的关键，是让能力声明、任务定义、测量工具与证据上限保持一致。

**关键词：** 视频生成；视频质量评价；FVD；条件遵循；多镜头叙事；人类评测；World Model；Video Reasoning

## 1. 问题定义：评测对象必须随能力声明变化

### 1.1 三个基本矛盾

第一，**单一参考与多模态未来之间存在矛盾**。给定同一段历史，人物左转、右转或停下都可能是合理未来；若测试集只记录其中一种，逐像素误差会把其他合理结果判错。早期视频预测研究已经指出，均方误差容易把多个可能未来平均成模糊的“条件均值” [[1]](#ref-1)，随机视频预测随后把问题转向生成分布是否覆盖合理未来 [[5]](#ref-5)。

第二，**外观真实性不等于时间、语义和物理正确性**。逐帧高质量不能保证帧序正确、身份长期一致、事件已经完成，或碰撞与材料行为符合规律。FETV 对自动指标与人工判断的比较 [[14]](#ref-14)，以及 VBench 的多维拆分 [[15]](#ref-15)，都表明总体质量分数会掩盖能力之间的显著差异。

第三，**视觉 plausibility 不等于因果与决策效用**。视频基础模型可以呈现若干看似合理的世界规律 [[28]](#ref-28)，但只有在固定初始状态、改变动作并验证后果，乃至检验规划收益时，证据才从相关性推进到干预和决策层面 [[35]](#ref-35)。

### 1.2 四个递进的评测单位

| 层级 | 基本评测单位 | 核心问题 | 典型证据 | 不能直接推出 |
|---|---|---|---|---|
| 样本层 | 单条输出；有参考时为输出—参考对 | 是否清晰、自然、连续；有参考时是否保真 | PSNR/SSIM、LPIPS、VQA、人评 | 分布覆盖、条件正确 |
| 分布层 | 真实集与生成集 | 质量与多样性是否共同接近数据分布 | FID/FVD、MMD、precision/recall | 具体 prompt 被正确执行 |
| 条件与任务层 | `condition × seed` | 对象、属性、动作、关系和事件是否遵循条件 | 原子事实核验、专项 benchmark、编辑/推理成功率 | 动作具有正确因果效应 |
| 交互与决策层 | `state × action branch × horizon × policy` | 动作后果、长 rollout 和策略排序是否可靠 | 干预差分、regret、return、闭环收益 | 跨任务、跨域的通用世界理解 |

同一模型还应按时间尺度分层：帧内结构、短期运动、跨遮挡状态、长时事件与多镜头叙事。所谓“可控”也至少包含 prompt steerability、轨迹/相机/动作可控，以及持续接收动作的 closed-loop interactivity。若不先声明测试处于哪一层，指标数值本身没有稳定含义。

## 2. 历史演化：四条并行而非单线替代的时间线

视频评测史并不是“PSNR 被 FVD 替代、FVD 又被大模型 judge 替代”的单线过程。更准确的理解，是指标、benchmark、World Model 和人类/部署评价四条时间线相互交织，各自扩大了评测对象。

| 时间线 | 早期关注 | 中期转向 | 近期前沿 |
|---|---|---|---|
| 指标 | 参考保真与主观质量 | 感知和生成分布 | 表征选择、有限样本与指标元评测 |
| Benchmark | 总体质量 | 多维能力诊断 | 组合、物理、长时、编辑、推理等专项任务 |
| World Model | 观察预测 | latent planning 与动作条件生成 | 反事实、闭环、策略排序和现实收益 |
| 人类与部署 | MOS/ACR | 学习式 judge 与 arena | 安全、来源、尾延迟、能耗和成本 |

### 2.1 指标时间线：参考保真 → 生成分布 → 指标元评测

在 full-reference 主线中，SSIM 于 2004 年把像素误差推进到结构相似性 [[3]](#ref-3)。2015—2018 年，确定性与随机视频预测揭示了“单一参考”无法覆盖多种合理未来 [[1]](#ref-1) [[5]](#ref-5)；随着 VideoGAN、MoCoGAN 等生成式视频模型出现 [[6]](#ref-6) [[7]](#ref-7)，评价单位随之由样本对扩展到样本集合，并在 2018 年形成面向视频分布的 FVD [[10]](#ref-10)。2020—2026 年的主要转折，则是从提出分数转向验证分数：有限样本偏差 [[53]](#ref-53)、内容偏置 [[12]](#ref-12)、替代表征与距离 [[11]](#ref-11)，以及受控退化元评测 [[39]](#ref-39) 依次成为研究对象。P.910 的 2026 版本在本章中是当前主观实验规范锚点，而非早期生成模型里程碑 [[2]](#ref-2)。

### 2.2 Benchmark 时间线：总体质量 → 多维画像 → 专项能力

2023 年 FETV 代表了从总体分数向细粒度 prompt 因素和人机一致性诊断的转折 [[14]](#ref-14)；2024 年 VBench、EvalCrafter、GenAI-Bench、VideoScore 与 T2VSafetyBench 将多维质量、组合性、学习式 evaluator 和安全拆成不同方向 [[15]](#ref-15) [[16]](#ref-16) [[19]](#ref-19) [[20]](#ref-20) [[26]](#ref-26)。2025 年以后，状态转移、物理、编辑、开放世界知识、人体与长时视频逐渐形成专项 benchmark [[21]](#ref-21) [[23]](#ref-23) [[41]](#ref-41) [[38]](#ref-38) [[40]](#ref-40)；到 2026 年，多镜头工作进一步把评测单位从 clip 扩展为 episode [[63]](#ref-63) [[64]](#ref-64) [[65]](#ref-65) [[66]](#ref-66)。与此同时，SLVMEval 把 evaluator 本身变成被测对象 [[39]](#ref-39)。各路线的方法差异与证据边界在第 4 节展开。

### 2.3 World Model 时间线：观察预测 → 规划充分性 → 交互与决策证据

这条脉络早期由观察预测展开，并首先暴露唯一参考与随机未来的冲突 [[1]](#ref-1) [[5]](#ref-5)。2019—2025 年，PlaNet、MuZero 与 DreamerV3 形成另一条以规划回报和数据效率验证 latent model 的路线 [[29]](#ref-29) [[30]](#ref-30) [[31]](#ref-31)。2024 年后，Sora 与 Genie 将大规模视频生成、可交互生成环境和 world-simulator 叙事重新连接 [[28]](#ref-28) [[33]](#ref-33)；WorldModelBench、V-JEPA 2 与 WorldMark 又依次加入视觉物理诊断、动作条件预测、机器人规划和交互响应证据 [[32]](#ref-32) [[34]](#ref-34) [[45]](#ref-45)。因此，近期争论的核心不再是名称，而是声明究竟停留在观察相关性，还是已经通过反事实、闭环与决策实验 [[35]](#ref-35)。

### 2.4 人类与部署时间线：主观锚点 → 学习式 judge → 可信运行

主观实验始终是质量与偏好的校准锚点，当前可依 P.910 组织受控呈现 [[2]](#ref-2)。VideoScore 代表把多维人工反馈蒸馏为可扩展 evaluator 的路线 [[20]](#ref-20)，但 2023—2025 年的 judge 研究也暴露了位置、长度、自偏好和任务外推等可靠性问题 [[61]](#ref-61) [[60]](#ref-60)。2024 年后，安全 [[26]](#ref-26)、水印鲁棒性 [[54]](#ref-54) [[55]](#ref-55)、来源凭证 [[27]](#ref-27) 与功耗/商业 API 运行方法 [[58]](#ref-58) [[59]](#ref-59) 进入验收范围，评价对象由模型输出扩展到完整生成系统。

## 3. 方法谱系：不同指标究竟测量什么

### 3.1 有参考保真与多模态未来

有参考指标回答“输出与指定参考有多接近”。PSNR 对像素误差敏感，SSIM 衡量局部结构相似性 [[3]](#ref-3)，VMAF 预测特定失真与观看条件下的主观质量 [[4]](#ref-4)，LPIPS 则在深层特征空间比较感知差异 [[13]](#ref-13)。它们适合重建、[视频退化修复](tasks/video-restoration.md)、插帧、带参考编辑和近确定性预测；在开放式生成中，合理但不与参考配准的结果也可能获得低分。

对于随机未来，常见的 best-of-$N$ 定义为：

```math
d_{\mathrm{best}\text{-}\mathrm{of}\text{-}N}
=\min_{i\in\{1,\ldots,N\}}d\!\left(\hat{x}^{(i)}_{1:T},x_{1:T}\right).
```

它只证明 $N$ 次采样中至少有一次接近记录未来，且会随采样预算增加而改善。随机视频预测因此应同时报告单次、样本平均、best-of-$N$、样本间多样性、非法模式率和计算成本，而不能只报最优样本 [[5]](#ref-5)。

### 3.2 无参考样本质量

无参考方法回答“单条视频自身是否有可见问题”。清晰度、闪烁、检测置信度、身份相似度和 DOVER 一类 VQA 可诊断技术质量与审美质量 [[17]](#ref-17)，但不能说明模型是否覆盖真实分布，也不能单独判断 prompt、事件或物理是否正确。因此，无参考质量分适合作为缺陷筛查器或分项指标，不适合作为开放生成的唯一总分。

### 3.3 分布指标：从 FID/FVD 到表征与距离的联合设计

IS 利用分类器的条件输出和类别边际分布衡量可辨识度与类别多样性 [[8]](#ref-8)，但不比较真实数据、难以刻画类内覆盖。FID 在图像特征上比较真实与生成分布的均值和协方差 [[9]](#ref-9)；FVD 将同一思想移到 I3D 视频特征 [[10]](#ref-10)。FID/FVD 的共同形式可写为：

```math
D_F=\lVert\mu_r-\mu_g\rVert_2^2+
\operatorname{Tr}\!\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
```

这类指标可扩展、适合在同一数据集和固定实现下做消融。其中，FID/FVD 这类 Fréchet 指标还假设所选表征与目标能力相关、均值和协方差足以支持所需比较，且有限样本估计足够稳定；IS 与 MMD 的假设并不相同。FID/IS 的有限样本偏差 [[53]](#ref-53)、FVD 的内容偏置 [[12]](#ref-12)，以及 JEDi 对 JEPA 表征和 MMD 的替代尝试 [[11]](#ref-11)，都说明绝对数值不能脱离 backbone、权重、预处理、样本数、片段长度、FPS、分辨率和重复次数比较。

因此，分布评价至少应把 fidelity 与 coverage 分开解释，并辅以 precision/recall、置信区间和按 prompt 类别的切片。跨论文抄取 FVD 排名通常不成立，除非真实集、生成预算、取帧、实现和预处理完全一致。

### 3.4 条件与事件一致性：从整体相似度到原子事实核验

CLIPScore 最初是图像—文本兼容性指标 [[18]](#ref-18)。在视频中逐帧计算再平均，能够识别主体和场景，却天然弱于数量、否定、左右关系、动作方向和事件顺序；帧均值甚至对帧置换不敏感。GenAI-Bench 把组合条件加入 text-to-visual 评价 [[19]](#ref-19)，TC-Bench 直接检查初始状态是否转化为目标状态 [[21]](#ref-21)，T2V-CompBench 进一步细分属性绑定、空间关系、动作绑定、对象交互和计数 [[22]](#ref-22)，T2VWorldBench 则扩展到开放世界知识 [[38]](#ref-38)。

这一路线的共同方法，是把 prompt 拆成可验证谓词：实体是否存在、属性是否绑定到正确对象、关系和动作方向是否正确、事件是否完成、关键状态是否持续。检测、跟踪、深度、pose、VideoQA 与 MLLM 可以分别服务这些谓词，但每个 evaluator 仍需在人类 gold set 上校准。

### 3.5 人类评价与学习式 judge：从可扩展性到测量可靠性

人评最适合复杂语义、自然度和使用偏好，但成本高且对实验设计敏感。可靠协议应隐藏模型身份、随机交换左右位置、统一编码和播放方式，对任务分层抽样，并允许“都差”“平局”和“无法判断”。统计单位应是独立任务实例，而非帧：开放生成通常对应 prompt，多镜头生成对应 story/episode，交互任务则对应初始状态或 episode。P.910 可作为主观呈现与评分设计的基础规范 [[2]](#ref-2)。

VideoScore 说明多维人工反馈可以训练专门的自动评测器 [[20]](#ref-20)，但其分布外泛化，以及评测器同时参与模型优化时的可靠性，仍需单独验证。通用 judge 研究中的位置、长度和自偏好问题 [[61]](#ref-61)，以及视频语言模型回答裁判的可靠性研究 [[60]](#ref-60)，提示生成视频 judge 也必须经过任务内校准。SLVMEval 进一步以已知长时退化检验 evaluator 是否真的响应目标错误 [[39]](#ref-39)。因此，训练 reward、开发 evaluator、冻结验收 evaluator 和最终盲测人评应彼此隔离。

### 3.6 任务与决策指标：评价“是否完成”而非“是否像”

不同任务必须采用不同成功定义。开放生成关注质量、运动、条件、覆盖和拒绝率；多镜头叙事还要分开镜头内生成、切镜边界、跨镜头实体/状态和剧情依赖；编辑同时测 edit success、source preservation 与时空 locality；个性化应分离身份、运动、绑定、时间漂移和参考泄漏；Video Reasoning 必须同时检查最终答案与中间状态合法性；声称动作或决策能力的 World Model 还要检验动作后果、反事实、闭环和策略价值，观察预测器则按其较低层声明评价。任务指标可以使用成功率、过程违规率、planning regret、real-environment return 等，它们不能被 FVD 或总体人类偏好替代。

## 4. Benchmark 谱系：按能力问题而非单一排行榜组织

### 4.1 开放域生成：从总体质量到可诊断维度

FETV 在 2023 年较早系统组织开放域 prompt 并比较自动指标与人工判断 [[14]](#ref-14)；VBench 将质量和语义拆成 16 个维度 [[15]](#ref-15)，EvalCrafter 以多指标与用户意见映射扩大了评价覆盖 [[16]](#ref-16)。后续 VBench++ 扩展任务与 trustworthiness [[36]](#ref-36)，VBench-2.0 转向 intrinsic faithfulness [[37]](#ref-37)。这些工作共同推动了“一个总分”向“能力画像”的转变，但维度名称并不自动保证测量有效：每一维仍继承 CLIP、DINO、RAFT、检测器、VQA 或 MLLM backbone 的盲点。

### 4.2 组合性、状态转移与世界知识

组合评价沿着“概念共现—正确绑定—状态变化”的方向演进。GenAI-Bench 强调组合式 text-to-visual 对齐 [[19]](#ref-19)，T2V-CompBench 对七类组合能力做细粒度诊断 [[22]](#ref-22)，TC-Bench 要求起始与结束状态构成正确转移 [[21]](#ref-21)，T2VWorldBench 则测试开放世界知识而非简单关键词存在 [[38]](#ref-38)。

### 4.3 物理、人体与长时一致性

物理评价也从常识判断走向规律和动作专项测试。VideoPhy 同时评价语义与物理常识 [[23]](#ref-23)，PhyGenBench 用 160 个 prompt 覆盖 27 条物理规律 [[24]](#ref-24)，VideoPhy-2 扩展到更困难的动作中心场景 [[25]](#ref-25)，Physics-IQ 覆盖流体、光学、固体、磁学和热学等现象 [[44]](#ref-44)。这些 benchmark 可以证明被测视频在特定可见规律上的表现，却不能证明改变动作会产生正确反事实，也不能证明模型可用于规划。

人体和长时视频暴露了平均分难以捕获的局部错误。HuM-Eval/HuM-Bench 把整体视觉判断与 2D pose、3D motion 细评结合 [[40]](#ref-40)；SLVMEval 通过受控插入长时退化，检查 evaluator 是否会漏掉短暂但关键的错误 [[39]](#ref-39)。评价长视频时，应报告事件完成、首次失败时间、身份/状态漂移和镜头级覆盖，而不能只增加均匀采样帧数。

### 4.4 Multi-shot / Story Video Generation：从 clip 升级为 episode

Multi-shot video 由可定位的镜头及硬切或设计转场构成；镜头之间允许时空跳跃，但人物、场景、道具、已发生事件和电影语言仍受同一故事约束。它不同于连续长单镜头、前缀续写和只输出静态图的 story visualization。因此，最低评测单位应是 `story × shot plan × reference set × seed × generation budget`，并把四类证据分开：镜头内质量与局部动作、切镜边界、跨镜头实体/状态、叙事顺序与电影意图。逐 episode 成功率、实体再次出现间隔 $g$ 的退化曲线及实际重试预算，都不能由均匀采帧后的平均 CLIP、FVD 或美学分代替。

| 工作 | 主要评测对象 | 主要评测设计 / 新增维度 | 证据边界 |
|---|---|---|---|
| MuSS [[63]](#ref-63) | 电影式连续叙事与跨镜头主体 | 同时检查主体保持与 anti-copy-paste 捷径 | 不能替代状态因果和系统成本评价 |
| MSVBench [[64]](#ref-64) | 层级脚本与参考图条件下的多镜头生成 | 组合 LMM 与专家模型；作者协议报告与人评较高相关 | 作者报告 Spearman $\rho=0.944$；这是特定协议结果，新模型仍需再校准 |
| EntityBench [[65]](#ref-65) | 人物、物体、地点的长间隔再次出现 | 140 个 episode、2,491 个镜头、最长 48-shot gap，并使用 fidelity gate | 聚焦实体一致性，不覆盖全部剧情与电影质量 |
| PersonaShot [[66]](#ref-66) | 人物中心的叙事连续性 | 约 1,000 个片段、16 个指标，区分物理、情感与电影连续性 | 人物中心且仍为预印本，不能外推到所有实体类型 |

评测还应以删除、交换、重复镜头和改变实体回归间隔等受控扰动检查指标是否会被投机；详细的状态、记忆、重试与回滚协议见[故事与多镜头视频生成专章](tasks/story-multishot.md)。

### 4.5 编辑及相邻任务边界

视频编辑的核心是同时回答“改对了吗”和“未要求修改的内容保住了吗”。VE-Bench 将文本—编辑结果、源—编辑结果以及感知质量联合建模 [[41]](#ref-41)；FiVE-Bench 引入对象级指令、mask 和更细粒度的保持评价 [[42]](#ref-42)；IVEBench 将范围扩展到更长视频和更多指令类型 [[43]](#ref-43)。需要特别区分：`VE-Bench` 评编辑输出质量，而 2026 年无连字符的 `VEBench` 评估 MLLM 的现实视频编辑知识与操作推理，二者任务不同 [[62]](#ref-62)。

开放集个性化应以 `subject × reference set × prompt/control × seed × adaptation budget` 为单位，把身份、运动、绑定、参考复制和成本分开；详见[开放集视频个性化](tasks/personalized-video-generation.md)。多视角/4D 则需区分 `seen/novel view × seen/novel time`，分别检查重投影、几何、遮挡、轨迹和未见区域不确定性；详见[多视角与 4D 生成](tasks/multiview-4d-generation.md)。这两类任务都不适合直接继承开放式 T2V 的总体分数。

### 4.6 Video Reasoning：结果、过程与预算三条证据线

2025—2026 年的工作并非都把生成过程当作忠实、可检验的内部推理轨迹。按评测对象和证据类型，可以区分四条路线：

| 路线 | 代表工作 | 主要证据 | 证据上限 |
|---|---|---|---|
| 广域零样本诊断 | MME-CoF [[46]](#ref-46)、Gen-ViRe [[48]](#ref-48) | 对空间、几何、物理、时序或规划任务采用带细则的 VLM judge；静态任务看目标输出，动态任务可看完整视频 | 支持特定 prompt、采样预算和裁判下的能力画像，不证明每帧都是忠实内部推理，也不证明通用因果世界模型 |
| 分层任务与终点/过程验证 | TiViBench [[47]](#ref-47)、V-ReasonBench [[49]](#ref-49) | 视觉、跟踪和 VLM 的混合评分；后者以末帧 `pass@k` 为主 | “终点正确”仍可能伴随“过程错误”，而 `pass@k` 还包含多次采样预算 |
| 程序化可验证基础设施 | VBVR [[50]](#ref-50)、VBVR-Pro [[52]](#ref-52) | 任务特定的确定性 scorer，并以人评校准；后续工作还将 scorer 用于训练和干预实验 | 支持所覆盖的受控任务与训练机制，不能外推为开放世界通用推理；VBVR-Pro 仍是新预印本 |
| 动作条件世界模拟与规划 | World Reasoning Arena [[51]](#ref-51) | 综合 VLM judge、运动/一致性指标、人评和系统级规划成功率 | 反映特定数据与 planner–world-model 协议下的系统表现，不能归因为基础模型独立推理能力，也不等于真实环境收益 |

因此，最小协议应分开初始约束保持、最终答案、可观察中间状态的合法性，以及 `pass@1`、`pass@k` 和 best-of-$k$ 对应的预算。能用程序验证时应优先使用确定性 scorer；依赖 VLM judge 时则需公开 rubric、校准集与弃权规则，而不能让同类模型既生成又充当唯一裁判。

### 4.7 安全、水印与来源

安全评价至少分成行为安全、AI 生成检测、水印和来源凭证四层。T2VSafetyBench 测试有害请求及安全—可用性权衡 [[26]](#ref-26)；检测器需要在固定低 FPR 下报告对未见生成器、重压缩和域外真实视频的 TPR。VideoMarkBench 明确把水印置于删除与伪造、白盒与黑盒等攻击条件下 [[54]](#ref-54)，SIGMark 则验证特定生成内水印方法在其攻击套件下的提取能力 [[55]](#ref-55)。

C2PA 2.4 提供的是签名来源声明和内容绑定，不是 deepfake 分类器，也不保证声明所描述的事件真实 [[27]](#ref-27)。正式报告应分别记录 manifest 是否存在、密码学验证是否通过、签名者是否受信以及处理链是否完整；具体版本入口应指向官方 2.4 索引 [[56]](#ref-56)，AI/ML 动作与来源类型的实施解释应遵循官方 guidance [[57]](#ref-57)。

## 5. World Model：从视觉诊断到决策效用的独立框架

### 5.1 概念边界

文献中至少有三类系统被称为 World Model：观察历史后预测未来画面的**观察预测器**；建模 $p(o_{t+1},r_{t+1}\mid o_{\le t},a_{\le t})$ 的**动作条件环境模型**；以及不追求像素重建、而预测规划所需 latent transition、reward、value 或 policy 的**决策型模型**。Sora 一类技术报告主要提供生成与视觉规律证据 [[28]](#ref-28)，PlaNet、MuZero 和 DreamerV3 则以规划或环境回报验证决策充分性 [[29]](#ref-29) [[30]](#ref-30) [[31]](#ref-31)。三者不能共享同一证据标准。

### 5.2 证据阶梯

| 证据层 | 核心问题 | 推荐实验 | 可支持的声明 |
|---|---|---|---|
| L0–L2：视觉诊断 | 画面、时间、语义与可见物理是否合理 | VQA/FVD、人评、事实与物理 benchmark | 高质量生成、常见规律拟合 |
| L3–L4：动作与反事实 | 给定动作后状态是否正确；换动作是否只改变应变因素 | 动作对齐、no-op、配对干预、branch consistency | 局部动作条件与反事实证据 |
| L5：闭环 rollout | 连续交互中状态、记忆和响应是否稳定 | horizon 曲线、失败时间、任务成功率、回环 | 可交互模拟能力 |
| L6–L7：决策与现实效用 | 模型能否正确排序策略并改善真实表现 | policy ranking、regret、return gap、optimization lift | 面向特定任务的决策价值 |

WorldModelBench 主要位于视觉与物理诊断层 [[32]](#ref-32)；Genie 推进到 latent-action 条件下的可交互生成 [[33]](#ref-33)，但这不等同于已具备统一语义动作空间。V-JEPA 2 把动作预测与机器人规划纳入同一证据链 [[34]](#ref-34)，WorldMark 则系统化了交互响应测试 [[45]](#ref-45)。决策中心框架强调，World Model 的声明上限不应高于其反事实、闭环和独立环境验证所在层级 [[35]](#ref-35)。

### 5.3 四类核心实验

**一步预测与自由 rollout 应分开。** Teacher forcing 主要测局部拟合，自由 rollout 才暴露误差累积。应报告状态误差、对象存活、身份保持、reward/value 误差和首次不可恢复失败随 horizon 的曲线。

**动作遵循应采用配对干预。** 固定初始状态和随机因素，只改变动作或使用 no-op，再比较目标状态差分。分别判断两条视频“看起来合理”无法隔离动作的因果效应。

**状态持久性与不确定性应独立检查。** “移动物体—遮挡—执行无关动作—重新观察”可测试记忆与回环；随机环境则应报告 NLL/Brier、校准曲线、rare-mode recall 和 spurious-mode rate，而不是要求模型命中唯一未来。

**对声称服务决策的 World Model，最高层判据是策略价值。** 对候选策略比较模型预测回报 $\hat J(\pi)$ 与真实环境或独立 simulator 回报 $J(\pi)$ 的排序、regret 和 return gap。若 planner 只在 learned simulator 中找到高回报、在真实环境中失败，就构成 model exploitation；更好的视觉分数无法消除这一风险 [[35]](#ref-35)。观察预测器若不作决策声明，则不必承担这一证据要求。

## 6. 统一且可复现的评测协议

![视频生成评测证据链：先声明能力与任务，再选择自动指标、人类校准、任务测试及安全和部署证据；World Model 若作动作或决策声明，还需动作干预与闭环决策。](../assets/diagrams/video-evaluation-evidence-chain.png)

**图 1：从能力声明到证据上限的评测链。** 自动指标负责规模化扫描，人类 gold set 负责校准，任务或闭环实验负责验证实际效用；三者结果不应被强行平均成一个缺乏解释的总分。

<a id="capability-evaluation-matrix"></a>

### 6.1 Claim–task–metric 对齐

评测前应先写 model claim card：模型面向 T2V、I2V、multi-shot/story、编辑、个性化、长视频、联合音频、Video Reasoning，还是 action-conditioned World Model；每项声明需要明确成功条件、失败代价和证据等级。下表列出代表性任务路由，而非穷尽所有子任务。“能力主张”中的编号均反链[基础模型能力地图](foundation-model-capabilities.md#capability-cross-table-index)；它们决定需要验证什么，不代表某个 benchmark 已经充分覆盖该能力。

| 任务 | 能力主张 | 最低评测单位 | 必须分开的结果 |
|---|---|---|---|
| 开放生成 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C7](foundation-model-capabilities.md#capability-c7) | `prompt × seed` | 外观、运动、条件、覆盖、拒绝/失败 |
| I2V / 修复 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | `input × condition/degradation × seed` | 输入保持、目标变化、参考保真、时间稳定；作语义或物理声明时另验 [C2](foundation-model-capabilities.md#capability-c2) / [C7](foundation-model-capabilities.md#capability-c7) |
| Multi-shot / story | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) · [C7](foundation-model-capabilities.md#capability-c7) | `story × shot plan × reference set × seed × budget` | 镜头内质量、边界、实体/状态、叙事/电影语言、错误传播与成本 |
| 编辑 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | `source × instruction × seed` | edit success、source preservation、locality、时间稳定 |
| 个性化 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | `subject × reference × prompt × seed × budget` | 身份、运动、绑定、漂移、泄漏、适配成本 |
| 多视角 / 4D | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) | `scene × view × time × seed` | 新视角/新时刻、几何、遮挡、轨迹、不确定性 |
| 联合音视频 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C6](foundation-model-capabilities.md#capability-c6) · [C7](foundation-model-capabilities.md#capability-c7) | `condition × seed × audiovisual event` | 音频与视频质量、事件正确、同步、跨模态一致性 |
| Video Reasoning | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C7](foundation-model-capabilities.md#capability-c7) · [C8](foundation-model-capabilities.md#capability-c8) | `problem × seed × budget` | 约束保持、最终答案、过程合法性、pass@1/pass@k |
| World Model | [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C7](foundation-model-capabilities.md#capability-c7) · [C9](foundation-model-capabilities.md#capability-c9)；明示推理或规划时再加 [C8](foundation-model-capabilities.md#capability-c8) | `state × action branch × horizon × policy` | 动作效应、反事实、rollout、校准、策略价值 |

### 6.2 冻结数据、版本与生成预算

所有模型应使用相同任务/场景集、每个独立任务实例（如 prompt、source、story 或 state）的样本数、最大重试次数和尽可能一致的时长、FPS、分辨率及音频设置。必须记录 checkpoint 或 API 版本、访问日期、seed、采样器、步数、guidance、prompt 重写、插帧/超分和安全过滤。成功、拒绝、超时、损坏文件和重试都应进入分母，避免高拒绝模型只靠筛选后的样本获得高分。

Benchmark 也需要版本化：保存所用论文/项目页/leaderboard 版本、prompt 集、scorer、evaluator checkpoint、judge prompt、帧采样、代码 commit 和访问时间。还要区分两类命名风险：VBench++ 与 VBench-2.0 名称相似但范围不同 [[36]](#ref-36) [[37]](#ref-37)；VE-Bench 与 VEBench 则是名称近似但任务不同 [[41]](#ref-41) [[62]](#ref-62)。因此，报告必须给出精确版本与任务定义，不能只写 benchmark 名称。

### 6.3 自动指标必须先接受元评测

对正确视频分别构造帧乱序、冻结/重复、速度改变、关键片段删除、颜色/数量交换、动作反转、短暂身份漂移和 codec 重压缩。对 multi-shot/story 还应交换/重复完整镜头、移动切点、改变转场、重置不可逆状态，并分别控制实体回归间隔与干扰镜头数。若指标声称测某种能力，其分数应随相应破坏强度单调恶化，同时对无关变化相对稳定。FVD 的内容偏置研究 [[12]](#ref-12) 与 SLVMEval 的长时退化实验 [[39]](#ref-39) 都说明，应先证明测量仪器有效，再用它给模型排序。

### 6.4 人评与统计不确定性

人类评测应遵循受控呈现和盲测原则 [[2]](#ref-2)。成对比较通常比绝对 1—10 分更稳定，但需要随机左右位置、允许平局/都差/无法判断，并说明 tie policy。模型胜率和自动指标应以独立任务实例为聚类单位给出 bootstrap 置信区间：开放生成用 prompt，multi-shot/story 用 story 或 episode，编辑用 source–instruction 对，交互任务用初始状态或 episode；同一标注者反复评价时，可用混合效应模型控制 annotator 与任务难度。作为可复现性要求，本章建议 Arena 结果另行报告用户来源、时间窗口、模型版本和匹配调度；已有 LLM-as-a-judge 研究可直接支持的是位置、长度和自偏好等裁判偏差 [[61]](#ref-61)，不应被扩写成对所有视频 Arena 协议的实证结论。

### 6.5 分项报告与 Pareto frontier

视觉质量、动态程度、条件遵循、多样性、安全、速度和成本之间存在真实权衡。除非权重在评测前确定并做敏感性分析，否则不应制造一个总分；更可解释的方式是发布各维结果、置信区间、失败类型，以及质量—速度—能耗—成本 Pareto frontier。

系统指标必须按实际使用方式定义。离线批量应报告吞吐、NFE、峰值显存、成功率和每个合格视频成本；交互系统应报告 time-to-first-frame/chunk、控制到可见响应、p50/p95/p99、jitter、deadline miss 和可持续 horizon；商业 API 应覆盖上传、排队、推理、编码和下载。功耗测量需与 workload window 同步，可参考 MLPerf Power [[58]](#ref-58)；行业 API 方法 [[59]](#ref-59) 可用于操作性参考，但不能替代同行评审或统一硬件复测。

### 6.6 安全与来源作为正式验收项

行为安全同时报告攻击成功率和正常请求误拒率 [[26]](#ref-26)；检测与水印报告 `TPR@固定FPR`、误报、载荷、bit error、删除与伪造鲁棒性，并覆盖跨生成器和重压缩条件 [[54]](#ref-54) [[55]](#ref-55)。C2PA 验证可记录 manifest presence、cryptographic validity 与 trusted signer；本章另建议以 provenance completeness 汇总所需处理链是否齐备，但它不是 C2PA 官方 validation status。来源凭证回答声明是否与资产可靠绑定，不回答内容语义是否真实 [[27]](#ref-27) [[56]](#ref-56) [[57]](#ref-57)。

## 7. 未来发展方向

### 7.1 从相关性指标走向因果与决策证据

物理 plausibility benchmark 仍主要评价可见结果。下一阶段需要成组动作干预、反事实分支、闭环任务和独立环境回报，把“看起来会模拟”推进为“改变动作时预测正确，并能改善决策”。WorldMark 的动作响应测试 [[45]](#ref-45) 和决策中心框架 [[35]](#ref-35) 已提供起点，但跨场景动作标准化、策略可利用性和现实迁移仍是开放问题。

### 7.2 从评模型走向 evaluator science

未来 evaluator 应像模型一样被版本化、校准和红队测试。关键方向包括受控破坏的单调性、对未见生成器和长视频瞬态错误的敏感性、对抗优化后的稳健性、概率校准，以及允许 abstain 的 coverage–risk 评价。SLVMEval [[39]](#ref-39) 和视频理解回答的 judge 可靠性研究 [[60]](#ref-60) 表明，“更大的 judge”本身不是终点；后者不是生成视频质量评价的直接实验。

### 7.3 从平均帧分数走向长时事件、多镜头叙事与随机未来

长单镜头需要事件完成率、首次失败时间和跨遮挡状态，且 evaluator 必须对短暂退化保持敏感 [[39]](#ref-39)；多镜头故事还需要显式切点、实体回归间隔、不可逆状态、镜头依赖、叙事覆盖和 anti-copy 诊断 [[63]](#ref-63) [[64]](#ref-64) [[65]](#ref-65) [[66]](#ref-66)；随机未来则需要概率评分、置信区间、稀有模式召回和非法模式率 [[5]](#ref-5)。三者的共同目标，是避免平均分掩盖短暂关键错误、跨切状态重置或低概率灾难性分支。

### 7.4 从静态 benchmark 走向动态治理

公开 prompt 容易被训练或后训练吸收，API 模型与 judge 也可能静默更新。未来 benchmark 应同时维护公开开发集、私有测试集和持续刷新的 challenge set，发布 prompt/scorer hash、evaluator checkpoint、访问日期与污染声明。结果应被解释为特定版本和时间窗口的测量，而非永久能力标签。

### 7.5 从共享总分走向可组合的任务接口

开放生成、多镜头叙事、编辑、个性化、4D、Video Reasoning 和 World Model 可以共享版本、预算、统计与失败报告接口，但不应共享一个万能成功定义。现有 episode 级评测 [[63]](#ref-63) [[64]](#ref-64) [[65]](#ref-65) [[66]](#ref-66)、可验证 Video Reasoning [[52]](#ref-52) 与细粒度编辑 benchmark [[41]](#ref-41) [[42]](#ref-42) [[43]](#ref-43) 已显示模块化趋势。尚未解决的是：如何为各模块定义稳定接口、如何在不丢失失败模式的前提下聚合不确定性，以及如何阻止一个方向的高分补偿另一个方向的致命失败。

### 7.6 从模型质量走向可信部署

未来的正式报告应同时覆盖安全、权利、检测、水印、来源、尾延迟、能耗和成本。VideoMarkBench 与 SIGMark 代表水印鲁棒性的不同路线 [[54]](#ref-54) [[55]](#ref-55)，C2PA 2.4 则提供来源与处理链的标准化表达 [[27]](#ref-27)。仍待解决的问题包括跨生成器攻击模型、权利声明与技术来源凭证的衔接，以及跨硬件能耗结果的可复现性；安全过滤不证明来源，水印不证明事件真实，签名来源也不替代生成质量与物理评价。

## 8. 结论

视频生成评测的演进，本质上是评测单位和证据上限不断扩大的过程：从参考帧保真，到生成分布，再到条件、故事 episode、任务和交互决策。PSNR/SSIM 适合参考保真，LPIPS 和无参考 VQA 适合感知诊断，FID/FVD 适合固定协议下的分布比较，多维 benchmark 适合定位开放域能力；multi-shot/story 评测必须把镜头内质量、切镜边界、跨镜头状态和叙事依赖分开，编辑与 Video Reasoning benchmark 评价各自的任务完成。World Model 若声称服务决策，则还必须以动作干预、长时 rollout、策略排序和独立环境收益验证其最高层能力。

因此，最可信的评测不是找到一个能压缩所有差异的总分，而是建立一条可审计的证据链：先声明模型要解决什么问题，再为每项声明选择难以被投机的指标、人工校准和任务实验；最后同时报告不确定性、失败样本、安全、来源与运行成本。只有当方法的测量对象、适用边界和引用证据都被明确写出，模型之间的比较才具有学术意义和工程价值。

## 参考文献

<a id="ref-1"></a>[1] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. ICLR. 2016.

<a id="ref-2"></a>[2] [P.910: Subjective video quality assessment methods for multimedia applications](https://www.itu.int/rec/T-REC-P.910-202607-P/en). ITU-T. Recommendation P.910 (07/2026), in force prepublished edition. 2026.

<a id="ref-3"></a>[3] [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. IEEE Transactions on Image Processing. 2004.

<a id="ref-4"></a>[4] VMAF: Video Multi-Method Assessment Fusion [![GitHub: Netflix/vmaf](https://img.shields.io/github/stars/Netflix/vmaf?style=social)](https://github.com/Netflix/vmaf). Netflix. Official implementation and documentation.

<a id="ref-5"></a>[5] [Stochastic Video Generation with a Learned Prior](https://proceedings.mlr.press/v80/denton18a.html). Emily Denton, Rob Fergus. ICML, PMLR 80:1174–1183. 2018.

<a id="ref-6"></a>[6] [Generating Videos with Scene Dynamics](https://proceedings.neurips.cc/paper_files/paper/2016/hash/04025959b191f8f9de3f924f0940515f-Abstract.html). Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. NeurIPS. 2016.

<a id="ref-7"></a>[7] [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993). Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz. CVPR. 2018.

<a id="ref-8"></a>[8] [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498). Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. NeurIPS. 2016.

<a id="ref-9"></a>[9] [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500). Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. NeurIPS. 2017.

<a id="ref-10"></a>[10] [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717). Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, Sylvain Gelly. arXiv preprint. 2018.

<a id="ref-11"></a>[11] [Beyond FVD: An Enhanced Evaluation Metrics for Video Generation Distribution Quality](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a57483b394a3654f4317051e4ce3b2b8-Abstract-Conference.html). Ge Ya Luo, Gian M. Favero, Zhi Hao Luo, Alexia Jolicoeur-Martineau, Christopher Pal. ICLR. 2025.

<a id="ref-12"></a>[12] [On the Content Bias in Fréchet Video Distance](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html). Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, Jia-Bin Huang. CVPR. 2024.

<a id="ref-13"></a>[13] [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html). Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. CVPR. 2018.

<a id="ref-14"></a>[14] [FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c481049f7410f38e788f67c171c64ad5-Abstract-Datasets_and_Benchmarks.html). Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, et al. NeurIPS Datasets and Benchmarks. 2023.

<a id="ref-15"></a>[15] [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html). Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, et al. CVPR. 2024.

<a id="ref-16"></a>[16] [EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.html). Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, et al. CVPR. 2024.

<a id="ref-17"></a>[17] [Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives](https://arxiv.org/abs/2211.04894). Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, et al. ICCV. 2023.

<a id="ref-18"></a>[18] [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://aclanthology.org/2021.emnlp-main.595/). Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi. EMNLP. 2021.

<a id="ref-19"></a>[19] [GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation](https://arxiv.org/abs/2406.13743). Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, et al. CVPR SynData4CV Workshop. 2024.

<a id="ref-20"></a>[20] [VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation](https://aclanthology.org/2024.emnlp-main.127/). Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, et al. EMNLP. 2024.

<a id="ref-21"></a>[21] [TC-Bench: Benchmarking Temporal Compositionality in Conditional Video Generation](https://aclanthology.org/2025.findings-acl.241/). Weixi Feng, Jiachen Li, Michael Saxon, Tsu-Jui Fu, Wenhu Chen, William Yang Wang. Findings of ACL. 2025.

<a id="ref-22"></a>[22] [T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.html). Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, et al. CVPR. 2025.

<a id="ref-23"></a>[23] [VideoPhy: Evaluating Physical Commonsense for Video Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fce2d8a485746f76aac7b5650db2679d-Abstract-Conference.html). Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, et al. ICLR. 2025.

<a id="ref-24"></a>[24] [Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://proceedings.mlr.press/v267/meng25c.html). Fanqing Meng, Jiaqi Liao, Xinyu Tan, Quanfeng Lu, Wenqi Shao, Kaipeng Zhang, et al. ICML. 2025.

<a id="ref-25"></a>[25] [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c02f6a1d5c55e16db50d339dad905b4d-Abstract-Conference.html). Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, Kai-Wei Chang. ICLR. 2026.

<a id="ref-26"></a>[26] [T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/74eed5f568354c2e77dd9b018f38a9d4-Abstract-Datasets_and_Benchmarks_Track.html). Yibo Miao, Yifan Zhu, Lijia Yu, Jun Zhu, Xiao-Shan Gao, Yinpeng Dong. NeurIPS Datasets and Benchmarks. 2024.

<a id="ref-27"></a>[27] [C2PA Technical Specification, Version 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html). Coalition for Content Provenance and Authenticity. April 2026.

<a id="ref-28"></a>[28] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-29"></a>[29] [Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html). Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, et al. ICML. 2019.

<a id="ref-30"></a>[30] [Mastering Atari, Go, chess and shogi by planning with a learned model](https://www.nature.com/articles/s41586-020-03051-4). Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, et al. Nature. 2020.

<a id="ref-31"></a>[31] [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2). Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. Nature. 2025.

<a id="ref-32"></a>[32] [WorldModelBench: Judging Video Generation Models As World Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4ec03ed08a3fcb59e1c815b5598beff1-Abstract-Datasets_and_Benchmarks_Track.html). Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, et al. NeurIPS Datasets and Benchmarks. 2025.

<a id="ref-33"></a>[33] [Genie: Generative Interactive Environments](https://proceedings.mlr.press/v235/bruce24a.html). Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. ICML. 2024.

<a id="ref-34"></a>[34] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985). Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, et al. arXiv preprint. 2025.

<a id="ref-35"></a>[35] [How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position](https://arxiv.org/abs/2606.15032). Yang Yu, Shiyuan Zhang, Yifei Sheng, Haoxiang Ren, Haoxin Lin. arXiv preprint. 2026.

<a id="ref-36"></a>[36] [VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models](https://doi.org/10.1109/TPAMI.2025.3633890) ([VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models — 2024 preprint](https://arxiv.org/abs/2411.13503)). Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, et al. IEEE TPAMI, 48(3):3268–3285. 2026 (online-first 2025).

<a id="ref-37"></a>[37] [VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness](https://arxiv.org/abs/2503.21755). Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, et al. arXiv preprint. 2025.

<a id="ref-38"></a>[38] [T2VWorldBench: A Benchmark for Evaluating World Knowledge in Text-to-Video Generation](https://openaccess.thecvf.com/content/WACV2026/html/Chen_T2VWorldBench_A_Benchmark_for_Evaluating_World_Knowledge_in_Text-to-Video_Generation_WACV_2026_paper.html). Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, Jiahao Zhang. WACV. 2026.

<a id="ref-39"></a>[39] [SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Matsuda_SLVMEval_Synthetic_Meta_Evaluation_Benchmark_for_Text-to-Long_Video_Generation_CVPR_2026_paper.html). Ryosuke Matsuda, Keito Kudo, Haruto Yoshida, Nobuyuki Shimizu, Jun Suzuki. CVPR. 2026.

<a id="ref-40"></a>[40] [HuM-Eval: A Coarse-to-Fine Framework for Human-Centric Video Evaluation](https://arxiv.org/abs/2604.25361). Bingzi Zhang, Kaisi Guan, Ruihua Song. Accepted to ICME. 2026.

<a id="ref-41"></a>[41] [VE-Bench: Subjective-Aligned Benchmark Suite for Text-Driven Video Editing Quality Assessment](https://ojs.aaai.org/index.php/AAAI/article/view/32763). Shangkun Sun, Xiaoyu Liang, Songlin Fan, Wenxu Gao, Wei Gao. AAAI. 2025.

<a id="ref-42"></a>[42] [FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FiVE-Bench_A_Fine-grained_Video_Editing_Benchmark_for_Evaluating_Emerging_Diffusion_ICCV_2025_paper.html). Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, Mengyu Wang. ICCV. 2025.

<a id="ref-43"></a>[43] [IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment](https://iclr.cc/virtual/2026/poster/10007517). Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, et al. ICLR. 2026.

<a id="ref-44"></a>[44] [Do Generative Video Models Understand Physical Principles?](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html). Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, Robert Geirhos. WACV. 2026. Benchmark: Physics-IQ.

<a id="ref-45"></a>[45] [WorldMark: A Unified Benchmark Suite for Interactive Video World Models](https://arxiv.org/abs/2604.21686v2). Xiaojie Xu, Zhengyuan Lin, Kang He, Yukang Feng, Xiaofeng Mao, Yuanyang Yin, et al. arXiv v2. 2026.

<a id="ref-46"></a>[46] [Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark](https://openaccess.thecvf.com/content/CVPR2026F/html/Guo_Are_Video_Models_Ready_as_Zero-Shot_Reasoners_An_Empirical_Study_CVPRF_2026_paper.html). Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, et al. CVPR Findings. 2026.

<a id="ref-47"></a>[47] [TiViBench: Benchmarking Think-in-Video Reasoning for Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_TiViBench_Benchmarking_Think-in-Video_Reasoning_for_Video_Generation_CVPR_2026_paper.html). Harold Haodong Chen, Disen Lan, Wen-Jie Shu, Qingyang Liu, Zihan Wang, Sirui Chen, et al. CVPR. 2026.

<a id="ref-48"></a>[48] [Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark](https://arxiv.org/abs/2511.13853). Xinxin Liu, Zhaopan Xu, Ming Li, Kai Wang, Yong Jae Lee, Yuzhang Shang. arXiv preprint. 2025.

<a id="ref-49"></a>[49] [V-ReasonBench: Toward Unified Reasoning Benchmark Suite for Video Generation Models](https://arxiv.org/abs/2511.16668). Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, et al. arXiv preprint. 2025.

<a id="ref-50"></a>[50] [A Very Big Video Reasoning Suite](https://arxiv.org/abs/2602.20159). Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, et al. ICML. 2026.

<a id="ref-51"></a>[51] [World Reasoning Arena](https://arxiv.org/abs/2603.25887). PAN Team, Qiyue Gao, Kun Zhou, Jiannan Xiang, Zihan Liu, Dequan Yang, et al. arXiv preprint. 2026.

<a id="ref-52"></a>[52] [VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning](https://arxiv.org/abs/2608.26105). Junxiang Xu, Ruisi Wang, Fanyi Pu, Maijunxian Wang, Ran Ji, Tongxi Zhou, et al. arXiv preprint. 2026.

<a id="ref-53"></a>[53] [Effectively Unbiased FID and Inception Score and Where to Find Them](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html). Min Jin Chong, David Forsyth. CVPR. 2020.

<a id="ref-54"></a>[54] [VideoMarkBench: Benchmarking Robustness of Video Watermarking](https://arxiv.org/abs/2505.21620). Zhengyuan Jiang, Moyang Guo, Kecen Li, Yuepeng Hu, Yupu Wang, Zhicong Huang, et al. arXiv preprint. 2025.

<a id="ref-55"></a>[55] [SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f3f6f1739b646e0bd20111261ce23adb-Abstract-Conference.html). Xinjie Zhu, Zijing Zhao, Hui Jin, Qingxiao Guo, Yilong Ma, Yunhao Wang, et al. ICLR. 2026.

<a id="ref-56"></a>[56] [C2PA Specifications 2.4 index](https://spec.c2pa.org/specifications/specifications/2.4/index.html). Coalition for Content Provenance and Authenticity. April 2026.

<a id="ref-57"></a>[57] [C2PA Implementation Guidance, Version 2.4](https://spec.c2pa.org/specifications/specifications/2.4/guidance/Guidance.html). Coalition for Content Provenance and Authenticity. 2026.

<a id="ref-58"></a>[58] [MLPerf Inference: Power Measurement](https://docs.mlcommons.org/inference/power/). MLCommons. Official methodology, accessed 2026-08-29.

<a id="ref-59"></a>[59] [Video Model Benchmark Methodology](https://artificialanalysis.ai/video/methodology). Artificial Analysis. Industry methodology, accessed 2026-08-29.

<a id="ref-60"></a>[60] [Is Your Video Language Model a Reliable Judge?](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dc4f891373d19087d1ddda33e81e00e4-Abstract-Conference.html). Ming Liu, Wensheng Zhang. ICLR. 2025.

<a id="ref-61"></a>[61] [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). Zheng et al. NeurIPS Datasets and Benchmarks. 2023.

<a id="ref-62"></a>[62] [VEBench: Benchmarking Large Multimodal Models for Real-world Video Editing](https://openaccess.thecvf.com/content/CVPR2026F/html/Deng_VEBench_Benchmarking_Large_Multimodal_Models_for_Real-world_Video_Editing_CVPRF_2026_paper.html). Andong Deng, Dawei Du, Zhenfang Chen, Wen Zhong, Fan Chen, Guang Chen, et al. Findings of CVPR, 2187–2196. 2026.

<a id="ref-63"></a>[63] [MuSS: A Large-Scale Dataset and Cinematic Narrative Benchmark for Multi-Shot Subject-to-Video Generation](https://arxiv.org/abs/2604.23789). Haojie Zhang, Di Wu, Bingyan Liu, Linjie Zhong, Yuancheng Wei, Xingsong Ye, et al. arXiv preprint. 2026.

<a id="ref-64"></a>[64] [MSVBench: Towards Human-Level Evaluation of Multi-Shot Video Generation](https://arxiv.org/abs/2602.23969). Haoyuan Shi, Yunxin Li, Nanhao Deng, Zhenran Xu, Xinyu Chen, Longyue Wang, et al. arXiv preprint. 2026.

<a id="ref-65"></a>[65] [EntityBench: Towards Entity-Consistent Long-Range Multi-Shot Video Generation](https://arxiv.org/abs/2605.15199). Ruozhen He, Meng Wei, Ziyan Yang, Vicente Ordonez. arXiv preprint. 2026.

<a id="ref-66"></a>[66] [PersonaShot: Benchmarking Person-Centric Narrative Continuity in Multi-Shot Video Generation](https://arxiv.org/abs/2608.16717). Yuji Wang, Yuheng Chen, Teng Hu, Ran Yi, Yijia Hong, Han Feng, et al. arXiv preprint. 2026.
