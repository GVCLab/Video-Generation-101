# 视频生成任务地图：导航剖面、任务合同与运行协议

> 一手来源复核截至 **2026-08-30**；本轮结构纠错于 **2026-08-31** 完成。本页是任务定义与验收入口，不是模型排行榜。检索式、纳排标准和历史图像记录见[研究日志](../sources/research_20260830_task_application_taxonomy.md)。

视频任务不应被排成“无条件 → 文生视频 → 图生视频 → 编辑 → 世界模型”的单线升级链。也不能把反馈方式、是否流式、持续时长和记忆能力压成一条“交互时域”轴。这里先用导航剖面

~~~math
\mathcal N=(C,R,\Pi), \qquad \Pi=(F,\Gamma,H)
~~~

快速定位任务：

1. $C$（condition source）记录模型看到了什么；
2. $R$（source relation）记录输出相对源内容必须保留或改变什么；
3. $\Pi$（run protocol）把反馈 $F$、提交语义 $\Gamma$ 与评测跨度 $H$ 分开。

这三项只负责**导航**，不是任务的正式定义。$\mathcal N$ 是 $\mathcal T$ 的导航投影：$C$ 从输入 $I$ 中提炼，$R$ 从 $I,O,K,\Delta$ 的继承与变化关系中提炼，$\Pi$ 直接保留运行协议。正式定义仍需输入、输出、保持项、变化项、运行协议和证伪协议。AR、masked、diffusion、flow、GAN 或混合系统属于实现机制，不能反向替代任务定义。

## 1. 从导航剖面进入任务合同

~~~mermaid
flowchart TB
    accTitle: 从三项导航剖面进入正式任务合同
    accDescr: 条件来源、源内容关系和运行协议只用于快速定位任务；正式定义还必须声明输入输出、保持变化账本和证伪证据，之后才选择实现机制。

    subgraph navigation_profile["导航剖面 N=(C,R,Π)"]
        direction LR
        condition_source["条件来源 C<br/>语义 · 观测 · 参考 · 控制"]
        source_relation["源内容关系 R<br/>创作 · 锚定 · 变换 · 补全等"]
        run_protocol["运行协议 Π<br/>反馈 F · 提交 Γ · 跨度 H"]
    end

    condition_source --> task_contract
    source_relation --> task_contract
    run_protocol --> task_contract
    task_contract["正式合同<br/>T=(I,O,K,Δ,Π,E)"] --> falsification["硬门与证伪 E"]
    task_contract -.->|据合同选择| implementation_mechanism["实现机制 M<br/>AR · Diffusion · Flow"]

    classDef coordinate_part fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef formal_contract fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef validation_part fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef implementation_part fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764

    class condition_source,source_relation,run_protocol coordinate_part
    class task_contract formal_contract
    class falsification validation_part
    class implementation_mechanism implementation_part
~~~

**图 1：导航不是定义，机制不是任务。** 先写 $C$，再写 $R$ 和 $\Pi$；三者帮助找到相邻任务。随后用 $\mathcal T$ 写出正式合同，先确定硬门和证伪证据，再比较实现机制。

顺序化文字替代：

1. 条件来源 $C$ 说明模型接收了哪些语义、观测、参考和控制信号。
2. 源内容关系 $R$ 说明输出需要从零创作，还是参考、锚定、变换、恢复、补全、外推或重建已有信息。
3. 运行协议 $\Pi$ 分别声明反馈方式、提交语义和评测跨度。
4. 三项是六元组任务合同 $\mathcal T$ 的导航投影：$C$ 来自 $I$，$R$ 来自 $I,O,K,\Delta$，$\Pi$ 保持不变。
5. 合同规定硬门与证伪证据；AR、Diffusion、Flow 等机制在合同之后选择。

## 2. 三项导航信息分别回答什么

### 2.1 条件来源 $C$：模型究竟看到了什么

条件来源按角色记录，而不是把“无条件”和“类别条件”混为一类：

~~~math
C=C_{\mathrm{sem}}\cup C_{\mathrm{obs}}\cup C_{\mathrm{ref}}\cup C_{\mathrm{ctrl}},
\qquad z\sim p(z).
~~~

严格无条件生成要求 $C=\varnothing$。类别标签属于 $C_{\mathrm{sem}}$，因此应称“类别条件生成”。随机噪声或随机种子 $z$ 是采样变量，不是外部条件来源。

| 条件角色 | 典型信息 | 必须记录 | 不能自动推出 |
|---|---|---|---|
| $C_{\mathrm{sem}}$ 语义 | 文字、类别、脚本、结构化事件 | 词义、关系、时间顺序和语言版本 | 身份保持、几何正确或动作因果 |
| $C_{\mathrm{obs}}$ 观测 | 首帧、历史视频、退化视频、两侧端点 | 时间戳、帧率、空间标定和缺失模式 | 未见区域真实存在或未来唯一 |
| $C_{\mathrm{ref}}$ 参考 | 身份图、服装图、音色、风格、场景参考 | 参考是否占输出时间轴、授权和绑定关系 | 模型确实使用参考；需反事实消融 |
| $C_{\mathrm{ctrl}}$ 控制 | mask、相机、轨迹、姿态、深度、状态、动作 | 单位、坐标系、频率、延迟和对齐 | 控制误差小、无串扰或闭环可用 |

### 2.2 源内容关系 $R$：输出必须继承什么

为避免表内自由造词，关系标签使用下列闭集；一个任务可以取多个标签：

~~~math
R\subseteq
\{\text{创作},\text{参考},\text{锚定},\text{变换},
\text{恢复},\text{补全},\text{外推},\text{重建}\}.
~~~

| 关系 | 最小合同 | 典型错误 |
|---|---|---|
| 创作 | 没有必须逐像素继承的源时空观测 | 语义、运动、多样性或物理失败 |
| 参考 | 保留参考所定义的身份、服装、音色、风格或状态；参考不一定占输出时间轴 | 参考被忽略、错误绑定、泄漏或机械复制 |
| 锚定 | 某个已知时刻、端点或区域必须在输出中守恒 | 首帧跳变、端点不守恒、已知区被改 |
| 变换 | 源素材与输出共享对象或时间支撑，只改变指令指定的属性、区域或动作 | 过度编辑、未指定区变化、控制串扰 |
| 恢复 | 由退化观测估计同一场景的未退化信号 | 幻觉文字/身份/结构、闪烁、分布外失效 |
| 补全 | 已知支持是硬证据，缺失的空间或时间支持需要补齐 | 边界接缝、已知区泄漏、错误遮挡关系 |
| 外推 | 从真实过去向尚未观测的未来延展 | 历史断裂、未来均值化、随机性失校准 |
| 重建 | 从观测恢复可重复查询的几何、外观或动态状态 | 只生成一条路径、重投影或遮挡不一致 |

“参考、变换、恢复、补全”可以组合，但验收不能合并。例如，统一模型可共享接口，修复仍需检查退化一致性，补全仍需检查 mask 外保护 [[5]](#ref-5)。

### 2.3 运行协议 $\Pi$：反馈、提交和跨度分别是什么

~~~math
\Pi=(F,\Gamma,H).
~~~

| 字段 | 建议取值 | 必须报告 | 常见误判 |
|---|---|---|---|
| 反馈 $F$ | open-loop；closed-loop | 条件何时可更新；已提交观测是否由控制器读取并据此选择下一输入/动作；动作/观测频率 | 有动作输入或自回归生成不等于闭环；完整动作序列也可离线给定 |
| 提交 $\Gamma$ | batch；prefix-commit stream；revisable | 首次输出时间、块/帧延迟、已提交前缀能否修改、回滚与分支规则 | 流式输出不等于闭环；可预览不等于前缀已提交 |
| 跨度 $H$ | 秒、帧、镜头、步骤或回访间隔 | 测试网格、终止条件和随跨度变化的误差曲线 | 长视频不等于长期记忆；短片也可能闭环 |

上下文窗口、memory/cache、状态压缩与淘汰是实现或系统状态披露，不属于 $H$ 本身；reset、rewind、branch 和 error recovery 也应单独声明。只有“已提交的第 $t$ 步观测由控制器读取，并据此选择第 $t+1$ 步输入或动作”才构成 closed-loop。closed-loop 也不自动等于实时；只有同时报告 deadline、TTFF、p95/p99 latency 和 deadline miss，才形成部署证据。

## 3. 全仓库任务的最小合同

下列“导航关系”只帮助定位；正式实验仍须写完整 $\mathcal T$。表格按相近合同拆分，避免移动端出现无法阅读的超宽总表。

### 3.1 创作、参考与人物生成

| 任务与专章 | 输入 → 输出 | 导航关系 $R$ | 一票否决式错误 |
|---|---|---|---|
| [严格无条件生成](tasks/unconditional-video-generation.md) | $z$ → 新视频样本 | 创作 | 使用类别/文字却仍称无条件；训练样本记忆或模式坍塌 |
| [文本到视频](tasks/text-to-video.md) | 文字 → 满足主体、关系、动作和镜头的片段 | 创作 | 只出现关键词，但关系或时间顺序错误 |
| [原生音视频](tasks/native-audio-video-generation.md) | 文字，可附图像/音色参考 → 同一生成过程中的画面与声音 | 创作；有参考时另加参考 | 实际是视频后配音；事件、说话人或声源错绑 |
| [图像到视频](tasks/image-to-video.md) | 已知首帧/时刻锚点，常附文字 → 从该时刻延展的视频 | 锚定、外推 | 首帧跳变、参考布局漂移；把任意身份参考误写成 I2V |
| [开放集视频个性化](tasks/personalized-video-generation.md) | 输出时间轴外的主体参考 + 文字 → 新场景/动作视频 | 参考、创作 | 参考姿态/背景复制、多主体融合、身份泄漏 |
| [细粒度可控生成](tasks/controllable-video-generation.md) | 语义/参考 + 相机、轨迹、姿态或几何控制 → 受控视频 | 按子合同为创作、参考 + 创作或变换 | 控制被忽略、坐标误读、主体/背景串扰 |
| [多视角视频](tasks/multiview-4d-generation.md) | 图像/视频 + 同时刻多相机查询 → 一致的多视角视频 | 参考、补全 | 一条相机路径伪装成多视角；同一时刻视图不一致 |
| [可渲染 4D 状态](tasks/multiview-4d-generation.md) | 多视角时序观测 → 可按 $(v,t)$ 重复查询的动态状态 | 重建、补全 | 只有生成视频，没有显式或可查询状态；重投影/遮挡失败 |
| [故事与多镜头](tasks/story-multishot.md) | 剧本、分镜、角色/场景参考 → 有叙事关系的镜头序列 | 创作、参考 | 单镜头漂亮，但人物、道具和事件因果跨镜头断裂 |
| [参考驱动数字人](tasks/digital-human.md) | 身份参考 + 音频/文字 → 身份稳定、同步的人体视频 | 参考、创作 | 未授权身份、口型/身体错拍、多人串扰 |
| [数字人重演](tasks/digital-human.md) | 源人物视频 + 驱动音频/姿态/表演 → 指定表演变化 | 参考、变换 | 未指定身份/背景变化；驱动者身份泄漏 |

### 3.2 变换、恢复、补全与虚拟试衣

| 任务与专章 | 输入 → 输出 | 导航关系 $R$ | 一票否决式错误 |
|---|---|---|---|
| [视频到视频编辑](tasks/video-to-video.md) | 源视频 + 指令/参考/轨迹 → 指定变化后的完整视频 | 变换 | 未编辑区域变化、目标变化未实现、时间传播不一致 |
| [源视频虚拟试衣](tasks/video-virtual-try-on.md) | 人物源视频 + 服装参考/商品资产 → 同一时间轴、动作与相机下的换装视频 | 参考、变换 | 人脸/身体/背景漂移，服装纹理标识错位或跨帧闪烁 |
| [姿态驱动虚拟试衣](tasks/video-virtual-try-on.md) | 人物/身份参考 + 服装参考 + 姿态序列 → 新时间轴试衣视频 | 参考、创作 | 把该合同当作普通 V2V；身份、衣服与驱动姿态错误绑定 |
| [视频退化修复](tasks/video-restoration.md) | 低质量观测 + 可选退化参数 → 同时间轴高质量视频 | 恢复 | 幻觉文字/人脸/结构、闪烁、真实退化分布外失效 |
| [视频补全](tasks/video-inpainting.md) | 视频 + 时空 mask → 缺失区域及其时间延续 | 锚定、补全 | mask 外像素被改、边界 seam、错误重现被删对象 |
| [帧插值](tasks/frame-interpolation.md) | 两侧/多侧已知帧 + 目标时间 → 已知端点之间的帧 | 锚定、补全 | 端点不守恒、遮挡层次反转、目标时间位置错误 |

### 3.3 预测与交互

| 任务与专章 | 输入 → 输出 | 导航关系 $R$ | 一票否决式错误 |
|---|---|---|---|
| [视频预测](tasks/video-prediction.md) | 真实过去帧 + 可选上下文 → 一个或多个可能未来 | 锚定、外推 | 历史不连续；把多模态未来平均成模糊唯一答案 |
| [动作条件预测](tasks/action-conditioned-prediction.md) | 历史观测 + 完整或分段动作序列 → 对应未来观测 | 锚定、外推 | 换动作而未来不变；动作单位、坐标或延迟错位 |
| [交互式世界](tasks/interactive-world-generation.md) | 当前状态 + 每轮到达的动作 → 可持续反馈的观测与状态 | 锚定、外推 | 动作无因果效应、回访失忆、deadline miss 或错误无法恢复 |

同一底座可迁移到多个合同，并不意味着任务相同。Video Diffusion Models 同时研究无条件生成、文字条件和视频预测；Stable Video Diffusion 的底座又可适配 I2V、相机运动和多视角生成。这些是技术复用证据，不是任务合并证据 [[1]](#ref-1) [[2]](#ref-2)。

## 4. 十一组最容易混淆的边界

### 4.1 I2V 与帧插值

- I2V 的图像通常是输出时间轴上的首帧或已知时刻；任务从锚点向未知未来**外推**。
- 插值同时知道区间两侧端点；目标时刻在已知观测之间，属于时间**补全**。
- 因而插值必须检查端点守恒、遮挡显隐和精确时间位置；I2V 更强调锚点连续、合理运动和多样未来。

### 4.2 I2V 与开放集视频个性化

- I2V 的参考图占输出时间轴中的已知时刻。
- 个性化参考只定义主体，不占输出时间轴；目标是在新场景、新动作和新镜头下保持绑定。
- 首帧保真不能证明开放集身份绑定，身份相似度也不能证明首帧守恒。

### 4.3 视频编辑、退化修复与视频补全

- 编辑允许把已知内容改成另一种内容，但变化范围由指令和保持账本限制。
- 恢复把全帧低质量信号当作观测，目标是逆转 blur、downsample、noise 或 compression。
- 补全把已知支持视为硬证据，只在空间/时间缺失支持内生成。
- “同一模型都能做”只说明接口复用，不取消 fidelity、hallucination 或 outside-mask protection 等专属硬门。

### 4.4 视频预测与普通条件生成

- 预测必须接续真实过去状态；生成一个“合理视频”不等于预测给定场景的未来。
- 随机未来需分别报告覆盖、校准和固定预算 best-of-$N$；best-of-$N$ 只是搜索/覆盖 oracle，不是校准。
- 若模型含训练期 future posterior，测试必须证明部署时只从 history-conditioned prior 采样。详见[变分随机视频生成](generative-models/variational-generation.md)。

### 4.5 动作条件预测与交互世界

动作条件预测可以一次性读取完整动作序列并离线生成；交互世界要求已提交的第 $t$ 步观测由控制器读取，据此选择第 $t+1$ 步输入或动作，并在每个 deadline 前响应。2015 年 Atari 工作是动作条件预测里程碑，不等于今天的低延迟闭环系统 [[3]](#ref-3)。

### 4.6 数字人的两类源关系与交互协议

- 参考驱动生成：身份参考不占输出时间轴，模型创建新表演。
- 源视频重演：输出需继承源视频身份/场景，只替换指定表演或驱动。
- 两类源关系都可叠加交互协议；可交互数字人还需状态、轮次、延迟、打断和恢复。

两类源关系及其交互扩展都增加身份授权、音视频同步、人体结构和冒用风险。OmniHuman-1.5 的结构化语义条件推进了表演控制，但作者预印本结果不能替代长期身份、同意治理或独立复现 [[6]](#ref-6)。

### 4.7 长视频与多镜头叙事

- 延长单镜头主要测试对象存在、持续运动和误差累积。
- 多镜头允许切镜，但必须保持人物、道具、地点、事件顺序和镜头意图。
- 将独立短片拼接起来不构成多镜头叙事，除非存在跨镜头状态账本和冲突处理。

### 4.8 原生联合音视频与视频后配音

Video-to-audio 学习的是

~~~math
p(a\mid v,c),
~~~

而原生联合生成主张的是

~~~math
p(v,a\mid c).
~~~

若实现公开，应检查双向 cross-attention、共享去噪状态或跨模态递归等耦合接口；若实现是黑盒，则必须用时间打乱、事件删除和单模态反事实观察画面与声音是否双向响应。只看到最终文件“有声有画”时，应标成**产品能力未知机制**，不能自动写成联合生成。MM-Diffusion 是较早的联合音视频 diffusion 起点；Ovi 是 twin-DiT 双向交互的预印本例子 [[13]](#ref-13) [[12]](#ref-12)。

### 4.9 视觉运动控制与环境动作

- 相机 pose、轨迹、姿态、depth 或 flow 描述“画面怎样变化”，可在生成前一次给出完整序列。
- 环境动作要求输入对应状态转移；若进一步声称交互，还需等待新观测后再规划。
- 因而准确沿轨迹移动不证明动作因果，动作条件预测也未必提供可编辑摄影机。VideoComposer 展示统一控制接口，但每种控制仍需独立误差和串扰测试 [[10]](#ref-10)。

### 4.10 相机控制视频、多视角视频与可渲染 4D

- 相机控制视频在每个世界时间只选择一个相机位置，是 camera–time 平面上的一条路径。
- 多视角视频要求同一世界时间存在多个一致视图。
- 可渲染 4D 还要求可重复查询的 dynamic radiance、surface、Gaussian 或其他状态表示。
- 应分别测试 freeze-time 多视角、freeze-camera 时间推进、novel-view/novel-time、重投影、遮挡和 loop closure。

### 4.11 视频虚拟试衣与相邻任务

- **与通用 V2V 的关系**：source-video VVT 是严格 V2V 的专项，须保持源视频时间轴、动作与相机；pose-driven VVT 创建新时间轴，不属于严格 V2V。两者都需验证人物—服装双参考绑定、纹理/logo、层叠、遮挡和跨帧稳定。
- **不是泛化数字人**：数字人关注身份与表演；试衣的主变化对象是服装，人体和非服装区域是硬保持项。
- **不是单图 VTON**：视频合同额外要求服装细节随姿态、视角、遮挡和出画再入画保持一致。
- **不是 3D 服装仿真**：只有在输出包含可验证的 3D 衣物状态、材料参数或物理轨迹时，才能声称仿真；生成逼真的 2D 视频不足以证明布料物理正确。
- 源视频试衣与姿态驱动试衣应分别验收，不能用同一总分掩盖源视频保持或新时间轴生成错误。详见[视频虚拟试衣专章](tasks/video-virtual-try-on.md)。

## 5. 正式任务合同：一页纸写清“什么算成功”

建议为每个实验写六元组：

~~~math
\mathcal T=(I,O,K,\Delta,\Pi,E), \qquad \Pi=(F,\Gamma,H).
~~~

导航剖面由此投影得到：$C$ 从 $I$ 中提炼，$R$ 从 $I,O,K,\Delta$ 中提炼，$\Pi$ 原样保留。

- $I$：输入变量、单位、坐标系、授权和时间对齐；
- $O$：输出变量、时长、帧率、分辨率以及可查询状态；
- $K$：必须保持的不变量，例如身份、mask 外像素、服装细节或世界状态；
- $\Delta$：允许或要求发生的变化；
- $\Pi$：反馈方式、提交语义与评测跨度；
- $E$：能证伪主张的协议网格、阈值、硬门和统计报告。

~~~mermaid
flowchart TB
    accTitle: 从任务合同到证伪结论
    accDescr: 先定义输入输出和保持变化账本，再用多随机种子的单变量干预覆盖评测协议网格，最后由不可补偿的硬门决定发布报告或失败回滚。

    define_io["定义 I / O<br/>输入、对齐、输出"] --> define_ledgers["划分 K / Δ<br/>保持项与变化项"]
    define_ledgers --> run_interventions["多 seed<br/>单变量干预"]
    run_interventions --> build_protocol_grid["评测协议网格 E<br/>条件 × seed × 难度"]
    build_protocol_grid --> check_hard_gates{"硬门全部通过？<br/>保持 · 安全 · 因果"}
    check_hard_gates -->|是| publish_report(["发布通过报告"])
    check_hard_gates -->|否| rollback_failure["记录失败并回滚"]
    rollback_failure -.->|修订后重测| define_io

    classDef contract_part fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef validation_step fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef success_result fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef failure_result fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class define_io,define_ledgers contract_part
    class run_interventions,build_protocol_grid,check_hard_gates validation_step
    class publish_report success_result
    class rollback_failure failure_result
~~~

**图 2：平均分不能冲销硬失败。** 顺序为：定义输入输出；划分保持项和变化项；执行多 seed 单变量干预；覆盖条件、随机种子和难度组成的协议网格；全部硬门通过才发布，否则记录失败、修订并重测。

### 5.1 用向量报告，用硬门判定

先分开报告请求变化、保持误差和时间误差：

~~~math
\mathbf m=(m_{\Delta},e_K,e_t).
~~~

验收采用不可补偿的合取，而不是任意加权总分：

~~~math
\operatorname{PASS}
=
(m_{\Delta}\ge \tau_{\Delta})
\land(e_K\le \tau_K)
\land(e_t\le \tau_t)
\land G_{\mathrm{safety}}
\land G_{\mathrm{system}}.
~~~

阈值必须按任务、数据切片和使用风险预注册。画质或变化成功率上升，不能抵消身份泄漏、mask 外变化、服装标识篡改、动作无效、deadline miss 等硬失败。

### 5.2 成对、多随机种子的反事实条件测试

固定样本、随机种子和其他输入，只改变一个条件；在预先声明的 seed 集上重复，并报告成对差值分布与置信区间：

- 改动作，未来状态是否按动作改变；
- 改音频，口型/节奏是否改变而身份保持；
- 移动 mask，变化是否随 mask 移动；
- 改相机轨迹，主体动作是否没有被错误改写；
- 交换人物或服装参考，绑定是否准确交换；
- 删除参考，相关保持指标是否显著下降。

若输出几乎不变，条件可能被忽略；若无关属性也大幅改变，系统存在串扰。只展示一个 seed 或最佳样例不能证明可控。

## 6. 里程碑按任务合同、技术底座和证据接口分层收录

本页不因参数量、演示热度或单一分数收录里程碑。每一行明确它改变的是任务合同、可复用技术底座，还是证据接口，避免把底座论文写成所有任务的共同起点。

| 首次公开 / 正式发表 | 工作 | 所属层与实际推进 | 当时仍未解决 |
|---|---|---|---|
| 2015 / NeurIPS 2015 | Action-Conditional Video Prediction | **任务合同：**把动作作为显式控制量注入未来帧预测 [[3]](#ref-3) | Atari 域、像素误差偏置、随机未来和现实控制迁移 |
| 2022 / NeurIPS 2022 | Video Diffusion Models | **技术底座：**将视频 diffusion、图像/视频联合训练与条件采样组织进同一框架，并系统报告无条件生成、文字条件、预测及空间/时间扩展 [[1]](#ref-1) | 不是这些任务的共同起点；像素空间训练、采样成本、长时状态和任务专属协议仍分离 |
| 2022 / CVPR 2023 | MM-Diffusion | **任务合同：**以耦合的音频与视频去噪子网从噪声联合采样对齐音视频对，是本页收录的联合 A/V diffusion 早期公开起点 [[13]](#ref-13) | 证据主要来自 Landscape、AIST++ 等受控数据与无条件协议，不能外推到开放域长视频、语音身份或现代生成质量 |
| 2023 / NeurIPS 2023 | VideoComposer | **任务合同 / 条件接口：**组合文字、空间条件、运动向量和条件序列 [[10]](#ref-10) | 控制冲突、精确 3D 相机/遮挡和大规模底座迁移 |
| 2023 / —（arXiv 预印本） | Stable Video Diffusion | **技术底座 / 开放工件：**系统化“文生图预训练 → 视频预训练 → 高质量视频微调”的三阶段 LDM 路线，并发布代码与权重；可继续适配 I2V、相机运动 LoRA 和多视角生成 [[2]](#ref-2) | 不是 I2V 或相机控制合同的起点；数据来源、参考保持、checkpoint/许可边界和产品能力仍需分别核验 |
| 2023 / SIGGRAPH 2024 | MotionCtrl | **任务合同 / 条件接口：**显式分离相机姿态与对象轨迹，并发布多底座适配工件 [[11]](#ref-11) | 2D/相对控制、估计噪声、遮挡和新视角几何仍限制精度 |
| 2024 / ICML 2024 | Genie | **任务合同：**tokenizer、autoregressive dynamics 与 latent action 形成逐帧可控环境 [[4]](#ref-4) | 低分辨率域、动作语义、现实转移和部署时延 |
| 2024 / ICLR 2024 | UniSim | **任务合同：**以统一 action-in/video-out 接口组合多域数据并支持交互模拟 [[7]](#ref-7) | “universal”是接口范围，不是模拟一切；声音等能力明确缺失 |
| 2025 / ICCV 2025 | VACE | **任务合同 / 条件接口：**统一 reference、editing 与 masked editing [[5]](#ref-5) | 统一模型不保证每个合同都达到专用模型的硬约束上限 |
| 2025 / —（arXiv 预印本） | OmniHuman-1.5 | **任务合同 / 条件接口：**从低层音频节奏扩展到音频、图像与文字的语义表演 [[6]](#ref-6) | 长时身份、同意/冒用、复杂多人和独立复现 |
| 2025 / —（arXiv 预印本） | Ovi | **技术底座 / 条件接口：**同构音频与视频 twin-DiT 在各 block 通过 scaled-RoPE 与双向 cross-attention 交换时间和语义信息，并支持文字或文字+图像条件同步生成音视频 [[12]](#ref-12) | 不是联合 A/V 起点；论文仍为作者预印本，训练脚本未公开，原始 5 秒论文模型与 Ovi 1.1 的 10 秒 checkpoint 应分开记录 |
| 2025 / CoRL 2025 | DreamGen | **证据接口：**把适配后的 I2V 世界模型、合成 neural trajectories、latent action / IDM 伪动作恢复和 policy training 串成可检验链，并关联世界模型基准与下游策略表现 [[8]](#ref-8) | 属于离线合成数据与策略训练管线，不是在线 planner；视频误差、伪动作误差和 embodiment/domain shift 会级联 |
| 2026 / 技术报告 | Cosmos 3 | **技术底座：**在同一 omnimodal 家族中组合语言、图像、视频、音频和动作输入输出 [[9]](#ref-9) | 技术报告/作者榜单不等于独立闭环复现 |

DreamGen 的价值不在“生成机器人视频”本身，而在把生成、伪动作恢复和策略学习串成可检查链；下游收益必须保留数据、机器人和评测设置边界 [[8]](#ref-8)。Cosmos 3 是截至复核日的重要前沿技术报告，不应写成已经形成同行评审共识 [[9]](#ref-9)。

## 7. 每类主张对应不同证据

| 主张 | 最少要报告 | 压力测试 | 不能用什么代替 |
|---|---|---|---|
| 条件遵循 | 条件解析、关系和时间顺序的分项结果 | 同 seed 单变量干预、否定词、稀有组合 | CLIP 类平均分或最佳样例 |
| 锚点 / 参考保持 | 身份、结构、颜色、状态和首帧连续的分项置信区间 | 遮挡、旋转、出画再入画、长时漂移 | 单帧相似度 |
| 开放集个性化 | 身份/属性、prompt/运动、绑定、泄漏和适配成本 | 参考交换、新姿态/背景、多主体冲突 | 单帧人脸或 CLIP 相似度 |
| 局部编辑 / 补全 | 目标区变化 + 非目标区误差 | 移动 mask、细边界、快速运动、scene cut | 整段感知质量平均分 |
| 退化修复 | paired fidelity、时间稳定、细节与幻觉率 | 未见 kernel/codec/相机、文字、人脸、多 seed | 只报 PSNR 或无参考美学分 |
| 视频虚拟试衣 | 人体/背景保持、服装忠实、纹理/logo、遮挡和跨帧一致 | 大幅姿态、宽松/透明/长服装、出画再入画、参考交换 | 单帧服装 CLIP 或精选正面样例 |
| 原生联合音视频 | 事件 onset、说话人/声源绑定、双向条件响应和延迟 | 单模态删除、时间打乱、声画矛盾、说话人交换 | 文件有音轨或单一 AV 平均分 |
| 随机未来 | 多样性、覆盖、校准、固定预算 best-of-$N$ | 罕见事件、多分支动作、不可观测状态 | 对单一未来的 MSE |
| 多视角 / 4D | 同时刻跨视图、跨时刻状态、novel-view/time 和重投影 | freeze-time、loop closure、遮挡、重复查询 | 一条相机路径或视频美学分 |
| 动作响应 | 状态转移、动作可辨识度、反事实差异 | 无动作、反动作、无效动作、延迟扰动 | “看起来像游戏/机器人” |
| 长期状态 | 对象存在、身份、拓扑和事件账本随跨度曲线 | 回访、跨段提示、上下文淘汰、长尾 horizon | 5 秒短片质量 |
| 闭环交互 | TTFF、deadline miss、p95/p99 latency、动作率和任务成功 | 峰值负载、快速动作切换、错误恢复 | 平均 FPS |

完整统计、主观评测、世界模型证据阶梯与部署 SLO 见[评测指南](evaluation.md)。

## 8. 如何从问题进入专章

| 你的问题 | 首选入口 | 必须再排除的邻居 |
|---|---|---|
| 从文字或随机变量创作视频 | [严格无条件](tasks/unconditional-video-generation.md)、[文生视频](tasks/text-to-video.md) | 类别条件不能写成严格无条件 |
| 从一个已知时刻继续生成 | [图生视频](tasks/image-to-video.md) | [个性化](tasks/personalized-video-generation.md)、[插值](tasks/frame-interpolation.md) |
| 参考只定义主体，不占输出时间轴 | [开放集个性化](tasks/personalized-video-generation.md) | I2V、数字人 |
| 修改已有视频 | [视频编辑](tasks/video-to-video.md) | 修复、补全、试衣 |
| 给人物更换指定服装 | [视频虚拟试衣](tasks/video-virtual-try-on.md) | 普通 V2V、数字人、单图 VTON、3D 仿真 |
| 去模糊、超分、去噪或去压缩 | [视频退化修复](tasks/video-restoration.md) | mask 缺失补全 |
| 补空间/时间缺失支持 | [视频补全](tasks/video-inpainting.md)、[帧插值](tasks/frame-interpolation.md) | 恢复、未来预测 |
| 同一历史产生多个概率可信未来 | [视频预测](tasks/video-prediction.md)、[变分随机生成](generative-models/variational-generation.md) | 普通条件生成 |
| 研究动作后的未来 | [动作条件预测](tasks/action-conditioned-prediction.md) | [交互世界](tasks/interactive-world-generation.md) |
| 逐轮动作—观测反馈 | [交互世界](tasks/interactive-world-generation.md) | 离线动作序列、仅流式输出 |
| 控制相机、轨迹、姿态或几何 | [细粒度可控生成](tasks/controllable-video-generation.md) | 环境动作、多视角与 4D |
| 同时刻多相机或可查询动态状态 | [多视角与 4D](tasks/multiview-4d-generation.md) | 一条相机路径 |
| 同步生成画面和声音 | [原生音视频](tasks/native-audio-video-generation.md) | 视频后配音 |
| 生成人物表演或重演 | [数字人](tasks/digital-human.md) | 个性化、虚拟试衣 |
| 生成跨镜头叙事 | [故事与多镜头](tasks/story-multishot.md) | 延长单镜头 |

使用顺序：

1. 写出 $I,O,K,\Delta,\Pi,E$；若写不清，先不要挑模型。
2. 用第二、三节找到最近合同，再用第四节排除相邻任务。
3. 进入专章查看机制、数据、评测协议和证据等级。
4. 一个系统支持多个任务时，为每个合同分别验收，不共享一个总分。
5. 若涉及现实动作或决策，继续阅读 [World Model](world-models.md)、[物理一致性](physical-consistency.md)和[相关应用](applications.md)。

## 参考文献

<a id="ref-1"></a>[1] [Video Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. NeurIPS. 2022.

<a id="ref-2"></a>[2] [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, et al. arXiv preprint. 2023.

<a id="ref-3"></a>[3] [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://proceedings.neurips.cc/paper_files/paper/2015/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html). Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L. Lewis, Satinder Singh. NeurIPS. 2015.

<a id="ref-4"></a>[4] [Genie: Generative Interactive Environments](https://proceedings.mlr.press/v235/bruce24a.html). Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. ICML. 2024.

<a id="ref-5"></a>[5] [VACE: All-in-One Video Creation and Editing](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.html). Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, Yu Liu. ICCV. 2025.

<a id="ref-6"></a>[6] [OmniHuman-1.5: Instilling an Active Mind in Avatars via Cognitive Simulation](https://arxiv.org/abs/2508.19209). Jianwen Jiang, Weihong Zeng, Zerong Zheng, Jiaqi Yang, Chao Liang, Wang Liao, et al. arXiv preprint. 2025.

<a id="ref-7"></a>[7] [Learning Interactive Real-World Simulators](https://openreview.net/forum?id=sFyTZEqmUY). Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, et al. ICLR. 2024.

<a id="ref-8"></a>[8] [DreamGen: Unlocking Generalization in Robot Learning through Video World Models](https://proceedings.mlr.press/v305/jang25a.html). Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, et al. Proceedings of the 9th Conference on Robot Learning (CoRL), PMLR 305:5170–5194. 2025.

<a id="ref-9"></a>[9] [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800). NVIDIA et al. Technical report. 2026.

<a id="ref-10"></a>[10] [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://proceedings.neurips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html). Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, et al. NeurIPS. 2023.

<a id="ref-11"></a>[11] [MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/abs/2312.03641). Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, et al. First preprint 2023; SIGGRAPH Conference Papers. 2024. [Official project and release surface](https://wzhouxiff.github.io/projects/MotionCtrl/).

<a id="ref-12"></a>[12] [Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation](https://arxiv.org/abs/2510.01284). Chetwin Low, Weimin Wang, Calder Katyal. arXiv preprint. 2025. Official code and weights: [character-ai/Ovi](https://github.com/character-ai/Ovi).

<a id="ref-13"></a>[13] [MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation](https://openaccess.thecvf.com/content/CVPR2023/html/Ruan_MM-Diffusion_Learning_Multi-Modal_Diffusion_Models_for_Joint_Audio_and_Video_CVPR_2023_paper.html). Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, Baining Guo. CVPR. 2023. [First public preprint](https://arxiv.org/abs/2212.09478), 2022.
