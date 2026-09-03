# 视频生成任务地图：如何定义和区分不同任务

> 一手来源复核截至 **2026-08-30**；任务地图结构于 **2026-08-31** 复核，表述于 **2026-09-03** 更新。本页是任务定义与验收入口，不是模型排行榜。检索式、纳排标准和历史图像记录见[研究日志](../sources/research_20260830_task_application_taxonomy.md)。

视频任务不应被排成“无条件 → 文生视频 → 图生视频 → 编辑 → 世界模型”的单线升级链。也不能把反馈方式、是否流式、持续时长和记忆能力压成一条“交互时域”轴。更实用的做法是先回答三个问题：

1. 模型接收了哪些输入条件？
2. 输出与已有素材是什么关系，哪些内容必须保留？
3. 系统如何运行，以多长的时间跨度评测？

这三项用于快速定位任务。完整的任务定义还应说明输入、输出、必须保持的内容、允许变化的内容、运行与评测设置，以及验收标准。AR、masked、diffusion、flow、GAN 或混合系统属于实现方法，不能反向替代任务定义。

## 1. 先定位任务，再写清完整定义

~~~mermaid
flowchart TB
    accTitle: 从三个问题到完整任务定义
    accDescr: 输入条件、输出与已有素材的关系、运行与评测设置用于快速定位任务；完整定义还必须说明输入输出、保持项、变化项和验收标准，之后才选择实现方法并按预定标准评测。

    subgraph task_positioning["快速定位"]
        direction LR
        condition_source["输入条件<br/>语义 · 观测 · 参考 · 控制"]
        source_relation["与已有素材的关系<br/>创作 · 锚定 · 变换 · 补全等"]
        run_protocol["运行与评测设置<br/>反馈 · 输出方式 · 评测跨度"]
    end

    condition_source --> task_definition
    source_relation --> task_definition
    run_protocol --> task_definition
    task_definition["完整任务定义<br/>输入 · 输出 · 保持项 · 变化项<br/>运行与评测设置 · 验收标准"] --> implementation_mechanism["实现方法<br/>AR · Diffusion · Flow"]
    implementation_mechanism --> evaluation["按预定标准评测<br/>关键要求 · 失败条件"]

    classDef coordinate_part fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef task_definition_part fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef validation_part fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef implementation_part fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764

    class condition_source,source_relation,run_protocol coordinate_part
    class task_definition task_definition_part
    class evaluation validation_part
    class implementation_mechanism implementation_part
~~~

**图 1：实现方法不等于任务定义。** 先根据输入条件、与已有素材的关系、运行与评测设置定位任务；再写清输入、输出、保持项、变化项和验收标准，最后选择实现方法并按预定标准评测。

顺序化文字替代：

1. 输入条件说明模型接收了哪些语义、观测、参考和控制信号。
2. 与已有素材的关系说明输出需要从零创作，还是参考、锚定、变换、恢复、补全、外推或重建已有信息。
3. 运行与评测设置说明反馈方式、输出方式和评测跨度。
4. 完整任务定义还需列明输入、输出、保持项、变化项和验收标准。
5. 在任务和关键要求明确后，再选择 AR、Diffusion、Flow 等实现方法。
6. 按预定标准执行评测，检查关键要求和失败条件。

## 2. 区分任务的三个关键问题

### 2.1 输入条件：模型究竟看到了什么

输入条件应按实际作用记录，而不是把“无条件”和“类别条件”混为一类。严格无条件生成不接收任何外部条件；类别标签属于语义条件，因此应称“类别条件生成”。随机噪声或随机种子 $z$ 是采样变量，不是外部输入条件。

| 条件角色 | 典型信息 | 必须记录 | 不能自动推出 |
|---|---|---|---|
| 语义条件 | 文字、类别、脚本、结构化事件 | 词义、关系、时间顺序和语言版本 | 身份保持、几何正确或动作因果 |
| 观测条件 | 首帧、历史视频、退化视频、两侧端点 | 时间戳、帧率、空间标定和缺失模式 | 未见区域真实存在或未来唯一 |
| 参考条件 | 身份图、服装图、音色、风格、场景参考 | 参考是否占输出时间轴、授权和绑定关系 | 模型确实使用参考；需反事实消融 |
| 控制条件 | mask、相机、轨迹、姿态、深度、状态、动作 | 单位、坐标系、频率、延迟和对齐 | 控制误差小、无串扰或闭环可用 |

### 2.2 与已有素材的关系：输出必须保留什么

为了统一表述，本页使用“创作、参考、锚定、变换、恢复、补全、外推、重建”八种常见关系；一个任务可以对应多种关系。

| 关系 | 基本定义 | 典型错误 |
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

### 2.3 运行与评测设置：如何反馈、输出和确定评测跨度

| 字段 | 建议取值 | 必须报告 | 常见误判 |
|---|---|---|---|
| 反馈方式 | open-loop；closed-loop | 条件何时可更新；实际提交并返回给控制器的观测是否被读取，并据此选择下一输入/动作；动作/观测频率 | 有动作输入或自回归生成不等于闭环；完整动作序列也可离线给定 |
| 输出方式 | batch；prefix-commit stream；revisable | 首次输出时间、块/帧延迟、已提交前缀能否修改、回滚与分支规则 | 流式输出不等于闭环；可预览不等于前缀已提交 |
| 评测跨度 | 秒、帧、镜头、步骤或回访间隔 | 测试组合、终止条件和随跨度变化的误差曲线 | 长视频不等于长期记忆；短片也可能闭环 |

上下文窗口、memory/cache、状态压缩与淘汰属于实现细节或系统状态，不等同于评测跨度；reset、rewind、branch 和 error recovery 也应单独说明。只有“实际提交并返回给控制器的第 $t$ 步观测被读取，并据此选择第 $t+1$ 步输入或动作”才构成 closed-loop。closed-loop 也不自动等于实时；只有同时报告 deadline、TTFF、p95/p99 latency 和 deadline miss，才形成部署证据。

<a id="task-capability-matrix"></a>

## 3. 各类视频任务的基本定义

下列“内容关系”帮助快速定位；正式实验仍须写清完整任务定义和验收标准。表格按相近任务拆分，避免移动端出现无法阅读的超宽总表。“主要能力”栏中的编号均链接到[基础模型能力地图](foundation-model-capabilities.md#capability-cross-table-index)中的对应条目，表示该任务通常需要验证的能力，不表示某个模型已经具备这些能力。

### 3.1 创作、参考与人物生成

| 任务与专章 | 输入 → 输出 | 主要能力 | 内容关系 | 关键失败 |
|---|---|---|---|---|
| [严格无条件生成](tasks/unconditional-video-generation.md) | $z$ → 新视频样本 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C7](foundation-model-capabilities.md#capability-c7) | 创作 | 使用类别/文字却仍称无条件；训练样本记忆或模式坍塌 |
| [文本到视频](tasks/text-to-video.md) | 文字 → 满足主体、关系、动作和镜头的片段 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C7](foundation-model-capabilities.md#capability-c7) | 创作 | 只出现关键词，但关系或时间顺序错误 |
| [原生音视频](tasks/native-audio-video-generation.md) | 文字，可附图像/音色参考 → 同一生成过程中的画面与声音 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C6](foundation-model-capabilities.md#capability-c6) · [C7](foundation-model-capabilities.md#capability-c7) | 创作；有参考时另加参考 | 实际是视频后配音；事件、说话人或声源错绑 |
| [图像到视频](tasks/image-to-video.md) | 已知首帧/时刻锚点，常附文字 → 从该时刻延展的视频 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) · [C7](foundation-model-capabilities.md#capability-c7) | 锚定、外推 | 首帧跳变、参考布局漂移；把任意身份参考误写成 I2V |
| [开放集视频个性化](tasks/personalized-video-generation.md) | 输出时间轴外的主体参考 + 文字 → 新场景/动作视频 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | 参考、创作 | 参考姿态/背景复制、多主体融合、身份泄漏 |
| [细粒度可控生成](tasks/controllable-video-generation.md) | 语义/参考 + 相机、轨迹、姿态或几何控制 → 受控视频 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) · [C7](foundation-model-capabilities.md#capability-c7) | 视具体任务而定：创作、参考 + 创作，或变换 | 控制被忽略、坐标误读、主体/背景串扰 |
| [多视角视频](tasks/multiview-4d-generation.md) | 图像/视频 + 同时刻多相机查询 → 一致的多视角视频 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) | 参考、补全 | 一条相机路径伪装成多视角；同一时刻视图不一致 |
| [可渲染 4D 状态](tasks/multiview-4d-generation.md) | 多视角时序观测 → 可按 $(v,t)$ 重复查询的动态状态 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) | 重建、补全 | 只有生成视频，没有显式或可查询状态；重投影/遮挡失败 |
| [故事与多镜头](tasks/story-multishot.md) | 剧本、分镜、角色/场景参考 → 有叙事关系的镜头序列 | [C1](foundation-model-capabilities.md#capability-c1) · [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) · [C7](foundation-model-capabilities.md#capability-c7) | 创作、参考 | 单镜头漂亮，但人物、道具和事件因果跨镜头断裂 |
| [参考驱动数字人](tasks/digital-human.md) | 身份参考 + 音频/文字 → 身份稳定、同步的人体视频 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) · [C6](foundation-model-capabilities.md#capability-c6) · [C7](foundation-model-capabilities.md#capability-c7) | 参考、创作 | 未授权身份、口型/身体错拍、多人串扰 |
| [数字人重演](tasks/digital-human.md) | 源人物视频 + 驱动音频/姿态/表演 → 指定表演变化 | [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) · [C6](foundation-model-capabilities.md#capability-c6) | 参考、变换 | 未指定身份/背景变化；驱动者身份泄漏 |

### 3.2 变换、恢复、补全与虚拟试衣

| 任务与专章 | 输入 → 输出 | 主要能力 | 内容关系 | 关键失败 |
|---|---|---|---|---|
| [视频到视频编辑](tasks/video-to-video.md) | 源视频 + 指令/参考/轨迹 → 指定变化后的完整视频 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | 变换 | 未编辑区域变化、目标变化未实现、时间传播不一致 |
| [源视频虚拟试衣](tasks/video-virtual-try-on.md) | 人物源视频 + 服装参考/商品资产 → 同一时间轴、动作与相机下的换装视频 | [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) | 参考、变换 | 人脸/身体/背景漂移，服装纹理标识错位或跨帧闪烁 |
| [姿态驱动虚拟试衣](tasks/video-virtual-try-on.md) | 人物/身份参考 + 服装参考 + 姿态序列 → 新时间轴试衣视频 | [C2](foundation-model-capabilities.md#capability-c2) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) | 参考、创作 | 把该任务当作普通 V2V；身份、衣服与驱动姿态错误绑定 |
| [视频退化修复](tasks/video-restoration.md) | 低质量观测 + 可选退化参数 → 同时间轴高质量视频 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C5](foundation-model-capabilities.md#capability-c5) | 恢复 | 幻觉文字/人脸/结构、闪烁、真实退化分布外失效 |
| [视频补全](tasks/video-inpainting.md) | 视频 + 时空 mask → 缺失区域及其时间延续 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C5](foundation-model-capabilities.md#capability-c5) | 锚定、补全 | mask 外像素被改、边界 seam、错误重现被删对象 |
| [帧插值](tasks/frame-interpolation.md) | 两侧/多侧已知帧 + 目标时间 → 已知端点之间的帧 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C7](foundation-model-capabilities.md#capability-c7) | 锚定、补全 | 端点不守恒、遮挡层次反转、目标时间位置错误 |

### 3.3 预测与交互

| 任务与专章 | 输入 → 输出 | 主要能力 | 内容关系 | 关键失败 |
|---|---|---|---|---|
| [视频预测](tasks/video-prediction.md) | 真实过去帧 + 可选上下文 → 一个或多个可能未来 | [C1](foundation-model-capabilities.md#capability-c1) · [C3](foundation-model-capabilities.md#capability-c3) · [C7](foundation-model-capabilities.md#capability-c7) | 锚定、外推 | 历史不连续；把多模态未来平均成模糊唯一答案 |
| [动作条件预测](tasks/action-conditioned-prediction.md) | 历史观测 + 完整或分段动作序列 → 对应未来观测 | [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C7](foundation-model-capabilities.md#capability-c7) · [C9](foundation-model-capabilities.md#capability-c9) | 锚定、外推 | 换动作而未来不变；动作单位、坐标或延迟错位 |
| [交互式世界](tasks/interactive-world-generation.md) | 当前状态 + 每轮到达的动作 → 可持续反馈的观测与状态 | [C3](foundation-model-capabilities.md#capability-c3) · [C4](foundation-model-capabilities.md#capability-c4) · [C7](foundation-model-capabilities.md#capability-c7) · [C9](foundation-model-capabilities.md#capability-c9)；含显式规划时再加 [C8](foundation-model-capabilities.md#capability-c8) | 锚定、外推 | 动作无因果效应、回访失忆、deadline miss 或错误无法恢复 |

同一基础方法可迁移到多个任务，并不意味着这些任务相同。Video Diffusion Models 同时研究无条件生成、文字条件和视频预测；Stable Video Diffusion 的基础模型又可适配 I2V、相机运动和多视角生成。这些是技术复用证据，不是任务合并证据 [[1]](#ref-1) [[2]](#ref-2)。

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

- 编辑允许把已知内容改成另一种内容，但变化范围由指令和保持要求限制。
- 恢复把全帧低质量信号当作观测，目标是逆转 blur、downsample、noise 或 compression。
- 补全把已知支持视为硬证据，只在空间/时间缺失支持内生成。
- “同一模型都能做”只说明接口复用，不取消 fidelity、hallucination 或 outside-mask protection 等必须单独满足的要求。

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

- [长视频生成](generative-models/long-video-generation.md)先区分 fixed-long、length extrapolation 与 open-horizon；延长单镜头主要测试对象存在、持续运动、事件进展、误差累积和资源曲线。
- [故事与多镜头](tasks/story-multishot.md)允许切镜，但必须保持人物、道具、地点、事件顺序和镜头意图。
- 将独立短片拼接起来不构成多镜头叙事，除非系统能记录跨镜头状态并处理冲突。

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
- **不是单图 VTON**：视频任务还要求服装细节随姿态、视角、遮挡和出画再入画保持一致。
- **不是 3D 服装仿真**：只有在输出包含可验证的 3D 衣物状态、材料参数或物理轨迹时，才能声称仿真；生成逼真的 2D 视频不足以证明布料物理正确。
- 源视频试衣与姿态驱动试衣应分别验收，不能用同一总分掩盖源视频保持或新时间轴生成错误。详见[视频虚拟试衣专章](tasks/video-virtual-try-on.md)。

## 5. 用一页纸写清任务定义和验收标准

建议每个实验至少明确以下六项：

- **输入**：输入变量、单位、坐标系、授权和时间对齐；
- **输出**：输出变量、时长、帧率、分辨率以及可查询状态；
- **必须保持的内容**：例如身份、mask 外像素、服装细节或世界状态；
- **允许或要求的变化**：例如动作、局部属性、镜头或未来状态；
- **运行与评测设置**：反馈方式、输出方式、评测跨度和终止条件；
- **验收标准**：测试组合、阈值、必须满足的关键要求和统计报告方式。

~~~mermaid
flowchart TB
    accTitle: 从任务定义到验收结论
    accDescr: 先定义输入输出、保持项、变化项和运行方式，再预先确定测试组合与验收阈值，然后以多个随机种子做单变量测试，最后判定通过或失败。

    define_io["定义输入和输出<br/>变量 · 对齐 · 输出形式"] --> define_changes["划分保持项和变化项"]
    define_changes --> define_run["说明运行与评测设置<br/>反馈 · 输出 · 跨度"]
    define_run --> set_criteria["预先确定测试组合与验收阈值<br/>条件 × seed × 难度"]
    set_criteria --> run_interventions["多 seed<br/>单变量测试"]
    run_interventions --> check_hard_gates{"关键要求全部满足？<br/>保持 · 安全 · 因果"}
    check_hard_gates -->|是| publish_report(["发布通过报告"])
    check_hard_gates -->|否| rollback_failure["记录失败并回滚"]
    rollback_failure -.->|修订后重测| define_io

    classDef definition_part fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef validation_step fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef success_result fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef failure_result fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class define_io,define_changes,define_run definition_part
    class set_criteria,run_interventions,check_hard_gates validation_step
    class publish_report success_result
    class rollback_failure failure_result
~~~

**图 2：平均分不能掩盖关键失败。** 顺序为：定义输入输出；划分保持项和变化项；说明运行与评测设置；预先确定测试组合和验收阈值；执行多 seed 单变量测试；所有关键要求满足后才判定通过，否则记录失败、修订并重测。

### 5.1 分项报告，关键要求必须全部满足

要求变化是否实现、应保留内容的误差、时间误差、安全性和系统性能应分开报告，不要压成一个加权总分。验收前需按任务、数据切片和使用风险预先确定阈值；任一关键项不达标，整体就不应判定为通过。画质或变化成功率上升，不能抵消身份泄漏、mask 外变化、服装标识篡改、动作无效、deadline miss 等关键失败。

### 5.2 成对、多随机种子的反事实条件测试

固定样本、随机种子和其他输入，只改变一个条件；在预先声明的 seed 集上重复，并报告成对差值分布与置信区间：

- 改动作，未来状态是否按动作改变；
- 改音频，口型/节奏是否改变而身份保持；
- 移动 mask，变化是否随 mask 移动；
- 改相机轨迹，主体动作是否没有被错误改写；
- 交换人物或服装参考，绑定是否准确交换；
- 删除参考，相关保持指标是否显著下降。

若输出几乎不变，条件可能被忽略；若无关属性也大幅改变，系统存在串扰。只展示一个 seed 或最佳样例不能证明可控。

## 6. 按任务与条件接口、基础方法和证据链梳理里程碑

本页不因参数量、演示热度或单一分数收录里程碑。每一行明确它主要推进了任务与条件接口、可复用基础方法，还是证据链，避免把实现方法写成新任务，也避免把基础方法论文写成所有任务的共同起点。

| 首次公开 / 正式发表 | 工作 | 所属层与实际推进 | 当时仍未解决 |
|---|---|---|---|
| 2015 / NeurIPS 2015 | Action-Conditional Video Prediction | **任务 / 条件接口：**把动作作为显式控制量注入未来帧预测 [[3]](#ref-3) | Atari 域、像素误差偏置、随机未来和现实控制迁移 |
| 2022 / NeurIPS 2022 | Video Diffusion Models | **基础方法：**将视频 diffusion、图像/视频联合训练与条件采样组织进同一框架，并系统报告无条件生成、文字条件、预测及空间/时间扩展 [[1]](#ref-1) | 不是这些任务的共同起点；像素空间训练、采样成本、长时状态和各任务的专门评测仍分离 |
| 2022 / CVPR 2023 | MM-Diffusion | **基础方法：**以耦合的音频与视频去噪子网从噪声联合采样对齐音视频对，是本页收录的联合 A/V diffusion 早期公开起点 [[13]](#ref-13) | 证据主要来自 Landscape、AIST++ 等受控数据与无条件评测，不能外推到开放域长视频、语音身份或现代生成质量 |
| 2023 / NeurIPS 2023 | VideoComposer | **条件接口：**组合文字、空间条件、运动向量和条件序列 [[10]](#ref-10) | 控制冲突、精确 3D 相机/遮挡和大规模基础模型迁移 |
| 2023 / —（arXiv 预印本） | Stable Video Diffusion | **基础方法 / 公开代码与权重：**系统化“文生图预训练 → 视频预训练 → 高质量视频微调”的三阶段 LDM 路线，并发布代码与权重；可继续适配 I2V、相机运动 LoRA 和多视角生成 [[2]](#ref-2) | 不是 I2V 或相机控制任务的起点；数据来源、参考保持、checkpoint/许可边界和产品能力仍需分别核验 |
| 2023 / SIGGRAPH 2024 | MotionCtrl | **条件接口 / 基础方法：**显式分离相机姿态与对象轨迹，并发布多基础模型的适配代码 [[11]](#ref-11) | 2D/相对控制、估计噪声、遮挡和新视角几何仍限制精度 |
| 2024 / ICML 2024 | Genie | **基础方法 / 系统设计：**tokenizer、autoregressive dynamics 与 latent action 形成逐帧可控环境 [[4]](#ref-4) | 低分辨率域、动作语义、现实转移和部署时延 |
| 2024 / ICLR 2024 | UniSim | **任务 / 系统接口：**以统一 action-in/video-out 接口组合多域数据并支持交互模拟 [[7]](#ref-7) | “universal”是接口范围，不是模拟一切；声音等能力明确缺失 |
| 2025 / ICCV 2025 | VACE | **统一任务与条件接口：**统一 reference、editing 与 masked editing [[5]](#ref-5) | 统一模型不保证每项任务都达到专用模型的关键要求上限 |
| 2025 / —（arXiv 预印本） | OmniHuman-1.5 | **条件接口：**从低层音频节奏扩展到音频、图像与文字的语义表演 [[6]](#ref-6) | 长时身份、同意/冒用、复杂多人和独立复现 |
| 2025 / —（arXiv 预印本） | Ovi | **基础方法 / 条件接口：**同构音频与视频 twin-DiT 在各 block 通过 scaled-RoPE 与双向 cross-attention 交换时间和语义信息，并支持文字或文字+图像条件同步生成音视频 [[12]](#ref-12) | 不是联合 A/V 起点；论文仍为作者预印本，训练脚本未公开，原始 5 秒论文模型与 Ovi 1.1 的 10 秒 checkpoint 应分开记录 |
| 2025 / CoRL 2025 | DreamGen | **证据链：**把适配后的 I2V 世界模型、合成 neural trajectories、latent action / IDM 伪动作恢复和 policy training 串成可检验链，并关联世界模型基准与下游策略表现 [[8]](#ref-8) | 属于离线合成数据与策略训练管线，不是在线 planner；视频误差、伪动作误差和 embodiment/domain shift 会级联 |
| 2026 / 技术报告 | Cosmos 3 | **基础方法 / 系统设计：**在同一 omnimodal 家族中组合语言、图像、视频、音频和动作输入输出 [[9]](#ref-9) | 技术报告/作者榜单不等于独立闭环复现 |

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
| 长期状态 | 对象存在、身份、拓扑和事件状态随跨度变化的曲线 | 回访、跨段提示、上下文淘汰、长尾 horizon | 5 秒短片质量 |
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
| 生成超出原生窗口的连续长视频 | [长视频生成](generative-models/long-video-generation.md) | 流式/实时、多镜头、视频预测 |
| 生成跨镜头叙事 | [故事与多镜头](tasks/story-multishot.md) | 延长单镜头 |

使用顺序：

1. 写清输入、输出、保持项、变化项、运行与评测设置、验收标准；若写不清，先不要挑模型。
2. 用第二、三节找到最接近的任务类型，再用第四节排除相邻任务。
3. 进入专章查看机制、数据、评测协议和证据等级。
4. 一个系统支持多个任务时，应分别验收每项任务，不共享一个总分。
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
