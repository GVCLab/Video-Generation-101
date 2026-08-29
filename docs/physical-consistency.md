# 物理一致性的视频生成

视频生成中的“物理一致性”不是让画面看起来足够真实，而是让对象、材料和环境在时间演化中遵守可检验的约束：物体应当持续存在，接触应当产生合理后果，运动应当与重力、惯性、碰撞、摩擦和材料属性相符，同一初始状态在不同外力或动作下还应产生正确的不同未来。

本页讨论的是 **physically consistent / physics-grounded video generation**。它与一般的时间连贯性、视频预测和 world model 相邻，但并不等同。

## 1. 最小定义

普通条件视频生成通常建模：

$$
p(x_{1:T}\mid c),
$$

其中 $c$ 可以是文本、图像或已有视频。物理一致性生成还希望输出满足某组物理约束 $\mathcal{C}$：

$$
x_{1:T}\sim p_\theta(x_{1:T}\mid c),\qquad
\mathcal{C}(x_{1:T},z_{1:T},u_{1:T})\approx 0.
$$

- $x_{1:T}$：最终可见的视频帧；
- $z_{1:T}$：几何、质量、速度、材质等显式或隐式状态；
- $u_{1:T}$：外力、相机、用户或智能体动作；
- $\mathcal{C}$：接触、动力学、守恒、材料响应或场景边界等约束。

关键难点在于：仅从 RGB 帧通常无法唯一恢复质量、摩擦系数、深度和外力；同一个开头也可能对应多个合理未来。因此，目标不是逐像素复现唯一答案，而是在条件允许的范围内生成 **物理可行、状态连续且与控制相符** 的未来。

## 2. 必须分清的四个层次

| 层次 | 核心问题 | 典型失败 | 能否证明动力学正确 |
|---|---|---|---|
| 视觉真实感 | 单帧像不像真实拍摄 | 纹理模糊、形状畸变 | 不能 |
| 时间连贯性 | 相邻帧是否平滑 | 闪烁、身份漂移、瞬移 | 不能 |
| 物理合理性 | 事件是否符合常识与规律 | 穿透、悬浮、无因反弹、液体增生 | 只能提供部分证据 |
| 可控动力学 | 改变状态、力或动作时，未来是否正确改变 | 不响应外力、反事实分支相同 | 可以更直接地检验 |

一段“杯子从桌上掉下”的视频可能运动流畅，但杯子穿过桌面；也可能落地后看似自然，却在改变地面材质或初速度时生成几乎相同的结果。前者缺少物理合理性，后者缺少可控、可验证的动力学。

## 3. 物理一致性具体包含什么

### 3.1 对象与空间状态

- 对象在遮挡前后保持身份、数量、形状和位置关系；
- 刚体不应无故变形、合并、消失或穿透；
- 相机运动不能被误当作物体运动；
- 2D 画面中的运动应与隐含的 3D 几何和深度相容。

### 3.2 运动与接触

- 无外力时速度变化应有原因；
- 重力、支撑、摩擦和阻力的方向与尺度应合理；
- 碰撞发生在正确的接触时刻和位置，并产生符合质量、速度和恢复系数的后果；
- 多对象因果链应按正确顺序传播，而不是只生成“碰撞风格”的视觉效果。

### 3.3 材料与连续介质

- 软体、布料、头发应呈现与刚度、弹性和约束相符的形变；
- 液体应保持大致体积连续，正确处理流动、飞溅和容器边界；
- 烟、火等现象需要兼顾输运、扩散、浮力和外部环境；
- 透明、反射、折射和阴影的变化要与几何、材质和光照相容。

### 3.4 守恒与因果

- 质量、动量和能量不能无缘无故产生或消失；
- 原因必须先于结果，接触点、受力对象和运动方向应匹配；
- 改变初始速度、材质、外力或动作时，结果应以可解释方式变化；
- 对不可观测属性或随机事件，模型应表达不确定性，而不是始终给出一个过度自信的未来。

## 4. 六条主要技术路线

### 4.1 从大规模视频中隐式学习

大型 T2V/I2V 模型通过预测视频分布，能够学到重力、运动和常见交互的统计规律。优点是开放域覆盖广、视觉质量高；缺点是互联网视频很少提供质量、力、摩擦和 3D 状态标签，模型容易学会“看起来像物理”的相关性，而非可干预的动力学。

这条路线适合建立强视觉先验，但规模增长本身并不能保证守恒、罕见碰撞和分布外材料行为正确。VideoPhy、PhyGenBench 与 VideoPhy-2 都观察到，较强的视频模型仍会在物理专项测试中显著失败 [[1]](#ref-1), [[2]](#ref-2), [[3]](#ref-3)。

### 4.2 物理感知的提示、规划与推理

先让 LLM/VLM 把文本条件展开为对象、阶段、轨迹和物理规则，再用这些中间条件引导视频模型。PhyT2V 通过物理规则生成与迭代自修正改善现有 T2V 模型 [[4]](#ref-4)；VLIPP 使用视觉—语言模型规划粗粒度运动，再将轨迹或变化交给视频扩散模型 [[5]](#ref-5)。

这类方法无需重训完整基础模型，适合快速增强和 OOD 提示；但语言推理只能提出约束，不能保证生成器逐帧执行正确，也难以精确处理连续接触、流体和多物体耦合。

### 4.3 轨迹、几何与结构条件

将点轨迹、光流、深度、分割、姿态、场景图或 3D 表示作为生成条件，可以把“往哪里运动”和“长什么样”部分解耦。结构控制能减少瞬移、穿透和对象漂移，也是连接视觉生成与显式模拟的常用接口。

它的上限取决于中间表示：稀疏轨迹无法表达材料形变，错误深度会把物理误差传给生成器，而满足几何约束也不等于满足力学规律。

### 4.4 显式物理模拟器与生成模型结合

先从图像估计几何、材质和物理参数，在刚体、软体或流体模拟器中计算状态，再由视频生成模型负责渲染、补全和视觉细化。PhysGen 在图像空间中组合场景理解、刚体模拟和视频扩散 [[6]](#ref-6)；PhysGen3D / MiniTwin 将单图重建为可交互的 3D 小世界，再根据速度、材质等初始条件模拟 [[7]](#ref-7)。

这条路线的优势是约束明确、可控且容易做反事实；瓶颈是单图 3D 与物性估计不适定，模拟器覆盖的规律有限，并且模拟结果与生成渲染之间仍可能不一致。

### 4.5 物理状态与视觉 latent 联合建模

模型不一定显式输出完整的质量、速度和力，也可以学习一个对动力学有用的物理 latent，并让它与视频 latent 共同演化。PHANTOM 将潜在物理状态预测直接纳入视频生成 [[8]](#ref-8)。这类方法比纯像素生成更容易保持动态规律，又比完整 3D 重建更灵活。

风险在于 latent 的语义可能不可辨认：在测试分布内表现好，不代表它真正编码了可迁移的物性。需要用属性干预、反事实 rollout 和 OOD 组合来验证。

### 4.6 物理监督、奖励优化与闭环校正

使用模拟数据提供可控物性标签，或以物理检测器、VLM 裁判、可微模拟器和守恒残差作为训练奖励。其一般形式可以写为：

$$
\mathcal{L}=\mathcal{L}_{\text{video}}
+\lambda_{s}\mathcal{L}_{\text{state}}
+\lambda_{c}\mathcal{L}_{\text{constraint}}
+\lambda_{r}\mathcal{L}_{\text{reward}}.
$$

视觉损失维持画质，状态损失监督轨迹或物性，约束损失惩罚物理违规，奖励项则鼓励最终视频通过专项评测。2026 年的 PhyCo 进一步用系统变化的摩擦、恢复系数、形变和外力数据训练可控物理先验 [[9]](#ref-9)。这一路线可扩展，但必须防止模型投机自动裁判，并检查模拟到真实的迁移。

## 5. 数据应该怎样构建

| 数据来源 | 能提供什么 | 主要缺口 |
|---|---|---|
| 互联网视频 | 外观和开放世界事件覆盖 | 缺少状态、力和反事实；剪辑与相机偏差严重 |
| 实验室真实视频 | 可校准的碰撞、材料和测量 | 场景窄、采集昂贵 |
| 物理模拟数据 | 精确状态、物性和成组干预 | sim-to-real 外观与动力学差距 |
| 游戏与机器人日志 | 动作—状态—后果和闭环数据 | embodiment 与环境分布有限 |
| 人工规则与偏好 | 可覆盖常识和开放域错误 | 标注主观，难以精确测量连续动力学 |

理想数据不是大量互不相关的漂亮视频，而是 **成组实验**：保持场景不变，系统改变质量、摩擦、初速度、外力或动作；同时保存 RGB、深度、对象 mask、轨迹、接触、物性和状态。只有这样，模型才较难靠背景和文本捷径猜答案。

## 6. 如何评测

### 6.1 先分解能力，不报一个总分

建议至少分别报告：

1. **条件遵循**：对象、数量、动作和阶段是否出现；
2. **状态连续**：身份、形状、数量和遮挡后状态是否保持；
3. **运动正确**：方向、速度、加速度和轨迹是否合理；
4. **接触与材料**：碰撞时刻、反弹、摩擦、形变和流动是否合理；
5. **守恒与因果**：质量、动量、能量和事件顺序是否违反规则；
6. **反事实响应**：改变初始条件或动作后，结果是否正确变化；
7. **视觉质量**：清晰度、闪烁、伪影和审美质量。

视觉质量应作为独立维度，而不能与物理正确性平均后掩盖失败。

### 6.2 三层证据

| 证据层 | 方法 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| L1：生成视频诊断 | VideoPhy、PhyGenBench、VideoPhy-2、人评/VLM 评审 | 开放域物理常识与可见违规 | 精确状态、可控动力学 |
| L2：状态与反事实 | 模拟 ground truth、轨迹/接触误差、参数扫描、成组干预 | 初始条件到后果的动力学关系 | 对真实开放世界的普遍泛化 |
| L3：闭环任务 | 机器人成功率、规划 regret、碰撞率、策略收益 | 模型是否对行动真正有用 | 不能由漂亮离线样例替代 |

Physics-IQ 将流体、光学、固体、磁学和热学等原理纳入专项测试 [[10]](#ref-10)。但任何自动评测器都可能受帧采样、视觉识别和语言先验影响，因此仍需人工校准，并公开“无法判断”和“所有样本都失败”的情况。

### 6.3 一个最小可复现实验

以“球落地并反弹”为例：

1. 固定场景、相机、球和提示词，只改变释放高度、重力或恢复系数；
2. 每个条件生成多个随机种子，避免用单个幸运样本下结论；
3. 跟踪球心、首次接触时刻、反弹高度和对象完整性；
4. 检查反弹高度是否随释放高度和恢复系数单调变化；
5. 同时报告轨迹误差、物理违规率、视觉质量和失败案例；
6. 用未见过的球材质、背景和参数范围测试 OOD 泛化。

比起询问“视频是否逼真”，这种参数扫描更接近对动力学模型的检验。

## 7. 常见误区

- **时间平滑等于物理正确。** 慢速穿透也可以非常平滑。
- **文本中写出物理规律，模型就会遵守。** Prompt 只能提供条件，不能保证生成执行。
- **VLM 判为合理，就不存在物理错误。** 裁判可能漏看短暂接触、数量变化和细粒度守恒。
- **模拟器参与就一定真实。** 错误的几何、材质或边界条件会得到精确但错误的结果。
- **单一参考视频是唯一真值。** 开放世界未来通常是多模态的，应评价可行域和统计规律。
- **物理 benchmark 高分等于 world model。** 决策型 world model 还需要动作条件、长期状态、反事实和闭环收益。

## 8. 与相邻方向的关系

```text
时间连贯性
    ↓ 加入对象、接触、材料和守恒约束
物理一致的视频生成
    ↓ 加入状态、动作、反事实和长期记忆
动作条件 world model
    ↓ 加入规划、策略和真实环境反馈
Physical AI 闭环系统
```

- [视频预测](tasks/video-prediction.md) 关注从历史到未来，物理一致性是其中的重要但非唯一要求。
- [动作条件预测](tasks/action-conditioned-prediction.md) 更直接检验“动作是否导致正确后果”。
- [从视频生成到 World Model](world-models.md) 讨论状态、记忆、动作、反事实和闭环证据。
- [评测指南](evaluation.md) 给出 FVD、VBench、物理 benchmark、人评与 world-model 分层协议。

## 9. 值得继续研究的问题

1. 如何从 RGB 视频中辨识不可观测的质量、摩擦、刚度和外力，并表达多解性？
2. 应该学习像素、对象状态、3D 场、物理参数，还是可规划的 latent？
3. 怎样把模拟器的可控性与基础视频模型的开放世界外观覆盖结合起来？
4. 如何让物理约束在 diffusion / flow 的整个采样轨迹中生效，而不是只做结果筛选？
5. 如何覆盖刚体之外的软体、流体、烟火、热学、电磁和复杂接触？
6. 如何建立不容易被 VLM 裁判投机、又能与人工和闭环收益一致的评测？
7. 模型何时应生成多个未来，何时应承认当前观测不足以确定结果？
8. 物理一致性的提升是否真的改善机器人规划、数据生成或科学模拟？

## 10. 建议阅读顺序

1. 用 VideoPhy 和 PhyGenBench 理解“物理合理性如何被拆成可测试问题” [[1]](#ref-1), [[2]](#ref-2)；
2. 用 PhyT2V / VLIPP 理解语言推理和运动规划的轻量增强路线 [[4]](#ref-4), [[5]](#ref-5)；
3. 用 PhysGen / PhysGen3D 理解显式模拟器与视频生成器如何分工 [[6]](#ref-6), [[7]](#ref-7)；
4. 用 PHANTOM / PhyCo 理解物理 latent、模拟监督与生成模型联合训练 [[8]](#ref-8), [[9]](#ref-9)；
5. 最后回到 [World Model 专章](world-models.md)，检查这些方法是否提供了动作、反事实和闭环证据。

## 参考文献

<a id="ref-1"></a>[1] [VideoPhy: Evaluating Physical Commonsense for Video Generation](https://arxiv.org/abs/2406.03520). Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, et al. ICLR. 2025.

<a id="ref-2"></a>[2] [Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://arxiv.org/abs/2410.05363). Fanqing Meng, Jiaqi Liao, Xinyu Tan, Quanfeng Lu, Wenqi Shao, Kaipeng Zhang, et al. ICML. 2025.

<a id="ref-3"></a>[3] [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](https://arxiv.org/abs/2503.06800). Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, Kai-Wei Chang. ICLR. 2026.

<a id="ref-4"></a>[4] [PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_PhyT2V_LLM-Guided_Iterative_Self-Refinement_for_Physics-Grounded_Text-to-Video_Generation_CVPR_2025_paper.html). Qiyao Xue, Xiangyu Yin, Boyuan Yang, Wei Gao. CVPR. 2025.

<a id="ref-5"></a>[5] [VLIPP: Towards Physically Plausible Video Generation with Vision and Language Informed Physical Prior](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VLIPP_Towards_Physically_Plausible_Video_Generation_with_Vision_and_Language_ICCV_2025_paper.html). Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, et al. ICCV. 2025.

<a id="ref-6"></a>[6] [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://arxiv.org/abs/2409.18964). Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, Shenlong Wang. ECCV. 2024.

<a id="ref-7"></a>[7] [PhysGen3D: Crafting a Miniature Interactive World from a Single Image](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_PhysGen3D_Crafting_a_Miniature_Interactive_World_from_a_Single_Image_CVPR_2025_paper.html). Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, et al. CVPR. 2025.

<a id="ref-8"></a>[8] [PHANTOM: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_PHANTOM_Physics-Infused_Video_Generation_via_Joint_Modeling_of_Visual_and_CVPR_2026_paper.html). Ying Shen, Jerry Xiong, Tianjiao Yu, Ismini Lourentzou. CVPR. 2026.

<a id="ref-9"></a>[9] [PhyCo: Learning Controllable Physical Priors for Generative Motion](https://openaccess.thecvf.com/content/CVPR2026/html/Narayanan_PhyCo_Learning_Controllable_Physical_Priors_for_Generative_Motion_CVPR_2026_paper.html). Sriram Narayanan, Ziyu Jiang, Srinivasa G. Narasimhan, Manmohan Chandraker. CVPR. 2026.

<a id="ref-10"></a>[10] [Do Generative Video Models Understand Physical Principles?](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html). Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, Robert Geirhos. WACV. 2026.
