# 视频生成与 World Model 评测指南

视频评测比图像评测更难，因为同一条件通常存在多个合理未来。一个结果可以画质很好但运动错误，也可以逐帧真实但整体故事不一致。本页提供一套分层评测框架。

## 1. 不要寻找单一总分

建议至少分开报告以下维度：

| 维度 | 要回答的问题 | 常见方法 |
|---|---|---|
| 单帧质量 | 每一帧是否清晰、自然？ | 人评、FID、LPIPS、审美模型 |
| 时间质量 | 是否闪烁、抖动或突然变形？ | FVD、temporal LPIPS、光流一致性 |
| 条件遵循 | 是否遵循文本、图像和参考素材？ | 人评、CLIP 类相似度、VLM judge |
| 运动质量 | 动作幅度、速度和方向是否正确？ | 轨迹误差、flow statistics、人评 |
| 身份与状态 | 人物、物体和场景状态是否保持？ | re-identification、tracking、状态探针 |
| 物理合理性 | 接触、重力、碰撞、材料是否合理？ | 专项 benchmark、程序化测试、人评 |
| 多样性 | 相同条件能否生成多种合理结果？ | pairwise distance、coverage、人评 |
| 安全与来源 | 是否涉及肖像、版权、误导或不可追溯内容？ | 红队测试、provenance 检查、政策评审 |

对 world model 还必须增加：动作遵循、反事实一致性、不确定性校准和规划收益。

## 2. 常见指标及其局限

### PSNR / SSIM

适合插帧、重建和未来比较确定的任务。对于开放式预测，它们会惩罚与 ground truth 不同但同样合理的未来，并鼓励平均化结果。

### LPIPS

比像素误差更接近感知相似性，但仍主要衡量成对视觉差异，不能单独判断运动和因果。

### FID

衡量生成帧和真实帧的特征分布差异，忽略时间顺序，不适合作为唯一视频指标。

### FVD

将视频映射到时空特征后比较分布，是历史上最常用的视频生成指标之一。但它对特征提取器、样本数量、预处理和实现细节敏感，也难以解释具体错误来自画质还是时间结构。

报告 FVD 时至少注明：

- 特征网络及权重版本。
- 生成和真实样本数量。
- 分辨率、帧数、帧率和裁剪方式。
- 置信区间或多次运行方差。

### CLIP / VLM-based score

可用于文本遵循和语义覆盖，但可能忽略数量、空间关系、时间顺序和细粒度动作。VLM judge 还会继承评判模型的偏差。

### VBench 等多维 benchmark

比单一 FVD 更容易定位问题，但不同维度仍可能被代理模型误判。最稳妥的方式是自动指标、专项可控测试和人工双盲比较结合。

## 3. 推荐的生成模型评测协议

### Prompt 分层

将测试条件分为：

1. 单主体、简单运动。
2. 多主体与相互作用。
3. 相机运动。
4. 遮挡和对象再次出现。
5. 材料、流体和非刚体运动。
6. 文字、数量和空间关系。
7. 长程动作与多镜头叙事。
8. 分布外组合和反常物理条件。

### 固定生成预算

- 所有模型使用相同 prompt 集合。
- 对每个 prompt 生成多个随机种子。
- 记录失败、拒绝和无法生成的样本，而不是只比较成功结果。
- 对分辨率、时长、帧率和后处理进行披露。

### 人工评测

使用 pairwise preference 通常比绝对打分稳定。问题应拆开询问：

- 哪一个更符合条件？
- 哪一个时间上更一致？
- 哪一个运动或物理更合理？
- 哪一个视觉质量更高？

不要把四个问题合并成一个含义模糊的“总体更好”。

## 4. World model 专项评测

### 4.1 动作遵循

固定初始状态，改变动作；测量目标物体、相机或机器人状态是否按动作变化。

### 4.2 反事实测试

对同一状态生成多个动作分支，例如：

```text
状态：球位于桌面边缘
动作 A：向左推
动作 B：向右推
动作 C：不接触
```

不仅检查每个分支是否合理，还要检查三者是否共享相同初始状态和不受动作影响的背景因素。

### 4.3 状态持久性

执行一次会永久改变世界的动作，例如打开抽屉、移动物体或留下痕迹；经过遮挡、转身或较长时间后再次检查状态。

### 4.4 多步 rollout

不要只报告一步预测。建议绘制指标随 rollout horizon 的变化，并记录首次发生不可恢复错误的时间。

### 4.5 不确定性校准

在随机或部分可观测环境中，检查模型生成的未来分布是否覆盖真实结果，以及置信度是否与实际错误相关。

### 4.6 闭环价值

最终测试应比较：

- 不使用 world model 的策略。
- 使用简单或传统 simulator 的策略。
- 使用 learned world model 规划的策略。
- 在真实环境或保留 simulator 中的任务成功率。

如果模型只能提高自身生成环境中的分数，却不能迁移到真实或独立环境，应警惕 model exploitation。

## 5. 失败案例分类

建议为每个失败样本标注一到多个标签：

```yaml
- identity_drift
- object_appearance_or_disappearance
- geometry_inconsistency
- temporal_flicker
- wrong_motion_direction
- contact_failure
- gravity_failure
- material_failure
- camera_control_failure
- action_ignored
- state_memory_loss
- audio_visual_desync
- text_rendering_failure
- unsafe_or_unlicensed_content
```

错误分类通常比一个聚合分数更能指导下一轮模型改进。

## 6. 最小可复现报告模板

```markdown
## Models
- checkpoint / API version:
- access date:
- inference settings:

## Data and prompts
- prompt source:
- sample count:
- preprocessing:

## Output specification
- resolution:
- fps:
- duration:
- post-processing:

## Metrics
- automated:
- human evaluation:
- confidence intervals:

## Failure analysis
- error taxonomy:
- representative cases:

## World-model tests, if applicable
- action sensitivity:
- counterfactual consistency:
- state persistence:
- planning utility:
```

## 7. 最重要的原则

评测必须与模型声称的能力一致。声称“电影级生成”，需要检查镜头、叙事和身份；声称“物理理解”，需要可控干预；声称“world model”，需要动作条件和闭环任务证据。
本页主要参考工作：Deep multi-scale video prediction beyond mean square error [[1]](#ref-1)、Video Diffusion Models [[2]](#ref-2)、Video Generation Models as World Simulators [[3]](#ref-3)、Genie: Generative Interactive Environments [[4]](#ref-4)、Dream to Control: Learning Behaviors by Latent Imagination [[5]](#ref-5)、Cosmos World Foundation Model Platform for Physical AI [[6]](#ref-6)。

## 参考文献

<a id="ref-1"></a>[1] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440). Michael Mathieu, Camille Couprie, Yann LeCun. arXiv preprint. 2015.

<a id="ref-2"></a>[2] [Video Diffusion Models](https://arxiv.org/abs/2204.03458). Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet. arXiv preprint. 2022.

<a id="ref-3"></a>[3] [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/). OpenAI. Technical report. 2024.

<a id="ref-4"></a>[4] [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. arXiv preprint. 2024.

<a id="ref-5"></a>[5] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. arXiv preprint. 2019.

<a id="ref-6"></a>[6] [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575). NVIDIA, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, et al. arXiv preprint. 2025.
