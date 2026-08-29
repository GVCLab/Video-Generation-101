# 视频与 World Model 数据集索引

数据集的许可、下载方式和可用状态可能变化。这里按研究用途分类，不代表推荐将其用于商业训练。

## 1. 小型视频预测与生成

| 数据集 | 内容 | 常见用途 | 入口 |
|---|---|---|---|
| Moving MNIST [[1]](#ref-1) | 移动数字的合成序列 | ConvLSTM [[2]](#ref-2)、长期预测入门 | [Reference](https://arxiv.org/abs/1502.04681) |
| KTH Actions | 六类人体动作 | 早期预测与生成 | [Dataset](https://www.csc.kth.se/cvap/actions/) |
| BAIR Robot Pushing [[3]](#ref-3) | 机器人推动物体 | 动作条件视频预测 | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small) |
| UCF-101 [[4]](#ref-4) | 人类动作视频 | 类别条件视频生成 | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/ucf101) |

这些数据规模小、视觉分布窄，适合教学和机制验证，不适合证明开放世界生成能力。

## 2. 视频理解与时序预训练

| 数据集 | 特征 | 入口 |
|---|---|---|
| Kinetics | 大规模人类动作类别 | [Project](https://deepmind.google/research/open-source/kinetics/) |
| Something-Something | 对对象交互和动作方向敏感 | [Project](https://developer.qualcomm.com/software/ai-datasets/something-something) |
| Ego4D | 大规模第一人称日常活动 | [Project](https://ego4d-data.org/) |
| EPIC-KITCHENS | 第一人称厨房交互 | [Project](https://epic-kitchens.github.io/) |

## 3. 文本—视频生成

| 数据集 | 内容 | 注意事项 | 入口 |
|---|---|---|---|
| MSR-VTT | 视频与自然语言描述 | 规模较小，常用于检索与条件评测 | [Project](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) |
| WebVid | 大规模网页视频—文本对 | 原始数据已停止分发；仅适合历史结果与数据治理讨论 | [GitHub](https://github.com/m-bain/webvid) |
| HD-VILA-100M | 大规模高清视频—语言数据 | 适合预训练研究，下载与许可需核验 | [Project](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) |
| Panda-70M | 自动过滤和描述的大规模视频数据 | 需检查具体分发与使用条款 | [Project](https://snap-research.github.io/Panda-70M/) |

## 4. 机器人与动作条件数据

| 数据集 | 内容 | 入口 |
|---|---|---|
| Open X-Embodiment | 多机构、多机器人 embodiment 数据集合 | [Project](https://robotics-transformer-x.github.io/) |
| DROID | 大规模真实机器人操作轨迹 | [Project](https://droid-dataset.github.io/) |
| BridgeData V2 | 多任务机器人操作数据 | [Project](https://rail-berkeley.github.io/bridgedata/) |
| CALVIN | 长序列语言条件机器人任务 | [Project](https://github.com/mees/calvin) |
| RoboNet | 多机器人、多视角交互视频 | [Project](https://www.robonet.wiki/) |

动作条件数据至少需要对以下字段进行标准化：

- 相机内外参和时间戳。
- 动作频率、坐标系和单位。
- 机器人 embodiment 与关节定义。
- episode 边界、任务、成功和失败标签。
- 观测延迟和动作延迟。

## 5. 驾驶与具身环境

- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)：感知、运动和驾驶研究。
- [nuScenes](https://www.nuscenes.org/) [[5]](#ref-5)：多传感器自动驾驶数据。
- [BDD100K](https://github.com/bdd100k/bdd100k)：多样化道路视频与标注。
- [Habitat](https://aihabitat.org/)：具身导航 simulator 和数据生态。
- [Procgen](https://github.com/openai/procgen)：用于泛化研究的程序化游戏环境。

## 6. 数据治理清单

在训练或发布模型前，至少记录：

```yaml
source:
license:
consent_and_likeness:
collection_period:
geographic_distribution:
language_distribution:
resolution_and_fps:
shot_detection:
deduplication:
caption_source:
safety_filtering:
train_validation_test_split:
known_limitations:
```

## 7. 数据泄漏与评测污染

大规模网页视频模型可能见过评测视频、相似剪辑或同源帧。建议：

- 使用感知 hash、视频 embedding 和音频 fingerprint 去重。
- 不只在公开视频 benchmark 上评测。
- 构建发布日期晚于训练截止时间的新测试集。
- 使用程序生成、可精确控制的物理与反事实测试。
- 披露无法确认训练数据时的风险，而不是默认没有污染。

## 参考文献

<a id="ref-1"></a>[1] [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681). Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov. ICML. 2015.

<a id="ref-2"></a>[2] [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo. NeurIPS. 2015.

<a id="ref-3"></a>[3] [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157). Chelsea Finn, Ian Goodfellow, Sergey Levine. NeurIPS. 2016.

<a id="ref-4"></a>[4] [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402). Khurram Soomro, Amir Roshan Zamir, Mubarak Shah. arXiv preprint. 2012.

<a id="ref-5"></a>[5] [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027). Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, et al. CVPR. 2020.
