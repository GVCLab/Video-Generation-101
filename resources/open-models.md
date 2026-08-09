# 开放模型与代码

本页收录适合阅读、运行或二次开发的视频生成与 world model 项目。状态截至 **2026-08**；权重、许可证和商用条件可能变化，使用前请检查各项目最新说明。

## 视频生成

| 项目 | 主要方向 | 入口 |
|---|---|---|
| HunyuanVideo | 大规模文本/图像到视频 | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo) |
| HunyuanVideo 1.5 | 更轻量的视频生成 | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) |
| Wan | 文本、图像、首尾帧和角色相关视频任务 | [GitHub](https://github.com/Wan-Video/Wan2.1) |
| CogVideo / CogVideoX | 文本和图像到视频 | [GitHub](https://github.com/THUDM/CogVideo) |
| LTX-Video | 高效 latent 视频生成 | [GitHub](https://github.com/Lightricks/LTX-Video) |
| Stable Video Diffusion | 图像到视频研究基线 | [GitHub](https://github.com/Stability-AI/generative-models) |
| Open-Sora | 开放视频生成训练与推理框架 | [GitHub](https://github.com/hpcaitech/Open-Sora) |
| Open-Sora-Plan | 开放视频生成方案与复现 | [GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan) |
| AnimateDiff | 为图像 diffusion 添加运动模块 | [GitHub](https://github.com/guoyww/AnimateDiff) |
| VideoCrafter | 开放文本/图像到视频工具箱 | [GitHub](https://github.com/AILab-CVC/VideoCrafter) |

## 视频 Tokenizer 与离散生成

| 项目 | 适合研究的问题 | 入口 |
|---|---|---|
| VideoGPT | VQ-VAE + 自回归视频 Transformer | [GitHub](https://github.com/wilson1yan/VideoGPT) |
| MAGVIT | 3D tokenizer、masked generation、多任务 | [Project](https://magvit.cs.cmu.edu/) |
| OmniTokenizer | 图像和视频统一 tokenizer | [GitHub](https://github.com/FoundationVision/OmniTokenizer) |

## World model 与 Physical AI

| 项目 | 主要方向 | 入口 |
|---|---|---|
| DreamerV3 | latent world model 与强化学习 | [GitHub](https://github.com/danijar/dreamerv3) |
| TD-MPC2 | 潜在动力学与 model predictive control | [GitHub](https://github.com/nicklashansen/tdmpc2) |
| NVIDIA Cosmos | world foundation model、tokenizer、Physical AI | [GitHub](https://github.com/NVIDIA/Cosmos) |
| V-JEPA 2 | 视频表征、物理预测、动作条件模型 | [GitHub](https://github.com/facebookresearch/vjepa2) |
| Genie 论文实现索引 | latent action 与交互环境研究入口 | [Paper](https://arxiv.org/abs/2402.15391) |

## 通用推理与训练工具

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)：提供多种视频 pipeline、scheduler 和训练示例。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：适合搭建节点式视频生成与编辑工作流。
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)：视频模型、数据加载和变换工具。
- [Decord](https://github.com/dmlc/decord)：高效视频读取。
- [FFmpeg](https://ffmpeg.org/)：切镜、转码、抽帧、音视频复用和质量检查的基础工具。

## 选择项目时检查什么

1. **任务是否匹配**：T2V、I2V、V2V、动作条件或交互式生成并不等价。
2. **许可证**：代码许可证和模型权重许可证可能不同。
3. **显存与推理时间**：避免只根据参数量判断实际成本。
4. **输出规格**：分辨率、帧数、帧率和是否经过插帧或超分。
5. **训练数据披露**：判断复现、研究和商业使用风险。
6. **评测版本**：同一项目的不同 checkpoint 不能混为一个模型。

## 不收录什么

- 没有可靠来源的“泄漏权重”或镜像。
- 仅转载宣传视频、没有项目或论文入口的模型。
- 绕过安全、肖像授权或内容来源限制的工具。
