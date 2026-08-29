# 开放视频模型研究审计（2026-08-30）

> 对应教材页：[`resources/open-models.md`](../resources/open-models.md)
> 冻结时间：2026-08-30，Asia/Shanghai
> 审计目标：判断“当前官方版本实际开放了什么”，而不是制作跨协议排行榜。

## 1. 检索问题与边界

本轮围绕五个问题检索：

1. 项目在冻结日的官方入口与当前推荐版本是什么？
2. 论文、代码、权重、推理、训练、数据、评测、许可和硬件说明分别是否存在？
3. README 的计划项、社区适配和官方已发布 artifact 能否严格分开？
4. 官方显存、速度和 benchmark 的协议边界是什么？
5. 哪些项目适合作为基础生成、长视频、控制编辑、原生音视频或 world model 的研究入口？

纳入标准：官方作者仓库、官方模型卡、官方项目页、论文原文或会议页面；社区框架只用于说明可用适配层，不用于替代模型本体的官方 release。排除标准：只有二手榜单、转载、未定位权重的 demo、无法区分版本的模型合集，以及把 `coming soon` 写成已发布的条目。

## 2. 检索路线

- 官方 GitHub 仓库：README、release/commit、license、安装与硬件段落；
- Hugging Face 官方组织或模型卡：权重文件、revision、许可和任务变体；
- arXiv、作者项目页与正式会议页面：方法、版本和 venue；
- GitHub API / raw README：在网页渲染不稳定时核对默认分支和最新 commit；
- 研究框架官方仓库：只登记其支持范围，不反推上游训练开放度。

检索过程中 GitHub API 在连续查询后出现速率限制，因此后续核验改用官方仓库页面与 raw 内容。未下载任何大模型权重，也未进行 GPU 推理；文中的显存、速度和 benchmark 均明确保留为作者入口口径。

## 3. 统一判定框架

每个版本登记发布向量：

```text
R = (paper, code, weights, inference, training,
     data_recipe, evaluation, license, hardware)
```

并按以下证据强度标记：

- R0：只有论文或项目页；
- R1：有可检查代码，但缺关键权重或运行入口；
- R2：代码、权重、推理、许可和硬件边界足以做固定版本运行；
- R3：R2 加可执行微调/训练入口与数据格式；
- R4：进一步开放数据配方、评测和固定环境，可审计训练。

该等级只描述发布完整度，不代表视觉质量。对许可证使用交集原则：代码、权重、adapter、数据、上游依赖和内容政策任一项更严格时，以交集为准。

## 4. 关键版本更正

| 项目 | 本轮更正 | 一手证据 |
|---|---|---|
| MiniMax H3 | H3-Base FL2VA / Ref2VA 可得；H3-Context-IR、2K regenerate 和稀疏注意力不能提前算作开放 | [官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)、[官方模型组织](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| LTX | 当前推荐线为 LTX-2.5；2.3 属兼容旧版。权重使用社区许可证并含商业门槛 | [官方仓库](https://github.com/Lightricks/LTX-2)、[许可证](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) |
| NVIDIA Cosmos | `cosmos-predict2.5` 已进入有限维护；当前入口为 Cosmos 3 模型仓与 Cosmos Framework | [模型仓](https://github.com/NVIDIA/cosmos)、[框架仓](https://github.com/NVIDIA/cosmos-framework) |
| MAGI | 冻结日应登记 MAGI-1.1 24B 与 4.5B/量化分支，而不是只停留在 MAGI-1 初版 | [官方仓库](https://github.com/SandAI-org/MAGI-1) |
| SkyReels | V2 仍是 diffusion-forcing 长视频入口；V3 已提供多参考、扩展和 talking-avatar 新任务入口 | [V2](https://github.com/SkyworkAI/SkyReels-V2)、[V3](https://github.com/SkyworkAI/SkyReels-V3) |
| V-JEPA | V-JEPA 2.1 / 2-AC 是 latent 表征与 action-conditioned world model，不能列作像素生成器 | [官方仓库](https://github.com/facebookresearch/vjepa2) |

## 5. 一手来源台账

### 5.1 当前基础与统一模型

| 项目 | 官方入口 | 本轮读取重点 |
|---|---|---|
| MiniMax H3 | [GitHub](https://github.com/MiniMax-AI/MiniMax-H3)、[HF](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 开放任务、未开放产品模块、native AV、许可 |
| LTX-2.5 | [GitHub](https://github.com/Lightricks/LTX-2) | 当前版本、distilled/DFR、trainer、许可与硬件路径 |
| Cosmos 3 | [Models](https://github.com/NVIDIA/cosmos)、[Framework](https://github.com/NVIDIA/cosmos-framework) | Reasoner/Generator、模型尺寸、训练计划项 |
| MAGI-1.1 | [GitHub](https://github.com/SandAI-org/MAGI-1) | 24B/4.5B、distilled/quantized、显存入口 |
| Bernini | [GitHub](https://github.com/bytedance/Bernini) | planner/renderer、Bernini-R、训练面 |
| Lance | [GitHub](https://github.com/bytedance/Lance) | 3B active、统一理解生成编辑、fine-tuning |
| HunyuanVideo-1.5 | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 8.3B、training/LoRA、未完成开放计划 |
| Wan2.2 | [GitHub](https://github.com/Wan-Video/Wan2.2) | A14B MoE、5B TI2V、多任务权重、官方硬件 |
| SANA-Video / LongSANA | [GitHub](https://github.com/NVlabs/Sana) | 线性注意力、长视频训练/测试、版本区分 |
| Open-Sora 2.0 | [GitHub](https://github.com/hpcaitech/Open-Sora)、[报告](https://arxiv.org/abs/2503.09642) | checkpoint、训练/推理、数据处理、成本边界 |
| CogVideoX | [GitHub](https://github.com/zai-org/CogVideo) | 2B/5B/1.5、SAT/Diffusers、分模型许可 |
| Step-Video | [GitHub](https://github.com/stepfun-ai/Step-Video-T2V) | 30B、T2V/Turbo/TI2V、显存、训练缺口 |
| Mochi | [GitHub](https://github.com/genmoai/mochi) | 10B AsymmDiT、Apache、LoRA、已知限制 |

### 5.2 长视频、控制、编辑与 world model

| 项目 | 官方入口 | 本轮读取重点 |
|---|---|---|
| FramePack | [GitHub](https://github.com/lllyasviel/FramePack) | context packing、低显存边界、社区加速影响 |
| SkyReels-V2 | [GitHub](https://github.com/SkyworkAI/SkyReels-V2) | diffusion forcing、extension、长时漂移边界 |
| DreamX-World | [GitHub](https://github.com/AMAP-ML/DreamX-World) | camera/event control、5B release、H20 作者成本 |
| Matrix-Game 3.5 | [GitHub](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) | 5B 权重、causal distillation、硬件与证据级别 |
| VACE | [GitHub](https://github.com/ali-vilab/VACE) | R2V/V2V/masked V2V、权重和继承许可 |
| Video-As-Prompt | [GitHub](https://github.com/bytedance/Video-As-Prompt) | 语义视频 prompt、官方与社区训练入口区分 |
| FlashMotion | [GitHub](https://github.com/quanhaol/FlashMotion) | trajectory、few-step、训练/推理/权重/评测 |
| Lumos | [GitHub](https://github.com/alibaba-damo-academy/Lumos) | AR + discrete diffusion、checkpoint、SFT/fine-tuning |
| V-JEPA 2 | [GitHub](https://github.com/facebookresearch/vjepa2) | latent prediction、action-conditioned 模型、非像素生成边界 |

### 5.3 框架与历史坐标

- 框架：[Diffusers](https://github.com/huggingface/diffusers)、[Finetrainers](https://github.com/huggingface/finetrainers)、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI)、[FFmpeg](https://ffmpeg.org/documentation.html)。
- 历史模型：[VideoGPT](https://arxiv.org/abs/2104.10157)、[MAGVIT](https://magvit.cs.cmu.edu/)、[OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer)、[Stable Video Diffusion](https://github.com/Stability-AI/generative-models)、[AnimateDiff](https://github.com/guoyww/AnimateDiff)、[VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)。

## 6. 冻结的核心代码版本

以下 SHA 是检索时从官方默认分支获得的代码快照，用于消除同名仓库随时间漂移；模型权重仍需在实际实验时单独冻结 revision 与文件 hash。

| 仓库 | 冻结 commit |
|---|---|
| `MiniMax-AI/MiniMax-H3` | `d21241f0a4b3…` |
| `Lightricks/LTX-2` | `a95ab856bf29…` |
| `NVIDIA/cosmos` | `e7ad5e77eecd…` |
| `NVIDIA/cosmos-framework` | `0e034bc98ffa…` |
| `SandAI-org/MAGI-1` | `0fcefdef8ce2…` |
| `Tencent-Hunyuan/HunyuanVideo-1.5` | `60783e704160…` |
| `Wan-Video/Wan2.2` | `42bf4cfaa384…` |
| `SkyworkAI/SkyReels-V3` | `28c771e84563…` |
| `hpcaitech/Open-Sora` | `7ad6a96a135f…` |
| `zai-org/CogVideo` | `7a1af7154511…` |
| `stepfun-ai/Step-Video-T2V` | `675e08c0dd08…` |
| `genmoai/mochi` | `0c86aebf386e…` |
| `lllyasviel/FramePack` | `97fe5dbe06ac…` |
| `SkyworkAI/SkyReels-V2` | `9351d1315220…` |
| `Riemann-Dynamics/Matrix-Game-3.5` | `fa6d2b628ac9…` |
| `ali-vilab/VACE` | `48eb44f1c4be…` |
| `facebookresearch/vjepa2` | `204698b45b37…` |

## 7. 图像资产记录

- 教材图：[`assets/diagrams/open-video-model-release-surface.png`](../assets/diagrams/open-video-model-release-surface.png)
- 生成提示核心：白底、16:9、无品牌标识；七步证据链为“官方身份 → 版本冻结 → 发布面拆分 → 许可交集 → 硬件预检 → 最小运行 → 复现清单”；从发布面、许可和最小运行设置红色隔离出口；只保留有限中文标签。
- 像素尺寸：1672 × 941，RGB PNG。
- SHA-256：`350f910762899322ca944bfccdc83d6af756678493d7c7055b81861cb4ad1938`。
- 视觉回读：七个主步骤顺序明确；三类发布面标签、三个红色失败出口、隔离框与颜色图例均可辨识；未出现模型名称、分数或额外结论。正文同时提供 Mermaid 与顺序化文字替代，以便编辑和读屏。

## 8. 证据限制与更新规则

- 本轮没有下载权重或运行 GPU smoke test，因此不能声称任何模型已经在本仓库环境复现。
- 硬件、速度、VBench 与 human preference 仅作为作者入口信息；跨项目比较前必须统一分辨率、帧数、FPS、步数、精度、offload、计时范围与 prompt 集。
- 仓库 HEAD、模型 revision、许可证和 `coming soon` 状态会快速变化；任何实验都应在开始日重新读取并生成 manifest。
- 对只有项目页或 technical report 的 world model，证据级别单列，不把未正式发表等同于错误，也不把作者演示等同于独立验证。
- 后续更新时必须保留“此前冻结值 → 新值”的变更记录，禁止无痕覆盖版本事实。

## 9. 本轮静态验收

- Markdown：`resources/open-models.md` 与本日志由 `markdownlint-cli2 v0.23.2` 检查，0 issues；本批 17 个 Markdown 文件合计同样为 0 issues。
- 引用：正文 35 个唯一引用编号与 35 个定义锚点完全闭合；missing、unused、duplicate 和编号错配均为 0。
- 本地链接：正文与日志所引用的研究记录、教学图均存在；全批 17 个 Markdown 共检查 41 个相对链接，missing 0。
- Mermaid：正文 3 个图块与本批其他页面合计 15 个图块，均用 `@mermaid-js/mermaid-cli 11.16.0` 和系统 Chrome 实际渲染为非空 SVG；15/15 成功，最小文件 23,712 bytes，均含 title/desc。
- PNG：教学图 1672 × 941、RGB PNG，SHA-256 与第 7 节一致；全批 8 张 PNG 均完成尺寸、格式、hash 和灰度统计检查。
- Git：`git diff --check` 无空白错误。远端 push 的 commit 与同步结果由提交后的最终交付记录给出，不在提交前伪写。
