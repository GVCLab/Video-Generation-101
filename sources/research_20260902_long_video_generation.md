# 长视频生成研究记录 — 2026-09-02

本文件记录 `docs/generative-models/long-video-generation.md` 的检索范围、纳排规则、主来源、证据边界与教学图生成过程。它不是第二份教程。所有会变化的论文状态和发布面冻结于 **2026-09-02（Asia/Shanghai）**。

## 1. 审查问题

1. fixed-long、length extrapolation 与 open-horizon 的最小合同分别是什么？
2. 整段联合、分层计划、滑窗协同、因果滚动和长期记忆改变的是同一个层面吗？
3. 哪些工作改变了可生成长度，哪些只降低计算或改善窗口边界？
4. 分钟级、小时级和 “infinite” 主张怎样保留作者协议边界？
5. 长视频与流式、实时、多镜头、视频预测和交互式世界模型怎样分工？
6. 哪些指标能发现冻结、循环、漂移、事件停滞和资源无界增长？

## 2. 检索范围与策略

- **审查日期：** 2026-09-02。
- **时间范围：** 2022-01-01 至 2026-09-02；更早工作只在解释机制不可替代时纳入。
- **来源优先级：** 正式 proceedings / 官方 venue → arXiv 原始记录 → 作者项目与官方仓库。
- **证据政策：** 技术事实回到论文，演示与作者数值只按作者报告。本章不做代码/权重可用性排名，也不对每篇工作的 release surface 作完备快照；正文若不提供逐项官方证据，就不声称代码、权重、数据或许可证已开放。
- **检索计数：** 本轮采用目标化检索和 exact-title 回读，不把搜索引擎返回数写成系统综述筛选统计。

目标化 query families 包括：

```text
"long video generation" fixed length extrapolation open horizon
"training-free long video generation" noise rescheduling spectral attention
"autoregressive long video diffusion" self forcing rolling forcing
"minute-length text-to-video" linear complexity
"infinite video generation" FIFO diffusion error recycling
"long video generation" context retrieval memory KV cache
"long video evaluation" freeze repetition first failure survival
site:proceedings.neurips.cc "long video generation"
site:proceedings.iclr.cc "long video generation"
site:openaccess.thecvf.com "long video generation"
```

检索还复核了仓库已有的因果/流式、多镜头、Video DiT、tokenizer、预测与交互式世界生成章节，避免为新章复制已有合同。

## 3. 纳入、排除与证据等级

### 3.1 纳入

- 直接改变长视频表示、生成分解、训练分布、外推机制、长期记忆或评测协议的一手工作；
- 正式 venue 页面，用于核验题名、年份和论文身份；
- 尚无正式 proceedings 的近期预印本，用于描述冻结日前沿；
- 与本章边界直接相关的系统论文，如因果少步生成、缓存或长时 evaluator；
- 能明确写出输入、输出、窗口、资源或失败合同的工作。

### 3.2 排除或降级

- 只增加插帧、慢动作、静帧复制或循环播放而没有新增状态的结果；
- 只有产品演示、媒体稿、聚合站或第三方转载的长度主张；
- 长视频理解、检索或摘要工作，除非只用于 evaluator/记忆边界反例；
- 只研究多镜头故事、4D 重建或动作闭环，却没有长时生成机制的工作；
- “infinite”“hour-long”“minute-long” 不按字面升级为任意时长稳定；
- 论文、代码、权重、数据和独立复现不互相继承证据等级。

### 3.3 证据等级

| 等级 | 一手证据 | 本章允许写法 |
|---|---|---|
| A | 正式 proceedings / venue + 论文 | 可确认论文身份，并写“论文提出/报告” |
| B | arXiv 或作者技术报告 | 写“作者提出/报告”，不称正式发表或独立确认 |
| C | 官方 repo、项目页、model card | 只支持发布面和维护者说明 |
| S | 本综述合成的合同、分类或实验 | 必须明确为综述框架或建议，不能归给单篇论文 |

## 4. 主来源账本

### 4.1 2022–2023：长序列表示、可变长度与层级计划

| 工作 | 主来源 | 状态 | 纳入原因与边界 |
|---|---|---|---|
| Generating Long Videos of Dynamic Scenes | <https://proceedings.neurips.cc/paper_files/paper/2022/hash/ce208d95d020b023cba9e64031db2584-Abstract-Conference.html> | A·NeurIPS 2022 | 分离长期低分辨率动态与短期高分辨率外观；受控动态场景不等于开放域故事 |
| TATS | <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5950_ECCV_2022_paper.php> | A·ECCV 2022 | time-agnostic VQGAN + time-sensitive Transformer；代表离散 token 长序列 |
| Phenaki | <https://iclr.cc/virtual/2023/poster/12256> | A·ICLR 2023 | 时间因果 tokenizer 与 masked generator 支持可变长度；整个 generator 不应误写为 strict AR |
| NUWA-XL | <https://aclanthology.org/2023.acl-long.73/> | A·ACL 2023 | 全局 keyframe diffusion + 局部递归插值；代表全局计划—局部渲染 |
| Gen-L-Video | <https://arxiv.org/abs/2305.18264> | B·arXiv 2023 | 多窗口 temporal co-denoising；training-free，但没有正式 venue 证据 |

### 4.2 2024：推理时外推与开放滚动

| 工作 | 主来源 | 状态 | 纳入原因与边界 |
|---|---|---|---|
| FreeNoise | <https://proceedings.iclr.cc/paper_files/paper/2024/hash/15ce8e7afe5ee95bad56e3b9be28d3d1-Abstract-Conference.html> | A·ICLR 2024 | 噪声重排 + window attention fusion；延长短模型，不自动新增长期语义 |
| FIFO-Diffusion | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/a397986e0f34d4b1f0b640686ceaeff7-Abstract.html> | A·NeurIPS 2024 | 斜向去噪队列与固定窗口；“infinite”是持续采样接口，不是质量保证 |
| FreeLong | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html> | A·NeurIPS 2024 | 全局低频与局部高频 SpectralBlend；作者示例从 16 帧扩到 128 帧 |
| Diffusion Forcing | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html> | A·NeurIPS 2024 | per-token noise level 连接 full-sequence diffusion 与 next-token；不自动表示少步/实时 |

### 4.3 2025：混合规划、线性骨干与 self-history

| 工作 | 主来源 | 状态 | 纳入原因与边界 |
|---|---|---|---|
| ARLON | <https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2fb048a8e37ad978fc895528102ce49-Abstract-Conference.html> | A·ICLR 2025 | 粗粒度 AR visual tokens 引导 DiT；混合全局计划与局部生成 |
| LinGen | <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LinGen_Towards_High-Resolution_Minute-Length_Text-to-Video_Generation_with_Linear_Computational_Complexity_CVPR_2025_paper.html> | A·CVPR 2025 | 线性复杂度时空模块面向高分辨率分钟级；结果按作者协议读取 |
| LongDiff | <https://openaccess.thecvf.com/content/CVPR2025/html/Li_LongDiff_Training-Free_Long_Video_Generation_in_One_Go_CVPR_2025_paper.html> | A·CVPR 2025 | 位置映射与信息帧选择处理位置歧义/信息稀释；one-go 不等于 open-horizon |
| StreamingT2V | <https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html> | A·CVPR 2025 | 短期条件 + 长期首段锚点；约两分钟为作者展示 |
| CausVid | <https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html> | A·CVPR 2025 | 双向 teacher → 少步 causal student；速度必须绑定硬件与端到端边界 |
| Self Forcing | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html> | A·NeurIPS 2025 | 在 self-rollout 历史上训练，直接处理 exposure bias |
| FramePack | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html> | A·NeurIPS 2025 | 重要性压缩固定上下文；部分 anti-drift 变体含终点条件，不都属于 strict causal |
| TokensGen | <https://openaccess.thecvf.com/content/ICCV2025/html/Ouyang_TokensGen_Harnessing_Condensed_Tokens_for_Long_Video_Generation_ICCV_2025_paper.html> | A·ICCV 2025 | 先生成 condensed global tokens，再逐片段还原 |

### 4.4 2026：检索记忆、滚动实时与超长纠错

| 工作 | 主来源 | 状态 | 纳入原因与边界 |
|---|---|---|---|
| Mixture of Contexts | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/94bbcb744bbada8808fda05b9d9290d6-Abstract-Conference.html> | A·ICLR 2026 | 动态选择历史块并保留局部窗口/文本锚点；把长上下文改写为检索问题 |
| Rolling Forcing | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html> | A·ICLR 2026 | 渐变噪声窗口、sink 与 self-history；实时数字按作者硬件设置读取 |
| LongLive | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html> | A·ICLR 2026 | frame AR、prompt KV recache、train-long–test-long；最长 240 秒/速度是作者协议 |
| Stable Video Infinity | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/2858f8c8683aaa8c12d487354cf328dc-Abstract-Conference.html> | A·ICLR 2026 | error recycling 与非循环 ultra-long；名称不表示已证明无限稳定 |
| Flow Caching | <https://proceedings.iclr.cc/paper_files/paper/2026/hash/85dc8f85ff978b9c606d3b2f5b0da69a-Abstract-Conference.html> | A·ICLR 2026 | 分块特征缓存和 KV 压缩；加速倍率不自动等于端到端实时 |
| LoL | <https://openaccess.thecvf.com/content/CVPR2026/papers/Cui_LoL_Longer_than_Longer_Scaling_Video_Generation_to_Hour_CVPR_2026_paper.pdf> | A·CVPR 2026 | 分析 attention sink collapse，并用多头 RoPE jitter 抑制；12 小时是选择性作者演示 |
| Free-Lunch Long Video Generation | <https://openaccess.thecvf.com/content/CVPR2026/html/Tian_Free-Lunch_Long_Video_Generation_via_Layer-Adaptive_O.O.D_Correction_CVPR_2026_paper.html> | A·CVPR 2026 | 重编码修正 + 层自适应稀疏 attention；重点是短模型 2×/4× 外推 |
| SLVMEval | <https://openaccess.thecvf.com/content/CVPR2026/html/Matsuda_SLVMEval_Synthetic_Meta_Evaluation_Benchmark_for_Text-to-Long_Video_Generation_CVPR_2026_paper.html> | A·CVPR 2026 | 用受控长时退化测 evaluator 敏感性；不是模型排行榜 |

### 4.5 冻结日前沿：只按预印本处理

| 工作 | 主来源 | 首版 | 技术方向 | 限制 |
|---|---|---:|---|---|
| LongLive-RAG | <https://arxiv.org/abs/2606.02553> | 2026-06 | 从已生成 latent 历史做内容寻址检索 | B；作者实验，未与正式论文混写 |
| RECAP-Forcing | <https://arxiv.org/abs/2608.26671> | 2026-08-27 | 按外观新颖性而非纯时间近邻保留 KV | B；冻结日为新近预印本 |
| DensityKV | <https://arxiv.org/abs/2608.27922> | 2026-08-28 | 逐 attention head 维护有界 token 级历史 bank | B；冻结日为新近预印本 |
| LayerRecall | <https://arxiv.org/abs/2608.28460> | 2026-08-28 | 状态条件检索并只向 memory-sensitive layers 注入 | B；冻结日为新近预印本 |

这些工作只用于说明“最近窗口 → 内容/层/头级路由”的趋势，不用于宣称该方向已经解决长期漂移。

## 5. 任务边界与负面核验

1. **长视频不等于流式。** LongDiff 的 one-go 与 FIFO/Rolling 的持续窗口属于不同输出合同。
2. **流式不等于实时。** 不含 codec、调度、传输和显示的 FPS 不能支持端到端 SLO。
3. **长视频不等于多镜头。** 连续镜头可能很长；多镜头系统也可能只是很多短 clip。
4. **多文本不等于故事。** prompt 可切换不证明镜头语法、事实保持、角色回归或回滚。
5. **视频预测不等于无条件长 rollout。** 真实历史条件与模型自生成历史的分布不同。
6. **长演示不等于世界模型。** 没有动作条件、可查询状态和闭环反事实时不升级为交互世界模型。
7. **有界 GPU memory 不等于恒定总成本。** CPU、外存、索引、带宽和总生成时间仍需报告。
8. **精选最长样例不等于最大可靠时长。** 必须给 prompt/seed 分布与 first-failure survival。

## 6. 本章综合框架的来源边界

正文的“五道证据门”、固定长片/长度外推/开放时域三分法，以及 `LongHorizon-1` 都是本综述的综合（等级 S）。它们从上述论文的长序列、外推、滚动、记忆和 evaluator 问题抽象而来，不被描述为某篇论文已经完整实现的统一协议。

四条图示路线也不是穷尽分类：整段联合、分层关键帧、滑窗协同、因果滚动可以出现在同一个系统中。方法表的用途是定位主要瓶颈和应有证据，不是把论文硬分为互斥家族。

`LongHorizon-1` 明确标为**建议实验、尚未执行**。本轮没有下载模型、生成样片或独立复现论文质量/速度结果。

## 7. 图像资产记录

- 项目文件：`assets/diagrams/long-video-generation-contract.png`
- 生成方式：内置 image generation；先生成完整概念图，再做一次局部文字修正。
- 最终尺寸：1672 × 941 px，RGB PNG，无 alpha。
- SHA-256：`3601fda0cc3782050865bf371993104861fc054f479a0da675b535dcefa4e620`
- 主提示核心：白底 16:9 科学教学图；“短片能力”经过“整段联合 / 分层关键帧 / 滑窗协同 / 因果滚动”四条并行路线和“有界记忆”，到达 5 秒、30 秒、1 分钟以上时间轴；底部放置非重复帧、局部质量、长程状态、事件进展、资源曲线五道独立证据门；使用色盲友好蓝/橙/绿/紫配色，无模型名、分数、logo 或水印。
- 局部修正：将第一稿可能歧义的“唯一帧”改为“非重复帧”，其余构图保持不变。
- 视觉回读：15 组文字标签均清晰；四路线、记忆模块、时间轴和五个证据门层级明确；没有乱码、裁切、logo、水印或额外能力结论。

## 8. 验收记录与边界

最终验收于 2026-09-02 执行：

| 检查 | 实测结果 |
|---|---|
| 新增文档 Markdown | 章节与研究记录共 2 个文件，0 issues |
| 章节引用闭环 | 29 条正文引用对应 29 条参考文献，无缺失、闲置、编号空洞或重复 |
| 范围内相对链接 | 12 个本轮相关文件、340 个相对链接，0 missing |
| 全库离线引用检查 | 扫描 82 个 Markdown 文件、3,070 个链接和 419 个 arXiv ID，0 errors / 0 warnings |
| 一手来源回读 | 新章的正式会议年份、题名与高风险 2026 条目未发现不一致；全库在线检查仍有其他历史文件的既有告警，本轮未扩大处理范围 |
| 图像资产 | 1672 × 941 px、RGB PNG、无 alpha；哈希与上节一致，并完成文字、裁切、层级和水印视觉回读 |
| 站点构建 | `python3 scripts/build_site.py --strict` 成功；464 个文件进入构建区，目标页写入 `_site/docs/generative-models/long-video-generation.html` |
| 页面回读 | 标题、目录、表格、公式、图像、`LongHorizon-1` 和参考文献均可见；页面、CSS、图像、脚本与搜索索引请求成功 |
| 补丁卫生 | `git diff --check` 通过 |

这些检查只能证明资料组织、引用路径与页面资产内部一致，不能把作者报告的长时质量、速度或资源结果升级为独立复现。本轮没有下载模型、生成样片或运行 `LongHorizon-1`。
