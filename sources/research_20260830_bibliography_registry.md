# 核心 bibliography 注册表审计与刷新记录

## 1. 任务、冻结时间与证据边界

- 审计日期：2026-08-30（Asia/Shanghai）。
- 目标：修复 `bibliography/registry.json` 与全仓正文引用之间的核心覆盖缺口，刷新机器生成元数据、BibTeX、GitHub Stars 与索引页。
- 一手来源限定：论文正式页、arXiv、Crossref、机构原始报告、论文作者或机构的官方 GitHub 仓库/项目页。
- 不做的事：不把全仓每个链接机械并入核心表；不把搜索摘要、二手综述或社区 fork 当作论文/代码状态的一手证据；不修改技术正文、timeline、reading list 或 coverage audit。
- 年份口径：citekey 与索引年份使用一手元数据中的首次预印本/来源年份；正式发表年份留在各章参考文献或来源的 journal reference 中。arXiv 编号中的年月不能替代 API 的 `published` 字段。

## 2. 扫描方法与审计基线

扫描所有 Markdown，识别以下显式标识：

```text
https://arxiv.org/abs/YYMM.NNNNN
https://arxiv.org/pdf/YYMM.NNNNN
arXiv:YYMM.NNNNN
```

版本后缀 `vN` 在比较前去除；随后与 registry 中 `source.kind == "arxiv"` 的 source id 集合比对。核心范围单独限定为：

- `docs/generative-models.md`；
- `docs/generative-models/*.md`；
- `docs/foundation-models.md`。

变更前基线：

| 指标 | 数量 |
|---|---:|
| registry 条目 | 58 |
| registry arXiv / DOI / manual | 44 / 4 / 10 |
| 全仓 Markdown 唯一 arXiv ID | 362 |
| 全仓已用但未注册 arXiv ID | 318 |
| 核心范围已用但未注册 arXiv ID | 51 |

全仓缺口大于 coverage audit 中的“至少 31”并不意味着 318 条都应进入核心表。`sources/` 保存发现与核验账本，任务、评测、数据集和时间线页面也各自保留完整章末参考文献；这些局部证据与跨章节核心书目是两层不同基础设施。

## 3. 核心注册表纳入与排除

### 3.1 候选纳入条件

候选至少满足一项，并经人工判断确为跨章节核心入口：

1. 是机制总览、机制分章或视频基础模型主章中的路线锚点、定义性论文或里程碑；
2. 在两个及以上机制/基础模型页面重复支撑跨路线主张；
3. 是核心模型家族、系统或 release surface 的唯一机构一手报告。

### 3.2 默认排除

1. 只在单个任务、评测、数据集、时间线或研究日志中作局部证据的条目；
2. 二手综述、新闻、聚合页，以及与核心论文没有直接对应关系的社区实现；
3. 单独的许可证、标准、产品或仓库页面；它们继续保留在章末参考文献中，除非本身就是核心系统唯一可引用的一手报告；
4. 题名或作者相近、但无法由原论文/项目页确认归属的 GitHub 仓库。

这套边界同时写入 `bibliography/registry.json` 的 `scope` 字段，并由更新脚本生成到 `docs/bibliography.md`，避免规则只存在于一次性审计日志。

## 4. 本轮纳入结果

- 新增 51 条核心记录，使 registry 从 58 增至 109 条。
- 51 条新增记录全部使用 arXiv 主标识，恰好闭合核心范围的 51 个显式 arXiv 缺口。
- `World Models` 原来把 arXiv URL 包在 manual metadata 中；本轮改为 `source.kind = arxiv`、`source.id = 1803.10122`。因此 arXiv source 总数从 44 增至 96，manual 从 10 降至 9。
- 51 条新增记录中，21 条由论文、项目页或官方仓库说明确认了 `official-code`；对应 20 个唯一 GitHub 仓库，Causal Forcing 与 Causal Forcing++ 共用同一官方仓库。
- fresh metadata 后，所有 citekey 中的年份均与一手 source 的首次年份一致。特别是 LTX-Video 的 arXiv ID 为 `2501.00103`，但 API `published` 日期属于 2024，因此 citekey 为 `hacohen2024ltxvideo`，不从编号前缀误写为 2025。

新增记录按三个扩展分组呈现：

- I：跨路线生成基础、测量与 tokenizer；
- J：现代视频基础模型与系统报告；
- K：因果、自回归流式生成与加速。

## 5. 新增官方代码关系

以下关系由官方 GitHub API 的 repository metadata、仓库描述/README 与论文题名或项目页交叉核对。`official-code` 只表示作者或机构维护的对应实现，不等于完整训练数据、权重、许可证或独立复现均已开放。

| Cite key | 官方仓库 | 判定 |
|---|---|---|
| `lee2018stochastic` | [alexlee-gk/video_prediction](https://github.com/alexlee-gk/video_prediction) | 仓库描述直接对应 SAVP |
| `song2020scorebased` | [yang-song/score_sde](https://github.com/yang-song/score_sde) | README 标注 ICLR 2021 官方实现 |
| `fuest2025maskflow` | [CompVis/maskflow](https://github.com/CompVis/maskflow) | 仓库描述直接对应论文题名 |
| `kong2024hunyuanvideo` | [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | 机构官方代码与权重仓库 |
| `hacohen2024ltxvideo` | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) | Lightricks 官方仓库 |
| `ma2025stepvideo` | [stepfun-ai/Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V) | 机构官方实现 |
| `zheng2025opensora2` | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) | Open-Sora 2.0 官方训练/发布仓库 |
| `wan2025wan` | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | 仓库描述直接对应 Wan 报告 |
| `chen2025skyreelsv2` | [SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) | 机构官方实现 |
| `sandai2025magi1` | [SandAI-org/MAGI-1](https://github.com/SandAI-org/MAGI-1) | 机构官方实现 |
| `low2025ovi` | [character-ai/Ovi](https://github.com/character-ai/Ovi) | 机构官方实现 |
| `wu2025hunyuanvideo15` | [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 机构官方实现 |
| `hacohen2026ltx2` | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | 官方推理与 LoRA 训练仓库 |
| `li2026skyreelsv3` | [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3) | 机构官方实现 |
| `chen2024diffusionforcing` | [buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing) | 论文作者发布的实现 |
| `yin2024causvid` | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | 仓库描述直接对应 CVPR 2025 论文 |
| `huang2025selfforcing` | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | README 标注 NeurIPS 2025 官方实现 |
| `yang2025longlive` | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) | NVIDIA Research 官方仓库 |
| `zhu2026causalforcing` | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | README 标注 ICML 2026 官方实现 |
| `zhao2026causalforcingpp` | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | 同一 README 明确覆盖 Causal Forcing++ |
| `zheng2026causalrcm` | [NVlabs/rcm](https://github.com/NVlabs/rcm) | 仓库描述与配方文档明确覆盖 Causal-rCM |

旧条目中标为“仓库已归档”的 `google-research/magvit`、`google-research/planet` 与 `facebookresearch/ijepa` 也通过官方 GitHub API 重新检查，三者的 `archived` 均仍为 true，因此保留原状态说明。

## 6. 元数据、年份与类型修正

1. arXiv metadata 按 40 个 ID 一批刷新，避免 96 个 ID 放在单个长请求中；题名、作者、首次年份、primary class、DOI 和 journal reference 均来自 [arXiv API](https://export.arxiv.org/api/query)。
2. 4 个 DOI 条目从 [Crossref REST API](https://api.crossref.org/) fresh-fetch；不使用搜索摘要补作者或年份。
3. `World Models` 的 arXiv feed 提供 Zenodo DOI `10.5281/zenodo.1207631`。更新器现在不会仅因 arXiv feed 含 DOI 就把预印本强制写成 `@article`；arXiv source 保持 `@misc`，DOI 仍作为可解析标识输出。
4. 对 arXiv 中的机构作者作最小 BibTeX 保护：NVIDIA、Sand.ai、Wan Team 与 Seedance Team 使用大括号保持团体作者语义；不删减其后个人作者。
5. registry validation 新增 citekey 格式、section、arXiv ID、manual 必需字段、标准化 source id 唯一性与 GitHub repository-root URL 检查。

## 7. 无法自动确认的项目

- 51 条新增记录中有 30 条未登记 GitHub。这里的 null 只表示本轮未从论文、官方项目页或官方组织确认到直接对应的 GitHub repository；不宣称代码绝对不存在。
- SDXL-Lightning 与 AnimateDiff-Lightning 有官方权重/演示入口，但本轮检查的 `ByteDance/…` 与 `ByteDance-Seed/…` GitHub candidate paths 由 GitHub 官方 API 返回 404；不能把 Hugging Face、DagsHub 镜像或社区 notebook 填成官方 GitHub，因此保持 null。
- 正式 proceedings、OpenReview、标准与机构项目页的完整覆盖不由 arXiv regex 衡量；它们仍以各章参考文献为准，只有被人工选为核心入口时才进入 registry。

## 8. 动态 GitHub Stars 快照

- fresh run：`python scripts/update_bibliography.py --all`。
- 50 个唯一 repository 均从公开 GitHub repository page 成功取得实时星标，终端没有出现缓存回退警告。
- `bibliography/github-stars.json` 记录 `fetched_at = 2026-08-29T22:39:55Z`；对应 Asia/Shanghai 为 2026-08-30 06:39:55。
- `as_of = 2026-08-29` 使用脚本既有的 UTC 日期语义，因此生成页显示 2026-08-29；本日志同时记录本地审计日期，避免把时区差误当成旧快照。
- Stars 只用于动态可见度快照，不支持代码质量、复现性、许可证或官方性结论。

## 9. 残余缺口与解释

变更后：

| 范围 | 唯一 arXiv ID | 已注册交集 | 未注册 |
|---|---:|---:|---:|
| 全仓 Markdown | 362 | 96 | 266 |
| 发布页 `docs/` + `resources/`（不含生成索引） | 330 | 95 | 235 |
| `sources/` 研究账本 | 280 | 82 | 198 |
| 核心机制/foundation 范围 | 59 | 59 | 0 |

266 个全仓残余中，31 个只出现在 `sources/`，235 个出现在任务、评测、数据集、timeline 等发布页。它们是章末完整书目候选，不是本轮核心 registry 漏加的 266 条。残余中只有 `2512.00425` 同时出现在三个发布页（physical consistency、video reasoning、datasets）；它属于跨评测/数据集证据，若未来把核心表扩展到评测基础设施，应优先重新审查。

## 10. 验收命令与结果

- JSON：`python3 -m json.tool` 检查 registry、metadata 与 stars，均可解析。
- 唯一性：109 个 citekey 唯一；109 个标准化 source identity 唯一；96 个 arXiv source id 唯一。
- fresh metadata：96 个 arXiv 与 4 个 Crossref DOI 均成功返回，0 缺失。
- dry run：`python scripts/update_bibliography.py --check`（别名 `--dry-run`）只读检查 registry、快照与生成物一致性。
- offline replay：`python scripts/update_bibliography.py --offline` 从快照重生成后，四个生成物与 fresh run 一致。
- BibTeX：109 个 entry、109 个 citekey；大括号平衡且与 registry key 集完全一致。
- 本地链接：本日志与 `docs/bibliography.md` 的相对链接全部存在。
- Git：`git diff --check` 无空白错误。

上述验收不包含论文复现实验、权重下载或远端 push；本轮没有 commit/push。
