# 2026-09-20 技术手册修订与检查记录

## 范围

对 README、贡献指南、技术章节与资源章节进行结构、术语、引用和站内链接检查，重整 55 个既有技术/资源章节，新增 `docs/manual-guide.md`。生成的文献索引纳入链接与引用检查，未重新生成其底层文献数据库。技术公式、原始参考文献和有来源的实验条件保留；不能确认的动态状态继续按原条目日期标注。

## 内容修改

- 删除数据集章节原 4.3 整节；OpenVid 条目只保留官方可核验的 OpenVid-1M 和 OpenVidHD-0.4M，并使用官方仓库入口及论文统计口径。
- 重写首页与章节入口，补充范围、前置知识和使用步骤；将论述式、纠错式标题改为定义、接口、方法、配置、评测与故障诊断标题。
- 将自拟实验名称改成描述性实验名称，明确建议设计未运行，示例参数不代表公认基准。
- 在评测章定义 L0–L7 为本手册的报告约定；WAM 采用同一口径，视频预测和交互系统分别采用 VP、IW 局部编号，避免相同编号具有不同含义。
- 修正“所有生成均从噪声开始”“模型不会记忆训练视频”“潜空间规划不使用像素重建”等过度概括；区分表示、概率分解和训练目标。
- 区分 AnimeShooter 数据集和 AnimeShooterGen 生成器；区分 BAIR 真实机器人数据与合成物理数据。
- 统一可移植的行内数学语法；替换视频预测中标记不一致或模块含义不清的概念图，使用与正文一致的 Mermaid 流程。
- 保留可兼容的旧章节锚点，更新正文跨章节链接；导航按六类技术内容组织。历史调研资料继续可链接，但不进入正文导航和搜索；实施计划不发布到站点。

## 实质性来源更正

| 项目 | 更正 | 一手依据 |
|---|---|---|
| SyncNet | 原 arXiv ID 指向量能器论文；改为正确论文标题与牛津作者页 | [Out of time: automated lip sync in the wild](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/) |
| Hallo-Live | 修正标题、首作者及文本驱动联合音视频的任务描述 | [arXiv:2604.23632](https://arxiv.org/abs/2604.23632) |
| LeVJEPA | 修正完整标题及 Lukas Kuhn 首作者名 | [arXiv:2608.27395](https://arxiv.org/abs/2608.27395) |
| DINO-WM | 首作者改为 Gaoyue Zhou | [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) |
| OmniAvatar | 首作者改为 Qijun Gan | [arXiv:2506.18866](https://arxiv.org/abs/2506.18866) |
| Branch-JEPA、Var-JEPA | 完整标题按返回的官方元数据更正 | [Branch-JEPA](https://arxiv.org/abs/2607.05238)、[Var-JEPA](https://arxiv.org/abs/2603.20111) |
| 多镜头方法 | CineTrans、AnimeShooter、CausalCine、CineWeaver、UnityShots、LogiShot、SEAM 的标题按官方元数据更正 | 对应章节参考文献 |
| Open-Sora 2.0 | 技术报告名称按官方元数据更正 | 对应章节参考文献 |
| LongCat Avatar | 链接改为包含 Avatar 实现的 LongCat-Video 主仓库 | [作者论文](https://arxiv.org/abs/2605.26486)、[官方仓库](https://github.com/meituan-longcat/LongCat-Video) |
| SadTalker | 修复 CVPR 官方论文链接中的错误路径 | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_CVPR_2023_paper.html) |
| Video Inpainting of Complex Scenes | 区分 2014 期刊发表与 2015 arXiv 上传 | [arXiv 的 journal reference](https://arxiv.org/abs/1503.05528) |
| Video Pixel Networks | 区分 2016 首次公开与 ICML 2017 发表 | [预印本](https://arxiv.org/abs/1610.00527) |
| Sora | 截至 2026-09-20，Web/app 已于 4 月 26 日停用；API 计划于 9 月 24 日停用 | [官方停用时间表](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation) |

## 自动检查与页面检查

- 在线引用检查：正文中 53 个含外部引用的文件，2,307 个链接记录，421 个唯一 arXiv ID；421/421 成功查询并返回元数据，0 个未查询，按脚本规则为 0 错误、0 警告。
- 检查器新增 HTML 链接支持，覆盖时间线卡片；已知方法别名与导航标签不误当完整论文标题。部分网络失败会返回未完成状态并保存覆盖率，而不会显示通过。
- 严格站点构建、正文结构、渲染后的本地文件与锚点、搜索范围检查通过。精确计数见随附验证记录。
- 6 项回归测试覆盖错误论文链接识别、未知与不存在的区分、部分查询失败、HTML 引用、坏锚点和调研记录的安全暂存。
- 浏览器抽查数据集章节：原 4.3 已不在目录/正文，公式已排版，无数学错误节点；抽查预测章节的 Mermaid 流程与公式。

## 检查边界

元数据核验能发现标题、链接身份及部分年份错误，不能证明论文所有机制或实验数字正确。本次没有训练或运行各论文模型，没有逐一下载媒体/权重，也未逐幅重绘全书历史插图。保留的作者实验按原来源解释；资源访问状态按其日期理解。没有将本次文档编辑描述为独立实验复现。
