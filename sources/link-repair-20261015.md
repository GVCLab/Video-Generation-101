# 2026-10-15 外链修复记录

## 范围与原则

依据 [Check links run 38708098450](https://github.com/GVCLab/Video-Generation-101/actions/runs/38708098450) 的 15 条错误、1 条超时（10 个唯一地址），核对论文身份和作者入口后更新正文及历史调研记录。保留参考编号、论文、数据集及技术论述；不删除引用，不接受 400/404，不扩大忽略范围。

## 对应关系与核验依据

| 对象 | 原问题 | 修复后的一手入口与身份依据 |
|---|---|---|
| GEARS | 相关 DOI 暂返回 404 | [同篇 arXiv](https://arxiv.org/abs/2608.29322)保留标题、作者及接收说明；元数据仍列出 `10.1145/3842526`，所以保留该 DOI 编号并注明解析状态，不冒称 DOI 已更正或可访问 |
| BulletTime | CVF 文件名多出 `Generation` | [CVPR 正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_BulletTime_Decoupled_Control_of_Time_and_Camera_Pose_for_Video_CVPR_2026_paper.html)，标题与 Yiming Wang 等作者一致 |
| Omni-LiveAvatar | 仓库所有者写错 | [作者仓库 Aoko955/Omni-LiveAvatar](https://github.com/Aoko955/Omni-LiveAvatar)，与[论文摘要给出的地址](https://arxiv.org/abs/2608.13602)一致；同步修正徽章目标 |
| MEAD | 原项目页路径返回 404 | [ECCV 原论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660698.pdf)与[作者仓库 uniBruce/Mead](https://github.com/uniBruce/Mead)，保持 ECCV 2020 同一数据集 |
| LongCat-Video-Avatar | 独立仓库路径不存在 | [Meituan LongCat-Video 主仓库](https://github.com/meituan-longcat/LongCat-Video)，README 包含 Avatar；[原论文](https://arxiv.org/abs/2605.26486)保留 |
| SadTalker | CVF 文件名多出 `Image` | [CVPR 正式论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SadTalker_Learning_Realistic_3D_Motion_Coefficients_for_Stylized_Audio-Driven_Single_CVPR_2023_paper.html)，仍为相同论文 |
| EDEN | 仓库所有者误写成 MCG-NJU | [作者仓库 bbldCVer/EDEN](https://github.com/bbldCVer/EDEN)，由[论文全文](https://arxiv.org/html/2503.15831v2)中的仓库链接互证 |
| LDF-VFI | 仓库所有者误写成 Xinyu-Peng | [作者仓库 xypeng9903/LDF-VFI](https://github.com/xypeng9903/LDF-VFI)，由[论文摘要](https://arxiv.org/abs/2601.14959)互证 |
| BDD100K | 项目域名请求超时 | 使用[官方工具与数据说明仓库](https://github.com/bdd100k/bdd100k)，不替换为社区重打包数据 |
| arXiv API | 裸查询端点没有参数时返回 400 | 可点击入口改为[官方 API 文档](https://info.arxiv.org/help/api/index.html)；实际端点作为配置文本保留，注明需要查询参数 |

链接核验日期为 2026-10-15。历史调研页增加维护说明，避免把入口修复解释成所有实验、代码、权重或许可证状态均已刷新。GEARS 的接收与 DOI 状态采用 arXiv 作者元数据归因，不补造出版记录。

## 验证

在仅含待提交内容的独立副本中完成：

- 完整 Lychee 扫描：5,785 条记录、2,161 个唯一地址；5,785 条成功，0 错误、0 超时、0 排除。接受状态与 CI 保持一致，未改检查配置。
- 在线引用元数据：56 个文件、416 个唯一 arXiv ID 全部返回，0 错误、0 警告、0 未核验。
- 严格构建、61 个源文件/63 个渲染页面的结构检查、14,696 个站内链接检查通过。
- 6 项 manual 检查、1 项数学检查通过；原参考编号与块级公式保留。
- 原有 bibliography 与生成文献索引等未提交修改保持不变，未纳入本次提交。

机器可读的替换入口 HTTP 核验及检查摘要见 [link-repair-20261015-http.json](link-repair-20261015-http.json)。GitHub 的最终检查状态以本次提交对应的 Check links 工作流为准。
