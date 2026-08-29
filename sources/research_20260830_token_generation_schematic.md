# 视频 token 生成路线图：生成与验收记录

> 记录日期：2026-08-30。该文件只记录图的语义规范、生成取舍和验收结果；论文证据分别见本批 representation/tokenizer、autoregressive 与 masked 研究记录。

## 1. 图要解决的误解

目标图只回答三件事：生成变量是什么、外层联合分布如何分解、条件分布由什么 head 实现。它必须阻止以下错误等同：

1. continuous latent 不等于 diffusion；它也可以置于 frame/chunk 自回归外层中；
2. discrete token 不等于 strict next-token；它也可以按集合并行 refinement；
3. frame/chunk AR 描述外层条件顺序，内部仍可使用 diffusion、flow 或 masked head；
4. 少步、流式和实时属于部署与测量问题，不能由路线名称直接推出。

## 2. 最终语义规范

最终图采用三条互不交叉的横向路线，以消除任意兼容性箭头带来的歧义。

### A. Continuous-latent route

`Pixel video → Causal 3D VAE → Continuous latent grid → Joint OR frame/chunk factorization → Diffusion / flow head → Decode video`

这条路线同时展示 codec ceiling、外层 data-time factorization 和内层 denoising/transport time，不把三者压成一个“diffusion 模型”标签。

### B. Discrete-token route

`Pixel video → VQ / LFQ / BSQ → Discrete token IDs → Strict token OR grouped AR → Categorical cross-entropy → Decode video`

这条路线把词表、token 数量、顺序、KV cache 与串行深度绑定，而不声称所有离散生成都必须逐 token。

### C. Masked / discrete-diffusion route

`Partly masked token set → Bidirectional prediction → Confidence selection → Commit high-confidence tokens → Remask uncertain tokens → Repeat until complete`

唯一反馈边从 `Remask uncertain tokens` 返回 `Bidirectional prediction`，表示下一轮 refinement；它不经过 strict next-token 链。

## 3. 生成迭代

第一稿采用四列兼容性图，但被拒绝：生成模型把 continuous latent 错连到 strict token AR，并把 discrete token 错连成固定的 frame AR。由于箭头的科学含义错误，该稿未进入仓库。

第二稿改用三条独立路线，逐项限定相邻节点与唯一反馈边。最终文件为：

- 路径：`assets/diagrams/video-token-generation-routes.png`
- 尺寸：1672 × 941 px，16:9
- 色彩：sRGB、8 bit、无 alpha；蓝/橙/绿三路线同时用 A/B/C 和独立文字编码
- SHA-256：`6c3fc69cd8c512c34af03beb2103065b103372e722e2ec003fbb185e94a174cf`

## 4. 验收

- [x] 所有主箭头只连接同一路线中的相邻节点；无跨路线箭头。
- [x] masked feedback 只从 remask 返回 bidirectional prediction。
- [x] 标题、框内文字和页脚逐字检查，无乱码、截断或水印。
- [x] 原图 1672 × 941，关键文字在原尺寸和常规页面宽度下可读。
- [x] 转为灰度后，A/B/C、边框、箭头和所有文字仍可区分。
- [x] 页脚明确写出 `Representation ≠ factorization ≠ training head ≠ deployment claim`。

## 5. 集成约束

正文图注必须说明：三条路线是常见兼容配置，而不是互斥且穷尽的模型家族；现代系统可以把 frame/chunk AR 作为外层，再在组内运行 diffusion、flow 或 masked refinement。确定性 Mermaid 应承担精确关系与可搜索文字，PNG 用于快速建立直觉。
