# Contributing

感谢你帮助维护 Video Generation 101。本仓库优先收录能澄清技术脉络、可以追溯到可靠来源的内容。

## 可以贡献什么

- 关键论文、项目页和官方代码。
- 开放模型、数据集和 benchmark。
- 论文年份、作者、链接或技术描述的勘误。
- 可复现的评测结果和失败案例。
- 对术语、时间线或 world model 定义的改进。

## 提交资源前

请确认：

1. 链接指向论文、作者项目页、官方组织或原始代码仓库。
2. 说明该资源为什么值得收录，而不只是声称“效果最好”。
3. 涉及模型时注明代码、权重和许可证是否分别开放。
4. 涉及性能时注明模型版本、数据、输出规格和评测方法。
5. 不提交来源不明的权重、盗版数据或绕过安全限制的方法。

## 推荐的条目格式

```markdown
- **项目或论文名称** — 作者/组织，年份。
  Paper · Project · Code：填写对应的原始来源链接。
  一句话说明它改变了什么，以及证据来自哪里。
```

## 修改流程

1. Fork 本仓库并创建新分支。
2. 只修改与本次贡献相关的文件。
3. 检查 Markdown 链接和相对路径。
4. 在 Pull Request 中说明新增内容、选择理由和来源。

## 风格

- 中文解释为主，论文和模型保留官方英文名称。
- 区分论文结论、官方声明和贡献者推断。
- 避免“真正理解世界”“全面超越”等无法验证的绝对表述。
- 对 closed model 的内部 benchmark 保持明确归因。
- 日期统一使用 `YYYY-MM` 或 `YYYY-MM-DD`。

## Commit 建议

```text
docs: add paper on action-conditioned video prediction
fix: correct publication year for MAGVIT
resources: add official model repository
```

提交贡献即表示你有权提供相关内容，并同意新增的原创文本按仓库许可证发布。
