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

## 技术手册写作要求

- 方法章节先定义范围、前置知识和输入输出，再说明机制、操作步骤、配置、结果解释和故障处理。
- 标题使用术语或操作名称；修订过程、检索流水和编辑争论写入 `sources/`。
- 数据集、模型、标准与基准名称必须有一手来源；自拟方案使用“实验设计示例”，注明尚未运行。
- 引用保留官方标题，分别记录预印本初次公开和正式发表年份。数值注明统计单位和实验条件。
- 图示说明适用范围。概念图不能当作模型输出或实验结果；正文与图内术语应一致。
- 通用约定与配置记录模板见[手册使用说明](docs/manual-guide.md)。

## 提交前检查

```sh
python3 -m pip install -r requirements-docs.txt
python3 scripts/verify_citations.py --offline
python3 scripts/build_site.py --strict
python3 scripts/check_manual.py --site _site
python3 -m unittest discover -s scripts/tests -p 'test_manual*.py'
```

涉及引用变更时，运行 `python3 scripts/verify_citations.py --json citation-findings.json --coverage citation-coverage.json`。在线检查应报告实际查询覆盖率；网络失败不代表文献不存在，也不能记为核验通过。检查只覆盖脚本列明的元数据与结构规则，不替代对论文机制和结果的阅读。

## Commit 建议

```text
docs: add paper on action-conditioned video prediction
fix: correct publication year for MAGVIT
resources: add official model repository
```

提交贡献即表示你有权提供相关内容，并同意新增的原创文本按仓库许可证发布。

### 公式渲染检查

公式使用行内 `$...$` 或块级 `math` 代码围栏。构建时会对围栏中的 HTML 特殊字符转义；不要手动把 TeX 中的 `<` 改成 HTML 标签或实体。发布前执行：

```bash
python3 -m unittest discover -s scripts/tests -p 'test_math*.py'
python3 scripts/build_site.py --strict
python3 scripts/build_math_audit.py --output /tmp/vg101-math-audit
python3 -m http.server 8766 --bind 127.0.0.1 --directory /tmp/vg101-math-audit
```

在浏览器打开 `http://127.0.0.1:8766/audit.html`，等待状态显示“完成”。检查覆盖所有已构建页面，在 1280 px 和 390 px 宽度下核对公式数量、MathJax 错误、残留源码和不可滚动的越界内容。`problemPages` 必须为 0；结果 JSON 可从页面保存。另需抽查复杂公式的视觉显示，并点击站内链接确认即时导航后仍正确渲染。此检查验证网站渲染，不替代公式数学含义的审校。
