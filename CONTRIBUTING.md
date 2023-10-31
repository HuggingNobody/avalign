# 贡献指南

欢迎参与 avalign！这份指南说明本地开发与提交 PR 的约定。

## 本地开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 核心 + 开发工具
pip install -e ".[dev,torch]"    # 如需运行 torch 相关测试
```

## 提交前检查

```bash
ruff check .
ruff format --check .
mypy
pytest
```

或者装上 `pre-commit` 让钩子自动跑：

```bash
pre-commit install
```

## 约定

- **核心层保持纯 NumPy。** 不要在 `import avalign` 的路径上引入 `torch`；
  神经网络相关代码放到 `models/` 或 `train/`，并通过惰性加载暴露。
- **测试跟着实现走。** 新增功能请配套单元测试；torch 相关测试用
  `pytest.importorskip("torch")` 保证无 torch 环境下会自动跳过。
- **提交粒度要小。** 一个提交只做一件事，信息写清楚“做了什么、为什么”。
- **公开函数写 docstring**，注释只在“为什么”不明显时才加。

## 提交 PR

1. 从 `main` 切出特性分支。
2. 保证上面的检查全部通过。
3. 如有必要，更新文档与 `CHANGELOG.md` 的 `Unreleased` 段。
4. 在 PR 描述里说明动机与改动范围。
