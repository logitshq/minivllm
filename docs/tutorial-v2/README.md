# 一章一章写一个推理引擎

当前完成第一、二章。每次读懂并验证一章，再写下一章。

教学分支是 `tutorial/step-by-step`，工作目录是 `/Users/ryanli/code/minivllm-tutorial`。原来的 `/Users/ryanli/code/minivllm` 保持在 `main`，已有引擎和旧教程都保留。

## 从第一章开始

[01 一次文本生成是怎么发生的](01-一次文本生成是怎么发生的.md)

[02 模型的基础计算块](02-从ID到向量再到分数.md)

第一章用 HF 做完整模型 baseline，自己写生成循环。第二章在 MLX 中实现基础计算块，用相同输入与真实权重逐项对照；生成循环暂时仍调用 HF。教学代码不导入已经完成的 `minivllm` 引擎。

## 代码怎样逐章保存

| 章节 | 固定版本 | 本章代码 |
|---|---|---|
| 01 | `tutorial-01` | [tutorial_code/generate.py](../../tutorial_code/generate.py) |
| 02 初版 | `tutorial-02` | Embedding、Linear 的固定版本 |
| 02 扩展版 | `tutorial-02-v2` | [layers.py](../../tutorial_code/layers.py)、[check_layers.py](../../tutorial_code/check_layers.py)：加上 norm、MLP、残差 |

每章在同一份 `tutorial_code/` 上接着改，并创建一个 commit 和 tag。读某章时以该章 tag 为准；后续章节的优化不会改变旧 tag 指向的代码。新增能力与性能优化都会在章节开头说明。

这个分支从已经完成的引擎仓库创建，因此 Git 历史中仍有原项目。逐章复写的代码单独放在 `tutorial_code/`，从第一章的参考生成循环开始积累。`minivllm/` 是可对照的完成版。

查看第一章代码：

```bash
git show tutorial-01:tutorial_code/generate.py
```

查看本章改动：

```bash
git show tutorial-01 -- docs/tutorial-v2 tutorial_code README.md
```

如需切换整个教学目录到固定版本，先确保本地修改已经保存，再执行：

```bash
git switch --detach tutorial-01
```

继续编写教程时回到：

```bash
git switch tutorial/step-by-step
```

换到其他机器时执行 `uv sync`，再按章运行示例。模型权重存放在 HF cache，不提交到 Git；每次示例 trace 记录实际使用的模型 snapshot 路径。Git tag 固定代码与依赖锁文件，trace 记录运行条件。
