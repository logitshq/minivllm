# 一章一章写一个推理引擎

教程按章累积实现；每章保留可运行代码、数值验证和固定 Git tag。

教学分支是 `tutorial/step-by-step`，工作目录是 `/Users/ryanli/code/minivllm-tutorial`。原来的 `/Users/ryanli/code/minivllm` 保持在 `main`，已有引擎和旧教程都保留。

## 从第一章开始

[01 一次文本生成是怎么发生的](01-一次文本生成是怎么发生的.md)

[02 模型的基础计算块](02-从ID到向量再到分数.md)

第一章用 HF 做完整模型的正确性参照，自己写生成循环。第二章在 MLX 中实现基础计算块，用相同输入与真实权重逐项对照；生成循环暂时仍调用 HF。教学代码不导入已经完成的 `minivllm` 引擎。

## 两条 baseline 的分工

| 参照 | 用途 | 入口 |
|---|---|---|
| HF Transformers | 复写模型数学、核对 logits 和 greedy；第一章无 cache 循环是教学版本 | [generate.py](../../tutorial_code/generate.py) |
| oMLX | 对照已有优化的 Apple Silicon 引擎性能：单流、并发、cache、投机 | [运行与配置说明](../../tutorial_code/baselines/README.md) |

HF 的教学循环不作为最终速度目标。后面性能对照以 oMLX 为主，使用与手写引擎相同的 MLX checkpoint 和测试条件。oMLX 的请求接口只用于测量外部 baseline，HTTP serving 不成为核心推理引擎的学习前提。

每章既展示“原始数学实现 → 手写对应代码”，也交代该组件在优化引擎中的位置。正确性通过与性能达到目标是两项分别验证的结果。

当前两条都能运行：HF 已有固定参考结果；oMLX 官方预编译版本在独立环境安装完毕，已通过同一 19-token 输入的 8-token 功能验证。正式 27B 持续性能测试留到完整引擎阶段。

## 每章的代码阅读格式

后续章节沿用第一、二章的标记方式：

1. 开头列出“之前 → 本章之后”，说明新增能力、替换范围或优化目标。
2. 在最小关键片段上用 `# ← 关键①` 等注释标出要读的行，旁边解释形状或数学含义。
3. 给出 baseline 的文件、类和函数，摘出框架原始计算，再展示手写对应计算；高亮每个替换点。仅展示一行框架调用不够，要让读者看见调用内部原本做了什么。
4. 明确替换状态：独立实现、同输入对照通过、已接入生成路径、仍由 baseline 执行。涉及组件重写时，先有同输入正确性证据；接入后再验证完整生成。
5. 用 `diff` 代码块摘出本章新增或修改的核心行，再给出相邻固定版本的 `git diff` 命令。
6. 说明怎样验证。新增组件比较正确性；性能优化同时报告相同条件下的前后速度、内存或延迟。

标记和节选放在教程正文，完整代码保持可运行。代码以各章 tag 为准；文档的解释和阅读标记可以继续完善，已有 tag 的内容保持固定。

## 第一、二章之间的实际代码变化

第一章的完整 baseline 入口是 [generate.py](../../tutorial_code/generate.py)：`AutoModelForCausalLM.from_pretrained` 加载 HF 模型，`model(input_ids, use_cache=False)` 计算 logits，我们写循环和 argmax。

第二章新增 [layers.py](../../tutorial_code/layers.py) 和固定版本的 [check_layers.py](../../tutorial_code/check_layers.py)，分别实现基础组件、做同输入验证。第二章没有修改 `generate.py`，所以完整生成仍走 HF。

```bash
# 没有输出：第一章到第二章没有改变完整生成入口
git diff tutorial-01 tutorial-02-v2 -- tutorial_code/generate.py

# 显示新增组件和验证代码
git diff tutorial-01 tutorial-02-v2 -- tutorial_code/layers.py tutorial_code/check_layers.py
```

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

## 后续章节

- [03 Attention](03-attention.md) — `tutorial-03`
- [04 GDN](04-gdn.md) — `tutorial-04`
- [05 完整模型](05-model.md) — `tutorial-05`
- [06 增量cache](06-cache.md) — `tutorial-06`
- [07 量化权重](07-quantization.md) — `tutorial-07-v2`（保留原 `tutorial-07`）
