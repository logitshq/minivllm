# 第一章 Notebook：来源、执行与审读

按用户要求，先完成第一章的可交互版本。主教程有18个代码单元格；独立答案有5个。没有制作后续章节的 Notebook，也没有修改引擎或历史 tag。

## 参考了什么

实际读取 Sebastian Raschka 的 GitHub 文件内容，采用“短讲解 → 小段代码 → 可观察输出 → 扩展为完整实现”的组织方式，以及独立练习答案。中文内容、例子和流程图围绕本项目重新编写，没有复制该项目的模型实现或插图。

| 文件 | 本次读取的 Git blob |
| --- | --- |
| [ch02.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb) | `53241d116966f19c8ab23a3a905a551496be4db1` |
| [ch03.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb) | `983bbe62c1061ae5200bc2a9fc0f7db827d1dc35` |
| [exercise-solutions.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/exercise-solutions.ipynb) | `be23260d2f77353ca8c92cb58e12a04436f1a665` |

上游链接指向可变的 main；blob 标识本次实际读到的文件版本。没有把 blob ID 当作 commit ID。

## 固定的本方 baseline

`tutorial-01` / `4f9952da4537a2503477365b789a276e3ff8033d`。原始脚本和历史 trace 精确复制到 [reference](../../notebooks/ch01/reference/manifest.json)，并记录原路径和 SHA-256。Notebook 从头显式执行 HF 前向、argmax 和拼接，没有导入当前完成版引擎。

主教程先用不修改原输入的一次 forward 观察输出，然后从原 prompt 的副本开始完整循环。模型加载格释放已有 model；结束格释放模型。独立答案顺序运行两个短命子进程，避免同时保留两份模型。

## 实际检查

2026-10-18，本机48 GiB内存，运行前系统报告78%内存空闲，未发现其他 Python / oMLX / Lithos 测试进程。使用 MPS、FP32、Python 3.13.14、torch 2.14.1、transformers 5.19.0、huggingface_hub 2.2.0。模型固定为 `Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`。

- 两份 Notebook 从空 kernel 按顺序执行，全部通过；真实输出随 Notebook 保存。
- 默认 prompt 的19个输入 ID 与历史 trace 一致。logits 从 `[1,19,248320]` 增至 `[1,26,248320]`。
- 主教程输出8个 ID：`[79852,18887,220,98435,98618,103806,95779,103725]`，与冻结脚本的实跑输出及历史 trace 一致。
- 冻结脚本生成12个 ID 的前8个与8-token运行一致；后4个为 `[111892,95726,108777,111734]`。
- 检查形状、最终输入拼接、单步与循环首个结果一致、EOS只在最后出现、提前退出条件。
- 另一 agent 只读核对 Notebook 的语义、状态重跑方式、固定源码和本地链接；主作者复查启动路径与下载步骤。
- 执行后导出 HTML，并在本地浏览器检查首屏、关键循环、真实输出、练习与章节对照表的排版。首次发现缺少进度条组件产生警告，补充锁定的 `ipywidgets` 后重跑。

这些是小模型的教学与正确性检查，不是性能数据，不改变三家 Pareto 比较或任何优化候选资格。历史 ID 对照在教学本中显示布尔结果；修改 prompt 时跳过。换设备或依赖可能需要重新分析数值差异，不能仅以 Run All 无异常推断历史 ID 对齐。

从项目根目录复跑（先按 [启动说明](../../notebooks/README.md)下载权重）：

```bash
uv run --frozen --with-requirements notebooks/requirements.txt \
  python -m jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/ch01/ch01.ipynb
uv run --frozen --with-requirements notebooks/requirements.txt \
  python -m jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/ch01/exercise-solutions.ipynb
```

执行只验证运行；还应打开 Notebook 检查输出内容与可读性。完整静态和输出检查摘要见 [JSON](2026-10-18-chapter01-notebook.json)。
