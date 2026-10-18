# 用 Jupyter 一格一格运行教程

目前只提供**第一章**：观察文字怎样变成 token IDs，模型怎样给出 logits，再由 greedy 逐个生成 token。后续章节的 Notebook 尚未制作，不要把现有章节文档当作已有 Notebook。

## 1. 从项目根目录启动

在包含 `pyproject.toml` 和 `uv.lock` 的仓库根目录运行：

```bash
uv sync --frozen
# 首次下载；已有这个固定版本的本地缓存时可跳过
uv run --frozen hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17
uv run --frozen --with-requirements notebooks/requirements.txt python -m jupyterlab notebooks
```

终端会给出本机 JupyterLab 地址。打开后，先进入 `ch01/`，选择主教程。模型权重不随仓库提供；上面的下载步骤需要网络和足够的磁盘空间。Notebook 只读本地缓存，不会自动下载模型。

Notebook 打印的解释器路径可能位于 uv 临时目录，这是额外 Jupyter 依赖的覆盖环境，属于正常现象；模型依赖仍来自项目的 `uv.lock`。

| 文件 | 怎样使用 |
| --- | --- |
| [ch01/ch01.ipynb](ch01/ch01.ipynb) | 主教程：先读问题，再运行短代码，观察形状、分数和新 ID。 |
| [ch01/exercise-solutions.ipynb](ch01/exercise-solutions.ipynb) | 独立练习答案：先自己做，再打开核对。不要和主教程同时加载模型。 |

## 2. 每次只前进一步

1. 从主教程顶部开始，先读当前单元格的问题。
2. 选中代码单元格，按 **Shift+Enter** 执行并移到下一格。左侧显示 `[*]` 时先等它结束。
3. 看清输出，试着预测下一格的输入长度、logits 形状和选择的 ID，再继续。
4. 修改 prompt 或生成长度后，从对应设置单元格向下重新运行。要检查整套步骤是否独立可复现，用 **Kernel → Restart Kernel and Run All Cells**，从空状态顺序复跑。

主教程把 `forward → 最后位置的 argmax → 拼回输入` 拆开执行，再写成循环。它不是用 `%run` 把整个生成脚本跑完；重点是亲手观察每一步。页面上已有输出也不代表当前 kernel 已执行过前面的代码。

答案本的8-token与12-token对照会分别调用冻结的参考脚本，检查共同前缀；这是额外核对，不代替主教程里的逐格计算。

## 3. 运行完要关闭 kernel

模型留在 kernel 的内存中。关闭浏览器标签页不等于停止 kernel。结束学习或准备切到答案本时，在 **Kernel** 菜单关闭当前 kernel，或在左侧 **Running** 面板找到它并关闭；这会释放该进程持有的模型内存。回到终端按 **Ctrl+C** 可停止 JupyterLab。

## 4. 本章固定哪一份 baseline

- 教学代码对应 `tutorial-01`，commit 为 `4f9952da4537a2503477365b789a276e3ff8033d`。
- 模型是 `Qwen/Qwen3.5-0.8B`，revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。
- 使用 HF 参考模型、fp32、greedy 和 `use_cache=False`，每轮重算整个前缀。这是观察生成过程的正确性参照，不是性能测量。

本目录的 [reference/generate.py](ch01/reference/generate.py) 和 [参考轨迹](ch01/reference/chapter01-trace.json) 保存第一章固定版本，身份见 [manifest.json](ch01/reference/manifest.json)。启动 Notebook **无需 checkout 旧 tag**；旧 tag 本身不含这些新 Notebook。也不要用当前分支不断更新的 `tutorial_code/generate.py` 替换这里的参照。

后续制作每章 Notebook 时，仍按[章节目录](../docs/tutorial-v2/README.md)标出的独立 tag 固定代码；不能把后面新增的 cache 或默认参数带回前面的课程。第一章的文字讲解见[一次文本生成是怎么发生的](../docs/tutorial-v2/01-一次文本生成是怎么发生的.md)。

教学组织参考 Sebastian Raschka 的 *LLMs from Scratch*：[第2章 Notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb)、[第3章 Notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb)。这里采用相同的学习节奏：先看问题、运行短单元格、观察输出，再预测下一步；模型与实现仍以本教程的固定版本为准。
