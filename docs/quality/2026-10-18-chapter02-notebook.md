# 第二章 Notebook 执行与来源检查

本次只增加第二章：主教程24个代码格，独立练习答案4个代码格。第一章正在由用户手动运行，其本地输出变化未纳入此次提交。原模型生成、调度和性能代码没有修改。

## 来源与学习顺序

主教程依次显式定义 Embedding、Linear、RMSNorm、SiLU、MLP、FeedForwardBlock。先用小矩阵手算，再比较真实0.8B模型的输入、权重和参考输出，最后列明哪些部分仍由 HF 执行。

固定版本为 `tutorial-02-v2`。这是 annotated tag：tag 对象是 `e55e7c59ade307c00e690dc8abcbce933c8464ee`，解引用后的代码 commit 是 `f3b203ba44b135693bf0c6d6dbc82dd8072e9a63`。两者在 [manifest](../../notebooks/ch02/reference/manifest.json) 中分别记录。

四份冻结文件逐字等于该 commit 的源文件。六个 Notebook 组件定义的 AST 与冻结 `layers.py` 一致。`generate.py` 在第一章与本章的 tag 之间没有变化。历史检查报告保留原值，新输出另存于 Notebook 和[检查 JSON](2026-10-18-chapter02-notebook.json)。

## 真实验证

本机运行前报告76%空闲内存；已有第一章 Jupyter 服务和 kernel，未关闭或更改。第二章独立运行验证 kernel，使用 MPS、FP32、MLX 0.32.3、torch 2.14.1、transformers 5.19.0。模型 revision 继续固定为 `2fc06364715b967f1860aea9cf38778875588b17`。

- 两份 Notebook 从空 kernel 顺序执行通过，保存真实输出。
- 小例子核对查表、内积、RMSNorm、SiLU、MLP；零 down 投影验证残差加回原始输入。
- 真实对照检查形状、FP32 dtype、有限值和最大绝对误差。八项组件/组合检查全部通过，最大误差为 `7.6293945e-6`，低于原章的 `1e-3` 阈值；Embedding误差为0。
- 最后 norm+head 的全部位置 argmax 与HF一致，最后位置选择79852，与第一章首token一致。
- HF采集辅助模块不导入手写组件；真正重算由Notebook当前定义的类完成。更改类后需重新创建实例，建议重启并全部运行。
- 六组hook在finally中移除；返回的NumPy数据独立于模型存储。HF完整模型在采集完成后释放，Notebook末尾再释放真实权重与中间数组。
- 浏览器检查执行后HTML中的数学步骤、代码、真实结果、范围说明和练习入口，未发现排版错误。
- 独立agent完成只读交叉审阅：当前定义与真实对照关系、残差输入位置、dtype/finite检查、hook清理、全局大数组引用和本地链接均通过；该审阅没有运行GPU。

以上是有限输入的教学正确性检查，不是完整模型替换、通用正确性证明或性能复测。没有新增Pareto数据或更改现有优化资格。

从项目根目录复跑，使用[同一套环境](../../notebooks/README.md)：

```bash
uv run --frozen --with-requirements notebooks/requirements.txt \
  python -m jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/ch02/ch02.ipynb
uv run --frozen --with-requirements notebooks/requirements.txt \
  python -m jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/ch02/exercise-solutions.ipynb
```
