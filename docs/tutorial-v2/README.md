# 一章一章写一个推理引擎

从 HF 的完整模型参照开始，逐步实现自己的模型、cache、调度、Metal kernel 与投机解码。**先读01—15基础章，再读[优化教程的6篇18节](optimization/README.md)。** 原16—45保留为30个实验编号的完整档案；新短章按概念组织，不表示所有候选都已被采用。

| 你现在想做什么 | 从哪里开始 |
| --- | --- |
| 理解模型如何逐个生成token | [01 生成循环](01-一次文本生成是怎么发生的.md) → 02基础层 → 03 Attention → 04 GDN |
| 手动逐格运行教程 | [Notebook 启动说明](../../notebooks/README.md) → [第一章](../../notebooks/ch01/ch01.ipynb) → [第二章](../../notebooks/ch02/ch02.ipynb)；后面按章增加 |
| 读懂自己实现的完整引擎 | 05完整模型 → 06 Cache → 07量化 → 08调度 |
| 学习GPU与服务优化 | 09 Metal → 10投机 → 11测量 → 12—15逐步优化 |
| 看目前三家吞吐—延迟比较 | [45 三家C1/C2/C4](../experiments/45-lithos-queued-concurrency.md)；注意其负载和测试批次 |
| 一步步学习后续优化 | [6篇18节](optimization/README.md)，从测量到融合、复用、执行、投机与Pareto |
| 查某次优化的完整代码与正负结果 | [原章对应表](optimization/coverage.md) → [实验档案](../experiments/README.md) |

01—15代码保存在 `tutorial/step-by-step` 的固定tag；当前优化分支还保存后续实验快照。仓库有三套不同用途的代码：`minivllm/`是原完成版参照，`tutorial_code/`是逐章教学代码，`docs/code-versions/`与实验source目录是冻结候选。运行教学代码不会导入完成版引擎。

## 阅读顺序与真正的替换点

| 章节 | 本章内容 | 完整生成接入状态 | 固定版本 |
|---|---|---|---|
| [01 一次文本生成是怎么发生的](01-一次文本生成是怎么发生的.md) | IDs、logits、greedy、生成循环 | 完整模型使用 HF | `tutorial-01` |
| [02 模型的基础计算块](02-从ID到向量再到分数.md) | Embedding、Linear、RMSNorm、MLP、残差 | 同输入核对组件；完整模型使用 HF | `tutorial-02-v2` |
| [03 Attention](03-attention.md) | Q/K/V、GQA、partial RoPE、因果 mask、输出门 | 同输入核对完整 Attention 组件 | `tutorial-03` |
| [04 GDN](04-gdn.md) | 因果卷积、门、delta 递推、矩阵记忆 | 同输入核对完整 GDN 组件 | `tutorial-04` |
| [05 完整模型](05-model.md) | 层装配、严格加载权重、独立 forward | **完整生成第一次使用自己的模型** | `tutorial-05` |
| [06 Cache](06-cache.md) | prefill/decode、KV/GDN 状态、分块 prompt | 单请求 cache 已接入 | `tutorial-06` |
| [07 量化](07-quantization.md) | packed weights、scales、量化算子、norm 存储约定 | MLX 量化 checkpoint 已接入 | `tutorial-07-v2` |
| [08 调度与批量](08-scheduler.md) | 请求生命周期、预算、chunked prefill、decode batch | 多请求核心引擎 | `tutorial-08` |
| [09 Metal](09-metal.md) | 融合 GDN 单步、chunk 三角求解 | 自定义 kernel 接入引擎 | `tutorial-09` |
| [10 投机解码](10-speculative.md) | DFlash2、verify、accept、bonus、回滚 | greedy 单流投机路径 | `tutorial-10` |
| [11 Baseline 与实验](11-benchmark.md) | 重复测量、指标边界、同条件比较 | 核心与外部引擎各自报告 | `tutorial-11-v3` |
| [12 执行过程优化](12-从同步到流水线再到算子融合.md) | 设备 token 流水线、GDN 列分组与 norm 表达式编译 | 普通 greedy decode 已接入；各步验证分列 | `tutorial-12-pipeline` → `tutorial-12-tiling` → `tutorial-12-v1` |
| [13 单核 norm](13-把norm写成一个Metal核.md) | 保留归约拓扑、FP32舍入与BF16门控边界的Metal核 | norm接入完整模型；保留guard与旧表达式回退 | `tutorial-13-v1` |
| [14 批量 prefill](14-把多个prompt一起prefill.md) | fresh同长prompt、独立Attention/GDN状态、量化dispatch回退 | 完整六文件组合17例数值/lifecycle通过；正式AC配对另测 | `tutorial-14-batched-ops` → `tutorial-14-batch-prefill` → `tutorial-14-v1` |
| [15 局部门编译](15-把GDN的小门计算一起编译.md) | 四输入纯函数、显式opt-in与局部/全模型/端到端分层验证 | 默认off；example启用普通B1 decode；C1内部配对吞吐约+0.94% | `tutorial-15-gates-optin` → `tutorial-15-v1` |

第二章初版 `tutorial-02` 只包含较少组件，阅读使用扩展版 `tutorial-02-v2`。第七章保留初版 `tutorial-07`，修正量化 Embedding 的激活 dtype 接口后使用 `tutorial-07-v2`；旧 tag 不覆盖。

每章先读数学图与小例子，再看 HF 内部代码、我们的对应实现和 diff。第一次阅读从01开始；已理解前两章可直接进03。

## 正确性参照与性能参照

| 参照 | 主要用途 | 入口与记录 |
|---|---|---|
| HF Transformers / MPS | 原始 fp32 权重下的模型数学与 greedy 参照 | [generate.py](../../tutorial_code/generate.py)、各章 checker |
| oMLX | 已有优化的 Apple Silicon 引擎性能参照 | [baseline 工具说明](../../tutorial_code/baselines/README.md)、第十一章重复测量 |
| Lithos Metal | 第三条外部引擎；0.8B/27B GPU 实测已完成 | [第十一章的资格说明](11-benchmark.md)、[上游模型/硬件表](https://github.com/lithos-ai/lithos-metal#models-and-hardware) |

我们自己的测量入口是 [benchmark_core.py](../../tutorial_code/benchmark_core.py)。HF 第一章的无 cache 循环用于理解与正确性，不作为最终速度目标。量化模型与同 checkpoint 的参考核对，不要求它与 raw fp32 一直生成相同 token。

第11章的oMLX驱动经HTTP/SSE，手写引擎计时从IDs输入开始；该章分列两个边界，不能据此直接计算相对速度。后续第43—45章补上同为完整HTTP请求的比较。Lithos在M4上使用实际支持的后端；第11章保留最初配置及accelerator-on实验，第45章补官方FIFO接口的C1/C2/C4。其DSpark与我们的DFlash2不是同一投机方法。

核心交付的原始 IDs 可能包含终端 EOS，API completion usage 不一定包含它；第十一章另列逻辑完成数与结束原因，不能只看两个输出计数就判断生成工作相同。

## 每章的代码阅读格式

1. 先读“之前 → 本章之后”，确认新增或替换的能力。
2. 在关键节选中用 `# ← 关键①` 标出数学、形状和状态变化。
3. 展示 HF 或上一章调用内部的原计算，再看手写对应计算。
4. 明确该组件是独立核对，还是已经进入完整生成路径。
5. 阅读 `diff`，并用相邻 tag 查看真实文件变化。
6. 看各章 checker 的实际 JSON，区分有限精度数值验证、完整输出验证与性能测量。

公式统一沿用第二章的排版：行内使用 `\( ... \)`，独立公式使用单独一行的 `\[` 与 `\]`。格式修订保存为 `tutorial-format-v1`，各章代码 tag 保持不变。

节选省略训练、视觉或接口细节时会明确说明；完整代码保持可运行。结构与 kernel 参考原 `minivllm` 并注明来源，教学代码放在独立的 `tutorial_code/miniengine/`，运行时没有导入完成版引擎。

## 第一章 baseline 与第二章的实际变化

第一章的完整入口 [generate.py](../../tutorial_code/generate.py) 用 HF 加载模型，`model(input_ids, use_cache=False)` 计算 logits，我们写循环和 argmax。

第二章新增 [layers.py](../../tutorial_code/layers.py) 与 checker，分别实现基础组件、做同输入核对，没有修改 `generate.py`。完整第二章 checker 以固定 tag 为准：

```bash
git show tutorial-02-v2:tutorial_code/check_layers.py
git diff tutorial-01 tutorial-02-v2 -- tutorial_code/generate.py
git diff tutorial-01 tutorial-02-v2 -- tutorial_code/layers.py tutorial_code/check_layers.py
```

第五章新增 [generate_mlx.py](../../tutorial_code/generate_mlx.py) 才换掉完整 forward。保留 HF 入口让两个结果可以继续对照；使用 HF tokenizer 只是输入输出边界，核心模型与引擎只处理 token IDs。

## 用固定版本阅读，避免后面代码混进前面

每章在同一份 `tutorial_code/` 上继续修改，并创建 commit 与 tag。后续优化不改变旧 tag 指向的代码。

```bash
# 首次从远端获取已有章节标签；不覆盖同名本地tag
git fetch origin --tags

# 查看第三章的固定代码
git show tutorial-03:tutorial_code/miniengine/model/attention.py

# 看第三章到第四章新增了什么
git diff tutorial-03 tutorial-04 -- tutorial_code/miniengine

# 第八章基于第七章的 dtype 修正版
git diff tutorial-07-v2 tutorial-08 -- tutorial_code/miniengine
```

正文里的相对源码链接打开的是**你当前checkout**中的文件，可能已包含后续章节的修改。复现本章必须看标注的tag；`git diff`只展示差异，不会替你切换运行版本。

建议保留当前目录，在一个新的同级worktree运行固定版本：

```bash
# 在本仓库根目录执行；目标目录应尚不存在
# worktree只写出代码，不会下载权重或运行模型
git worktree add --detach ../minivllm-ch03 tutorial-03
cd ../minivllm-ch03
uv sync --frozen
# 模型已按第01章下载后，再运行本章checker
uv run python -m tutorial_code.check_attention
```

只阅读时用 `git show` 即可。若选择切换当前目录，先保存本地修改，并记录原分支；学习结束后回到原分支，不默认要求跳到另一条开发线。

模型权重留在本地 HF cache 或模型目录，不提交到 Git。章节报告记录实际 checkpoint 路径、精度和验证条件；tag 固定代码，报告固定已运行的案例，不宣称覆盖所有输入。

## 运行环境

命令从仓库根目录或上面的固定版本worktree执行，默认使用该目录的锁文件和环境：

```bash
uv sync --frozen
uv run python -m tutorial_code.check_attention
```

模型下载方式见第01章；量化checkpoint见第07章。文中 `/Users/ryanli/...` 是原测量机器的路径记录，换机器需要替换，不能把本机已有模型当成仓库自带。oMLX/Lithos使用独立环境，见[baseline说明](../../tutorial_code/baselines/README.md)。核心教程处理token IDs，tokenizer与HTTP serving作为输入输出适配；后续服务实验才测完整HTTP边界。

## 基础篇之后：六篇优化教程

[优化目录](optimization/README.md)按“先测清楚 → 融合 → 复用 → 减少主机工作 → 投机 → 完整服务比较”组织成6篇18节，每节都有前后对照、不变量、实测边界与理解题。六篇是主题层级，不是把30章硬压成六个大章。

[原章对应表](optimization/coverage.md)列出16—45全部30个编号的知识点去向，并明确哪些复杂推导、支线和完整数据仍需打开原文。原章、旧tag、源码和失败记录均保留；只读18节不会逐项复述全部日志。[实验档案地图](../experiments/README.md)继续解释章号、V编号和tag，以及ICB、CAP等术语。

[46论文到实验](../experiments/46-next-ideas-from-papers.md)和[47近期论文精读](../experiments/47-recent-paper-reading.md)是待验证方案，尚无新性能点。历史章节的STOP/PASS描述当时的实验，后续更新由目录与章首链接说明；不覆盖旧tag或原始结果。
