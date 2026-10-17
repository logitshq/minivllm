# 一章一章写一个推理引擎

从 HF 的完整模型参照开始，逐步实现自己的模型、cache、调度、Metal kernel 与投机解码。01—15 的代码与讲解放在同一教学分支，每章保存固定版本。

教学分支是 `tutorial/step-by-step`，工作目录是 `/Users/ryanli/code/minivllm-tutorial`。原 `/Users/ryanli/code/minivllm` 保留在 `main`；完成版与旧教程仍可作为源码参照。

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

下一步从第三章读起。可以先只读前半章的数学图与小例子，再看 HF 内部代码、我们对应的实现和 diff；不需要一次读完后面的调度与 kernel。

## 两种验证，加一个候选性能参照

| 参照 | 主要用途 | 入口与记录 |
|---|---|---|
| HF Transformers / MPS | 原始 fp32 权重下的模型数学与 greedy 参照 | [generate.py](../../tutorial_code/generate.py)、各章 checker |
| oMLX | 已有优化的 Apple Silicon 引擎性能参照 | [baseline 工具说明](../../tutorial_code/baselines/README.md)、第十一章重复测量 |
| Lithos Metal | 第三条外部引擎；0.8B/27B GPU 实测已完成 | [第十一章的资格说明](11-benchmark.md)、[上游模型/硬件表](https://github.com/lithos-ai/lithos-metal#models-and-hardware) |

我们自己的测量入口是 [benchmark_core.py](../../tutorial_code/benchmark_core.py)。HF 第一章的无 cache 循环用于理解与正确性，不作为最终速度目标。量化模型与同 checkpoint 的参考核对，不要求它与 raw fp32 一直生成相同 token。

oMLX 当前驱动经 HTTP/SSE；手写引擎计时从 IDs 输入开始。两个计时边界在报告中分列，不把不同边界的数字相除称为优化倍数。Lithos 在 M4 Max 的上游状态是未测 native fallback；它的 DSpark 与这里的 DFlash2 也不是相同实验，真实 0.8B/27B 本机报告在第十一章扩展版分别列出，未测其他场景不填成绩。

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
# 查看第三章的固定代码
git show tutorial-03:tutorial_code/miniengine/model/attention.py

# 看第三章到第四章新增了什么
git diff tutorial-03 tutorial-04 -- tutorial_code/miniengine

# 第八章基于第七章的 dtype 修正版
git diff tutorial-07-v2 tutorial-08 -- tutorial_code/miniengine
```

如果当前目录有自己的修改，优先用 `git show` 阅读固定版本。要切换整棵目录时先保存本地修改，再 `git switch --detach tutorial-03`；继续开发时回到 `git switch tutorial/step-by-step`。

模型权重留在本地 HF cache 或模型目录，不提交到 Git。章节报告记录实际 checkpoint 路径、精度和验证条件；tag 固定代码，报告固定已运行的案例，不宣称覆盖所有输入。

## 运行环境

下面命令里的 `python` 指已有 MLX/Torch 环境的解释器。本机路径是 `/Users/ryanli/code/minivllm/.venv/bin/python`；例如运行第三章：

```bash
cd /Users/ryanli/code/minivllm-tutorial
/Users/ryanli/code/minivllm/.venv/bin/python -m tutorial_code.check_attention
```

其他机器先按照仓库依赖准备环境和所需 checkpoint；oMLX 使用独立安装环境，详见 baseline 说明。教程核心不要求实现 tokenizer 或启动自己的 HTTP 服务。

## 第三条 baseline 的当前准备状态

Lithos 已从完整官方源码 `5485ad6bac0a5649d66a347f3e01a655ea29d937` 构建并安装到独立 Python 3.12 环境；没有本地 kernel 或模型补丁。0.8B/27B 的 CPU checkpoint 格式、19 个输入 IDs 与完整 EOS 集合均已核对，复现步骤见 [Lithos README](../../tutorial_code/baselines/lithos-README.md)。安装与准备记录用 `baseline-lithos-setup-v1` 保存。

用户解锁后关闭空闲应用、保留正在执行的远端 CUDA 工作与 Codex，随后独占 GPU 串行运行。0.8B/27B 的 smoke 与三次重复测量均已完成，报告标记为真实 target-only 本机测量；`tutorial-11-v2` 保存扩展结果。原 `tutorial-11` 和 `baseline-lithos-setup-v1` 的资格/CPU 准备记录保留，不覆盖旧 tag。

本机 M4 官方 accelerator-on 的独立实验已记录，见 [实验说明](../../tutorial_code/baselines/m4-accelerator-README.md)；完整 M5 配置被官方设备检查拒绝，未绕过验证。推荐第十一章阅读版本为 `tutorial-11-v3`，默认 profile/off 与原 `tutorial-11-v2` 记录保留。

## 独立实验阅读（未采用）

[18：先确认优化真的更快](18-先确认优化真的更快.md)记录GDN四投影与MLP两投影的输出行拼接。两条路线均未通过收益门，当前26个引擎源码及V19 default OFF保持不变；章节编号只作实验阅读索引，不表示已集成的新能力。

[19：把量化矩阵乘拆开看](19-把量化矩阵乘拆开看.md)记录独立NVFP4 QMV几何实验：原QMM→R4数值控制→每SIMD行数4→2；严格数值通过、性能STOP，未接入引擎。

[20：沿实际实现读三条执行路径](../experiments/20-donor-implementation-map.md)对照实际版本的oMLX、手搓引擎与Lithos，区分已具备ahead、待验证conv预处理、ICB运行时工程及allocator未知项；仅源码阅读，无新增收益结论。

[21：把GDN真卷积、SiLU和原始tail合成一核](../experiments/21-gdn-conv-fusion.md)记录V23的Y4证据、synthetic严格数值与局部micro；host-wall中位减少12.21%，尚未接入引擎，真实preconv与完整模型/E2E资格仍待完成。

[22：融合和图复用的真实 HTTP 检查](../experiments/22-combined-decode-screen.md)补齐组合版数值、185-token生命周期和六新进程C1测试；X+1.54%/Y+0.76%，收益不足，未采用，继续优化。

[23：继续拆解 kernel 与真实 GPU 诊断](../experiments/23-kernel-tests-and-gpu-diagnostics.md)记录权重重排、SIMD递推、真gate/up融合的严格结果、负结果及整模失败；解释为什么 Metal buffer 时间戳不能直接作为单算子占比。

[27：少搬 state，再把真实 Metal 命令留在 ICB 里](../experiments/27-native-icb-pingpong.md)保存48层S轮换、独立buffer克隆负结果、实际双bank Metal ICB及全数值/B-only计时证据；36.188ms/步、27.633tok/s未作服务或Pareto获胜声明。

[28：把卷积、归一化复用和state保留接到真实ICB](../experiments/28-native-gdn-composition.md)保存V39/V40/V44的严格完整资格、V40绑定失败/修复与B-only计时；候选默认关闭，无HTTP/Pareto采用结论。

[29：把原生 decode 接进调度器，再测真实请求](../experiments/29-native-runner-http.md)说明原生状态的生命周期与普通 runner 回退，保留 V42 负结果和 V45 未完成测试及全部原始证据。

[30：提前提交一步](../experiments/30-native-ahead.md)对照同步 runner 与 depth2 FIFO，说明状态与输出的所有权、提前停止和实际数值范围；真实 HTTP 仍未超过已测前沿。

## 当前继续阅读

[38：投机事务与量化乘法](../experiments/38-spec-transaction-and-qmm.md)保留整模旧序列失败、官方MTP较慢的配对结果及逐算子诊断。

[39：保留逐行量化乘法的结果](../experiments/39-verify-broadcast-m1.md)从旧代码到广播M1逐步对照，附固定候选与diff；局部497项通过，整模仍待新资格。[进展图与台账](../progress/README.md)保存已验证结果和负结果，章节持续迭代。

[40：整模通过后再测吞吐和延迟](../experiments/40-m1-verify-and-bounded-loader.md)补上完整投机路径资格与加载资源边界。

[41：HTTP 传输选项](../experiments/41-native-http-nodelay-pilot.md)和[42：短请求投机解码](../experiments/42-startup-warm-short-request.md)分别保存一次无收益和一次明显变慢的真实对照。

[43：新的三家 C1 对照](../experiments/43-fresh-c1-throughput-latency.md)以完整请求 latency 和总输出 throughput 作图：本方三轮均领先；当时缺少的同场并发点已在第44章补齐。正文对照旧 native15、当前 CAP64/direct 与本轮测量代码的变化。

[44：完整吞吐—延迟前沿](../experiments/44-fresh-full-throughput-latency.md)先解释 C1/C2/C4 与非支配点，再展示固定启动配置如何接入真实 batching。21 次全新加载的同轮结果中，本方 C1/C2/C4 覆盖全部已测外部点；图包含全部重复测量。结论限定于该模型、短负载及已测配置，未验证的 kernel 不计入收益。

另读[共享权重多行 M1 补记](../code-versions/44-shared-w-m1/LOCAL-NUMERIC.zh.md)：从已经正确的 broadcast-M1 出发，尝试在多个 token 之间复用读取的权重，同时保留每行累加顺序。代码和局部数值记录已保存；它尚未取得整模或性能资格。

[45：补测 Lithos 的并发请求](../experiments/45-lithos-queued-concurrency.md)对照 OpenAI 与 Ollama 请求代码，解释客户端并发、服务端排队和 GPU batching 的区别。官方排队接口实测 C1/C2/C4，吞吐基本持平而完整延迟增加；三家九点图保留测试批次标签。

[46：读新论文，找下一处值得替换的代码](../experiments/46-next-ideas-from-papers.md)区分已实现机制与真正的新改动，用前后伪代码解释 decoded-W 复用，并拆开调度排序、时间预算、投机成本策略。包含近期论文与作者代码；候选尚未实现或计入性能图。
