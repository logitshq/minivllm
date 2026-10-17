# E018 后续设计评审：保留单 token helper，只在 literal 多 token 块复用状态

2026-10-17。只读四个原始 JSON 及当前源码；没有 GPU、MLX import、模型加载或新 kernel。E018 v1 的总体重复性结论仍为 **FAIL**。本文件是下一项实验建议，不修改原实验的门或结果。

## 已有证据限定了下一步

| 进程 | T1 | T3 | T5 | T8 | GM T3/5/8 | 原门 |
|---|---:|---:|---:|---:|---:|---|
| 初筛 | 0.982453 | 1.199745 | 1.219095 | 1.578766 | 1.321746 | PASS |
| repeat 1 | 1.023888 | 1.220317 | 1.461622 | 1.530151 | 1.397486 | PASS |
| repeat 2 | 0.939084 | 1.270615 | 1.413632 | 1.478124 | 1.384694 | FAIL |
| repeat 3 | 1.066419 | 1.256273 | 1.483812 | 1.504110 | 1.410092 | PASS |

来源是 `.artifacts/m5-ultra-256gb/gdn-verify-v1/summary.json` 与三个 `gdn-verify-repeat-v1-run*/summary.json`，没有池化。660 项逐字节比较通过。repeat 2 的 T1 reference/candidate median 分别 0.2546875/0.2712085 ms，差约 16.5 µs；其三个 ABBA block 的描述性比值为 1.073/0.924/0.860。这个分布不能证明回退只是噪声，原 0.97 门确实失败。

T1 没有相邻 token 的状态读取或 launch 可消除，因而无需使用 persistent kernel。多 token 各进程收益存在，但当前数据无法分开归因于 launch、Python 建图、输入 cast、状态带宽或寄存器分配。不要把每线程 `float carried[32]` 写法当成编译器已经无 spill 的证明。

## 最小下一版：改变分派，不改 kernel 数学

建立新的实验编号和预注册，仅增加显式分派。普通 decode 的 T1 入口继续直接调用原 `gated_delta_step_metal`；它不应先进入 persistent wrapper、构建 history 或改变返回形状。多 token 的 literal verification 接口才调用现有 persistent kernel。不要把全部普通 decode 迁移到一个新的通用 backend 框架。

若统一的块接口必须接受 T1，则 T1 分支调用原 helper，再用增加长度为 1 的维度满足 `[T,B,H,D]` / `[T,B,H,D,D]` 的返回合同；这个 wrapper/view 的成本必须计入该接口的 T1 检验。不能拿原 helper 自己对自己计时，宣称实际路由没有开销。只读 observer 应记录每个请求实际走了 original/persistent 哪条路。

第一阶段可以只优化已资格的 T∈{3,5,8}；T2/4/6/7 的预算尾块显式走同一个 serial literal helper，报告路由。另一种选择是把模板扩至 T2..8，但那是新增设备编译 shape，必须先逐点资格，不能直接从 T3/5/8 外推。无论选择哪一种，在新预注册中先固定，之后不按结果变更。

这是无请求私有状态的算子：输入是明确的 q/k/v/g/beta 和 initial S，输出明确的全部 history。保留一个默认 OFF 的独立模块选择即可；不用闭包捕获 cache，不新增请求游标，不改动态块长策略，不改 SPLIT=4/VALUE_TILE=32、norm/归约顺序或 FMA 表达式。

## 多层真实输入资格

只把同一份 layer-0 输入贴上多个 layer 编号不合格。应以普通 target 路径，从固定数学/代码/中文 prompts 的真实 prefill cache 起步，运行八次正常单 token decode，在每个实际 GDN 层调用 helper 时捕获该层自己的 q/k/v/g/beta。每层只保留第一次调用的 initial S；不要为取得输入而保留八份旧 S。原始 token IDs、层编号、输入 dtype/shape、每 tensor SHA 和源/模型身份都入档。

深层输入依赖上游层的实际输出，必须跑完整 target 才能捕获。使用 teacher-forced continuation 也可做算子资格，但必须明确它不是新的 greedy rollout；若宣称 greedy，保存各步真实 argmax 及后续输入对应关系。

建议在新计划中固定一张有限矩阵，而非盲目做所有轴的笛卡尔积：

- 代表层 0/16/32/48/62（加载后断言均为 GDN），三个真实 prompts，T1/3/5/8，真实 initial S：60 cases。
- 同五层、一个固定 prompt，T3/8，加有 offset 的视图和 stride 视图；值和逻辑 shape 保持相同：20 cases。
- 同五层、一个固定 prompt，T1/8，zero 和独立随机 initial S：20 cases。
- 全部 48 个 GDN 层、一个固定 prompt、T8 的真实输入再扫描一次：48 cases（重复代表层点可保留，但不能充作独立样本）。

每 case 与原 helper serial chain 比较全部 output、全部 FP32 history、每个 keep=0..T 的独立前缀重跑，并检查 initial/input 不变。两个不同 initial state 与输入序列应交错调用同一候选，证明无隐式缓存串用。保持 bit-exact，失败保存首个层/位置/数组，不放宽为 allclose。

非连续输入必须是实际 MLX 视图，不能先人为 contiguous 后再宣称覆盖 stride。若 `metal_kernel` 自己为合法输入做 contiguous 转换，这是实际路径成本，应在两臂中自然保留并记录构造方法。

## history 生命周期是实际成本

T8 每层完整 state history 为 24 MiB，48 层约 1.125 GiB。commit 选择 `history[keep-1]` 很可能仍引用整个 history backing allocation；删除 Python 的 history 变量不自动释放它。下一块开始时可能同时保留前一块 backing 与新的输出，即仅这部分就接近 2.25 GiB。这个估算与 256 GiB 机器容量相容，但应报告实际 active/peak，不能把 commit 后的驻留写成每层 3 MiB。

第一版可以接受 view 生命周期，避免为了省内存额外复制；每次 transaction 完成清掉不再需要的 history、tape、observer 引用。若后来引入显式分离拷贝，必须证明所选 API 确实分配新 storage；`contiguous(view)` 在 view 已连续时不保证脱离大 backing。该拷贝的读写时间也应算入完整提交路径。

S history 只覆盖 GDN recurrent state。它不自动提供 conv history、target attention KV、draft context 或 committed logical position。整模 commit 必须把同一个 keep 原子应用于这些状态。GDNCache 目前是无 batch 维的 S；micro 的 `[T,1,H,D,D]` 在接入时应显式处理 batch 维，不以偶然广播隐藏 shape 错误。

## 新计时计划与采用条件

先完成新资格，再对新路由做三个固定 fresh processes。T1 仍在矩阵和非回退门内，全部无资格回退/短尾块有独立路由记录。新实验可以预注册更长的固定 ABBA schedule，例如每点 10 blocks、每个 timed observation 连续 16 次调用并在每次调用后 eval，报告总 wall/16；两臂同样执行，首次及 warm 单独保留。这样减少亚毫秒 host timer 的抖动，但仍不是冷编译或完整生成时间。不能在看到第一组结果后再调 repetitions。

记录逐层完整输出/history 的消费与 keep selection 的成本、source pins、owner/reap、资源和服务恢复。实际 production T1 路由和 literal-block T1 wrapper 若不同，应分别标注，不把两个口径混起来。每进程独立判门，保留全部负点，不与 E018 初筛或旧复测池化。

通过 micro 仍只允许进入整模 transaction 资格。当前 target chunk-WY verifier 与 serial decode 之间的差异、M4 的 QMM/broadcast-M1 资格是独立问题。仅替换 GDN kernel 不能证明完整验证器的 greedy 轨迹或 rollback 无损。必须先取得真实整模 token/state 证据，随后才测 HTTP/端到端收益。
