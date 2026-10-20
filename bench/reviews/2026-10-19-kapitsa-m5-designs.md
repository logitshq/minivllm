# Kapitsa 与 M5 tensor operations：可移植设计审阅

2026-10-19。仅通过网页读取官方仓库和论文；没有 clone、build、模型执行或本机 CPU/GPU 测试。此笔记不修改 E045 或教程 036 的结果。

结论：优先借鉴**按并发选择验证深度，以及验证与 recurrent-state 提交分离**。Metal 4 矩阵单元是另一种算术路径，值得小范围数值筛查，但不是现有 NVFP4 逐字节合同下已成立的替换。

## 读到的源与比较边界

- Kapitsa 当前网页为 `main`，读取日期如上；[CMakeLists.txt](https://github.com/nace-ai/kapitsa/blob/main/CMakeLists.txt) 标记版本 0.1.38。本轮网页没有给出可核验的 HEAD SHA，不能把这些活动分支链接称为已封存实现。
- [架构文档](https://github.com/nace-ai/kapitsa/blob/main/docs/architecture.md) 明确 Qwen 后端读取 affine quantization；Nemotron 路径另外支持 NVFP4。它还明确批量路径可能因舍入改变 token，ANE prefill split 会引入额外量化。以上都不能继承我们同权重、全 logits/state 的逐字节资格。
- [QLinear 定义](https://github.com/nace-ai/kapitsa/blob/main/src/model/qwen35.h) 的专用 Q4 为 `bits=4, gs=64`，每组 BF16 scale **和 bias**；我们的 NVFP4 为非线性 FP4 code、group16 scale。二者虽然都称 4-bit，布局、解码和算术均不同。
- [官方 benchmark 条件](https://github.com/nace-ai/kapitsa/blob/main/docs/benchmarks.md) 中 M5 Max 128GB 的 27B 数字来自 affine 权重、DFlash2、各引擎最快设置；该文还标注 Kapitsa 两次 27B run 有其他应用 GPU contention。它的 writing/decode 口径不是我们的 HTTP 全请求 latency–throughput frontier。README 的 102 tok/s **不与我们的约 48 tok/s 作倍数比较**。

## 设计一：用接受前缀提交 recurrent state，先打通 C1 的成本判断

[Qwen35 的实际接口](https://github.com/nace-ai/kapitsa/blob/main/src/model/qwen35.h) 把 `encode_verify` 和 `commit_rows` 分开：最多八行验证先写 tape，不提交 recurrent state；`commit_rows(s,n)` 本身只设置 `s.n_pend=n`。下一次验证或 `encode_flush` 折叠 pending 行；plain decode、prefill、checkpoint 前需要 flush。`encode_verify_multi` 共享各请求的 projection/MLP/head，但 convolution、recurrence、KV attention 按请求分别处理。

[DFlash2 接口](https://github.com/nace-ai/kapitsa/blob/main/src/model/dflash2.h) 使用 target hidden captures 驱动每请求的 context KV ring。anchor 加七个 mask 产生草稿；greedy 接受最长匹配前缀，随后只把保留行加入 ring。两段 command buffer 可先提交 draft，再编码 verify；下一轮 draft 也可读取 GPU acceptance state。其“exact speculative sampling”不等于与另一种 kernel 的 logits/state 逐字节相同。

**移植假设（尚未测量）：**把我们的已通过 E042 的 `rowwise_native_verify/candidate.py::verify` 当作可信前缀 oracle，先获得 T4/T8 整模型验证成本，再决定是否值得接 draft。计时覆盖完整 forward、最终 logits 求值和所选 prefix 的 state 提交；不包含模型加载、raw dump、哈希。比较的门槛是：

`平均每轮实际提交 token 数 > (draft + verify + commit 的总时间) / native C1 单步时间`。

E042 的约 58 秒包含准备、参考推进与原始证据写盘，不是上述 verify latency。当前逐行 projection 会多次访问主干权重，局部 head 提速不能证明整个 verifier 划算。

最小后续实现应保留原 oracle，只加一个默认 OFF 的 accepted-prefix commit 接口和必要的 `flush` 边界；先使用现有完整 prefix state 选取，检验接受 0/部分/全部、继续下一步、EOS/取消。只有发现 state materialization 是实测瓶颈，才尝试保留精确运算顺序的短链 GDN scan 或 lazy materialization；不能先用代数低秩重排替换原舍入顺序。网页未成功返回 `qwen35_v8.cpp`/相关 Metal kernel 正文，本笔记**没有确认其 tape 的内部算术或 claimed exactness**。

## 设计二：C8 保留共享 batch，按 workload 关闭昂贵 speculation

[Engine::batch_rows / run_batch](https://github.com/nace-ai/kapitsa/blob/main/src/engine/engine.cpp) 不是统一开最大草稿：dense Qwen 两请求各四行，三请求不组成该 batch，四请求各两行；超过四请求只有一行/请求，源码注明未测。总行数上限八。这是作者在其设备上的策略，不能直接复制阈值。

**对我们的判断：**C8 已约 98.8% decode step 为 B8，主要机会不是再增加 scheduler batch。先完成现有 `gdn_prework_batched` 的 B2/B8 real-layer canary，再在同一个 combined graph 路径测 C8 GDN ON/OFF；保留 native batched projections、各请求独立 state，不额外串行 draft。它比重建一个 `encode_verify_multi` runtime 更小，也直接覆盖当前 B>1 GDN fallback。C1 与 C8 可以采用不同配置，但每个配置仍需自己的正确性与端到端测量，不能由单层或不同并发继承。

## M5 tensor operations 的作用与不适用条件

[BaseRT 论文 v1，2026-07-21](https://arxiv.org/html/2607.19438v1) 使用 Metal 4 `mpp::tensor_ops::matmul2d`、驻留 cooperative accumulator 和按 shape dispatch，主要加速 compute-bound prefill；M1 decode 保留专用 GEMV。实验是 M5 Pro、4/8-bit、tg128 与不同 prompt 长度，既没有我们的 NVFP4 Qwen3.8-27B，也没有 C8 HTTP frontier 或全状态 bit-exact 证明。GPU 每核 Neural Accelerator 与独立 ANE 是两件事。

Kapitsa 的 [V8Knobs / enc_qmm8 声明](https://github.com/nace-ai/kapitsa/blob/main/src/model/qwen35.h) 确实另有 `qmm8t` 的 M5 八行路径，限定 affine Q4/group64；[构建配置](https://github.com/nace-ai/kapitsa/blob/main/CMakeLists.txt) 把 tensor kernels 分库编译为 Metal 4。这比“所有 decode 都只适合 GEMV”更具体：M=8 可以研究，但收益和 exactness 必须测。

若后续筛查 tensor path，最小对象应是两种真实 NVFP4 projection 的 M8，保持原 packed codes/scales，比较完整输出 bits，再考虑性能。新的 matmul reduction 与原 `float4 dot + shuffle` 次序不同，可能立即不满足资格；不得改成 affine 重量、放宽容差或只看 argmax 后称同输出优化。C1 M=1、很小的 T、原算术必须逐字节一致时，不应预设 tensor_ops 更快。BaseRT [官方仓库说明](https://github.com/basecompute/baseRT#whats-in-this-repo) 当前只开放 CLI/格式/API/bindings，engine 为另行许可的预编译 binary；论文不是可直接复制的完整公开 kernel 实现。

下一步顺序：完成 C8 batched-GDN canary；测 C1 full-verifier break-even；之后才用小 M8 数值筛查决定 tensor path 是否值得继续。以上均为实验建议，没有新增性能结论或 Pareto 点。
