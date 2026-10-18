# 2026-10-17：BaseRT 可借鉴设计与本机实验边界

这次重读两篇 **2026 年 7 月**的一手论文，不算发现新的 10 月论文。最值得继续的是两条已具备简单开关的方向：保留原 kernel 路由的 decode 图复用，以及去掉无意义的状态搬运。论文不构成我们 M5 Ultra 的性能或逐位正确性证据。

## 实际阅读与可迁移范围

| 来源 | 日期、阅读范围 | 与本机区别 |
| --- | --- | --- |
| [BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators](https://arxiv.org/html/2607.19438v1) | [7 月 21 日 v1](https://arxiv.org/abs/2607.19438)；读 §2.1–2.3、§3.1–3.2、§4.1、§4.4、§5.1–5.2。 | M5 Pro 48GB；BaseRT 0.1.6，对照 MLX 0.32.0 / mlx-lm 0.31.3。Q4/Q8、转为 `.base` 格式，pp128–2048、tg128，5 次重复；论文明确未覆盖 continuous batching / 多请求并行 decode。NAX 用 Metal 4 `matmul2d` 加速计算密集的 GEMM/attention prefill，decode 继续专用路径。 |
| [BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal](https://arxiv.org/html/2607.00501v1) | [7 月 1 日 v1](https://arxiv.org/abs/2607.00501)；读 §3.1–3.4、§4.1。 | 主设备 M4 Pro 24GB，另测 M3 16GB；Q4/Q8，pp128–2048、tg128。重点是预分配 scratch、区分 GEMV/GEMM、减少 dispatch 与中间内存往返；未给出可移植的逐位 NVFP4 kernel。 |

我们的合同是固定 **M5 Ultra 256GiB、MLX 0.32.3、Qwen3.8-27B 原始 NVFP4 权重、8 个 D fixture、normal EOS / cap512、C1/C8 HTTP**。服务采用既有 gates ON、norm reuse OFF。设备带宽、量化字节、请求调度和指标均不同；论文的 prompt/decode 速率不能换算为我们的 POST→DONE 前沿，文中的 numerical equivalence 也不等于原 dtype 逐位一致。

作者 [GitHub 仓库 README](https://github.com/basecompute/baseRT#whats-in-this-repo) 明确公开 CLI、格式、headers、bindings；engine 是独立专有预编译二进制。本轮只读公开网页，没有读取到其 Metal/NAX 内核实现，不声称已核 tile 或归约顺序。当前 README 的 continuous batching 功能也不能倒填进 7 月论文实验。

另读当前 [公共 C API](https://github.com/basecompute/baseRT/blob/main/include/baseRT/baseRT.h) 中 `baseRT_set_baked_decode`、`baseRT_set_decode_replay`、`baseRT_reload_metallib`、prefill/batch scratch 与 GDN snapshot 注释：有 baked dispatch 的 A/B 开关；CUDA stream-capture 开关明确对 Metal 无效；更换 kernel 库必须使旧 KV/递归状态失效。这里是 **2026-10-17 观察到的 main 接口合同**，未固定其提交，也未验证二进制实现。

## 两个最小可试设计

### 1. 固定路由、复用准备好的 decode 调用

**本机推论：** 复用 [E024 candidate](../../../bench/m5_ultra_256gb/route_decode_graph/candidate.py) 的单 runner callable，降低逐层 Python 建图/dispatch 成本；不改 NVFP4 字节、split/reduce、attention 边界或请求顺序。BaseRT 的 baked table 是机制参考，不是我们的 `mx.compile` 实现证明。

继续当前 E024 HTTP 接入即可，无需再造 kernel。开关默认 OFF，保持 gates ON / norm OFF / E028 OFF；真实 service context 仍为 2048，候选只按自身已注册 bound、batch、容量守门。先用实际 C1/C8 捕获的到达、EOS、refill 和 ahead 生命周期重演，核完整 logits、128 状态原 dtype 位、输入不变及实际 route-call；参数替换仍走已有独立检查，不能把首次编译捕获的旧权重当常量。Python callable 数量不能冒充机器图编译次数。

**证伪：** 任一原 dtype 差异、错误 fallback/状态复用即拒绝。数值资格后才 observer OFF、相同 fresh/warm/formal 协议配对，主指标为 committed tokens / 共同窗口与 POST→DONE p50/p95；若 forward 收益被同步、排队或专门化成本抵消，就没有 HTTP 收益。当前研究不授予尚未执行的 HTTP 资格。

### 2. 保留状态所有权，先删 identity 搬运

**本机推论：** BaseRT 的零分配 decode 思路可缩为 [E028 identity-refill](../../../bench/m5_ultra_256gb/identity_refill/candidate.py) 的一个局部开关：旧行顺序确为 identity、旧 batch 非空且所有 KV 容量不变时省去 gather，仍保留原 concat 与非 identity fallback。先走已有候选的真实 HTTP 对照，暂不引入大 arena、原地写缓存或新同步 fence。

最小验证复用已有两层 alias/offset/strided/delayed-eval 字节证据与完整模型资格，再单独测实际 C8 refill；E024 关掉以隔离主因。核源和结果分别写入、retire 后复用、ahead 丢弃、容量变化的 fallback，以及全部 BF16/F32 状态。去重 backing、allocator 和物理占用分别记录，不能把逻辑字节当成实际搬运量，也不能把共享 buffer 重复相加。

**证伪：** 源/输出别名污染、延迟图保留增长、原 dtype 差异均拒绝；若 MLX 已消除 identity gather，或其成本不在端到端关键路径，完整请求指标可能无收益。该设计不承诺“零分配”，也不改变当前 uncached D 工作负载。

## 不从论文直接重启的方向

NAX 的收益条件是算子强度与具体 dispatch，不是芯片名字。我们已有 E025 same-split NAX partial 逐位拒绝、E027 native multirow→M1 bridge 逐位拒绝；原生 kernel 不因此被判错。论文未提供能够推翻这两个结果的 NVFP4 归约证据，本轮不重新推荐同一路径、不换 Q4/Q8 或降低正确性门。若未来找到新的具体 dispatch 或算术保持办法，应单独预注册原 kernel 对照，并从原 dtype partial/final 字节门开始。

本轮仅网页和小源码阅读、写笔记；没有下载引擎、改变运行时、执行模型/测试/性能测量或服务操作。
