# E029：把 GDN 预处理融合接到真实长上下文路径

2026-10-17：预处理融合候选已通过真实激活局部验证与独立原始数组复核；整模型验证和性能测量仍待完成。SIMD 标量候选尚未运行设备。现有正式图只包含已审计的实际计时，局部正确性结果不进入 Pareto。

## 先找没有进入热路径的优化

目标 Qwen3.8-27B 有 48 个 GDN 层。原生实现已包含卷积、SiLU、拆分、q/k 归一化和新卷积缓存的融合辅助函数，但现有调用经由短上下文投影打包路径。D512 的八个真实提示从 886–978 tokens 开始，因此不能用短路径已经存在的优化推断它在此负载生效。

安装的 oMLX Qwen3.5 B1/T1 路径会启用类似预处理融合。我们先学习这个局部设计：把现有原生融合辅助函数接到普通 decode；不引入其通用服务层，也不同时改变投影、attention、batch 调度或权重格式。

[候选源码](../../../bench/m5_ultra_256gb/gdn_prework_unbound/candidate.py) 默认 OFF。显式安装只替换三个或指定 GDN 实例的方法，作用域退出后撤回。适用范围限定为 M5 Ultra / MLX 0.32.3、NVFP4、B1、原 BF16 输入与 FP32 状态；其他正常路径回原实现。仍执行原来的五次 Linear，读取当前参数，没有额外的派生权重缓存。尚未验证与 E024 图编译组合。

## 融合不能省掉舍入语义

卷积的 BF16 乘法/加法、SiLU 舍入、q/k 归一化及 FP32 状态更新都可能改变后续输出。数学上等价不等于逐位一致。已有辅助函数显式指定 safe，普通状态核使用默认选项；本机 MLX 0.32.3 的 CompileOptions 默认同样是 Safe，因此这不是已证实的数学模式差异。融合边界和编译器 lowering 仍需实际逐位验证，“它在短路径测过”不能代替本次长路径验证。

小实验先回答一个明确问题：在模型真实产生的 activation 与状态上，这个独立替换是否保持完整状态？

1. 固定最短、最长两个 D512 原始提示，使用原模型的正常 greedy token；不循环提示、不强制越过 EOS。
2. 捕获前、中、后三个 GDN 层，分别在 decode 1–3 和 129–131 形成三步连续链。
3. 每条候选链只从参考状态初始化一次，之后使用自己的上一状态；后续 activation 仍来自原模型。因此这是局部层验证，不是候选整模型 rollout。
4. 每一步比较完整 BF16 输出、卷积缓存和 FP32 递归状态，核 shape、dtype、finite 与逐位相等，保留原 dtype 原始数组；任何失败停止该尝试。
5. 独立审计保存的数组和实际 token 来源。通过之后才进入 37 请求的完整 logits / cache / ahead / 调度验证，再做真实 HTTP 验证和 observer-OFF 正式计时。

[实验源码与范围](../../../bench/m5_ultra_256gb/gdn_prework_unbound/canary-plan.md) 记录固定 18 个局部调用、6 条状态链和未覆盖场景。CPU 假测试只验证接线和失败处理，不能证明 Metal 算术正确。外层必须单独完成真实子进程管理、资源监控及原服务恢复；旧 lifecycle 的前序白名单不接受当前新实验，不能只因 owner 的 CLI 通用就直接运行。

## 另一个独立假设：共同标量只算一次

[SIMD 标量候选](../../../bench/m5_ultra_256gb/gdn_simd_scalars/README.md) 保持原状态计算和归约顺序，仅尝试由 lane 0 计算共同的 q/k 归一化与门控标量，再广播给同 SIMD 的线程。它同样默认 OFF、尚未运行。编译器可能已经消掉重复计算；只有真实检查和计时才能决定是否保留。先不与预处理融合叠加，以免失去定位失败和收益的能力。

[研究笔记](../research/2026-10-17-gdn-real-inputs-and-apple.md) 区分了近期真实激活失败报告、Apple 不同协议的实测和较早的 GDN / launch overhead 论文。外部数字不混入本机吞吐—时延图。

## 怎样判断值得留下

局部核更快只是候选证据。正式比较仍用相同权重、原提示、normal EOS、正确 committed token 数与完整 POST→DONE 时延，分别展示 p50 和 p95。更大的 batch 可以摊薄权重读取，但每条序列的 GDN 状态仍独立增长：本模型每条序列 144 MiB，不能因为 256 GiB 内存充足就假定并发越高越好。

先比较独立开关，再对值得保留的组合重新验证。没有足够的整请求收益，就保留失败记录、移除运行时负担。


## 256 GiB 容量不等于更低的单请求延迟

对这份固定模型，仅递归状态就有 `48 layers × 48 heads × 128 × 128 × 4 bytes = 144 MiB/sequence`。一次 decode 至少要读旧状态、写新状态，因此它的理论最低流量为 `288 MiB × B`；卷积缓存、投影和 attention KV 另计。这是数组尺寸推导，不是已测显存带宽。

| GPU decode B | 递归状态驻留 | 每步状态最低读写量 |
|---:|---:|---:|
| 1 | 144 MiB | 288 MiB |
| 8 | 1.125 GiB | 2.25 GiB |
| 16 | 2.25 GiB | 4.5 GiB |
| 32 | 4.5 GiB | 9 GiB |
| 64 | 9 GiB | 18 GiB |

假设这些状态读写落到被测内存层、每步实际读权重 W，简化的带宽下界是 `step_time ≥ (W + 2 B S + KV_traffic) / effective_bandwidth`，其中 W 是本步实际读取的权重字节，S 是一条序列的递归状态字节；cache 命中、分块和额外读写会改变实际流量。整体输出率近似 `B / step_time`，单请求输出 N tokens 则需要约 `N × step_time`，另外加排队和 prefill。增大 batch 可以摊薄 W，却不能摊薄每条序列自己的状态流量。

因此更大内存首先允许更大的模型或更多在途请求；它不直接让 decode 更快。提高 B 的实验必须同时记录完整请求延迟、实际 GPU batch 分布、状态读写和内存压力。只用 C1/C8 两端还不足以确定中间的最佳折中；C2/C4 的实际 HTTP 点与 C16 以上的容量探索各有价值。最终按经过验证的实测点选择工作区间，不以最高并发数替代 Pareto 目标。


## 已完成的 CPU 准备

2026-10-17 18:35 PDT：候选、真实激活采集器、原始数组审计器和serving接入的44项假CPU测试已有PASS证据；新增单子进程执行驱动的12项测试也全部PASS，完整[源码/命令/结果归档](../../experiments/m5-ultra-256gb/gdn-canary-driver-cpu-v1/README.md)已保存。驱动覆盖默认不执行、绑定改变拒绝、失败/未知启动与finally恢复路由，禁止真实设备、子进程、网络和信号。这些结果不代表实际GPU算术或恢复成功；这项记录保留准备时状态；随后实际 18 调用结果见下节。


## 实际局部验证结果 · 2026-10-17

注册的 18 次局部调用已完成，独立原始数组、资源及恢复检查全部通过。60 个原始文件、54 组完整输出/conv/S、24 组连续状态及 6 次全词表参考 argmax 检查均通过。候选在每条三步链中使用自己的上一状态，后续 x 仍来自参考模型；因此尚不能推出候选整模型正确。原服务已恢复。下一步是在观察器安装前接入该候选，运行原有 37 请求完整状态与调度检查。[完整结果与原始证据](../../experiments/m5-ultra-256gb/E029-gdn-prework-canary-v1/RESULTS.md)。

The math-mode correction above was checked against this actual pinned installation: `mlx/include/mlx/backend/common/metal_kernel.h:14-15` sets `MathMode::Safe`; `mlx/core/fast.pyi:205-211` documents the same default. It does not prove convolution, SiLU or normalization lowering merely from source spelling. Actual saved-byte checks provide the narrow canary evidence.
