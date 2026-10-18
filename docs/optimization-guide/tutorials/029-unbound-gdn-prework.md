# E029：把 GDN 预处理融合接到真实长上下文路径

2026-10-17：预处理融合候选已通过真实激活局部验证及原 37 请求的整模型同调度检查；真实 HTTP 已完成（见末节），正式性能测量仍待完成。SIMD 标量候选尚未运行设备。现有正式图只包含已审计的实际计时，局部正确性结果不进入 Pareto。

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


## 从局部状态链进入整模型轨迹

2026-10-17 19:26 PDT：原 37 请求完成，独立数值、资源与恢复审计全部 PASS。8 个 D-C1、16 个 D-C8、13 个生命周期请求累计 5,163 步、12,411 committed tokens。独立复核 6,622,533 组完整数组逐位收据；成功数组的离线算术没有重算。

本次证据也确认了开关的边界：长 B1 在每个 GDN 层进入候选，多行 decode 回到原实现；没有暗中把 B1 的正确性推广到 B8 融合。两个模型对分别安装/撤回，48 层计数与实际调用相符。没有观察到 EOS 终止，因而保留这一未覆盖范围。原服务恢复完成。[逐项结果与全部原始记录](../../experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1/RESULTS.md)。

下一道门是同样真实 HTTP 的记录与重放，再关闭观察器进行配对计时。通过局部和整模型轨迹，只说明值得继续测量；不能提前把预处理融合画成更快的点。E024 的独立 OFF/ON 正式计时先运行，组合实验仍需自己的正确性检查。


### 下一道门：真实 HTTP 调度（源码已准备，尚未执行）

新增 [独立 HTTP 接入](../../../bench/m5_ultra_256gb/gdn_prework_http/README.md) 复用原服务、客户端和逐状态比较器；只在48个GDN实例上安装局部开关。先捕获、回收服务进程，再让两套模型按实际到达历史独立生成。记录的token只用于比较，不能喂给候选来掩盖分歧。

HTTP侧只读Python调用计数，不增加tensor求值；数值检查留给replay。消费已有ahead不算新调用，实际B>1必须核原路径回退。16项CPU测试已写、未跑；外层资源owner/controller和真实注册尚未完成。这是接入准备，不能增加Pareto点。


### C8 的下一项假设：保持原投影，扩展预处理覆盖

当前长历史候选仅在 B1 生效。源码检查发现现有 fused kernel 已有 B2–8 的 batch 地址计算，因此下一项小实验应先保留五次原生 Linear，只扩展卷积、激活、Q/K norm 与状态核的调用范围。B>1 继续使用原 gates；不改变投影归约、权重布局或状态提交顺序。

[具体设计与逐行风险](../../../bench/reviews/2026-10-17-gdn-prework-batched-next.md) 先要求 B2/B8、早晚真实激活、三个层位的独立三步状态链，再验证混合历史/row reorder/refill 与完整 HTTP。这一方向仍未实现、未测量；已有 B1 结果不能覆盖它。它减少中间计算，但不减少五次投影的权重流量和每行递归状态的读写，收益取决于真实瓶颈。此前 packing 和 per-row M1 的失败保留为约束。


2026-10-17 后续：上述批量方向已落成 [独立默认关闭的候选源码](../../../bench/m5_ultra_256gb/gdn_prework_batched/README.md)。核心修改只有 B1–8 的形状边界与两处 reshape；原有 batched Metal kernel、五次投影和 gates 分支继续复用。12 项假 CPU 测试已写但未执行，逐位真实状态验证仍是下一道门。原 B1 候选与证据保持独立，不能把旧 PASS 改标为新 batch 域已通过。


## 真实 HTTP 捕获与重放已通过

2026-10-17 23:33 PDT：C1×8 与 C8×16 的四阶段周期已完成，独立数值、完整 HTTP/资源与原服务恢复审计全部 PASS。共5,124步、12,288 committed tokens，6,573,486组完整数组逐位收据通过；成功数组没有离线重新计算。两个历史均达到cap512，没有观察到EOS。全部原始记录和首次审计保存在[HTTP结果](../../experiments/m5-ultra-256gb/E029-gdn-prework-http-v1/RESULTS.md)。

这次实际dispatch也划清收益范围：C1的4,088次B1 forward触发196,224次逐层候选调用；C8的1,027次多行forward全部走原fallback，候选调用为零。不能把C8正确性通过说成批量融合加速。下一轮三对OFF/ON正式计时会关闭观察器，分别测单请求收益和并发回归。E024图编译仍关闭；合并两开关需另行验证。

同时读DeltaLog后实现了一个严格FP32顺序的[状态日志CPU toy](../../../bench/m5_ultra_256gb/state_log_exact_reference/README.md)：首次8项机制测试与24条轨迹通过，但[成本分析](../../../bench/m5_ultra_256gb/state_log_exact_reference/GO-NO-GO.md)给出当前NO-GO。省去部分状态写入，仍需每步完整读取base并重放更新；严格保持舍入显著增加算术，不能引用论文的容差等价收益当作本机加速。先保留学习与失败边界，继续测真实关键路径。
