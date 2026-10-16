# 23：哪些实现差异值得继续，哪些实验先停下

第22轮的组合版 C1 收益不足。本轮继续从 oMLX 的 GDN state 布局和 Lithos 的量化 GEMV 实现拆出小实验；目前没有新增被采用的运行时代码，也没有新官方 Pareto 点。负结果同样保留。

| 实验 | 从原路径换成什么 | 实际检查 | 结果 |
| --- | --- | --- | --- |
| V26 | NVFP4 原 row-major 地址 → row/K-block 重排；R4 算术不变 | 27项严格输出检查、逆重排和只读检查通过 | 局部描述性减少0.1249%，控制组漂移；停止 |
| V27 | 原 GDN 递推 → compact Q/K、SIMD 归约、寄存器 state、持久转置存储 | 15组输入×两布局和16步链在预设误差界内；局部描述性减少13.87%，控制组漂移 | 整模第一步 logits 最大差异0.125超过预设1/16，整个整模资格失败；不进入 HTTP 对比 |
| V28 | 两次 QMV 加 SwiGLU → 一核内共享 activation、保留两套 R4 累加并融合激活 | 70项严格中间/最终输出、只读检查通过，不复制权重 | 局部描述性反而慢0.7684%，仅1/6对获胜且控制漂移；停止 |

V27 的局部误差小，不代表经64层传播后仍能通过整模门。失败时保存了原始 logits 与比较记录；即使保存的单步 argmax 恰好相同，也不能把没有完成的32-token检查称为通过。后续会另写保留原求和顺序的版本，不修改这次失败的门槛或旧源码。

V28 的 build1 未执行；根据独立 V26/V27 的首 slot 较慢现象，在执行 V28 之前创建 build2，将每 block prime 从128增加到2048。repeat512、配对计划和控制门不变，最后仍然漂移。因此加长 warmup 不是稳定性的保证，更不能只挑正向配对。

## 用真正的 Metal 时间戳定位

这台 M4 Max 支持 stage-boundary counters，不支持 dispatch-boundary timestamp；本机也没有 Xcode/xctrace。因此只在诊断子进程注入一个本地记录器，保存 command-buffer 的 GPUStartTime/GPUEndTime 和实际 kernel 成员，没有改动系统或其他应用。

| 诊断协议 | warm / 测量步数 | 实际 dispatch 分组 | 可得结论 |
| --- | --- | --- | --- |
| 原 OPS50 / MB50 | 3×32 / 4 | 336个 buffer、7896次 dispatch，99.83%的 envelope sum 为混合组 | 平均每步 GPU envelope union 约35.14ms，不能拆成单 kernel 耗时 |
| OPS1 / MB50 | 3×32 / 1 | 多数仍是两次 dispatch；strict `buffer_ops > max_ops` | 减小到1也不保证每次只含一个算子 |
| OPS0 / MB50 | 1×32 / 1 | 1894个 buffer、1974次 dispatch；1830个单 dispatch，48个双 dispatch，16个三 dispatch | 可以观察更细的成员和 envelope，仍有提交开销、等待和重叠 |

最后一轮所有 envelope 相加是54.66ms，合并重叠后的 union 只有35.95ms。**不能把相加的54.66ms当作生成一个 token 的时间，不能把各类相加值除以它当作生产占比。** 单 dispatch 的 buffer envelope 也不是独占 shader 执行时间。OPS0 的强扰动配置只用于诊断，未作为服务提速配置或 Pareto 点。

真实捕获中量化 QMV 有497次单 dispatch，而 GDN 递推有48次。它们给出了继续研究量化解码与减少小算子边界的依据；不能由次数直接推算收益。后续每个候选仍要接受数值、完整生成和实际 HTTP 检查。

[源代码与范围说明](23-kernel-tests-source/README.md)、[全部文本证据索引](23-kernel-tests-source/evidence-index.json) 和 [压缩证据](23-kernel-tests-source/evidence.tar.gz) 包含原 PASS/STOP/FAIL、配对结果、资源和退出证明。所有实际 GPU 进程均已回收并经外部严格 PID 检查。原始张量与原生 binary 不放入此压缩包；收据保留其 SHA 和本地路径，不冒充独立重新采集。
