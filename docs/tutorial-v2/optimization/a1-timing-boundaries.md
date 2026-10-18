# A1：先找到时间花在哪里

[六篇目录](README.md) · [下一节 A2](a2-controlled-comparison.md)

读过[第11章测量](../11-benchmark.md)后，先回答一个问题：我们说的“慢”，究竟是矩阵乘慢，还是完整请求慢？同一条生成链可以有不同计时边界：

```text
客户端 POST → 等待调度 → prefill → decode 多步 → HTTP 终止帧
                           └── 单层 → 投影、卷积、GDN、MLP
```

局部 kernel 计时回答“这个计算怎样实现更快”；完整请求计时回答“用户等多久”。二者都要测，但不能用前者代替后者。即使一个占总时间 10% 的阶段快一倍，总时间也只从 100 降到 95；若额外的数据搬运花掉 5，整条请求就没有收益。这是假设数字，用来解释边界，不是我们的实测占比。

**baseline → 新做法：** 原先只看一项 decode token/s；现在先画出计时区间，再给每项成本命名。旧31章的原生 runner 就有一个容易漏掉的区间：

```text
t0 ── prefill，产出第一个 token ── t1
    ── reset / 上传初始状态 ── 31 次原生续写 ── t2

decode_ms = (t2 - t1) / 31 × 1000
whole_core_tps = 32 / (t2 - t0)       # 时间单位为秒
```

第一项包含 reset 和初始状态上传，不能叫“纯 GPU 单步”。第二项覆盖32个输出的核心生成，仍不含完整 HTTP。该次 B-only 探索的约 28.83 decode token/s，也不能直接拿来对比外部引擎的服务吞吐。

阅读 oMLX、Lithos 时也先沿**真实调用路径**走。旧20章发现本方与 oMLX 都已有一步提前提交，不能把已有机制当作新优化；Lithos 的命令复用则要求固定 buffer 和明确的状态轮换。借鉴之前先写出“我准备替换哪一行、它现在实际调用什么”。

**保持不变：** profiling 不能改变正式对照的输出、计数和资源采样边界。把 encoder 切段、加 fence 本身可能扰动时间；单位未确认的时间戳不能换算成瓶颈百分比。

**理解题：** 单个 kernel 快了 20%，完整请求却没变，下一步应该补测哪些阶段？

原始记录：[20 三家实际源码路径](../../experiments/20-donor-implementation-map.md)、[23 GPU 诊断](../../experiments/23-kernel-tests-and-gpu-diagnostics.md)、[31 计时边界与扰动](../../experiments/31-native-direct-profiling.md)。下一节：[A2 公平比较](a2-controlled-comparison.md)。
