# 实验篇阅读地图

[返回基础教程](../tutorial-v2/README.md) · [最近三家对比](45-lithos-queued-concurrency.md) · [进度与原始证据索引](../progress/README.md)

01—15教你构建一个可运行的引擎。这里记录之后怎样定位瓶颈、尝试替换、验证并决定是否保留；**实验编号不代表已合入的优化数量。** 有些只通过局部检查，有些完整正确但更慢，有些最终进入了特定服务配置。

初次阅读先走这条线：**16—17准备工作融合 → 27原生命令 → 29—30接入服务 → 38—40投机正确性 → 43—45公平比较**。其他章节用于回答下面的具体问题，不必顺序读完所有日志。

## 1. 每组实验解决什么问题

| 学习问题 | 建议正文 | 其他实验与证据 |
| --- | --- | --- |
| 融合少几个操作，为什么整模未必更快？ | [16 SiLU与拆头](../tutorial-v2/16-实验篇从哪里开始.md)、[17 Q/K范数复用](17-gdn-norm-reuse-walkthrough.md) | [17阶段结果](17-gdn-norm-reuse-status.md)、[18投影拼接](../tutorial-v2/18-先确认优化真的更快.md)、[18证据](18-projection-packing-evaluation.md)、[19量化矩阵乘](../tutorial-v2/19-把量化矩阵乘拆开看.md)、[19几何实验](19-nvfp4-qmv-geometry.md) |
| 从其他引擎借什么，哪些数学不能直接替换？ | [20三条源码路径](20-donor-implementation-map.md)、[21卷积融合](21-gdn-conv-fusion.md) | [22组合后的HTTP](22-combined-decode-screen.md)、[23 kernel诊断](23-kernel-tests-and-gpu-diagnostics.md)、[24数值与QMV](24-numerics-and-qmv.md) |
| 为什么要固定buffer、减少每步主机提交？ | [27 ICB与状态轮换](27-native-icb-pingpong.md)、[28组件接回ICB](28-native-gdn-composition.md) | [25固定buffer](25-fixed-buffer-replay.md)、[26 GPU反馈](26-gpu-feedback.md) |
| 单步变快后，怎样进入真实请求循环？ | [29原生runner与HTTP](29-native-runner-http.md)、[30提前提交一步](30-native-ahead.md) | [31直接反馈与诊断](31-native-direct-profiling.md) |
| 哪些看起来合理的局部优化没有留下？ | [35局部采用门槛](35-local-probe-stops.md) | [32残差carry/四投影](32-native-residual-projection.md)、[33整步编译](33-native-range-compiled-step.md)、[34门控预计算](34-native-ba-gates-prework.md)、[36容量/RMS版本](36-cap64-and-rms-profile.md)、[37异步提交](37-simd4-async-and-spec-handoff.md) |
| 为什么verify数学相同，输出仍可能不同？ | [38事务与量化乘法](38-spec-transaction-and-qmm.md)、[39逐行M1](39-verify-broadcast-m1.md)、[40整模检查](40-m1-verify-and-bounded-loader.md) | [冻结候选与diff](../code-versions/39-broadcast-m1/README.md)、[共享W局部候选](../code-versions/44-shared-w-m1/README.md) |
| 怎样判断真的改善吞吐—延迟？ | [43 C1比较](43-fresh-c1-throughput-latency.md)、[44同轮前沿](44-fresh-full-throughput-latency.md)、[45补Lithos并发](45-lithos-queued-concurrency.md) | [41传输选项负结果](41-native-http-nodelay-pilot.md)、[42短输出投机负结果](42-startup-warm-short-request.md) |
| 之后值得试什么？ | [46论文与替换点](46-next-ideas-from-papers.md)、[47近期论文精读](47-recent-paper-reading.md) | 研究方案；没有因阅读论文而新增实现或性能点 |

## 2. 三种编号不要混用

- **章号**（如39）：文档阅读顺序，不是代码版本。
- **V编号/候选名**（如V114、candidate116）：某次实验的源码、配置或运行记录；同一章可以出现多个版本。
- **Git tag或冻结目录**：真正用于定位代码。基础篇用`git show tutorial-08:路径`；实验篇优先使用该章给出的source目录、source-map与before/after diff。

正文里的“当前”“待验证”默认指该章的记录时点。后续进展看章首补注与最新进度，旧负结果不删除。例如第39章首次整模加载中止，后续第40章才取得相应完整路径证据。

基础代码所在的`main`、教学代码所在的`tutorial_code/`、独立实验快照不是同一个自动升级的运行包。第44章的本方前沿是两种静态启动配置的集合：C1原生单流，C2/C4普通batch；没有实现同进程自动切换的统一最优策略。

## 3. 读后半部分前先认识这些词

| 词 | 本教程里的含义 |
| --- | --- |
| C1/C2/C4 | 客户端同时发出的请求数；不保证GPU实际组成同样大小的batch |
| B、T、CAP | B是模型batch行数，T是本次处理的token位置数，CAP是该路径分配的缓存容量 |
| ICB | Metal Indirect Command Buffer：把可复用的GPU命令保留下来；仍要正确更新输入buffer和状态 |
| prefill / decode / verify | 处理已有prompt / 续写 / 一次验证若干投机候选；verify还可能包含anchor位置 |
| micro / B-only | 局部计时 / 只测候选B的探索；都不足以证明完整服务胜过对照A |
| 数值资格 | 在指定输入、形状、精度和状态范围内完成正确性检查；不是所有输入的数学证明 |
| STOP | 某项预先约定条件不满足而停止；可能是资源或性能问题，要读原因，不能一律当数值错误 |
| adopted / default OFF | 已在明确路径采用 / 代码存在但默认未启用；局部PASS不自动改变开关 |

## 4. 怎样读一章，避免陷进收据编号

先回答四个问题：原来哪一段代码花时间？本章替换哪一段？计算与状态有没有变化？完整请求是否因此受益？正文先看流程图、before/after和结果表，SHA、PID与原始日志放到source目录按需核查。

正确性检查要说清是浮点容差、greedy IDs、还是状态/输出逐字节一致；它们不是同一个标准。前沿横轴采用完整请求延迟、纵轴采用整组输出吞吐。模型、checkpoint、请求长度、并发、电源与计时边界不同的结果不能直接拼成一条曲线。

本文只是导航，不为任何候选新增数值或性能资格。新论文方案也必须走“实现 → 正确性 → 局部成本 → 完整HTTP比较”的步骤。
