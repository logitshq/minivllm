# 原30章的内容去了哪里

[返回六篇目录](README.md) · [机器可查映射](coverage.json)

原16—45是30个实验编号，不等于30个文件。16的完整章保存在固定tag，当前目录另有入口；17、18、19各有配套记录。下表全部覆盖，旧文件不删除、不改写，源码与原始数据仍在各原章链接的目录中。

“进入短章”说明新正文实际讲解的核心；“仍需展开原文”明确哪些细节没有复制进短章。**18节不是原30章的逐字替代品。** 完整内容保留在“短章＋原记录”两层里，需要复现实验或深入推导时继续读原文。

| 原章 | 进入短章的知识点 | 新节 | 仍需展开原文的细节 |
| --- | --- | --- | --- |
| [16](https://github.com/logitshq/minivllm/blob/tutorial-16-experiment-v1/docs/tutorial-v2/16-把GDN的SiLU和拆头并入状态更新.md) / [配套1](../16-实验篇从哪里开始.md) | typed SiLU、拆头映射、重复计算与dispatch的取舍 | [B1](b1-postconv.md) | 精确逻辑计数、开关/fallback、独立示例与每轮检查 |
| [17](../../experiments/17-gdn-norm-reuse-status.md) / [配套1](../../experiments/17-gdn-norm-reuse-walkthrough.md) | Q/K范数只算一次；L2与RMS区别；共享scale不共享S | [C1](c1-norm-reuse.md) | 串行归约顺序、临时空间、阶段资源STOP、完整结果索引 |
| [18](../18-先确认优化真的更快.md) / [配套1](../../experiments/18-projection-packing-evaluation.md) | 输出行拼接；顺序漂移；对称AB/BA与AA/BB | [A2](a2-controlled-comparison.md)、[C2](c2-weight-reuse.md) | GDN和MLP各轮原数值、prime/cold、hash放置及采用门 |
| [19](../19-把量化矩阵乘拆开看.md) / [配套1](../../experiments/19-nvfp4-qmv-geometry.md) | NVFP4原数学与R4/R2几何；数值门和速度门 | [A2](a2-controlled-comparison.md)、[C2](c2-weight-reuse.md) | qdot推导、scale表示、27项逐例检查、冷启动及完整配对 |
| [20](../../experiments/20-donor-implementation-map.md) | 从实际调用链识别已有机制；正确借鉴oMLX/Lithos | [A1](a1-timing-boundaries.md) | wheel与上游版本差别、源码哈希、allocator/in-flight逐项证据 |
| [21](../../experiments/21-gdn-conv-fusion.md) | 卷积、SiLU与原始tail同核；BF16规约次序 | [B2](b2-conv-fusion.md) | 原sum_y推导、带正零partial、6个合成case、8步chain与84项严格比较、Metal完整源码 |
| [22](../../experiments/22-combined-decode-screen.md) | 融合接回整模；真实状态与生命周期；组合结果不归因单项 | [B2](b2-conv-fusion.md) | 函数图复用细节、B2误用gold的修正、所有HTTP组与资源检查 |
| [23](../../experiments/23-kernel-tests-and-gpu-diagnostics.md) | 权重布局与几何；控制漂移；profiling边界 | [A1](a1-timing-boundaries.md)、[C2](c2-weight-reuse.md) | rowblock编码/逆变换、gate/up融合、仪器能力及各次失败 |
| [24](../../experiments/24-numerics-and-qmv.md) | 因子搬移与浮点边界；改归约后需重验 | [C2](c2-weight-reuse.md) | factor-fold旧/新协议、GDN shuffle变体、全部数值和停止记录 |
| [25](../../experiments/25-fixed-buffer-replay.md) | 固定buffer、offset与所有权；单步重放不是连续生成 | [C3](c3-state-buffers.md) | 捕获planner、原生同步失败与修复、reset及完整binding表 |
| [26](../../experiments/26-gpu-feedback.md) | 一次seed；GPU反馈预测/位置/长度；参数owner存活 | [D1](d1-command-reuse.md)、[D2](d2-device-feedback.md) | stage范围、同步/别名细节、原生单臂时间与原参数只读证据 |
| [27](../../experiments/27-native-icb-pingpong.md) | S双bank；真正ICB；资源声明与同步；克隆负例 | [C3](c3-state-buffers.md)、[D1](d1-command-reuse.md) | PSO重建、常量bank、dispatch计数、arena克隆/barrier诊断 |
| [28](../../experiments/28-native-gdn-composition.md) | 融合与命令复用分层；避免双重SiLU；重查新绑定 | [D1](d1-command-reuse.md) | held-state/norm内核、S_new index错误与修复、各组合资格 |
| [29](../../experiments/29-native-runner-http.md) | 整请求选择路径；native状态所有权；不能回退旧seed | [D3](d3-service-ownership.md) | 依赖planner支线未减少barrier、三路径逐例生命周期及电源STOP |
| [30](../../experiments/30-native-ahead.md) | 两票FIFO；预算减首token；EOS后drain/discard再drop | [D3](d3-service-ownership.md) | seed mutex竞态修复、slot/bank完整规则、tiny/整模与控制失败 |
| [31](../../experiments/31-native-direct-profiling.md) | 直连替代stage；诊断与4B生产读取；计时含seed的区别 | [A1](a1-timing-boundaries.md)、[D2](d2-device-feedback.md) | alias证明、allocation未释放、kernel支线、分段时间戳与采样扰动 |
| [32](../../experiments/32-native-residual-projection.md) | 多投影复用x；原指针与packed数学；整模通过不等于提速 | [C2](c2-weight-reuse.md) | residual carry独立分支、四投影的参数/归约/绑定与完整检查 |
| [33](../../experiments/33-native-range-compiled-step.md) | 命令复用与整步compile分层；纯函数动态输入 | [D1](d1-command-reuse.md) | ICB范围R8微测、cold位置、完整纯step与未完成的服务结论 |
| [34](../../experiments/34-native-ba-gates-prework.md) | BA小投影与gates并入prework；BF16→FP32边界 | [B3](b3-gate-fusion.md) | 48层实际布局/只读检查、buffer反射范围、raw-native计时 |
| [35](../../experiments/35-local-probe-stops.md) | 门槛先定；6/6赢不等于达到收益门；负结果也是结论 | [A2](a2-controlled-comparison.md) | half2 FP4、K展开、scaled-ready三候选的diff/控制/原始结果 |
| [36](../../experiments/36-cap64-and-rms-profile.md) | 容量与有效长度不同；CAP64范围；RMS另立数学profile | [C3](c3-state-buffers.md) | CPU补零对照、原ID失败、RMS精度族各参考与新资格记录 |
| [37](../../experiments/37-simd4-async-and-spec-handoff.md) | CAP在构造前固定；异步交付与终态；资源STOP不补点 | [D2](d2-device-feedback.md) | SIMD4独立分支、投机checkpoint探索、service生产/诊断逐例检查 |
| [38](../../experiments/38-spec-transaction-and-qmm.md) | 交付与计算frontier不同；真实历史提交；同输入QMM诊断 | [E1](e1-spec-transactions.md)、[E2](e2-verify-m1.md) | 部分producer异常覆盖缺口、外部官方MTP实验、历史M4/M5交换 |
| [39](../../experiments/39-verify-broadcast-m1.md) | 广播batch=T/M=1；限定verify作用域；首次整模资源STOP | [E2](e2-verify-m1.md)、[E3](e3-draft-loading.md) | 497项输入明细、源码dispatch推断边界、首次加载全过程 |
| [40](../../experiments/40-m1-verify-and-bounded-loader.md) | 逐tensor pop/eval/release；参数不变；后续完整投机资格 | [E2](e2-verify-m1.md)、[E3](e3-draft-loading.md) | 175参数/350对象对照、49事务逐项结果、allocator与后续ready STOP |
| [41](../../experiments/41-native-http-nodelay-pilot.md) | 完整HTTP边界；TCP_NODELAY单变量负例 | [F1](f1-http-measurement.md) | 实际socket读回、warm/formal、启动与trace时钟及每请求记录 |
| [42](../../experiments/42-startup-warm-short-request.md) | 接受率不等于总成本；短输出投机慢；fresh与重复分清 | [E4](e4-spec-cost.md)、[F1](f1-http-measurement.md) | 启动warm修正、全部raw32与HTTP/资源记录、不足以归因的计数缺口 |
| [43](../../experiments/43-fresh-c1-throughput-latency.md) | 三家同负载C1完整HTTP；正确性和结束计数先核对 | [F1](f1-http-measurement.md) | 九次fresh的全部请求、轮换顺序、资源与退出证据；不扩成C4 |
| [44](../../experiments/44-fresh-full-throughput-latency.md) | 吞吐与完整延迟；离散非支配点；两种静态启动profile | [F2](f2-pareto-configurations.md) | 21次fresh、聚合层级、配对比与中位数之比、全部请求/资格/图数据 |
| [45](../../experiments/45-lithos-queued-concurrency.md) | 官方FIFO与真实batch区别；协议终止；跨轮图边界 | [F3](f3-queue-vs-batch.md) | 九次fresh、134文本核对、C1校准、cache日志、资源和全部可复算数据 |

## 代码和原始证据

旧tag与冻结source没有改名或改指向；本次只新增教学层、更新入口和记录。映射保存原文在整理前提交 `095bbb8` 中的SHA256，16完整原文则固定在 `tutorial-16-experiment-v1` 的提交 `f7d9156`。脚本逐字节核对这些文件，结构检查不能代替对知识覆盖的人工审读。

从仓库根目录运行，只有CPU文件读取：

```bash
python3 docs/tutorial-v2/optimization/check_structure.py
```

[46](../../experiments/46-next-ideas-from-papers.md)与[47](../../experiments/47-recent-paper-reading.md)保留在研究区；它们提出后续验证方案，没有被并入“已提速”的教学结论。
