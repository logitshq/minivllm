# V131：调度与 hybrid prefix cache 的三个可测增量

研究截止 **2026-10-17**；仅论文、会议官网与作者仓库，网页已实际浏览。建议顺序：**先 aging-only，再时间预算，最后精确 hybrid prefix cache**。全部是新实验建议，未运行 GPU、未改引擎或既有正确性门。论文中的 GPU 收益不外推到 Apple Silicon。

## 先排除“已经做过”

本方 candidate44 的 `engine/scheduler.py:48–71` 已经先安排 decode、再用剩余 `max_batched_tokens` 处理未完成 prefill 和新请求；`model_runner.py:136–150` 已按 computed cursor 执行 continuing chunks。因此 continuous batching、decode-first、chunked prefill **不是待发明的增量**。待改的是 FCFS 顺序和固定 token 预算。`cache.py:14–18,52–56` 的 snapshot 是事务/回滚设施，不是跨请求、可淘汰的 prefix cache。

当前 C1 是 CAP64/direct native feedback，C2/C4 是 accepted25/CAP256 ordinary batching；两种静态启动 profile 没有运行中迁移资格。以下调度实验先放 ordinary batch profile，C1 原路径不动。19→32 的等长、同时到达请求几乎没有长 prefill 干扰，不能用它证明调度策略有益。

据 root 的 M5 tip86757b8 交接：E024 是 forward-only bound 性能，E028 是 identity-refill 数值资格，D512 Lithos 正式测量尚待完成。它们不是现成 HTTP 收益。本报告不把 M5 已列的 SiliconBench、TWP、lookahead、GLM cache、PhaseGate 再称双方新发现。

## 1. 最小改动：只替换 prefill FCFS 为 aging/remaining-work 排序

**来源与日期。** Liu 等《Fairness-Aware and Latency-Controllable Scheduling for Chunked-Prefill LLM Serving》，arXiv v1 **2026-06-08**。作者把等待时间与剩余 prefill 工作组成优先级，decode 仍优先；不是缩短模型执行时间的 kernel 优化。[论文 §3.1](https://arxiv.org/html/2606.09061v1#S3.SS1)；[作者仓库](https://github.com/Charmstok/Chunked_Prefill_Serving)；[作者 Ascend aging 分支](https://github.com/lhx-666-cool/vllm-ascend/tree/aging)。

**真正 delta。** 保留本方 token budget=512、模型/kernel/cache/采样与 decode 顺序，仅在可调度 prefill 候选中按 `α·wait − |β|·remaining` 排序；同分用稳定 request 序号。不加攒批 sleep，不把 native C1 改成 batch。先不用论文的预测器、APC 或抢占，避免把多项机制混成一次结果。

**最小实验。** 两机各自重放相同的 32 请求 arrival trace：24 个短 prompt（128）＋8 个长 prompt（1024），output cap=512，normal EOS；加一个全19→32对照。先固定 Cmax=4，固定一个预声明权重比，A=原FCFS、B=aging-only。记录实际 batch/continuing chunks、各类 waiting/TTFT/E2E p50/p95、总输出 TPS、每请求最大等待；总吞吐不升也可能换得更好尾延迟，不能只报短请求获益而隐去长请求饥饿。

**正确性门。** 先用冻结新 cohort 的独立基线运行保存每请求 IDs 与相同 computed frontier 的 cache/logits；B 自己推进，不能喂 A 的未来 token。调度改变 batch 组成可能改变量化路径：旧同C资格不自动覆盖它。原19→32/cohort硬门、所有新请求输出、无丢失/重复/越序 token、EOS/cancel/drop/重入/readonly 都必须过；不匹配则 STOP，不以“只是排序”免除数值资格。

## 2. 独立第二步：时间预算替代固定 token 预算，随后才加 APC

同一 **2026-06-08** 论文的 LPRS/APC 已有作者实码：[`opt_sarathi_scheduler.py`](https://github.com/Charmstok/Chunked_Prefill_Serving/blob/main/sarathi/core/scheduler/opt_sarathi_scheduler.py) 的 `_predict_latency_ms`（约108–136行）使用 decode/context/prefill 特征；`_get_active_prefill_seq_cap`（195–219）限制活跃 prefill；`_schedule`（319起）先放 decode，再搜索 chunk。该源码当前 main 已读，但尚未锁作者 artifact commit，不是可直接复制到 Metal 的已资格实现。

**真正 delta。** 本方不缺 chunk cursor，缺的是“这轮允许花多少毫秒”。先只把固定512改为**本机实测保守时延表**选择最大可容纳 chunk，保持FCFS、关闭APC；再单独比较 APC on/off，不能直接搬作者 CUDA/Ascend MLP、显存特征或论文阈值。表至少包含 decode batch、KV深度、prefill已处理长度、候选chunk；不要只按总token数估算。

**最小实验/门。** 在同一 ordinary profile 上，固定一条4条D512 decode中途注入1024-token prefill的 trace；A=fixed512，B=time-budget（候选64/128/256/512，阈值在测量前按本机 decode-only envelope 声明）。先逐 chunk 对比整段 prefill 的 logits、KV有效前缀、GDN S/conv 与完整 greedy 输出；chunk改变GDN/QMM规约若失败直接保留STOP。通过后报告总TPS—完整HTTP E2E p50曲线，同时报告TTFT与可观测token gap p95/p99、deadline违例率/调度开销。客户端文本块gap只是交付指标，不能伪称内部逐token ITL。

补充新论文 [SLOWeave，2026-09-07](https://arxiv.org/abs/2609.07883) §3 提供“下一decode deadline以内尽量多做prefill”的窄控制器；本轮未找到可核的作者代码链接，所以不把它列为可直接移植包，也不继承其准确预测下的保证。我们的保守表/回退是待验证设计。

## 3. 长上下文方向：Marconi 式精确 hybrid prefix-state admission

**来源与日期。** 《Marconi: Prefix Caching for the Era of Hybrid LLMs》，**MLSys 2025**；arXiv首版 **2024-11-28**，所读v3 **2025-04-10**，不能写成2025首次提出。[会议论文](https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf)；[作者 artifact](https://github.com/ruipeterpan/marconi)；[`radix_cache_hybrid.py`](https://github.com/ruipeterpan/marconi/blob/main/radix_cache_hybrid.py)。实码 `HybridStates` 同时持有 recurrent/conv 与KV，`insert` 要求新分叉位置的状态，`evict_v2` 考虑低分叉节点。artifact主要针对Mamba/SSM，**不是已证明Qwen GDN实现**。

**真正 delta。** 不做每token recurrent checkpoint；先做1–2个条目的精确前缀缓存，只在明确可复用分叉保存完整 KV＋GDN S＋conv tail＋position/长度/必要最后logits，并把 cache key 绑定完整token前缀、模型/数学profile、dtype、RoPE参数。随后才测“第二次命中再收录”对一次性前缀污染的影响，暂不移植复杂自调 eviction。

**最小实验/门。** P=1024的共同前缀＋两个不同32-token suffix、完全重复prompt、较短部分重叠、同长度不同内容、evict后重载；输出512。A每次fresh full prefill，B精确命中后只算真实suffix。KV有效区、所有S/conv、computed位置、first-logits及完整IDs严格相等；snapshot不与活动native arena别名，分叉后各请求隔离。完全重复prompt须保存最后logits或复用P−1再消费最后token，不能把最后prompt token重复算一遍。未命中必须原路径，缓存内存/复制开销及零复用负载退化也要报。

主指标为包含查找/快照/恢复的完整 E2E latency—总TPS，辅报TTFT、命中token数、缓存bytes、重建/复制耗时；cached workload单独建曲线，不能改写旧uncached19→32胜负。M5 D512先用已资格容量界内的cohort；M4 CAP64 C1不够，先ordinary长容量资格，不临时放宽guard或假称自动迁移。

**明确不选。** [Tail-Replay，2026-08-31](https://arxiv.org/abs/2608.30310) 用短suffix近似重建GDN状态，属于近似质量—性能方案；它不满足当前exact state/IDs合同，不能当精确prefix cache捷径。M5已看的[Token Window Pacing，2026-10-13](https://arxiv.org/abs/2610.17689)与[lookahead batching，2026-10-12](https://arxiv.org/abs/2610.16013)日期均早于截止，但不重复登记为新idea；后者固定batch处理时长的理论模型也不能直接当混合Qwen step时延模型。

## 分工建议

M4先验证 **aging-only** 的真实ordinary mixed-load路径与正确性，M5先在其D512/长容量已资格范围内测 **时间预算**；两机各自校准成本，不交换绝对阈值。精确prefix cache先由一机做checkpoint/分叉/无别名资格，另一机复核相同token workload的恢复边界，再各自做HTTP。所有实验先冻结cohort/门/metric，再跑新marker；不把forward-only、同profile数值PASS或论文结果称新HTTP Pareto点。
