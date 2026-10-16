# M4 Max 最终七点测试与边界

**固定协议、资源守卫与生命周期独立审查PASS；没有稳定双轴整体win，稳定热态性能仍未解决。** Native明确为f976 + gatesON（显式opt-in）、V19 OFF，25源不变，代码默认仍gatesOFF；此前内部C1 gates y+0.935558%低于1%门槛。

实际attempt2完成21/21 fresh，根session57621 exit0。每点3fresh，每fresh3warm+3measure；总270请求/8640tokens，只有135请求/4320tokens用于测量。C1为N/O/L、O/L/N、L/N/O三个共享block；C2/C4为AB/BA/AB。

|配置|C|x 客户端tok/s|y 整机tok/s|TTFT秒|POST→DONE秒|
|---|---:|---:|---:|---:|---:|
|native|1|27.951541|25.135605|0.161493|1.273095|
|omlx|1|29.646924|25.401468|0.214130|1.259770|
|lithos|1|29.488619|21.766048|0.420394|1.470180|
|native|2|13.487015|22.628166|0.491682|2.799684|
|omlx|2|13.156233|20.303770|0.762054|3.115390|
|native|4|12.200735|38.886488|0.731406|3.291086|
|omlx|4|11.623697|34.445485|1.027729|3.640560|

C1 native相对oMLX aggregate x−5.718581%/y−1.046646%；C2/C4 aggregate y分别+11.448102%/+12.892845%，但C2 pair3 x为负，Lithos C1 block3 y为负，不能称稳定双轴整体胜出。TTFT在全部12个signed对照及4个aggregate均更低；保留所有正负、逐fresh值与范围，未筛除慢点。

oMLX为0.7.1.dev1 balanced默认.03：已试C2/C4最佳，C1 balanced/aggressive历史近并列，不能称全配置最优。Lithos为真实m4_max/noDraft/acceleratorON、保守apple-m4-max-40c profile；当前OpenAI服务single admission，仅C1。旧aggressive screen不混入本轮正式点。

缓存口径补充（2026-10-16源码复核）：无跨请求前缀复用与MLX allocator缓冲池是两件事，不能把旧摘要的`cache0`读成allocator上限为0。本轮实际`child.py:35`未传`cache_limit_gb`，采用引擎默认2GiB；oMLX实际CLI在`total_mem>0`时将上限设为`total_mem`。这些是调用代码中的策略，不是实际池内字节、复用率或速度因果的测量；本轮未记录这些allocator指标。原分数和raw不变，详见[实现对照](../../experiments/20-donor-implementation-map.md)。

前8 arms均nominal(0)，包括第8个已经变慢的native C1 block3；第9个oMLX C1 block3才变fair(1)，全部C2/C4均fair。359份原始资源均AC/pressure1，Swapouts总delta0；thermal0/1都符合原guard。不能把变慢因果归为热/降频：本轮没有频率、温度或功率telemetry，也不能称单一稳定热域Pareto。

[跨campaign静态审计](CROSS-CAMPAIGN-SANITY.md)未找到具体源码分母、barrier、pipeline、scheduler或cache bug；旧actual MLX controls为UNKNOWN，新controls显式50/50/20/FAST_SYNCH0，probe cadence由5变17。旧nominal campaign约两倍快的绝对速度差异原因仍UNKNOWN；不同campaign不pool，不事后按热状态筛行。

21server均wait/reap(-15)；实际工具返回后22PID严格absent、21ports ECONNREFUSED。原raw GPU_released=false保留，外部GPU_RELEASED单列。V19仍fullqual严格PASS、E2E资源STOP/DO_NOT_ADOPT/defaultOFF；两次旧fullqual STOP、V19 Swapouts STOP、矩阵attempt1标题cleanup STOP与缺失OMwait均保留。

x是客户端stream delivery proxy，y包括HTTP/queue/prefill；TTFT/E2E按request→group median→process median3→fresh median3，warm不进入测量。Native C2/C4只要求每request raw32=own canonical，未宣称跨batch或官方内部raw-ID等价。

[独立审查](INDEPENDENT-REVIEW.md) · [独立统计](INDEPENDENT-STATS.json) · [全部12 signed对照CSV](SIGNED-COMPARISONS.csv) · [逐fresh与范围](RESULTS.json) · [原证据路径/SHA](EVIDENCE.json)。原独立receipt SHA `ef5b41a3c4d63ea3e75bbb3f081dde25bf06e0a1f0dccc51cdcb68458b7c695c`，seal SHA `aea07cc9eaccda23fa576925da9dc38a5db0564fcb7cd011243f5b568e2ffc9b`。

[静态参考包](m4-final-static-reference.tar.gz)为50文件、343408字节；[包清单](REFERENCE-PACKAGE.json)说明脱敏与限制。历史APIkey已脱敏、模型路径为placeholder、原pins仅historical index；须adapt路径/环境并refreeze，不能直接跑旧命令。原4MB handoff source-reference.tar.gz不变。

[256GB Mac任务书](../../handoffs/256gb-mac/HANDOFF.md)仍要求目标机重建harness、重新完整模型资格并同机重测；源机PASS、旧GO/latch和旧raw不能转移为目标授权。
