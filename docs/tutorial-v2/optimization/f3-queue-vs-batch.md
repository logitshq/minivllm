# F3：能接收四个请求，就等于 batch=4 吗

[六篇目录](README.md) · [上一节 F2](f2-pareto-configurations.md)

接着 [F2](f2-pareto-configurations.md)，客户端 C4 表示一组并发发出四个请求，不保证服务端同时收到；GPU 是否一起算，要读服务实现。Lithos 的这次对照正好说明两种能力的区别：

```text
排队：R0、R1、R2、R3 → FIFO → [算 R0] [算 R1] [算 R2] [算 R3]
批量：R0、R1、R2、R3 → batch → [一轮处理四行，再进入下一轮]
```

**baseline → 新做法：** 旧44章使用 Lithos 的 OpenAI 入口；pinned 源码表明重叠请求忙时会拒绝。再读源码后找到同版本官方 Ollama 入口：它会等待 FIFO 锁，然后仍调用同一个生成后端。因此只改协议适配并重新验证，不给 Lithos 私自加调度器。

```text
before  POST /v1/chat/completions → SSE，终止标记 [DONE]
after   POST /api/chat           → NDJSON，终止帧 done:true
```

测量仍从 POST 前开始，等待锁的时间包括在完整延迟里。`max_sessions=8` 缓存的是编译变体，不是八路 GPU batch；服务端 ticket 顺序也不保证等于客户端编号顺序。旧入口没有保存到原始429响应，所以只陈述源码行为，不捏造实测拒绝率。

新入口完成三个并发档位、每档三次 fresh。C1/C2/C4 总吞吐约21.18/20.75/21.10 token/s，完整 p50 约1.51/2.33/3.80秒。请求增多主要拉长排队，总吞吐没有随 C 增长；这与一次只生成一个请求的实现一致。真实 batching 才可能让多行计算共享权重读取，但也有自身成本，须实测。

![包含Lithos官方FIFO并发的描述性对照](../../experiments/45-lithos-queued-source/lithos-queued-throughput-latency.png)

图里本方/oMLX 来自旧44章，Lithos 来自另一轮 AC 测试。两轮的硬件、模型、19→32负载与AC供电域相同，但运行时间与Lithos入口不同，也不是同轮配对，因此只作描述性比较。原图和每次请求记录都保留；没有用新点覆盖旧实验，也没有因此新增本方优化。

**理解题：** 排队服务在 C4 时吞吐没提高，是否意味着四个请求没有成功完成？要看哪些记录才能回答？

原始记录及 pinned 上游源码链接：[45 官方 FIFO 对照](../../experiments/45-lithos-queued-concurrency.md)；[请求时钟](../../experiments/45-lithos-queued-source/requests-normalized.json)、[全部图数据与来源](../../experiments/45-lithos-queued-source/plot-data.json)。读完后回到[六篇目录](README.md)，或按[旧章映射](coverage.md)查实验细节。
