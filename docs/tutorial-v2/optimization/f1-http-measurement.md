# F1：从“模型跑得快”到“用户等得少”

[六篇目录](README.md) · [上一节 E4](e4-spec-cost.md) · [下一节 F2](f2-pareto-configurations.md)

有了 [D3 的服务入口](d3-service-ownership.md)，才能把自己的实现和 oMLX、Lithos 放在同一请求边界比较。此时我们关心的是：请求成功完成了吗？总共产出多少？用户从发出请求到结束等了多久？

**baseline → 新做法：** 从模型内部计时，移到客户端 POST 前开始，到协议的终止帧结束。下面是测量伪代码；SSE 和 NDJSON 要使用各自的 parser：

```python
start = monotonic()                     # ← 在 POST 之前
for event in post_and_read_stream():
    if first_content(event):
        record_first_content_time()
    collect_output(event)
    if is_terminal(event):
        end = monotonic()               # ← 完整协议结束
        break
require_success_and_expected_output()
latency = end - start
```

首个有效输出的等待时间回答“多久开始看到内容”，完整延迟回答“多久完成”。若只能观察文本 chunk，就必须标成首文本时间，不能假设一个 chunk 正好等于一个 token。只从首文本开始计时，会漏掉排队、prefill 和首段交付。

吞吐要以并发组计。假设四个请求各输出32个 token，最早 POST 到最晚终止共2秒，组吞吐就是 `128 / 2 = 64 token/s`；不能把四个单请求速度直接相加。实际实验按组、同进程重复、fresh 启动逐层聚合；同一进程测三次不等于三个独立 fresh。

旧41章只改变 TCP_NODELAY，保留同样的测量开销：候选约 26.01 token/s，对照约 26.08，未采用。它只有每边一次 fresh，不能包装成稳定收益。旧43章随后做了三家各三次 fresh 的 C1 完整 HTTP 比较；该结论仍只覆盖 C1，后面才补并发。

**正确性边界：** 服务成功、终止原因、usage 和可见文本重分词都要核对。外部接口不提供 raw IDs 时，不能声称已逐 raw ID 相等；不完整请求也不能悄悄从统计里删掉。测试使用的 checkpoint、请求和检查范围应随结果保存。

**理解题：** 若候选首文本更早，但完整请求更晚，它改善了哪项指标？能否直接称为全面提速？

原始记录：[41 传输负例](../../experiments/41-native-http-nodelay-pilot.md)、[43 完整 C1 比较](../../experiments/43-fresh-c1-throughput-latency.md)；可复算时钟：[44 请求](../../experiments/44-fresh-full-source/requests-normalized.json)、[并发组](../../experiments/44-fresh-full-source/groups-normalized.json)。下一节：[F2 Pareto](f2-pareto-configurations.md)。
