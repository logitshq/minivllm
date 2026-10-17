# 44：重新测完整的吞吐—延迟曲线

本轮在 M4 Max 48 GiB 上，重新测量 Qwen3.8-27B NVFP4 的同一条 19 输入、32 输出负载。7 个已测配置的联合 Pareto 前沿全部来自本方：native C1、C2、C4。三个 oMLX 配置及 Lithos C1 都被同轮的本方配置支配：本方同时具有更低的完整请求延迟和更高的总吞吐。

这次补齐的是一次全新、可比较的 C1/C2/C4 测量。第 43 章的 C1 结果和更早的 C4 数据均没有拼入本图。

## 1. 先看坐标，再看 C

![M4 Max Qwen3.8-27B：吞吐与完整 HTTP 请求延迟的 Pareto 图](44-fresh-full-source/fresh-full-throughput-latency.png)

横轴是客户端发出 HTTP POST 前，到收到 SSE `[DONE]` 的完整请求延迟，单位毫秒，越低越好；它包含 prefill、decode 和 HTTP 交付。纵轴是同一并发组的总输出 token 数，除以最早 POST 到最晚 DONE 的时间，越高越好。旧的“首段文本之后的速度”只作辅助记录，未用作主横轴。

C1、C2、C4 分别表示每组同时发出 1、2、4 个请求。每个请求都生成 32 个 token，所以 C4 的纵轴是 4 个请求合计的吞吐，不是单个用户的 token/s。正确性是进入比较的先决条件，不是图的一根轴。

每个配置有 3 次 fresh 启动，每次先运行 3 个 warm 组，再运行 3 个正式组。延迟先取组内请求 p50，再取 3 个正式组的中位数，最后取 3 次 fresh 启动的中位数；吞吐采用同样的组、启动聚合层次。图中的小点保留全部 21 次 fresh 结果，大点是 7 个配置的中位数。

## 2. 为什么不能直接把 C1 的并发改成 4

当前 C1 使用 CAP64、单请求的原生 ICB 异步反馈路径。它固定了单请求的输入、状态和命令布局；把客户端并发或 `max_seqs` 改成 4，不会自动得到一个合格的四行 batch。

因此，本轮本方是**两种静态启动配置组成的集合**，每个测点重新启动进程：

| 测点 | 实际模型与执行路径 | 调度上限 / 物理容量 |
| --- | --- | --- |
| native C1 | 当前 candidate44；conv + norm reuse + gates；native101 + adapter102 的原生反馈 | max_seqs=1 / CAP64 |
| native C2、C4 | accepted25 / native15 的真实 batching 与原有 B>1 数学路径，gates ON | max_seqs=4 / CAP256 |

C2/C4 的队列、admit 和模型 batch 都是真实并发执行。资格运行已分别观察到 B=2、B=4，拆掉 warm 观察 hook 后再次入队仍形成对应 batch；正式测量也保留了实际 batch 见证。这里没有实现同进程中从 C1 原生反馈迁移到多请求 batch。

## 3. 改的是启动入口，底层已资格路径保持原样

之前的 V125 campaign 只测 C1，入口固定 `batch = 1`，直接启动原生单请求服务。本轮 V128 从配置读取 C，交给 V126 的统一入口。入口在任何模型导入、token 执行之前，选择一次启动 profile：

```python
# before: V125 run_arm
engine = item['engine']; label = item['label']; batch = 1

# after: V128 run_arm
engine = item['engine']; label = item['label']; batch = item['batch']
```

选择规则是一个很小的、默认关闭多并发扩展的函数：

```python
# V126 entry_policy.choose；参数校验见归档原文件
if concurrency != 1 and not enable_full_curve:
    raise ValueError('full-curve startup profile is default OFF')
return 'direct_c1' if concurrency == 1 else 'legacy_batched'
```

`direct_c1` 委托给逐字节相同的 V125 服务；`legacy_batched` 委托给保留旧 native15 batching 核心的服务，不加载原生 replay adapter。该替换没有引入新 kernel、speculative decoding 或 V124 shared-weight 实验。完整选择代码、原服务快照及 before/after diff 在 [source 目录](44-fresh-full-source/README.md)。

## 4. 同轮重测的结果

| 配置 | 完整 HTTP 请求 p50（ms，低优） | 总输出吞吐（token/s，高优） | 联合前沿 |
| --- | ---: | ---: | --- |
| native C1 | 1226.301 | 26.0947 | 是 |
| oMLX C1 | 1274.119 | 25.1154 | 否 |
| Lithos C1 | 1461.269 | 21.8988 | 否 |
| native C2 | 1470.632 | 43.5117 | 是 |
| oMLX C2 | 1570.562 | 40.1541 | 否 |
| native C4 | 2011.992 | 63.3790 | 是 |
| oMLX C4 | 2306.974 | 54.3799 | 否 |

按配置中位数相除，本方对 oMLX 的完整延迟分别降低 **3.753%、6.363%、12.787%**，总吞吐分别提高 **3.899%、8.362%、16.548%**。C1 对 Lithos 的延迟降低 16.080%，吞吐提高 19.161%。

配对百分比的中位数是另一个统计量：对 oMLX，C1/C2/C4 的延迟改善分别为 3.719% / 6.101% / 12.906%，吞吐改善为 3.863% / 8.060% / 16.548%。四种比较各保留 3 对原始结果，本轮 12 对均同时改善两根主轴。计算明细同时保存两种口径，没有用“中位数之比”替换“配对百分比的中位数”。

黑线只连接已经观察到的联合非支配点；各家的细线同样只是已测配置之间的阅读引导，不证明线段上的任意新吞吐—延迟组合可实现。被支配的配置仍保留为空心大点。Lithos 的固定 endpoint 不支持 C2/C4 admission，记 N/A，不画零点或串行替代点。

## 5. 这个结论覆盖到哪里

这轮完成了当前 **19→32、C1/C2/C4、已测配置集合**的完整同轮比较。oMLX 使用此前已测的 balanced.03 配置；Lithos 使用实际 M4/noDraft 服务配置。它们是明确版本、明确配置的对照，不代表任意模型、长度、并发或所有可选优化下的全局最好结果。

21 个 fresh arm 按预定顺序运行，之间固定留出 60 秒已卸载冷却；资源在请求组以外采样，要求 AC、thermal nominal=0、pressure=1，且 Swapouts 与 swap-used 不增加。所有测点的请求、退出与端口证据均保存。图中离散重复点是测量波动，不是置信区间；3 次 fresh 重复不足以宣称统计显著或解释某个优化贡献。

本方 C1 与 C2/C4 的 source、物理容量及执行路径不同，结果应读成一个可选择的静态配置集合，不能归因为“某一个 kernel 让整条曲线翻倍”。旧 STOP、未资格实验、M5 数据和旧轮次点均未参与统计。

复核入口：[安全时钟与 token rows](44-fresh-full-source/requests-normalized.json)、[组聚合](44-fresh-full-source/groups-normalized.json)、[配置与配对结果](44-fresh-full-source/plot-data.json)、[来源、复算与资源范围](44-fresh-full-source/README.md)。
