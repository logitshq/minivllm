# E011：把图复用和投影打包拆开验证

2026-10-17；状态：第一次尝试完整完成，未过性能门，不接入。基线`0d8037b`，计划提交`bb8d628`。

当前C16吞吐低于C8，不能仅凭曲线推断根因。已知代码在B>8退回eager；
旧的大batch融合候选既改变图执行也改变投影打包，曾触发数值差异和B32回退。
这次只问：保留原投影和GDN核，只缓存大batch的计算图，能否移除有价值的主机开销？

原型对现有CompiledDecode隔离地替换两行：资格扩大到B64，融合仍限制B≤8。
不修改生产源码，不把二维权重拼成新投影，不引入原生ICB或新状态owner。
图复用不等于预录Metal命令；它仍可能受state搬运和矩阵执行成本支配。

先检查B8/9/16/32/64各五步的全部logits与128个cache数组严格字节相等，
包括连续步、最后优化位置255与首个fallback位置256。任何差异立即STOP。
只在检查通过的配置测两个预热轮、三个ABBA块，保留所有计时。
B16整模前向至少1.10×且B8/32/64均不回退超过3%，才进入独立调度及HTTP资格。
这是单进程筛查，不能加入HTTP Pareto曲线。计划及源码SHA在GPU工作前落盘。

这一实验对应[MSInfer技术报告](https://arxiv.org/html/2607.16831v1)的启示：
decode的运行时/分派与prefill的结构改写应分别验证；其B200结果不能外推到M5。
另一条更大方向来自[TreeWY](https://arxiv.org/html/2608.20961v1)：减少投机验证状态开销。
该论文明确不是bit-identical token流；不能直接当作本项目exact路径。其内存收益在
并发受限时才转换为吞吐，本机256 GiB是否到该拐点需要先测。

源码：[candidate.py](../../../bench/m5_ultra_256gb/large_graph/candidate.py)与
[screen.py](../../../bench/m5_ultra_256gb/large_graph/screen.py)。
下一节将逐次追加命令、结果、失败和采用决定；不覆盖此前尝试。

## 尝试1：数值通过，收益不足

25组整模前向、3,225个logits/cache数组比较全部逐位相同。
每个配置保留两次首次计时、两个预热轮和三个ABBA块；没有重跑或删除慢点。
首次计时发生在数值资格之后，不能叫冷编译时间。

| 实际batch | 原路径ms | 只加图复用ms | 加速比 |
|---|---:|---:|---:|
| 8控制 | 36.486 | 37.207 | 0.9806× |
| 9 | 45.272 | 42.943 | 1.0542× |
| 16 | 74.133 | 72.467 | 1.0230× |
| 32 | 96.999 | 96.285 | 1.0074× |
| 64 | 209.764 | 206.553 | 1.0155× |

B16未达到预注册1.10×，停止该候选的HTTP升级；不修改生产资格范围。
峰值MLX活跃内存59.329 GiB。自有子进程已wait/reap；授权暂停的外部服务已恢复。
资源和全部原始样本、计划、两行变更后的精确类源码见
[归档](../../experiments/m5-ultra-256gb/E011-large-graph-v1/)。

执行入口为`large_graph/screen.py --workload .../fullqual-build2/workload.json
--out .../large-graph-screen-v1`，由`guarded_v2.py`运行，再由`suspended_service.py`
管理已授权服务的暂停/恢复。完整argv保存在归档`raw/large-graph-jobs-v1.json`。
MLX环境与E010相同：50 ops、50 MB、BFS20、FAST_SYNCH0；源码SHA见运行前plan。

这说明只消除大batch主机构图没有解决本负载的大部分成本，不能进一步推断
某一个GPU核是唯一根因。下一项E012转向DFlash2选路中的逐token主机往返；
测选择器也测完整draft，避免再次以局部收益替代整体收益。
