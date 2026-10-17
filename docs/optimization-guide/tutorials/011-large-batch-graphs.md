# E011：把图复用和投影打包拆开验证

2026-10-17；状态：计划与原型已写，尚未测量。基线`0d8037b`。

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
