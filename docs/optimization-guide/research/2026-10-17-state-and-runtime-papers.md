# 新近论文：状态表示、验证和运行时边界

2026-10-17 只读一手资料。以下外部结果没有在本机复现，不新增 Pareto 点；我们的实验推断另行标注。

## DeltaLog：减少完整状态写回

[DeltaLog，2026-08-16](https://arxiv.org/html/2608.15533v1)把 recurrent state 表示为 dense base 加有界更新日志，周期性合并，覆盖 GDN/KDA/RWKV6。其 5.5 节明确区分精确算术等价与混合精度误差：实现改变累计次序，以容差检查通过，不能直接继承我们逐位状态门。研究主要考察较大 batch，不能外推本机 C1。

**我们的新假设**：只延后状态的物理写回，保留每步原 FP32 运算及舍入次序。记录每次的 decay、key 和 delta，在读 base 时按原次序重放，避免把 decay 相乘或累计更新重关联。这样可能以额外计算换写流量；是否划算取决于日志长度、状态大小与硬件。先做纯 CPU 小递推独立 oracle，检查每步输出/逻辑状态与 merge/reset/clone，再考虑 Metal。CPU 一致不能代替 Metal 的 FMA、exp、归约、原 dtype、ahead 或整模型资格。

本机 Qwen 的48层 FP32 recurrent state 每序列144 MiB；B8整状态1.125 GiB。仅省写回的潜力必须与权重、读流量、attention和重放开销一起计量，不等于整个模型能按写回减少比例加速。当前在跑的 E029 仍只融合 prework，并没有启用日志状态。

## TreeWY：减少投机验证快照

[TreeWY，2026-08-21](https://arxiv.org/html/2608.20961v1)用树形 WY 变换计算 draft 节点，只在接受后重建状态。其 §4 明确生产 kernel 使用 BF16 容差且 token 流不逐位相同；真实树还会使其 vLLM 路径退出整图 capture。收益主要来自内存受限时能接纳更多请求，低并发并不自动更快。

**我们的判断**：这是未来 GDN speculative state 所有权设计的参考；现阶段不直接移植三角求解并宣称无损。先测实际 rollback/snapshot成本及接受分布；保留原状态精度和顺序的有界日志先行。

## BaseRT 与 Apple 上的可迁移性

[BaseRT M5，2026-07-21](https://arxiv.org/html/2607.19438v1)按工作形状把 compute-heavy prefill/MoE交给 Metal4矩阵单元，单 token decode保留专用路径。论文实验机器是M5 Pro，模型/量化不是当前同合同对照。[项目](https://github.com/basecompute/baseRT)可补入外部资格调查，公开速度不直接并图。**我们的判断**：继续使用显式shape dispatch；不能为了使用NAX而改变归约或把B1强行改成更大矩阵。

[MacPaw作者报告，2026-09](https://research.macpaw.com/publications/llm-optimizations-apple-silicon)在其单请求工作负载下发现多种投机方法变慢，说明技术名不能保证跨runtime/硬件/工作负载收益。这里只读到作者公开概要，未据此声称覆盖本机所有投机实现。**我们的判断**：统一记录 draft成本、验证成本、接受数和完整HTTP窗口，失败配置同样保留。
