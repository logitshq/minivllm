# 共享 W-M1：从严格同输入基线走到新 kernel

现有正确基线是已经整机数值资格通过的 broadcast-M1：它用原始 W/scales 的 singleton views，让每个 verify 行保留 M1 数学，而不是把一个多行 QMM 的结果当成 M1。V124 是其上的独立 kernel 探针：在每个 K512 轮次先把同一输出行的 packed words 和 scale 读到线程局部，再供 T=2/4/5 个输入行使用；每个输入仍保持原 per-lane K 顺序、qdot 表达式、simd_sum 和 BF16 输出舍入。没有重排、解量化整份权重或替换模型参数。缓存寄存器也可能降低 occupancy，不能从减少读权重直接推出提速。

V124 先完成40个 synthetic case、80次完整BF16矩阵比较，只覆盖九种目标几何中的四种。V127 再固定原 V108 实际 B proposal fixture，复现 V112/V113 原始497个 Linear 的 T5 输入、原batch输出及独立native-M1结果SHA。旧V108整体FAIL不变；固定失败运行的真实提案是锁定operator输入，不是重判整模正确。

V127 实际1491/1491严格通过：497原真实T5输入，另994是同X的T2/T4 prefix，并非另两条真实verify轨迹。九种N/K几何×三T均命中新共享W，包含lm_head，无fallback。完整shape/dtype/finite/每字节比较由源内执行并记录；497条输入/原Y/W/scales只读、1349参数只读以及keep0全部128缓存恢复通过。公开选字段保存哈希与执行证据，没有把未保存的全部raw矩阵或post-keep0 tensor虚构成可离线重算材料。

这次只加载target一次，先做ordinary19→32 warm，不加载闲置drafter，也没有live drafting。它不将新kernel结果推进后续层，因此不能继承V117的propagated speculative整模资格。复核者也是harness作者，报告诚实标为author recheck；root另行阅读、执行与实时检查了清理。

下一步缓存计划尚未接入：一个kernel factory配每个原Linear一个prepared对象，只在qualified114 verify scope的T2/4/5启用，默认OFF，其余继续原broadcast-M1；X与输出不能被跨调用缓存。候选性能基准应是现有broadcast-M1，concat多次native-M1只是数值参照。V129仍unfinished/unrun，没有新速度点，也没有采用结论。

M4 Max/MLX0.32.3的结果不会自动转移到M5 Ultra。源码保留M4 guard；M5需独立确认本地MLX、编译器、geometry/actual strides和严格数值，再谈整模与端到端性能。
