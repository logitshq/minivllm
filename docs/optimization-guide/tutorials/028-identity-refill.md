# E028：新请求加入时，跳过旧状态的恒等 gather

本轮目标是减少 refill 的状态搬运。正确性是前提，最终指标仍是整机输出吞吐和完整请求延迟。
v3 已通过10个真实层形状合成状态场景、88项完整原字节检查；独立原件审计与root复核均已通过。
模块仍默认 OFF，尚未完成全模型与 HTTP 资格，也没有新的 Pareto 点。

## 从真实路径选优化点

`ModelRunner._sync_batch` 在成员变化时调用 `DecodeBatch.rebuild`。当旧请求全部保留且
顺序不变、只追加新请求时，原实现仍先对 K/V、GDN 卷积状态和 FP32 recurrent state
做一遍 `old[keep_idx]`，然后拼接新请求。这里的 index 恰好是 `[0, 1, ..., B-1]`。

实际 E023 长请求日志的 step514/515 均有 B4→B6、L1536 的该类转换。按真实形状求和，
旧 B4 状态逻辑 payload 为 971.25 MiB。这不是实测复制量、峰值节省或速度预测；
MLX 可能消去部分操作，只有实测才能判断。[源证据与算式](../../../bench/reviews/2026-10-17-identity-refill-gather.md)。

## 最小改动

候选保持原切片、padding、concat、行顺序、长度和模型运算，只在 identity refill
且所有 attention KV 容量不变时，把原数组直接传给 concat。新请求必须非空，
因此仍有必要的多输入拼接。其他路径直接调用原 `DecodeBatch.rebuild`。

开关属于独立 batch 实例，默认 OFF，没有全局 monkeypatch，也不默认安装到引擎。
源码与固定执行计划在[实验模块](../../../bench/m5_ultra_256gb/identity_refill/README.md)。

## 先证明数组生命周期

少一次 gather 会改变临时数组与旧 backing 的引用关系。仅比较一次拼接结果不足以
证明后续写入安全，所以第一阶段使用独立构造的 OFF/ON 状态，检查：

- B2→4、B4→6、B6→8 的真实层形状，以及有 offset 的非连续输入。
- 重排、删除、没有新增请求、扩容和空旧 batch 的原路径 fallback。
- 完整 BF16/F32 原始字节、未使用 KV padding、请求 ID 与长度。
- 延迟求值前修改输入，结果仍与独立的未修改原生 oracle 一致。
- 修改 source 或输出时，另一方的值保持正确；两臂不共享可写 array 对象。

这是一个 attention 层加一个 GDN 层的合成状态诊断，沿用已审查的 1 GiB MLX / 2 GiB
物理观察上限和外部 owner 清理。固定计划在实跑前提交。只有这一层证据通过，才能进入
64 层完整状态与真实模型调度验证，然后做独立进程 HTTP 配对；局部诊断不替代这些门。


## 实现依据

已核对固定版本 [MLX v0.32.3 的 concatenate Metal 实现](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/slicing.cpp#L11-L38)：
多输入拼接为输出分配空间，并按输入真实 strides 复制；
[ops.cpp 的单输入分支](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/ops.cpp#L1297-L1299)可直接返回原数组。
因此本候选要求存在新请求，保留多输入 concat，并实测非连续输入和延迟求值。
源码行为支持实验设计，不能直接换算成本机的带宽或吞吐收益。


## v1 输入生成失败与 v2 小修

先提交 `cb3c4fd` 再执行 v1。MLX 已初始化并通过设备门，但第一组输入构造中的
`0x3e00 | low` 抛出 TypeError：array 实现正向 OR，没有 `__ror__`。因此0组比较、
0份状态snapshot、2个资源阶段；factory和OFF batch构造到达，rebuild尚未调用。
child58427 exit1已回收，原服务74384同identity恢复。随后owner的footprint读到已失败
退出的child而报错，这是次生监测失败，不是候选数值差异。

v2另存screen/owner，不改v1：只把两处OR改成array在左，保持全部位模式、候选、
10/88/53覆盖与预算。新增不支持反向OR的fake array，直接运行实际pattern函数，
复现旧写法失败并核对BF16/F32整数位域。23CPU测试与独立审读通过，573项来源固定；
[新计划](../../../bench/m5_ultra_256gb/identity_refill/prereg-v2/README.md)仍须先提交再执行。


## v2 合成输入峰值越界与 v3 时序修复

先提交 `4c134d0` 再执行 v2。B2→4 的8项检查全部通过，B4→6 的输入一致性及
两臂 rebuild 只读检查再通过3项；尚未比较该案输出。第8个资源阶段记录历史
MLX peak 1,131,872,976B，超过原定1GiB门，按计划停止；当时active105,611,296B，
physical peak1,109,951,736B仍小于2GiB。11项已执行检查无数值差异，剩余覆盖未完成。
child58999 exit1并回收，原服务74384同identity恢复S。[原件归档](../../experiments/m5-ultra-256gb/E028-identity-refill-v2/README.md)。

候选不变。v3仅逐个求值合成输入的完整parent，使位运算构图的临时数组可及时释放；
非连续视图仍在parent求值后按原slice建立。原v2也在rebuild前求值全部输入，因此
候选和延迟输出的求值边界不变。保留历史peak、不缩小场景、不放宽预算。26项CPU
测试通过，核对实际value()时序和pattern/run_case AST；设备后再核对v2已有输入字节。
[固定v3计划](../../../bench/m5_ultra_256gb/identity_refill/prereg-v3/README.md)先提交再跑。


## v3 设备结果与可切换安装

`896e5f1` 提交后单次运行，10case/88check/53phase全部完成，108份完整safetensors保留。
历史MLX峰值614,891,549B、物理峰值1,623,885,336B，均在原预算内；cache0，
14次资源记录pressure2稳定、swap无增长。child59771 exit0回收，原服务74384同identity
恢复S。11.76秒是含生成/保存/验证的诊断用时，不是性能计时，不计算优化倍数。

[跨版本独立复核](../../experiments/m5-ultra-256gb/E028-identity-refill-v3-cross-version/README.md)
覆盖v2全部16份文件（14同名、2份失败时输入映射到相同只读边界），每侧671,577,268B
逐字节相同。另从输入原bytes独立重建B2/B4的完整拼接与KV逐head补零，40个输出tensor
每侧493,928,448B完全相同；包括后续source写入隔离和B4首scalar输出写入。root重跑
得到逐字相同回执。此证据证明修复未换输入或缩小已跑场景，不推断原始GPU物理strides。

新的 `identity_refill/integration.py` 提供 `install(runner, enabled=False)`。OFF不读取
runner，不导入候选；ON仅在构造后、加入请求和安装观察器前使用，拒绝非空或spec runner、
重复安装、源码/backend不匹配。所有安装检查只执行一次，成功后只赋值该runner的batch，
不patch全局、不改变eval。7项CPU检查通过；没有接入生产默认，没有实际整模型结果。

下一门复用E023的37请求矩阵与既有完整logits/raw IDs、staging/ahead、退场和EOS检查，
在实际eligible refill后、decode前补全64层完整状态比较。只要这门通过，再用相同模型、
量化、请求与参数做fresh-process OFF/ON：先C1控制与C8 refill，测实际token总数/共同
请求窗口、POST→SSE DONE的p50/p95。较高并发先补对应资格。既有pressure1要求和
服务冷停/恢复授权边界保持；不将本次生成输入的eval带入真实runner或计时路径。


## 独立复核与完整归档

[完整v3原件](../../experiments/m5-ultra-256gb/E028-identity-refill-v3/README.md)保留全部
108份snapshot、880个tensor、5,312,069,632B原始payload。CPU审计独立生成所有输入
位模式（包含非连续view的父索引）并重建10case输出；88组/724个tensor pair共
4,334,575,616B比较全部相同，24组-8/-16/-32首scalar写符合计划。576来源、53资源
阶段、14主机快照、11个外部physical采样及回收/恢复证据核对通过；root重跑完全相同。
1366成员完整归档26,103,106B，每个成员的原字节SHA均经独立与root复读。

[v1输入生成失败](../../experiments/m5-ultra-256gb/E028-identity-refill-v1/README.md)和
[v2资源中止](../../experiments/m5-ultra-256gb/E028-identity-refill-v2/README.md)同时归档，
不覆盖、不删除失败记录。安装helper经独立审读及7项CPU测试；三个设备实验及归档
没有改变任何生产默认。当前收益是局部状态门通过，吞吐/请求延迟收益仍未知。
