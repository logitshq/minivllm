# E028：新请求加入时，跳过旧状态的恒等 gather

本轮目标是减少 refill 的状态搬运。正确性是前提，最终指标仍是整机输出吞吐和完整请求延迟。
当前为默认 OFF 的实验模块准备阶段；尚无设备结果，也没有新的 Pareto 点。

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
