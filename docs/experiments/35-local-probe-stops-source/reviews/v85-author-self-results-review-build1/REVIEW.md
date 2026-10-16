# V85 作者自审（非独立审计）

仅 CPU 复核保存的结果/源码/哈希；未运行 GPU、资源探测、micro 或集成，旧证据未改。

- receipt SHA `83adff180b95e13e1aa9ba006399d7ab37e7a2d135c40abf8e4eba0cd7d6c083`，状态仍为 `STOP_MICRO_GATE`。
- 64 pins 当前字节一致，含原 56 pins 严格不变子集；原 V81 receipt 绑定正确。
- 30 public +60 captured +480 native strict；20 timed blocks ×30 最终 o/S 数组 =600 strict。不能宣称每个 dispatch 输出都验证。
- 17 numeric +10 pair 只读检查，各 255 原数组；240 原 allocation 与180 mutable arena extent 重新核验不重叠；实际 S6 ReadOnly/S_new9、四 PSO 与依赖 barrier 一致。
- numeric/prime/timed dispatch =2,760/76,800/614,400，总693,960；每 timed block 30,720。
- 重算 A/B = 9481.0831542969/9165.8732096354 ns/dispatch；下降 3.32461956%，6/6 胜；controls PASS，仍低于5%门槛。
- 24 原始资源样本与汇总一致：AC、thermal0、pressure1；swap-used5006.88MB / Swapouts17460761 均未变；已知推理进程仅归属 driver95239。
- owner95238/driver95239，rc2、wait/reap、两 PID absence、external release PASS 均有保存证据；这里只核查这些快照，不声称持续观测所有 GPU 使用。
- 20 个 official CB seconds 时间戳有效；仅支持本微测整个 CB envelope。A 中位数为 18.912111 µs/bank，wall 则为 18.962166 µs/bank。
- `×48≈0.91ms` 只能是同成本假设下的外推情景，不能证明真实整模份额或严格上界。详见 SOURCE-ADVICE.md。

结论：保存计数、统计、hash、资源与 STOP 一致；不重跑、不采用。复核断言 3169 项，脚本/源映射附后。
