# E019：hidden history 尾部提取

独立审计 PASS；实验结论 **REJECT_PERFORMANCE_GATE**，不采用当前覆盖全部点的候选。
150 个资格点全部通过；120 个非空计时点中，48 个 N512/N2048 tail 点几何平均 7.247945×，但其余 72 点有 **27 点低于 0.97×**。完整预注册门失败，未剔除 single/reset 或坏点，未重跑。

CPU 审计直接解析四份原始 BF16 safetensors，逐字节检查固定输入 pattern，按独立的 offset、行 stride、列 stride 和压缩布局重建全部 history 与输出 SHA，再核对 150 份独立记录、总表和计时后输入记录。它不导入 MLX，也不复跑 GPU。输入 pattern 有 257 元素周期，不能排除任意长距离错行；仍需真实 features/full-draft 资格。

全部 1440 formal、480 warm、240 first-post-qualification 样本保留。逐点重算 build+eval+release、ABBA 顺序、中位数、speedup 与完整 gate。首个计时调用发生在资格后，不称冷编译。6 项破坏性记录检查均被独立审计拒绝，17 项 CPU 边界/CLI 检查通过。

所有 4 个资源样本为 AC、pressure=1、thermal=0，swap 268.81 MiB 与 Swapouts 保持不变。child PID 5981 exit 0、wait/reap/strict-absence 记录通过；owner PID 5928 完成，授权 oMLX 服务 PID74384 恢复 T→S。进程审计仅覆盖已知命令/精确 owner PID，未声称检测所有 GUI GPU 使用。

archive 保存全部四份 tensor 输入（原样约 265 MiB，未省略）、全部 raw JSON/timing、owner/lease/jobs、源码及审查。7 份测量 pin 源码均从 **be12f82** 读取并对照 plan SHA，包含当时 tutorial；后续文档追加不会改变归档 pin。310 个 tar 成员逐 size/SHA 复核通过，完整 manifest 随附。

这是单进程纯提取 micro。没有目标模型、完整 draft、token/accept 或 HTTP 结果，**不进入主 throughput–latency Pareto 图**。原 history/64 块压缩/核心路径未改。

## 所有未过 0.97× 的 other 点

| Point | Reference / candidate |
|---|---:|
| N19-single-tail1 | 0.935693× |
| N19-single-tail3 | 0.838453× |
| N19-single-reset | 0.611333× |
| N19-short_tail-reset | 0.709184× |
| N19-cross_chunks-tail3 | 0.152938× |
| N19-cross_chunks-reset | 0.946386× |
| N19-offset_single-reset | 0.651591× |
| N19-column_stride_cross-tail7 | 0.867080× |
| N128-single-tail1 | 0.947778× |
| N128-single-tail3 | 0.933843× |
| N128-single-tail7 | 0.869494× |
| N128-single-reset | 0.677260× |
| N128-cross_chunks-tail7 | 0.892529× |
| N128-offset_single-tail1 | 0.950655× |
| N128-offset_single-tail3 | 0.885017× |
| N128-offset_single-reset | 0.656614× |
| N128-column_stride_cross-reset | 0.901485× |
| N128-old65_compress_then_append-reset | 0.876167× |
| N512-single-reset | 0.606195× |
| N512-short_tail-reset | 0.958662× |
| N512-cross_chunks-reset | 0.963575× |
| N512-offset_single-reset | 0.605429× |
| N512-old64_then_append-reset | 0.947826× |
| N2048-single-reset | 0.580915× |
| N2048-short_tail-reset | 0.938529× |
| N2048-offset_single-reset | 0.523429× |
| N2048-row_stride_cross-reset | 0.907573× |
