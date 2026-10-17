# M5 Ultra 进展图

每批实验得到可复查结论后新增一个快照。旧图不覆盖；正收益、回退、失败和未完成项一起保存。
每张图附原始 JSON 副本、SHA、逐点数据和可离线重画的脚本。HTTP、模型和单模块分别标明范围。

| 快照 | 本批内容 | 结论 |
|---|---|---|
| [2026-10-17 v6](2026-10-17-v6/README.md) | 新 D512 native C1/C8，128正式请求，各1新进程 | 全周期独立审计PASS；45.37 /167.40 tok/s，p50 11.28 /24.43 s；oMLX/Lithos D512待测，不拼旧cap32 |
| [2026-10-17 v5](2026-10-17-v5/README.md) | 历史C2–64与同轮E020 C1分面；217份来源独立核对 | 正确性硬门下比较真实吞吐/完整延迟；保留被支配点，历史外部C1干扰排除证据不足，未拼接跨轮前沿 |
| [2026-10-17 v4](2026-10-17-v4/README.md) | E020 官方 MTP / DSpark 与同轮 target-only；15 fresh servers / 90 请求 | C1 的 latency–throughput 最优为 oMLX target-only；MTP 吞吐 GM 0.906×、DSpark 1.086×，不与旧并发曲线混用 |
| [2026-10-17 v3](2026-10-17-v3/README.md) | throughput–完整请求latency Pareto；48进程/2304请求从时间戳重算 | 正确性为前置；本负载联合前沿为native C1/2/4/8/32，外部为历史target-only实测 |
| [2026-10-17 v2](2026-10-17-v2/README.md) | E018 单层 GDN 初筛及三次复测；1 张 PNG/SVG | 660 项精确比较通过；复测 PASS/FAIL/PASS，当前候选不采用 |
| [2026-10-17 v1](2026-10-17-v1/README.md) | E010 HTTP、E012/E014/E016 模块及状态表；7 张 PNG/SVG | 全部 E014 进程保留，核心接入的短点失败没有被平均值掩盖 |

最新实验细节见[实验账本](../experiments.md)；完整源代码、资源与进程回收证据见每个实验的归档。
曲线只连接已测点，不推算未测负载；局部收益不相乘为整机加速，也不代表全网最快。

v5随后完成的 [E022 v2](../../experiments/m5-ultra-256gb/E022-native-transport-v2/README.md)
36请求正确性/资源/回收均通过，完整请求配对GM仅1.014×且一对TTFT回退，未过采用门。
这是单独transport配对实验，不池化进v5的历史并发或E020 C1数据；并非数值失败。

[E026 layer v2](../../experiments/m5-ultra-256gb/E026-conv-tail-layer-v2/README.md)随后通过24单层真实权重数值/状态用例；两ON与OFF完整逐位一致，T978驻留state减少20,037,632B。它没有HTTP性能资格，所以v5曲线不增加点。本轮从217来源重画v5，chartdata SHA与原版一致。正确性是硬门槛，优化轴始终为整机throughput与完整请求latency。

M4同步：已接收并离线核验[第44章同轮C1/C2/C4曲线](../../experiments/44-fresh-full-throughput-latency.md)，native在其7个已测配置中构成联合前沿。硬件、轮次和资格范围独立，不合并进M5图。

2026-10-17 E027：M5共享原权重M1通过33个局部数值点、66次原始BF16比较和110个
GPU descriptor核对；[完整证据](../../experiments/m5-ultra-256gb/E027-shared-w-m1-v1/README.md)
与[教程](../tutorials/027-shared-weight-m1.md)已保存。默认OFF，整模型与HTTP性能待验证。
**这不是新的吞吐—完整请求延迟测量，该批没有改变v5的历史Pareto。** 正确性是入图硬门槛，
内存/核级收益作为诊断证据分别保存。
