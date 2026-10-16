# 实验账本

编号是本攻略的索引；旧实验保留原目录与提交，不重新编号原始文件。
所有百分比需连同测量层级、负载和对照一起读。

## 已有成果与反例

| ID | 工作 | 结果/状态 | 证据 |
|---|---|---|---|
| E001 | 固定环境、官方oMLX和Lithos对照 | 已有；Lithos有本机M5 profile，官方接口只测C1 | [原基准](../experiments/m5-ultra-256gb/report.txt)、[Lithos](../experiments/m5-ultra-256gb/lithos-adaptation-v1/m5-ultra-lithos-report.txt) |
| E002 | 有界decode图 | 已采用、opt-in | [图复用报告](../experiments/m5-ultra-256gb/decode-graph-v1/m5-ultra-graph-report.txt) |
| E003 | 单行NVFP4 QMV | 已采用 `f44bc5a`，范围受限 | [QMV报告](../experiments/m5-ultra-256gb/decode-qmv-r1-v1/m5-ultra-qmv-r1-report.txt) |
| E004 | B1–B8投影/GDN融合 | 已采用 `044a263`；更大batch的投影打包数值/性能失败均留档 | [范围及失败](../../bench/m5_ultra_256gb/decode_batch/README.md) |
| E005 | 同一步结束只整理一次状态 | 已采用 `dc745bc`；C64吞吐+46.9%，C32+2.95%，C8−4.35%；旧新非逐对配对 | [结果与限制](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-report.txt) |
| E006 | C64请求、active cap32 | 已测独立配置；排队计入窗口和TTFT | 同上；不能把64请求声称成32请求 |

## E007 · 2026-10-16 · 大batch独立GDN融合

**假设。** B9–B64跳过已失败的投影打包，只融合conv/SiLU/归一化准备与state
更新，可能保留局部收益并显著改善整体。基线commit `266d05c`，生产源码未改。
来源是已有B1–B8 kernel、[oMLX/Lithos阅读](papers.md)和消除中间搬运的设计原则。

**唯一变量。** 四个GDN投影、norm、out_proj都保留原调用，只在实验进程中
替换GDN消费者。新包装复用原Metal kernel，扩大独立batch维到9–64，不改算术。
原型在[代码](../../bench/m5_ultra_256gb/gdn_standalone/candidate.py)。

**预注册门。** 全部逐位检查通过；B32和B64完整前向都至少快5%，才进入独立
HTTP资格验证。先2轮预热，3个ABBA计时块；首个计时调用保留；不删慢样本。
v1结果中的`cold_ms`命名不准确：它发生在正确性求值之后，不代表冷编译。

**正确性。** 16个合成kernel case（4batch×2seed×2scale，非连续输入）通过；
15次完整前向的logits与128个cache数组逐位一致；交错结束和迟到请求的
引擎逐步输出/存活cache一致。B8为不命中的控制组。

| Batch | 原前向ms | 候选ms | 前向加速倍数 | 局部GDN加速倍数 |
|---|---:|---:|---:|---:|
| 8 控制 | 38.361 | 38.504 | 0.9963 | 不适用 |
| 9 | 45.028 | 44.107 | 1.0209 | 1.2745 |
| 16 | 74.529 | 73.037 | 1.0204 | 1.1116 |
| 32 | 89.871 | 88.742 | 1.0127 | 1.1451 |
| 64 | 170.161 | 167.791 | 1.0141 | 1.0671 |

**决定：不接入生产路径。** 未达到预注册门，局部加速没有变成足够大的整体
收益。这是单进程筛选，不是独立进程置信区间或HTTP结果，不加入Pareto图。
峰值MLX活跃内存46.114 GiB；owned子进程已退出并核对不存在。

运行命令（M5仓库根目录，输出目录必须不存在）：

```sh
env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 \
  MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
  .venv/bin/python -B bench/m5_ultra_256gb/lithos/guarded.py \
  --out .artifacts/m5-ultra-256gb/gdn-standalone-owner-v1 \
  --cwd "$PWD" "$PWD/.venv/bin/python" -B \
  bench/m5_ultra_256gb/gdn_standalone/screen.py \
  --workload .artifacts/m5-ultra-256gb/fullqual-build2/workload.json \
  --out .artifacts/m5-ultra-256gb/gdn-standalone-screen-v1
```

原始数据位于命令给出的两个目录；[归档](../experiments/m5-ultra-256gb/systematic-v1/)
包含计划、检查、全部计时与校验清单。输入workload沿用旧实验的本机模型路径。

## E008 · 2026-10-16 · allocator工作集配对

**假设。** 2 GiB池不足以复用高batch的大状态buffer，扩大到16 GiB可能减少
分配开销。来源：M4 agent的[allocator源码发现](../experiments/20-donor-source/omlx/allocator-source-observations.json)
和E007的全模型成本。池上限不是实际已分配量，也不是prefix cache。

**唯一变量。** 同一冻结 `dc745bc` 源码、相同模型/采样/HTTP shim，只改
`cache_limit_gb=2/16`。C1和C64，各三对新进程，顺序2→16、16→2、2→16；
每个进程3组预热和3组正式测量，19输入/32输出，正常EOS。两臂都采集
active/cache/peak和相同的步骤计数；不与旧版未instrument的HTTP绝对值直接配对。

**预注册门。** C64三对都提升且配对吞吐几何平均提升≥5%；C1中位数x/y
回退不超过3%；请求计数、资源、清理验证通过。不是扫描后挑最佳池大小。

**结果：拒绝扩大默认池。** 12个新进程、2,340条含预热请求，其中1,170条正式
请求、37,440个输出token。独立重算原始请求时间戳、计数和组窗口全部通过；
168个资源样本均pressure1/thermal0，12个子进程均已wait/reap。

| 并发/池GiB | 每用户交付估计tok/s | 总吞吐tok/s | TTFT秒 |
|---|---:|---:|---:|
| C1 / 2 | 57.307 | 51.592 | 0.07907 |
| C1 / 16 | 57.292 | 51.579 | 0.07915 |
| C64 / 2 | 4.577 | 177.608 | 4.68467 |
| C64 / 16 | 4.621 | 178.629 | 4.69417 |

表格是三进程中位数。预注册判断用**配对比值的几何平均**：C64三对吞吐变化
依次为−0.656%、+0.819%、+0.235%，几何平均仅**+0.131%**，且不是三对全胜。
C1的x/y分别约−0.027%/−0.025%。不够支持采用，更不能把中位数之比当成配对结果。

C64活跃内存峰值都约43.43 GiB；实际记录的最大空闲缓存从约2.08 GiB升至
16.08–16.10 GiB。缓存限制不是硬实时上限，观测值会略越过设置值。这说明
多保留buffer确实发生了，但在此负载下未产生足够收益；不据此否定所有内存布局优化。

代码：[配对驱动](../../bench/m5_ultra_256gb/allocator_http/run.py)、
[独立重算](../../bench/m5_ultra_256gb/allocator_http/summarize.py)。运行前计划和原始
事件在`.artifacts/m5-ultra-256gb/allocator-http-v1/`；
[汇总](../experiments/m5-ultra-256gb/systematic-v1/E008/summary.json)进Git。

## E009 · 2026-10-16 · 单线程保留GDN状态片段

**假设。** 原`gdn_metal.py`先读S计算`k·S`，再读S计算更新和输出。
每个线程暂存自己的32个FP32元素，可能消除第二次源码级读取。数学、归约、
线程布局、投影均保持不变，输出仍是独立buffer。原型在
[candidate.py](../../bench/m5_ultra_256gb/gdn_state_reuse/candidate.py)。

**反例也预先写明。** 编译器可能已经复用旧读值，缓存可能已命中；显式保存
可能增加寄存器占用或spill。未抓到硬件trace之前，不能宣称DRAM流量已经减少。
不改state精度，不用原地状态覆写，不扩展到M4硬件域。

**门槛。** 与E007相同：B8控制，B9/16/32/64逐位检查、连续步骤和调度检查；
3个ABBA块；B32/B64全前向均快≥5%才进入HTTP资格。GPU执行在E008完成后进行。
运行前已落盘计划和源文件SHA。

**结果：不接入。** 16个合成case、15次全前向及交错调度的逐位检查均通过。
完整前向B32为87.425→87.491 ms（0.9992×），B64为175.524→172.717 ms
（1.0162×）；B8控制0.9988×，B9为1.0045×，B16为1.0006×。没有达到5%门。
峰值活跃内存46.207 GiB，owner已确认子进程退出。

局部B16测出2.2003×，但原始调用在约0.33–1.32 ms之间剧烈变化，完整前向
只变0.06%。保留所有12个局部样本，并明确该单次短micro不足以支持2.2×
可复现收益。没有据此扩大候选范围、重跑挑好数或将其加入HTTP图。
源码级减少读取不等于已证明减少DRAM传输。

[全部结果](../experiments/m5-ultra-256gb/systematic-v1/E009/complete.json)与
[局部原始计时](../experiments/m5-ultra-256gb/systematic-v1/E009/kernel-timing.json)均归档。

## E010 · 待开工 · M5成组prefill/NAX

源码证据：`model/layers.py:prefill_batch_backend_supported`明确排除generation≥17，
因为NAX的M-dependent tile/归约与legacy不同；`model_runner.py`因此在M5对prefill
逐请求调用。E008 C64的TTFT约4.7秒，说明端到端仍有明显的首字阶段成本，
但TTFT也含排队，不能全算成GPU prefill时间。

下一轮先捕获这段实际耗时和NAX矩阵路径，检查是否能保留每序列T维和归约，
仅共享batch/权重读取。小原型覆盖T19/128/512、B1/8/32，先做逐层logits、GDN
时间轴和cache检查，再做TTFT/完整窗口配对。不能只删掉现有硬件guard。
若必须改变归约，提前定义数值误差和留出任务质量门；与exact实验分别记录。

原生固定buffer/设备执行计划、自适应投机、长KV路线继续在[攻略](README.md)
中保留，但按实测成本决定次序。E010此处只有设计和源码证据，尚未运行。
