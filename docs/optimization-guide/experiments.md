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

## E010 · 2026-10-16 · M5成组prefill/NAX · 已通过配对采用门

**设计。** MLX0.32.3的单请求QMM可能先走split-K；改变M会改变分区及BF16
舍入。M5 NAX还在M=32处切换BM。直接拼成B×T并不满足旧路径的逐位合同。
本轮用广播的三维packed weight保留每条序列的M=T，只共享独立batch维；仅在
原单序列已是splitK1、K满足NAX约束时启用，其余投影保持逐请求调用。
[实现](../../bench/m5_ultra_256gb/prefill_nax/candidate.py)与
[MLX固定版本源码](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp)
可复核；广播不是已测得的DRAM读取次数保证。

**预声明门。** 所有logits与cache逐位一致，B26/T19完整prefill耗时下降至少10%，
再测HTTP。HTTP要求C64三对吞吐全部提高、配对几何平均≥1.05，TTFT比值≤0.90；
C1控制及C8/C32吞吐不能回退超过3%。三对新进程按AB/BA/AB，三组预热和三组正式
测量，不删除慢样本。所有门和源SHA已在运行前写入plan。

**数值结果。** 初始7组：B2/8/26/32×T19、B2/8×T128、B2×T512；448个层级
检查覆盖输出及每行cache，末logits及后续三步decode严格。另22组全模型边界
检查覆盖T1、7/8/9、13/14/16、31/32/33、63/64/65、129、257，B3/7/15/25/31，
以及B8×T512和B32×T128。调度16步涵盖C64、512预算分块、首token停止、交错结束、
混合长度601及迟到请求，输出和全部存活cache签名一致。

初筛prefill加速B8/T19=1.2859×，B26/T19=1.2700×，B32/T19=1.2584×，
B8/T128=1.4197×。**这些计时仍是临时观测**：后续HTTP发现本机已有改名为
`omlx-server`的服务活跃，原命令行matcher漏识别，C1从约56降至37 tok/s。
HTTP v1已中止并保留全部完成/未完成数据，不能用于收益门或新Pareto图；
初筛也不据此声称隔离性能。数值一致性结果仍然有效。

**修正。** 新`resources_v2.py`同时检查可执行进程名；保留v1探针与证据原样。
用户已明确授权临时暂停外部服务并在结束后恢复。新版实验带SIGCONT恢复和看门狗，
已复核最终核心实现，再完成C1/8/32/64 HTTP配对。
本分支的核心候选仅限MLX0.32.3/Apple M5 Ultra、B2–32、T1–512且B×T≤4096；
未知环境及超范围退回原路径。**最终核心实现通过预声明HTTP门，采用；完整前沿补测单独记录。**

[阶段证据](../experiments/m5-ultra-256gb/prefill-nax-v1/)记录计划、逐层结果、
原始计时及所有本机raw文件哈希；后续隔离结果追加，不覆盖本轮受干扰记录。

### 暂停已知并行推理后的最终配对

核心是`b33b282`，复核时直接运行最终实现；reference只关闭runner的分组，
两臂共用相同核心、2 GiB池、graph/fuse decode、HTTP与内存计数instrumentation。
最终核心再次通过22组边界整模logits/cache及16步固定调度输出/状态签名检查。

24个新进程，3,780条含预热请求；1,890条正式请求、60,480输出token。
336个资源样本pressure1、thermal0；已知外部oMLX服务明确处于授权暂停状态。
GUI和未识别的GPU活动不在探针的排除能力内。所有进程均wait/reap并确认PID不存在，
服务通过同一PID/start身份恢复。初次受干扰HTTP仍原样保留，不进入下表。

| 并发 | 配对吞吐几何平均变化 | 配对TTFT几何平均变化 | 配对交付速率x变化 |
|---|---:|---:|---:|
| C1 控制 | +0.32% | −1.21% | +0.22% |
| C8 | +14.68% | −26.29% | +2.62% |
| C32 | +12.30% | −24.42% | +2.89% |
| C64 | +14.54% | −23.02% | +6.40% |

C64三对吞吐比值1.20184、1.17141、1.06731，全部通过；最后一对明显较慢，
未剔除或重跑。C64原型micro的1.27×不能替代这项端到端结果。C64峰值MLX活跃
内存约43.43 GiB，两臂接近；未增加模型权重副本。不是系统RSS或实际DRAM流量。

HTTP允许自由到达，batch划分并非每进程完全相同。C32和C64在两臂中分别都出现
2种、3种原始32-token流；保留各流及频数，不声称跨异步进程每条输出恒同。
**逐位合同仍是相同输入和调度下的logits/cache/输出相同**，固定trace检查未放宽。
请求文本重编码、使用量、32-token长度、全部时间戳和组窗口另经独立CPU重算。

采用后另补C2/C4/C16，各三次新进程，得到当前代码的完整C1–C64曲线。
扩展中的legacy `pair`字段表示重复编号，没有新增reference臂，不计算伪配对增益。
外部oMLX/Lithos仍引用先前归档的固定对照，不声称本轮重跑了外部运行时。
[配对汇总与完整证据](../experiments/m5-ultra-256gb/prefill-nax-v2/)包含全部33进程，
受干扰尝试、两个实现的资格、恢复记录、源码和可校验压缩归档。

最终共2,088条正式请求、66,816个输出token，462个资源样本。当前7个并发配置的
中位数覆盖历史oMLX/Lithos全部9个配置中位数；联合前沿为native C1/2/4/8/32。
C16被C8支配，C64被C32的中位数支配。C32与C64吞吐仅差约0.3%，后者波动更大，
不能据此声称C32具有显著吞吐优势；C32的交付速率和TTFT更好。
外部对照未在本轮重跑，且全部为target-only；尚未比较合法启用草稿模型的外部配置。

## E011 · 2026-10-17 · 大batch只复用图 · 不采用

基线`0d8037b`。把大batch图复用与旧投影打包分开，只改变B9–64的主机构图路径，
不改变投影或GDN核。B8/9/16/32/64先做严格logits/cache检查再做ABBA整模前向。
B16至少1.10×且B8/32/64不回退超过3%才继续调度与HTTP资格。
[逐次tutorial](tutorials/011-large-batch-graphs.md)保存假设、源码和全部后续尝试。

尝试1通过25组整模/3,225数组严格比较，但B16仅1.0230×，B32为1.0074×、B64为
1.0155×，未过收益门，不进入HTTP。B8控制0.9806×、B9为1.0542×，全部样本保留。
原型未接入；[完整归档](../experiments/m5-ultra-256gb/E011-large-graph-v1/)含源码、资源和服务恢复。

## E012 · 2026-10-17 · GPU保留DFlash2选路 · 默认OFF模块采用

基线`604ce07`。用GPU依赖索引代替每个草稿token的两次主机取值，只在最后取回整条路径。
独立开关保持真实旧函数对照；24组合成score/IDs、9组真实draft/全部KV检查后，再分别
计时选择器与完整draft。全draft几何平均≥1.05且所有组≥0.97才继续接入资格。
[逐次tutorial](tutorials/012-device-draft-selector.md)；不调用或声称修复现有SpecRunner回滚。

原型33组score/IDs及9组完整draft/KV通过；selector为1.782–2.809×，完整draft九点
几何平均1.177892×。接入独立模块与默认关闭开关，预声明三次新进程复测实际核心。
第一次CPU接口检查为测试夹具构造错误，保留于tutorial；不属于GPU数值失败。

实际核心`f05c3cf`的三次新进程全部过门，完整draft几何平均1.185085/1.185097/1.190796×，
99组selector和27组完整draft/KV通过。所有27个点均提升，最小1.094524×。
核心31增3删，独立`draft_device_selection`默认OFF；没有改verify/commit，也不计入HTTP前沿。
[接入资格与完整证据](../experiments/m5-ultra-256gb/E012-draft-selector-integrated-v1/README.md)。
