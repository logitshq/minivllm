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

## E013 · 2026-10-17 · draft分层成本 · 完成定位

固定E012 ON、历史32/256/1024与block3/5/8。v1脚本名遮蔽stdlib profile，导入失败，未加载模型；
v2改名costs.py，同一计划9组108数组相等，全部441个正式计时保留。最大组件均为backbone，
lm_head为完整draft的22.9%–28.9%，未达40%分界。下一步E014窗口KV与E016图复用分别调查。
同步组件不相加，不是优化收益或端到端结论。[教程](tutorials/013-draft-cost-profile.md)、
[含失败的完整证据](../experiments/m5-ultra-256gb/E013-draft-profile-v2/)。

## E014 · 2026-10-17 · 有界draft KV · v1拒绝

136数值检查通过，32K/131072状态回放较快，但2048窗口边界0.9102/0.9556×未过门。
三组ABBA的candidate前后顺序还有稳定差异，可能含跨臂缓存释放成本；不改写v1，不授予
生产或长生成收益。后续分批裁剪与明确本臂生命周期归属另做v2。
[教程](tutorials/014-bounded-draft-kv.md)、[完整原始证据](../experiments/m5-ultra-256gb/E014-draft-window-v1/)。

v2预注册：只在物理KV超过2W时裁到W；每个样本分别计prepare、draft和自己的teardown，
采用门按draft+teardown计算。固定跑2/16 GiB两个新进程，16 GiB作为更强reference的采用门，
2 GiB兼容性单独保留。184数值用例涵盖4095/4096/4097与变长block链；尚待GPU运行。

v2实测两个进程全部368数值检查、240正式计时通过。16 GiB强reference的32K为1.19–1.31×，
131K为1.83–2.22×，短点最小0.9834×，过预注册门。16 GiB末次采样Swapins增加3，
swap用量和Swapouts稳定，无法定位于计时内；按原门保留，不选样本重跑。
[全部phase/资源与审计尝试](../experiments/m5-ultra-256gb/E014-draft-window-v2/README.md)。

实际核心05d75cf四个fresh process完成：2944严格数值检查全过；16 GiB三次门为
PASS/FAIL/PASS，2 GiB兼容性PASS。第二次16 GiB的N32/B3为0.946272×，超3%回退门，
所以整体接入采用门未过，保留默认OFF实验模块。其余点和重复测量照常保留；
不能将原型v2或首轮成功冒充实际核心全面采用。继续看长历史收益与短点波动的原因，
不追加挑选重跑，不授予HTTP或真实长生成收益。

## E015 · 2026-10-17 · 官方oMLX MTP · engine数值资格通过

fresh普通/MTP两进程各3请求，192原始IDs完全相等。MTP每请求16轮、31提案、15接受、
32交付；实际load/attach/route、verify宽度、proxy计数链和draft列表一致。双审在GPU前
修复了误读underlying计数的观察器问题，原模型计数实测为0，不能代表实际投机是否运行。
10资源快照正常，子进程reap与原服务恢复完整。插桩engine API资格，无HTTP速度排名。
[教程](tutorials/015-official-mtp.md)、[原始事件和独立审计](../experiments/m5-ultra-256gb/E015-official-mtp-v1/README.md)。

## E017 · 2026-10-17 · Lithos DSpark · engine数值资格通过

固定Lithos2236580、原target5ff8ef与DSpark b169bc4，三个权重shard已SHA验证。
实际serving.prepare→assets.options→load_session，自然M5 Ultra backend、无Max recipe，
普通/DSpark各fresh process三请求19→32，关闭prefix cache、保留normal EOS。
新pack分别2048/2054容量；固定gamma7与实际accept日志、编译program和rawIDs作资格。
完整最后GPU块可多提交，必须记录真实accepted/committed并与返回前32IDs分开。
9项CPU腐败/CLI检查通过；两进程六请求192个raw IDs严格相等，实际DSpark路由已运行。
首个分歧位置的普通/DSpark同机控制均为ID100606，不能混用oMLX的60277作为oracle。
独立审计每DSpark请求14轮/98verified/17accepted/31decode committed，另加prefill的1 token。
全部327归档成员逐SHA通过，19资源快照稳定，子进程reap、服务恢复通过。
[完整证据](../experiments/m5-ultra-256gb/E017-lithos-dspark-v1/README.md)；不作HTTP/全状态或速度结论。
[教程](tutorials/017-lithos-dspark.md)。

## E016 · 2026-10-17 · draft backbone图复用 · 拒绝

24组312数组严格通过，9点完整draft几何平均1.040008×，最低1.006598×，未过1.10×门。
首次图调用205.702 ms；9次trace包含增长shape，位置变化与同shape切换请求可复用。
108正式计时全部保留；5资源快照正常，exit/reap/恢复完整。不继续增加bucket复杂度。
[教程](tutorials/016-draft-backbone-graph.md)、[独立审计与证据](../experiments/m5-ultra-256gb/E016-draft-graph-v1/README.md)。

## E018 · 2026-10-17 · 短块GDN状态驻留 · GPU初筛预注册

单个kernel串行处理T=1/3/5/8，保留原SPLIT4归约和每步完整FP32状态；
每线程32元素跨token驻留，减少重复读S与dispatch。仅隔离helper，不改core。
目标首层真实输入、真实/零/随机初始状态，165项输出/状态/独立keep前缀严格比较，
任何失败即停止计时。通过后2 GiB池、每臂2warm、3ABBA；T3/5/8几何平均需≥1.25×，
任何T需≥0.97×。6项CPU接口测试与双人源码检查已过，尚待唯一GPU队列。
不以单层结果替代整模事务、projection几何或HTTP资格。
[教程](tutorials/018-persistent-gdn-verify.md)。

v1单进程165严格比较通过，T1/3/5/8为0.982453/1.199745/1.219095/1.578766×，
短块几何平均1.321746×，初筛过门。固定再跑相同源码三个fresh process；
每次单独满足原门才继续，不与初筛池化或挑选补跑。核心尚不使用。

三个fresh复测完成，全部495严格比较通过；连初筛共660项。短块几何平均分别
1.397486/1.384694/1.410092×，但第二进程T1=0.939084×，未满足≥0.97门；
三次结论PASS/FAIL/PASS，因此整体重复性门失败，不采用当前覆盖全部T的候选。
T3/5/8在三个进程中都更快这一有限事实保留。后续可独立验证T1保留原路径、
多token才驻留状态的分派设计；当前没有该分派实现或整模结果，不能改写本轮FAIL。

## E019 · 2026-10-17 · 只提取新增hidden history · REJECT

当前每轮完整concatenate后只取[fed:start]，候选反向收集相交尾部，单piece直接返回，
多piece仅拼所需行。保留完整history、batch reset后的完整重喂、原64/65块压缩逻辑。
独立函数只读取元数据和切片，不导入MLX；严格同dtype/width及sumrows==start契约。
17项CPU边界/CLI测试通过（包括544个小分块/fed组合），主线程与独立静态审查通过。
be12f82冻结后的设备micro：150个资格点通过，120非空点全部计时；48个N512/N2048
tail点几何平均7.247945×，但72个other点中27个<0.97×，完整预注册门失败。
全部single/reset和坏点保留，未重跑，核心不采用；不进入主throughput–latency图。
独立stdlib审计从四份真实BF16 backing重建全部逻辑输入/输出SHA，核原始ABBA与gate，
17测试及6破坏记录检查通过。周期257元素的输入不能覆盖任意长距离错行；真实features、
完整draft和HTTP资格仍未完成。归档保存全部265MiB输入、raw、owner/lease和310个成员，
原七份source pin含教程从be12f82取出后验证，授权服务恢复S。
[教程与预注册门](tutorials/019-drafter-history-tail.md)。
[全部坏点与完整证据归档](../experiments/m5-ultra-256gb/E019-history-tail-v1/README.md)。

## E020 · 2026-10-17 · 官方投机 HTTP 强对照 · 合同通过，收益分化

09568bc 预注册15个fresh server：native target-only三个锚点、oMLX none/MTP三对、
Lithos none/DSpark三对。固定19→32、C1、正常EOS，每进程3warm+3formal；
90请求全部满足各自输出合同，45正式请求实际1440tokens，无重跑或剔除负样本。
主指标是POST→DONE完整E2E和实际completion_tokens/E2E，TTFT另报。

oMLX MTP/none配对throughput为0.903280/0.911512/0.901819×，GM **0.905527×**，
三次完整请求吞吐均下降；TTFT更低不能掩盖E2E回退。
Lithos DSpark/none为1.077841/1.099826/1.080440×，GM **1.085991×**，三次均提升。
native当前target-only三个锚点44.687/44.029/45.615tok/s；
oMLX none为47.106/46.908/47.128tok/s，本轮native尚未超过该普通基线。
允许按明确质量合同比较三方性能：native/oMLX与Lithos的qualified continuation不同；
18次native有当次raw IDs，其余72次官方HTTP只核visible文本重新编码与既有资格，
不声称新raw-ID/full-state等价，也不把其他模块收益乘加进入本轮图。

13项CPU fixtures、双人静态审查、原始SSE/官方日志/三对ratio独立重算通过。
15个直接owned listener PID全部reap及端口消失，原服务identity不变T→S恢复。
225资源快照中Swapins/Swapouts增量均0；首native整个owner区间Pageouts+1、
Compressions+813047，完整保留，不能称内存状态完全不变。
9131个归档成员逐size/SHA复核，8590份冻结source/config独立核hash，
68个大payload/runtime哈希保留runner前后receipt；不归档大weights或运行环境。
六个旁置负向挑战全部被拒绝（visible IDs、时钟、MTP日志、listener/parent PID、restore identity），原tar不变。
[教程及全部fresh数值](tutorials/020-external-spec-http.md)、
[独立审计和原始证据](../experiments/m5-ultra-256gb/E020-external-spec-http-v1/README.md)。

## E022 · 2026-10-17 · native HTTP TCP_NODELAY · v1 readiness FAIL

只切换 stdlib accepted-socket Nagle 选项，冻结复用 E020 模型/core/SSE/client。
6 fresh servers、36 请求的 OFF/ON 三对计划已固定；每 socket readback 不符即整个进程退出。
16 CPU tests 双方通过；新 23 small pins 与继承 8658 pins 分层保留，尚无速度结论。
全部数值、资源与回收先过门，才看三 pair 均 E2E >1、GM≥1.05、TTFT每对≥0.97。
GLM 下载/SHA 后才调度 GPU；不与 E010/E020 历史锚点池化，不声称 GPU kernel 加速。
[教程](tutorials/022-native-transport.md)。

v1 已执行：OFF 第一进程 3 warm+3 formal 完成；ON 第一进程在 readiness 时
得到原始 TCP_NODELAY=4，而校验要求整数 1，exit70；ON 无请求计时，矩阵停止。
这属于 harness 布尔读回检查缺口；没有 paired effect，不进入 Pareto，不复用 OFF 单点。
两个自有进程回收、端口消失，外部服务 identity 不变 T→S。全部 66 文件 SHA 复核归档见
[E022-v1](../experiments/m5-ultra-256gb/E022-native-transport-v1/README.md)。修正版独立 v2，原门保持。

v2 真实 CPU socket OFF=0、ON=4，按布尔语义修复，18 tests root复核通过。
新 plan SHA `e850dcdcc1d160ecddf512529758b32fc9a8241c6af515fb72901c4a6b28c81c`，
24 small pins 与继承 pins 已绑定；此条为执行前准备，未赋予速度资格。

v2 已在3873101提交后完成6server/36请求，独立输出/资源/回收审计通过、0 errors。
完整请求三对speedup=1.017017/1.016630/1.007713×，GM=**1.013777×**；
TTFT三对=.899891/1.005569/1.021203×。GM<1.05且首TTFT<.97，拒绝采用，
正确性通过的全部观测仍保存；不称数值失败、不混进其他批次Pareto。223文件归档SHA通过、
外部服务identity不变T→S，见 [v2全部证据](../experiments/m5-ultra-256gb/E022-native-transport-v2/README.md)。

## E023 · 2026-10-17 · 长生成 throughput–latency · CPU 准备

固定八份编码任务，实际模板后输入886–978 tokens、cap512、正常EOS；两种tokenizer API
与decode round trip一致，七份输入依赖SHA核对通过，未观察答案/速度后选任务。
新HTTP入口每step接收pending，client以C槽固定有限队列及时补位，按实际tokens/共同窗口
算吞吐，按POST→DONE算延迟。服务端18项与client 22项纯CPU测试root复核通过。
独立审查发现的完整请求绑定和raw EOS越cap问题已修复，负测及全部root输出保留；
没有GPU/监听或任何新合格性能点。

长输入还揭示已有快路覆盖缺口：compiled/fused decode guard≤256；grouped prefill
要求同长、T≤512。当前任务应先如实测fallback基线，扩展快路需独立模块、资格与配对。
GLM下载及加载预算审查同步推进，未把CPU准备或模型下载算作运行成功。
[教程](tutorials/023-realistic-throughput-latency.md)。

同调度完整数值资格脚本现已实现，root共52项CPU机制测试通过；37请求、两模型驻留上限、
30分钟绝对预算，实际GPU尚未执行。原cpu-v1保留，TCP布尔读回修复和新资格对应cpu-v2。

## E024 · 2026-10-17 · 长上下文固定图 · CPU 原型

默认 OFF 的单独 graph-only callable，B1–8、used257–1536、K/V backing均≥1536；
固定closure长度、资格上限与backing下限同步定义，不改原256模块、不改D输入。
显式状态输入输出、模型参数作为compile inputs、无packed/fuse；其余形状回原路径。
15 CPU mock tests经root复核，31core source pins一致，两人静态审读无阻断。
真实L1536 attention归约、ahead生命周期、参数替换及完整长任务性能尚未验证，不能进入Pareto。
[实现、推导和GPU候选计划](../../bench/m5_ultra_256gb/long_decode_graph/README.md)。


E023 实机资格计划现已冻结并在 `e9600a8` 后启动：plan SHA
`21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb`，122执行依赖pins。
root独立复核owner的8项CPU检查；子进程1800秒、owner1830秒预算，资源守卫与外部服务恢复
归独立lease负责。运行中的逐步PASS不等于整组通过；计时带完整数值观察，不进入性能图。

E024 新增真实GPU诊断入口（尚未调度）：32个B/长度点、真实参数值替换与恢复、B1→B8→B1、
实际compiled callable见证和cap4 ahead生命周期。root与独立reviewer均复核12项CPU检查；
SHA与日志在 `long_decode_graph/cpu-qualification-v1`。完整D512/gates-ON门仍需后续运行。


E023 v1 实机已因资源门中止：owner859.55s；C1完成8请求4096IDs，C8最后完整step515，
未完成全16请求及13请求lifecycle。首错误pressure2，swap324→469.06MB；无已保存数值差异。
child TERM(-15)、reap与独立PID absence通过，外部服务同identity T→S恢复。资格INCOMPLETE、
无末尾model/source recheck、无性能点；完整332成员归档保留。E024暂不调度，先诊断资源。

## E025 · 2026-10-17 · 相同split-K的NAX prefill部分乘法 · CPU准备

仅T19的GDN z、MLP down与out三形状，保持原split2/2/3与BF16中间舍入，尝试官方MLX NAX helper代替SIMD部分乘法。默认OFF、独立模块，其余形状回原QMM；重建SIMD归约必须先逐位等于官方QMM，然后才能运行候选。root复核10项CPU tests通过；未编译Metal、未运行GPU、未获得数值或性能资格。
[源码、来源、推导与严格执行顺序](../../bench/m5_ultra_256gb/prefill_nax_same_split/README.md)。

## E026 · 2026-10-17 · GDN卷积尾状态独立存储 · CPU准备

已从MLX0.32.3官方源码证实小slice共享整块prefill父buffer；默认OFF的uint16 contiguous复制仅保留最后3行，不改模型算术，立即eval另设开关。root复核7项CPU checks通过；未运行GPU、不归因E023中止、无性能点。先做无权重buffer ownership实测，再做完整引擎资格和吞吐/延迟配对。
[教程](tutorials/026-conv-tail-ownership.md)。

E026 tiny v1已在e03ff2d之后实机PASS：3fresh arms、48case/528phase、全部uint16位域与finite样例通过；T978 retained backing 20,103,168→65,536B，两种ON相同。根线程与独立审计一致，665原始成员逐SHA归档；11资源采样pressure2不恶化、无swap增长，全部reap，原服务T→S恢复。仅该无模型位/所有权诊断通过；不是完整模型资格、不是E023根因或性能点。见[证据与图](../experiments/m5-ultra-256gb/E026-conv-tail-tiny-v1/README.md)。

Cold-restart preparation remains unexecuted: root reviewed the fixed-target controller and reran 23 CPU tests (PASS). Independent read-only interface review confirmed the parsers/saved payload schemas but found required service-absent owner/auditor, pre-spawn-failure recovery, and live reload-monitor/evidence work. User cold-restart approval remains pending; no stop/load occurred. [Review](../../bench/reviews/omlx-controller-interface-audit.md).

E026 scoped integration is now CPU-reviewed (root 12 tests PASS, independent review no blocker), default OFF and limited to ordinary eager prefill. Production source and the original numerical qualifier are unchanged. A single-layer real-weight feasibility analysis is recorded; no integration GPU run or new Pareto point exists. [Hook contract](../../bench/m5_ultra_256gb/conv_tail_owned_copy/README.md).

E026 layer v1 is preregistered before device execution: 608 source/runtime/bundle pins, 14 real layer-0 tensors (65,239,488 original bytes), three fresh arms and eight cases per arm. Root 27 CPU tests and independent auditor 31 negative checks pass. Whole-model correctness and throughput/latency remain unqualified by this diagnostic. [Frozen plan](../../bench/m5_ultra_256gb/conv_tail_owned_copy/prereg-layer-v1/README.md).

E026 layer v1 was aborted before its first case because the footprint parser incorrectly required two OS current readouts to be equal (65,389,240 vs 65,323,704 B). Both are below 2 GiB; this is not an observed memory overrun or numerical mismatch. OFF zero cases, ON arms not started; child/owner reaped and original service restored. Full failed evidence and independent/root rejection are archived; no Pareto point. [Failure record](../experiments/m5-ultra-256gb/E026-conv-tail-layer-v1/README.md).

E026 layer v2 is separately preregistered to correct OS footprint observation handling and cancellation reporting, with the original 1 GiB MLX / 2 GiB physical budgets and model math unchanged. Root32 CPU tests and auditor47 negative checks pass; device execution pending. [Plan](../../bench/m5_ultra_256gb/conv_tail_owned_copy/prereg-layer-v2/README.md).

E026 layer v2 real-device PASS after preregistration aba84a5: 3 fresh arms,24 cases,204 phases; root/independent receipts identical,168 full cross-arm array comparisons/655,616,512 bytes exact. T978 state23,248,896→3,211,264B in both ON arms; active/unique each save20,037,632B. All pins/resources/reap/service restoration pass. Single-layer numerical/state qualification only, no new Pareto point. [Full archive](../experiments/m5-ultra-256gb/E026-conv-tail-layer-v2/README.md).

## E027 · 2026-10-17 · M5 多行共享原始 NVFP4 权重

**预注册前状态：UNRUN，默认 OFF。** 从 M4 第44章保留的原始 shader 建立独立
M5 adapter；不删除 M4 guard，也不把 M4 的局部资格当作 M5 结果。目标是在
2/4/5 行间复用 packed weight/scale 的源码读取，保持每行 K 遍历、qdot 表达式、
SIMD 归约和 BF16 store。准备对象持有原 W/scales，lazy callable 可复用，不缓存 X/Y。

本轮只做数值筛选：27 synthetic +6 real-weight/synthetic-input cases，分别与
独立 native M1 和原表达式控制核比较，共66次完整 BF16 字节比较。固定三个
(N,K)、三个输入/权重bank、三种行数；真实权重只覆盖 N48/N10240。实际 GPU
shape/stride descriptor 独立读取，原输入与所有输出全量保存，第一处不等立即停止。

小型数值实验复用 E026 的独立资源合同（120s child、1GiB MLX观察、2GiB physical
观察，稳定既有 pressure2 可准入），不是修改整模型/性能的 pressure1 门槛。
原 omlx-server74384 只通过用户已授权的暂停/恢复 lease 控制。
[代码与计划](../../bench/m5_ultra_256gb/shared_w_m1/README.md)；运行前另存确切计划、
源码/runtime/bundle SHA 和命令，并先 commit。数值PASS之后仍需整模型和HTTP资格；
寄存器占用、缓存与调度开销可能抵消读取复用，没有预先宣称速度或新增 Pareto 点。

**E027 v1 实际执行追加。** 先提交 `a33249d` 再运行，33case/66原始BF16比较全PASS；
原始M1、表达式control与shared输出逐位一致。110份实际GPU布局记录，68个self
physical phases，11prepared banks均复用一次callable，原X/W/scales只读检查通过。
root重新读取319份独立raw文件（260,130,624B），66对输出实际比较1,945,856B，全通过。
MLX peak30,097,544B、cache0、finalactive0；最大physicalpeak413,172,552B。
6个资源样本pressure2稳定，swap/Swapouts不增，真实free≥62.522GiB。child45274
exit0/reaped/strictabsence，原服务74384 sameidentity T→S恢复。原始证据1,173成员
全保存并重新解包验SHA，见[归档](../experiments/m5-ultra-256gb/E027-shared-w-m1-v1/README.md)。
这是三个几何的局部数值通过，未建立actualmodelactivation/整模型/性能资格；
没有把诊断耗时或memory数字加入throughput—complete-latency Pareto。

E027独立审计追加：PASS；576个pins、33cases/66comparisons、319个raw文件，
以及physical/resource/reap/lease证据均复核通过。审计器6项CPU测试覆盖篡改拒绝。
资格仍限局部矩阵，整模型与HTTP待验证。

E027 native-route bridge追加：先提交28b67d0，再实际执行。当前M5 Linear直接提交multirow QMM，
T2/4/5在g17d走已共享权重的fp_qmv_wide；先前M1逐行oracle不是当前原生路由。
第19案synthetic-N10240-random-T2发现5/20,480个BF16差异；18case完整通过，
56比较=55PASS+1FAIL，其余14case未运行。失败案尚未调用shared，不称原生错误，
拒绝整个三几何直接替换。无放宽误差/forced-M1重设基线/挑前18案入生产。
独立audit与root重跑逐字相同，六种篡改拒绝；580pins、203raw tensor/descriptor文件、
39physical phases、3资源快照核实。child52256 exit1/reaped/strictabsence，服务74384
原identity T→S。942成员原件完整归档31,005,640B，逐成员复读SHA；保留早停缺失项。
[失败归档](../experiments/m5-ultra-256gb/E027-native-bridge-v1/README.md)。无新Pareto点。

Cold恢复v2与service-absent owner已通过root46项CPU tests（16+30），独立审查未发现
错误放行；另诊断API9项通过。真实只读prepare v1与v2两次self sysFootprint接口通过，
后者pressure2、systemphysical约164.6–164.7GiB，绝非cold资源准入。原服务仍同identity S。
新的独立数值/lifecycle审计与运行注册还需完成，停服/重载授权仍待答；未执行cold mutation。
[CPU与只读证据](../../bench/reviews/omlx-controller-v2-cpu-v1/README.md)。


## E025 v2 · 相同 split-K 的 NAX 实机筛选：拒绝

运行前已提交 `4d97a8d`，保持三形状原 split2/2/3、BF16 partial store、原模型权重格式；
候选默认 OFF。第一形状 GDN z/T19 的 OFF 与原始 QMM、重建 SIMD 归约与原始 QMM
两项全量逐位通过，随后 NAX partial 的 233,472 个 BF16 元素有 11 个不同。
首差异 [0,0,2350]：0x3d29 → 0x3d2a。按预注册门立即拒绝；NAX 最终归约、
ON callable、fallback、其余两形状与模型均未运行，不能据此声称最终模型输出错误。

root 与独立 CPU 审计的 audit.json 及原件 manifest 逐字一致；输入全字节未改变，
570 项来源核对，原服务 74384 同 identity T→S 恢复，child55031 exit1/reaped/absence。
保留观察中 MLX peak45,219,886B、physical peak335,659,752B；pressure2 稳定且无 swap 增长。
这些是小型数值筛选的观测，不是性能测量，也不是 full-model pressure1 准入。
[教程](tutorials/025-same-split-nax.md)与[完整失败原件](../experiments/m5-ultra-256gb/E025-same-split-nax-v2/README.md)
保留两项 PASS、首个 FAIL 与未运行范围。无新 throughput–latency 点，不改变现有前沿。

## E023 service-absent v1 · 审计和只读准备完成

原 37 请求数值计划不变。新的独立审计器冻结 SHA `1b7b526d…ddaf6d`，root25项CPU测试通过，
独立审查发现的旧时间采样误用已修复并加负测。实际 controller_v2 只读 prepare 通过，
journal 仍 PREPARED，owner 不存在；实际审计返回 INCOMPLETE/qualified=false，未打开数值日志。
未执行停服、重载或模型工作；cold restart 授权仍待答，现有 pressure2 不满足全模型资源门。
[固定计划、只读记录与 CPU 回执](../../bench/m5_ultra_256gb/pareto_realistic/prereg-cold-service-v1/README.md)。
正确性是入场门，目标仍是完整请求 p50 延迟↓、整机输出 tokens/s↑；本次准备不记作性能进步。


## E028 · 2026-10-17 · refill 跳过恒等 gather · 预注册

独立默认OFF batch adapter，仅旧行完整按原顺序保留、新请求非空且所有KV容量不变时，
省去旧K/V/conv/S的gather，保留原concat及算术。20CPU检查和8个错误计划拒绝通过；
独立静态审读无阻断。固定10cases/88完整字节比较/53资源阶段，570pins；先提交再实跑。
第一阶段只用真实层形状的一attention+一GDN合成状态，检查非连续视图、延迟eval和双向
写入隔离，不代替64层/模型/HTTP资格，无性能点。[教程](tutorials/028-identity-refill.md)与
[固定计划](../../bench/m5_ultra_256gb/identity_refill/prereg-v1/README.md)。
