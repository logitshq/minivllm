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


E028 v1实际在首输入生成器停止：MLX不支持int|array，0checks/0snapshots/2phases；
candidate factory到达但rebuild未调用，不称数值拒绝。child58427已reap，原74384恢复S。
v2只修两处array-first OR和版本绑定，候选/位模式/预算不变，23CPU tests通过；
[独立v2注册](../../bench/m5_ultra_256gb/identity_refill/prereg-v2/README.md)冻结573pins后再运行。


E028 v2实际：1完整case/11检查PASS后，B4构图后的历史MLXpeak1,131,872,976B越过
1GiB门而停止。B4输出未比较，无完整资格。child58999已回收、原服务恢复。v3只将
输入parent逐个求值，保持全部位值、非连续view、输出delay、10/88/53与预算；26CPU
检查通过，576pins固定，[v3预注册](../../bench/m5_ultra_256gb/identity_refill/prereg-v3/README.md)
提交后才运行。无新性能点。


E028 v3（提交896e5f1后单次设备运行）完成10cases/88checks/53phases/108全量snapshot。
MLXpeak614,891,549B、physicalpeak1,623,885,336B，原预算通过；child59771exit0/reap，
原74384恢复S，14资源采样pressure2稳定且swap不增。独立v2/v3全16文件字节相同，
B2/B4的40输出tensor由原始输入重新拼接核对PASS，root复跑一致。另新增默认OFF
单runner安装helper，7CPUtests通过，未接生产。两层合成state≠64层模型/HTTP资格，
无新Pareto点；完整原件审计与归档随后补记。


E028 v3完整独立审计与root复核完成：所有108snapshot/880tensor原始5,312,069,632B
解析并独立重建各case全部输出，88groups/724pairs通过；576pins及完整资源/回收/
恢复证据通过。1366成员无损归档26,103,106B，逐成员SHA重复核验。v1/v2失败同时
保留。[v3原件](../experiments/m5-ultra-256gb/E028-identity-refill-v3/README.md)。
资格严格限两层合成state矩阵，完整模型和HTTP仍未完成，Pareto图未新增点。


E028整模型准备追加：新的私有qualifier保留原37请求，补实际identity refill后的
64层/128数组比较与D-C8命中门；新owner复用原资源生命周期。root66 CPU checks和
157+5来源的真实CPU prepare通过，未GPU、未服务操作。固定计划在
[prereg-model-v1](../../bench/m5_ultra_256gb/identity_refill/prereg-model-v1/README.md)。
新增证据仍需独立审计、真实整模型及HTTP资格，当前没有新的throughput–latency点。


E028整模型独立审计追加：新数值审计保留原37请求门，独立重建identity refill与
fallback计数并核128数组；新owner审计核真实nested路径/CLI及原资源恢复证据。
root实际CPU prepare检查157+5文件和241,872,766B runtime，PASS。新controller只读
prepare后journal仍PREPARED；真实审计INCOMPLETE/qualified=false/ledger_opened=false，
owner目录不存在。冷停/恢复授权仍待答，未GPU或服务变更。

Lithos官方/api/chat NDJSON客户端已补，按实际终场接收时间统计完整请求延迟，C8
测试保持8个HTTP在途、mock单生成峰值1。首次60CPU tests通过后，独立审读发现
EOF后closer取消漏报，保留原版与窄修复负测。[审计和修复记录](../../bench/reviews/2026-10-17-pareto-adapters-cpu-v1/README.md)。
客户端只条件校验计数；实际raw-ID/每条零cache日志与全模型仍待验证。无新Pareto点。


## Recorded HTTP replay preparation and approved E028 execution

Native HTTP raw SSE/UUID/scheduler join、独立双引擎实际调度回放及固定来源wrapper已完成，
28 CPU tests通过；真实24条D请求经冻结server/tokenizer解析通过。root修复startup identity
的int/float timeout不一致，保留原版和重跑证据。只有CPU/mock资格，暂无新Pareto点。
实际HTTP仍需capture owner及native client取消close的窄修复；不把准备工作当速度提升。
[可复跑记录](../../bench/reviews/2026-10-17-http-replay-cpu-v1/README.md)。

用户现已明确授权“允许停止、实验后恢复”。接下来按冻结E028 model v1执行冷停→完整
37请求→回收→恢复原oMLX模型，并核新日志中的MTP/resident PLE。先前pending记录是历史
快照，当前不再等待同一授权；失败也执行恢复。M4分支53d4906已在本分支祖先中。


## E028 full-model result and restoration: PASS (2026-10-17)

Frozen model v1 completed 37 requests / 5163 steps / 12411 raw=committed tokens.
Twelve actual identity refills (C8 seven, lifecycle five) passed full 64-layer/128-array
checks. The full ledger records 6,624,069 bit-equal comparisons. All requests hit caps;
normal EOS was enabled but early EOS did not occur. Original v1 audit rejected the
C8→lifecycle free-slot ordering; separately reviewed v2 carries the independently
reconstructed reused scheduler state. Root 5+6 CPU tests and the full v2 audit PASS;
original failure and sources remain intact, with no GPU rerun or relaxed comparison.

Original oMLX is RESTORED as PID74456 with its original model, Lightning MTP and
resident PLE confirmed in fresh logs. Model child68871 and load worker74512 were
reaped. Pressure1 held across212 samples; swap usage/Swapouts did not grow, but
Swapins increased40. Instrumented elapsed1087.833s and dual-engine MLX peak69.648GiB
are not production performance. No HTTP/performance qualification or new Pareto point.
[Final evidence](../experiments/m5-ultra-256gb/E028-identity-refill-model-v1/README.md)
archives all692 files with full member/source SHA verification. This supersedes earlier
pending/running status snapshots; approved cold stop and restoration both completed.


## Actual HTTP baseline: frozen C1/C8 capture/replay preparation

The repeat cold cycle now binds the previously verified RESTORED PID74456 through
immutable journal/audit hashes. A real v3 source owns its POST worker and preserves
the original persisted-visible contract; no source-path spoofing or arbitrary
listener adoption is used. The new outer owner runs exactly C1 capture→reap→replay,
then C8 capture→reap→replay, with four separately recorded direct children.

Root18 controller +17 owner CPU tests PASS, including cross-module recovery for
preflight failure, a known returned Popen with failed identity probe, and the
capture's inner intent written before a rejected Popen call. Unknown spawn and
unreaped children still block restoration; failed numerical work does not.
Repeated cancellation waits for an in-flight resource probe before reap.

Actual small-source CPU prepare and service read-only prepare both passed;
journal is PREPARED, no service change or GPU execution at this recorded point.
Plan SHA `d641a95b7a454ce45ea01c3f3b3b0bc59f08ec71de0f0d5c79973315f7861429`.
[Registered workload, exact plan and receipts](../../bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v1/README.md).
Fixed D/normalEOS/cap512 requests are8 at C1 and16 at C8; E028 isOFF. Captured real
UUID arrivals are replayed by independent engines; recorded schedules/tokens are
comparison targets. No performance claim follows from CPU or instrumented timing.
The existing user authorization applies to the same stop/experiment/restore scope.


## HTTP baseline v1 readiness failure and v2 retry (2026-10-17)

Actual v1 cold stop succeeded, but the first capture rejected the owned listener
before any HTTP request. macOS `lsof -Fpn` returned the valid three fields
`p81230`, `f4`, `n127.0.0.1:18023`; the old parser required exactly two fields.
This is a harness readiness failure, with zero requests and no numerical result.
Child81230 was terminated, waited/reaped and strictly absent; the original oMLX
was restored as PID81337 with its original model and fresh MTP/resident PLE logs.
A separate offline restoration audit PASS independently checks raw physical/log
and child-absence evidence. Warm KV and exact hidden runtime state are not restored.
All original sources and failed artifacts remain unchanged.

The separate v2 capture permits one optional ASCII decimal descriptor field while
still requiring exactly the owned PID and localhost endpoint. V2 owner changes only
the capture source binding. Root28 capture/owner tests and12 restoration-audit tests
PASS; independent source reviews found no remaining blocker. V2 retains exactly
the D workload, C1 eight/C8 sixteen requests, normal EOS, cap512, model/precision,
resource limits and numerical contract. E028 remains OFF for this baseline.

[Complete failed-v1 evidence](../experiments/m5-ultra-256gb/E023-http-baseline-v1/README.md)
and [registered v2 retry](../../bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v2/README.md)
separate the historical failure, restored predecessor and next attempt. No new
Pareto point follows from this retry preparation; only independently qualified,
separate production timing can update throughput vs POST→DONE p50.


## Actual D512 HTTP qualification PASS (2026-10-17)

C1/C8 completed 24 actual HTTP requests, 5124 replay steps and 12,288 committed tokens.
Independent full-ledger audits account for 6,572,973 original-dtype bit-equal checks
and 13,145,946 finite checks across full vocabulary logits/cache/ahead state. All
requests hit cap 512; normal EOS was enabled but actual early EOS coverage is zero.
This covers recorded arrivals only. E028/E024 remain OFF and no timing point is awarded.

Both audit failures remain preserved: integer 0 versus the HTTP producer's float 0.0
sampling metadata/ahead key, and a strict-positive file-size check that rejected
three valid empty package files. Separate v2 auditors fix only those expectations;
root CPU regressions and complete actual audits PASS, with no raw record edits or
GPU rerun. Full source/runtime/model provenance and 249 raw resource samples passed;
pressure 1, zero swap/Swapouts/Pageouts growth, Swapins +3. All four children were reaped.
Original oMLX recovered asPID 90204 with the original model and fresh MTP/resident PLE
logs; warm KV/exact hidden runtime state are not restored.

[Complete D512 evidence](../experiments/m5-ultra-256gb/E023-http-baseline-v2/README.md).

Observer-OFF formal native C1/C8 specs now bind these actual independent receipts
and pass CPU prepare/validate: 8 warm + 64 formal requests each. The next graph will use
actual committed tokens/common window versus full POST→DONE p50, separate from the
historic cap 32 panel. Official oMLX's D512 route/count observer and Lithos launch/cache
contract are prepared; neither has a new D512 measurement yet.


## Native D512 observer-OFF timing: two audited points (2026-10-17)

One fresh process per C, eight warm then 64 formal requests, same fixed NVFP4 model
and D fixtures, normal EOS/cap 512. Complete-cycle independent audit PASS:

| Concurrency | Requests | Committed tokens | Common window (s) | Output tok/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 64 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| C8 | 64 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |

These are single-run baseline observations. C8 raises total throughput and also
request latency; both measured native points are nondominated among these two.
They do not establish dominance over oMLX/Lithos, whose D512 points are pending.
The cap 32 history stays in its own chart. Actual early EOS coverage here is zero.

All 144 warm+formal raw requests passed protocol/count/text checks. Representative
captured C1/C8 histories supply full-logit/state numerical eligibility; no claim
is made that every new formal schedule was replayed. All 216 resource samples
passed, pressure 1, swap−8 MiB, Swapouts/Pageouts 0. Original service recovered
asPID 98702 with fresh MTP/resident PLE evidence; warm KV/exact hidden state are not
restored. Both actual experiment children were strictly reaped.

[Complete 935-member evidence](../experiments/m5-ultra-256gb/E023-native-formal-v1/README.md) · [New D512 chart](progress/2026-10-17-v6/README.md).


## E024 long decode graph rejected at its first point (2026-10-17)

The unchanged candidate failed exact BF16 full-vocabulary logits at B1/used257:
228,516/248,320 finite elements differ, maximum absolute difference 0.28125.
Independent raw safetensors audit and root rerun agree. Zero of 32 matrix points
passed; later argmax/cache/parameter/lifecycle and final source/model rechecks
were not reached. No speed or Pareto point is awarded; no tolerance was relaxed.

The original service recovered as PID 832 with fresh MTP/resident PLE evidence;
independent recovery audit PASS. Five resource samples had pressure 1, zero swap
growth and zero Swapouts/Swapins/Pageouts. Warm KV/exact hidden state are not restored.

The first-principles lesson is to isolate tensor shape from compilation. E024
changed attention length from 257 to 1536 as well as compiling the forward pass.
Matching MLX 0.32.3 source predicts different SDPA reduction paths on g17d across
K=1024. This is a hypothesis about the failure, not a measured cause. The next
separate diagnostic compares eager/compiled at each length from identical state.
Original evidence and the default-OFF candidate remain frozen.

[Complete 637-member failed-attempt evidence](../experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md).


## Lithos D512 qualification preparation (2026-10-17)

The official FIFO server now has qualification-only taps and a raw NDJSON join.
Actual returned IDs, full prompt, terminal EOS/count/text, cache miss and route
are bound through original wire lines; response completion order is never used
as identity. Actual POST→DONE interval scanning rejects serial histories labelled
C8. Observer-OFF timing will use separately bounded original log windows for
cache=0/setup=0; logs have no per-request wire IDs. Root and author each passed
15 focused CPU tests. This prepares request/route/count qualification; it does
not grant full-state correctness or a performance point.

[Implementation, limits and recorded preparation attempts](../../bench/m5_ultra_256gb/lithos_d512/README.md).


## Four-arm graph diagnosis: bound, not a speed result (2026-10-17)

Independent raw audit confirms all129 output tensors bit-equal for eager versus
compiled at the same bound (257 or1536). Changing bound257→1536 reproduces the
original difference:121/129 tensors,35,960,404 words. Four131-array inputs stay
identical; all outputs finite, parameter value receipts unchanged. Scope is only
B1/used257. This isolates bound as sufficient at that point; no runtime kernel
trace, full candidate qualification or performance point is claimed.

Original service recovered asPID11548 at16:43:37 UTC, independently audited;
13 resource samples had pressure1/zero swap growth. Complete small evidence is
archived;13 original full-state tensor files (3,312,040,693B) remain locally with
committed hashes/paths. Next source-backed hypothesis uses1023/1536 bounds to
preserve the native reduction route; it still needs40 boundary points and lifecycle.

[Result and evidence](../experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md) · [Tutorial](tutorials/024-decode-graph-dispatch.md).


## Route-preserving graph canary passes (2026-10-17)

The new default-OFF two-bound candidate passes its first B1/used257 point.
Independent original-byte audit: eager257 versus compiled1023 all129 outputs
equal (89,803,264 elements/255,104,000B each), both131-array inputs unchanged,
actual compiled key1023/invocation1. All finite;1349 parameter value receipts
unchanged. The successful initial weight/prefill payloads were not independently
recomputed. No higher-bucket, full matrix/lifecycle/HTTP or speed qualification.

Child13629/owner13596 were reaped; service restored asPID13908 at17:04:28 UTC,
independently audited with original model and fresh MTP/resident PLE evidence.
Warm KV/exact hidden state are not restored. Ten samples pressure1/zero swap
growth. The seven full-state tensor files (1,783,330,137B) remain local;677 small
evidence files and exact local hashes are archived. Candidate gates remain OFF.

NEXT: 40 fixed batch/length points, actual cross-boundary/mixed-row/batch reuse,
then parameter replacement/ahead lifecycle and gates-ON D512/HTTP. No new Pareto
point; external D512 baselines remain pending. M4 tip53d4906 is already an ancestor.

[Evidence and limits](../experiments/m5-ultra-256gb/E024-route-canary-v1/README.md).


## Route graph boundary matrix passes (2026-10-17)

The default-OFF 1023/1536-bound candidate now passes all52 fixed diagnostic
steps: B1..8 at used257/1023/1024/1280/1536, uniform and mixed-row actual
cross1024 chains, and B1→B8→B1 reuse in both buckets. Source-pinned device
checks cover6708 full output tensors plus6192 prefill tensors; full-byte input,
output and unchanged1349-parameter receipts close. Independent audit ran once:
ten complete raw snapshots, three full output contrasts/387 tensor pairs, all
bit-equal. The remaining successful tensors are device/hash receipts, not saved
full raw payloads. No parameter-replacement, ahead/scheduler or HTTP eligibility.

Child17212/owner17177 reaped; original service13908 restored as18261 at17:32:45
UTC with original model/config and fresh MTP/resident PLE. Independent restore
audit passes; warmKV/hidden state not restored. Resource46 samples pressure1,
thermal0, zero swapgrowth/Swapins/Swapouts/Pageouts. Child222.566seconds includes
loading, checks and serialization: not performance. MLX peak56,951,132,480B.

Evidence:760 small files archived (1,184,892B); ten full tensor files remain
local (7,902,883,133B), exact hashes/paths committed. Source plan was pushed as
294f2a5 before execution. M4 tip53d4906 remains an ancestor after fresh fetch.
M4 must independently qualify its hardware reduction routes before enabling.

NEXT: narrow forward-only OFF/ON graph versus eager timing at B1/B8 and
used1023/1024, with same-state full bit gates, excluded warmup, fixed ABBA and
observer-free timing. This go/no-go diagnostic precedes further HTTP/ahead
integration. NativeD512 C1/C8 remains45.372/167.399 outputtok/s at p50
11.283/24.432seconds (n=1 per point); no new Pareto point or oMLX/Lithos
dominance is established.

[Matrix evidence and limits](../experiments/m5-ultra-256gb/E024-route-matrix-v1/README.md).


## Route-preserving forward timing passes (2026-10-17)

Preregistered in b221b83, eight B1/B8 × used1023/1024 × gates OFF/ON
configurations complete all32 warm/96 formal samples. Each arm has six formal
samples in three ACCA blocks. All full original-bit checks pass. Compiled median
forward-rate gain is5.44–10.67%; gates ON: B1 6.68–7.01%, B8 5.68–5.92%.
All24 block ratios are positive observations within one process, not confidence
intervals. No HTTP throughput/latency qualification and no competitor comparison.

Frozen numerical auditor ran once:1548 qualification outputs,2096 inputs,
3096 prefill checks,33280 sample checks,1349 parameter receipts. Successful raw
payloads were not saved; device/hash receipts only. Resource21 samples pressure1,
thermal0, zero swap growth/Swapins/Swapouts/Pageouts;Decompressions+264. Child
90.565s includes loading/checks. MLX diagnostic peak56,450,814,156B.
Owner19484/child19518 reaped; service restored as19870 with fresh MTP/resident
PLE and original model/config; independent restoration passes. WarmKV not restored.
706 small files archived at800,396B with complete manifest/hash verification.

NEXT: actual Lithos D512 C1/C8 qualification then formal timing. Graph remains
OFF pending parameter replacement, ahead/scheduler and HTTP. Native D512 points
remain45.372/167.399 outputtok/s at p50 11.283/24.432s, n=1 each.

[Results and evidence](../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md) ·
[Forward-only figure](progress/2026-10-17-v7/README.md).


## Lithos D512 first actual qualification rejects double templating (2026-10-17)

The preregistered actual run (47ba0fd) completed eight C1 HTTP requests, then
rejected the first full-prompt identity comparison. Our request builder put the
already templated fixture.prompt_text inside user.content, so official Lithos
applied its chat template a second time. All eight actual inputs gained eleven
tokens. The complete official CPU tokenizer reproduction matches both the
incorrect captured inputs and, from fixture.messages, the original frozen IDs.
This is a request-construction bug, not an inference numerical failure.

C8 was deliberately skipped after C1 failed. No throughput/latency or Pareto
point is awarded. The original prompt-ID oracle remains the contract; the retry
uses the original messages and keeps that exact gate. A source-preserving new
harness version avoids overwriting any failed raw evidence or registered source.

Owner24167 and direct server24265 were reaped. oMLX recovered asPID25221 with
the original model/config and fresh MTP/resident PLE evidence; independent
restoration audit PASS. Warm KV/hidden state are not restored. The frozen
qualification/resource auditors reject this incomplete two-phase cycle before
full evidence inspection; their FAIL status is not a finding of resource pressure.

[Complete failure evidence](../experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v1/README.md).


## Lithos D512 retry: actual C1/C8 qualification PASS (2026-10-17)

The two-line original-message correction was preregistered and pushed as418281d.
Actual C1 eight/C8 sixteen requests pass the original full prompt-ID, route,
raw-token count, decoded-text and zero-cache gates. Independent post-restoration
qualification/resource/restoration audits each ran once and all PASS. Total
12,288 raw/committed tokens, all cap512; early-EOS coverage remains zero. C8 is
concurrent HTTP with official FIFO GPU execution. No full-state cross-framework
equality or formal performance point is granted.

All70 resource samples pass: pressure1, swap growth0MiB, Swapouts/Pageouts0;
Swapins+12, Decompressions+1634. Owner26867 and servers26920/27564 reaped.
oMLX restored asPID28531 with original model/config and fresh MTP/resident PLE;
warm KV/hidden state not restored. Archive603members,1,562,748bytes, all member
hashes and source trees rechecked. Failed v1 stays preserved separately.

Next: observer-OFF8warm+64formal per process, repeated C1/C8, preserving actual
committed throughput and full POST→DONE latency. The existing native n=1 points
and D512 graph are unchanged; Lithos/oMLX formal D512 points remain pending.

[Complete qualification evidence](../experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2/README.md).

## E023 Lithos D512 formal v1 registered · 2026-10-17

Preregistered six fresh observer-OFF official processes: r1-c1,r1-c8,r2-c8,
r2-c1,r3-c1,r3-c8, each8warm+64formal. Exact real-v2 qualification, original
messages, normalEOS cap512, fixed NVFP4 weights/pack and same five dependency
groups. Plan SHA ab8d035bf53f65855143ac9c5f2713d0a119bcd81a9504766a0aa8b376612523.
89 integrated CPU tests plus5 fake execution/recovery paths PASS. Actual closed
qualification → mock official HTTP/child → full trial offline audit also passes.
Source/metadata and read-only service prepare PASS; current original PID28531.
No actual new formal request or performance point at registration.

The resource auditor's numeric filename membership correction supports more
than999 samples without removing chronological/resource checks. Pre-registration
schema mismatches (raw point metadata and scheduler SHA field) were repaired;
fixture-only test failures are recorded. Fixed originals and failed experiments
remain untouched. M4 bea0a02/ce068d8 were fetched and merged before registration;
response and proposed work split are in the Git handoff.


## E023 Lithos D512 formal v1 completes · 2026-10-17

Six observer-OFF fresh processes,48warm+384formal,196608 formal committed tokens. Each C has three process observations. Frozen independent formal/resource/restoration audits ran once and PASS. C1 process-metric medians44.910185tok/s, p50 11.402589s; C8 44.916329tok/s, p50 91.183519s. All cap512, early-EOS0. Native historical n1/C remains45.372147/167.398907tok/s and p50 11.283287/24.431815s. New v8 plots eight observations without pooling/CI/interpolation. oMLX D512 remains pending; no global-frontier claim.

999 resource samples pressure1,swap growth0MiB,Swapouts/Pageouts0. All six actual child servers reaped; original oMLX restored as85613 with original model/config plus fresh MTP/resident PLE evidence. WarmKV not restored. Evidence archive2695members/3769816bytes; complete member/source/tree verification PASS.

Next prepared CPU suite actually ran91tests:77 unaffected tests pass; oMLX's14-test module had9errors/1failure (loaded-identity fixture, macOS resolved temporary-path mismatch, missing Pydantic in root test interpreter). Original output preserved; fixture/interpreter repair runs separately without weakening production gates. E024's59 lifecycle/integration/schedule/auditor/driver CPU tests pass. New HTTP entry is only source-reviewed, not covered by these tests.

[Formal evidence](../experiments/m5-ultra-256gb/E023-lithos-formal-v1/README.md) · [D512 plot](progress/2026-10-17-v8/README.md).

Focused oMLX test repair subsequently PASS14/14 in0.385s under the existing pinned oMLX Python. Only two fixture path normalizations changed; production gates unchanged. Original failed stdout/stderr, all actual tested v1 source bytes, focused before/after/command/pins are in [CPU evidence](../experiments/m5-ultra-256gb/prepared-cpu-2026-10-17/README.md).

## E024 route serving: original37 numerical/resource/restoration PASS · 2026-10-17

The default-OFF per-instance adapter completes D-C1 eight, D-C8 sixteen and lifecycle thirteen requests:5,163 steps/12,411 committed IDs. Candidate short compile/fuse/gates ON, norm OFF; reference eager/gates OFF; E028 OFF. Independent schedule reconstruction and all6,622,533 device array-comparison receipts PASS:10,315 full-vocabulary logits,6,596,864 state arrays,15,354 ahead-token arrays. Successful tensor bytes were not retained for offline arithmetic recomputation.

Idle parameter coverage is42 direct calls (6 baseline+18 changed+18 restored), three parameter paths at used1023/1024 and B1→B8→B1,18 meaningful eager-change receipts and17,068 additional comparisons. Actual D graph calls: C1 bound1023/1536=723/3365; C8=88/938. C8 observes7 partial retirements and7 survivor refills; lifecycle8/5. Consumed ahead totals5,118, but discarded ahead0 and observed EOS0. No cancellation/sampling-key invalidation, live weight replacement, combined E028 or unseen-capacity qualification.

Independent resources217samples PASS: pressure1/thermal0, swap growth0MiB, Swapouts/Pageouts/Compressions0; Swapins+12, Decompressions+6053. Owner86493/child86527 reaped; original oMLX85613 restored as92152 with original model/config and fresh MTP/resident PLE. WarmKV/exact hidden state not restored. Independent restoration PASS. Archive935members/3,850,220compressed bytes; original1,628,001,846-byte numerical ledger and6,012,475-byte route ledger preserved, all tar members/source trees rechecked.

No HTTP/performance point is added. Next route-ON C1/C8 capture→strict server reap→independent replay/route audit must pass before observer-OFF throughput vs POST→DONE latency. Current native/Lithos formal frontier is unchanged; oMLX D512 remains pending.

[Closed evidence and limits](../experiments/m5-ultra-256gb/E024-serving-v1/README.md) · [Tutorial](tutorials/024-decode-graph-dispatch.md).


## oMLX D512 actual C1/C8 qualification closes · 2026-10-17

After committed registration bccad9c, two fresh official servers completed C1 eight/C8 sixteen D512 requests:12,288 committed tokens, all cap512, EOS0. Full prompt/raw IDs/text/usage/cache and real target-only route pass independent raw reparse. Peak HTTP outstanding1/8 and actual GPU batch1/8; C8 contains881 batch-8 response events. This is protocol/route/count qualification, not cross-framework numerical/quality equality or timing.

Independent resource41samples PASS: pressure1, swap growth0MiB; Swapouts/Pageouts/Compressions0, Swapins+8, Decompressions+373. Both servers93587/94094 and owner93524 reaped; original service92152 restored as94592, fresh MTP/residentPLE/originalmodel/config verified, warmKV/hiddenstate not restored. Separate restoration PASS. Owner inventories match before/after; independent auditor does not rehash weight payloads or claim recursive runtime closure.

Observer request seals preceded owner SIGTERM-15 and strict process/listener absence. They are not generator EOF/process-close certificates; actual observer_closed=false and generator_eof_proven=false remain visible. Archive414members/3,267,180compressed bytes, all member/original hashes and tree membership checked. Next is observer-OFF warm8/formal64 fresh-process timing, not reuse of diagnostic timings. [Complete evidence](../experiments/m5-ultra-256gb/E023-omlx-d512-qualification-v1/README.md).


## Route HTTP replay preparation and progress v9 · 2026-10-17

The new per-instance HTTP entry/capture passes the 12 server and11 capture CPU tests (the original23-test attempt preserves one macOS temporary-path fixture failure; only that fixture was fixed). The new replay adapter separately passes13 guarded CPU tests; no GPU/runtime/model/network/process backend was exercised. A later one-string docstring correction has an AST-equivalence receipt and is not claimed as retested. Actual HTTP capture/replay, independent route audit and formal performance remain pending. [Replay CPU evidence](../experiments/m5-ultra-256gb/route-http-replay-cpu-v1/README.md).

[Progress v9](progress/2026-10-17-v9/README.md) retains exactly the same eight formal points as v8 and shows new E024/oMLX qualification status separately. PNG was visually checked; no diagnostic timing or invented new Pareto point was added.


## oMLX D512 six-process formal registration · 2026-10-17

Observer-OFF official CLI, C1/C8–C8/C1–C1/C8, warm8/formal64 per process, same D512/target-only/cache0 protocol. Four existing actual inventory groups retained, mandatory owner pre/post rehash;48 source pins and actual PREPARED-journal sealed. No service stop or performance measurement yet. [Registration](../../bench/m5_ultra_256gb/omlx_d512_formal/README.md).

CPU first attempt79:78 PASS and one correct rejection when a real outer-runtime inventory test ran under vendor Python. The unchanged exact test then PASS under its actual outer Python. Both attempts/source snapshots preserved, zero real subprocess/network/signals/device imports;80 executions,79 distinct tests passed, no combined rerun. No weakened production gate. [CPU archive](../experiments/m5-ultra-256gb/omlx-formal-prepared-cpu-v1/README.md).


## oMLX formal v1 stopped at real usage alias gate · 2026-10-17

First server completed8warm/4096committed; official BaseUsage includes input_tokens/output_tokens aliases not represented in the synthetic fixtures/allowlist. Counts were correct; local validator rejected before formal. Owner/server were reaped, and original94592→98900 recovery independently PASS. Formal/resource audits retainFAIL (six phases not completed); no performance point. Full356member/2,997,492B archive retains exact failure. New versioned retry adds exactint alias equality and actual original SSE regression; old sources unchanged. [Evidence](../experiments/m5-ultra-256gb/E023-omlx-d512-formal-v1/README.md).


## oMLX formal retry v2 registration · 2026-10-17

New realsourcefolder/outputcycle; strict optional official count aliases, original count/cache/wire unchanged.83 CPU casesPASS (82vendor+1actualouter), including exactfailedwarmSSE for producer and independent auditor. Current98900 PREPARED after formal-v1 recoveryPASS, earlier24request successfulcountqualification retained separately.49 sourcepins sealed; same6fresh processes/warm8/formal64. [Registration](../../bench/m5_ultra_256gb/omlx_d512_formal_retry/README.md); no formalpoint yet.


## E029 GDN 长路径预处理 / SIMD 标量候选准备 · 2026-10-17

当前 oMLX formal-v2 独占 GPU 期间只准备新源码：gdn_prework_unbound 保留五次原投影，独立启用现有融合辅助函数；gdn_simd_scalars 尝试共同标量单 SIMD 计算一次。均默认 OFF，未运行设备、无性能资格。真实激活 canary 固定两个 D512 提示、三层、两组三步链，18 局部调用；完整输出/conv/S 原 dtype 比较、候选状态自续、normal EOS，不能替代整模型验证。CPU 测试源码尚待本轮服务恢复后执行。新 HTTP cycle 及 GLM physical guard 也仅源代码准备；未封存前序拒绝实际服务操作。[教程](tutorials/029-unbound-gdn-prework.md)。


## oMLX formal-v2 execution closes; first offline parser fails · 2026-10-17

All six actual observer-OFF processes report completion of48warm/384formal requests. Original service restored98900→19019; independent restoration PASS. The frozen resource auditor's outer child/port-absence parser incorrectly expects stderr, which that pinned producer does not retain; resource FAIL and formal FAIL-before-raw are preserved. The six points are not yet added to the chart. All original1707 archive members/189,576,831 logical bytes and49 registered sources are verified. A separate corrected offline audit must bind actual inner stderr evidence and reparse all432 retained requests; no source/data/failure overwrite. [Original evidence](../experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2/README.md).

## E024 real HTTP capture/replay registered · 2026-10-17

New135-pin plan `ba7556ffddd28323ce0a7541cc1c6a8557378a687fbdace981c610bfc7b2129d`; actual read-only PREPARED from independently restoredPID19019. Four direct phases are C1 capture/replay then C8 capture/replay, with8+16 actual requests, original37/42 serving prerequisites, source/runtime/model checks, observed graph dispatch, full numerical replay and cold-service restoration. Registration grants no actual HTTP or speed result.

[CPU evidence](../experiments/m5-ultra-256gb/next-modules-prepared-cpu-v1/README.md):173 initial tests,172pass/1fixturefail; production unchanged, repaired fixture's entire12-test module passes. Both attempts archived,185 executions/173 distinct tests with passing evidence; no claimed clean full-suite rerun. GDN/SIMD/GLM/v3 candidates remain defaultOFF and device-unqualified.


## oMLX formal-v2 post-run independent audit PASS; progress v10 · 2026-10-17

New versioned auditors correct only the outer reader schema and bind12 actual earlier inner stderr/reap receipts. Six CPU regressions PASS; actual resource597samples pressure1/zero swap growth and full432 raw request reparse PASS. Original two FAIL receipts,49 pins,1707-member archive remain byte-identical. New40-member/127072B supplement SHA fbd9378fa4e2f6c8a83ac97629e8b061caa040e6822f656f8ea73b78b85ac501 records every new source, command, audit and source-manifest binding; this is explicitly post-run, not preregistered.

oMLX process-metric medians C1 50.781568tok/s,p50 10.079909s,p95 10.128053s; C8 168.365034tok/s,p50 24.263937s,p95 26.130037s. Native historical C1 45.372147 and C8 167.398907tok/s: C1 still behind, C8 nearly equal with native p95 25.979063s. No cross-cycle significance or overall frontier claim. New visually checked [v10](progress/2026-10-17-v10/README.md) shows14 actual points, separate p50/p95, source-pinned lineage and all prior eight point values unchanged.

E024 actual HTTP four-phase qualification started after registration4ec464d; C1 capturePASS and numerical replay currently live. No E024 performance timing promoted to chart. GDN source candidates remain defaultOFF; registration/driver preparation proceeds independently without device overlap.


## E024 actual HTTP capture/replay independent PASS - 2026-10-17

Four actual phases close: C1eight/C8sixteen,12,288 committed IDs,5,124 replay steps,6,573,486 full-array device receipts independently checked. Both bound families execute; C8 has seven partial retirements and seven survivor refills, ahead invalidation0/EOS0. Full successful tensor payloads are not saved; source-bound exact receipts are audited, not offline arithmetic.250 raw resource samplesPASS, zero swap growth; all direct children reaped. Original19019 restored27913 with fresh MTP/residentPLE, independentPASS. Archive1333members/4,544,516B SHAaec7947a4afd39d2b17c560ef45014b7d4b2b2181732d1bff44292437c09f717. [Evidence](../experiments/m5-ultra-256gb/E024-route-http-v1/README.md). No new formal speed/Pareto point.

Next GDN canary registered: ownerplana54e3e74,child17e3e23e,actualPREPARED journal81263452,68pins. v3 migrated only four actual predecessor hashes, algorithmAST unchanged,24 fake lifecycle testsPASS;12 driver testsPASS. One18-call local real-activation gate, defaultOFF and not combined withE024. Source-only formal adapter12tests first attempt had5failures/1error from unresolved macOS fixture path; fixture-only resolve() fix yields full12PASS, oldattemptretained and productionunchanged.


## E029 actual real-activation canary independently PASS · 2026-10-17

Registered 18 calls / six candidate-owned three-step state chains pass. Independent audit rechecks all 60 saved raw files, 54 output/conv/S pairs, 24 causal pairs and six reference full-vocabulary argmax checks; 162 device comparison arrays, fallback 0. Reference supplies each next activation, so this remains local rather than whole-model candidate rollout. Seven resource samples PASS, zero swap growth. Child 29163 / owner 29126 reaped; original service 27913 restored as 29328 with fresh MTP/resident PLE and separate recovery PASS.

Complete archive: 1,040 members, 185,486,960 logical bytes, four compressed parts totaling 166,005,544 bytes, stream SHA46ba33137f1729cb42d2ca673211e022c73b81b563ce1828d6057693e0b2d33f. [Result and evidence](../experiments/m5-ultra-256gb/E029-gdn-prework-canary-v1/RESULTS.md). No performance point. Next is the unchanged37-request full-state scheduling gate; separate E024 graph formal timing has twelve actual prepared specs but no execution yet.

[Prepared CPU archive](../experiments/m5-ultra-256gb/cold-v3-route-formal-prepared-cpu-v1/README.md) retains v3 four-constant seal/unchanged algorithm, whole24 lifecycle PASS, original formal12 fixture failures and fixture-only whole12 retry PASS. [Actual formal spec preparation](../experiments/m5-ultra-256gb/route-http-formal-prepare-v1/README.md) binds all12 OFF/ON specs to actual HTTP PASS; it is CPU preparation, not timing.


## GLM direct-child owner: 12 CPU tests PASS · 2026-10-17

The separate GLM owner now has complete fake-only wiring/cleanup tests:12PASS, no failures/errors/skips or forbidden backend/process/network/signal operation. Unknown launch remains unknown; a returned child is reaped even if identity inspection fails. It retains actual physical observations separately from performance/full-correctness claims. [Evidence](../experiments/m5-ultra-256gb/glm-owner-prepared-cpu-v1/README.md):142 members,2,114,303 logical bytes,322,436 compressed bytes,SHAae431c2229e4be17f476d7aadde2f2b1d11d1e1d3ca0830acabdb38c1a4d3bc2. The controller/registration remains absent and execution fails closed. No GLM load, capacity or inference performance result is claimed.


## E029 original37 same-schedule numerical qualification PASS · 2026-10-17 19:26 PDT

All37 requests complete: C1 8/4096steps, C8 16/1027steps, lifecycle13/40steps;12,411 committed tokens. Independent scheduling/full-logits/state/ahead receipt audit PASS:6,622,533 bit checks,13,245,066 finite checks. Actual48-layer candidate/fallback submissions independently reconcile; B1 candidate stays separate from original multirow fallback. No observed EOS, no HTTP/performance/parameter-replacement/composition claim.211 resource samplesPASS, pressure1/thermal0/swapgrowth0; Swapins+8/Decompressions+2126. Original29328→43565 restored, independentPASS; warmKV/exact hidden runtime state not restored.

[Actual result and evidence](../experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1/RESULTS.md):1343members,1,642,768,789logicalB,10,656,372xzB,streamSHA83343891d17826fb1045660863ad72b71e2e5ca05b9de3086e6e182b4adc316c. All source/raw bytes retained; no successful full-tensor offline arithmetic or new Pareto point. Next E024 formal cycle has62 new fake-only CPU testsPASS with unchanged source,0fail/error/skip; no formal speed result yet.


## E024 actual registration rejects overwritten plan digest · 2026-10-17

The real guarded registration built twelve specs, but the source-copy loop reused the variable holding the owner-plan SHA. Saved commands then named the final test-source SHA instead of the actual plan digest; the real owner CPU preparation rejected it before any service preparation/stop. Preserve the actual v1 registration FAIL and partial preregistration unchanged. The earlier62 fake CPU tests remain their actual PASS evidence but did not cover this registrar bug. A separate v2 registration and saved-command regression are required; no measurement is lost or promoted.

Progress [v11](progress/2026-10-17-v11/README.md) was actually rendered and visually checked. All14 formal observations and six process-metric median groups remain exactly v10 values; only the independent qualification table advances.


## E024 paired formal v2 actually PREPARED · 2026-10-17

Fixed only the registrar digest variable and moved execution paths to v2. All64 CPU testsPASS, including two actual saved-command/seal regressions; original62PASS and failed realv1 registration remain immutable. CPUv2 archive335members/824452B/SHAa84ae43c0d6ba2c6cf0f917235f3dd154ce3400e5d27b3611c5eea6b0658ae42. Original registration FAIL archive103members/328280B/SHA371f79397c093ac8bdaf0fa03028e494eea045fc967aa657bdad10f7a35cf8ae.

Actual guarded v2 registration and CPU owner preparation PASS; subsequent read-only service prepare bindsPID43565. Plan0920aceae,PREPAREDjournalda0e9518,76pins. Twelve fresh processes retain3 adjacent OFF/ON pairs each at C1/C8,96warm+768formal requests, identical cap512/normalEOS/gatesON/normOFF/E028OFF and frozen transport. [Preparation evidence](../experiments/m5-ultra-256gb/route-formal-registration-v2/README.md):115members/335704B/SHA23e5ed64fb5c2394e98b91b91bfed6d42b592959f63414d0d2ab34cbc2cbb86e. No measured graph performance point yet.


## E029 GDN-only HTTP adapters source-prepared · 2026-10-17

Added a separate default-OFF HTTP entry, captured-history replay and independent audit under [gdn_prework_http](../../bench/m5_ultra_256gb/gdn_prework_http/README.md). Native worker/transport/scheduler and the original numerical comparator are reused. Only48 GDN instance hooks change; E024/E028 remainOFF. Capture counters observe real Python submissions without array evaluation. C8 requires actual multirow fallback, not an invented B8 candidate hit; its tail may executeB1.

Root and independent agent reviewed the four modules; AST parsing succeeded. Sixteen focused CPU test methods are written but **not executed**, to keep the ongoing E024 formal timing isolated. Source manifestSHA9d1ba2ab648f9ea10e0c0fea512257314358f838abb6705e4f27ca260fc98a1a. No new HTTP, numerical or speed qualification. Outer lifecycle/registration remains required; prefill_batch and unobserved EOS/cancellation/composition remain outside this adapter's current scope.


## E024 future display policy and C8 hypothesis · 2026-10-17

Prepared [v12 chart source](progress/2026-10-17-v12/README.md) before ON results: retain all14 historical observations unchanged and append all12 paired OFF/ON points only after the entire formal/resource/restoration/archive chain passes. Historical native n=1 remains separate from OFF n=3. No rendering or current-link change yet.

A [source-only GDN B2–8 design](../../bench/reviews/2026-10-17-gdn-prework-batched-next.md) identifies reuse of existing batched prework/state kernels while retaining five native Linear calls, original B>1 gates and state ownership. It preserves packing regressions and per-row M1 arithmetic failures. No candidate implementation, test or speed claim for this extension.


## GLM cold-load cycle source prepared · 2026-10-17

Added separate controller, input inventory builder, registration/seal, explicit execution driver and independent restore/load audit. The existing original probe/owner/physical guard remain frozen. The input builder retains original62 shard SHA provenance and stats current files; it hashes only the supplied runtime/source set, not204GB of weights. No real input plan, service preparation or GLM load has happened.

Root and independent agents reviewed the source. Review caught an auditor error before execution: aggregating token events by case could accept wrong order or an empty drain map. It now checks actual before/after/terminal/reset ordering and all seven queue counters. Thirty-three fake tests are written and AST-parsed, not run during active E024 timing. [Prepared source and usage](../../bench/m5_ultra_256gb/glm_flash_cycle/README.md). The future load diagnostic is separate from full-model correctness, quality, physical-capacity or speed qualification.


## Batched GDN prework candidate source prepared · 2026-10-17

Implemented the earlier [B2–8 design](../../bench/reviews/2026-10-17-gdn-prework-batched-next.md) as a separate [default-OFF B1–8 adapter](../../bench/m5_ultra_256gb/gdn_prework_batched/README.md). The small runtime delta is a batch-shaped guard and two reshapes, plus a pin for the original state kernel. It preserves all five native Linear calls/current parameters, original B>1 gates fallback and state commit/exception behavior; existing fused Metal math is reused. Original B1 source and current formal pins remain unchanged.

Root and independent source review found no concrete blocker. AST verification confirms unchanged loader/install scopes and all8 native source pins. Twelve new fake test methods are written but unexecuted. This new domain has no device/full-model/HTTP/performance qualification; it is not added to the active GDN HTTP16 CPU task or any speed chart.


## Offline continuation actually queued · 2026-10-17

The active E024 driver will attempt its original service recovery before the separate audit/archive continuation. New one-cycle waiting processes now serialize GDN HTTP16 and GLM33 complete fake CPU modules after actual closure, then generate the fixed v12 figure only on full formal PASS. Actual parent wait-return evidence prevents overlapping offline stages; a child result file alone is insufficient. No tests/render had begun at the saved snapshot, and the batched GDN12 tests are outside this queue. [Exact scripts and startup states](../experiments/m5-ultra-256gb/queued-continuations-v1/README.md). Current v11 stays unchanged until visual review; failures remain inspectable and do not create speed points.


## E024 paired formal closed; original parser failure retained · 2026-10-17

All12 registered fresh servers completed96warm+768formal; original43565 restored73594 and independentlyPASS. First frozen formal audit rejected a legitimate registered HF tokenizer symlink before rawHTTP, so automaticv12 correctly producedNO_NEW_SPEED_CHART. OriginalFAIL/3,733-member archive preserved and pushed688b68c. Separatev2 parser retains genericcanonical receipts, validates exact snapshot/model-manifest/targethash+stat and repeats target check after audit. Actual8 focusedCPU regressionsPASS; every original count/timing/resource function remains unchanged.

The complete same864 raw requests then independentlyPASS with1,267resource samples and zero swapgrowth. C1 OFF/ON process medians45.405789/45.835028tok/s,p50 11.270521/11.165923s; C8 167.536857/170.619844tok/s,p50 24.418582/23.976082s. Six adjacentpairs improve throughput0.847–3.807% with lowerp50/p95. Threeprocesses/configuration/C, no pooledpercentiles/CI, no all-workload victory. The separate corrected archive retains3,843members/38,203,144xzB SHA63e211a8eecaccd7f41e7b690e33fc01964d164c7c45ff5f82d4b3fc2575f7db. [All pair metrics, correction and limits](../experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1/RESULTS.md).

## Completed next-module CPU checks · 2026-10-17

GDN HTTP16 actuallyPASS with no backend/model/network operations; [full239-member CPU archive](../experiments/m5-ultra-256gb/gdn-http-cpu-v1/README.md). BatchedGDN12 firstattemptPASS with14unchanged source snapshots; [full33-member CPU archive](../experiments/m5-ultra-256gb/gdn-batched-cpu-v1/README.md). These do not establish actual HTTP or batched device qualification.

GLM first33-test run had32PASS+1fixtureERROR: the preintent test omitted the external registered plan while testing absence of the owner-local copy. Only thatfixture was repaired; realregistration/hash and corrupt-external-plan rejection remain. New complete33-test runPASS, production sourcesunchanged; [both complete runs and repair tutorial](../experiments/m5-ultra-256gb/glm-cycle-prepared-cpu-v2/README.md). No actualGLM input preparation/service action/model load occurred.


## v13 actual rendering and layout repair · 2026-10-17

Predeclaredv12 policy retained:14historical+12newprocess points,10groups,6pairs. v13preparePASS; first renderFAIL missingMatplotlib inroot.venv, identical source/data rendered in existingplotvenv. Root+independentvisualreview found x-axis/tableheading overlap. Separate render_v2 moves only subplotbottom.585→.615 and output/provenance names. FinalPNG viewedPASS/SVGparsed, chartdataSHAff7d06adbe1ecda482173a6747a7e673aa71a73a54713bfd2d8f2f6b52551ef0 unchanged. Allthree actualcommand attempts, bothimages/sources/receipts and failedreview retained. Currentroot/guide links [v13](progress/2026-10-17-v13/README.md). No new timing samples collected.


## E030–E033：组合资格与正式计时完成 · 更新2026-10-19

[E030](../experiments/m5-ultra-256gb/E030-gdn-graph-combined-serving-v1/README.md)
组合GDN+E024完成原42参数诊断与37请求同调度比较；独立数值、资源、恢复通过。
[E031](../experiments/m5-ultra-256gb/E031-gdn-graph-combined-http-v1/README.md)
随后完成实际C1/C8的24请求、捕获后独立历史重放，四子进程回收与恢复通过。
这两轮只有数值/HTTP资格，插桩耗时未进入Pareto。

[E032](../experiments/m5-ultra-256gb/E032-gdn-graph-combined-formal-v1/README.md)
正式driver误传owner不支持的--run，真实CLI以2退出，模型子进程和请求均为0；
原失败代码、命令与正常恢复证据保留。新的retry-v2在停止服务前执行真实
prepare-only入口，19项focused CPU通过，源与计划在启动前提交。

实际E033 retry-v2于20:06:57 UTC启动owner93089，现已完成12进程、96warm+768formal。
同一combined server、graph恒ON、GDN OFF/ON、每C1/C8三对fresh process；
正式计时/资源与恢复审计PASS，服务91373→21938，四阶段退出码全0。
C1中位数45.81897→48.35777tok/s；C8 170.46064→170.37898，无稳定增量收益。
[3472成员完整归档](../experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2/README.md)
和031教程保留所有配对。C1仍低于oMLX约4.77%，尚未实现逐batch领先目标。

[首轮C1时间戳诊断](../experiments/m5-ultra-256gb/combined-v2-r1-c1-diagnostic-v1/README.md)
保留三组各64请求和首次计算错误。当前ON比历史oMLX平均E2E慢0.51371s，
其中post-first-output为0.43960s；这是HTTP边界分解，不能归因为GPU或ITL，
也不是同时配对外部对照。新默认OFF候选与对应未执行测试源、最新论文和M4同步
均已提交df21dab。完整正式运行结束后才执行新候选检查、绘图与下一轮设备工作。


## 下一批候选CPU首跑 · 2026-10-19

正式运行完全闭环后执行155项fake/小数组CPU检查。v1因runner未使用package名称导入
相对import的测试模块而在执行任何测试前失败；修正runner、保留v1后，v2实际执行155项，
154通过，GLM注册负例因期待ValueError而实际得到Refuse产生1error。
生产代码正确拒绝尚未封存的前序证据；修复测试期待并将前序状态改成确定性fixture后单独重跑。
这些CPU检查没有加载模型、使用GPU或赋予任何新性能资格；两次尝试与源码快照完整保留。


## E034：GLM加载v3实际FAIL，服务恢复PASS · 2026-10-19

实际计划dba39963057c9db9d8c77f40afbdae3df9d0b175b69812c9262632752cea9023先提交8c3a640再启动。
stop0/owner1/recover0，状态RESTORED，服务21938→34623。物理样本0031触发runtime swap growth，
系统swap比冷基线增加242,871,173B，Swapouts增加14,880；子进程182.45 GiB、系统203.92 GiB。
底层官方加载线程在取消清理时返回，EnginePool await未正常返回，0请求、0tokens。
独立恢复审计PASS/诊断审计FAIL，未得到GLM数值或性能资格。
[032教程](tutorials/032-glm-load-and-memory.md)记录假设、原门槛、实际边界和下一步。
首次离线审计wrapper因analysis目录已存在但为空而在启动审计前失败；修正目录创建后两个真实审计执行。
首次归档程序拒绝模型目录symlink，未跟随或打包权重；保留失败后显式记录该已注册别名再归档。


## E035: actual C1 diagnostic and restoration PASS - 2026-10-19

Three fresh off/runner/full processes, each8warm+8diagnostic:48requests/24576outputtokens. All natural exits/reaps; stop0/owner0/recover0, service34623 restored39988; independent diagnostic/restorationPASS. Archive1084members/96,896,227rawB/6,629,976xzB, SHA2f1b1b8abc64ac822c295ce588b64d69bd798ac0183d694d0936ce0b03d2b763. Both analyses and first static rendering succeeded; root visual reviewPASS. No new Pareto point.

4072 duplicate-current-root async calls total18.523ms(0.02191%); successor calls average18.620ms. Visible long-wrapper interval0.4609%; delivery99.831% overlaps worker. Prioritize actual model/backend and multirow weight reuse, not removal of small wrapper costs. These are Python wall intervals, never exclusive GPU time. [Tutorial033](tutorials/033-profile-before-large-optimization.md) retains the method and limits.


## E036：GLM 离线 expert 重排实际完成 · 2026-10-19

计划先提交096a7af；真实执行502.305588s，退出0，PASS_BYTES_ONLY。
完整原始分片SHA校验、分组复制SHA和输出独立回读均通过。
113446→4960张量，378个backbone stacks，58个输出分片；payload保持203976457080B。
原checkpoint保留，MTP原字节和官方处理路径保留。无数值转换。
[E036 metadata归档](../experiments/m5-ultra-256gb/E036-glm-expert-repack-v1/README.md)
区分运行时完整payload校验与事后独立header/index/manifest核验，Git不打包模型权重。
未执行实际loader、整模型正确性或性能；不宣称解决了E034的内存问题。

## E037：NVFP4 多行权重复用真实 GPU smoke · 2026-10-19

源码和运行器先提交2dd1e44；实际子进程退出0，36/36固定用例PASS_SYNTHETIC_BYTES_ONLY。
T2/4/8、三个矩阵形状、连续/stride2，小形状另含正负零和较大有限值；
三路完整BF16输出逐字节相同且有限：candidate、独立原生M1、现有customB1。
MLX观测peak71516160B。原oMLX已恢复、离线GLM复制已结束；本轮无速度计时结论。
[E037原始输入/输出/执行与独立审计](../experiments/m5-ultra-256gb/E037-nvfp4-multirow-device-v1/README.md)。
候选仍默认OFF、107行；尚未做实际整模型prefix/state、接受/拒绝、回滚或HTTP资格。
[034教程](tutorials/034-real-weight-reuse-and-glm-layout.md)记录两个真实结果和后续范围。
正式Pareto保持v16，没有把正确性通过画成加速测点。

E037 independent offline audit v1 exited1 because it wrongly required empty stderr; the actual104B stderr was only the known mx.metal.device_info deprecation warning. The failed source/commands/logs/results are retained. With only that exact warning accepted, v2 exited0:36cases/108BF16 matrices,559104pairwise word comparisons all finite and byte-identical. No device rerun or numeric gate relaxation. E036 independent header/index/manifest audit also PASS;164metadata members verified.


## E038：真实投影筛选数值通过，第一版多行 kernel 因速度拒绝 · 2026-10-19

实际使用 head K5120/N248320、layer0 up K5120/N17408 权重，合成 BF16 输入，T2/4/8。
六组逐行原生 M1/custom B1/candidate 全输出字节相同且有限，2,400 原始计时完整。
十二个 median baseline/candidate 比值全部低于1；head 对原生为
0.931513/0.922243/0.820917，up 为0.931196/0.857143/0.801152。
默认OFF候选因速度拒绝，不自动执行旧整模型资格。原服务39988暂停后同身份恢复。
[独立审计/图/原始证据](../experiments/m5-ultra-256gb/E038-real-projection-screen-v1/README.md)
保留70成员，权重payload排除。首次离线reader因null metadata格式假设失败和首张图
排版修复均保留；不是设备数值失败。无新引擎Pareto点。

## E039：58分片重排GLM实际加载失败，恢复通过 · 2026-10-19

真实E036派生checkpoint进入官方loader；物理0018触发不变的swap增长门槛：
全系统+1,247,805B（约1.19MiB），Swapouts+76，pressure1。
子进程148,261,492,112B、system171,148,001,280B，均低于观测门槛，但不能据此认定fit。
0完成请求/0token batch事件，无loader-after、loaded-engine、complete或协作shutdown收据。
子进程最终-9且严格回收；stop0/owner1/recover0，RESTORED，服务恢复52818。
独立恢复PASS、诊断FAIL，不能将E036字节通过或恢复通过升级为加载/性能通过。
[860成员实际证据](../experiments/m5-ultra-256gb/E039-stacked-glm-load-v1/README.md)
归档SHA ff464cfdfe7df2def16148d0fb229c8c08cb441cd6db45597a4922c7261985f3。
原始计数器是全系统观察，不证明具体换页原因；较低的提前中止footprint不是内存收益。

## E040：shared tile 保留B1算术，head T4/T8有局部收益 · 2026-10-19

新默认OFF候选将两个输出列的packed权重/scale装入5,760B shared tile，token SIMD
并行，各自解码/累加。K5120、T2/4/8、BF16专用，无生产安装。
六case/18完整矩阵三路finite且exact；独立复算2,400timings/60group ratios。
Head T4/T8对native为1.169216/1.258057，对custom为1.095522/1.136031；
两个T下所有五组均赢两个基线。Head T2为0.991231/0.928371；up T2/T4/T8
对native为0.910122/1.002304/0.995709，对custom为0.985041/1.022069/1.073009，仍mixed。
每组20对AB/BA，ratio为组内各自延迟median之比，再取五组median；不是独立进程置信区间。
服务52818同身份T→R恢复、health通过，所存资源pressure1且swap/Swapouts不增长。
[图和78成员独立归档](../experiments/m5-ultra-256gb/E040-shared-projection-screen-v1/README.md)
首轮audit/render均PASS，tar SHA57800d10708e247bba3dac8d6f4635644fede945b22275f8e2a2a3f6d106fcb7。
这只是实际权重+合成hidden的局部投影筛选，不是整模型、HTTP或engine Pareto。

[035教程](tutorials/035-shared-weight-screen-and-load-boundary.md)串联上述三轮实际结果。
正式v16的50观察/18组与三引擎前沿完全不变；后续资格进度另记如下。

## E041：tiny CPU file-like I/O实际3/3字节通过 · 2026-10-19

实际MLX0.32.3/CPU，276B fixture（84B payload，U32[9]/F16[11]/BF16[13]）依次通过
filename、cached-file-like、F_NOCACHE-file-like三条路径；默认描述命令与唯一实际run均退出0。
同一fd在首个I/O前fcntl48返回0；两个file-like均关闭fd后CPU eval/save成功、关闭后I/O事件0。
输入和三份输出原dtype/shape/字节独立复核相同，包含正负零、subnormal与有限极值。
[实际证据](../experiments/m5-ultra-256gb/E041-glm-filelike-io-smoke-v1/README.md)已提交327441c。
只取得小型API/字节/生命周期资格，不代表大分片内存、swap、官方loader或性能通过。
后续cached-file-like与uncached-file-like对照才能隔离cache假设；filename的lazy时机不同。

## E042：shared head固定prefix与独立字节审计实际PASS · 2026-10-19

父head-prereg-v2计划SHA55dbd68bb64a80b3091e629a20b6915a19d3ab92fb92c0b29417700aaefa6b7a，
实际启动工具e2753e/session60473→b38bd6退出0，producer九case/42prefix/九head调用PASS，
58.171206s；47个子进程物理样本、lease闭环、原服务52818同身份恢复。
独立字节审计c20a5f/session53351→09357a退出0，40.454174s；全部九case/42prefix、
5418输出tensorpairs、96原始文件、九head调用与1349参数值收据核对通过。
父v1仅CPU准备未运行，保留因子进程回收/环境审查修正的源；模型和子计划冻结。
只有forward_prefix_qualified=true；full_model/compiled/rollback/HTTP/performance仍false，
不代表接受/回滚协议或端到端收益。
[474成员metadata归档](../experiments/m5-ultra-256gb/E042-shared-head-prefix-v1/README.md)首跑PASS，
tar SHA1f1cc198e8632886c8f97a17295bae73fd06900ca69da3381f6c4705d535efc9。
96份24,476,249,870B原始快照留本地，以path/size/已被独立reader验证的SHA引用；
归档本身只回读metadata，不重复22.8GiB扫描。全部原始attempt/父v1准备保留；正式v16仍不变。


## E043：native M8 shared wide局部数值通过，速度明确拒绝 · 2026-10-19

计划先提交cc3b2bb，SHA9f0dbea058ee647c0ccf16eb1f9e3badf86a06686e9385ca4ae81d81130ac2ed。
实际device/owner退出0；两T8 case真实head/up权重、合成BF16输入，baseline为一次native M8，
不是八次M1。四矩阵2,125,824 pairwise原word全部finite/equal，独立复算400timings/10组ratio。
Head native/candidate=0.5458300112，up=0.6937056746，全部组<1，因此拒绝速度采用。
五child物理观察、六indexed资源样本+after，pressure1/swap无增长，原52818同身份T→S恢复。
[92成员归档](../experiments/m5-ultra-256gb/E043-native-m8-shared-screen-v1/README.md)首轮audit/archive均PASS，
tar SHA635604cff085caa99c8866f1c211db531c87f5d0ff2dc917015d970d39d72ecb。
765,296,968B真实权重slice排除，全部实际输入输出/计时保留；CPU8与11两次PASS准备保留。
新candidate默认OFF，无full-prefix/engine/HTTP资格；E040不同oracle，不能当同基线回退。
热重复局部访问与整模型缓存驻留可能不同，无cache容量/硬件计数器归因。v16完全不变。


## E044：native nv8 register局部仍慢，原父FAIL/后续恢复单独保留 · 2026-10-19

两T8真实head/up投影与native M8完整BF16 exact/finite；独立400timings、十组ratio均复算通过。
Head native/candidate=0.7647025469、up=0.8177880323，全部组<1，拒绝采用。
数值child57173 waited=true/0；原父却因自然退出后的identity检查竞态FAIL，
随后lease57165恢复被中断，原lease−2/closedfalse，stderr KeyboardInterrupt15保留。
Watchdog同身份T→S恢复52818；fresh只读reconciliation PASS_RESTORATION_ONLY，
三个health端点与57173/57165不存在核对，SHAbee0a1ca402151f9f30dd975e8fe194efe64aae4715c71655f0cc90c80607d0a。
原parent FAIL不变，数值/计时PASS不升级为完整性能闭环。四child物理/五indexed资源+after保留。
[108成员原始证据与E043/E044合图](../experiments/m5-ultra-256gb/E044-native-m8-register-screen-v1/README.md)，
tar SHA f1400de1d2f0fa94c012fe98ae7cdf964e94c97c5fa679e7d7f19157d6e43982；真实模型slice排除。
父v2清理修复仅CPU5/7 PASS，无新设备周期。首次archive在audit前因未来测试5→7的
current-source SHA差异失败，源码/命令/日志完整保留后改用各自历史快照，第二次PASS。
合图20组点和四medians不跨轮配对，明确E044原父FAIL；作者与root均实际看图PASS。
正式v16完全不变，无新engine Pareto、HTTP或整模型速度点。


## E045：按C1/C8独立选择提交配置，预注册 · 2026-10-19

[执行器与方案](../../bench/m5_ultra_256gb/concurrency_config_v1/README.md)仅扫描
MLX_MAX_MB_PER_BUFFER=16/50/200/800，其他OPS50/BFS20/FAST_SYNCH0固定。
C1使用E033 GDN+graph，C8使用同入口graph-only；每C均有前后50基线。
每新进程8 warm+8 admission_probe，完整raw SSE与输出IDs留存；不称formal/Pareto。
候选需两端漂移均≤2%、吞吐与p50均对两端改善≥3%才能进入确认。
旧环境数值资格不继承，新环境仍需完整logit/cache数值与长配对验证。
本段为运行前计划；实际结果在闭环和独立审计完成后追加，包括失败。


## E045：C1配置收益不足门槛，C8基准自身返回 IDs 不一致 · 2026-10-19

v2真实运行退出0、PASS_CAPTURE_AND_RESTORE，十个新服务器共160请求、每次512 tokens。
独立raw审计和第二份stdlib reader均复算20组；C1所有返回IDs相同，MB200最好49.140388tok/s、
p50 10.419658s，对两端最小吞吐收益1.5002%、p50速度1.4787%，未到预设3%。
C8默认MB50 warm/probe已有4/8 fixture IDs不同，所有配置不合格；计时首尾漂移约0.42%仍通过，
不能混淆计时稳定和token一致。MB800原始171.965865tok/s不升级为优化收益。
输出原因未定位，不能归因MB或断言engine bug；先记录真实scheduler trace再做同schedule重放。
400物理/400metadata观察，最大系统183.895GiB、child current22.374GiB、swap增长0，
服务52818同身份恢复，lease闭环；独立saved-observation审计PASS，不是未采样峰值保证。
[完整1805成员证据/图](../experiments/m5-ultra-256gb/E045-concurrency-config-screen-v2/README.md)
全部回读size/SHA PASS，tar SHA939b8f187eed63f94e91328fe9535ff8d0f95a772ae76242a7a12bcde36b23dc。
v1 duplicate-label FAIL、首CPU失败和首图布局失败全部保留；正式v16不变。
[036教程](tutorials/036-concurrency-specific-configurations.md)记录按C独立选择的办法与失败。

## E046：B2/B8 GDN原始字节canary准备完成 · 2026-10-19

仅本段准备状态：新owner CPU实际12/12 PASS、现场source/服务检查PASS；未运行GPU。
父plan SHA08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b，
child plan SHA7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003。
保留native batched projections，仅融合prework；四条B2/B8早晚链、36calls、原始输出/conv/S，
graph OFF。它是独立局部数值门槛，不解决E045 schedule变化、不构成HTTP/速度资格。
执行后追加真实结果，原准备记录保留。


## E046：B2/B8真实激活原字节canary与独立审计实际PASS · 2026-10-19

已提交执行计划40eeb84后运行，tool2a413a/session16488→deea7d exit0，
outer PASS_CAPTURE_AND_RESTORE、39.526630s；producer34.993563s仅数值资格时长。
四个B2/B8早晚reference chains、三层×三步，共36candidate calls/零fallback全部通过。
独立audit edb9ec/session12053→acfa34 exit0，140原始文件、108输出/state pairs、
48continuing-state pairs、80greedy rows、324device comparisons全部PASS。
原文件1,777,163,588B留本机；不把局部canary提升为整模型、HTTP、compiled或速度资格。
13physical/13metadata独立saved-observation审计PASS；最大系统201556377600B、
child current28110909200B、aux peak28744347504B，swap增长0/Swapouts807478，
最后done→resource→ack→wait/reap、原52818同身份恢复和health闭环通过。
[402成员metadata归档](../experiments/m5-ultra-256gb/E046-gdn-batched-device-v1/README.md)
全部回读size/SHA PASS，tar SHAec41ba39432f7a0ae3850eddd9f3b43fb9353dce930b712d58d661cf6e806a95。
原始pending字段不改写，后续独立审计单独记录；首次device和audit即通过，无重跑。


## E047：C8默认MB50两轮调度观察实际完成 · 2026-10-19

计划先commit+push0bcb086；父plan6d31b9f9efbdb453b02675c3d0fcba666f49b48b3e1b8fc7aca8189f42fae9b7。
9项CPU通过后实际8849da/session19491→705ce4 exit0、54.379622s、PASS_CAPTURE_AND_RESTORE。
同PID52818恢复。两轮各八请求完整raw及scheduler trace已保存；闭环后join d2fb75 exit0。
这是observer ON的诊断，会改变调度时机；不加入性能/Pareto。精确trace分析和原生同schedule
重放尚未完成，不能根据输出差异直接断言实现错误。

## E048：真实B2/B8 GDN局部计时准备 · 2026-10-19

从E046已审计real captures选12点：B2/B8、early/late、layer0/32/62。保留native batched
五个Linear；每臂独立seed，先完整output/conv/S字节比较，再5warmup+5组×10pairs AB/BA。
共1200 measured durations，完整forward+output/state eval+sync，reset/load/hash/I/O在计时外。
owner13CPU及screen6CPU均实际通过；现场metadata/evidence预检查通过。尚未设备执行。
父plan8c1fdfe2bcaa5d45e49b87120650949b879eef616ec1aaf881e49a00a381386a。
只能判断local warm-resident层成本，不把微测速度或producer时长变成引擎吞吐。


## E048：真实激活局部GDN层计时独立PASS · 2026-10-19

计划先commit+pushb85d6b1；e7e79b/session65941→af7b46exit0，owner15.352015s且原52818恢复。
12case/1200measured/120warmups、672candidatecalls/fallback0；所有完整原输出/state数值通过。
独立audit003e46/session91399→0bc50bexit0，48rawfiles、72bytepairs、60groups全部重算；
report SHA22141e12f3eda94f762ec77fbbe51d87c825853b62152510dee3985055b8ba1a。
B2六case median1.071695–1.089616；B8六case1.047935–1.066748。全部12median>1，
B8earlylayer32 group0=.880700仍保留，无删点。仅local eager/warm-resident层成本，不是HTTP或Pareto。
[全部组图及证据](../experiments/m5-ultra-256gb/E048-gdn-batched-local-timing-v1/README.md)。
6资源观察与原服务恢复独立saved-audit通过；首图轴标题裁边FAIL与layout2双方viewPASS分别保留。

## E049：warm49 faithful prefix准备 · 2026-10-19

原E047recording/caps不改，eager和原graph同实际schedule到49步，明确INCOMPLETE_BY_DESIGN。
7项prefix CPU PASS（首轮toy期望错误FAIL保留），13项owner CPU PASS，现场prepare PASS。
父plan eb89c92e329f7bc1e5629ae05ac911457ed86bd656a92cc1fed9ac63be6099a5，
child71f2891f24045f539a195711e8cdc97ea190bedd214e42173ab006b7eb700f1c。
64GiB child/240GiBsystem/no swap原gate不改，实际prefix峰值未知。尚未设备运行。

### E049 actual / E050 prepared: C8 batched GDN composition

E049 actual device execution d8e806/session53359 → 7e5ce0 returned 0, PASS_CAPTURE_AND_RESTORE (51.460662s diagnostic lifecycle, not engine latency), original oMLX PID52818 restored. Independent ledger audit 06dc48 PASS and resource audit 1a6652 PASS; 49 steps/380 IDs/65,005 bit receipts/130,010 finite receipts. E049 remains INCOMPLETE_BY_DESIGN: 0 completed, 8 active, 8 future. Archive 262 members, SHA256 `2c854df325a00847883cdca0bf79c1432bb672754077983811e1839a7faa7967`.

E050 adapter10/10 CPU PASS (687d78) and owner13/13 CPU PASS (cf7efb). Prepare86df9c: parent `14b61d284a088e7c799fb91ee11d45dce8379c79d5e2a071bcb4f40da5072fcc`, child `b939013497400d24ef4d5a26d24142fddc424d6f611b09a9f63a07995caa4155`. Same warm49 schedule, native oracle versus actual batched GDN+graph. Scope remains numerical prefix only; all raw failures retained. Runtime not yet executed at this commit.
