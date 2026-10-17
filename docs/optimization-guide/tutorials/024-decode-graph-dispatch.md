# E024 · 图编译优化必须保持归约路径

2026-10-17。已完成一次失败候选、一次四组诊断、新两档候选首点验证及52步数值矩阵；
矩阵与独立审计通过；随后完成8配置的隔离前向计时，**没有新增HTTP Pareto点**。
目标仍是正确性通过后的实际 output throughput ↑ /完整 POST→DONE latency ↓。

## 为什么整段图编译值得做

长 decode 会反复提交大量小操作。把稳定的前向过程编译成可复用图，有机会减少
Python 和 GPU 提交开销，但前提是模型参数、请求缓存与数值语义都保持正确。
我们把候选做成默认关闭的独立模块，模型参数作为编译输入，缓存显式输入、输出。

原候选为了复用图，把 attention 长度统一补到1536。它在 B1/used257 首点失败：
两个有限 BF16 logits 中有228,516/248,320元素不同。完整原始失败和恢复证据
[单独保留](../../experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md)，没有放宽容差。

## 先拆开两个变量

旧比较同时改变了“是否编译”和“attention张量长度”。因此失败不能直接归因于编译。
新的诊断只做同一个真实前缀、B1/used257，四组使用完全相同但各自独立的输入缓存，
共享前后所有参数字节哈希不变的同一模型：

| 对照 | 只改变什么 | 独立原始字节结果 |
| --- | --- | --- |
| eager257 → eager1536 | 补齐长度 | 121/129张量不同 |
| eager257 → compiled257 | 是否编译 | logits及128状态全部逐位一致 |
| eager1536 → compiled1536 | 是否编译 | logits及128状态全部逐位一致 |
| eager257 → compiled1536 | 原候选的两个变化 | 121/129张量不同 |

全部输出有限，四组输入前后与canonical逐字节一致。每组完整logits和128个状态
均保存并由独立CPU审计读取。参数哈希及原文件前后检查通过；独立审计没有重新读取
未保存的整套device参数值。四组argmax都是1559，仍不能用这个事实替代逐位门。

**这个点已证实补齐长度本身足以重现差异。** 相同长度编译没有增加差异；结论只覆盖
这一个点，不能推广到所有batch、状态更新或生产服务。

## 数学上被mask，不等于实现上没有影响

[官方MLX0.32.3分派源码](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L1014-L1019)
在这台g17d设备上，K达到1024就切到两遍SDPA。即使新增位置都被mask掉，改变长度
仍可改变有效元素被分给哪个线程、以什么顺序累加。浮点归约顺序属于正确性合同。
我们没有捕获运行时kernel trace，因此源码解释仍与实际四组因果对照分开报告。

下一候选的源码依据更具体：原bool mask会直接跳过无效key的分数、指数与累加。
在257..1023内保持单遍32路归约，在1024..1536内保持两遍128 blocks，有机会只增加
被跳过的循环而保持有效key的归约顺序。拟选择bound1023或1536，而不是全部1536。
[完整源码依据与待验证边界](../../../bench/reviews/2026-10-17-e024-route-preserving-bounds.md)
包含保留的官方shader、许可证与哈希。

这仍是待验证方案。需要B1..8×257/1023/1024/1280/1536共40点，以及跨1024、异长rows、
B1→B8→B1图复用，再过参数替换、ahead生命周期和gates-ON D512/HTTP检查。
两个bound只是两个callable族；不同batch形状仍可能编译出更多机器图。

## 每次尝试如何保留

44项不同CPU检查通过后先提交计划，再临时停止原服务、执行唯一子进程，最后恢复。
此次四组诊断47.76秒，原oMLX已恢复为PID11548，独立恢复审计通过；暖KV不恢复。
这些耗时包含加载、保存与校验，不能作为推理性能点。13个资源样本pressure1、swap增长0。

[完整结果与证据索引](../../experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md)
包含689份小文件归档；13份原始整状态张量约3.31GB保留在本机，精确路径/哈希已提交。
Git归档不包含这些大张量，远端读者需要原文件或重跑才能再做全量字节审计。

[当前D512吞吐—延迟图](../progress/2026-10-17-v6/README.md)仍只有已通过资格的
native C1/C8基线。oMLX/Lithos同负载点尚待测量，不宣称已经超过它们。


## 按归约路径分段：第一个新点通过

我们没有改变权重、mask或浮点容差，只把默认关闭的新模块做成两档：
used257..1023使用bound1023，used1024..1536使用bound1536。物理cache仍要求
至少1536，因此本次没有顺便扩大cache容量的适用范围。

同一B1/used257，原eager257与新compiled1023的完整129个输出全部逐位一致：
每臂89,803,264个原dtype元素，零差异。两个独立输入各131数组也都没有被修改。
实际一次compiled调用及key1023已记录；独立审计直接读7个原始文件约1.783GB。
这支持“在同一归约分段内补齐”的方向，但只证明这个点，尚不能证明整段或速度收益。

运行35.05秒包含加载与验证，不能标为推理耗时。原oMLX已恢复为PID13908，
恢复审计通过，暖KV不恢复；10个资源样本pressure1、swap增长0。
[本次完整证据](../../experiments/m5-ultra-256gb/E024-route-canary-v1/README.md)
保留计划、代码、失败的首次只读prepare命令及成功执行。

下一轮固定40个长度×batch点，再加真实跨1024链、异长B8链及两档B1→B8→B1复用。
实际ahead/retire、参数替换及生产gates-ON D512/HTTP仍要另外过门。图编译减少
提交开销是待测假设，额外被mask的循环也有成本；当前Pareto图不增加这一点。

## 两档矩阵与跨界复用：52步全部通过

实际完成了预注册的40个B1..8×used257/1023/1024/1280/1536点，再加B1的
1023→1024→1025链、异长B8的同一跨界链，以及used1023和1024各一次B1→B8→B1复用。
总计52步、52次实际compiled调用，完整logits加128个状态共6,708次原dtype逐位比较
全部通过；48次独立prefill对照另有6,192项检查。输入与1,349个参数的前后字节哈希门
也通过。候选仍默认关闭，core compile/fuse/gates compile/norm reuse保持OFF。

长history用冻结D任务的prompt IDs重复、截断构造，是诊断延长，不是实际长生成。
异长B8从257/511/767/900/978/1000/1022/1023开始，各臂随后使用自己的greedy输出。
本次加载了完整64层模型，但没有取得完整D512请求生命周期或HTTP资格；两个callable族
也不意味着只有两个机器图。

独立CPU审计完整读取10份原始快照：B1/used257的canonical输入，以及B1/used1023、
B1/used1024和异长B8最长行used1024的canonical输入与两臂输出。三个完整输出对照
共387个张量逐字节零差异。其余49个输出点保留设备逐位比较与完整字节哈希账本，
不能说成所有52点都有离线完整raw复核。成功prefill和参数也遵守这一证据范围。

诊断共222.57秒，包含加载、编译、校验和保存，不能当推理速度。MLX观察峰值53.040GiB
包含两套验证模型、诊断状态与图缓存，不代表生产占用。46个资源样本均为pressure1；
swap读数不变，Swapins/Swapouts/Pageouts增长均为0，Decompressions增加283，
因此也不把它写成所有paging活动完全不变。

原服务PID13908已恢复为18261，独立恢复审计通过，resident PLE与Lightning MTP
可见证据齐全；暖KV和隐藏运行时状态不恢复。[归档说明与独立结果](../../experiments/m5-ultra-256gb/E024-route-matrix-v1/README.md)
包含760份小文件；10份原始张量共7,902,883,133字节保留在本机，Git内有路径与哈希，
远端重新做完整raw审计仍需要这些原文件。

下一步先做B1/B8×used1023/1024的forward-only收益筛选：gates-OFF对照，再要求
gates-ON完整逐位一致，计时段关闭观察器并排除复制、哈希和预热。这个诊断只决定是否
值得继续接入。参数替换、ahead/退场、实际scheduler和HTTP门通过后，才测真实吞吐与
完整POST→DONE延迟；本次52步通过不向Pareto图增加点，也不宣称超过oMLX或Lithos。


## 保持同一数值路径后，隔离前向获得约5–11%收益

52步矩阵通过后，我们按提前提交的计划运行B1/B8×used1023/1024×gates OFF/ON。
每配置4次预热，再运行3组ACCA；每臂6个正式样本，共32预热、96正式样本。
A是原eager，C是两档compiled。全部样本的完整logits与128个状态逐位检查通过，
gates ON两臂也先对照原gates OFF oracle；没有放宽误差门槛。

| Batch | Used length | Gates | Eager median ms | Graph median ms | Forward rate gain |
| --- | --- | --- | --- | --- | --- |
| 1 | 1023 | OFF | 24.874 | 22.475 | 10.67% |
| 1 | 1023 | ON | 23.680 | 22.197 | 6.68% |
| 8 | 1023 | OFF | 40.624 | 38.528 | 5.44% |
| 8 | 1023 | ON | 40.595 | 38.414 | 5.68% |
| 1 | 1024 | OFF | 24.315 | 22.524 | 7.95% |
| 1 | 1024 | ON | 23.638 | 22.089 | 7.01% |
| 8 | 1024 | OFF | 40.469 | 38.367 | 5.48% |
| 8 | 1024 | ON | 40.565 | 38.297 | 5.92% |

上表收益是前向中位时间的比值减一。gates ON时B1为6.68–7.01%，B8为5.68–5.92%。
每组ACCA都有正向差值，但它们来自同一进程，不能代替跨新进程重复或置信区间。
计时仅含forward、全部129个输出求值与同步；复制、哈希、观察器、验证与teardown
都排除。单独记录的forward+teardown也不是请求延迟。这个实验筛选出了值得继续接入
的方向，还不能说明整条生成链、更高并发或HTTP端到端提升。

数值审计读取冻结源码产生的设备逐位比较与完整哈希记录，这次没有另存成功张量raw，
不能说成又做了一轮成功输出的离线完整字节复核。服务已恢复为PID19870，MTP与
resident PLE独立验证通过；21个资源样本pressure1、thermal0、swap增长0。

[完整结果及证据](../../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md) ·
[前向诊断图和可重画源码](../progress/2026-10-17-v7/README.md)。
下一优先级是补齐Lithos同任务D512 C1/C8，再继续参数替换、ahead、scheduler和HTTP门。
原native D512 Pareto两点保持不变；本次不能用来宣称超过Lithos或oMLX。


## Next serving boundary: one existing decode call site

The forward screen does not require a new scheduler. Both normal decode and
ahead call `runner.decode_model`; an initially default-OFF per-instance adapter
can reuse that point, preserving the existing short-context decoder as fallback.
The remaining device gates are changed-value parameter replacement, real
ahead consumption/discard and retirement, then C1/C8 HTTP capture and replay.
Existing ahead keys do not include a weight version, so idle parameter tests
cannot establish safe hot reload while a future is outstanding. Keep weights
fixed through a serving run and record this boundary explicitly.

The tested physical KV capacity is1536. The candidate accepts larger capacity,
but that wider domain still needs evidence before production support is claimed.
The first D512 workload fits1536. Retain bounds1023/1536 so compilation does not
change the attention reduction route already isolated in the failed experiment.

[Source-backed integration plan](../../../bench/reviews/route-graph-serving-plan.md)
records the minimal adapter and missing gates. It is a design, not an implemented
or qualified HTTP optimization; the feature remains OFF.

## 真实调度接入：37请求与参数替换诊断通过

本轮将默认关闭的实例适配器接到原有 `runner.decode_model`，保留原scheduler，
同时覆盖当前decode和ahead调用。候选保持short compile/fuse、GDN gates compile ON，
norm reuse OFF；reference仍为eager/gates OFF，E028 identity refill OFF。没有扩大
物理KV资格范围，本轮实际使用1536容量。

原37请求全部完成：D-C1八请求、D-C8十六请求各为normal-EOS cap512，另有十三个
变长cap生命周期请求，共5,163步、12,411个committed IDs。完整词表logits、64层的
128状态以及ahead检查全部通过。独立审计按实际记录重建FCFS、slot复用、退场与ahead
配对，核对6,622,533项设备数组比较收据，其中logits 10,315、状态6,596,864、
ahead tokens 15,354；不是只核对最终greedy IDs。

| Case | Consumed ahead | Partial retirements | Survivor refills | Bound1023 / bound1536实际调用 |
| --- | ---: | ---: | ---: | ---: |
| D-C1 | 4,080 | 0 | 0 | 723 / 3,365 |
| D-C8 | 1,014 | 7 | 7 | 88 / 938 |
| Lifecycle | 24 | 8 | 5 | 本case走原适用路径；不借用D长图覆盖 |

参数诊断在请求开始前、engine为空时进行。三个真实参数路径——最终norm weight、
第0层GDN `in_proj_a.weight` 与 `scales`——分别在used1023/1024做B1→B8→B1。
42次direct calls由6次baseline、18次changed、18次restored组成；18条收据确认实际
改值确实改变了eager logits，恢复后输出逐位回到原值，另有17,068项数组比较通过。
这证明的是注册的空闲参数替换诊断，不能说成支持请求中途热换权重。

两个未覆盖边界必须保留：所有case的discarded-prior-ahead均为0；normal EOS虽开启，
实际EOS退休也为0。因此不授予取消、sampling-key变化、ahead失效或EOS退休路径资格。
本轮成功张量没有保存完整raw，离线审计检查的是完整设备比较/哈希收据和独立调度重建，
不能写成把所有成功tensor字节又离线重算了一遍。

数值、217个记录资源样本和恢复三份独立审计均PASS。资源为pressure1、thermal0，
swap增长0，Swapouts/Pageouts/Compressions增长0；Swapins+12、Decompressions+6,053。
owner86493/child86527已回收，原oMLX85613恢复为92152，原模型配置与新MTP/resident PLE
证据通过；暖KV和隐藏状态不恢复。[归档与精确范围](../../experiments/m5-ultra-256gb/E024-serving-v1/README.md)
保留935个成员、完整1.628GB原始ledger与独立route记录的压缩证据。

**这次闭合的是固定调度下的完整模型数值资格，没有新增HTTP或性能点。** 下一步让真实
C1/C8 HTTP到达顺序进入新entry，严格回收server后再独立replay，绑定实际图调用与完整
状态；通过后才关observer测吞吐和完整POST→DONE延迟。前向诊断的约6%收益仍不能
直接投射端到端曲线，也不意味着与E028组合或更高batch已通过。
