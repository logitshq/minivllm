# 33. 换到ICB边界后，R8和整步编译是否更快

第32章的V69把每SIMD独立行数从4增至8，数值通过，CPU逐次调用/eval微基准只有描述性的1.046412%改善，未过预声明5%门槛。本章没有重判这个STOP，而是用新V72检查另一条执行边界；同时记录V74把完整纯decode先编译、再捕获进native ICB，以及V77的B-only screen。

结果均没有形成采用结论：V72新边界下R8反而慢约0.5166%，0/6胜；V74通过限定工作负载的数值资格，V77没有公平配对或HTTP胜出证据。五次运行和原始失败见[provenance](33-native-range-compiled-step-source/provenance.json)、[证据索引](33-native-range-compiled-step-source/evidence-index.json)与[独立复核](33-native-range-compiled-step-source/reviews/v72-v74-v77-results-review-build1/REVIEW.md)。

## V72：一次CPU调用，执行一组静态ICB范围

V69的两臂虽都已 `mx.compile`，计时仍逐次经过Python API/eval。V72改测静态range循环：八组真实MLP gate权重来自layers0、8、16、24、32、40、48、56，各自使用保存的真实输入X。权重与scale共401,080,320字节，没有concat、repack或新布局。

```text
一次capture：A0 A1 … A7 | B0 B1 … B7
一个16-command ICB：range A=[0,8)，range B=[8,16)
一次ffi/一个CB：repeat { execute选中range8；Buffers barrier }
```

A仍是compiled原生QMM/R4，B是冻结V69的R8/SG2。两个arm使用同一bank的原始W/scales/X。每个arm有自己的八个输出slot，共16个原始输出owner；没有跨arm输出复用。源扩展include冻结native40，保留旧API，不启用fullmodel feedback/S ping-pong。代码见[V72 build2](33-native-range-compiled-step-source/prototypes/v72-native-range-icb-build2/replay.mm)。

### build1的拒绝必须保留

build1在 `v72_build_ranges` 拒绝了 `active readwrite/unknown access`。保存的72项MLX比较已完成，但native loop为0，整个run是FAIL_STOP，不能算native资格通过。

实际反射中，A函数 `nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0` 的arg4 `y`、B冻结typed函数的arg3 `output` 都是RW。build2没有把所有RW放开：只对这两个精确function/index/name加入冻结源码的store-only输出推断。installed原实现与冻结69 header的函数体一致，y只偏移指针并被赋值，不读取旧输出；B的output只传给同一实现。

**反射仍是RW，store-only是源码推断。** [策略说明](33-native-range-compiled-step-source/prototypes/v72-native-range-icb-build2/output-access-policy.json)和[source proof](33-native-range-compiled-step-source/prototypes/v72-native-range-icb-build2/source-write-proof.json)保留这个区别。其他RW/unknown仍拒绝；输出必须是原始allocation基址、offset0、整块allocation独占，且与任何输入或其他输出allocation不相交。maxBind覆盖全部实际记录/编码bindings，不猜测unused global-scale/batch参数。

build1原文件、FAIL receipt及释放证据原样保存。build2的计时protocol字节未改；旧protocol里全拒绝RW的准备字段只由新的窄策略覆盖，不能把旧文字当成实际反射WriteOnly证明。

### 数值资格先于新的性能marker

| 层次 | 实际覆盖 | 不包含的声明 |
|---|---:|---|
| MLX资格 | 8banks×3输入(real/zero/fixedrandom)×3比较=72 | 所有输入/所有shape |
| native资格 | 16次完成loop×最终8slot=128个原始输出 | loop中每次dispatch都验值 |
| 性能后检查 | 20个timed block×最终8slot=160 | 每个8192-dispatch block逐次验值 |

native资格的group次数为1、2、3、4、5、7、8、16，每arm各一次。原始输出与16份reference字节严格一致，且reference匹配历史对应层真实gate哈希；40个原输入和16个捕获输出owner的readonly是保存的运行时证据。进程退出后没有重新构造这些owner。

实际映射有24个readonly输入allocation、16个独占输出allocation、48次W/scales/X解析，maxBind7；zero-copy共401,162,240字节，其中81,920字节为八个X。arena/snapshot为786,432字节，即16×49,152字节物理输出allocation；每个逻辑输出34,816字节。arena只在首次build恢复一次，没有每block重拷400MB权重。

每组8个dispatch后都显式插入Buffers barrier，包括最后一组，以控制重复写slot的WAW。该屏障与完成计数证明不等于逐次值观测：只在loop完成后读取最终八个结果。

## V72性能：controls通过，R8仍未过门槛

数值run退出/reap/release后，新的性能marker明确绑定该成功receipt SHA，再做相同资格和固定协议：prime128groups、timed1024groups，每timed block编码8192dispatches；六个AB/BA比较及四个AA/BB控制，没有重试或调参。

| 项目 | V72 build2实际结果 |
|---|---:|
| A臂中位/每dispatch | 100.570989 µs |
| B臂中位/每dispatch | 101.090497 µs |
| `1−B中位/A中位` | −0.516559% |
| 比较对获胜 | 0/6 |
| controls | 每次5%、每臂中位2%均PASS |
| 预声明收益门槛 | 至少5%降低且至少5/6胜 |
| 结果 | STOP_MICRO_GATE，未采用 |

计时是一次ffi加native编码/commit/wait的wall，再除8192；没有每kernel Python/eval。20份timed报告均有完成CB、8192encoded、1024barriers和有效的官方GPUStart/End秒值。总计56个完成loop、23,132groups、185,056encodeddispatches；其中还包含16个数值loop和20个prime loop。GPU时间是整个CB envelope，不是各kernel独占时间之和。

不能把本表与V69逐次调用边界的绝对数值直接相减成收益，也不能用这次STOP覆盖或修正旧STOP。这里仅排查“Python逐次边界是否掩盖R8”的假设；新边界仍未达到采用条件。

## V74：whole pure step先compile，再捕获native ICB

V74保持candidate44、native40、数学与gates/conv/normreuse来源不变，在外部建立纯函数：tokens/positions/lens加128个state为131个显式输入，返回logits、128state、argmax prediction、next position/lens，共132项。

attention用独立 `BatchKV(K[:],V[:])` 句柄，GDN直接持有显式旧array再整体返回新state。closure只固定当前模型、类型和max_len256，不捕获某个请求的token或cache。factory见[functional_step.py](33-native-range-compiled-step-source/prototypes/v74-compiled-step-icb-qual-build1/functional_step.py)。

```text
pure131→132 -> mx.compile(shapeless=False) -> cold eval/132旧A哈希/seed readonly
                                              ↓
                         warmed compiled_fn(*seed) -> native capture -> ICB replay
```

cold只有一次trace、48次GDN Python调用。之后warm capture的trace增量0、GDN Python调用0；hook移除后仍为同一个compiled图。这里没有要求compiled hit重复执行Python hooks，也没有把 `mx.compile` 本身称为ICB。

| V74实际资格 | 结果 |
|---|---|
| 单次cold | 132项shape/dtype/finite/raw SHA匹配旧44 A，131seed readonly |
| 三请求(base/reset/changed) | 93×132=12,276项严格输出，全部32IDs一致 |
| 参考A | 原accepted25 GDN，逐项SHA又对旧38/44 A |
| capture | 1498dispatches，旧S PSO/S6与S_new9，48eligible/83staged |
| 完成 | 93steps、3reset、ICB banks48/45，参数owner不换 |

1498是实际capture调度条数，不是硬件指令数或加速比例。首次cold与整模资格都保留严格字节门槛，不能用容差或自参考掩盖编译规约变化。

### 更正oMLX的实际调用链

此方案是独立探索，**不是从被测oMLX移植whole-generation-step compile**。实际安装版本为omlx0.7.1.dev1与mlx-lm0.31.4.dev132+g94cdcae13；wheel没有可声明的被测oMLX VCS commit。

所读 `mlx_lm/generate.py:1355–1367` 的 `GenerationBatch._step` 没有compile decorator，直接调用 `self.model`。`omlx/scheduler.py:773–798` 的wrapper处理mRoPE/rows/grammar后调用原step，也没有整步compile。gated_delta里的compiled leaf不等于整代生成step，也不能仅凭源码存在推断某个leaf实际engaged。原始安装路径/行号/SHA保留在[SOURCE-NOTES](33-native-range-compiled-step-source/prototypes/v74-compiled-step-icb-qual-build1/SOURCE-NOTES.md)和[source-map](33-native-range-compiled-step-source/source-map.json)。

## V77：raw-native B-only screen

V77绑定实际V74资格与释放记录，复用同一factory/native40/candidate44。先做普通32 warm和独立cold资格，再做三fresh warm、三fresh measure。每次请求末尾132项对旧A，六次共792；263原始capture owner readonly在区间外检查。

```text
t0 -> fresh prefill + 首token/flatseed准备 + reset/131seed上传 -> t1
       native31步 + 每步4B prediction观察                         -> t2
```

| 三次measure中位 | V77实际 |
|---|---:|
| `(t2−t1)/31` | 34.620622 ms/步 |
| `31/(t2−t1)` | 28.884518 token/s |
| `32/(t2−t0)` | 25.520283 token/s |

timed decode没有A/model calls、整state CPU复制/hash、资源或写文件；186步、6reset、banks96/90，compiletrace仍为1。这里t1在131seed上传**之后**，不同于67/71的engine时钟包含native begin/upload。它不是HTTP服务指标，没有平衡配对A；whole32与decode31都没有证明公平胜出、Pareto或采用。

## 归档与CPU自验证

本章只收V72build1FAIL/build2numeric+STOPperf、V74fullqual、V77screen及对应独立review。V76/V78/V79等后续结果不纳入。五个owner均已wait/reap，外部记录严格PID absent；82份保存的child资源记录为AC/thermal0/pressure1，swap-used和Swapouts稳定。它们是保存证据，不是新的现场探测。

[归档索引](33-native-range-compiled-step-source/README.md)包含确定性文本tar、原路径/长度/SHA、跨24–32章精确副本及外部二进制/fixture身份。失败和历史CPU_PENDING文字原字节保留，以实际receipt为结果；MLX/MLX-LM MIT与oMLX/Lithos Apache全文分别保留。归档不包含权重、tensor或native/runtime二进制，也不是可重定位执行的模型包。

离机命令只验证文本/哈希/统计与旧gold记录；tensor字节复核必须保留本机原始fixtures，使用 `--originals`。两者都不导入MLX/native，不执行模型/GPU，不授权重跑性能：

```sh
python3 -B docs/experiments/33-native-range-compiled-step-source/verify_archive.py
python3 -B docs/experiments/33-native-range-compiled-step-source/verify_archive.py --originals
```

结果见[verification](33-native-range-compiled-step-source/verification.json)、[manifest](33-native-range-compiled-step-source/manifest.json)与[terminal-fixture-index](33-native-range-compiled-step-source/terminal-fixture-index.json)。
