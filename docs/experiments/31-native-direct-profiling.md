# 31. 直接反馈、采样扰动与局部投影实验

第30章让CPU提前提交下一步，却仍为每步非S输出做完整stage快照。本章先用V57消除同步路径的这段复制，再用V62/V64区分诊断快照和production交付。后续V63的采样只为定位候选方向；V60/V61的局部性能门槛均失败，V66只通过局部数值资格。

V67最终B-only引擎screen为 **28.825675 decode token/s、25.922431 whole32 token/s**；decode区间包含native seed设置。它没有形成HTTP、配对收益或Pareto胜出结论，所有路径仍未采用。冻结来源、实际PASS/FAIL/STOP与生命周期见[归档索引](31-native-direct-profiling-source/README.md)和[provenance](31-native-direct-profiling-source/provenance.json)。

## V57：省去同步路径的完整stage副本

原同步反馈包含两段blit：先复制84个非S输出到stage，再从stage写回83条反馈。V57在两个S bank上检查注册输入/输出range的bounds与交叉别名，允许直接从arena输出写回arena输入；同步command buffer完成后再读取最终arena/current S bank。

```text
原路径：graph -> 全部非S输出stage快照 -> stage到输入反馈 -> host wait
V57：   graph -> arena到输入直接反馈 -> host wait -> 读取已完成输出
```

83条反馈共19,726,348字节。原stage分配20,223,012字节仍保留；不执行这段复制不表示释放了allocation，也不能从字节数推算节省多少毫秒。48个S pair继续换bank，参数、PSO与算术图不变。实现与range证明见 [V57原生实现](31-native-direct-profiling-source/prototypes/v57-native-direct-feedback-build1/replay.mm)。

实际tiny为3请求、15步、75项严格输出检查；27B完整资格为3请求、93步×132=12,276项，并逐项核对旧A哈希。同步B-only screen三个measure中位是34.545681 ms/步、28.947178 decode token/s、25.603767 whole32 token/s；没有公平配对A/B或HTTP比较。

## V62：直接反馈可以复用，输出读取合同必须改变

异步队列里，后继可能覆盖共享arena。把V57的“同步完成后可读全部输出”原样放进depth2，会破坏结果所有权。V62保留直接反馈，并分为两种固定模式，见 [两模式设计](31-native-direct-profiling-source/prototypes/v62-native-direct-pipeline-build1/DESIGN.md)。

| 模式 | 每slot交付存储 | 完整输出何时可读 |
|---|---|---|
| diagnostic | 全部84个非S输出快照；S按last delivered bank读取 | 每步消费期间，下一次refill之前 |
| production | 只有4字节prediction | 仅严格quiescent请求末尾 |

两种模式都保留graphDone、stageDone、feedbackDone和前一票反馈依赖。production每步做prediction snapshot及83条直接反馈，不做20MB完整stage复制；旧stage allocation仍保留。只读4字节预测的生产路径不自动获得诊断模式的每步全状态证据。

`v62_copy_quiescent_output` 要求pending和retained jobs均空、最后交付票是最新票、submitted/completed/delivered一致、该请求无discard/failed且native无错误。持锁复制时，直接反馈的dest与所有graph输出source不重叠。这里证明的是注册range的copy安全，没有猜测shader访问范围。

空drain可保留最后交付结果；带剩余票的drain会丢弃并使读取条件失效。因此早停丢弃后不能把某个共享arena值当作可验证的请求终态。

### build1的reporter失败必须保留

V62 build1留下15步中间小图报告，却在最终report阶段失败：ABI绑定循环把argparse的 `args` 覆盖为list，`args.mode` 抛出AttributeError。没有正式receipt，不能把中间文件改判为PASS。

build2只将该循环变量改为 `arg_types`；native、core、ABI、library字节相同。实际tiny分别通过diagnostic的75项逐步检查，以及production的15次prediction检查加15项请求末尾检查。原FAIL日志、rc1、wait/reap和外部释放仍单独保存。差异见 [reporter差异](31-native-direct-profiling-source/prototypes/v62-native-direct-pipeline-build2/smoke-from-build1.diff)。

## V64：同一生命周期，两种不同覆盖

V64 adapter把V56的enable路由到V62；剩余预算、最多两票、FIFO等待、交付4字节预测和EOS discard保持原合同。源码见 [V64 adapter](31-native-direct-profiling-source/prototypes/v64-native-direct-async-runner-build1/native_runner.py)。两种完整资格各用独立进程加载一次模型。

| 模式 | 严格输出哈希检查 | 实际范围 |
|---|---:|---|
| diagnostic | 12,540 | 核心93步×132，加回退后和早停后新请求的两次最终132 |
| production | 660 | 五个quiescent请求末尾各132；另核对161个delivered prediction IDs |

两种模式均为161 delivered、162 submitted/completed、1 discarded、6 resets，S bank执行84/78。**production的660项是末尾检查，不能写成12,540项逐步检查。** early-stop六个交付步也不宣称逐步完整132项比较；两个模式的数字不能合并成更强的证明。

production请求末尾接口要求队列完全交付；发生discard的早停请求不做该快照。随后新请求的终态检查证明reset后可重用，不能反向补成被丢弃请求的全部状态证据。保存结果的独立核对见[模式审查](31-native-direct-profiling-source/reviews/v64-results-review-build1/REVIEW.md)。

## V67：引擎计时包含seed设置

V67正式执行production direct + async adapter。三个warm、三个measure全部生成相同32个IDs，累计186 delivered/completed、0 discard、6 resets、bank96/90；每请求最后132项与旧A哈希一致，共792项。只读capture owner是保存的运行时断言，不能从已退出进程重新构造原字节。

```text
t0 -> 普通prefill产生首个token -> t1
   -> native reset/131范围seed上传 -> 31步native交付 -> t2
```

| 三次measure的中位数 | 结果 |
|---|---:|
| `(t2-t1)/31`，包含seed设置 | 34.691296 ms/步 |
| `31/(t2-t1)` | 28.825675 token/s |
| `32/(t2-t0)`，整条请求 | 25.922431 token/s |

哈希复制与资源采样放在这些区间外；首次warm捕获只作描述。V57的34.545681 ms与这里的34.691296 ms涉及不同执行路径和区间，不能直接相减为异步收益。V67是B-only引擎探索，没有真实HTTP或平衡配对对照，不晋级为正式性能胜出。

## V60/V61：数值通过，局部性能仍STOP

V60将48个head的exp值移到预处理；V61试用更宽的value tile。两者以卷积后、已激活输入为边界：激活在候选外完成，算子没有接入引擎。实际各用9个历史真实bank与6个synthetic bank完成30项严格FP32输出比较；V60另有15个exp-bank检查。register/spill收益没有被测量证明。

局部micro的A是原V44 activated算子，B分别是V60或V61；两侧都不经过mx.compile。只读saved输入轮转，每次消费o与S_new，输出不回填下一次输入。每block固定prime2048、timed8192；10 pairs含4个同臂控制和6个AB/BA比较，共20个timed与20个prime区间。

门槛保持原样：每个控制比值在[0.95,1.05]，每臂两次控制中位在[0.98,1.02]；性能还需至少5%耗时降低、6对中至少5胜。这里的降低率是 **1−B臂中位/A臂中位**，不是逐对降低率的中位数。

| 候选 | A中位 | B中位 | 耗时降低 | 胜数 | 结果 |
|---|---:|---:|---:|---:|---|
| V60 exp48 | 126.225121 µs | 127.740087 µs | −1.200210% | 1/6 | STOP-MICRO-GATE |
| V61 tile64 | 126.996638 µs | 129.742444 µs | −2.162110% | 1/6 | STOP-MICRO-GATE |

两轮控制均通过；负降低率表示B更慢，不是统计控制漂移。原STOP、不重试和不采用保留，局部数值PASS不授予性能、fullmodel或E2E资格。完整控制比值与原时间戳见[结果审查](31-native-direct-profiling-source/reviews/v67-results-review-build1/REVIEW.md)及文本归档。

## V63：用raw counter选题，不能换算生产份额

V63把每个原ICB命令放入单独encoder，并加入链式fence和采样。93个数值步骤仍有12,276项严格输出证据；90步用原路径，另外3步是1个unsampled segmented control与2个sampled诊断。

每份profile有1625条命令。两份sampled的全部3250个uint64 delta有效；unsampled的1625个delta按设计为null。按原PSO实参K/N及原参数range，审查解码每份497条QMV，三份共1491条映射，unknown为0。数据见 [全部QMV映射](31-native-direct-profiling-source/reviews/v63-results-review-build1/qmv-bindings.csv)。

| sampled profile的类别 | 命令数 | reset sum(raw delta) | changed sum(raw delta) |
|---|---:|---:|---:|
| QMV | 497 | 34,454,870 | 34,935,333 |
| GDN state | 48 | 866,664 | 871,666 |
| SDPA | 16 | 280,248 | 277,581 |

表中raw数值仅属于这次分段采样。**SDK timestamp字段单位UNKNOWN，捕获没有校准或频率查询；encoder、链式fence和采样也扰动了执行。** 不把raw delta换成exclusive毫秒、生产时间份额或可节省毫秒。

CommandBuffer的GPUStart/End另有明确的秒单位，control包络68.945833 ms、两个sampled包络101.538667/124.894750 ms也只是这种诊断形态的CB包络，不是生产逐kernel独占时间。QMV映射为下一步投影实验提供线索，不能直接承诺收益。

## V66：四组原始指针，先过局部数值门槛

V66把GDN qkv、z、b、a四组原始weight/scale指针接入一个投影实验，没有预打包权重。build1在projection比较前被输入布局守卫拦住：BF16 `[1,5120]` 的一个实际view为stride `[0,1]`。检查数为0，是测试守卫失败，不是输出数值不一致。

build2按singleton row的地址规则修正守卫：B1时列stride为1即可，row stride不参与地址计算；多行weight/scale仍须canonical布局。原kernel body/header不变，旧FAIL保留。差异见 [V66 build2守卫差异](31-native-direct-profiling-source/prototypes/v66-gdn-four-projection-build2/source-diff.patch)。

实际3个selected layer×11个vector=33个层向量案例；qkv/z/b/a各3种比较，共396项严格BF16比较，35个原始数组只读，额外packed weight为0。四个输出都消费。receipt明确模型未加载、未接入引擎、没有计时；历史向量来自MLP路径，不能写成新采集的GDN激活。旧V16 STOP也没有被覆盖。

## 归档边界

本章收录V57、V60、V61、V62、V63、V64、V66、V67及16次实际运行；保留build1失败、后续修正、全部raw文本、owner/release与历史CPU-READY措辞。V65与V68/V69/V70属于后续工作，只提准备方向，不收其源码或变化中的运行目录，也不报告它们稍后的结果。

归档逐字节保存文本并用24–30章精确内容引用继承依赖；确定性gzip/tar记录每个成员的原路径、SHA与长度。MLX MIT与Lithos Apache许可分别保留，不能用Apache替换MIT。runtime/native二进制、权重与张量不复制，只保留外部身份；这不是脱离原环境可直接执行的模型包。

独立CPU复核重算micro与screen原时间戳、核对V57/V63/V64/V67的旧A哈希范围、校验V66计数与全部保存释放证据；不导入归档模块或MLX/native，不运行GPU。结果见[verification](31-native-direct-profiling-source/verification.json)、[manifest](31-native-direct-profiling-source/manifest.json)与[依赖表](31-native-direct-profiling-source/dependency-map.json)。归档和数值资格不授予采用或新运行权限。
