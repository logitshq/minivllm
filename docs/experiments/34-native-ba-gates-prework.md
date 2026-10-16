# 34. 把BA投影和FP32 gates放进GDN prework

本章记录一条独立扩展：V76在已有activated norm prework里计算两路小BA投影和FP32 gates，V82把它接入29py候选并通过整模数值资格，V86随后做B-only原生decode screen。三次运行均已退出/reap/外部释放；候选仍defaultOFF，没有采用或HTTP胜出结论。

| 阶段 | 已保存的实际结果 | 边界 |
|---|---|---|
| V76 local numeric | 45cases、720项严格比较、71原数组readonly | 不是连贯整模输入或时间反馈链 |
| V82 full qualification | 3请求、93×132=12,276项、96IDs对双旧gold | 数值资格，不是性能点 |
| V86 B-only screen | 3fresh warm+3fresh measure、792项终态哈希 | raw native计时，无配对A或HTTP客户端 |

原始receipt、资源、owner/release在[证据索引](34-native-ba-gates-prework-source/evidence-index.json)；[provenance](34-native-ba-gates-prework-source/provenance.json)只列这三次actual run。V78/79/80/81/83/84/85等实验不并入本章，必要旧source pins仅作来源依赖。

## 为什么不照搬Qwen4的gates

所读oMLX补丁源码 `qwen35_gdn_prework.py:2054–2100` 的Q35 decode融合conv/SiLU/L2预处理，gates和recurrence仍分开。该文件`:300–322` 的Qwen4分支使用不同的BF16 gates/outer exp，不能把它当成Q35精确数学的替代。V76是沿预处理结构提出的扩展；不声称被测oMLX已经使用同一个融合，也不声称存在whole-generation-step compile。

精确路径、SHA和分支说明保留在[source proof](34-native-ba-gates-prework-source/prototypes/v76-gdn-gates-prework-build1/source-proof.json)。这份是所读补丁源码身份；不把当前main或补丁revision冒称被测wheel的VCS commit。

V76保留原QK16组的serial norm顺序；48个V组各用128线程复制V，其中前32lane分别算B/A的一个输出行。每lane仍按原K512顺序执行10步，使用原qdot、FP4/FP8解码、scale和simd_sum。

BA结果先各自转BF16，再转FP32，之后调用已冻结的installed Sigmoid、Add、Exp、Negative、Multiply、LogAddExp/log1p float functors。softplus仍是 `logaddexp(x,0)`；没有换成另一个exp/log公式。state40的body/name/header/options和S input6都保持原样。

```text
原始normX -> 原WB/SB与WA/SA -> BF16 b/a -> FP32 beta/g
原conv23 activated -> 原Q/K norm + V/scales ----------------┐
                     六FP32输出(Q,K,V,scales,g,beta) -> 原state40
```

生产prework声明8输入+6输出，共14个buffer参数；资格probe多导出BF16 b/a，共16个。**这是源码ABI计数，不是实际逐参数反射观察。** native40没有导出该表，V82/V86实际字段均保留`NOT_OBSERVED`。

## V76局部资格检查了什么

只读取checkpoint中layers0/32/62的18个范围，共830,016B；原U32权重/U8 scales不重排、不concat。保存的BF16 A_log/dt_bias按原loader转换为FP32，未加载整模。

45cases来自3layers×15份历史u/S bank(9real、6synthetic)。activated为保存RAW u经原SiLU；X是八个预声明generic向量，按bankindex%8选择。它们不是同一次模型forward的隐藏状态与激活元组，不能拿这些局部通过替代整模资格。

每case16项：两路QMM与诊断BA比较，原QMM/compiled gates/44 prework与生产六输出比较，生产与诊断六输出比较，以及最终o/S两项。共720项shape/dtype/finite/raw-byte严格比较，全部通过；每layer后71个原数组地址与SHA保持不变。

这里BA与gates进入同一safe prework编译环境，CPU源码相同不证明Metal contraction/JIT上下文等价。因此保留实际BA、六输出和o/S严格门槛，不用容差放宽。原README里的CPU准备/PENDING状态保留历史原字节，以actual receipt为结果。

## V82：真实布局与完整模型

29py候选保留原decode/prefill方法，默认关闭新路径。先在BA OFF下执行一次额外普通decode，在capture和93步资格之外取得48层真实input-layernorm X及conv23 activation；再用独立GPU stride probe记录每层六参数+两激活，共384条布局。

实际384条均为canonical mapping；`ensure_row_contiguous=False`，internal MLX flags仍明确未观察。48个prepared helper保留各自六个原参数对象，无packing、无额外权重副本，prepare前后active bytes不增加。288个原参数范围共13,289,472B，不相交并保持完整range readonly；全部1349个模型参数owner身份不换。

B的capture观察到48个distinct GDN对象和48个prepared helper各一次，六FP32输出shape/dtype都匹配。两个观察hook通过finally恢复，之后才native replay。原参数地址在实际captured buffer范围中逐项匹配，并带有新prework函数的readonly角色记录；这与未观察的14参数反射计数是两项不同证据。

A在每次参考调用时关闭BA/conv/gates，临时绑定accepted25原_decode AST，eval位于该context内，finally恢复。B只在每请求开始上传131seed，随后消耗自己捕获的GPU argmax反馈；A或保存的continuation不进入B的逐步输入。

| V82实际检查 | 结果 |
|---|---|
| base/reset/changed-prefill | 各32IDs；changed长度20，实际改变129个seed输入 |
| 完整输出 | 每步logits+128state+pred/nextpos/nextlens，共132项 |
| 双旧gold | 12,276项A SHA逐项同旧V38与V44，B字节严格等A |
| 原始owner readonly | 每步131capture seed+132capture outputs+131request seed；A retained输入另查 |
| native capture/replay | 1433dispatch、2ICBs、93完成、banks48/45 |
| feedback | 48S eligible、83staged links；每步无CPU seed重写 |

原始readonly基线在capture/replay前一次性保存，不是当前值同自身比较。数值通过只覆盖此限定模型、设备、版本、prompt和路径，不能从1433条调度数直接推断速度。独立CPU复核见[REVIEW](34-native-ba-gates-prework-source/reviews/v82-independent-results-review-build1/REVIEW.md)。

## V86：六fresh请求的原生绝对时间

V86绑定V82实际成功receipt SHA及release，candidate82/native40不变；一次weights load，先普通32 warm、布局准备与一次capture，再三fresh warm、三fresh measure，均无KV请求间复用。

```text
t0 -> fresh prefill/首token/flatseed eval + native reset/131seed upload -> t1
                       31 native步，每步只观察4B prediction          -> t2
```

| 三个measure中位 | V86实际 |
|---|---:|
| `(t2−t1)/31` | 34.852046 ms/步 |
| `31/(t2−t1)` | 28.692720 token/s |
| `32/(t2−t0)` | 25.415153 token/s |

计时decode内没有A/model调用、整state CPU复制/hash、资源探测、日志或写文件。每请求完成后132项终态哈希对旧44 A，六次共792；全部32IDs一致，186步/6reset/banks96/90。263个捕获原owner的341,939,224B一次性CPU readonly备份与288参数检查均在时钟外。

t1在所有seed上传**之后**，不同于67/71的engine时钟包含native begin/upload。该表是B-only raw native绝对时间，不是HTTP指标，没有公平配对A，未证明speedup、Pareto或采用。native CB GPU envelope/profile也不是kernel独占时间或生产占比。

## 归档与复算

[归档README](34-native-ba-gates-prework-source/README.md)列出精确来源、29py候选和完整MIT/Apache许可证；[source-map](34-native-ba-gates-prework-source/source-map.json)和[dependency-map](34-native-ba-gates-prework-source/dependency-map.json)支持`tar.gz::成员`引用。旧V24/V25大gold保留在第33章的精确压缩成员中，本章不重复收录；本章所需旧V38/V44 gold直接引用已有文本归档。

只归档文本证据；权重、tensor fixture、native/runtime二进制仅保留身份。离机命令验证文本/来源哈希/保存结果统计；`--originals`额外检查本机原件与外部fixture身份，不重跑MLX/GPU或重建已退出进程的readonly状态。

```sh
python3 -B docs/experiments/34-native-ba-gates-prework-source/verify_archive.py
python3 -B docs/experiments/34-native-ba-gates-prework-source/verify_archive.py --originals
```

[verification](34-native-ba-gates-prework-source/verification.json)、[original-path verification](34-native-ba-gates-prework-source/verification-originals.json)和[manifest](34-native-ba-gates-prework-source/manifest.json)封存复算结果。候选仍defaultOFF，归档不是可重定位的模型执行包，也不授权复跑性能。
