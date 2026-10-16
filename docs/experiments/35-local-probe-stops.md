# 35. 三条局部优化通过数值，仍未过采用门槛

本章只收9次局部运行：V78/V79数值、V80数值及micro、V81数值、V83数值、V84数值及micro、V85 micro。V76/V82/V86整模路线已在第34章隔离；V89/V90等后续结果不收。全部原始receipt、owner/release与资源记录见[证据索引](35-local-probe-stops-source/evidence-index.json)。

| 机制 | 数值资格 | 平衡micro实际结果 | 判定 |
|---|---|---|---|
| V79→V80 paired-half2 FP4 decode | 36局部；再72MLX+128native | A100.684446/B166.393819µs，慢65.262686%，0/6胜 | controls PASS，STOP |
| V83→V84 outer-K unroll2 | 36局部；再72MLX+128native | A100.891930/B100.295504µs，下降0.591153%，6/6胜 | controls PASS，<5%，STOP |
| V78→V81→V85 scaled-ready GDN | 30ready+30diagnostic+30o/S；再30public+60captured+480native | A18.962166/B18.331746µs/两dispatch bank，下降3.324620%，6/6胜 | controls PASS，<5%，STOP |

正值均按`1−B中位/A中位`重算。三个micro门槛在执行前固定：至少5%下降、至少5/6胜；controls每次偏差不超过5%、每臂中位不超过2%。没有调整门槛、重判旧STOP或采用。

## FP4解码：half2和unroll是不同实验

V79在原R4/SG2的qdot内尝试成对half2解码，保留原权重布局、每个block scale、归约和BF16输出，不折叠16384因子、不重排权重。V79的一份真实L0gate权重配8synthetic+1saved-real向量，留下36项有限、严格BF16记录：9项原R4桥接与27项compiled A/B比较，不是整模证明。

V80把它放入第33章V72的native范围边界：layers0/8/16/24/32/40/48/56八份真实gate W/scales，共401,080,320B；各arm8个独占输出slot。一次捕获A8+B8，静态ICB range[0,8)/[8,16)，无fullmodel feedback。72MLX比较和128个native终态输出均严格通过；新性能marker绑定成功资格后才计时。

V80的A/B中位100.684446/166.393819µs是每dispatch wall，包含一次ffi、编码、commit/wait再除8192，没有逐kernel Python/eval。它表明这次M4范围微测里的half2实现明显退步；不修改V79数值通过，也不把别的硬件路径当成本机性能。

V83是另一项最小变化：以原66 R4/SG2 header为基线，只在外层K循环前加入一次 `#pragma clang loop unroll_count(2)`。删除这条pragma后header逐字等原66，qdot/scale/K顺序及几何未改。思路来自所读Lithos NV_UNROLL2，不能推断其M4实际recipe已engaged。

V83实际manifest为`3d250d55…69ab9`、micro为`d7078854…954ab`；V84同时绑定它们、actual identity、owner、36项qualification/readonly及release。V83 result JSON恰与旧36项结果同SHA，**只看该通用result不能确定执行了哪版源码**。早先cpu-inert1的preseal标签错误已由CPU2修正；历史identity保留，不能当当前runtime身份，实际GPU源码未受影响。

V84使用同八bank、原128prime/1024timed groups、六AB/BA与四AA/BB，B精确function及完整active schema必须匹配。数值72+128通过，随后只有0.591153%下降；6胜仍没有达到5%门槛。

## Scaled-ready：真实四PSO，不是十个active state参数

V78把Q/K的FP32 scale乘法提前到prework输出，后继state读取scaled-ready。15份保存bank(9real、6synthetic)固定g/beta/S，没有自回归反馈。30项ready、30项diagnostic和30项o/S严格记录通过；15份输入before/after一致。它们不是新模型capture，也不是整模数值资格。

V81一次捕获两arm×15bank×两kernel，共60dispatches，构造两range；prework→state producer buffer逐项匹配，每range15个内部依赖barrier。15个原S allocation由A/B共同读取且固定，不让输出回填输入。

实际四PSO为原44 prework、原40 state、scaled78 prework、scaled78 state。原state S/S_new为arg6/9；**scaled state的scales arg3实际inactive**，不能凭源码签名声称十个active buffer。原binding仍按捕获原样编码，maxBind10覆盖recorded bindings；其他访问权限/原allocation证据原样保留。

V81有30public、60captured、480native严格记录。480=8种group次数×2arm×15bank×o/S；只另保存group16的60个native raw和30个reference raw。255个原数组的17个readonly检查均为保存的运行时证据；不是作者重新跑GPU。

V85复用V81同native/math/四PSO和15个固定bank。每group30dispatches，prime128groups、timed1024groups；20timed blocks共614,400dispatches，每block只检查最终15bank的o/S，合计600项终态检查。下降3.324620%、6/6胜、controls PASS，仍STOP；不能把终态检查写成逐dispatch验证，也不能把两kernel平均换算成真实整模份额。

## 保存证据的复核边界

三个micro都在每group结束保留Buffers barrier，包括末组；CB completion、encoded计数和官方GPUStart/End秒值可复算整个CB envelope。它们不提供逐kernel独占时间，数值检查也没有观测所有重复dispatch的中间状态。

[review provenance](35-local-probe-stops-source/review-provenance.json)区分来源：V81是独立CPU保存结果review，V80和V85是作者自查，V85不能标为独立审计。原78/79独立review还含V76内容，所以本章只留其身份，不搬入额外actual scope。本文verifier复核保存文本/哈希/统计，不是新的GPU数值运行。

所有9个owner均已wait/reap并有外部strict absence/release。归档不把旧运行readonly flags变成当前现场观测，也不从资源快照推断持续的全GPU占用。各run的原始资源域与检查口径原样保留，不混为新的性能矩阵。

[归档README](35-local-probe-stops-source/README.md)、[source-map](35-local-probe-stops-source/source-map.json)、[dependency-map](35-local-probe-stops-source/dependency-map.json)保留精确原字节身份并支持跨章`tar.gz::成员`。旧大gold通过已有成员复用，不复制裸JSON；权重、tensor、native/runtime二进制均不收payload，1076个终态fixture仅存身份。

```sh
python3 -B docs/experiments/35-local-probe-stops-source/verify_archive.py
python3 -B docs/experiments/35-local-probe-stops-source/verify_archive.py --originals
```

离机只验证文本/哈希/保存统计；`--originals`需本机原件，额外核外部身份及可用的终态raw bytes/有限数。结果封存在[verification](35-local-probe-stops-source/verification.json)、[originals](35-local-probe-stops-source/verification-originals.json)和[manifest](35-local-probe-stops-source/manifest.json)。三条路线均无整模/HTTP收益或采用声明。
