# 031：把GDN融合放进长decode编译图

更新：2026-10-19。组合数值、HTTP及正式计时均已完成，资源与恢复独立复核PASS。
首轮入口失败仍保留。正式重试三对C1约提升5.5–5.7%，C8增量无稳定收益；
C1仍落后oMLX，不能宣布目标达成。

## 第一性问题：图究竟编译了哪一个函数

GDN融合减少普通B1路径的预处理算子；E024把长decode的显式状态计算编译复用。这两个开关作用在不同层次。若图先完成tracing，再替换GDN实例方法，已有图可能继续执行旧路径。图缓存的key没有包含Python方法身份，因此安装顺序是正确性的一部分。

[组合安装器](../../../bench/m5_ultra_256gb/gdn_graph_combined/integration.py)仅接受全新、无请求、无ahead、无已编译缓存的runner。先安装48个GDN实例hook，再安装长图dispatcher，最后安装观察器。两个开关默认关闭、可独立设置。退出按反序移除观察器，清除此scope的图和short缓存，恢复原方法。没有新权重包或新投影公式。

## 不能把tracing次数当执行次数

编译图第一次构建时会经过GDN的Python方法；后续图复用可以完全不增加这个Python计数。旧GDN资格中“每次模型调用增加48次”的规则不能直接移到组合路径。

新资格记录真实compiled callable调用，同时记录所有48层在该调用前后的候选/回退计数。同一个bound/B首次使用必须观察到相应trace；之后零计数增量才可解释为复用。B1应走候选，B2–8保留原回退。计数用于确认tracing选择了哪个实现；是否算对仍由独立eager reference的完整数组比较决定。

## 首轮注册与验证

复用原42次参数变更/恢复调用与37请求的完整logits、128状态数组、调度和ahead检查。参数变更覆盖原norm与GDN in_proj_a矩阵，不扩称所有参数热更新已验证。组合输出不得作为参考模型的输入。

- integration机制CPU首跑10/10通过。
- 组合qualifier机制CPU首跑10/10通过。
- 独立审计器新证据负例首跑9/9通过。
- 恢复事实桥接CPU首跑6/6通过；它保留GLM首次加载和恢复监控FAIL，不将其改成RESTORED。
- 2026-10-18 12:10:19 PDT，实际GPU子进程74841开始；owner74808。运行计划SHA `a2af16324fb323dcd9a93e4b9f07d4f127415bcaa0a12f04ca01792caf2696d9`，数值子计划SHA `fa59abb503725b060dd7e26f87e178bb1770178c6094b359fb666c21cac76dcf`。

初始化的`INCOMPLETE`标记只说明当时尚未结束。此轮随后完成：producer与owner PASS，原服务72953恢复为80649，stop/owner/recover退出码均为0。独立数值、原始资源重解析、恢复审计均PASS。

独立账本核到37请求、5,163个调度步骤、6,622,533次完整数组位比较、13,245,066次finite检查。组合图观察到12次tracing、5,144次复用和38次eager fallback；这些是Python/compiled callable事件，不是GPU kernel计数。参数诊断42次；本轮未观察到discarded-prior-ahead，不能补称覆盖。成功完整数组是在设备上比较并保留检查收据，离线审计未重算原始tensor payload。

[完整E030证据](../../experiments/m5-ultra-256gb/E030-gdn-graph-combined-serving-v1/README.md)含969个归档成员、1,660,774,222原始字节、9,958,428压缩字节；每成员SHA与原件复核通过。原数值、资源和恢复结论分别保留。

## 下一轮已经启动

2026-10-18 12:32:47 PDT，新owner81295启动C1 capture→reap→独立replay→C8 capture→reap→独立replay。计划SHA `7783b4c35798d590f1a326739c38a410ac59a3d0655cd384813d6e5b9f0a02e8`，两候选均ON；原服务在driver的finally恢复。新HTTP入口14项CPU机制、独立HTTP审计9项、四阶段生命周期14项通过。初次HTTP异步测试被过严的本地socketpair guard阻断，修正runner后的整14项通过，两个尝试都保留。

此时尚无HTTP整轮结论或正式速度点。正式计时将使用同一入口、route恒ON、GDN OFF/ON，比较组合相对E024-only的增量收益；两个旧单项百分比仍不能相加。


## E031: actual combined HTTP history passed

The next actual cycle completed on2026-10-18 at12:53 PDT. C1 served8 requests and C8 served16, all normal-EOS/cap512 requests; these particular requests each reached512 committed tokens and did not observe terminal EOS. Each capture server was strictly reaped before its independent same-history numerical replay. All four direct children exited through their registered cleanup, and the owner passed. The external Qwen service80649 was replaced by90614; the driver stop, owner and recovery codes were all0.

Independent C1/C8 audits passed5,124 total steps and6,573,486 full-array bit comparisons, with13,146,972 finite checks. The saved successful tensor bytes were compared on device; the offline ledger audit does not recompute those tensor payloads. C1 recorded4,080 ahead consumptions and C8 recorded1,014; neither recorded discarded-prior-ahead coverage. C8 actually exercised B2/B4/B6/B8. The GDN optimization still only applies at B1; larger batches use its original fallback.

The independent cycle audit also passed247 retained resource samples, pressure1, zero swap-use growth and zero new Swapouts. Swapins increased130; retained process inventory cannot rule out every GUI or unknown GPU runtime. Both graph trace records were independently reconstructed: C1 had2 Python tracing and4,086 reuse events; C8 had8 tracing and1,019 reuse events. These are Python compiled-call observations, not GPU kernel counts or speedups.

[The complete E031 archive](../../experiments/m5-ultra-256gb/E031-gdn-graph-combined-http-v1/README.md) retains1,100 members,1,875,145,734 raw bytes and11,470,864 compressed bytes. Every member and original SHA was verified; archiveSHA742a5988d72924beedaf5bf24914413997e569ed09bf2fb38c046b33623c6639. Restoration, C1 numerical/HTTP, C8 numerical/HTTP and full-cycle resource audits all independently passed. This remains observer-enabled qualification, with no new performance point.

The next timing plan compares the same combined server with graph always ON and GDN OFF/ON. It uses12 sequential fresh processes, three pairs at C1/C8,8 warm and64 formal requests per process, and no observers. Its actual CPU preparation validated all12 specs against the closed E031 receipts. Do not multiply the earlier independent GDN and graph speedups; use the resulting paired measurement.


## E032：入口失败也要完整记录

2026-10-18 12:56 PDT，root编写的正式driver把`--run`传给了不接受该选项的owner。真实argparse以2退出；owner输出目录尚未创建，模型子进程、warm请求、formal请求均为0。70项factory/生命周期/审计CPU检查没有覆盖这个新driver，不能把它们写成driver也通过。原失败代码、计划、命令、stdout/stderr全部保留。

外层driver正常wait了失败的CLI，再执行`recover --no-experiment`；原服务90614恢复为91373，独立恢复审计PASS。恢复通过不改变实验失败，也不能产生任何速度点。[E032完整记录](../../experiments/m5-ultra-256gb/E032-gdn-graph-combined-formal-v1/README.md)含182个成员、8,519,786原始字节、305,300压缩字节；逐成员和原文件SHA复核PASS。

修复使用新的retry-v2入口、计划和日志。原12个测速spec、HTTP服务器、数值实现和测量算法不变；新的生命周期绑定实际E032恢复证据。driver用一个函数构造正式与prepare-only命令，并在停止服务前调用真实owner的`--prepare-only`。这是检查真正解析器和计划的便宜入口验证；原失败记录不会被覆盖。


## Retry-v2正在执行：先保留完整配对，再决定采用

修正入口的19项focused CPU通过，提交3fbc028后，实际prepare-only退出0，
随后stop退出0，owner93089于20:06:57 UTC启动。原12个测速spec不改，
计划SHA f4e19bd0a842b48ea26fb2d27f870fdf62409897c7f862067c7800de5949b6f1。
13:52 PDT已完成六个进程，第二轮C1 ON继续；此处不预写最终PASS。

第一对C1为45.7519→48.3191tok/s，第一、二对C8分别仅约+0.0417%/+0.1087%。
这些未经整轮独立审计的中途点只用于选择后续诊断，不能提前作为合格前沿。
完整配对及其反序重复会保留；失败也会归档，不按中途成绩挑选后续样本。

[首轮C1诊断](../../experiments/m5-ultra-256gb/combined-v2-r1-c1-diagnostic-v1/README.md)
显示剩余历史oMLX均值差约85.6%位于首输出之后。其解释仍包括模型执行、
主机提交、线程与交付等；已准备的profile将区分这些范围及观察器自身开销。
[最新权重复用与有限精度笔记](../research/2026-10-18-finite-precision-and-weight-reuse.md)
给出下一项rowwise verifier的成本门：所有prefix精确通过后，再比较完整校验
与同数量普通步骤的成本；未来draft接受率不能代替这一步。


## E033最终闭环：C1改善稳定，C8保留原图配置

复核日期2026-10-19。原服务91373已恢复为21938；prepare/stop/owner/recovery退出码均为0，
12个实际子进程全部回收。独立审计重新读取864条HTTP请求，其中768条正式请求，
不是用中途显示的点代替最终结果。归档3472成员、37156548压缩字节，逐成员和原件SHA复核PASS。

| HTTP并发 | GDN OFF吞吐 | GDN ON吞吐 | ON请求p50 | ON请求p95 |
|---|---:|---:|---:|---:|
| C1 |45.81897|48.35777|10.58590s|10.63603s|
| C8 |170.46064|170.37898|24.01408s|25.49321s|

表为每配置三个独立进程指标的中位数，吞吐单位tok/s，graph两臂恒ON。
C1三次配对吞吐增益为5.611%、5.530%、5.658%；C8为0.0417%、0.1087%、−0.0675%。
这支持C1融合的增量收益，不支持C8进一步加速。没有将此前两个独立开关的百分比相加。
正式运行关闭观察器；其资格来自此前实际数值/HTTP历史，不扩称每种未来调度都已逐位证明。

[E033完整证据](../../experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2/README.md)
保留正式与恢复的各自结论。恢复确认服务重新可用；精确运行时签名与暖KV缓存未恢复。
下一步着重首输出之后的decode路径、每个有效token的权重读取与状态成本，
同时补齐中间并发点和GLM；小幅胜出不能替代逐batch大幅领先的长期目标。
