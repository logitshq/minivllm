# 031：把GDN融合放进长decode编译图

2026-10-18。组合数值、资源与恢复的独立复核已通过；实际组合HTTP验证已启动，尚无组合吞吐成绩。单项成绩见[v15](../progress/2026-10-17-v15/README.md)：GDN改善C1，E024保留C8优势，不能将两个百分比相加。

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
