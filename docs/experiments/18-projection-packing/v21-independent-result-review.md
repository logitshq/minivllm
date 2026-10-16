# Independent v21 results replay — PASS audit; STOP-MICRO-GATE retained
2026-10-16；只CPU stdlib读取/复算；无新tests、MLX/GPU/model/探针/process命令；source与raw未改。
Driver SHA256: c42c704ce1709e5e4f71d5cf3610a2436f8a47fbc95d8d537bc3efb38dee6961；identity一致，26冻结source及runtime模块SHA全部复验。
范围MLP gate/up L0/32/63，各17408行，12目标tensor共300810240B；未加载完整shard/模型，未做E2E。
66条唯一byte记录完整覆盖3×(8 synthetic+1历史real+B2/B4)×两段；shape/dtype/nbytes正确、finite/exact真、差字节0、左右SHA一致。
这是GPU记录/hash一致性复核；原输出tensor未保存，未把CPU复核描述为重跑GPU数值。
三份历史real输入文件SHA逐一匹配固定旧receipt SHA94a9f5ab7ae3780f306e66d71e4f3b0f2fb559eef4d5a89ae179ef63ef3355c8；均[1,5120] BF16/10240B。
来源为历史V17 normal-decode MLP输入，非本轮capture；旧receipt整体STOP_BUILD1保留，只用其ACTUAL64_NUMERIC_PASS数值出处。
8个synthetic timing输入hash唯一；startup三组选定slab均记录packed QMM count1。
10对schedule吻合：前后各AA/BB+6个均衡AB/BA；20block prime128/repeat512、3组/6段，duration与ns/call逐项复算相同，pair两侧全部6hash一致。
控制A ratios=1.0037577707/1.0003949866，中位1.0020763786；B=.9954247293/.9948800573，中位.9951523933。
各控制均在[.95,1.05]且中位在[.98,1.02]，tight control PASS；这只通过预声明位置门，不单独证明细小性能差异稳定。
A/B中位761211.58984375/762147.6240234375 ns；reduction=-0.12296635944279988%，3/6胜（AB3/3，BA0/3）。
≥5%且≥5/6的微门两项都未通过，独立状态STOP-MICRO-GATE与result.json完全吻合；无adopt。
27资源样本均Battery/thermal0 nominal/pressure1；从原vm_stat逐样本重解析：SO/SI/PO/Compressions增量0，仅Decompressions+24。
原swap文字解析为4719.56MB且每样本不高于起点，Swapouts全程未增；known matcher0的scope不排除未知/GUI GPU使用或采样间变化。
active601733120→300922880B，减少300810240B，正好对应deleted原payload；peak602617996B、after cache302891149B。
该active观察不能扩展为物理释放或full48内存资格；原output/weight完整allocation指针未证。
child29827 recorded exit2/finally退出；controller cleanup与root posttool absence均ps rc1、stdout/stderr literal empty，正常STOP已清理。
adopted/full_model_qualified/E2E_run/AC_frontier_entry全false；GDN旧STOP及本次MLP STOP均封存，不继承旧21.9%主张。
本次没有达到指定Battery小组探索门槛；不据此宣称所有rowpacking/整模型/workload普遍无效，不要求新测试。
原13文件SHA+bytes seals、完整统计/控制/资源/退出复算见INDEPENDENT-RESULTS-RECEIPT.json；封存前全部raw seal二次核对未变。
