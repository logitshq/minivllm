# V127 保存结果复核

PASS，仅限同输入局部数值。复核者是 V127 harness/计划/pilot 作者，这是 author recheck，不能称独立源码作者审查；实际 GPU 执行与实时释放证据由 root 完成。

77 pins 与冻结 manifest 原字节一致；1491完整 BF16 矩阵严格比较记录通过：497原真实B轨迹T5、994同X派生T2/T4；九几何×三T均走共享W，无fallback。497旧输入、原batch输出、native-M1输出SHA全部复现；含lm_head。2485 actual GPU shape/stride descriptor一致，内部MLX flags未观察。

497条X/原Y/W/scales owner与字节RO、1349参数字节前后及源内identity断言通过。literal histories与producer先同边界eval，keep0全部128签名回到原initial fixture由源内断言完成；未另存逐项post-keep0或完整literal-history RO快照，不能声称本次CPU重算了这些原tensor。完整1491输出raw也未保存，本审查重放比较旗标/shape/dtype/SHA与精确bytes比较源码。

21worker+20owner+1post=42资源样本均AC/thermal0/pressure1，Swapouts17519593、swap5038.06MB不变。allocator峰值16,952,614,436B，非RSS；cache配置64MiB而sample最大69,883,654B，不声称硬上限未超。一次targetload、零draftload，一次普通raw32warm；固定原B proposal fixture，不是live drafting。

root tool82729 exit0，50896/50911已wait/reap并有严格externalabsence；复用root实时证明，未liveps。无整模spec、性能或采用结论，V129仍unfinished/unrun。
