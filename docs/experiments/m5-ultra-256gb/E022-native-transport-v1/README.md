# E022 v1：socket 布尔读回检查失败

状态：**实验未完成，不进入 throughput–latency Pareto，无性能结论。**

第一组 OFF 完成 3 warm + 3 formal；第二个进程 ON 在 readiness 时退出，尚无请求计时。
设置 `TCP_NODELAY=True` 后，本机 `getsockopt` 返回原始值 `4`；冻结实现错误地要求整数必须等于 `1`。
因此 fatal callback 按设计退出整个自有进程（exit 70），停止余下矩阵。没有据此判定候选慢或数值错误。
早先 CPU mock 只模拟了 0/1，未覆盖操作系统返回非零 flag 的语义。

OFF 的 772.208 ms / 41.440 tok/s 仅为失败批次的诊断数据，不与历史点拼接，也不复用于下一版。
原始源码、计划、所有请求、失败、资源和 lease 共 66 文件完整保存于 [evidence.tar.gz](evidence.tar.gz)，
逐文件大小及 SHA 见 [manifest.json](manifest.json)，归档重开后逐成员校验通过。
[summary.json](summary.json) 保存停止原因和计数；这是失败与归档完整性检查，不是完整矩阵数值资格。

两个自有进程均已 wait/reap，PID 和 listener 消失；外部服务原 identity 不变，T→S 恢复。
继承的 E020 core/runtime/model 身份由 v1 plan 和 pins-before 绑定，完整基线归档见
[E020](../E020-external-spec-http-v1/README.md)。本归档不包含权重。

修复必须使用独立 v2 源码、计划和全新的 OFF/ON 三对；保留原 v1 失败，不修改原门槛。
完整实验方法见 [E022 教程](../../../optimization-guide/tutorials/022-native-transport.md)。
