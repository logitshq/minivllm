# E012：实际设备端选路模块资格

2026-10-17，M5 Ultra 256 GiB，MLX0.32.3。核心`f05c3cf`，运行HEAD `44fcdf1`。
同Qwen3.8-27B NVFP4与DFlash2 checkpoint、affine4bit草稿加载，具体SHA见归档plan。
结论：默认关闭的独立模块通过。完整draft三次几何平均加速1.185085/1.185097/1.190796×。
不构成target验证、回滚、生成或HTTP资格，不更新Pareto图。

- [运行前计划](preregister.json)：固定三进程、三prompt×三block、全部严格检查和ABBA计时。
- [汇总及全部27点](summary.json)、[审计](audit.json)、[CPU接口检查及首次失败](cpu-tests.txt)。
- [证据归档](evidence.tar.gz)与[逐文件SHA256清单](manifest.json)：97文件，含被测核心、原始计时、
  selector/全部draft KV检查、资源、进程回收、服务恢复、模型配置和工作负载。
- [原型初筛](../E012-draft-selector-v1/)、[完整tutorial](../../../optimization-guide/tutorials/012-device-draft-selector.md)。

99组selector分数/IDs及27组完整draft检查严格通过，三进程ID结果一致。12个资源样本
pressure1/thermal0，swap/Swapouts不变；3个子进程均回收且PID不存在，授权暂停的外部
oMLX服务同身份恢复。探针不能排除GUI或未知运行时的GPU工作。

从仓库根目录运行`python -B bench/m5_ultra_256gb/draft_selector/audit_integration.py`可以复核
本机原始证据并生成一次性归档；已有结果不覆盖。跨机器先解包核验manifest，再按本机路径、
硬件和模型重新跑，不能把本机绝对路径或性能资格直接迁移。
