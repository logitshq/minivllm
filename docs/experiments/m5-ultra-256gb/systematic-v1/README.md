# 系统优化第一轮：证据索引

完整叙述、复现命令和取舍见[实验账本](../../../optimization-guide/experiments.md)。

- `E007/`：大batch独立GDN融合。保留所有合成/全模型检查、完整计时和调度trace。
- `E008/`：2/16 GiB allocator配对。保留预注册计划、冻结源码身份和独立重算汇总。
- `E009/`：GDN线程内状态片段复用。保留成功或失败的资格检查及原始决定。
- `manifest.json`：本地原始文件路径、字节数、SHA256，以及本目录文件SHA256。

计划中的绝对路径是M5上的证据位置。原始HTTP SSE事件留在`.artifacts/`，没有
作为远程仓库依赖。模型revision来自固定的Hugging Face snapshot；模型权重
文件未包含在这份小档案中。不要把运行时源码pin数量说成权重文件校验数量。

E007旧字段`cold_ms`实际是正确性求值后的第一次计时，非冷编译；E009改正了
字段命名。两者的正式中位数都来自预声明的预热之后的ABBA块。

独立HTTP重算和归档：

```sh
.venv/bin/python -B bench/m5_ultra_256gb/allocator_http/summarize.py \
  --root .artifacts/m5-ultra-256gb/allocator-http-v1 \
  --out .artifacts/m5-ultra-256gb/allocator-http-v1/summary.json
.venv/bin/python -B bench/m5_ultra_256gb/archive_systematic.py
```

归档器检查源文件pin、运行状态和进程清理收据，复制汇总并计算原始文件清单。
它不重跑GPU，也不替代独立硬件复现。生产引擎只接受通过完整资格与收益门的改动。
