# 256GB Mac 交接包

**最新进展先读 [2026-10-16补记](PROGRESS-20261016.md)**：V99完整资格、V103诊断/生产资格，以及V105 HTTP thermal STOP；整体尚未胜出。

先将 [HANDOFF.md](HANDOFF.md) 交给目标机 agent。clone 实验分支或 `m4-max-final-tests-20261015-v1` 后进入本目录，先核验再展开：

```sh
shasum -a 256 -c SHA256SUMS
TASK_ARCHIVE="$PWD/source-reference.tar.gz"
TASK_ROOT="$PWD/minivllm-256"
mkdir -p "$TASK_ROOT"
tar -xzf "$TASK_ARCHIVE" -C "$TASK_ROOT"
```

- `source-reference.tar.gz`：accepted15 25源、V19 26源、numeric新增文件、精选harness文本和官方Python参考；仅保存一份压缩包。
- [SOURCE-VERSIONS.json](SOURCE-VERSIONS.json)：原f976/b158/8aba/abcbe74身份及逐文件hash；压缩包内也保存相同manifest。
- [LOCAL-RESULT-ADDENDUM.md](LOCAL-RESULT-ADDENDUM.md)：本机build3后续结果与资格边界。
- [FULLQUAL-BUILD3-REVIEW.md](FULLQUAL-BUILD3-REVIEW.md) / [收据](FULLQUAL-BUILD3-REVIEW-RECEIPT.json)：独立严格正确性审查；历史绝对路径仅为源机证据索引。
- `SHA256SUMS`：本目录交付文件校验；压缩包内另有`PAYLOAD-SHA256SUMS`。

本机 48GiB 的 fullqual build3 已独立 FULLMODEL_QUAL_PASS；随后唯一一次 E2E actual attempt 已执行 6 fresh，末尾因 Sustained Swapouts 资源守卫 STOP，未获正式 E2E/性能/采纳资格，整体 win=0，default OFF。 详情见 [E2E原始证据索引](E2E-RESOURCE-STOP.json)。旧两次STOP永久保留。
有真实可用引擎源码；完整目标机harness尚需迁移、重建、冻结和审核。目标机必须重新执行完整模型资格与官方baseline对照。
本包不含模型、大raw、用户check_layers.py改动、旧GO或latch。历史SHA和路径不是目标机授权；解包目录没有.git，完整Git历史由实验分支及相应source/static/numeric tags提供。

[M4最终七点归档](../../benchmarks/m4-max-final/README.md)：21/21完成且固定协议/资源/生命周期独立PASS；稳定热态性能未解决、无稳定双轴整体win。含原证据SHA、独立统计、全部signed对照与脱敏静态reference包。
