# 256GB Mac 交接包

先将 [HANDOFF.md](HANDOFF.md) 交给目标机 agent。clone 实验分支或 `tutorial-17-handoff-v1` 后进入本目录，先核验再展开：

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

本机build3 FULLMODEL_QUAL_PASS；E2E actual=0、待测，无整体胜出，V19默认OFF。旧两次STOP永久保留。
有真实可用引擎源码；完整目标机harness尚需迁移、重建、冻结和审核。目标机必须重新执行完整模型资格与官方baseline对照。
本包不含模型、大raw、用户check_layers.py改动、旧GO或latch。历史SHA和路径不是目标机授权；解包目录没有.git，完整Git历史由实验分支及相应source/static/numeric tags提供。
