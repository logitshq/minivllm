# 第37章精确源码与结果归档

配套[正文](../37-simd4-async-and-spec-handoff.md)。本目录保存V95/V99/V101–107的冻结源码、独立审查，以及8次实际运行的文本证据；V108/V109仍在进行，未包含。

- `source-map.json`：258项原始源码身份；相同历史文本引用前面章节，新增文件按原字节保存。
- `evidence.tar.gz` / `evidence-index.json`：566个文本成员、8次运行；无模型、raw tensor或原生可执行文件。
- `dependency-map.json`：797项冻结依赖。`external-fixture-identities.json`仅列外部未打包文件身份（含二进制、数组与模板），离机模式不验证它们的内容。
- `provenance.json`：实际运行、释放凭证与明确边界。V105为热状态STOP，没有完整HTTP曲线或Pareto胜出；V107为failclosed HOLD。

在完整仓库checkout内运行（CPU/stdlib）：

```sh
python3 -B docs/experiments/37-simd4-async-and-spec-handoff-source/verify_archive.py
```

源机可加 `--originals` 核对外部原始路径及字节。离机模式只检查归档文本、hash与保存的计数，不执行MLX/native代码、不重新计算tensor、不提供目标机运行资格。原harness绝对路径及动态库需要目标机独立适配、编译和重新冻结。依赖前面章节的精确文本，不能只复制本目录当独立发行版。

MLX/MLX-LM MIT、oMLX及Lithos Apache-2.0等许可证在 `licenses/` 或源码映射的前章位置保持；不改变第三方授权。
