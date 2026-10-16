# NVFP4 QMV：证据与源码参考

V22 `STOP-MICRO-GATE`，未采用；原QMM对R2六对降低0.212667%、5/6胜，未过5%门。

- `result.json`、`startup.json`：原结果和首次调用记录逐字副本；首次调用不是纯编译时间。
- `evidence.tar.gz`：原样保存完整10对（含全部正负AB与两个AA/两个BB）、27项byte摘要、24份资源、输入/identity/runtime/qualification、startup及child/log/双strict absence/root工具返回。
- `reference.tar.gz`：原型driver/test/Metal header/body、协议/source manifest/CPU收据/CPU独立review、原source研究与source map。保留MLX MIT许可证；没有checkpoint权重。
- `source-map.json`：原研究源码映射逐字副本；源码参考包内另有精确原型的source manifest。
- `EVIDENCE.json`：每个tar成员的原路径、大小、SHA及压缩包SHA。原临时证据未修改；解包后成员应与清单逐字相同。
- `INDEPENDENT-CPU-REVIEW.md`：原型发射前独立CPU审查；`INDEPENDENT-RESULTS-REVIEW.md`与`INDEPENDENT-RESULTS-RECEIPT.json`收据复算原统计、原始hash/资源/退出，确认STOP，不重建未保存输出。
- `ARCHIVE-RECEIPT.json`：归档者仅核对已有元数据/资源/退出/SHA，不独立重建原输出张量；GPU运行由root完成。
- `SOURCE-UNCHANGED.json`：本章前HEAD及26个引擎源码SHA；V19 default OFF，未集成R2。

查看：`tar -tzf evidence.tar.gz`、`tar -tzf reference.tar.gz`。参考包不是自包含发行包：本机绝对路径、v16 source manifest、checkpoint、历史L0输入、resource wrapper及MLX环境仍为外部依赖，换机需适配后重新冻结、重新资格测试。原型不能直接作为默认引擎补丁。
本Battery探索不合并AC结果，不构成完整模型、cache/lifecycle、E2E或正式性能前沿资格；本轮不再重跑/sweep。
