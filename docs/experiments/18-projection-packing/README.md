# 输出行拼接：证据与源码参考

v20 GDN4→1 与 v21 MLP2→1 均为 `STOP-MICRO-GATE`，没有采纳、默认启用或新 E2E。

- `v20-result.json`、`v21-result.json`：原始结果逐字副本。
- `evidence.tar.gz`：原样保存两次 run 的 result/bytes/pairs/resources/identity/runtime/startup/inputs、child.log、CPU review、root 工具退出和 strict absence；含旧 AC 的 summary/result/5 个配对。
- `v20-independent-result-review.json`：独立 CPU replay 收据；核验已有 hash、统计、资源与退出记录，没有重跑 GPU 或重建未保留的输出字节。
- `v21-independent-result-review.json` / `.md`：独立结果与协议 replay，确认原 STOP；两份 CPU source review 也保存在压缩包的 reviews/ 目录。
- `EVIDENCE.json`：每个归档成员的原路径、原 SHA256 与大小；原临时目录未修改，也没有 sanitized 派生数据。
- `reference.tar.gz`：冻结的两份 driver/test，以及原 v16 generic helper、layers/norm 依赖、source manifest。**不是可直接运行的独立发行包**：本地路径、MLX 环境、checkpoint 和历史 real 输入仍是外部依赖；换机须适配并重新冻结、重新资格测试。
- `SOURCE-UNCHANGED.json`：本分支 `6b50184` 上 26 个引擎 Python 源文件的 SHA；V19 仍 default OFF，未集成投影拼接。

查看归档目录：`tar -tzf evidence.tar.gz`；解包后可直接阅读全部 10 对的两个 block 和控制对，负配对完整保留。
两次独立 receipt 各自标明有限作用域；旧 AC、v20 Battery 与 v21 Battery 不合并成一个性能结论。
参考包不含 checkpoint tensor、完整模型或历史全部权重文件清单。
