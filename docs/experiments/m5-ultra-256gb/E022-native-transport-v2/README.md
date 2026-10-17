# E022 v2：正确性通过，transport 收益未过采用门

绑定提交 `3873101` 与 plan
`e850dcdcc1d160ecddf512529758b32fc9a8241c6af515fb72901c4a6b28c81c`。
6 个 fresh server、18 warm + 18 formal 请求全部完成；576 个正式输出 tokens。
完整输出、socket、资源、回收与外部服务恢复的独立审计通过，原始全部请求保留。

| 配对 | OFF/ON 完整请求加速 ↑ | OFF/ON TTFT 加速 ↑ |
|---|---:|---:|
| 1 | 1.017017× | 0.899891× |
| 2 | 1.016630× | 1.005569× |
| 3 | 1.007713× | 1.021203× |

完整请求加速几何平均 **1.013777×**，小于预注册 1.05 门；第一对 TTFT 小于 0.97 下限。
因此 **不采用当前候选**。这是性能门失败，正确性没有失败；不能删掉合格但慢的观测。
本轮仅测 transport，不能声称 GPU/kernel 加速，不能与 E020 或 v1 partial OFF 拼接前沿。

直接结果见 [result.json](result.json)，完整独立审计见本目录 audit 输出；
[summary.json](summary.json) 给出决策和归档摘要。
[evidence.tar.gz](evidence.tar.gz) 保存 223 个运行、lease、源码、计划和审计文件，
[manifest.json](manifest.json) 为逐成员大小/SHA；归档重开后全部校验。
继承的模型/core/runtime 身份仍引用 [E020 原始归档](../E020-external-spec-http-v1/README.md)。

六个自有 PID 全部 wait/reap、listener 消失。外部 PID74384 原 identity 不变，T→S 恢复。
Apple 布尔 getter 的修复有效：ON 实际 raw4、OFF raw0，原值和 normalized 值均保存。
v1 原失败保持，不从本次成功反向改写。下一条大方向仍是长任务 decode 图覆盖，独立验证。
