# E018：persistent GDN micro 初筛及三 fresh process 复测

独立审计通过；复测门结果 **REJECT**。预注册要求三次 fresh process 各自 T3/5/8 几何平均至少 1.25×，任何 T 至少 0.97×。初筛单独呈现，没有池化，也没有补跑挑选。

| Run | T1 | T3 | T5 | T8 | GM T3/5/8 | Gate |
|---|---:|---:|---:|---:|---:|---|
| initial | 0.982453× | 1.199745× | 1.219095× | 1.578766× | 1.321746× | PASS |
| repeat 1 | 1.023888× | 1.220317× | 1.461622× | 1.530151× | 1.397486× | PASS |
| repeat 2 | 0.939084× | 1.270615× | 1.413632× | 1.478124× | 1.384694× | REJECT |
| repeat 3 | 1.066419× | 1.256273× | 1.483812× | 1.504110× | 1.410092× | PASS |

每进程 12 cases、165 个输出/FP32 history/独立 keep-prefix 比较。四进程共 **660** 个记录逐项通过；shape、dtype、reference/candidate SHA 与前缀关系独立核对，initial state 保持不变。四份 safetensors 的每个输入 tensor 字节相同，BF16 q/k/v 与 FP32 gates/state dtype 保留。真实输入来自同一 19-token prefix 后首层的八个归档 continuation tokens，不声称为整模 greedy rollout。

每进程保留 48 formal、16 warm、8 first-post-qualification 样本，总计 192/64/32；ABBA 顺序及 median/speedup 独立重算。first 是资格后的调用，不能称冷编译。逐进程 memory、资源、release 均保存于 summary。所有资源快照 pressure=1、thermal=0，各 run swap/Swapouts 不增长；四个模型子进程 exit 0、wait/reap、strict absence 通过，两轮授权服务 lease 均恢复 T→S。

`evidence.tar.gz` 保存 initial 与三 repeats 的全部 raw、owner、lease/jobs、CPU 结果、34 个被测源、workload、模型 metadata 和 commit fe4ea8f 的复测预注册文本。188 个成员逐 size/SHA 核验通过。权重本体不入 archive；其前后 SHA 校验由每次 screen 执行，CPU audit 不重复读取大 payload。

本实验只比较同一 literal GDN helper 的串行链与一个 kernel 内的串行链。CPU 审计核查设备比较记录，不重新执行 GPU 数值运算。不证明 chunk-WY 数学等价、整模 rollback/事务正确性、长生成或 HTTP 速度；核心默认未采用。

## 审计程序格式兼容修复

CPU 审计第一次运行把 safetensors header 的全部键当作 tensor，遇到 MLX 写出的 `__metadata__: null` 后触发断言。审计程序随后允许并移除这个可选 metadata 键，再核对六个 tensor 的 shape、dtype、offset 与数据 SHA；完整审计和归档通过。第一次失败没有单独保存旧源码 SHA 或磁盘日志，此处不补造这些证据。四次 GPU 原始数据没有修改，这不是设备数值资格失败。
