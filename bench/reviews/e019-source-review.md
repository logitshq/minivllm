# E019 静态复核：history 尾部片段提取

2026-10-17。范围：`bench/m5_ultra_256gb/history_tail/candidate.py`、`test_cpu.py`，以及随后冻结的设备 screen。仅源码和小 JSON 审阅；没有真实 MLX import、GPU、模型加载或性能测试。核心未改。

## Candidate：有效域内没有发现尾部次序缺陷

`pending_features` 先检验完整 history 的 rank/width/dtype 和总行数，再从后向前取 `start-fed` 行，最后反转 selected pieces 恢复时间顺序。完整 piece 原对象返回、部分 piece 取负向 suffix、多个 piece 才 concatenate。零行块由 `if take` 跳过，避免 `[-0:]`；完整 list 的行数前置条件保证非空 suffix 一定取得足够行。`fed==start` 返回 None，空 history/start=0 合法；负位置、fed>start、bool 或非 int 位置早失败。

这里所有 piece 都做 dtype 检验，包括未选中的老块，是必要的当前合同：原 full-concat 可能由老块触发 dtype promotion，只检查 selected suffix 会改变结果。CPU fixtures 的任意位模式、offset/strided 映射、544 个切分/fed 组合、64/65 压缩边界、reset fed=0 与 concat spy 能支持索引和字节保持的静态结论。它们不能证明实际 MLX view/copy 行为。

## Alias 与生命周期

一个完整 tail piece 会返回同一个 `mx.array` 对象。这比旧的 full-concat→slice 更强地共享对象，当前 `DFlash2Drafter.draft` 的 fc→norm→attention 链只读 features，因而在现有消费者下可行；不能泛化成任何下游 in-place consumer 都安全。真实资格应在消费者完成后再次核对原 history 内容，不能只核 helper 返回前的输入 hash。

本轮 screen 若只测试提取，必须准确标成数组提取资格，不能借用 CPU fixture 当成完整 drafter consumer 的证明。后续真实 draft 资格应比较 hidden/logits/全 KV/IDs，明确 device_selection 与 bounded_kv 的设置，两臂相同。

不得删除已 fed 的前缀。`ModelRunner._sync_batch` / `drop_many` reset 共享 drafter 后，batch→solo 需要 fed=0 全量重建。现有压缩在 step 完成后、append 前，条件严格为 len(history)>64；E019 保留该规则。list 被替换或旧 Python 引用被释放，不等于可变数组可被改写；lazy consumer 需要继续拥有其输入 backing。

## 首要性能风险：完整 metadata 扫描

候选每次扫描全部 history 的 ndim、shape、dtype，复杂度 O(piece 数)。在 single-piece、zero-copy suffix 或小 history 场景，没有大的 GPU copy 可消除，额外 Python/C++ 属性访问可能主导成本，并触发已冻结的 0.97 门失败。batch 阶段 list 可以超过 65，不能只用 solo 上限估计此项。

本轮不修改候选或门。报告应区分提取 micro、CPU 建图、GPU eval 与完整 draft；真实 concat 是否发生也是测量问题。若实际失败，下一版可以把 producer 保证的 dtype/width/full-prefix 不变量与热路径检查分开：在模型/fixture 资格阶段独立断言，内部 helper 保留明确前置条件与必要标量检查。`ModelRunner.feats` 目前只有真实 capture 的 prefill、batch decode 与 committed verify-prefix 三类写入，可逐处证明相同模型下的格式。此方向不需要缓存游标、新请求状态或为了 3% 门堆一层抽象，但必须另行预注册与验证，不能悄悄削掉本实验的检查。

## MLX lazy 图：不要人为制造对照成本

旧路径必须保留 `full_history_reference` 的原表达式，在同样的最后输出或同样的消费者上 eval。若先 `eval(full)` 再 slice，会强制完成完整 concat，可能破坏运行时本来的消除/延后机会。两臂每次重建相同逻辑的新表达式，输入 history 的生成/压缩都在计时外且已物化，输出 eval/释放的边界一致。

concat spy 只证明 Python 提交了哪些输入，不证明 Metal 实际读写了多少字节。2048×25600×2=100 MiB 是逻辑材料量，不应写成已测节省流量。即使 lazy 图中仍有 concat 节点，也不能单凭图证明设备完整执行了它。保留参考原式后以实际 wall/active/峰值或单独诊断 trace 为证据；若运行时已消除了大部分 concat，照实记录低收益或拒绝。

真实设备资格需包括非零 offset、row stride 和 column stride，保存可重建视图的 backing 与构造说明；直接保存 view 再 load 通常只保留其逻辑值，不能证明 reload 后仍非连续。资格在每次 arm 消费结果后检查全部原 history，避免候选 alias/in-place 改动被对照掩盖。

## 预注册和 owner 边界

已冻结的 gate 应按教程逐字执行：N=512/2048 的全部 tail 点 GM≥1.25，其余非空输出点各≥0.97；fed=start 不计比例。不能因为 metadata 扫描或 zero-copy 点失败再把那些点排除。若大点只有 group GM 门而无逐点门，报告仍应列每个点和最小值，不能把 group PASS 写成每点都更快。

独立 screen 应在 CLI parse/unknown flag 检查后才 import MLX，由父任务既有 sole-owner guard/lease 启动；worker 不擅自暂停服务或创建新的计时 owner。必须保留 source/workload pins、真实 device identity、首调用/warm/全部 ABBA、资源、进程回收和服务恢复。初筛仅提取 micro，即使通过也不授予完整 draft 或端到端采用资格。

## Screen 复核状态

最终审读 `screen.py` 与 `test_screen_cpu.py` 后，结论为 **可由 sole GPU owner 启动，无阻塞接口/索引/计时缺陷**。最后新增的两函数 `code.co_filename` 来源断言和 post-timing history 对象 ID 检查也已读过。本次审阅不自行复跑 CPU tests；主线程及作者报告的 17 项 CPU PASS 与本静态结论分开记录。

150 个资格点、120 个非空计时点、48 long-tail / 72 other 分组与冻结门一致。N19 不制造 64/65 个非空 chunk，其余 N 的压缩在计时外完成。原式只在返回端 eval；每次 result 的 eval/释放属于当前 arm，输入 backing/history 全程保留。未发现跨 arm 保留 full-concat 的隐藏引用。

本机 MLX 0.32.3 stub 定义 `array.view(dtype)`；BF16→uint16 同 itemsize reinterpret 后经 NumPy 读取 C-order 逻辑字节是合适的位比较接口，没有 BF16→FP32 重编码。offset/row/column stride 从重新加载的一维 backing 重建，资格输出求值后及全部计时后再核完整 history，CPU metadata mock 没有被当成设备结果。源/输入前后 hash 和失败 JSON 路径清楚；实际 owner/reap/service 恢复仍由主线程 guard 提供。

一个非阻塞覆盖限制：synthetic flat 值每 257 elements 重复，因此相隔 257 行的完整宽 25600 行也相同。候选索引源码和 CPU 边界测试未见此类错误，但这份固定数据不能识别任意 257 行错位；本轮只宣称这些输入上的提取资格。未来可用非周期随机数据或坐标 tags 扩展，不修改这次冻结输入。真实模型 feature/consumer 与端到端资格仍在后续范围。

全部结论已直接发送 root 与实现 agent。两份被测文件、原门和核心未修改。
