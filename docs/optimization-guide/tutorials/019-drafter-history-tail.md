# E019：只拼接 drafter 尚未接收的 hidden history

2026-10-17，M5 Ultra 256 GiB。阶段：设备 micro 与独立 CPU 审计已完成，**REJECT_PERFORMANCE_GATE**。150 个字节资格点通过，48 个长历史尾部点几何平均 7.247945×，但其余 72 点有 27 点低于预注册的 0.97× 门。所有代码独立放在 `bench/m5_ultra_256gb/history_tail/`，核心未启用。这里测量数组提取，没有模型加载、完整 draft 或 HTTP 速度结论。

## 一个主因

`SpecRunner.step` 当前先 `concatenate(history)`，再取 `[fed:start]`。`fed` 是 drafter 已接收的上下文长度，`start` 是目标模型已经计算的 token 数。在连续 solo 解码中，新一轮通常只缺上一轮提交的一个 feature piece，但当前路径会构造包含旧前缀的完整拼接。若特征宽度为 5×5120、dtype 为 BF16，2048 行完整前缀为 100 MiB；只缺 3 行时有用数据为 150 KiB。这是数据量推算，并非已测显存流量或加速。

候选保留完整 history 列表，从尾部反向收集 `start-fed` 行，恢复原顺序后交付。仅一个完整 piece 时直接返回同一对象；仅一个部分 piece 时只切片；跨多个 piece 时只拼接与所需区间相交的部分。没有缓存游标、新请求状态或周期清理机制。

## 明确的契约

`pending_features(history, fed, start, width=..., dtype=..., concatenate=...)` 是不导入 MLX 的独立函数。调用者提供后端的 concatenate；后续设备资格限定当前模型的 width=25600、BF16。CPU 另验证 FP16/FP32 下的同 dtype 位保持。

- history 必须是现有 list；每个 piece 都是 rank 2，宽度与 dtype 必须精确等于声明值。即使旧 piece 不被选中，也检查其 dtype，避免跳过原 full-concat 会触发的类型提升。
- 所有 piece 的总行数必须恰好等于 start，`0 <= fed <= start`，位置与宽度必须是整数且排除 bool。不能用长度更短的缓存或含未提交后缀的列表代替完整 history。
- `fed == start` 返回 None。零行 piece 可以存在，但不会调用 `[-0:]` 或空参数 concatenate。
- 不转换 dtype、不要求 contiguous、不写入数组、不增删或替换 history 内容。输入可以是有 offset 或 stride 的视图；消费者也必须保持只读。

元数据验证仍为 O(piece 数)，当前 solo 路径列表长度通常不超过 65。优化目标是取消无用数组拼接，不是消除所有 Python 元数据检查。batch 阶段可能积累更多 piece，reset 后仍需要完整前缀。

## reset 与压缩边界

`ModelRunner._sync_batch` 和请求释放会 reset drafter，随后 `ctx_len=0`。batch 缩回 solo 时，候选必须返回全部 `[0:start]`；不能因为此前已喂过相同请求就保留旧 fed 或丢弃旧 history。

现有压缩条件是 **`len(hist) > 64`**，并且发生在 `SpecRunner.step` 返回之后、append 新 features 之前。E019 不修改它：64 个旧 piece 时 append 后成为 65；下一轮在 65 个旧 piece 上取尾部后，先把完整列表压成一个 piece，再 append 新 piece。候选必须同时支持压缩前跨 chunk、压缩后跨“大前缀 + 新尾部”、以及压缩后的 fed=0 完整重建。

## CPU 资格与设备预注册

已运行11项CPU测试并通过，主线程独立审读及复跑也通过。三块长度各1..4与全部fed边界
合计544个组合；另有reset、压缩、stride/offset与非法契约检查。这是设备运行前的 CPU 资格；设备结果追加在文末。

CPU fixtures 用小字节缓冲保存任意位模式，包括 signed zero 和 NaN payload；不执行浮点算术。每个有效边界都与原表达式 `concatenate(history)[fed:start]` 比较结果字节、shape、dtype，并验证输入对象与内容不变。concat spy 独立验证单 piece 无 concat，以及多 piece 仅拼接所需行。固定小网格覆盖三块长度各 1..4 的全部 fed 边界；另覆盖空块、非零 offset/stride、fed=0、fed=start、64/65 压缩、非法元数据。

设备预注册要求由唯一 GPU owner 安排，先冻结独立 raw 输入与 core/harness source SHA。在真实 MLX 上对 BF16/width25600、N∈{19,128,512,2048}、尾部长度∈{1,3,7}，以及 fed=0 和 fed=start 做严格逐字节资格。历史分块必须包括单块、短尾块、跨块和 64/65 压缩后的形式；全部质量点通过后才能计时。两臂都保留原完整 history，两个返回结果均 eval，单独记录首调用、warmup 和 ABBA 原始样本；不能把输入生成或已有 history 压缩的成本只算给 reference。

微基准的继续门预注册为：N=512/2048 的全部尾部点几何平均至少 1.25×，其余有非空输出的点各至少 0.97×；fed=start 是 None 的行为资格，不以空操作计时制造比例。若元数据检查抵消小点收益，记录拒绝，不事后排除小点。通过仍仅说明数组片段提取有效；随后需真实 DFlash 输入、raw draft IDs、接受数、最终 token 和状态资格，才考虑默认关闭的核心开关。本实验不改变 verify 算术，也不解除已有 whole-verifier 精确性限制。

## 设备 runner 的具体冻结范围

`screen.py --out <new-directory>` 是单进程 micro，只在解析 CLI 后才导入 MLX。候选与原始 reference 的已加载路径必须指向本目录，记录 core/harness/tutorial 的 SHA，结束时重验。固定 MLX/MLX-Metal 0.32.3、Apple M5 Ultra 和 2 GiB allocator pool。源码与 CLI/矩阵测试共 17 项通过；冻结后的设备资格与计时结果如下。

输入为可复现的有限 BF16 位模式，不来自真实模型：`bf16(((arange + 19019) % 257 - 128) / 128)`。每个 N 只保存一个 `(2*N+4)*25600` 元素的一维 safetensors backing，四个文件总计约 265 MiB，写盘后重新加载作为实际资格与计时输入。历史视图由 plan 和源码重建，因此不依赖 safetensors 保存 stride。单块、短尾块、跨块使用前 N 行；offset 单块从元素 25601 开始；另外分别构造真实行 stride=2 与列 stride=2 的三块 history。64/65 边界按原 `>64` 规则预先准备，且在两臂计时外 eval。

N=19 有六种布局；N=128/512/2048 各加上两种 64/65 边界布局。N=19 的旧前缀只有 12 行，不能拆成 64/65 个非空旧块，因此这两种布局在 plan 构建时就不适用。每种布局有 tail1/3/7、reset、caught-up 五个点，总计 150 个资格点。120 个非空点全部计时：48 个长 N 尾部点进入 1.25× 几何平均门，72 个其余点逐点执行 0.97× 门；不删 single 或 reset。

资格逐点将 BF16 **reinterpret 为同宽 uint16** 后读取逻辑字节，避免用 FP32 数值转换丢掉位信息。比较输出 shape/dtype/全部字节，并在输出求值与读取后重新 hash 整个原 history；输入数组对象也必须不变。Python concat spy 只在资格阶段核对调用和选中行数，它不证明实际 Metal 内存流量。任何点失败则保存失败数组和全部已有记录，并禁止全部计时。

计时保留原 reference 的惰性 `concat→slice` 图，只 eval 最终返回值；不会提前 eval 完整 concat 来强迫复制。MLX 若已经优化掉旧前缀搬运，结果必须如实反映。每次采样重新构建表达式、完整 eval，然后释放本次结果引用；两臂都保留所有原 history/backing。每次记录 build+eval、结果释放、合计三种时长，性能门使用合计。候选返回原 piece 时只释放本次新增引用，不能释放 history 持有的对象。

每点单记首个**资格后的计时调用**，不称 cold compile；随后每臂 2 warmup、3 ABBA blocks，全部原始样本写盘。输入生成、已有 history 压缩和资格 hash 均不算给任一臂。计时结束再次核验全部 history 字节/对象、输入文件 SHA 与 source SHA。这里的 consumer 仅是输出 eval 和位读取，真实 drafter 的 fc/norm 可能因 stride 或 alias 选择不同路径，仍需后续完整 draft 资格。

## v1 结果与独立归档审计

设备执行采用预注册 commit **be12f82** 的全部七个 source pin。150 个资格点全部通过，包含 30 个返回 None 的点；120 个非空点全部计时，保留 1440 个正式样本、480 个 warmup 与 240 个资格后的首调样本。48 个 N512/N2048 尾部点几何平均为 **7.247945×**，但其余 72 点中 **27 点低于 0.97×**。本轮结论是 **REJECT_PERFORMANCE_GATE**，未剔除 single/reset，未追加重跑，核心不采用当前候选。

| Gate / example | Reference / candidate | Result |
|---|---:|---|
| N512/N2048 全部 48 个 tail 点几何平均 | 7.247945× | ≥1.25，通过此子门 |
| 其余 72 个非空点 | 27 点 <0.97× | 完整继续门失败 |
| N19 / cross_chunks / tail3 | 0.152938× | 最低的 other 点，原样保留 |
| N2048 / offset_single / reset | 0.523429× | 完整前缀重建回退 |

这些结果支持“减少所需区间之外的拼接值得继续研究”这个有限方向，但不证明所有长历史点适合替换原路径，也没有测出 Metal 实际搬运字节数。Python 元数据校验、构图、切片与释放都包含在合计时长中；目前没有因果隔离实验，不能把坏点归因到其中某一步。若后续提出分派设计，需要独立预注册与完整覆盖，不能改写本轮失败。

独立审计脚本位于 `bench/m5_ultra_256gb/history_tail_audit.py`，不导入 MLX。它直接解析四份 BF16 safetensors，逐字节核验固定输入，再按独立实现的 offset、行/列 stride、分块及压缩布局重建全部逻辑 history 与输出 SHA。150 份单点资格记录、总表、计时后 history 记录三方一致；120 点的 ABBA 顺序、原始时长、中位数与 gate 重新计算一致。另有六项记录破坏检查与 17 项 CPU 测试通过。审计中的第一次 resources 总表比较因总表额外包含 guard 生成的相邻 delta 而拒绝；确认源码后由审计独立重建 delta 再比对，原始运行记录未改。

输入固定模式每 **257 个元素**重复，因行宽 25600 与 257 互素，整行偏移 257 行也会重复。因此通过仅覆盖这些固定输入，不能排除任意长距离错行；后续真实模型 features、draft IDs、接受数、最终 token 与状态资格仍未完成。独立 CPU 回放增强了记录可信度，没有扩大输入的覆盖范围。

四个资源样本均为 AC、pressure=1、thermal=0；swap 268.81 MiB 与 Swapouts 没有增加。child PID5981 exit 0 且 wait/reap/strict-absence 通过，owner PID5928 结束，授权暂停的 PID74384 服务恢复 T→S。进程检查只覆盖已知命令与精确 owner PID，不证明所有 GUI GPU 都闲置。

[完整归档](../../experiments/m5-ultra-256gb/E019-history-tail-v1/README.md) 保留全部四份原始 tensor（约 265 MiB）、所有 raw/timing、owner/lease/jobs、审查和源码，310 个 tar 成员逐 size/SHA 复核通过。压缩包约 1.45 MiB；较高压缩率来自输入的周期性。归档先从 **be12f82** 读取七个源文件并验证运行 plan SHA，包括本教程追加结果前的版本，避免文档更新改变实测 source pin。

本实验是单进程数组提取 micro，**不进入主 throughput–latency Pareto 图**。当前没有完整 draft 或 HTTP 加速结论。
