# E019：只拼接 drafter 尚未接收的 hidden history

2026-10-17，M5 Ultra 256 GiB。阶段：静态微原型与 CPU 边界资格；没有 GPU、模型加载或速度结果。所有代码独立放在 `bench/m5_ultra_256gb/history_tail/`，核心未启用。

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
合计544个组合；另有reset、压缩、stride/offset与非法契约检查。未运行任何真实MLX测试。

CPU fixtures 用小字节缓冲保存任意位模式，包括 signed zero 和 NaN payload；不执行浮点算术。每个有效边界都与原表达式 `concatenate(history)[fed:start]` 比较结果字节、shape、dtype，并验证输入对象与内容不变。concat spy 独立验证单 piece 无 concat，以及多 piece 仅拼接所需行。固定小网格覆盖三块长度各 1..4 的全部 fed 边界；另覆盖空块、非零 offset/stride、fed=0、fed=start、64/65 压缩、非法元数据。

下一阶段由唯一 GPU owner 安排，先冻结独立 raw 输入与 core/harness source SHA。在真实 MLX 上对 BF16/width25600、N∈{19,128,512,2048}、尾部长度∈{1,3,7}，以及 fed=0 和 fed=start 做严格逐字节资格。历史分块必须包括单块、短尾块、跨块和 64/65 压缩后的形式；全部质量点通过后才能计时。两臂都保留原完整 history，两个返回结果均 eval，单独记录首调用、warmup 和 ABBA 原始样本；不能把输入生成或已有 history 压缩的成本只算给 reference。

微基准的继续门预注册为：N=512/2048 的全部尾部点几何平均至少 1.25×，其余有非空输出的点各至少 0.97×；fed=start 是 None 的行为资格，不以空操作计时制造比例。若元数据检查抵消小点收益，记录拒绝，不事后排除小点。通过仍仅说明数组片段提取有效；随后需真实 DFlash 输入、raw draft IDs、接受数、最终 token 和状态资格，才考虑默认关闭的核心开关。本实验不改变 verify 算术，也不解除已有 whole-verifier 精确性限制。
