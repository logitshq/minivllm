# E014：让 drafter 的 KV 随可见窗口有界

2026-10-17；状态：运行前计划，独立原型，尚无性能或正确性结果。生产路径不变。

## 从可见信息反推存储

当前 DFlash2 的五层 attention 只读取最后 `2048 - block_size` 个上下文 KV，
但每轮把新上下文拼进完整历史。旧行已经不会再被读取，继续复制这些行的成本会随历史增长。
这个实验保持窗口内信息和计算不变，把持久物理 KV 限制在 2048 行。

[Adaptive DHiSparse v1 §3.4](https://arxiv.org/html/2610.13708v1#S3.SS4)
按 drafter 可见窗口与临时状态寿命分配存储。这里借用这个原则，先用简单的尾部连续张量，
没有实现论文的环形分页、目标 KV 分层或 target verification staging，也不借用其性能数字。

## 独立模块与正确性合同

原型位于 `bench/m5_ultra_256gb/draft_window/candidate.py`，通过
`enabled(bounded=False/True)` 临时替换 `DraftAttention.__call__`，退出恢复原方法。
没有生产配置、模型修改或全局默认开启。

新上下文仍按原尺寸、原位置、原算子完成完整 K/V 投影；裁剪发生在投影和拼接之后。
这避免改变矩阵形状和浮点累加路径。`ctx_len` 保留逻辑历史长度，RoPE 的 `start`
保持不变，attention 的 `lo` 则按当前物理 KV 长度计算。保留 2048 行而不是
`2048 - 当前 block` 行，使下一轮更小 block 仍能访问所需后缀。裁剪后调用 `mx.contiguous`。

同一 cache 轨迹不能从有界模式直接恢复完整历史：丢掉的行无法补回。基准为两个模式
各自恢复完整、独立的已准备 state，并单独推进链式步骤。此开关只属于基准作用域；
如果未来接入生产，必须另定 reset/切换合同。

E012 设备 selector 在两边始终打开。检查真实原 `_select` 收到的 hidden/logits 逐字节一致、
草稿 IDs 一致、五层全部十个 KV 保留后缀与实际 attention 可见后缀逐字节一致、逻辑长度一致。
计时不安装观察 hook，不省略 lm_head、selector 或 cache 更新。

## 固定输入与边界

目标模型 revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，DFlash2 revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`，drafter 默认 affine 4bit/group64。
使用数学 prompt 的真实 target 历史特征初始化短 KV，随后重复这些已经投影的 KV
构造长历史状态。**长长度是 synthetic cache-state replay，不是实际长 prompt 或长生成。**
重复后的旧 KV 不声称对应正确长位置 RoPE；两臂读取同一组字节，测量存储路径差异。
新增上下文投影仍使用声明的逻辑位置。

正确性长度固定为 0、31、2040、2047、2048、2049、4096、32768，block 为 3、5、8。
每点连续喂入 4 行、None、空特征、1 行、7 行，以独立轨迹比较切窗后的多步行为。
另在初始长度 31 上给入 2049 行特征，检查超过窗口的单次投影输入；三种 block 全测。
这些大特征同样来自重复短真实特征，不是新增目标历史。

另测真正 cold `ctx_k=None` 加非空真实上下文。长度 0 的普通轨迹使用显式空 KV 张量，
因为原实现不支持 `None KV + None/empty context`；不把原本未支持的调用包装成新语义。

## 计时与运行前门槛

完整 draft 计时固定历史长度 32、2048、8192、32768、131072，block 3、8，
每轮追加 4 行真实短特征。准备长缓存和裁剪种子在计时前完成；两臂都恢复自己的缓存元数据，
计时包含完整 draft 与 KV 求值。报告第一次资格检查后的计时、两轮 warmup、三组 ABBA
全部样本，不按结果选择重跑。每个样本完成后立即写入 JSON，失败保留已经完成的样本。

采用门：数值全部通过，32768 和 131072 的每一点至少 1.10×，32 和 2048 的每一点
至少 0.97×；8192 只报告。未过门保留原型和失败/负面结果，不直接接入核心。
通过也只进入独立进程和接入资格，不自动授予端到端吞吐结论。

内存只报十个物理 KV 张量的 `nbytes` 之和。`mx.contiguous` 不等于已证明真实 allocator
释放了所有底层历史 buffer，因此不会把逻辑 bytes 当成 active allocation 或可容纳会话数。
未来采用后需要独立进程测实际分配，不能用本基准中同时保留两种种子的峰值作对比。

脚本先写计划再加载模型，记录所有核心及实验源码 SHA、目标与 drafter 配置和权重 SHA、
workload SHA、设备、MLX、环境变量及每个失败。拒绝复用输出目录。
由主线程独占 GPU 队列，通过既有 guarded owner 与经授权的服务暂停/恢复 lease 运行。
本实验不调用 SpecRunner，不验证 target GDN 回滚、接受率、生成输出分布或 HTTP Pareto frontier。

## 尝试1：数值通过，收益门失败

136组数值检查全部通过。32K两点1.2698/1.1598×，131072两点2.7926/1.9206×，但2048历史
两点只有0.9102/0.9556×，8192也约0.99×，因此不采用v1。所有样本保留在
[54文件归档](../../experiments/m5-ultra-256gb/E014-draft-window-v1/)；5个资源样本正常，
进程回收和服务恢复已核实。

另发现计时归属问题：131072/B3每组第一个candidate为8.52/9.98/9.91ms，随后的第二个
为5.39/6.24/7.60ms；B8也有相同顺序差异。计时内restore可能把上一reference的大KV释放
记到candidate，尚未用allocator trace证明因果。v1照原计划报告，不把长历史倍数当实际生成结论。
v2将明确区分draft与本臂teardown，并调查每步裁剪改为有余量的分批裁剪。

## 尝试2：有余量的裁剪与正确的生命周期归属（运行前计划）

v1 的失败与样本保持原样。v2 是新的因果比较：当物理行数 **大于 4096** 时才裁到 2048，
其余轮次保留至多 4096 行，减少每轮一次额外 contiguous copy。追加瞬间可能临时达到
`4096 + incoming_rows`，完整投影输入仍不裁剪；attention 仍只读取 `2048 - block` 的后缀。
逻辑 `ctx_len` 持续增长，物理容量不决定位置。

正确性把 retained suffix 检查改为 reference 的最后 `candidate.shape[2]` 行，逐字节比较；
visible suffix、hidden/logits、IDs 和逻辑长度继续严格比较，并检查物理行数 ≤4096。
保留原有边界，新增初始长度 4095 和 4097，新增连续 block 8→3→5→8→3，
incoming 1→None→empty→1→7，在 4095/4096/4097 三个起点逐步比较。
单次 2049 行投影检查保留，再增加 4097 行，验证超过 2W 时也先按原尺寸完整投影。
本轮计划共 184 项数值检查：cold3、固定block链150、大输入6、变block链15、计时前10。

每个计时样本有三个显式阶段，两边执行相同的元数据工作和 KV 求值：

1. `prepare_ms`：恢复本臂已准备种子并同步，记录但不纳入主门。
2. `draft_ms`：完整 draft、实际 lm_head/selector、持久 KV materialize 并同步。
3. `teardown_ms`：恢复同臂种子、释放本次产出的 KV/IDs 并同步。`finally` 保证失败也清理，
   返回前检查 drafter 只持有种子；不把当前产物留给下一臂承担释放。

主指标为 `owned_lifecycle_ms = draft_ms + teardown_ms`。同时报告三个阶段、首次资格检查后的
样本、两轮 warmup、三组 ABBA 的全部原始值；不清 cache、不只清某一臂、不筛选 first/second
candidate。元数据 restore 与 framework destructor 成本如实按阶段归属，不将它们叫纯 kernel 时间。

预先固定两次 fresh process：`--cache-gib 2` 与 `--cache-gib 16`，默认参数是 2。
脚本同时将 MLX cache limit 和 LLMEngine `cache_limit_gb` 设为该值，计划记录该值。
每次内部两臂使用同一上限，两个池的所有点都保留，不看结果再选择更好的池。
原因是 reference 的 131072 行 KV 单份约 2.5 GiB，2 GiB 缓存池可能引入 allocator 抖动；
短 HTTP 负载的旧缓存池实验不能排除这里的影响。

**采用主门必须在 16 GiB 的强 reference 下通过**：全部数值一致，32K/131072 各点
owned-lifecycle ≥1.10×，32/2048 各点 ≥0.97×；8192 仍只报告。2 GiB 结果单独标为兼容性，
不能替代 16 GiB 的采用资格。脚本单进程 `advance` 只表示对应池的数值/速度门；必须结合
预注册的两次运行再决定是否独立接入。真实 allocator 数量、长生成和端到端 HTTP 仍未被本实验证明。
