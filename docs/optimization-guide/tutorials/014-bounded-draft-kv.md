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
