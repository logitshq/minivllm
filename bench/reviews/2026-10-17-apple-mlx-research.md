# Apple MLX 与 TreeWY：本机可用方向与适用边界

2026-10-17；仅阅读官方文章、论文原文和已冻结的小型源码。未运行 GPU、升级依赖、改模型、改变状态算术或新增性能点。

**优先做存储所有权与已完成快照的释放，再测实际请求曲线。两篇资料都不能证明我们已超过外部框架。**

## Apple 官方文章的可比范围

[Local and Distributed Large LLM Inference with MLX and RDMA on Mac Studio](https://machinelearning.apple.com/research/mlx-distributed-inference)，2026-10-15；已读正文、单机段和两组原始表。

- 单机实验为 512GB Mac Studio，比较 M5 Ultra 与 M3 Ultra；Qwen3.8-27B 用 **BF16**，上下文 8192/65536、生成 128、示例 batch=1。本文没有 GLM5.3，也不是 M4 Max 对照。
- 作者将长 prefill 的改善联系到 GPU Neural Accelerators，将单流 decode 联系到带宽。它没有公开适合我们 NVFP4 的逐投影分派阈值或位一致证明。
- RDMA、双线 mesh、tensor/pipeline parallel 都是跨机器方案。文中长上下文 pipeline 收益来自机器间重叠；不能算作一台机器收益。
- `MLX_METAL_FAST_SYNCH=1` 被描述为实验性 CPU/GPU 同步功能，出现在通信命令里。文章没有给它的独立单机消融。

我们的 256GB、固定 NVFP4 checkpoint、normal-EOS HTTP workload 与上述条件不同；文章的 decode tok/s 也不是 POST→DONE 延迟。可作机制线索，不能直接并入同轮 Pareto 图。文章说使用当日 MLX，但没有给完整 runtime/source SHA，不能把其全部优化归给当前 pinned 0.32.3。

补读 [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)，2025-11-19：官方说明 MLX 通过 Metal 4 TensorOps/MPP 使用 GPU 内矩阵加速单元。其设备是基础 M5/M4 MacBook Pro；数字不能替代 Ultra/Max 或当前 checkpoint 的实测。

## TreeWY 能借什么，不能直接接什么

[TreeWY v1](https://arxiv.org/html/2608.20961v1)，2026-08-21；重点阅读 §2–5、附录 C/D/E。

论文用树形 WY 表示替代每个 draft 节点的完整 GDN 状态，接受后重建状态。§4 明确生产 kernel 按 BF16 容差验证，token stream **不逐位一致**；直接采用其算术不满足我们当前硬门。其 B200/vLLM/Qwen3.5 实验不能外推到 M5 Ultra/Qwen3.8 或 GLM。内存收益在 admission 受限时才转化为吞吐/排队改善；宽树改善 acceptance 也没有自动成为吞吐胜利。附录 D 将 deferred-write 解释标作尚未验证的归因，不能当已证实优化。

本仓库也不是该论文的 `store_all` 对照：冻结 [cache.py](../../.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/cache.py) 的 `GDNCache` 保存一个 `_snap` 加 tape，`rollback(n)` 重放接受前缀后清掉 tape；[spec_runner.py](../../.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/engine/spec_runner.py) 每个 verify block 做一次 snapshot/rollback。当前 GDN S 是 FP32，不能套论文 BF16 状态容量或宣称获得其倍数。

## 具体候选：以下均为本机推导，非论文已验证结果

| 默认关闭的候选 | 最小切点与保留合同 | 下一证据 / 边界 |
| --- | --- | --- |
| `conv_tail_owned_copy` | 已有 E026 helper，只压紧最终 BF16 三行尾部，以 uint16 复制；materialization 另有开关 | 用 evaluated `get_array_buffer_size` 验证 backing，再过完整 logits/cache/normal-EOS 门；不能仅看逻辑 shape 或归因 E023 abort |
| `observer_release_after_compare` | 新资格 harness 才能试：本步所有比较、`finish()` 与 ledger 落盘后清两臂 `calls/postrun`；继续保留 `previous_future` 的完整证据，直到其语义输入被消费/丢弃并核对 | 前后 check 数、标签、语义 key、原 dtype bits 必须一致；测 observer-exclusive backing。它是资格工具的内存改进，不是生产推理加速点 |
| `identity_keep_no_gather` | `DecodeBatch.rebuild` 中只在 keep 恰好为原顺序全部行且 L 不变时省略 gather，仍执行原 concatenate | 独立位比较、mixed-length refill/retire、alias/mutation 资格；不合并退休与入场，不改 scheduler |
| `prefill_nax_same_split` | 继续既有 E025 的 T19 三投影域，只换 partial kernel，保留 split、BF16 舍入点和原 reduce | 重构 reference 先等于原 QMM，再比每个 partial；Apple 文章不提供该 exactness 证明，失败就拒绝 |
| `fast_synch_single_host` | 后续独立 fresh-process 环境开关实验，维持 checkpoint、allocator、调度和 workload | 当前沿用 0；先核实际 runtime 路由和数值，再测单机 HTTP。只有多机命令证据，优先级低于消除已知无效存储 |

前两项有互补但不同的对象：压紧 tail 去除 core 的大 backing 引用；提前释放已经比完的观察记录，避免把旧版本带进下一次 reference prefill。不能为了省内存提前删掉尚未验证的 candidate ahead logits/state；也不能把两个开关同时打开后只归因其一。

当前不实施 TreeWY 重建、改变 GDN 精度、宽树、RDMA 或扩大 prefill chunk。改变 chunk/归约顺序可能改浮点结果；机器间 pipeline 的 8192 chunk 并非单机最优值。GLM 继续先完成加载/路由/正确性资格，不能把 Qwen 的 GDN 存储结构套过去。

## 实验顺序与记录

先完成 [E026 无权重所有权 screen 的方案](../m5_ultra_256gb/conv_tail_owned_copy/README.md)；其后一次只改变一个所有权或 gather 开关。原 E023 资源中止仍保留为未完成资格，不由 tiny screen 的通过补成 PASS。当前系统压力不允许模型实验时，只做 CPU/source 工作。

实际请求曲线仍以 **有效输出 tokens / 整组 window** 对 **POST→DONE 完整请求延迟**，并保留真实 C、排队/准入数、输出长度与所有失败点；TTFT/TPOT 和分阶段耗时作辅助。固定 offered C 时，还要看实际 admitted/running C，区分释放内存带来的准入改善与同一批量下的每步成本。不同内存预算、不同工作负载、模型精度或机器数分别成图，不合并成一个速度倍数。

相关本机证据：[NAX 分派审查](m5-nax-prefill-opportunities.md)、[E023 cache 容量审查](e023-cache-memory-budget.md)、[E026 源码与 SHA 清单](../m5_ultra_256gb/conv_tail_owned_copy/source-pins.json)。此笔记没有改 `papers.md` 或任何冻结实验文件。
