# D512 C16/C32/C64：最小扩展矩阵与资源预算

2026-10-17，CPU 源码与既有归档审查，**候选方案，未注册、未执行高并发 D512**。不改冻结实现、输入或旧结果；没有导入 MLX、运行测试、访问服务或读取大权重。目标仍是通过正确性硬门后，画 **实际 committed output tok/s 对 POST→DONE p50**，保留排队延迟。

## 已知事实与不能继承的结论

固定模型为 Qwen3.8-27B NVFP4 revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`；八份 D 输入为 978/886/926/921/954/935/903/958 token，normal EOS、输出 cap512、无 prefix cache。现有 [native D512 基线](../../docs/experiments/m5-ultra-256gb/E023-native-formal-v1/README.md) 仅 C1/C8：各一 fresh process、8 warm、64 formal；分别 45.372/167.399 tok/s，POST→DONE p50 11.283/24.432 s。全部触 cap，未覆盖实际早 EOS。这些是已测点，不是高 C 的预测。

历史 **19 输入 / cap32** 的 [独立完整延迟重算](pareto-v3-v4-audit.md#verified-v3-numerical-values) 提供容量与调度探索经验：

| 历史 native C | POST→DONE p50，s | 总输出 tok/s |
|---:|---:|---:|
| 8 | 1.662854 | 153.890592 |
| 16 | 3.464325 | 147.720561 |
| 32 | 5.286723 | 190.444616 |
| 64 | 10.718857 | 189.827696 |

C16 被 C8 支配；C64 中位数被 C32 支配。C32/C64 吞吐只差约 0.3%，不能声称显著差异，更不能外推 D512。旧组是有限 C-request burst，且短 prompt 可以走 E010 grouped prefill，D 的完整 prompt 不满足其 T≤512 域。[E010 归档](../../docs/experiments/m5-ultra-256gb/prefill-nax-v2/README.md) 的约 43.43 GiB MLX active peak 也不是高 C 长任务的预算上界。旧 cap32 与新 D512 分图，不拼成一条 frontier。

## 四个不同的“并发/容量”量

| 量 | 实际含义 | 新基线的冻结选择/证据 |
|---|---|---|
| HTTP C | 已发 POST、尚未 DONE 的请求上限 | 由原始请求 start/DONE 区间重建实际 outstanding；client 不得偷偷限到更小 C |
| `max_seqs` | native running 槽，包含尚在 prefill 的请求；waiting 可超过槽数 | 主矩阵设为 C。若另测 C64/max_seqs8，应单独命名 queued configuration，不能叫 B64 |
| actual decode B | 本步 prompt 已完成并进入 decode 的行数 | 保存 qualification 的 B histogram、混合 prefill/decode 和 refill/retirement；不能由 C 或槽数推断 B=C |
| `max_batched_tokens` | 每步 decode 每行先扣 1，再给 continuing/fresh prefill 的预算 | 第一轮保持 **2048**，`max_len=2048`、allocator cache2GiB；必须 ≥max_seqs，不随 C 自动乘倍 |

依据：[Scheduler.schedule](../../tutorial_code/miniengine/engine/scheduler.py:48)、[ModelRunner.run](../../tutorial_code/miniengine/engine/model_runner.py:31)、[native CLI](../m5_ultra_256gb/pareto_realistic/native_server.py:329)。C16/32/64 饱和 decode 时，2048 预算分别剩 2032/2016/1984 token 给 prefill；这些是 token 数，不是内存配额。FCFS 逐步 admission、分块、客户端/服务器等待及 drain 都保留在延迟中。D 完整 fresh prompt 不走现 grouped-prefill 域；分块 continuing prefill 也不因此变成 grouped fresh prefill。

## 真实数组预算：不是把 256 GiB 当可全部分配的 cache

由固定 64 层（48 GDN、16 attention）和原 dtype 推导，详见 [cache/backing 审查](e023-cache-memory-budget.md)：

- 每个 decode 行的 GDN S：`48 × [48,128,128] × FP32` = **144 MiB**。
- 逻辑 conv tail：`48 × [3,10240] × BF16` = **2.8125 MiB**。
- K+V：`16 × 2 × [4,L,256] × BF16` = **L/16 MiB**。
- D 的 staging KV 以 256 对齐，完整 prefill 容量1024；`prompt+cap` 的 decode backing 全部对齐到 **L1536**。`max_len2048` 不代表每行已经分配2048；rebuild 会保留尚存 batch 的更大 L，必须从实际数组记录容量。

以下为**命名数组的逻辑 payload 算术**，GiB=2³⁰ B。后一列只展示两个比较模型各保留 committed+ahead 的分量，不是完整资格器峰值或可用容量预测。

| actual B | 一份 L1536 全 cache | 一模型 committed+ahead 两份 | 两模型各两份 |
|---:|---:|---:|---:|
| 8 | 1.89697 GiB | 3.79395 GiB | 7.58789 GiB |
| 16 | 3.79395 GiB | 7.58789 GiB | 15.17578 GiB |
| 32 | 7.58789 GiB | 15.17578 GiB | 30.35156 GiB |
| 64 | 15.17578 GiB | 30.35156 GiB | 60.70313 GiB |

单行 L2048 是274.8125 MiB，而非242.8125 MiB；若实际出现更大 backing，预算必须按实际 L 重算。S 在 L1536 单份状态中占144/242.8125 MiB，优化 KV 并不会消除这部分。

必须另计或实测以下部分：两份/一份真实模型及参数缓存、staging、旧/新 state 版本、rebuild 的 keep gather/pad/concat、retirement compaction、ahead、prefill workspace、延迟图、Metal/runtime 和 allocator。快照 view 可能共享 buffer，不能按 Python handle 个数加总；旧版本被替换后，observer 又可能使多个版本同时存活。一个 source-level `old+gather+new` 例子在同 B/L 下可达约三份状态，**不能据此断言三份一定同时物化，也不能把它当全部内存上界**。

普通 prefill 的三行 conv tail 可保留 `(T+3)×10240×2` 的整个 parent。按全48层 payload，T886–978 对应 **833.4375–919.6875 MiB/请求**，远大于2.8125 MiB逻辑尾片；这是可达 backing 路径，不是实测全模型物理减少。2048 token budget 限制每步 prefill 工作，但不能把所有64槽都算成64个完整同驻 staging，也不能假定只会留当前两个 parent：按 trace 的真实分块、旧引用与 eval 时点盘点。ordinary decode 的四行 parent 比三行 tail 另多0.9375 MiB/行。

实测提醒：[E028 两模型 B≤8 数值资格](../../docs/experiments/m5-ultra-256gb/E028-identity-refill-model-v1/README.md#resources-and-original-service-restoration) 的 MLX peak 已达 **69.648 GiB**，包括模型和 observers，不能按表中的7.59 GiB预测。暂停大 oMLX 服务仍会保留驻留内存；正式大矩阵沿已授权 cold-stop/reap/restore 合同执行，不能把 SIGSTOP 当释放。`set_cache_limit(2GiB)` 只限制可回收 allocator cache，不是 live-array、进程物理或操作系统硬上限。

资源递进门：C16先做独立一模型实际 admission/retire capture；记录 MLX active/peak/cache、实际 unique backing（只在诊断 eval 边界）、self/current+peak physical footprint、系统 physical/free/pressure/Swapins/Swapouts。通过后再进入两模型 replay；其 observer 开销单独标记。沿用 pressure1、AC/thermal、原 swap/外部进程门和 bounded owner/reap/recovery；不能因 C 增大放松。每个新 C 在运行前冻结具体预算/timeout，不根据失败后现场提额。资源拒绝保留证据并暂停更高 C；不给失败配置补零吞吐，也不承诺256 GiB一定可运行C64。

## 最小下一轮矩阵

先只扩 **target-only baseline**，E024/E028/E026保持OFF；每个C完成资源与正确性资格，才测对应formal。模型、tokenizer/template、original messages、精度、normal EOS/cap、无prefix cache和已校准usage语义不变。

| 新 HTTP C | native max_seqs | token budget | qualification 候选请求数 | formal请求数 N | 初始发布策略 |
|---:|---:|---:|---:|---:|---|
| 16 | 16 | 2048 | 32（2C） | 128（8C） | 第一优先；通过后先交该真实点 |
| 32 | 32 | 2048 | 64（2C） | 256（8C） | C16资源/资格完成后 |
| 64 | 64 | 2048 | 128（2C） | 512（8C） | C32资源/资格完成后 |

2C仅是用于观察 refill 的**最低候选队列**，不是预先授予资格：若实际历史未覆盖新增 decode B、跨1023/1024路线、部分退休与补位，就明确缺口并另注册补充数值诊断。请求自然EOS不足时不能强迫忽略EOS；另加生命周期诊断不混入formal。native qualification复用真实 HTTP trace→严格join→两独立模型同调度 replay，完整 logits、所有64层state、ahead、RID/原始tokens/normalEOS检查。新B及9、17、33等实际经过的边界必须被对应历史或补充诊断覆盖；不能套C1答案作为所有batch的token oracle。external逐请求核原prompt IDs、route/raw输出与计数binding；这不等于跨引擎全state/质量相等。

正式每个点建议3 fresh processes，轮换engine顺序；每进程固定8 warm覆盖fixtures，完全drain后开始formal。初始C个POST放行，DONE后立即补下一固定job，直至N全完，禁止显式等wave或重试隐藏错误。记录资格中的actual B分布和正式真实HTTP outstanding；正式关闭observer。若另加高C形状预热，应给三引擎/相关重测点同一新协议，不能只为某个引擎补热后与旧8-warm点无标记比较。

**N=max(64,8C) 是新的高并发协议扩展，必须有新 workload/protocol ID。** C64只有N64时是一波请求，几乎没有持续补位，启动与末尾drain可能主导；不能把这一波叫稳态。128/256/512仍是有界closed-loop实验，8波等价工作量也不保证达到稳态。若需所有 C 共用固定N512以严格比较有限样本边界，则登记另一套矩阵并重跑对应C1/C8；不能把旧N64样本池化进去。已有C1/C8可在独立标记的历史panel展示，不作为新protocol的配对重复。

每点仍算 `sum(actual committed tokens)/(last DONE-first POST)`，横轴为所有请求 `median(DONE-POST)`；p95用预先固定的nearest-rank。计入服务器排队，client等待并发槽发生于POST之前需明确。保留全输出长度/EOS比例、每进程散点和全部失败，不裁去warm后首批或末尾尾部以美化吞吐。延迟/吞吐都不作数值预测。deadline需随新N预注册为有限值，不能直接使用旧N64的超时后再静默延长。

## 三个优化与最大 C 的关系

- **E024**：[route candidate](../m5_ultra_256gb/route_decode_graph/candidate.py:68) 仅实际 **B1..8、used257..1536、KV backing≥1536**；B>8走eager，即使HTTP C64。只有起步/退场阶段B≤8可能命中。现[forward-only gain](../../docs/experiments/m5-ultra-256gb/E024-route-timing-v1/README.md)来自B1/B8，不是HTTP收益；不能乘到C16/32/64。不要直接删B guard；扩B是独立默认OFF数值实验，需原route/flags/全状态和真实调度资格。
- **E028**：[identity refill](../m5_ultra_256gb/identity_refill/candidate.py:34)没有硬B8算法guard，但只在old/new非空、keep精确identity且所有KV容量不变时跳过旧行gather；concat与实际新状态仍存在。潜在避免的复制字节随survivor B增大，但次数取决于异步retire/refill，不能外推线性收益。现全模型资格只有C1/C8/lifecycle共12次eligible refills；高C、真实HTTP与性能仍须分别验证。它不消除纯retirement gather或容量增长fallback。
- **E026 conv ownership**：作用是缩短普通prefill tail对大parent的保留，收益关联实际T/分块/同时staging与eval生命周期，不直接等于C倍。已有[tiny/单真实层](../../docs/experiments/m5-ultra-256gb/E026-conv-tail-layer-v2/README.md)存储证据；[staged integration](../m5_ultra_256gb/conv_tail_owned_copy/staged_integration.py)仍默认OFF，复用原runner eval，无每层eval。copy与parent可共存到原eval返回，不能宣称降低build/物理峰值；它也未消除占大头的FP32 S。完整engine与HTTP资格未完成，不能当作C64容量已获保障。

## 最少代码接点，不另造框架

现 [native_server CLI](../m5_ultra_256gb/pareto_realistic/native_server.py:329) 已能显式传max_seqs/token budget；[http_join](../m5_ultra_256gb/pareto_realistic/http_join.py:113) 基础trace推导也不局限C8。但冻结 [http_qualification](../m5_ultra_256gb/pareto_realistic/http_qualification.py:67)、[http_replay](../m5_ultra_256gb/pareto_realistic/http_replay.py:25)、数值audit及 [native_formal](../m5_ultra_256gb/pareto_realistic/native_formal.py:120) 的C1/C8、8/16或N64约束确实需要**新的薄适配/明确计划域**，不是改命令就已支持。复用原Comparator/Observer、raw parser、cleanup和资源恢复；保留旧源、旧失败及完整数字类型/状态门，只扩实际C/N与对应receipt范围。

oMLX按其官方target-only配置逐C资格，记录实际admission限制，不把native max_seqs参数强塞给它。Lithos继续原官方 `/api/chat` 排队路径：它的HTTP C可大于1而GPU仍单请求FIFO；所有排队时间必须进入POST→DONE，不能改OpenAI429路径或另起C份模型冒充batch。两者当前D512 binder也只接受C1/C8，需新C route/count资格后才能接formal。先交三引擎合法C16点，再往32/64递进；任何不支持/资源失败写coverage表，不画伪造性能点。

源身份依据沿现冻结记录：E023 numeric plan `21ba52e4…547fb`、core cache `056a3bf3…57a791`、runner `14fd964c…b7b1`、E024 route candidate `6531be25…525c3`。本审查仅读小源码/Markdown/计划和算术，未重新认证运行时或模型payload，也不授予任何新正确性/资源/性能资格。
