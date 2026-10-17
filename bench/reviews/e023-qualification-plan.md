# E023：D 长任务的最小数值资格方案

2026-10-17，以下保留最初的只读设计。现已实现独立 `pareto_realistic/qualify_native.py`，11 项 CPU 机制测试通过；**尚未执行 GPU 数值资格**。不改任务、不放宽生产 guard。GPU 资格由 root 独占，在资源门通过后调度。

建议用一个限定于本轮配置的 paired engine 脚本：两个普通 `LLMEngine`、同一份离散到达计划、逐步比较全部 logits/cache/IDs；再用现有 HTTP qualification trace 检查真实连接的 rid 路由。无需新的调度器、插件框架或通用 replay 系统。正确性通过后才测无 observer 的 throughput–E2E latency；资格耗时不作性能点。

## 已冻结任务与实际路径

输入是 [prepared-D-v1/fixtures.json](../m5_ultra_256gb/pareto_realistic/fixtures/prepared-D-v1/fixtures.json)，receipt 中 fixture SHA 为 `6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`。8 个长度依次为 **978、886、926、921、954、935、903、958**，总计 7,461 个 prompt tokens，每请求 cap 512，正常 EOS。加载 fixture 内的 exact `prompt_ids`，不重新模板化、补齐或截断。

| 当前机制 | 源码 guard 与调用点 | D 实际会走什么 |
|---|---|---|
| 完整 decode graph / fused decode | [`CompiledDecode.__call__`](../../tutorial_code/miniengine/engine/compiled_decode.py:22)，第 27–28 行要求 `1 <= B <= 8` 和 `0 < ctx.max_len <= 256`；第 36 行还要求物理 KV 长度至少 256；第 39–40 行失败就调用 eager model | 所有 prompt 已超过 256，首 decode 即失败；CLI compile/fuse ON 不等于实际启用了该路径 |
| grouped fresh prefill 的设备与形状域 | [`prefill_batch_backend_supported`](../../tutorial_code/miniengine/model/layers.py:133)，第 141–142 行在 M5 上限定 `2 <= B <= 32`、`T <= 512`、`B*T <= 4096` | 所有 D prompt 超过 512，必定不满足完整 fresh grouped 路线 |
| runner 能否形成 prefill group | [`ModelRunner.run`](../../tutorial_code/miniengine/engine/model_runner.py:111)，第 125 行要求 `computed == 0` 且本次 `n == prompt_len`；第 126 行按**相同长度**分组；第 128 行再次调用形状 guard | 8 个 prompt 长度各不相同；小 budget 的 partial prefill 也不会因 chunk 长度相同就自动成组。重复 fixture 即使同长，仍被上面的 T guard 拦住 |
| compiled GDN gates | [`GatedDeltaNet._decode_gates`](../../tutorial_code/miniengine/model/gdn.py:87)，第 91 行只接受 `x.shape == (1,5120)`，并检查实际 backend | C1，以及 C8 最后 drain 到 B1 时仍可能实际启用；没有 256 长度条件 |
| norm reuse | [`GatedDeltaNet._norm_reuse_eligible`](../../tutorial_code/miniengine/model/gdn.py:120) 与 `_norm_reuse_initial_eligible` | 本轮若明确 OFF 就不启用；不得用历史失败/未采用实验暗中替换 |
| lookahead | [`ModelRunner.run`](../../tutorial_code/miniengine/engine/model_runner.py:64)，第 66–68 行消费 `_ahead`，第 82–102 行产生下一步；第 170 行后的 `_sync_batch` 与 `drop_many` 失效处理 | 长任务仍实际使用。必须检验新请求进入、row compaction、EOS 以及 cap 时不会消费旧 key/旧 cache |

因此 D 的诚实 baseline 是“长上下文 eager decode + 当前 gates/allocator/lookahead/调度策略”。不能把 compile ON/OFF 的相同 fallback 数值通过说成“长上下文 fused graph 已验证”，也不能把 E010 短任务优势直接搬到 D。

## 旧资格能直接借用的部分

| 现有文件 | 具体可复用部分 | 不能直接继承的结论 |
|---|---|---|
| [`prefill_nax/screen.py`](../m5_ultra_256gb/prefill_nax/screen.py) | `arrays(cache)` 遍历每层两份单请求状态；`exact(a,b)` 先核 shape/dtype，再用 BF16→uint16、FP32→uint32 view 做逐位比较；原脚本逐层输出、cache、最终 logits 加三步 decode 的结构 | 原 full-model cases 最大为 B2/T512、B8/T128 等，且是固定 19 IDs 的旋转构造；不是这 8 份真实长 prompt |
| [`prefill_integration/qualify.py`](../m5_ultra_256gb/prefill_integration/qualify.py) | `integrated(active=False)` 只暂时使 runner grouping 判定返回 False，恢复时还原；candidate 是实际 core，无旧 candidate monkeypatch | 现有脚本在 import 时读 argv 并导 GPU，不能作为新 CLI 顶层依赖；只抽取小 contextmanager |
| [`prefill_nax/boundaries.py`](../m5_ultra_256gb/prefill_nax/boundaries.py) | 同模型独立 caches，检查完整 logits 与每层 state；失败保存后立即停止 | 22 个集成边界已 PASS，但 T 上界 512；不是 886–978 的资格 |
| [`prefill_nax/lifecycle.py`](../m5_ultra_256gb/prefill_nax/lifecycle.py) | 每 step 保存 delivered tuples、`batch.req_ids/lens`、所有 staging 的 offsets/cache、结束后空状态断言 | 现有 C64 9 步、mixed 7 步，短 cap 和循环 prompt。它未逐步检查全部 logits，也未保存 `_ahead` 的完整数组/键 |
| [`decode_batch/qualify.py`](../m5_ultra_256gb/decode_batch/qualify.py) | `arrays(layers)`、`copies(layers)`、uint view exact；mixed positions、权重不变、`schedule`/`staggered` 与空 cleanup | 当前短域大量位置围绕 19/255/256；`digest()` 经 FP32 cast 的历史写法不要作为新 BF16 bit-level 签名实现 |
| [`retire_batch/qualify.py`](../m5_ultra_256gb/retire_batch/qualify.py) | `signature()` 的原 dtype view；独立保存 outputs/batch/lens/requests 和 survivor state，真实退休日志 | 不要再次动态换成旧 Engine/Runner 类；E023 用当前 core，比较的是不同执行配置下的**同一调度** |
| [`pareto_realistic/native_server.py`](../m5_ultra_256gb/pareto_realistic/native_server.py) | 已冻结的 `BatchWorker`、`observe_runner`、final SSE raw/committed IDs、actual CLI identity | 18 项 CPU mocks 是 routing/liveness 合同，不证明 GPU batch 数学。HTTP qualification trace 没有 full cache/logit 内容 |

这些函数的模式足够，不必把含有模型 import/脚本顶层执行的整个历史文件 import 进新 CLI。新脚本顶层保持 stdlib，`--help`/未知参数在 MLX import 之前结束。

## 第一层：同调度的完整长 rollout

最小新文件可叫 `pareto_realistic/qualify_native.py`，只接 fixture、冻结 source manifest、配置、out。先明确所有服务器容量参数、allocator、gates、compile/fuse、norm、EOS，来源与后续 HTTP identity 一致。`max_len` 至少覆盖 1,490；本轮若冻结 2,048 就两臂都用 2,048，不能给 reference 更小预算。

**选择两个正常构造的 engine，单 process、串行 step。** reference 全部 eager，gates/norm OFF，grouping OFF；candidate 是将要测量的实际配置。两个 engine 各自加载同一固定 checkpoint，拥有独立 scheduler/requests/staging/batch。这样无需手工构造半个 engine、复制 nn.Module 或切换同一个模型的私有参数。两份权重增加内存，只用于一次不计时资格；先经过既有资源门，若容量不允许就停止报告，不临时更换精度或卸载用户服务。

reference 在**每个 step 前**清掉自己的 `_ahead`，使本步必须重新从 committed state 算；candidate 保留原 lookahead。reference 仍会在原 runner 中构造下一步 lookahead，清除只禁止下一步复用，不删除任何已提交 cache。这是一行 bench-only reference 控制，不修改 core，也不把它拿来当性能 baseline。每次清除前先完成上一轮比较和必要求值，避免对仍待比较的数组失去引用。

两臂按下列固定有限序列运行，全部 `ignore_eos=False`、greedy；不强行补足 cap：

1. **C1：8 个 D fixture 各一次、顺序 D01…D08、cap512。** 完整 drain 后才加下一份，所有长 token 都比较。
2. **C8：固定 16 jobs，D01…D08 重复两轮、cap512。** 初始提交前 8 个；在每个 reference/candidate 已一致的退休边界，按固定 FIFO 向两臂同时补下一 job，直到 16 个全部提交并 drain。这是离散 step 的 closed-loop trace，不依赖 wall clock。
3. **短生命周期 probe，仍用真实 D prompt：** 初始 8 个 cap `[1,2,3,5,8,13,21,34]`，在第 2 个 step 前再加 D01…D04，cap `[2,5,9,17]`，正常 EOS。完全 drain 后，复用已退休首 rid、改用另一份 fixture、cap3 再跑一次。该 probe 检查 pending、mixed decode/prefill、不同退休、清空后 reuse；它明确是诊断形状，不替换 cap512 性能任务。

每步先比较两臂真实 `SchedulerOutput`：有序 decode rids、prefill `(rid,n)`、finished_ids，以及请求 `num_computed_tokens`、输出计数、状态与 slot。只有一致后才继续；candidate 若提前出错/EOS，立即记录数值或调度失败，不能把它的新路径作为下一步 reference 输入。

输入 token 用每臂各自的 greedy 输出推进；每一步要求它们一致。不能在 candidate 上强行喂 reference 的 continuation 以掩盖分歧。C8 与 C1 之间**不**要求 raw IDs 相同；GPU reduction 随 batch shape 改变可能改变舍入，正确的对照是同 B、同 row order、同 history、同 arrival/retirement trace 的 reference。

### 每一步确实比较什么

- 输出 `(rid,token,done)` 全部相同，rid→fixture 对应不变；EOS 的 raw ID 包含末尾 EOS，committed IDs/usage 仅剔除这个终止 ID，不剔除普通 reasoning/control tokens。
- 每次完整 model call 的全部 vocabulary logits，而不是 argmax 或选中的 top-k。prefill 包括分块调用的 last-only logits；decode 包括本步以及真实 ahead 产生的 logits。
- 每个活跃 batch 所有 64 层、128 数组的完整 shape/dtype/bit pattern，包含 attention padded storage；每个 staging cache 的 offset、K/V 或 conv/S、record/tape 状态；`batch.req_ids/lens` 与 `requests` 的 metadata 单独对齐。
- `_ahead` 的 key、预测 token IDs、全部独立 layers：存在性与两臂 key 对齐，保存状态不允许回写 committed array handles。reference 丢弃 ahead 后再算的 logits/cache 必须等于 candidate 实际复用的结果。
- 每次退休后 rid 从 requests/staging/batch 消失；survivors 的原有顺序、history、lens 对齐。所有任务结束后 `has_work=False`、requests/staging/feats 空、batch size 0、`_ahead is None`。

**最小 observer：两个局部 call wrapper + 一个 step ledger。** 包住每臂 `runner.model` 的 prefill call 和 `runner.decode_model`，仅记录输入定位信息、返回 logits 引用；其他 model 属性代理到原对象。prefill key 为 `(rid,start,n)`，decode key 为 `(ordered rids,positions,input IDs)`。前者可从 staging cache identity 与进入该步的请求 metadata 得到，后者取当前 batch order/ctx；不要依赖“第几次 Python call”配对。

这个 key 很重要：candidate 本步可能消费上一步算好的 ahead，而 reference 本步重算；两者 Python call 数不一定相同。保留每臂当前 step 与一个未来 key 的 logits，按**语义输入**比较，消费或失效后释放。已被退休/换 batch 取消的未来结果明确标 discarded，不当作输出；不得把“callback 少一次”解释成缺失或凭空复制 logits。wrapper 不能给 candidate 的 `ForwardContext` 增加 record/capture 字段，否则会禁用正在验证的分支。

bit comparison 复用 `prefill_nax.screen.exact` 的 unsigned view 思路，批量在 GPU 做 `array_equal`，只取小布尔向量回 CPU；同时检查所有 logits/state finite、数组数量、shape/dtype。**不在每 step 将整个 2 GiB cache 传 CPU 算 SHA**，也不把 BF16 cast 为 FP32后声称原始字节相同。JSONL 逐步保存每层命名/shape/dtype/bit_equal、调度 metadata、IDs、ahead key；checkpoint 或首次失败才保存指定原 dtype safetensors 与 SHA。结束清引用，模型/源 pins 前后复核沿用 root 已有 owner 流程。

任何失败立即保留 partial ledger、失败左右数组与 exact input key，写非 PASS complete 并退出。不得临时改 allclose 容差、删任务、改到达顺序或换 C1 oracle；若原路径不同数学确实非逐位，需要另立具体合同和方案后重做，当前点不进图。

## 第二层：HTTP route 与真实 admission

复用冻结 server 的 `--qualification-trace` 和新 client，单独开资格进程；正式性能进程省略此 flag。最小 live probe 为 C1 的 8 个请求和 C8 的 16-job closed loop，仍用原 D/cap512。保存每个 HTTP 请求的 fixture、POST/DONE、`benchmark_request_id`、prompt/raw/committed IDs、finish/usage，与 server before/after step 和 actual runner batch trace join。

硬门：每个 HTTP rid 恰有一条对应完整输出；不存在跨 rid 重复/漏 token；trace 的输出逐 rid 合并后等于 SSE raw IDs；每个 done 之后不再有该 rid 输出；usage/EOS 与冻结 tokenization 一致；实际 trace 出现至少一个多 rid decode，并记录最大实际 B；server 最终排空。C=8 只是 outstanding 请求数，若始终 B1 则不能标已验证 C8 GPU batching。

若需要声称“**这一条 live HTTP 调度轨迹**也有 full numeric proof”，最小追加是把 trace 中每 step 的新增 rid/prompt/cap 转成离散 `arrivals_before_step`，交给上面的 paired harness replay，并断言 replay 的实际 SchedulerOutput 和 delivered IDs 与录制一致。只录申请时刻或假定八请求同时进入不够。server trace 尚未提供缓存数值，不能直接称它做了 full-state 检查。

无需先给所有无 observer 性能请求逐条 replay；固定路线资格证明所覆盖的执行机制，正式 client 仍逐请求检查 route/count/完整输出并保留全部失败。HTTP arrival 导致的 batch 差异不与 C1答案强制 exact；需要更强逐条主张时就做上述实际 trace replay，增加的花费如实列出。

## 实现与运行花费上界

最小新增内容：一个窄用途 CLI、两个 call wrappers、状态遍历/bit comparator、三份固定 workload loops、JSONL receipt；加少量 CPU mock corruption tests 验证错误 rid/shape/bit/遗漏 ahead/到达顺序会拒绝。预计约 250–400 行 runner 和小测试文件，是实现预算，不是已写代码；不新增生产开关、调度器或 daemon。

固定核心资格每臂 **37 个请求，最多 12,411 个 raw 输出 IDs**：C1 4,096、C8 8,192、短 probe 123；双臂最多 24,822。EOS 可使其更少，但保留实际覆盖，不补样本。首层真实长 prefill 的重复次数明确随这 37 请求发生；不另跑 22 个旧 synthetic cases。

同时最多两份 target 权重、两臂 committed caches、两臂一个 ahead、当前 logits；不保留全 rollout 的 tensor history。按当前 config，单 row 的 GDN S 为 48×48×128×128×4 = 144 MiB，16 层 attention 在 L1536 的 K/V 合计 96 MiB，conv 约 2.8125 MiB；一套 C8 state 约 1.90 GiB，四套 committed/ahead 约 7.59 GiB，再加 staging/瞬时重排与两份权重。**这是结构推算，不能替代实际 peak/RSS/pressure 门**；不以“256 GiB 足够”跳过 owner 检查。

不承诺墙钟速度。建议本阶段 owner 固定 **30 分钟绝对 deadline**（含双加载、对照与清理，具体应与现 owner lease 配置同时冻结），到时保存 timeout 并失败退出，不能只交成功前缀。每 case 另设保守 step 上界 `sum(prompt_tokens + caps) + 1`，同时拒绝有 work 但无 computed/output 进展的 step；此上界宽松但有限，不用 token/s 推算超时。

live HTTP route probe另有 24 请求/最多 12,288 raw IDs；若还 replay 它的完整 actual trace，paired 核心工作再加每臂 24 请求/12,288 raw IDs。将这项强合同与成本明确写入 prereg，不把它藏在“轻量 audit”里。模型权重 SHA 复用既有 owner pins 的前后校验，不在每 step 或每 fixture 重读大权重。

## 256 为什么存在，以及后续可做的独立模块

**256 在当前实现中是静态 attention shape，也是已验证范围；不是模型数学上的最大上下文。** `CompiledDecode.__call__` 第 28 行把 used length 限在 256，第 36 行要求 KV backing 至少 256；其 compiled closure 第 66–67 行实际构造 `ForwardContext(... max_len=256)`。[`Attention._decode`](../../tutorial_code/miniengine/model/attention.py:79) 第 87–89 行用 `ctx.max_len` 截 K/V 并构造 mask。只删除 eligibility 的 256 条件，会让长 history 仍只读前 256 个 keys，产生错误；不能这样“扩容”。

固定 `max_len` 让不同 decode positions 共用相同 K/V view/mask shape，避免每个 token 的 L 变化产生新 graph shape。旧文档与资格明确只保证短域，检查 254/255/256 边界、fallback、参数替换和插入/删除；源码没有给出“256 是显存/寄存器极限”或“超过它数学不成立”的证据，不编造这种原因。选哪个更大 bound 的编译、内存和速度收益必须测量。

**可另开默认 OFF 的 long-context graph 模块，首版只做一个固定上限 1536、B1…8。** D 的 `prompt+cap` 最大 1490，而 DecodeBatch 已按 256 对齐给到 1536，所以该候选不需要改 prompt 或 cache 的逻辑长度，也不需要通用 bucket manager。新模块必须同时固定 closure 的 attention L、资格上限、KV backing 下限，若未来有多个 bound 则 bound 必须进入编译函数 key；保留原 256 模块与所有不匹配 fallback。

先做 graph-only，再决定是否组合原 fuse；不要一并扩大 B、改量化、换 attention kernel。长 mask 会改变 SDPA dispatch/reduction 或编译图，数学上屏蔽 padding 并不自动保证 bit-exact。新资格应覆盖 D 的真实 mixed positions、L1024/L1280 边界、1535 最后支持位置与 1536 首个 fallback、KV backing 不足、batch/metadata fallback、参数替换、ahead 失效和完整 rollout。静态 bound 让 padded attention 扫描更多列，尤其 near887 时可能更慢；cold compile 与 warm 三 fresh 完整 HTTP 必须独立保存，不仅看 graph micro。

**grouped prefill 的 T512/B*T4096 也是本机 NAX 的经验资格边界，不是普适数学上限。** 数学风险来自 [`prefill_linear`](../../tutorial_code/miniengine/model/layers.py:145) 第 148–153 行解释的 BF16 splitK 舍入和 M/tile 路线；它用 broadcast weights 保留每行 M=T。新长 T 即使满足部分不等式，也没有被旧 full-state 实验覆盖。

单纯把 T guard 扩至 1024 对当前 8 个各异长度的首轮请求未必有益。把 partial chunks 硬塞入现 `_prefill_batch` 更不安全：当前 GDN grouped 实现用 fresh convolution/初始 S，attention grouped RoPE 从 0 开始，均不是任意 continuing cache 的接口。支持 mixed-length/continuing chunk 将要求显式 per-row offset、conv/S、attention history 与 mask，复杂度明显高于上述 single-bound decode graph。优先保留 D fallback baseline，测试 bounded long decode 的真实收益，再按测得瓶颈决定是否值得做 chunk prefill。

本文件不授予任何新数值/性能资格。C1/C8 通过后，C16/32/64 仍须逐级验证资源和相应形状，不能由 C8 外推。主图保持 actual throughput 对 actual E2E latency，正确性始终是入图门槛。
