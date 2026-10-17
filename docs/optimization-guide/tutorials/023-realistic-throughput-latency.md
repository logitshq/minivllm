# E023 · 正确性过门后，测长生成的 throughput–latency Pareto

2026-10-17。状态：**CPU 输入与测量入口准备；GPU 未运行，无新性能点。**

主优化目标始终是同一模型、同一任务集上的整机输出吞吐与完整请求延迟。
正确性不参与权衡：候选先通过数值、状态、请求隔离和计数检查，才允许进入图。
吞吐分子为实际提交的 target tokens；延迟从客户端 POST 起点到 SSE DONE，
包含排队、prefill、decode 和交付。TTFT 和 p95 是配套指标。

## 为什么扩展负载

已有并发图覆盖19输入/32输出；E020又补了同轮官方MTP、DSpark的C1对照。
这两组结果都保留，但32-token输出容易受请求边界支配，不能证明长生成收益。
E020的native shim还会排空一个请求才取下一个；只增大客户端C并不等于测到batch。

因此新实验先固定八份有不同需求的编码任务：TTL LRU、DAG调度、事件窗口、
增量frame解析、重试预算、区间预约、版本化KV事务和滑动限流。输入内容在
任何输出/性能观察之前写定；模型继续使用同一个Qwen3.8-27B NVFP4 revision。

| 任务 | 模板后输入 tokens | 输出上限 |
|---|---:|---:|
| D01 TTL LRU | 978 | 512 |
| D02 DAG scheduler | 886 | 512 |
| D03 event rollup | 926 | 512 |
| D04 frame parser | 921 | 512 |
| D05 retry budget | 954 | 512 |
| D06 interval booking | 935 | 512 |
| D07 versioned store | 903 | 512 |
| D08 sliding limit | 958 | 512 |

这些是自然886–978长度，明确取代早期候选方案的1024±32目标桶；没有人为padding。
output cap不是强制长度；正常EOS会结束请求，短回答仍保留。生成代码是否解决任务
是模型能力问题，不把CPU输入准备称为答案正确性检查。优化正确性还须验证引擎
在固定调度下保留应有的logits、状态和输出，不能把C1答案硬套到不同batch数学。

冻结输入的[源码、精确IDs及receipt](../../../bench/m5_ultra_256gb/pareto_realistic/README.md)
位于仓库。两个tokenizer API结果一致、decode可逆、七份输入依赖SHA一致；
fixture SHA为`6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`。

## 最小测量入口

服务端仍只有一个GPU worker，每step边界接收新请求，由原scheduler控制实际batch；
每个rid独立交付与退休。资格模式记录真实prefill/decode/retire路径，正式模式无该观察器。
raw IDs保留终止EOS，committed IDs只去掉末尾EOS，usage直接数后者；不会隐藏reasoning
或其他special token再冒充原始计数。core尚无cancel API，已进入引擎的请求断连使整次
服务测量失效，由owner回收，不从成功子集拼速度。

异步client在C个槽位中运行固定有限队列；任一DONE后立即补位。正式N=max(64,8C)，
首POST到末DONE组成共同窗口，实际tokens总数除以该窗口得吞吐；每请求独立算延迟。
保存原始SSE、EOS/usage、全部错误与未提交尾部；request/group均有绝对deadline，不重试。
资格receipt强制绑定完整请求body SHA、path及整组request→fixture映射；修改prompt、model、
cap、sampling或模板参数不能复用旧资格。后续controller还须验证receipt来源、实际加载模型、
源码与调度，不能由客户端的一段`status=qualified`声明自行授予数值资格。

## 当前边界与下一步

服务端18个、client 22个纯CPU mock检查已由作者和root分别通过，覆盖动态到达、并发提交、
独立退休、EOS计数、断连/超时和socket选项。独立审查发现并修复完整请求绑定不足与raw EOS
越cap边界；负测保留。root完整输出与五份源码SHA保存在`cpu-v1/receipt.json`。
这些检查不加载MLX、不创建listener。
新长形状的真实数值资格、完整controller及逐C资源资格尚未运行。

后续新增同调度双引擎资格入口 `qualify_native.py`：真实 eager reference 与现有候选
逐步比较全 vocabulary logits、全部 cache/ahead 状态、原始 IDs、到达与退休；首个差异
保留原 dtype 张量。C1 与 C8 分别正常构造双引擎，中间销毁并清理，最多两份模型驻留；
37 请求、最多每臂 12,411 raw IDs、含加载清理的 30 分钟绝对预算。
该入口尚未运行 GPU。root 复核全部 **52 项 CPU tests**，记录在 `cpu-v2/receipt.json`。

E022 v1 暴露了 macOS 的 TCP_NODELAY 非零 flag 读回（4）会被 `==1` 错拒问题；
本入口已按布尔语义修正并补 0/1/4 正反测试。原 `cpu-v1` 结果保持，不能以 mock
通过代替真实 socket 或模型资格。

已发现的快路范围值得独立优化：现有compiled/fused decode的used-length guard为≤256，
而grouped prefill需要同长、T≤512。当前八份输入从首decode就走普通fallback，也不满足
grouped prefill范围；不会把开关ON误写成实际kernel命中。先保留这个真实baseline，后续
扩大范围的候选另行默认OFF、验证与测量。

[具体数值资格方案](../../../bench/reviews/e023-qualification-plan.md)列出同调度双臂、
完整logits/cache/ahead比较、真实HTTP rid对照和确定的工作上界。256还写在compiled closure
的attention读取长度中，只删除guard会截断历史，不能作为正确的扩容实现。

执行时先D的C1/C8资格，再扩展C1/2/4/8/16/32/64。oMLX MTP须报告实际MTP/fallback覆盖率；
Lithos当前OpenAI接口的单inflight限制需要显式C2探针，不能静默重试串行化。
GLM-5.3-Flash沿用相同语义任务但单独tokenize、资格和作图，下载或CPU审查不会产生速度点。
完整候选协议见[下一轮负载方案](../../../bench/reviews/pareto-next-workloads.md)。


## v1 实机结果：正确性前缀通过，资源门中止

冻结计划在 `e9600a8` 后实际执行。C1完成8请求/4096 raw IDs，观察到的full logits/cache/
ahead/输出均一致；随后C8最后完整写入step515，在第一波结束和补位附近触发pressure2。
owner859.55秒停止child，returncode-15且wait/reap/严格absence通过；独立ps确认owner也退出。
外服务74384保持原启动identity并恢复T→S。完整37请求未完成，不能把C1完成或C8前缀PASS
提升为整组资格，更不能拿带逐步观察器的时长生成性能点。

资源第164个样本（编号从0开始）记录pressure2、swap324→469.06MB；postflight仍pressure2。
child RSS约29GiB并不代表全部GPU/system物理使用；外服务SIGSTOP仍占约143GiB footprint。
下一步核查双模型生命周期、快照与refill瞬时峰值。没有足够证据归因某个kernel、模型泄漏或
外部请求；不通过放宽资源门、缩短任务、重试成功子集来修饰结果。E024计划保持未执行。

[完整失败归档](../../experiments/m5-ultra-256gb/E023-realistic-qualification-v1/README.md)
保留332个成员与全部step前缀、资源和恢复证据；独立stream audit与root重跑逐字段一致，结果在归档内。模型/source
末尾recheck未运行，因此这次状态明确是未完成数值资格，而不是正确性PASS或数值FAIL。

审计独立重建调度、batch维、全层覆盖及退休：4,617完整行、4,611成功step记录、无末行截断。
全部已完成请求均到达cap，实际EOS早退覆盖为0；因此既不授予整组资格，也不声称覆盖EOS早退。


## Recorded HTTP replay preparation and approved E028 execution

Native HTTP raw SSE/UUID/scheduler join、独立双引擎实际调度回放及固定来源wrapper已完成，
28 CPU tests通过；真实24条D请求经冻结server/tokenizer解析通过。root修复startup identity
的int/float timeout不一致，保留原版和重跑证据。只有CPU/mock资格，暂无新Pareto点。
实际HTTP仍需capture owner及native client取消close的窄修复；不把准备工作当速度提升。
[可复跑记录](../../../bench/reviews/2026-10-17-http-replay-cpu-v1/README.md)。

用户现已明确授权“允许停止、实验后恢复”。接下来按冻结E028 model v1执行冷停→完整
37请求→回收→恢复原oMLX模型，并核新日志中的MTP/resident PLE。先前pending记录是历史
快照，当前不再等待同一授权；失败也执行恢复。M4分支53d4906已在本分支祖先中。


## HTTP baseline v1 readiness failure and v2 retry (2026-10-17)

Actual v1 cold stop succeeded, but the first capture rejected the owned listener
before any HTTP request. macOS `lsof -Fpn` returned the valid three fields
`p81230`, `f4`, `n127.0.0.1:18023`; the old parser required exactly two fields.
This is a harness readiness failure, with zero requests and no numerical result.
Child81230 was terminated, waited/reaped and strictly absent; the original oMLX
was restored as PID81337 with its original model and fresh MTP/resident PLE logs.
A separate offline restoration audit PASS independently checks raw physical/log
and child-absence evidence. Warm KV and exact hidden runtime state are not restored.
All original sources and failed artifacts remain unchanged.

The separate v2 capture permits one optional ASCII decimal descriptor field while
still requiring exactly the owned PID and localhost endpoint. V2 owner changes only
the capture source binding. Root28 capture/owner tests and12 restoration-audit tests
PASS; independent source reviews found no remaining blocker. V2 retains exactly
the D workload, C1 eight/C8 sixteen requests, normal EOS, cap512, model/precision,
resource limits and numerical contract. E028 remains OFF for this baseline.

[Complete failed-v1 evidence](../../experiments/m5-ultra-256gb/E023-http-baseline-v1/README.md)
and [registered v2 retry](../../../bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v2/README.md)
separate the historical failure, restored predecessor and next attempt. No new
Pareto point follows from this retry preparation; only independently qualified,
separate production timing can update throughput vs POST→DONE p50.


## Actual D512 HTTP qualification PASS (2026-10-17)

C1/C8 completed 24 actual HTTP requests, 5124 replay steps and 12,288 committed tokens.
Independent full-ledger audits account for 6,572,973 original-dtype bit-equal checks
and 13,145,946 finite checks across full vocabulary logits/cache/ahead state. All
requests hit cap 512; normal EOS was enabled but actual early EOS coverage is zero.
This covers recorded arrivals only. E028/E024 remain OFF and no timing point is awarded.

Both audit failures remain preserved: integer 0 versus the HTTP producer's float 0.0
sampling metadata/ahead key, and a strict-positive file-size check that rejected
three valid empty package files. Separate v2 auditors fix only those expectations;
root CPU regressions and complete actual audits PASS, with no raw record edits or
GPU rerun. Full source/runtime/model provenance and 249 raw resource samples passed;
pressure 1, zero swap/Swapouts/Pageouts growth, Swapins +3. All four children were reaped.
Original oMLX recovered asPID 90204 with the original model and fresh MTP/resident PLE
logs; warm KV/exact hidden runtime state are not restored.

[Complete D512 evidence](../../experiments/m5-ultra-256gb/E023-http-baseline-v2/README.md).

Observer-OFF formal native C1/C8 specs now bind these actual independent receipts
and pass CPU prepare/validate: 8 warm + 64 formal requests each. The next graph will use
actual committed tokens/common window versus full POST→DONE p50, separate from the
historic cap 32 panel. Official oMLX's D512 route/count observer and Lithos launch/cache
contract are prepared; neither has a new D512 measurement yet.


## Native D512 observer-OFF timing: two audited points (2026-10-17)

One fresh process per C, eight warm then 64 formal requests, same fixed NVFP4 model
and D fixtures, normal EOS/cap 512. Complete-cycle independent audit PASS:

| Concurrency | Requests | Committed tokens | Common window (s) | Output tok/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 64 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| C8 | 64 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |

These are single-run baseline observations. C8 raises total throughput and also
request latency; both measured native points are nondominated among these two.
They do not establish dominance over oMLX/Lithos, whose D512 points are pending.
The cap 32 history stays in its own chart. Actual early EOS coverage here is zero.

All 144 warm+formal raw requests passed protocol/count/text checks. Representative
captured C1/C8 histories supply full-logit/state numerical eligibility; no claim
is made that every new formal schedule was replayed. All 216 resource samples
passed, pressure 1, swap−8 MiB, Swapouts/Pageouts 0. Original service recovered
asPID 98702 with fresh MTP/resident PLE evidence; warm KV/exact hidden state are not
restored. Both actual experiment children were strictly reaped.

[Complete 935-member evidence](../../experiments/m5-ultra-256gb/E023-native-formal-v1/README.md) · [New D512 chart](../progress/2026-10-17-v6/README.md).


## E024 long decode graph rejected at its first point (2026-10-17)

The unchanged candidate failed exact BF16 full-vocabulary logits at B1/used257:
228,516/248,320 finite elements differ, maximum absolute difference 0.28125.
Independent raw safetensors audit and root rerun agree. Zero of 32 matrix points
passed; later argmax/cache/parameter/lifecycle and final source/model rechecks
were not reached. No speed or Pareto point is awarded; no tolerance was relaxed.

The original service recovered as PID 832 with fresh MTP/resident PLE evidence;
independent recovery audit PASS. Five resource samples had pressure 1, zero swap
growth and zero Swapouts/Swapins/Pageouts. Warm KV/exact hidden state are not restored.

The first-principles lesson is to isolate tensor shape from compilation. E024
changed attention length from 257 to 1536 as well as compiling the forward pass.
Matching MLX 0.32.3 source predicts different SDPA reduction paths on g17d across
K=1024. This is a hypothesis about the failure, not a measured cause. The next
separate diagnostic compares eager/compiled at each length from identical state.
Original evidence and the default-OFF candidate remain frozen.

[Complete 637-member failed-attempt evidence](../../experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md).
