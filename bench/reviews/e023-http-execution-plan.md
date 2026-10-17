# E023 HTTP 实机执行补全方案

2026-10-17，CPU 只读设计，尚未实现、冻结或运行。主图固定为 **实际 committed output throughput 对 POST→DONE 的完整 E2E latency**；数值、身份、计数、资源和回收全部是入图前置条件。本文件不修改既有 server/client/qualifier/E020 source，不授予任何新资格。

下一步只需补一个窄用途 HTTP controller、一份严格的 trace/SSE join 与一个 recorded-arrival replay loop。先完成当前 native baseline 的 C1/C8；E024 必须先补真实 D512、gates ON 的全状态资格，再作为单独 opt-in 臂使用同一 HTTP controller。无需生产服务器、请求取消 API、插件注册表或新调度器。

## 本次读取的固定源码

以下 SHA 是这次设计的依据；下一次执行计划必须 pin 实际加载文件，不能只记 Git HEAD。

| 文件 | SHA-256 |
|---|---|
| `pareto_realistic/native_server.py` | `9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f` |
| `pareto_realistic/client.py` | `ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc` |
| `pareto_realistic/qualify_native.py` | `26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf` |
| `external_spec_http/run.py` | `892e22277969709f2abf46b45949856cebb9045b27a86d0c3a5a12cf9b7bea31` |

路径均在 `bench/m5_ultra_256gb/` 下。已有 E023 qualification plan 的真实配置是 `max_len=2048`、`max_batched_tokens=2048`、pool2GiB、每 case max_seqs=C；current candidate 为 compile/fuse/gates ON、norm OFF。E024 diagnostic CLI 的 batch budget16384、gates OFF 与此不同，不能直接签发当前 HTTP 配置的资格。

## 现成代码能复用到哪里

| 现成函数/位置 | 直接复用 | 必须补的边界 |
|---|---|---|
| `native_server.BatchWorker.tick`，约77–152行 | 每 step 开始 drain pending；单 worker 调实际 engine；rid 独立 delivery queue；before/after step trace | trace 的 `newly_submitted_to_scheduler` 是进入 scheduler 的请求，不能当成 GPU decode batch。真实 B 看 `actual_scheduler_output.decode` 与实际 runner rows |
| `native_server.observe_runner`，163–174行 | 录制真实 SchedulerOutput 的有序 decode、prefill(rid,n)、finished IDs，及模型运行后的 batch | 只有 metadata，不含 full logits/cache；不能把 HTTP trace 本身称 full-state proof |
| `native_server.completion/parse_request`，177–216行 | cap512/正常 EOS、完整 prompt/raw/committed IDs、唯一 terminal EOS、usage；严格 greedy 参数 | controller 需要独立检查实际模型、body、完整 prompt IDs 与响应身份；`model` 是接受的字段但当前 parse 不用它选择/验证 loaded model |
| `client.run_group`，520–617行 | async finite queue、同一初始 barrier、立即补位、绝对 request/group deadline、bounded cleanup、全部 raw SSE、N=max(64,8C) | 它信任外层传入的 qualification；SHA 字段只验格式。controller 必须真正读/hash/验证 receipt 和身份 |
| `qualify_native.Comparator/snapshot/state/Observer/pair_calls` | 全 vocabulary logits、完整原 dtype state、ahead 语义 key、每数组 finite/bit-exact、首失败 artifact | 可直接按固定 SHA 动态 import；只保留当前 step 和一个 future 的 tensor 引用 |
| `qualify_native.run_case`，350–444行 | 原 C1/C8/refill/lifecycle 的完整 D512 数值资格 | arrival 只有固定 refill 与13-job lifecycle 分支，不能直接传 HTTP arrival trace 冒充已支持 replay |
| `E020.run.ps/listeners/lease_check/sample_resources/close_server` | 直接 Popen server 的 PID/PPID、唯一 listener、资源门、TERM→KILL→wait/reap、PID/端口消失、外层 service lease | `run_one` 固定19/32 oracle、三 warm+三 formal串行请求及旧 identity；`verify_pins` 还绑定 Lithos HEAD。两者不能原样用于 E023 |

最小 controller 可借鉴并复用上述 E020 lifecycle primitives；HTTP orchestration 改为调用现有 E023 `run_group`。不能为了复用旧 `run_one`，把64请求伪装成一个旧 row 或弱化旧 validator。导入冻结 E020 module 时，沿用已验证的绝对 sibling import/source-root 检查，防止通用名字 `client` 指向 E023/E020 的另一份文件。

## 从 body 到真实 server rid 的严格绑定

controller 在 CPU 阶段冻结每一个 job：`request_id`、`fixture_id`、`/v1/completions`、完整 body、`canonical_body_sha256(body)`。body 的 prompt 是已有 prepared D 的原始 chat-template 文本，cap512、temperature0、top_p1、repetition_penalty1、stream/include_usage/n1；不重新模板化、不插空白、不重分词后改 IDs。body 的 model 标识与 readiness 返回的服务标识严格一致；loaded snapshot/source/weights 由 controller 另核。

**客户端 ID 与服务器 rid 分开保存。** `client._request` 第301–302行发送 `X-Request-ID`，当前 server 不读取该 header，`BatchWorker.submit` 生成 UUID。无需为此改冻结 server；对每条 HTTP response row 使用最终 SSE 的 `benchmark_request_id` 建立 `client_request_id → server_rid → frozen fixture/body` 的一对一表。不能要求两种 ID 字符串相等，也不能用文本相同猜同一个请求。

controller 在所有资格与 formal 结果上补以下纯 CPU post-checks；它们在计时窗口外执行：

- 每条成功 row 恰有一个非空 server rid，整组无重复；每个 SSE `id` 必须唯一且严格为 `cmpl-<server_rid>`。client 现有的“至多一个 response id / 不跨请求复用”还不检查这个等式。
- `benchmark_prompt_ids` 必须逐整数等于对应 fixture 的全部 frozen IDs。当前 client 第404–407行仅核长度与 usage，完整相等需补；只验长度不能排除同长串线。
- 原始 output IDs/committed IDs/EOS 的一致关系采用现有 `native_committed_ids` 语义：raw = committed 加至多一个实际 terminal EOS；中途无 EOS，cap 边界正确；usage 是 committed 数，cached count 明确为0，finish 与 EOS/cap 相符。保留 reasoning/control token，只剔除 terminal EOS。
- 用同一 pinned tokenizer 将 committed IDs `decode(skip_special_tokens=False)`，与客户端拼出的实际 answer 文本精确相等。不要 tokenize 文本后反推出 raw IDs；tokenization 非唯一，已有 final SSE 提供真实 IDs。
- 完整 body hash、path、client request ID、fixture映射不允许被 response 反向改写；client request_id 的全队列表必须与计划一致。失败、缺字段、非200、截断、缺 DONE、异常清理一律不签发有效点。

资格 HTTP group 首次采集可传 `qualification=None`，保留客户端“不可信计数”的原始结果。不要先造一个 `status=qualified` 来让第一轮变绿。controller 的严格 join 和 numerical replay 完成后，另写正式 qualification receipt；原始采集 group 保持原状态且永远不是速度点。

## 最小真实 HTTP 数值资格：录制、join、重放

每个 `(arm,C,model/source/config/body contract)` 独立执行一次资格，服务器带 `--qualification-trace NEW_FILE`。初始 baseline C1 用8个 D jobs，C8用16个 jobs，cap仍512、正常EOS。客户端闭环与正式测量相同；资格 N 允许小于64，但必须原样标 `phase=qualification`。

录制服务器 trace 和 HTTP group 后先 **停服、wait/reap**，再启动 paired numeric replay；禁止让 live HTTP server 与两份 replay engine 同时占 GPU。保存 startup identity、每步有序事件、所有 SSE 与上述 ID 映射。worker 在单线程串行写 trace，可用事件顺序赋严格递增 step index，不需要比较不同机器或线程的 wall-clock 时间。

trace parser 必须验证完整事件序列，拒绝残缺、重复或未知事件。每个 step 将下列数据变成一条有限 schedule record：

```text
before_step:
  ordered arrivals = newly_submitted_to_scheduler
  scheduler_running / scheduler_waiting
actual_scheduler_output:
  ordered decode rids / ordered prefill(rid,n) / finished_ids
actual_model_runner_batch:
  actual post-run decode_rows
after_step:
  ordered delivered(rid,token,done) / retired / surviving_decode_rows
```

只接受有因果对应的完整 step；若 `engine.step()` 没有进入 runner，必须明确证明是空输出，不能随意补一个 missing runner event。正常 trace 若遇到这种尚未支持的序列，保留并停止扩展，不猜测。

用最终 SSE 的 prompt IDs/cap 补全每个 arrival 的输入。每个 rid 全程恰有一次 admission、一次 retirement；按 trace 合并输出 IDs 必须等于该 HTTP row 的 raw IDs，最终 done 后没有输出，所有计划 jobs 都完成。FIFO待执行队列、running列表、surviving rows、实际 B 分布都保留。C8 至少实际出现多行 decode；若要声称覆盖 B8，还必须实际看见 B8，不能只凭8条 HTTP outstanding。

**新的 `run_recorded_case` 只需要一个约70–110行有限 loop。** 原 frozen `run_case` 没有 arbitrary-arrival hook；不通过 monkeypatch scheduler 返回值强迫重放，也不把 recorded tokens 喂给 candidate。按 step 向两个正常 engine 同时 `add_request` 录到的 arrivals，之后调用原 engine 的 `step()`，复用以下原函数完成所有比较：

1. `state`/`Comparator.compare` 比较 before-state；reference 丢弃上一 ahead，但清除前必须已比较。
2. `Observer.begin`，reference 在 `grouping_disabled` 中 step；candidate 以 reference 的 schedule 为约束 step，保留实际 ahead。
3. reference/candidate 有序 SchedulerOutput 均须等于 HTTP trace：decode/prefill/finished IDs、before running/waiting、post-run batch；将原 `schedule_metadata` 的 request 元数据同时保留。
4. `pair_calls` 比全 logits 与128数组/层状态；比较 before-retirement、after-retirement 全 committed/staging/ahead，输出 tuples 与 recorded HTTP tuples 全等。
5. `Observer.finish` 保存 future 的产生、消费、丢弃；终态 `assert_empty`，最终 rid→raw IDs 再与 SSE 全量 join。

两臂分别使用自己的 greedy 输出推进；如果不能复现 HTTP 实际输出，就失败，不以录到的 token 覆盖分歧。C8 轨迹的 oracle 是其同调度 eager reference，**不是 C1 的 raw IDs**。只复用 E023 工具和已审阅的单步比较顺序，不复制整个611行 qualifier，不新增通用 replay DSL。

这份 receipt 证明的是记录下来的真实 HTTP 路由与完整同调度数学。正式 observer-OFF 进程的 arrival 时序会改变，因此 receipt 的 scope 不能写成“逐条证明未来所有 formal request 的所有 logits/cache”。正式点依赖已验证配置/每C执行域，并逐请求执行协议/身份/计数/完整输出绑定检查。若要求 formal 每条新调度也 full-state replay，就必须另登记新的采集方式；不能同时声称当前 formal 零 step observer且拥有那条未知轨迹的全状态证据。

## Qualification receipt 与 formal 的边界

receipt 至少绑定：arm/C、8份输入与tokenizer SHA、完整 body/path contract、实际 model revision/全部shards、core/server/client/replay源SHA、四个MLX环境变量和runtime identity、max_len/max_seqs/max_batched_tokens/pool、compile/fuse/gates/norm/long_graph、normalEOS/prefixcache、socket开关、trace与HTTPgroup SHA、numeric replay结果SHA、实际 B/position/backing/ahead覆盖、resources/reap有效性。

receipt 不含自身 hash。controller 读取并验证该文件后，才将其实际 SHA 填入 client qualification 对象的 `receipt_sha256`；同时为本次全队列生成精确 `request_fixture_map`、fixture body hashes、`native_committed_ids`、terminal IDs和prompt计数。这里是“已合格机制 + 新的固定队列绑定”，不能伪装成该 formal 队列已被逐请求 replay。

`expected_raw_ids_by_request` 只用于已有对应实际 schedule 的资格验证。formal 不填旧资格/C1的 output oracle；新调度本身可能改变 batch reduction和greedy路径。raw ID 与文本、EOS、计数仍全部验证并保存。

启动 formal child 前 readiness 必须确认 `step_observer=false`、`qualification_trace_enabled=false`、`per_token_observer=false`、drafter=None、normalEOS=true、prefix_cache=false以及实际配置/路径/hash。必须是新的模型进程；不要在资格服务上关 trace 后计时。warm与formal之间不 clear cache、不reload、不改allocator、不改温度/输出cap。

## 初始 C1/C8 baseline 的有限性能计划

只在当前完整 E023 数值资格和各C实际 HTTP replay全部通过后执行。每个C各3个fresh processes；每process先固定8个原D warm requests，覆盖所有fixtures，然后一次 `N=max(64,8*C)=64` formal closed-loop group。formal job次序固定为D01…D08循环8遍，client request ID在phase/process间唯一。正常EOS提前完成照实保留且立即补下一份；不补token、不重抽任务、不替换样本。

client直接给出：

```text
Y = sum(actual committed output IDs) /
    (max(all DONE timestamps) - min(all POST start timestamps))
L = median(each DONE - its POST start)
p95 = nearest-rank sorted[ceil(.95*N)-1]
```

每个formal request的E2E包含server排队、prefill、decode、网络和client排队；不减 TTFT，不用 `tokens / mean latency` 替代整组 throughput。TTFT保留首 answer/reasoning输出口径；同时报告p95、实际输出长度、EOS比例、峰值HTTP outstanding和资格中的实际B分布。资源probe与磁盘落盘放在group外，不能在每token或每request插入探针而改变闭环补位。

每个process只有一个 `(Y,L)` raw point；3个process分别画小点，并另给预先指定的3次process中位汇总点。不要把192条请求混成一个“伪单次进程”的点，也不要只选择最快一次。任一process不完整/资格/资源/reap失败，该配置不能作为完整三次通过点；全部失败/partial留下单独状态记录。

该baseline计划共6个fresh性能进程、48 warm +384 formal请求，最多221,184 raw output IDs；资格HTTP另24请求/最多12,288 raw IDs，同轨迹 paired replay每臂另24请求/最多12,288 IDs。这里只给工作量上界，不承诺完成时长。

每个C是一份新资格和receipt。C8通过不能自动授予C16/32/64，HTTP C不等于GPU B。后续每新C先资源与实际route形状资格，再正式N=max(64,8C)：C16/32/64分别128/256/512请求；不因显存/超时失败而悄悄缩短输入输出、换精度或关功能。

## 有界 controller 与资源恢复

新 controller 的每个 HTTP process只直接 `Popen` 一个实际listener server，保存PID/PPID/command/environment/sourcepins；不用owner再fork一个不拥有GPU的伪server。readiness轮询核 `child.poll()` 与唯一listener PID，`/v1/models`完整identity核对通过后才发warm。

复用E020的bounded readiness/probe/reap语义，但数值预算须按长任务重新冻结，不能继承420秒“下一条短请求”余量。建议首版显式计划给单请求300秒、warm group600秒、formal group1800秒、readiness900秒、HTTP清理5秒、整process3600秒；外层lease至少还需覆盖两个resource probes与TERM/KILL/reap、source复核/恢复余量。若预算和最坏路径之和不相容，prepare直接拒绝，不能运行后再延长。具体值由owner在开始前冻结，本设计不授予超时延期。

`asyncio.run(run_group(...))` 的结果先保存。组失败或 `GroupCancelled` 时保留其 `.result`，立即进入server cleanup；如果客户端cleanup不完整，不能继续下一组。native active cancel会永久fail整个worker，不能复用同一个服务“试下一条”。readiness/resource/transport failure均无自动重启或替代样本。

finalizer短暂屏蔽重复SIGTERM/SIGINT，按E020限时TERM→KILL→wait，核PID不存在、listener空、连接拒绝，写release和cleanup；之后恢复信号mask。外层仍只由已有 `suspended_service.py` 持有停/恢复授权并核服务同一identity，controller不能自行再暂停/恢复用户服务。全部模型/source/runtime pins在owner阶段前后核，计时中只做必要轻量身份检查，不重复hash大权重。probe与SIGALRM不能和另一个Python alarm叠用；group使用asyncio绝对时间，process超时由单独owner承担。

## E024 opt-in 的最短后续路径

1. 先保留本轮 baseline C1/C8数据。E024的32点diagnostic/gatesOFF通过只打开下一步，不能代替真实任务资格。
2. 另一个小CLI加载相同2048/2048/pool2配置，复用 E023 `workloads`/`run_case` 跑原37份D512/lifecycle traces。candidate只是 `runner.decode_model = LongContextDecode(model, enabled=True)`，gates ON/norm OFF；reference依旧明确eager/gates OFF。完整model/cache/ahead逐位门与实际compiled-callable证据不变。D实际backing应达1536且所有graph calls必须真的eligible；不能把fallback计为graph成功。
3. 新建极窄 `serve_long_graph.py` wrapper，按固定SHA加载冻结E023 server。只在该process的 `BatchWorker.initialize` 前接入long graph，并给 readiness identity增加实际selected decode callable、candidate SHA/bound和是否启用qualification observer；正式进程不安装GraphWitness。不改core/冻结server、不改payload/批处理算法。CLI原compile/fuse flags若仍记录为requested，需要同时明确实际替换后long path为graph-only、无packed/fused projection，不能让identity暗示fuse正在执行。
4. 对E024 C1/C8也录制自己的真实HTTP trace并paired replay。它与baseline的arrival/输出不必相同；每条都相对自己的同调度reference验数学。然后才能用相同controller进行无observer性能。
5. 若要判断E024采用门，必须同时运行新的baseline/candidate三fresh配对，按事先固定OFF→ON / ON→OFF / OFF→ON逐C执行。先取得的baseline点可以保留展示，但不能用历史baseline拼成新的配对比值。两臂保持同任务队列、正常EOS、socket设置、容量/pool、gates、客户端与warm策略；候选差异仅long graph路由。门沿用另行冻结的E024 throughput–complete-latency计划，不在结果出来后因某个C好看而挑选。

E024只支持B≤8；C16+不得从本次推出已获graph覆盖，更不能把大量fallback当候选收益。先完成C1/C8的真实长任务，再讨论扩B、融合或第二bound。

## 最小新增交付与 CPU 反例

建议只新增 `http_execute.py`（CLI/plan/Popen/group/lifecycle），`http_replay.py`（SSE/trace join、约70–110行recorded loop、receipt），以及对应CPU tests；E024届时另加小server wrapper。预计合计约250–400行业务实现，不含复用的已冻结库。先交controller/replay，再扩外部引擎，不提前为oMLX/Lithos写通用接口。

CPU反例要覆盖：两个同长prompt串线；client ID与UUID映射重复；错误SSE response ID；缺/重复trace事件；真实admission与计划初始barrier不同；ordered decode rows调换；raw/committed/EOS/usage错误；文本与raw IDs不符；正常EOS不足实际B覆盖；旧C1 oracle被误传C8；伪receipt SHA或错误source/config；formal observer未关；少于64个formal请求；组取消和清理超时；直接serverPID/port不能reap。用mock HTTP/engine/owner，不启动模型或listener。

最终一个point能否进入Pareto图由独立audit重新从raw group计数、wall窗口、每请求latency、source/receipt/identity/trace/replay/resource/reap计算，renderer只消费已审计结果。所有未通过点仍有可追溯状态；正确性没有被压成可与速度交换的第三轴。
