# 下一轮并发 throughput–latency sweep：候选执行方案

2026-10-17，**候选，未运行、尚未冻结为GPU执行计划**。只读 E020、v3图的计量协议与现有服务源码后提出；不修改这些实验的输入、门或结果。目标是先扩展 Qwen3.8-27B NVFP4，再为 GLM-5.3-Flash 的固定 mixed-bit checkpoint 建独立曲线。

主图始终是 **横轴请求 POST→SSE DONE 的实际 E2E latency，纵轴实际完成输出token总数/同一批请求的完整时间窗口**。正确性是入图前提。这里不增加“牺牲质量换速度”的维度，也不把微模块倍数放到主图。

## 先修正两个会让“并发”名不副实的入口

E020 的 `external_spec_http/native_server.py:54` 是取一个job、一直执行到 `engine.has_work()==False` 才取下一个。HTTP并发打入它会得到串行排队，不能据此声称已测native batch。新shim应保持单GPU worker，在每个step边界drain pending并由原scheduler限制admission，把 `rid→output queue` 分开。v3 的 `frontier/native_child.py` 有这种结构，但它带逐步timer、trace及旧配置，不能原样冒充无观察器性能server。新的qualification先证明实际有多request同处decode batch、不同输出长度独立退休；正式server移除这类观察。

E020 的client只支持C1，使用主线程 `SIGALRM`。不能直接把该函数放进线程池。新client采用一个 `asyncio` loop、`httpx.AsyncClient`、每请求绝对deadline和总组deadline，保存部分SSE及全部失败；client构造在计时外，每个POST计时从发起前开始。连接池上限至少C，不因客户端默认连接限制暗中把C64变成小并发。保留无keep-alive或明确固定的连接复用策略，两臂一致；native的TCP_NODELAY选择必须预先冻结，不能按E022后续结果在本轮中途切换。

## 最小有价值的负载矩阵

长度是**输入目标桶 / 输出上限**，不是强迫生成的输出长度。每桶冻结8份不同文本fixture；输入长度包括最终chat template后的全部token，在资格前记录每份确切IDs。Qwen与GLM使用相同语义任务、各自tokenizer/template，分别发布图，不混为一条模型曲线。

| 负载ID | 输入目标桶 | 输出cap | 固定内容 | 并发计划 | 解决什么缺口 |
|---|---:|---:|---|---|---|
| S：短对话 | 256±16 | 128 | 8份短文/记录的解释与摘要，包含自然较早EOS的请求 | **1/2/4/8/16/32/64** | 真实短请求，测排队与批处理，避免只比较19-token prompt |
| D：长生成 | 1024±32 | 512 | 8份自包含Python函数/小模块需求和上下文，要求完整实现及简短说明 | **1/2/4/8/16/32/64** | 让持续decode、投机接受与kernel成本有机会主导，而非32-token边界 |
| L：长上下文 | 8192±64 | 128 | 8份固定代码/文档中的局部信息提取，问题在尾部、答案位置预先分散 | 先 **1/2/4/8**；16/32/64在容量资格后按相同规则扩展 | prefill、KV/DSA增长；不把短prompt吞吐外推到长上下文 |

第一轮发布 **S+D 的完整七个C**；L是独立第二批，避免一开始做低价值的全笛卡尔积。若只有预算完成一个负载，先D：它直接纠正32输出过短的问题。旧19/32只可用作每个新server版本的C1接口回归记录，不混进新负载统计，也不作为本轮主曲线。

fixtures应在速度运行前就冻结：原文、messages、实际渲染prompt、token IDs、output cap、greedy设置、正常EOS集合、thinking/stop/template配置。挑选依据是任务内容及输入桶，不依据哪个引擎更快、接受率更高或恰好生成到cap。normal EOS提前结束不删样本、不重试、不补足到128/512；若D的全部fixture都自然短答，则如实报告“实际长输出覆盖不足”，另立下一版任务集，不能在本轮替换。

## 请求到达方式：固定并发的有限工作队列

每个 `(model, engine-policy, workload, C)` 使用3个fresh server；同一重复编号的不同engine-policy轮转顺序，C升降顺序在plan中写定，不能跑完挑更有利顺序。

每个fresh process：

1. 先固定8个C1 warm requests覆盖8个fixture，再做一批C个并发warm请求，独立记录并完全排空。不开prefix cache；allocator保留预注册配置。
2. 正式任务数 **N=max(64,8×C)**，因此C1/2/4/8是64，C16是128、C32是256、C64是512。fixture循环构成相同比例的8任务混合，顺序用固定seed冻结；每一臂使用同一请求ID→fixture映射。
3. barrier同时释放最初C个POST；任一请求到达DONE立即发下一个冻结job，直到N个job均已发出，再排空。不中途sleep、不在request timer里采资源，不因EOS早而等待其他请求凑齐一个wave。
4. 全组窗口从首个POST start到最后一个DONE；含所有排队、prefill、decode、draft、verify、HTTP交付与末尾排空。每请求记录start、首非空输出、DONE、usage、finish及stream events。实际outstanding trace由client时间戳重建；记录达到C的时段和尾部下降，不把requested C当已证明GPU batch size。

这是一组有界closed-loop并发实验，非固定外部到达率/生产SLA压测。它的目的正是扫描C对应的吞吐—等待时间折衷。p95至少来自每进程64个正式请求，仍只作描述统计；不将三个fresh进程当作192个独立环境重复。后续若要混合到达率、排队SLA或更精确尾部，另注册固定arrival trace，不与本次结果池化。

为什么用可变任务长度：S中自然早EOS和D中的长生成会实际检验retire/refill与head-of-line blocking，而不只是64个相同32-token请求同时结束。每个请求的EOS不会截断其他rid；跨prompt输入/输出串线属于正确性失败。

## 输出计数与统计公式

令每请求 `s_i=POST start`、`f_i=first emitted nonempty output`、`d_i=SSE DONE`，`n_i`为**已提交target输出token数，排除终止EOS/stop与所有未接受draft proposal**。正常输出里的reasoning若模型配置产生，应计入模型输出并完整核对；thinking策略必须在模型内各臂固定。可见answer与reasoning应分别记录，不能通过吞掉reasoning降低耗时后仍称同一负载。

- 组吞吐：`Y=sum(n_i)/(max(d_i)-min(s_i))`。绝不使用 `N*cap`，也不求各请求tok/s的平均。
- 请求延迟：`L_i=d_i-s_i`；主横轴为该process全部正式请求的 `p50(L_i)`。p95用nearest-rank `sorted(L)[ceil(.95*N)-1]`，写死算法。
- 每配置的大点：3个fresh process的Y中位数与各自p50 latency的中位数；同时显示3个真实process小点。p95和TTFT也先按process算，再列三个值与中位数，不能将旧样本池化提高表面置信度。
- 每点附实际输出长度min/p50/p95/总数、EOS提前结束比例、触cap比例、prompt总数，以及成功数/提交数。不同输出长度不能隐藏在一个速度数字后面。
- TTFT是辅助延迟，不替代主横轴。HTTP一chunk可能含多个token；只有准确token事件时才报告token级ITL，否则明确标chunk gap，不把反分词估计当真实逐token p95。

`usage.completion_tokens` 必须在独立资格阶段与上述 `n_i`定义核对。EOS是否算入usage、reasoning字段和模板补发文本可能因server不同；正常EOS后不再要求旧协议的 `finish=length` 或恰好32token。合法结束是模型EOS/stop或到cap，仍必须有DONE及一致计数。

原始token IDs可用时直接数提交ID，按已冻结EOS语义剔除终止token；不能重新tokenize文本后假设恢复原始ID。外部正式HTTP若没有raw IDs，只能使用已通过同负载独立raw-ID资格的usage语义并核对完整输出；若新fixture出现不可逆的特殊token、隐藏reasoning、缺usage或无法校准计数，该点不进入主吞吐图，保留失败证据。不能给它补一个cap或可见文本估算值。

每模型、每负载单独求Pareto：延迟≤且吞吐≥、至少一个严格改善才支配。S、D、L以及Qwen/GLM不互相求联合前沿。相同任务集但不同framework自然输出长度不同，应连同长度分布解释任务级比较；不称为等长生成kernel对比。主轴仍是throughput与latency，不另造质量坐标。

## 正确性资格必须覆盖本轮新形状

E010的19/32、E015/E017的C1短输出不能直接授予1024/512、C64或8192上下文资格。首次每个新增形状先做不计时的原始IDs/状态资格；自研调度用固定输入、固定batch/admission/retirement trace与reference逐步比较logits/cache/输出。HTTP自由到达改变batch划分时，不能把一个C1答案强行作为所有C的逐位oracle；需用资格阶段验证的实际调度路径解释并保留完整请求结果。候选在相同固定调度下有任何数值或状态失败，则该候选点不画成有效速度。

正式矩阵出现timeout、HTTP错误、计数错误、跨rid输出或状态错误，保留全部已得数据，当前配置无有效主图点。统计不能只用成功子集；诊断吞吐可以另列 `successful committed tokens / full attempt window`，同时显示reject/error rate，它不是Pareto候选。资源不足则该C记 `not qualified: capacity/resource`，后续更高C暂停，不能记零吞吐，也不能外推“支持64”。

## Lithos：当前OpenAI接口只有一个in-flight请求

本机 `lithos-metal/monolith/serve.py:483` 在OpenAI adapter中 `lock.acquire(blocking=False)` 失败即返回429 `model_busy`；同文件Ollama分支允许排队，是另一种API语义。E020所用OpenAI路径因此预期仅C1全成功，**不能在client增加semaphore或重试，把串行化隐藏成C8/C64**。

候选执行方式：每个S/D/L负载的Lithos none/DSpark做正常C1性能；另做一次预注册C2 admission probe（同步到达、无重试、保存200/429和时间戳）。若确认当前单inflight限制，C2/4/8/16/32/64在coverage表均写“当前OpenAI单inflight设计不支持；仅C2探针实测，其余按固定源码能力未执行”，主曲线只放C1。若C2意外全成功，也必须先核实际重叠与源码/config，不能立即宣布批处理支持。

若将来需要比较Lithos排队服务，可以单独登记官方Ollama queued API或公开队列wrapper；那条曲线必须标 `queued single-engine`，计时从外部最初到达到完成、包括队列等待，并明确GPU仍单request。不得与当前OpenAI baseline无标记替换，也不通过启动C份模型实现“并发”。

## oMLX MTP：按实际policy覆盖率命名

本机 `omlx/scheduler.py:10338` 的 `_route_to_vlm_mtp` 有明确条件：drafter忙则回退；已有waiting、running或其他prefilling request也回退到BatchGenerator。因此C>1时“MTP开关ON”不是“全部request实际MTP”，也不应因为安全fallback就判它数值错误。

1. 独立资格对每请求记录 `(internal_request_id, route=mtp|target, reason)`，以及实际proposal/accept/verify计数、原始输出IDs与target-only同调度对照。复用E015实际proxy counter所有权规则；normal EOS/cap缩短最后一轮时，不能拿 `rounds*(block_size-1)` 冒充实际proposed数。
2. 正式测量移除observer，保留原生INFO logs：unique MTP start/stats配对；unique fallback request及busy/contention等原因；startup证明同一drafter已attach。用phase日志byte offsets保存warm/formal边界，不截断原日志。所有实际请求都必须能归类，重试/重复route记录要dedup并说明；不足则标route evidence incomplete，不把该点称为MTP加速。
3. HTTP SSE response ID、client `X-Request-ID`、内部request ID不能凭名字相似直接join。本机 `server.py:4143` 仅把X-Request-ID用于preflight，而内部 `_request_abort_id` 与 `cmpl-*` response ID分别生成。资格阶段建立准确mapping；正式原生logs若没有可验证mapping，只报告可闭合的process级覆盖率，不虚构逐HTTP请求归属。
4. 图例用 **“oMLX MTP-enabled policy（实际MTP x/N，target fallback y/N）”**。x=0时明确“全部target fallback”；0<x<N为mixed policy；不能把两者叫纯MTP曲线。target-only与enabled policy是两套完整配置，重复后实际coverage不同也原样报告。

正式logs的accepted/nominal capacity仍可作明确标注的诊断，真实proposal acceptance率只来自独立资格。MTP C1是否更快、C8 fallback是否回到普通batch性能、尾部队列清空时是否重新进入MTP，都是这个长输出工作队列能回答的新问题，不提前许诺收益。

## Qwen与GLM的实际执行次序

**Qwen先行：**冻结新shim/client/8×3 fixtures及exact input IDs → D负载的C1与C8资格 → 扩展D全七C → S全七C → L短列表。运行前新harness必须把 `max_len` 改为覆盖实际prompt+cap并固定该设置；当前2048的E020 shim不能直接接8192。各层source/quantization不变，scheduler/admission参数写入identity；无需为测L先实现新kernel。

**GLM随后：**先通过固定snapshot的C1加载和同样S/D计数资格，使用GLM自己的EOS/模板与loaded vendor source。未有独立自研GLM engine时，只有官方oMLX baseline点，不能借Qwen曲线或把裸官方model call称自研胜出。Lithos本机未见GLM adapter，标unsupported、无数据点。GLM native MTP与Qwen外部MTP路径不同，另查其实际route，不套用上段覆盖结论。

GLM的190 GiB权重与被暂停但仍驻留的其他服务、加载副本及并发KDA/DSA cache共同决定容量。按C1→2→4→8→16→32→64逐级资格，实际内存门失败保留到达的supported range；C64是目标网格，不是保证可运行。不要通过更低精度、SSD offload或改变context长度让点悄悄通过；若另选这些配置，需独立命名和预注册。

最终每负载输出一张实际throughput–p50 E2E图、一张配套p95/TTFT/输出长度/实际route coverage表，以及七个C的状态表。所有fresh过程点和不支持/失败原因可追溯；只有完整且合格的三次过程形成配置大点。这个方案尚无任何新性能数据。
