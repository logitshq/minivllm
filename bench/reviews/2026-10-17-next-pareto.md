# 下一轮真实 Pareto：复用客户端，先固定合同

本笔记为静态规划及CPU输入重演，未注册/运行新的性能工作。正确性是硬门，主轴固定为实际整机
committed output tokens / 首POST至末DONE窗口，以及完整请求POST→DONE的p50；
p95、TTFT、长度/EOS/cap比例作为诊断。全模型仍需既定pressure1资源门与服务冷停/
恢复授权。E028两层合成state已过，真实模型37请求与64层refill状态门尚未跑。

## 最小native消融

复用E023 fixtures/client.run_group与现有owner/load/reap。先当前stack的E028 OFF/ON，
C1控制和C8 refill，各3fresh process，次序交错；不在正式计时挂step observer。仅新
server入口加默认off开关并在构造后调用integration，身份记录开关；不另造HTTP客户端。
原D组8条输入886–978token、greedy/cap512/normalEOS可作为负载候选；计划正式64请求、
固定8fixture轮换与完成后补充C个outstanding。下述CPU重演已确定这份固定D的零prefix
保存边界；实机资格/正式日志仍须验证零命中。各process点全部保留，配置点取各process统计中位数，不把请求
混池、不用64×512代替实际生成量。

官方对比另做同轮重测，连native胜出配置也重跑，不拼入旧19/32或前一消融作为新前沿。
oMLX的CLI/settings并发上限需都匹配C；target-only和MTP policy保留强合法配置，
实际MTP→target fallback必须记录且取得资格。

## Lithos实际可配置范围

已查当前editable安装0.1.2、clean commit `223658090ae549e59220bdec459957cbdad7926a`，
不是只根据旧429推断。`lithos-metal serve`→cli.py→serve.main，与E020的
`python -m monolith.serve`进入相同代码。源码SHA见[来源回执](2026-10-17-lithos-concurrency.json)。

- serve.py:195–210用单GPU锁包住整次generation；427为app级FifoLock，490–491让
  非Ollama请求在重叠时429。单cached session和max_sessions编译variant不是请求batch。
- parse_args:731–770没有batch/concurrency/workers参数，uvicorn workers=1；
  kernel-config、prefill-chunk-size、draft-block-size调整kernel/prompt/proposal，
  generate.py的in_flight调command buffer，均不代表多条请求同时生成。
- 官方 `/api/chat`（662–668）允许真实并发HTTP在同一FIFO排队，一次生成一条。
  可以作为“HTTP C8 / single-generation FIFO”测服务吞吐与含排队的完整延迟；它是
  合法并发负载，不能因为GPU不batch而直接删掉。不能改为客户端串行、隐藏重试或
  多worker复制模型后冒充当前官方配置。OpenAI入口的429也须保留。
- 本版没有 `/v1/completions`。chat/ollama必须发送原messages并独立核模板后IDs，
  不将已rendered prompt塞chat再套一次模板；Ollama流的final done语义需要明确适配。

## 固定D的零prefix保存边界：CPU已重演，实机仍待验证

[可复跑脚本](lithos_d_prefix_cpu.py)与[完整结果/来源SHA](2026-10-17-lithos-d-prefix-cpu.json)
已完成16项CPU检查：8个fixture分别经过实际`ChatRequest`和`ollama_request`转换，
用固定Jinja模板与tokenizer重演；每条完整prompt text/IDs均等于冻结fixture。
fixture SHA为`6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`。
脚本仅导入protocol/Jinja/tokenizers/Pydantic，禁止MLX、torch、transformers以及Lithos
serve/generate/runtime模块；后者只读并绑定源码SHA，运行前后均复核，不加载权重。

```sh
/Users/ryanli/code/lithos-metal/.venv/bin/python -I -B /Users/ryanli/code/minivllm/bench/reviews/lithos_d_prefix_cpu.py --out /tmp/lithos-d-prefix-new.json
```

输出路径必须不存在。结果是`CPU_SOURCE_REPLAY_PASS`，不是device/cache运行资格。
根agent已独立复跑16项检查并人工核对serve probe/generate checkpoint逻辑，生成JSON
逐字节一致：SHA `d2a154aecf6022b7ebf5180dbee5f1e116592210a158644c92629e774021fe14`。

具体原因：D的8条消息都是**单条user字符串**，无system/tools。serve.py:347–374的
两个稳定前缀probe经serving/protocol.py:141–153都变成empty-user消息，模板输出
`<|im_start|>user\n<|im_end|>\n`，IDs为`[248045,846,198,248046,198]`；与每条D
共同前缀恰为3token。generate.py:517–523先保留合法候选`[3]`，因此不会使用`p-1`
后备候选；随后`3 < min_tokens=128`使checkpoint集合为空。fresh空cache在只重复
这些请求时不会建立新checkpoint。**这是这份消息结构的零保存性质，非缓存功能关闭。**

可保留原官方`128-exact --no-warmup --keep-warm 0`，每个进程fresh、不配置
`--prefix-cache-dir`、不发送额外生成请求。固定暖机为8个C1请求覆盖D01–D08，再C个
并发warm，排空后开始64个formal请求；warm与formal之间不重启/清cache。
独立设备资格和正式运行仍必须保存完整日志及phase字节边界，每个完成的登记请求
对应一条Generation记录，全部`cached_prompt_tokens==0`。记录缺失、非零命中、
额外请求或source/config变化均使零命中合同无效，不能靠本CPU结果放行。

serve.py:129–134仍固定`prefix_cache=True`。`N-exact`的min_tokens是`min(N,128)`，
所以`1024-exact`不能关闭缓存；plain `1024`虽然能阻止这些长度≤978的输入保存，
同时改变prefill分块/执行路径，不能作为中性cache-off替代。本轮无需这项改动。
消息变成多轮/system/tools/多个content块时必须另审；PrefixCache最多保留2个内存
checkpoint也不代表任意8任务循环都无命中，更不代表全部warm hit。

原oMLX明确`--no-cache`和native无prefix可以保留为**uncached baseline**，不把它称为
各家最强服务配置。对“最强服务前沿”，另登记oMLX官方cache-enabled配置，在同一D
重复负载上允许自然命中；其冷/暖路径先获同负载资格，命中数据实记录。无需让不同
实现具有相同命中率，也不能静默关闭其缓存。当前client.py:371–372拒绝非零cached
tokens；若执行该服务负载，新增显式cache合同版本，保留旧uncached守门，不移除检查。

## Ollama FIFO所需的最小客户端适配合同

复用现有client.run_group的并发队列、连接配置、deadline、原始结果保存和统计；
只增加协议适配。Lithos的C1和C8都走`/api/chat`，标注`HTTP C / single-generation FIFO`。
旧OpenAI C1可另保留，不能与Ollama C8无标记连线。全局窗口是首POST至末终止记录，
排队时间属于请求延迟；不通过客户端semaphore、重试或多个模型进程隐藏服务器策略。

- 原wire body使用冻结messages、`stream:true`与`options={num_predict:512,
  temperature:0,top_p:1,top_k:0,seed:0}`，不加入stop/minimum/ignore_eos等覆盖。
  请求前对**原body**按现有sorted compact UTF-8 JSON算法保存SHA；适配成内部字段后
  不能重新计算并冒充原wire body合同。资格绑定path、body SHA、fixture和C。
- 保存全部原始NDJSON响应bytes及每次receive时间。增量解析允许跨chunk的UTF-8/JSON；
  只接受一个成功的`done:true`，终止时间取含完整该记录的接收时钟，不能取后续解析/
  清理时间、HTTP EOF或服务端duration。保留原terminal对象、`done_reason`和全部usage。
  error、缺/重复terminal、截断记录、terminal后额外数据均不能成为有效性能点。
- 内部完成事件可以统一接口，但不能伪造原始SSE `[DONE]`。TTFT取首次非空输出内容的
  接收时间；吞吐依旧是全部有效请求的**实际committed token总数/共同完整窗口**，
  p50/p95取各请求POST→done时间，不用服务端`eval_duration`代替。
- events.py:78–85的`eval_count`来自serve.py:408的`len(tokens)`；visible_text只在
  展示时截掉EOS。因此保留`reported_eval_count`，不能直接按现client的`identity`
  count rule进入排除终止EOS的主吞吐。独立资格必须验证完整raw IDs、terminal EOS集合
  `[248046,248044]`以及无EOS后额外token；只有已证明normal stop恰含一个末尾EOS时
  才规范化`committed_count=eval_count-1`，length路径不减。未证明则不计有效主图点。
  不从文本重新tokenize恢复IDs，不因为finish为stop就未经资格盲减1。
- `prompt_eval_count`对照完整模板后prompt长度；每个fixture保留输出/停止原因，EOS
  提前结束不删、不补足512、不改prompt。资格先覆盖实际FIFO route、正常EOS与cap路径，
  正式无observer。这里没有给新的长输入路径授予数值资格。

这不推翻已有E020图：其输入只有19token、Lithos chunk为128-exact，并且已冻结的
external_spec_http/contract.py:115–132要求全部六条Generation日志
cached_prompt_tokens==0；旧事实是“无命中”，不是“功能禁用”。本次没有重跑旧
数据，也没有新增吞吐/延迟点。


## 从初始消融到完整曲线

C1/C8只是首轮OFF/ON消融，不冒充完整Pareto扫描。三家同轮主图后续覆盖
C1/2/4/8/16/32/64中资源和对应正确性资格允许的档位，逐C冻结请求数与暖机安排；
保留全部失败及被支配点，仅连接已测非支配配置。横轴完整请求p50，纵轴整机实际
committed tok/s；p95、TTFT和内存另作诊断。正式窗口期间不运行重型CPU哈希、压缩
或原始数组审计，避免统一内存争用；这些准备/验收在计时前后完成。
