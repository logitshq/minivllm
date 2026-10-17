当前 pinned Lithos 可以测 C1/C2/C4：用官方 Ollama `/api/chat` 排队入口；不需要升级或改服务端。
实际安装 lithos-metal0.1.2/commit5485ad6bac0a5649d66a347f3e01a655ea29d937 的196个 registry 文件当前字节均匹配；8个关键 installed 源与 pinned checkout 相同。新上游未查，V128/第44章数据未改。

| 问题 | 实际源码与结论 |
|---|---|
| 旧429 | `monolith/serve.py:482–491`：非 Ollama 只试 nonblocking acquire，busy 返回429/model_busy；不是模型/GPU不支持多个HTTP连接。 |
| 官方可测入口 | `serve.py:662–668` POST `/api/chat` → `ollama_request` → 同一 dispatch/Backend.complete。只有此协议在 busy 时 blocking acquire。 |
| FIFO证据 | `serving/lifecycle.py:26–46` ticket/Condition 队列；nonblocking 有holder或waiter即失败；实际CPU4waiter票序[0,1,2,3]。指服务端取得ticket顺序，不保证客户端编号顺序。 |
| GPU并发范围 | `serve.py:207–210,427,811`：一 app FifoLock、一 Backend GPU锁、uvicorn workers1。C2/C4是客户端并发+服务器串行排队，不是独立序列batch2/4。 |
| max_sessions=8 | `serve.py:39,116–148` CPU recipe/sampling variants缓存，仅selected Session保有GPU buffers；切换adopt后release。不是8个并发槽。 |
| 独立Session | `generate.py:81–105,156–178` 可由Python手工构造、各有states/engines；CLI没有并发Session池/batch/worker选项，模型sampler也会被构造器改写。自行多进程/多Session不是当前官方单实例入口，内存/共享权重/线程安全未经资格，当前不建议。 |

请求见 `request-template.json`，model从 `/v1/models` 读取，messages与旧Chat完全相同。`options.num_predict=32`→max_tokens32；temperature0/top_p1/top_k0/seed0/中性repeat_penalty1；不把max_completion_tokens放Ollama顶层（protocol.py:450–490）。
CPU用被测Python+真实AutoTokenizer验证两协议 template_inputs相同、greedy/limit相同、apply_chat_template精确等workload.prompt_ids=19；没有加载27B模型或导入MLX。
这只能证明转换/输入相同；新的Ollama实际32输出、visible token stream、无EOS截短、缓存0，必须由root下一次GPU资格确认，不把旧Chat生成证据挪作Ollama实测。
NDJSON（非SSE）：逐行解JSON并累加 `message.content`；任意error即FAIL。终止 `done:true`、`done_reason:length`、`prompt_eval_count:19`、`eval_count:32`；末帧content为空，不是完整文本，没有`[DONE]`（events.py:76–86,141–151,206–209）。
client时钟 POST前→收到done:true，必须包含服务器等待/header等待；Y仍为组总output/(最后done−首个POST)，不能把服务端eval_duration当请求延迟。
为避免协议混轮，新C1/C2/C4均用Ollama并独立fresh进程；3warm+3formal groups，同一组C个并发POST，不能复用V128的Chat-C1当新Ollama-C1。

同一模型/session会复用GPU buffers；每请求 `generate.py:551–553` reset状态（KV位置归零后先写后读），不等于prefixcache被禁用。
`serve.py:129–134,347–375` 开启prefixcache并传稳定prefix；`runtime/prefix_cache.py:39–60` 允许精确前缀命中，save受min_tokens限制，min由prefill设置决定。19短prompt旧实测cached0仅是历史证据；新各请求仍查Generation.cached_prompt_tokens=0/prefix_sourceNone，缺少协议缓存字段记unknown，不能硬填0。
不要新增prefix-cache-dir、HEAD /api/chat自动load、keep_alive0或改变prefill recipe；省略keep_alive最接近原CLI(no TTL)。Ollama的num_ctx等loader选项会被忽略，不能用于设置服务端capacity（protocol.py:419–420）。
stream worker finally释放FIFO（serve.py:549–553）；消费者finally只设置cancelled（606–607），后端pump按取消回调退出。排队阶段同步lock.acquire不检查断连/超时；客户端断开不证明已移出队列/服务端idle，不能用断连代替owned TERM/KILL/wait/reap。
保留现有严格资源门、clean parent、50/50/20/0、MONOLITH_ACCELERATOR=on、noDraft、单GPU所有权与原pack/max-context2048/keep-warm0/no-warmup。一个服务端和C≤4个等待handler；没有增加C份GPU模型驻留，但当前没有实测C2/C4峰值或速度。

可执行CLI形状/配置和资格边界见 `launch-plan.json`；这是root待执行计划，非已经运行。CPU复算：被测Lithos `.venv/bin/python -B check_cpu.py`。
源码精确路径/SHA/permalink见 `source-map.json`；官方pinned设计 `docs/design/serving.md:41–43` 也明确“一generation，其它429，Ollama排队”。
