# V130: official Lithos queued-load qualification (CPU draft)

最小 root-only 资格：一个 fresh 官方进程，OpenAI C1 1请求 → Ollama C1 1请求 → Ollama C2 2请求 → Ollama C4 4请求。8HTTP/256输出，全相同19-token chat fixture、greedy、正常EOS/max32，无重试、无外部队列。固定 `--power AC|Battery`；若切换或thermal非0/pressure非1/Swapouts或swap-used增加，STOP并清理，不能混入旧AC图。

资格要求两个C1及C2/C4所有请求的visible32等旧实际OpenAI visible gold、prompt19/output32/length；official raw token IDs没有暴露。C>1 使用barrier；所有客户端pre-POST t0均早于第一个terminal，不冒称已观测wire/server到达顺序；不称GPU batch。请求status/拒绝/error/时钟/usage/增量内容完整保留，失败group主指标为null，不因丢弃拒绝而制造曲线。OpenAI终点是SSE `[DONE]`，Ollama终点是NDJSON `done:true`；POST前→相应terminal含服务排队，清理不含。不能拿eval_duration代替完整延迟。

原pinned `serve.py:482–508` 的非Ollama忙时429，Ollama使用同一FifoLock blocking acquire；`/api/chat:662–668`转换到同Backend。`FifoLock lifecycle.py:26–46`只保证取得内部ticket之后的顺序；网络到达/FastAPI线程调度先后未知，不宣称HTTP公平性或并行GPU batching。stream worker finally释放锁，断开客户端不保证排队请求立即取消，必须原owner TERM→wait/reap→PID/port absent。

旧429证据更正：保存的chapter11 contract/gap-audit描述源代码语义；定向检查没有找到可证明请求级429分布的原始HTTP响应。不能把源码推断叫已实测C2/C4拒绝率。本轮客户端如实记录状态，不自动重试；旧OpenAI baseline/source/结果不改。

原官方launch使用 `official_command('lithos',1,...)`，其中1只是继承原服务器命令配置，无自定义queue、workers、多Session或vendor修改。C请求数仅在本客户端选择。keep_alive不传，保留原CLI生命周期行为。engine Generation日志必须实际m4_max/noDraft且8条cached_prompt_tokens0；NDJSON缺少cached字段不填0。

`driver.py`与`plan-nine-arms.json`准备下一性能切片，但run硬绑定本源资格PASS和root释放文件：3个C各3fresh，旋转C1/C2/C4、C2/C4/C1、C4/C1/C2；每fresh3warm3measure，共126HTTP/4032输出（正式63/2016），8固定60s unloaded间隔。它尚未执行。Root已选择Battery Lithos三配置与旧V128 AC七配置分面呈现；不跨域pool/支配比较，不重新称同一轮完整Pareto。

CPU检查不importMLX、不加载dylib/模型、不起服务器。`qualify.py --cpu-dry-run --power Battery --pins-sha256 <SHA>`安全；实际必须root独占GPU与600s有界owner运行 `--run --out <NEW ABS> --power Battery`。新目录资格marker一次性；CPU-READY提供确切CLI。性能driver同样显式固定电源，跨arm追加全局guard、8个固定60s卸载冷却；实际资格和释放原始SHA均由root作为run参数绑定。本目录仍无数值/性能资格。
