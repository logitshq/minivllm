# V130 保存结果独立复核
PASS_WITH_SCOPE_LIMITS；只读保存证据，无 GPU、网络、live PID/port 探测或源修改。V130 源码由另一作者完成。
- 性能：9 fresh arms，54 groups，126 HTTP / 4032 输出；其中正式 27 groups / 63 请求 / 2016 输出。
- 资格另计：4 groups / 8 HTTP / 256 输出（1 OpenAI、7 Ollama）；合计 134 请求 / 58 groups。
- 逐请求重算 pre-POST→协议终帧时钟；逐 group p50 与吞吐，再 3 formal group→arm median、3 fresh→配置 median，全吻合。
- C1：1511.125000 ms / 21.1762759535 tok/s；C2：2326.114458 ms / 20.7543789802；C4：3797.967271 ms / 21.1008248112。
- 全部文本由 CPU tokenizer 独立重新编码，等 saved visible32 gold；usage 19/32、length、无重试、NDJSON 唯一 done:true 均通过。
- 内部 raw token IDs 不可得；visible roundtrip 与 usage 检查不能改写为内部 raw32 或整模张量资格。
- 126+8 条实际 Generation 记录逐条等 participation，cached_prompt_tokens=0、prefix_source=None；m4_max / draft=None。
- 全部 preimport 为同一 196 文件源树身份、50/50/20/0；冻结 218 项 pins 的 hash 与主三脚本已核，未重新审计整棵 vendor 源树。
- 资源：性能 135 arm+27 campaign+1 release=163，资格 11+1=12，共 175；AC / thermal0 / pressure1。
- 全部样本 Swapouts=17554594、swap usage=1065.94 MiB；采样均在 HTTP 时钟外，不能排除样本间瞬态。
- 8 个固定、非自适应卸载冷却间隔均 ≥60s，均在请求时钟外。
- 全部 owned server 已保存 wait(-15)/birth identity/严格 absent 与拒连；perf 11 PID/9 port、qual 3 PID/1 port 外部释放绑定通过。
- 此处复用 root 当时实时释放证据；审查者没有 fresh live absence 检查。owner/driver rc0 与 receipt/source/release SHA 已绑定。
- participation.original_log_sha256 是卸载前日志前缀身份；10/10 前缀精确匹配，清理追加部分无 Generation。最终日志另行 sealed。
- per-arm no_new_performance_point=true 是继承资格元数据占位；suite 与正式原始时钟确有测量，不能据该占位否认或扩大测量。
- 客户端 start barrier 只证明 pre-POST 时间先于首终帧，不证明 wire arrival 公平性；排队等待计入 X，精确等待时长未知，无 GPU batching 声明。
- 与 V128 同图只能作标明 campaign/协议的描述性比较；无配对增益、显著性、自动采用或完整同轮 Pareto 胜出声明。
复算入口：`recompute.py`（CPU tokenizers 依赖及原始私有路径；不是离机全证据重放包）；安全结果见 verification.json/recomputed-rows.json。
