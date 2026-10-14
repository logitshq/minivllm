# 05 server、async 引擎、和 mlx-lm 的正面对比

日期：2026-10-14。文件：`minivllm/engine/async_engine.py`、`minivllm/server.py`、`minivllm/tokenizer.py`、`tests/test_server.py`、`bench/run_ours.sh`。

## 三个自己写的东西

- **分词 + chat 模板**：`tokenizers`（Rust 库，只读 tokenizer.json）+ 十行手写的 ChatML。增量 detokenize：一个 token 可能只是一个 UTF-8 字符的一部分，decode 出来末尾是 � 就先攒着。
- **AsyncLLMEngine**：一个后台任务不停 `step()`，每个请求一个 `asyncio.Queue` 收 token；没活时等 event。step 是同步的，GPU 在算的时候 Python 在等，每步之间 `await asyncio.sleep(0)` 让 HTTP 层收发。
- **HTTP server**：纯 `asyncio.start_server`，自己解析请求头和 Content-Length，流式响应用 chunked + SSE，`data: [DONE]` 收尾。
  支持 `/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/health`，认 `ignore_eos`（vLLM 扩展）和把 EOS 压到 −100 的 `logit_bias`（等价 ignore_eos）。没有框架，120 行。

## 跑分（同一台 M4 Max，同一 nvfp4 checkpoint，同一 client，缩短协议：32 条 prompt，1024 进 256 出）

| | 并发 8 | | 并发 32 | |
|---|---|---|---|---|
| | mlx-lm | **minivllm** | mlx-lm | **minivllm** |
| 完成 | 32/32 | 32/32 | **崩溃**（Metal OOM，38 GB） | 32/32 |
| 端到端输出 tok/s | 27.8 | 25.2 | — | **32.8** |
| 平均 TTFT | 32 s | **24 s** | — | 90 s |
| 平均 TPOT | 165 ms | 225 ms | — | 605 ms |
| 进程峰值 | 17 GB | 18 GB | 38 GB | **24 GB** |

并发 32 是 mlx-lm 做不到的事（内存多占 13 GB）。并发 8 我们端到端略慢（25 vs 28）：
它 2 条一起 prefill（M=2048 的 GEMM 更高效）而且 prefill 时 decode 全停；我们分块 prefill 和 decode 交错，TTFT 低 25%，
但 decode 的人每遇到一个 512 的块就等 2 秒（TPOT 225 ms、ITL p99 3 s）。这是"TTFT 低"和"TPOT 稳"之间的取舍，vLLM 也一样。

引擎本身（不经 server）跑同样的 32 条并发 8：289 s，其中 decode 975 步 115 s（118 ms/步）、带 prefill 块的步 69 步 170 s。
server 层多了 37 s（13%），asyncio 的逐 token 写 SSE，待查。

## 踩的坑

- **第一次跑并发 32 整机 20 tok/s**：桌面 app 把 swap 吃到 12 GB。跑分前退出所有 app，脚本里记 swap 和进程占用。
- **MLX 的 allocator 缓存**：释放的 buffer 留着复用，每步的 K/V 拷贝尺寸各不相同，几分钟攒出 20 GB，逼近 37 GB 工作集后系统压缩内存。
  `mx.set_cache_limit(2 GB)` 之后并发 32 的峰值从 31 GB 降到 24 GB，端到端从 28 提到 33 tok/s。dflash-mlx 默认 4 GB 也是为这个。
- mlx-lm 的 server 默认把跑完的请求的 KV 留在 LRU 里复用，`--prompt-cache-bytes 1` 关掉才公平。

## Pareto 曲线（2026-10-14 补）

`bench/run_pareto.sh` 扫并发 1 到 32，`bench/pareto.py` 画成 `docs/pareto.html`（InferenceX 的画法：x = 每用户 tok/s，y = 整机 tok/s）。
并发 1 到 16 两条线重合（kernel 都是 MLX 的）；24 和 32 时 mlx-lm 的静态 batch 是 116 和 142，我们是 103 和 115，差 20%，原因没查；
但它的 server 并发 8 只有 48、并发 32 OOM，静态 batch 的数字它自己的 server 拿不到。单用户开 DFlash2 的三个点在最右边。
