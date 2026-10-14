# minivllm（本地版）— 手搓计划 v2

> 远程版（surf-gpu，CUDA/Triton，目标 unsloth/Qwen3.8-27B-NVFP4 打平 vLLM）由另一组 agent 并行在做，接口契约见 `ARCHITECTURE.remote.md`。
> 本地版在 M4 Max（48 GB）上用 **MLX** 一步一步带着搭，目标和远程同级：**Qwen3.8-27B NVFP4 + 投机解码**，对标 `mlx_lm.server`。

## 2026-10-13 的两个决定

1. **引擎用 MLX 写，不用 torch MPS。** torch MPS 的 kernel 比 MLX 慢一倍以上（0.8B bf16 单流 110 vs 220 tok/s），而且 torch 没有 Mac 上的 NVFP4 kernel；27B 只有量化才装得下。`engine/`（调度器、block pool、request）是纯 Python，和远程完全共用；`model/` 和 `kernels/` 本来就要按后端重写（远程 Triton，本地 Metal）。
2. **目标模型 Qwen3.8-27B，用 `mlx-community/Qwen3.8-27B-nvfp4`（16.1 GB）。** MLX 0.32 原生支持 `nvfp4` 量化模式（e2m1 + e4m3 组内 scale，group 16），和远程 NVIDIA NVFP4 是同一种数值格式，`mx.quantized_matmul(mode="nvfp4")` 直接可用。bf16（55.6 GB）和 FP8（30.9 GB）在 MLX 推荐工作集 37.4 GB 下装不下或装不下 cache。

内存预算（27B nvfp4，并发 32，上下文 2048）：权重 16.1 GB + KV（16 层 × 4 kv 头 × 256 × 2 × bf16 = 64 KB/token，32 × 2048 token → 4.3 GB）+ GDN 状态（48 层 × 48 × 128 × 128 × fp32 = 151 MB/请求 → 4.8 GB）≈ 25 GB，在 37.4 GB 之内。

算力预算：27B 每 token 54 GFLOP。M4 Max 实际 bf16 GEMM 约 10 TFLOPS 量级，1024 token 的 prefill 约 5 s；decode 并发 32 每步约 0.2 s，合计约 150 tok/s。远程契约里的 bench（1024/1024/100 请求/并发 32）本地一次约 20 分钟，所以**所有开发在 0.8B 上做，27B 只做端到端和跑分**。

## 模型

| 用途 | 模型 | 位置 |
|---|---|---|
| 开发（和 27B 同架构、同量化格式） | Qwen3.5-0.8B nvfp4（自己 convert，4.5 bit/weight） | `models/Qwen3.5-0.8B-nvfp4` |
| 开发（逐 token 对齐 HF） | Qwen3.5-0.8B bf16 | HF cache |
| 目标 | mlx-community/Qwen3.8-27B-nvfp4 | HF cache |
| 投机解码草稿 | 待定：Qwen 自带 MTP 头（`mlx-community/Qwen3.8-27B-MTP-nvfp4`，0.2 GB；mlx-lm 0.32 还不支持，得自己实现）或 DFlash2（`incoai/Qwen3.8-27B-DFlash2`，3.8 GB bf16，远程在做的那个） | 第 6 步决定 |

## 基线（mlx-lm 0.32，`mlx_lm.benchmark` 静态 batch，1024 in / 1024 out）

| 模型 | batch 1 decode tok/s | batch 32 decode tok/s |
|---|---|---|
| 0.8B bf16 | 220 | 1812 |
| 0.8B nvfp4 | 441 | 2614 |
| 27B nvfp4 | 28.8（p=256 g=64） | 68（b=8 已饱和在 76） |

单流投机解码基线：oMLX + MTP k=3 在 M4 Max 上 53 / 72 tok/s（文 / 代码）；dflash-mlx（装在 `.venv-mlx`）。这台机器上 verify k 个 token 的成本：k≤4 免费，k=5 +32%，k=8 ×2（见 tutorial 00）。
serving 基线：`mlx_lm.server --decode-concurrency 32 --prompt-concurrency N` + 自己写的 bench client（复刻 `vllm bench serve --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 32`），记 TTFT / TPOT / 吞吐。
正确性基线：HF transformers bf16 greedy（`tutorial/00-环境与基线.md`），我们的 bf16 路径必须逐 token 相同；nvfp4 路径对比 mlx-lm 同一 checkpoint 的 greedy。

## 步骤（每步 = 一个文件 + 一个测试 + 一页教程）

| # | 做什么 | 验证什么 | 对应远程 |
|---|---|---|---|
| 0 | venv、模型、基线 | 48 个 greedy token id；mlx-lm 数字 | setup |
| 1 | `model/`：Qwen3.5 的 MLX 前向 + loader（bf16 和 nvfp4 两种 linear），无 cache 整句前向 | logits vs HF；nvfp4 logits vs mlx-lm | A1 A4 |
| 2 | `kernels/`：paged KV（slot_mapping / block_table）+ GDN 状态槽，先 gather + 现成算子做对，再写 Metal kernel | decode 序列 == 一次性前向；greedy == 步骤 0 | A2 A3 |
| 3 | `engine/`：Request、BlockPool / KVCacheManager、Scheduler（V1 统一调度、token budget、chunked prefill、抢占） | CPU 单测，200 请求模拟 | A5 |
| 4 | `ModelRunner` + `LLMEngine` + `Sampler`：ForwardContext 拼装，continuous batching，`mx.compile` / lazy eval 的 step 图 | greedy == HF；并发 32 吞吐 vs mlx-lm batch 32 | B |
| 5 | `AsyncLLMEngine` + OpenAI server + bench client；27B 端到端 | 1024/1024/100/并发 32 对比 `mlx_lm.server` | A6 C |
| 6 | 投机解码（MTP 头或 DFlash2）+ 优化（prefix caching、投影合并、Metal kernel 调优） | 每项前后数字 | D |

## 规则

- `minivllm/` 不 import vllm、不 import mlx_lm、不 import transformers 的模型类（只用 tokenizer + config）。mlx-lm 的源码可以读、可以抄思路，不能 import。
- 先写测试再改代码；每步结束写 `docs/tutorial/NN-*.md`，写清为什么这样设计、否掉了什么、测到的数字、踩的坑。
- 最少代码。没有消费者的功能不写。
