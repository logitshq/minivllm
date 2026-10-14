# minivllm（本地版）— 手搓计划

> 远程版（surf-gpu，CUDA/Triton，目标 27B NVFP4 打平 vLLM）由另一组 agent 并行在做，接口契约见 `ARCHITECTURE.remote.md`。
> 本地版在 M4 Max 上用 PyTorch MPS，目的是**一步一步带着搭**：同一套设计、同一个 ForwardContext 契约、同一个 dev 模型（Qwen3.5-0.8B），
> kernel 全部用纯 torch 写（Triton 在 Mac 上没有）。本地跑通的 engine 代码可以原样搬到远程，只换 kernels/。

## 为什么是 PyTorch MPS 而不是 MLX

- 和远程共用 model/ engine/ 代码，学的是 vLLM 的结构，不是 Apple 的 API。
- torch 2.14 的 MPS 已有 bf16 GEMM（实测 3.5 TFLOPS）、GQA causal SDPA、index scatter，够做 paged KV 的纯 torch 版。
- MLX 更快，但另起一套；等引擎跑通之后如果想榨 Mac 性能再做一版 kernels/mlx。

## 本地没有 vLLM，拿什么当基线

- 正确性基线：HF transformers 的 greedy 输出（见 `tutorial/00-环境与基线.md`），我们的引擎必须逐 token 相同。
- 性能基线：HF generate batch=1（110 tok/s）；后面加上 mlx-lm server 作为"Mac 上最接近 vLLM 的东西"。
- 远程契约里的 bench 命令（1024 in / 1024 out / 100 prompts / 并发 32）本地用自己的 client 复刻，参数一致。

## 步骤（每步 = 一个文件 + 一个测试 + 一页教程）

| # | 做什么 | 验证什么 | 对应远程 |
|---|---|---|---|
| 0 | venv、下模型、HF 基线 | 48 个 greedy token id 存档 | setup |
| 1 | `model/`：Qwen3.5 纯 torch 前向 + safetensors loader（无 cache，整句一次性前向） | logits vs HF max abs diff | A1 |
| 2 | `kernels/`：paged KV（slot_mapping / block_table）+ GDN 状态槽；prefill + 逐 token decode | decode 序列 == 一次性前向；greedy == 步骤 0 | A2 A3 |
| 3 | `engine/`：Request、BlockPool/KVCacheManager、Scheduler（V1 统一调度、token budget、chunked prefill、抢占） | CPU 单测，200 请求模拟 | A5 |
| 4 | `ModelRunner` + `LLMEngine` + `Sampler`：ForwardContext 拼装，continuous batching | greedy == HF；32 并发吞吐 vs batch-1 | B |
| 5 | `AsyncLLMEngine` + OpenAI server + bench client | 1024/1024/100/并发 32 跑通，记录 TTFT / TPOT | A6 C |
| 6 | 优化：prefix caching、合并投影、减少 H2D、（可选）MLX kernels | 每项给前后数字 | C |

## 规则

- `minivllm/` 不 import vllm，不 import transformers 的模型类（只用 tokenizer + config）。
- 先写测试再改代码；每步结束写 `docs/tutorial/NN-*.md`，写清为什么这样设计、否掉了什么、测到的数字、踩的坑。
- 最少代码。没有消费者的功能不写。
