# 04 批量 decode、紧凑 cache、第一个 Metal kernel

日期：2026-10-13/14。文件：`minivllm/cache.py`（DecodeBatch）、`minivllm/forward_context.py`、`minivllm/engine/model_runner.py`、
`minivllm/engine/llm_engine.py`、`minivllm/sampler.py`、`minivllm/kernels/gdn_metal.py`、`tests/test_engine.py`。

## 先测 MLX 的三个事实

| 问题 | 结果 | 决定 |
|---|---|---|
| `mx.fast.rope` 能不能每行不同位置 | 可以（offset 传数组） | 批量 decode 一次 RoPE |
| 往 128 MB 的 buffer 散写 8 行是不是原地 | **不是，整块拷贝**（0.24 ms） | 不能用"大池子 + 按槽散写" |
| 按槽 gather 32 行 | 0.74 ms（拷贝） | 同上 |

MLX 的数组不可变，`a[idx] = b` 是生成新数组。"固定槽位的大池子"在 CUDA 上理所当然，在 MLX 上每步要把 4.8 GB 的 GDN 状态池拷一遍（10 ms）。

## 设计：紧凑的批量数组，每步整体替换

decode batch 的 B 个请求排成紧凑数组：注意力层 K/V `[B, Hkv, L, D]`，GDN 层 conv `[B, 3, C]`、S `[B, Hv, 128, 128]`。
- K/V 每行按**绝对位置**写（第 i 行写第 pos_i 列，一次 scatter），读时用长度掩码。不需要 mlx-lm 那种左对齐再整体右移。
- GDN 状态每步整体替换：递推的输出就是新状态，零拷贝。
- 请求进出 batch 时重排一次（gather + concatenate）。每个请求一生两次。
- prefill 阶段每个请求用自己的单序列 cache（第 2 步的类），算完 prompt 才并进 batch。

于是 ModelRunner 一步 = 一次批量 decode 前向（所有在跑的）+ 若干次单序列 prefill 块前向。模型的每一层有 `_prefill` 和 `_decode` 两条路。

## 验证：批量不变性

fp32 的 0.8B：单请求 == HF greedy 32 个 token；三个请求合批 + 预算 8 的分块 prefill == 各自单跑；5 个请求挤 2 个槽，EOS 正常停、槽全部归还。

## 第一个 Metal kernel：GDN 的 decode 一步

27B decode 一步的 profile（并发 32）：GDN 状态更新 90 ms、MLP 108 ms、注意力 24 ms、lm_head 6 ms，共 260 ms。
MLP 是 10 TFLOPS 的算力上限，动不了；GDN 状态更新是纯 MLX 把 100 MB 的状态读三遍写一遍（214 GB/s），这就是 mlx-lm 手写 Metal 的地方。

`mx.fast.metal_kernel`：一个 threadgroup 管一个 (序列, 头) 的 128×128 状态，线程 t 管第 t 列，dk 再拆 4 段提高小 batch 的并行度：
```
第一遍  kv[t] = Σ_dk k[dk]·S[dk, t]                      读 S
        δ[t]  = (v[t] − e·kv[t])·β
第二遍  S'[dk, t] = e·S[dk, t] + k[dk]·δ[t]；o[t] += q[dk]·S'[dk, t]    读 S，写 S'
```
相邻线程读写相邻地址，访存合并。q/k 的 L2 归一化、缩放、exp(g) 也折进去，每层省十来个小算子。

| 并发 | 纯 MLX | Metal | GDN 48 层合计 | decode 一步 | tok/s |
|---|---|---|---|---|---|
| 1 | 0.66 ms | 0.55 | 26 ms | 39.5 ms | 25.3（mlx-lm 24–29） |
| 8 | 0.66 | 0.34 | 17 | 88 | 90（mlx-lm server 48） |
| 32 | 1.86 | 0.79 | 38 | 186 | 172；**持续负载下 250 ms → 127** |

## 踩的坑

- B=16 比 B=8 还慢：MLX 的 nvfp4 `quantized_matmul` 在 M=16 和 M=32 一样贵（0.65 ms vs M=8 的 0.37）。M 在 9–16 之间是最差的点，先记下。
- **MacBook 的 GPU 撑不住峰值**：250 步连续 decode，前 20 步 195 ms，之后稳定在 250 ms。跑分要看持续数字，不要看前几步。
- 第一次 decode 步 460 ms：包含把 32 个请求的 cache 并进 batch 的拷贝，之后 207 ms。
