# 02 KV cache、GDN 状态、分块 GDN

日期：2026-10-13。目标：decode 一步只算一个新 token。文件：`minivllm/cache.py`（55 行）、`minivllm/kernels/gdn.py`（110 行）、
`tests/test_cache.py`、`tests/test_gdn_kernel.py`。

## 两种层存的东西不一样

| 层 | 存什么 | 大小（27B，每个请求） | 随长度增长？ |
|---|---|---|---|
| 全注意力（16 层） | K、V 各 [Hkv=4, T, 256] bf16 | 每 token 64 KB | 是，线性 |
| GDN（48 层） | 卷积尾巴 [3, 10240] + 状态 S [48, 128, 128] fp32 | 每层 3.2 MB，共 151 MB | **否** |

这就是混合架构的意义：2048 token 的上下文，KV 只占 128 MB，GDN 状态 151 MB 固定。普通 Qwen3-27B 同样上下文要 2 GB 以上。

AttnCache 按 256 个 token 一块预分配、按位置写入（slice 赋值），和 mlx-lm 的 KVCache 一样。vLLM 的 paged cache 是它的推广：把"每个序列一段连续内存"变成"每个序列一张 block 表"，第 4 步讲。

## 唯一的正确性判据

**prefill 之后逐 token decode 的 logits == 整句一次前向在对应位置的 logits。** fp32 下每个位置差 < 0.01，30 步 greedy 相同；
分块 prefill（7 个一块）== 一次性 prefill；bf16 带 cache 和不带 cache 的 greedy 32/32 相同。

细节：注意力带 cache 时 T 个 query 对 offset+T 个 key，MLX 的 `mask="causal"` 按右下角对齐，刚好是分块 prefill 要的语义。
RoPE 的位置从 `cache.offset` 起算。

## 分块 GDN：prefill 从 103 到 208 tok/s

逐 token 递推 prefill 1024 个 token 要 1024 × 48 层的小算子，27B 只有 103 tok/s。分块算法（HF 的 `torch_chunk_gated_delta_rule`，源自 FLA）
把 64 个 token 的 delta rule 合并成几次矩阵乘，块间再递推：

1. 块内累计衰减 `cum = cumsum(g)`，两两衰减 `exp(cum_i − cum_j)`（上三角为 0）。
2. 解一个单位下三角方程组 `(I + A) u = β·v`，A = β·(k kᵀ)·衰减。u 是"块内 delta rule 合并后的新值"。
3. 对每块：读旧状态 `q·S`，加块内注意力 `(q kᵀ·衰减)·u'`，更新 `S = S·exp(cum_末) + kᵀu'`。

| 27B nvfp4 | prefill tok/s |
|---|---|
| 逐 token 递推 | 103 |
| 分块（64） | 128 token：162，512：213，**1024：208**，2048：188 |
| mlx-lm 同一 checkpoint | 138 |
| mlx-lm affine 4bit | 241 |

## 踩的坑：显式求逆在真实数据上会炸

第一版用幂级数显式求 `(I + A)^{-1} = (I−A)(I+A²)(I+A⁴)…`（A 严格下三角，A⁶⁴ = 0，6 次矩阵乘精确）。随机输入下误差 1e-7，
**真实数据下 logits 误差从第 16 个位置开始滚雪球，到第 59 个位置差 10**。原因：真实序列里相邻 token 的键高度相关（k_i·k_j ≈ 1）、β ≈ 1，
(I+A)^{-1} 的元素按 2^(i−j) 增长再相消，fp32 扛不住。解本身（u）是有界的，所以改成前向代入直接解，不形成逆矩阵，误差回到 1e-4。
教训写成了测试 `test_chunk_with_correlated_keys`：kernel 测试必须覆盖真实数据的统计形状，随机正态是最友好的输入。

## 速度与内存（27B nvfp4，单流）

decode 22.3 tok/s（mlx-lm 24–29，差距是每步 64 层的 Python 算子开销，第 6 步用 mx.compile 收）。
prefill 2048 token 峰值 20.1 GB，其中 1 GB 是对全部 2048 个位置算了 lm_head：引擎里只对最后一个位置算（第 4 步）。
