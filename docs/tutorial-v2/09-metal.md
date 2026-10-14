# 09：数学不变，让 GDN 在 GPU 上少走几趟

第八章已经能合批生成。本章优化执行方式：把 GDN 单步的多个运算融合到一个 Metal kernel，并把 chunk 内的三角求解搬进 GPU kernel。

| 之前 | 本章之后 |
|---|---|
| batched GDN 用通用 MLX 运算组合 | 自定义 Metal 执行同一个状态更新 |
| chunk 三角求解由 Python 逐行组织 MLX | 一个 Metal kernel 内按行前向代入 |
| 只有数学与引擎对照 | 加上 kernel 数值对照与执行路径记录 |

代码版本：`tutorial-09`；上一个版本是 `tutorial-08`。核心文件 [gdn_metal.py](../../tutorial_code/miniengine/kernels/gdn_metal.py) 整理移植自原 [minivllm Metal kernel](../../minivllm/kernels/gdn_metal.py)；原 MLX 参考函数保留，方便验证。**是否更快需要计时，写成 Metal 本身不证明提速。**

## 1. 先看单步：大头是读写 S

第四章的数学保持不变：

\[
\bar S=e^g S,\quad\delta=\beta(v-k^\top\bar S),
\]

\[
S_{new}=\bar S+k\delta^\top,\quad o=q^\top S_{new}.
\]

Q/K 仍需 L2 归一化，Q 再乘 \(D_k^{-1/2}\)。Metal 路径把它们放在 kernel 内，调用前不能再归一化一次。

一个头的状态是 `Dk×Dv`。假设两个维度都是 128，单头有 16,384 个 fp32 数；很多头、很多层、多个请求一起运行时，反复读写这些状态会产生明显流量。

```text
通用算子组合：归一化 → exp → 读状态预测 → delta → 写状态 → 读状态输出
融合 kernel： 进入一个 kernel，完成这些步骤，返回 S_new 和 o
```

融合减少中间 tensor 与 GPU 调度开销。它没有减少模型的层数，也没有把结果近似成另一套公式。

## 2. Metal 的线程怎样分工

无需先记所有 Metal 语法，先看工作分配：

```text
一个 (请求 b, 头 h) → 一个 threadgroup
S 是 Dk 行 × Dv 列
每条线程负责一个 value 列，以及 key 方向上的一小段
  ↓ 合并各段的 kᵀS，得到完整预测
  ↓ 计算 delta，各线程更新自己的状态片段
  ↓ 合并各段的 qᵀS_new，得到输出
```

这里 key 方向拆成 4 段，所以需要两次跨段归约和同步。输出列与状态片段之间没有随意互相写入，分工明确才能避免数据竞争。

源码的关键节选：

```cpp
float delta = (v[...] - ex * kv) * be;           // ← 关键① 预测包含旧状态衰减
for (...) {
    float s = ex * S[...] + normalized_k * delta;
    S_new[...] = s;                             // ← 关键② 更新状态
    acc += normalized_scaled_q * s;             // ← 关键③ 顺便累积输出
}
```

`kv` 是归约后的 `kᵀS_old`，先乘 ex 才得到 `kᵀS_bar`。这与第四章“先衰减，再预测”完全一致，只是调整了运算组织。

当前实现有形状边界：Dk 必须能被 4 整除、Dv 不大于 128。它服务于这里的模型维度，不能声称适用于任意 GDN 配置。

## 3. 多 token 为什么能用 chunk

第六章已使用纯 MLX chunk 计算。直觉是：一块内的状态影响可以写成矩阵运算，块与块之间继续传递末尾 S。

从逐 token 更新展开可得到每个 token 的修正量 \(\delta_i\)：

\[
\delta_i=\beta_i\left(v_i-k_i^\top e^{G_i}S_{start}
-\sum_{j<i}e^{G_i-G_j}(k_i\cdot k_j)\delta_j\right),
\]

其中 \(G_i=\sum_{r\le i}g_r\) 是块内累计 log 衰减。前面 token 的写入影响后面的预测，因此修正量不是简单并行的 `β×v`。

把相互影响收进严格下三角矩阵 A，写成：

\[
(I+A)\Delta=V_{rhs}.
\]

这就是 chunk 实现里需要三角求解的原因。后面的输出与块末状态，再由这些修正量和矩阵乘计算。

一个只有两行的例子：

```text
x0            = 2
0.5*x0 + x1   = 4
所以 x0=2，x1=4−0.5×2=3
```

第 i 行只依赖前面已经求好的行，叫 forward substitution（前向代入）。这里不显式构造逆矩阵，避免无谓运算和中间值相消放大的数值问题。

## 4. 第二个替换点：三角求解

第六章纯 MLX 参考版的主线：

```python
rows = [V[..., 0:1, :]]
for i in range(1, C):
    X = mx.concatenate(rows, axis=-2)
    rows.append(V[..., i:i+1, :] - A[..., i:i+1, :i] @ X)
```

它按正确顺序求解，但 Python 每行组织多个小 GPU 运算。Metal 版在一个 kernel 内完成相同顺序，列可以并行，行之间同步：

```cpp
for (uint i = 0; i < C; i++) {
    float acc = Vn[i * M + m];
    for (uint j = 0; j < i; j++)
        acc -= An[i * C + j] * Xn[j * M + m];   // ← 关键④ 读已解的前面行
    Xn[i * M + m] = acc;
    threadgroup_barrier(mem_flags::mem_device); // ← 关键⑤ 下一行前确保可见
}
```

完整 chunk 公式可对照 [HF 的 torch_chunk_gated_delta_rule](../../ref/modeling_qwen3_5.py) 与 [教学 gdn.py](../../tutorial_code/miniengine/kernels/gdn.py)。初学者先抓住“相同 delta 递推，块内用下三角关系批量求修正”即可。

## 5. 代码替换与实际接入

```diff
  # 第八章 GDN batched decode
- o, S = gated_delta_step_batch(q, k, v, g, beta, st.S)
+ o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)

  # 第六章 chunk 内求解
- return _solve_unit_lower(A, V)
+ return solve_unit_lower_metal(A, V)
```

这里两个优化都接入引擎：多请求 decode 调第一种 kernel，多 token GDN chunk 调第二种。单 token 单序列的参考递推仍保留；“用了 Metal”不等于所有模型运算都走自定义 kernel。

## 6. 验证计算，也验证确实执行了 Metal

```bash
python -m tutorial_code.check_chapter09 \
  --out tutorial_code/examples/chapter09-checks.json
python -m tutorial_code.check_chapter08 \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 12 --out tutorial_code/examples/chapter09-engine-checks.json
git diff tutorial-08 tutorial-09 -- tutorial_code/miniengine
```

[kernel 报告](../../tutorial_code/examples/chapter09-checks.json) 比较 B=1/3 的单步输出与新状态，以及 chunk size=8/16/64 的三角求解。记录 dispatch counter，并在结果 `mx.eval` 完成后保存；函数可导入不等于实际 GPU 路径已经执行。

[引擎报告](../../tutorial_code/examples/chapter09-engine-checks.json) 重新核对合批与串行输出，检查局部优化没有破坏请求级行为。计时必须对相同形状、输入和精度预热后重复执行，并等待 GPU 完成；本章还分别对 MLX 参考与 Metal 做 2 次预热、5 次完整求值计时，包含 Python 建图与发射开销；第十一章测完整引擎，两者的边界不同。

[下一章：10 投机解码](10-speculative.md)

## 本次局部计时怎么看

本次 fp32 微基准，GDN step 的中位耗时：B=1 从约 0.393 ms 到 0.270 ms，B=3 从约 0.418 ms 到 0.248 ms；chunk=64 的三角求解从约 2.550 ms 到 0.551 ms。这是给定小形状、2 次预热和 5 次重复、包含 Python 建图与执行的局部测量，样本可见报告。它说明我们确实测到了这两个局部路径，不能据此推导完整 27B 引擎加速倍数。

另外用 73 个 token、chunk=16、非零初始状态核对递推与分块，最后一块只有 9 个真实 token，输出/状态误差为约 1.49e-8 / 1.49e-7。padding 没有把额外时间步写进最终状态。

## 本机验证记录

完整结果见 [chapter09-checks.json](../../tutorial_code/examples/chapter09-checks.json)。这是本章代码实际运行的结果：

```json
{
  "checks": [
    {
      "kernel": "decode_step",
      "batch": 1,
      "shape": [
        1,
        4,
        128,
        128
      ],
      "dtype": "float32",
      "output_max_abs": 9.313225746154785e-09,
      "state_max_abs": 5.960464477539063e-08,
      "mlx_reference_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.00045900000259280205,
          0.00039308299892582,
          0.0003635419998317957,
          0.000630208000075072,
          0.0003345420118421316
        ],
        "median_s": 0.00039308299892582,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      },
      "metal_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.00023441600205842406,
          0.0002274169964948669,
          0.0002710410044528544,
          0.00026950000028591603,
          0.00044533399341162294
        ],
        "median_s": 0.00026950000028591603,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      }
    },
    {
      "kernel": "decode_step",
      "batch": 3,
      "shape": [
        3,
        4,
        128,
        128
      ],
      "dtype": "float32",
      "output_max_abs": 8.381903171539307e-09,
      "state_max_abs": 5.960464477539063e-08,
      "mlx_reference_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0003490000090096146,
          0.0003885000041918829,
          0.00042891599878203124,
          0.0004181669937679544,
          0.0005873340123798698
        ],
        "median_s": 0.0004181669937679544,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      },
      "metal_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0002472500054864213,
          0.00024458300322294235,
          0.00036204200296197087,
          0.0002795409964164719,
          0.00024774999474175274
        ],
        "median_s": 0.00024774999474175274,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      }
    },
    {
      "kernel": "unit_lower_solve",
      "chunk": 8,
      "shape_A": [
        2,
        8,
        8
      ],
      "shape_V": [
        2,
        8,
        256
      ],
      "dtype": "float32",
      "max_abs": 4.76837158203125e-07,
      "mlx_reference_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0003768330061575398,
          0.0003819590056082234,
          0.0018105829949490726,
          0.0003965419891756028,
          0.0005012080073356628
        ],
        "median_s": 0.0003965419891756028,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      },
      "metal_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0001646250020712614,
          0.00022695799998473376,
          0.00029362500936258584,
          0.00020658300491049886,
          0.0001740830048220232
        ],
        "median_s": 0.00020658300491049886,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      }
    },
    {
      "kernel": "unit_lower_solve",
      "chunk": 16,
      "shape_A": [
        2,
        16,
        16
      ],
      "shape_V": [
        2,
        16,
        256
      ],
      "dtype": "float32",
      "max_abs": 7.152557373046875e-07,
      "mlx_reference_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0006409999914467335,
          0.0005972499930066988,
          0.0020252499962225556,
          0.0006495840061688796,
          0.0008986250031739473
        ],
        "median_s": 0.0006495840061688796,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      },
      "metal_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.00023966700246091932,
          0.00023150000197347254,
          0.0002000840031541884,
          0.000218084009247832,
          0.00022362500021699816
        ],
        "median_s": 0.00022362500021699816,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      }
    },
    {
      "kernel": "unit_lower_solve",
      "chunk": 64,
      "shape_A": [
        2,
        64,
        64
      ],
      "shape_V": [
        2,
        64,
        256
      ],
      "dtype": "float32",
      "max_abs": 1.430511474609375e-06,
      "mlx_reference_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.002468416001647711,
          0.0027721250080503523,
          0.0025501249911030754,
          0.0031441250030184165,
          0.002518624998629093
        ],
        "median_s": 0.0025501249911030754,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      },
      "metal_timing": {
        "warmups": 2,
        "repeats": 5,
        "samples_s": [
          0.0005314160080160946,
          0.0005360419891076162,
          0.0005729999975301325,
          0.0008220839954447001,
          0.0005507500027306378
        ],
        "median_s": 0.0005507500027306378,
        "timing_scope": "one complete evaluated operation; includes graph building and dispatch"
      }
    },
    {
      "kernel": "chunk_vs_recurrent",
      "tokens": 73,
      "chunk": 16,
      "tail_tokens": 9,
      "nonzero_initial_state": true,
      "output_max_abs": 1.4901161193847656e-08,
      "state_max_abs": 1.4901161193847656e-07
    }
  ],
  "evaluated_dispatch_counts": {
    "decode_step": 16,
    "unit_lower_solve": 25
  },
  "scope": "correctness, evaluated Metal execution, and warmed operation-level timing; no full-engine speedup claim"
}
```
