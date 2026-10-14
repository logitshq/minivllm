# 在 M4 Max 上手搓一个跑 Qwen3.8-27B 的推理引擎

2026-10-14 · Ryan Li

## 结论：一台 48 GB 的 MacBook 能把 27B 跑到这个速度

Qwen3.8-27B 用 nvfp4 量化只有 16 GB，在 M4 Max 48 GB 上单流 25 tok/s，开 DFlash2 投机解码 56 到 62 tok/s，结果和不投机逐 token 相同。并发 32 跑得起来、峰值 24 GB，而 mlx-lm 的 server 在并发 32 时直接 Metal OOM；不过它的静态 batch 在并发 24 以上比我的引擎快 20%，这个差距还没追平。整个引擎只依赖 mlx 和 tokenizers，1700 多行 Python，含两个 Metal kernel，每一步都有测试和教程。

| 场景 | 这个引擎 | 对照 |
| --- | --- | --- |
| prefill，1024 token | 248 tok/s | mlx-lm 同一 checkpoint 138；affine 4bit 241 |
| 单流 decode | 25 tok/s | mlx-lm 24 到 29 |
| 单流 + DFlash2，无损 | 56（数学）到 62（代码）tok/s | dflash-mlx 64 |
| 并发 8，decode 阶段 | 90 tok/s | mlx-lm server 48 |
| 并发 32 | 持续 127 tok/s，峰值 24 GB | mlx-lm 的 server Metal OOM，38 GB；它的静态 batch 能到 142 |

所有数字都是同一台机器、同一个 checkpoint、同一个 client 测的，跑分前退出了桌面上所有其他 app。代码：https://github.com/logitshq/minivllm

## 为什么是 Qwen3.8-27B、nvfp4、MLX

选择其实只有一条路：能装进 48 GB 的最强开源模型是 27B，装得下的格式只有 4bit，Mac 上能跑 4bit 矩阵乘的库只有 MLX。

**模型。** Qwen 官方的 3.8 系列只有三个：27B（稠密）、Flash-Next（512 个专家的 MoE，下一代架构）、和 2.4T。27B 的 config 里 `model_type` 就是 `qwen3_5`，架构和 Qwen3.5 完全相同：64 层里 48 层是 Gated DeltaNet（线性注意力），只有 16 层是普通注意力。这个比例决定了后面很多事：KV cache 很小，但每个请求要额外存一份递推状态。开发用同架构的 Qwen3.5-0.8B，因为 3.8 没有小尺寸。

**内存账。** bf16 是 55.6 GB，FP8 是 30.9 GB，nvfp4 是 16.1 GB。MLX 在 48 GB 机器上的推荐工作集是 37.4 GB，超过就开始换页或者直接 Metal OOM。并发 32、上下文 2048 时，KV 只要 4.3 GB（16 层注意力，每 token 64 KB），GDN 状态 4.8 GB（每请求 151 MB，不随长度变），加权重约 25 GB。

**格式。** nvfp4 是 e2m1 权重加 e4m3 的组内 scale，16 个一组，和 NVIDIA 的 NVFP4 是同一种数值格式。MLX 0.32 原生支持它，`mx.quantized_matmul(mode="nvfp4")` 直接可用，mlx-community 有转好的 checkpoint。同一份权重在 5090 和 Mac 上都能跑，这是选它而不是 affine 4bit 的原因。

**算力账。** 27B 每个 token 54 GFLOP。M4 Max 实测 bf16 矩阵乘 10 TFLOPS，内存带宽 460 GB/s。decode 每出一个 token 要把 16 GB 权重读一遍，35 ms，所以单流上限 28 tok/s；prefill 1024 个 token 要 55 TFLOP，5 秒。这两个数定了后面所有优化的边界：decode 卡带宽，prefill 卡算力。

**为什么不是 torch。** 我先用 torch MPS 试了：同一个 0.8B bf16 模型单流 110 tok/s，MLX 220；torch 的拷贝带宽只有 94 GB/s；而且 Mac 上没有 nvfp4 的 torch kernel，27B 根本跑不起来。vLLM 在 Mac 上没有正式版本。引擎用 MLX 写、调度器是纯 Python，后者可以原样搬到 CUDA 版。

## 先跑基线，再动手

动手之前先把现有工具在这台机器上跑一遍，否则不知道自己在追什么，也不知道机器本身有没有问题。Mac 上没有 vLLM，最接近的三个参照物：

| 工具 | 是什么 | 在这台机器上的数字（27B nvfp4） |
| --- | --- | --- |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple 官方库，server 带 continuous batching | 单流 24 到 29 tok/s；并发 8 decode 48 tok/s、TTFT 32 s；并发 32 进程 38 GB，Metal OOM |
| [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | DFlash2 投机解码的 MLX 实现 | 单流 64 tok/s（块 5），2.2 倍 |
| [oMLX](https://github.com/jundot/omlx) | 社区的“Mac 版 vLLM”，paged KV + MTP + DFlash | 别人在 M4 Max 128 GB 上：53 tok/s 文本、72 代码（[Weschera](https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac)） |

别人在 48 GB 上的结果和我的对得上：[oMLX 社区跑分](https://omlx.ai/benchmarks/performance/p3kbtfg6)里同款 M4 Max 48 GB 跑 6bit 是 20 tok/s，按权重大小换算和 4bit 的 25 一致；[SiliconBench](https://arxiv.org/abs/2609.19169)（2026 年 9 月，Penn State 和 EXO）测了九个 Mac 引擎，27B 只测到并发 4，而且是 64 GB 机器，mlx_lm 在它们那里也是 Metal OOM。没有人发过 27B 在 48 GB 上并发 8 以上的数字，这就是空白。

跑基线时踩的四个坑，每一个都会让数字差一倍以上：

1. **桌面 app 吃 swap。** 第一次跑并发 32 整机只有 20 tok/s，比单流还慢。查下来是浏览器、Slack、Notion 这些把 swap 吃到 12 GB，GPU 的数据被钉在内存里，其他东西全在换页。跑分前退出所有 app，脚本里每 20 秒记一次 swap 和进程占用。
2. **MLX 的 allocator 缓存。** 它把释放的 buffer 留着复用，每步的 K/V 拷贝尺寸各不相同，几分钟就攥出 20 GB。`mx.set_cache_limit(2 GB)` 之后峰值从 31 GB 降到 24 GB。dflash-mlx 默认 4 GB 也是为这个。
3. **mlx-lm server 的 prompt cache。** 它默认把跑完的请求的 KV 留在 LRU 里复用，`--prompt-cache-bytes 1` 关掉才公平。
4. **MacBook 持续负载会降频。** 连续 250 步 decode，前 20 步 195 ms，之后稳定在 250 ms。所有吞吐数字要看持续值，不要看前几步。

## 手搓路线图：六步，每步一个判据

每一步只加一件事，而且有一个能自动判对错的标准，这是整个过程里最重要的一条。第一步故意不带 cache，每生成一个 token 就把整句重算一遍，慢但能把“模型算对了”和“cache 算对了”分开验证。

| 步 | 加什么 | 判据 | 结果 |
| --- | --- | --- | --- |
| 1 | 模型前向 + 权重加载，整句算 | fp32 对 HF transformers fp32，48 个 greedy token 逐个相同 | logits 最大差 0.0001；27B 对 mlx-lm 24/24 |
| 2 | KV cache、GDN 状态、分块 GDN | prefill 后逐 token decode 的 logits == 整句一次前向 | prefill 103 → 248 tok/s |
| 3 | 调度器（纯 Python） | CPU 单测，200 个随机请求全部正确结束 | 固定槽位、不需要抢占 |
| 4 | 批量 decode、采样、引擎 | 三个请求合批 == 各自单跑（批量不变性） | decode 一步 260 → 186 ms |
| 5 | async 引擎、HTTP/SSE server | 自己写的 client 复刻 vllm bench serve | 并发 32 跑得起来 |
| 6 | DFlash2 投机解码 | 投机的 greedy == 不投机的 greedy，128/128 | 单流 2.2 到 2.5 倍 |

关于判据的两个经验。一是**只有 fp32 对 fp32 才是正确性证据**：bf16 跨框架累加顺序不同，top-2 打平的位置会翻，我的实现和 mlx-lm 都在第 12 个 token 分叉、之后又合上，追这个是浪费时间。二是 **kernel 测试要用真实数据的统计形状**，随机正态是最友好的输入，后面那个显式求逆的 bug 就是随机数据全过、真实数据全炸。

还有一个没料到的坑：mlx-community 转 checkpoint 时把零中心 RMSNorm 的 +1 烧进了权重（存的是 1 + w），而 HF 原版存 w。症状是 bf16 对 HF 完美、nvfp4 全盘错，逐层对比才抓到。

## Mac 和 CUDA 不一样的三件事

vLLM 的设计假设了 CUDA 的三个性质：显存可以原地改、算力相对带宽很富余、kernel 发射很便宜。在 MLX 上这三条都不成立，照抄 vLLM 会慢得莫名其妙。

**MLX 的数组不可变，散写是整块拷贝。** 实测：往一个 128 MB 的 K/V buffer 里写 8 行要 0.24 ms，恰好是把整块拷一遍的时间；按槽 gather 32 行 0.74 ms，也是拷贝。vLLM 的“大池子 + 按槽散写”在 CUDA 上理所当然，在 MLX 上每步要把 4.8 GB 的 GDN 状态池拷一遍，10 ms。所以我的 decode batch 是紧凑的批量数组：K/V 每行按绝对位置写一列（每步一次拷贝，不可避免），GDN 状态每步整体替换（递推的输出就是新状态，零拷贝），请求进出 batch 时才重排一次。调度器里的“槽”只是并发上限，不是一块预分配的内存。

**算力带宽比低一个量级，很多“免费”的事在这里要付费。** H200 上一步验证 8 个 token 和 1 个差不多，M4 Max 上实测：2 个免费，4 个 +11%，5 个 +32%，8 个翻倍。所以投机解码的块长是 5 不是 8，这和 oMLX 的默认值一致。同样的原因，并发 32 时 MLP 已经在 10 TFLOPS 的算力上限上，投机解码反而变慢，引擎只在单请求时开它。还有一个怪点：MLX 的 nvfp4 矩阵乘在 M=16 和 M=32 一样贵，并发 16 比并发 8 还慢，选并发数时要跳过 9 到 16。

**kernel 发射不便宜，逐 token 的 Python 循环要合成矩阵乘。** GDN 的递推逐 token 写最直白，但 prefill 1024 个 token 就是 1024 × 48 层的小算子，27B 只有 103 tok/s。按 64 个一块把块内的 delta rule 合成几次矩阵乘（FLA 的算法，HF 有纯 torch 参考实现），再把块内那个 63 步的三角求解写成一个 Metal kernel，就到了 248。安全的做法是先用递推版对齐 HF，再用它当分块版的参考答案。

## 两个 Metal kernel

两个 kernel 加起来不到 100 行 Metal，都是用 `mx.fast.metal_kernel` 写的，不用碰 Xcode。做法都一样：先用纯 MLX 写一个慢但对的版本当参考答案，再写 kernel，测试里两者必须一致。

**第一个：GDN decode 一步的状态更新。** 并发 32 时 decode 一步 260 ms，其中 GDN 状态更新 90 ms，比 MLP 的 108 少不了多少。纯 MLX 版要把 100 MB 的状态读三遍写一遍，只有 214 GB/s。kernel 里一个 threadgroup 管一个（序列, 头）的 128×128 状态，线程 t 管第 t 列：第一遍算 kv = Σ k·S，第二遍 S' = e·S + k·δ 顺手把 o += q·S' 也算了。相邻线程读写相邻地址，访存合并；q/k 的归一化和 exp 也折进去，每层省十来个小算子。结果：每层 1.86 → 0.79 ms，decode 一步 260 → 186 ms。小 batch 时线程数不够（并发 1 只有 48 个头），把 dk 拆成 4 段用 threadgroup 内存归约。

**第二个：分块 GDN 里的三角求解，先讲它的 bug。** 分块算法每 64 个 token 要解一次 (I + A) u = βv，A 是严格下三角。我第一版用幂级数显式求逆：(I+A)^{-1} = (I−A)(I+A²)(I+A⁴)…，A⁶⁴ = 0，6 次矩阵乘精确。随机输入下误差 1e-7，测试全过。真实数据上 logits 误差从第 16 个位置开始滚雪球，到第 59 个位置差 10。原因是真实序列里相邻 token 的键高度相关、β 接近 1，逆矩阵的元素按 2^(i−j) 增长再相消，fp32 扁不住。解本身是有界的，所以改成前向代入直接解、不形成逆矩阵，误差回到 1e-4。教训变成了一个测试：用相关的键和 β = 0.98 跑 kernel。

前向代入在 Python 里是 63 步小算子，每层约 190 次发射，所以写成第二个 kernel：一个 threadgroup 解一个（头, 块），256 个线程各管一列，按行顺序算、行之间 barrier。分块 kernel 从 14.4 ms 到 4.8 ms，27B prefill 从 208 到 248 tok/s，超过了 mlx-lm 的 affine 4bit。

## DFlash2 投机解码：单流翻倍，结果不变

单流 decode 一步 40 ms 里 35 ms 在读权重，算力闲着；一次验证 4 个 token 几乎不加钱。投机解码就是让一个小模型猜接下来几个 token，大模型一次前向验证，猜对几个白赚几个，从第一个猜错的位置截断。greedy 下结果和不投机逐 token 完全相同，我用 128 个 token 的数学和代码 prompt 验过，128/128。

**Drafter 是什么。** [DFlash2](https://inco.ai/blog/dflash2/) 的草稿模型是 5 层 Qwen3 式 transformer，1.9B 参数，加载时量化成 4bit 只有 1.2 GB，接受率和 bf16 一样。它的“记忆”是目标模型第 5、19、33、47、61 层的输出拼起来（每个 token 25600 维），所以目标模型 prefill 和 decode 时都要顺手把这五层的输出留下来。一轮：块 = \[锚点, mask × 4\] 过 5 层，块内非因果（block diffusion），每层夹一对两点的动态卷积；最后几个位置过目标模型的 lm_head，每个位置取 top-16，一个选路器按位置顺序用 codebook 挑一条连贯的路。

**验证和回滚。** 验证就是把块走一遍 prefill 路径，现成的 kernel 都能用。麻烦的是回滚：注意力 cache 把 offset 退回去就行，多写的几列下次覆盖；GDN 的递推状态退不回去，所以验证前快照、验证时把每个 token 的递推输入记在“磁带”上，接受 n 个之后从快照重放 n 步，1 ms。这个回滚在 0.8B fp32 上有精确测试：验证 8 个、只留 m 个、继续 decode，等于直接按这 m 个 decode。

**数字（27B，decode-only）。** 块 5：数学推理 56.5 tok/s（2.23 倍，每轮接受 3.3/4），写代码 62.3（2.46 倍），短中文问答 26.6（1.05 倍，每轮只接受 1.25）。每轮成本：draft 14 ms、verify 约 55 ms、回滚 1 ms。块 8 的 verify 要 95 ms，在 Mac 上不划算。接受率低的 prompt 用自适应策略缩到块 3，短中文这一条变成 29.3（1.16 倍）。dflash-mlx 在同一个数学 prompt 上是 64.2，差的 12% 在 verify 那步的小 M 量化矩阵乘，它有专门的 kernel。

**两个整合 bug。** 第一版引擎里接受率只有 1.7/7、6 tok/s：投机路径改了调度器的输出对象，调度器收不到 token，每轮都在同一个锚点上重来。能很快定位是因为单独测 drafter 的脚本给出 4.6/7，说明问题在整合不在模型。第二个：自适应块长用“平均接受数 + 1”，块缩到 3 之后最多接受 2 个，平均数再也涨不回去，正反馈锁死；要用逐位置接受率，它不随块长变。

## Pareto 曲线：吞吐 vs 每用户速度

一个数字说不清推理性能，因为吞吐和每个用户的速度是对立的。[InferenceX](https://inferencex.semianalysis.com/glossary/pareto-frontier) 的画法是扫一遍并发数，每个并发数是一个点，连起来就是一台机器能提供的所有取舍。

![Pareto 曲线：整机吞吐 vs 每用户速度](pareto.html)

（可交互版本：`docs/pareto.html`，用 `uv run bench/pareto.py` 重画。）

怎么读：一个人用，开 DFlash2，56 到 62 tok/s；2 到 4 个人同时用，合批几乎免费，每人仍有 19 到 24 tok/s；8 个人以上每人跌到 9 tok/s 以下，只适合批处理。并发 1 到 16 两条线重合，说明矩阵乘和注意力 kernel 都是 MLX 自己的，引擎层没有额外开销；24 和 32 时 mlx-lm 的静态 batch 快 20%，差在哪里还没查。但它的 server 在并发 8 只有 48 tok/s（空心点）、并发 32 直接 OOM，静态 batch 的数字它自己的 server 拿不到。DFlash2 的三个实心菱形旁边是 dflash-mlx 在同样 prompt 上的空心菱形：数学 54.6 对我们的 56.5，代码 56.1 对 62.3；短中文它 40.9 对我们 26.6，但它默认开 thinking，答案长得多，这一条不能直接比。

## 还没做好的

四件事没做到，按重要性排：

1. **verify 的小 M 矩阵乘。** 8 个 token 的前向要 95 ms，是 1 个 token 的 2.4 倍，理论上应该接近 1 倍。MLX 的 nvfp4 矩阵乘在 M 为 2 到 16 时效率很差，dflash-mlx 为此写了专门的 verify kernel。做好这个，投机解码能从 2.2 倍到 3 倍，并发 8 的 decode 也能快 30%。
2. **并发 8 的端到端比 mlx-lm 慢 10%。** 25 对 28 tok/s。分块 prefill 和 decode 交错让 TTFT 低 25%，但正在 decode 的请求每遇到一个 512 的块就等 2 秒。这是 TTFT 和 TPOT 之间的取舍，vLLM 也一样；块长该做成参数按场景选。
3. **server 层比裸引擎慢 13%。** 同样 32 条请求引擎 289 s、经 HTTP 326 s。怀疑是 asyncio 里逐 token 写 SSE 的 drain 和引擎循环争事件循环，没查。
4. **prefill 卡在 10 TFLOPS。** 1024 个 token 4.1 秒，全是矩阵乘，已经是这块 GPU 的算力上限。并发 32 时 TTFT 90 秒就是 32 条 prompt 排队算的结果，除非换机器，没有办法。

完整的 100 条 × 1024 进 1024 出的协议还没跑过，在 mlx-lm 上要两小时，在这个引擎上估计 25 分钟，留着夜里跑。还有一条是画 Pareto 曲线时才发现的：并发 24 到 32 的批量 decode 比 mlx-lm 的静态 batch 慢 20%（16 以内打平），差在哪里没查。

## 复现清单

一台 Apple Silicon，48 GB 以上，macOS 15 以上，装好 uv。全部命令在仓库根目录跑。

```bash
uv sync                                   # 引擎 + 开发依赖 → .venv，锁在 uv.lock
uv run hf download Qwen/Qwen3.5-0.8B       # 开发模型，1.6 GB
uv run hf download mlx-community/Qwen3.8-27B-nvfp4   # 目标模型，16 GB
uv run hf download z-lab/Qwen3.8-27B-DFlash2         # 投机解码的草稿模型，3.8 GB
uv run tests/make_refs.py                  # 用 HF transformers 生成参考答案（一次）
uv run pytest -q                           # 22 个测试，约 10 秒
```

起 server 和跑分：

```bash
M=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/*)
D=$(ls -d ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*)
uv run python -m minivllm.server --model $M --max-seqs 32 --max-len 4096 --port 8000          # OpenAI 兼容
uv run bench/bench_serve.py --base-url http://localhost:8000 --model $M --num-prompts 100 \
    --input-len 1024 --output-len 1024 --max-concurrency 32 --eos-mode vllm                  # 复刻 vllm bench serve
uv run bench/throughput.py $M --batch 1 8 32                                                 # 不经 server 的吞吐
uv run python -u bench/spec_check.py $M $D 128                                               # 投机解码：速度 + 无损验证
```

跑分前退出其他 app，`bench/run_ours.sh` 会每 20 秒记一次 swap 和进程占用。基线环境（mlx-lm、dflash-mlx）单独一个 `bench/setup_baseline.sh`，和引擎的 venv 分开。
