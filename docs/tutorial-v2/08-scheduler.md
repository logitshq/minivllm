# 08：从一个生成循环，变成多个请求共用的引擎

前面的模型已经能接续生成。这一章处理新的问题：A 正在生成，B 又进来，应该怎样分配计算并保存各自的状态？

| 之前 | 本章之后 |
|---|---|
| 一个请求独占生成循环 | 引擎管理 waiting、running、finished 请求 |
| 每次只算一个新 token | 多个请求的新 token 合成一次 decode batch |
| 一次处理完整 prompt | 长 prompt 分块，先给正在生成的请求安排 decode |

代码版本：`tutorial-08`；上一个版本是 `tutorial-07-v2`。新增 [scheduler.py](../../tutorial_code/miniengine/engine/scheduler.py)、[model_runner.py](../../tutorial_code/miniengine/engine/model_runner.py)、[llm_engine.py](../../tutorial_code/miniengine/engine/llm_engine.py)，并扩展 [cache.py](../../tutorial_code/miniengine/cache.py)。参考原 `minivllm/engine/`，教学包运行时不导入完成版。

## 1. 分清模型、调度器和 runner

```mermaid
flowchart LR
    P[输入 token IDs] --> E[LLMEngine]
    E --> S[Scheduler: 决定本步算哪些请求]
    S --> R[ModelRunner: 组织输入与 cache]
    R --> M[我们自己的 MLX 模型]
    M --> A[Sampler: logits → token IDs]
    A --> U[更新请求长度 / 完成状态]
    U --> E
```

Scheduler 只操作 Python 请求与计数，不认识矩阵。Runner 把这些决定变成 tensor 和 cache。模型执行的仍是第五章那套数学，只新增“每行对应一个请求”的路径。

这个分工让我们能不加载模型就测试调度逻辑，也能不做 HTTP serving 就运行完整引擎。

## 2. 一个小例子：每步最多算 6 个 token

假设本步 A、B 已经进入 decode；C 的 prompt 还剩 10 个 token。

```text
预算 6
  A：decode 1                    剩余 5
  B：decode 1                    剩余 4
  C：prefill 4                   剩余 0
下一步：再先安排 A/B，余量继续处理 C
```

预算是**模型本步输入的 token 数**，不是输出的字数。C 的 prefill 块没到 prompt 末尾，就不会产生第一个输出 token。

当 C 的 prompt 最后一块算完，其末尾 logits 就能预测第一个 token。不要再额外跑一次空 decode 才开始采样。

decode 优先可以避免长 prompt 在每步预算上占满，但一个 prefill 块仍要花真实时间。本实现不能据此保证全局 TTFT 或每个请求的严格延迟上限。

## 3. 从原来循环到调度代码

第七章的 baseline 是我们自己的单请求循环：

```python
for _ in range(max_new_tokens):
    logits = model(current, ctx, last_only=True)
    token = argmax(logits[-1])
    current = [token]
```

原 `minivllm` 的 scheduler 思路被整理成教学代码，以下是省略请求队列细节的主线：

```python
budget = self.max_batched_tokens
for req in self.running:
    if req.prompt_done:
        out.decode.append(req)                  # ← 关键① 每个活跃 decode 先占 1
        budget -= 1
for req in self.running:
    if not req.prompt_done and budget > 0:
        n = min(req.remaining_prompt_tokens, budget)
        out.prefill.append((req, n))             # ← 关键② 剩余预算处理 prompt
        budget -= n
```

`remaining_prompt_tokens` 是这里便于阅读的简写；源码用 `num_prompt_tokens - num_computed_tokens`。最后才从 waiting 队列按先到先服务接入新请求，还需要空闲槽位。

## 4. 真正的 batch：每行一个请求

假设 A 已算 20 个 token，B 已算 7 个 token：

```text
ids = [A 的最新 token, B 的最新 token]          [B=2]
positions = [20, 7]                             [B]
hidden = Embedding(ids)                         [B,hidden]
logits                                         [B,vocab]
```

数组里的两个位置属于**两条不同序列**，不能当成同一条序列的两个时间位置。Attention 的 RoPE 为每行使用自己的 position；attention mask 为每行使用自己的有效历史长度。

```python
mask = mx.arange(L)[None, :] < ctx.lens[:, None] # ← 关键③ 防止读到别人的 padding
```

GDN 状态增加 batch 维：`S=[B,Hv,Dk,Dv]`、`conv=[B,K−1,C]`。本章用普通 MLX 向量运算并行更新各行，数学仍是第四章的衰减后预测与 delta 更新，没有提前调用自定义 Metal。

## 5. 状态怎样随请求进出

prefill 时，每个请求先使用第六章的单序列 cache。prompt 算完之后，runner 把它移入 compact decode batch。

```text
prefill 暂存： A 的独立 cache；B 的独立 cache
             ↓ prompt 完成后加入
decode batch： row 0 = A；row 1 = B
             ↓ A 完成，C 加入
decode batch： row 0 = B；row 1 = C
```

请求到 batch 行的映射也必须更新，否则输出和状态会交叉串线。不同长度 K/V 用 padding 放到共同长度，靠 mask 读取各自的有效部分。

这是 **compact batch cache**，没有分页 block table、抢占或跨请求 prefix cache。slot 用于限制并发请求数；实际 K/V tensor 随批内容量需要增长，不能把它描述成所有槽位的完整 `max_len` 状态始终预分配。

结束请求当步释放 runner 的状态；最后一个请求结束时也清理，不依赖“再调用下一次 step”触发清理。

## 6. 引擎入口与核心 diff

```python
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.sampling_params import SamplingParams

engine = LLMEngine(model_dir, max_seqs=3, max_len=256, max_batched_tokens=32)
outputs = engine.generate([prompt_A_ids, prompt_B_ids], SamplingParams(max_tokens=12))
```

也可以 `add_request` 后反复 `step`，并在运行中添加新请求。`step` 返回 `(request_id, token_id, finished)`，调用者自行决定怎样展示。

```diff
- 每个 prompt 单独循环调用 model
+ out = scheduler.schedule()
+ sampled = runner.run(out)             # 多请求 decode 一起 forward
+ scheduler.update(out, sampled)
+ 释放本步完成请求的状态
```

采样参数默认 greedy；普通路径还可选择随机采样。本章正确性对照使用 greedy，避免随机数把 batch 与串行的差异混在一起。空 prompt、非正 `max_tokens`、超出总长度和重复活跃 ID 会在边界报错。

## 7. 验证调度与 batch

```bash
python -m tutorial_code.check_chapter08 --scheduler-only \
  --out tutorial_code/examples/chapter08-scheduler.json
python -m tutorial_code.check_chapter08 \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 12 --out tutorial_code/examples/chapter08-checks.json
git diff tutorial-07 tutorial-08 -- tutorial_code/miniengine
```

纯调度检查预算、生命周期与边界。GPU 检查用长度不同的三条 prompt，比较各自串行运行与合批运行的 greedy 输出；再测试运行中加入请求、结束后清理 cache。结果分别保存在 [调度报告](../../tutorial_code/examples/chapter08-scheduler.json) 与 [完整引擎报告](../../tutorial_code/examples/chapter08-checks.json)。

现在我们有多请求核心引擎。下一章保持这些行为，优化 GDN 的 GPU 执行。

[下一章：09 Metal](09-metal.md)

## 实测发现：先核对 dtype，再解释 greedy 差异

首次测试指定 float32，但 NVFP4 Embedding 实际返回 bfloat16，导致短 prompt 的批量和单请求在第七个输出出现不同 ID。我们保留了 [修复前诊断](../../tutorial_code/examples/chapter08-bf16-diagnostic.json)，没有把这次失败记成通过。不同计算布局在低精度下产生数值差，teacher-forced 诊断中存在 top-two 分数完全相同的位置；greedy 在这种边界尤其敏感。

第七章 `tutorial-07-v2` 修复了量化 Embedding 的输出 dtype。修复后检查脚本直接断言真实激活与 logits 为 float32；同样的三个长度 19/7/12 的 prompt、每条 12 个输出、运行中加入请求均与独立单请求一致。raw HF 权重的 float32 路径也通过同样检查。这是这些测试条件的结果，不承诺任意低精度布局都逐 token 完全相同。

## 本机验证记录

完整结果见 [chapter08-checks.json](../../tutorial_code/examples/chapter08-checks.json)。这是本章代码实际运行的结果：

```json
{
  "scheduler_budget_and_admission": true,
  "eos_stop": true,
  "max_tokens_zero_rejected": true,
  "events": [
    {
      "decode": [],
      "prefill": [
        [
          "a",
          3
        ]
      ]
    },
    {
      "decode": [],
      "prefill": [
        [
          "a",
          2
        ],
        [
          "b",
          1
        ]
      ]
    },
    {
      "decode": [
        "a"
      ],
      "prefill": [
        [
          "b",
          1
        ]
      ]
    },
    {
      "decode": [
        "a",
        "b"
      ],
      "prefill": []
    },
    {
      "decode": [
        "b"
      ],
      "prefill": [
        [
          "c",
          1
        ]
      ]
    },
    {
      "decode": [
        "c"
      ],
      "prefill": []
    },
    {
      "decode": [
        "c"
      ],
      "prefill": []
    }
  ],
  "same_format_batch_greedy_equal": true,
  "late_arrival_equal": true,
  "finished_cache_released": true,
  "activation_dtype": "mlx.core.float32",
  "logits_dtype": "mlx.core.float32",
  "max_new_tokens": 12,
  "prompt_lengths": [
    19,
    7,
    12
  ],
  "single_outputs": [
    [
      79852,
      18887,
      220,
      95761,
      21966,
      220,
      95789,
      98618,
      99580,
      97792,
      99125,
      103935
    ],
    [
      2005,
      95883,
      828,
      10992,
      198,
      95826,
      110827,
      98682,
      108829,
      2005,
      95883,
      828
    ],
    [
      248045,
      198,
      248068,
      198,
      90700,
      8340,
      25,
      271,
      16,
      13,
      220,
      2972
    ]
  ],
  "batch_outputs": [
    [
      79852,
      18887,
      220,
      95761,
      21966,
      220,
      95789,
      98618,
      99580,
      97792,
      99125,
      103935
    ],
    [
      2005,
      95883,
      828,
      10992,
      198,
      95826,
      110827,
      98682,
      108829,
      2005,
      95883,
      828
    ],
    [
      248045,
      198,
      248068,
      198,
      90700,
      8340,
      25,
      271,
      16,
      13,
      220,
      2972
    ]
  ]
}
```
