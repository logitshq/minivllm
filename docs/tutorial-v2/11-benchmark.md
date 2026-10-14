# 11：怎样证明实现正确，又怎样比较速度

我们已经有自己的模型、cache、调度、Metal kernel 与投机路径。最后一章把验证和计时整理成可复现实验，回答两个不同的问题：算得对吗，以及在指定工作负载上花了多少时间？

| 之前 | 本章之后 |
|---|---|
| 各章组件与生成检查 | 固定模型、输入与条件的实验记录 |
| 8-token oMLX 功能冒烟 | 全长度预热、重复测量、保存每轮实际输出数 |
| 只看一个 tok/s | 分清 TTFT、后续生成、引擎和 HTTP 边界 |

代码版本：`tutorial-11`；上一个版本是 `tutorial-10`。入口 [benchmark_core.py](../../tutorial_code/benchmark_core.py)、[run_omlx.py](../../tutorial_code/baselines/run_omlx.py) 和 [probe_baselines.py](../../tutorial_code/baselines/probe_baselines.py)。现有 [baseline 说明](../../tutorial_code/baselines/README.md) 保留安装、版本固定和短请求记录。

## 1. 几个 baseline 各负责什么

| 对象 | 要回答的问题 | 这里的测量边界 |
|---|---|---|
| HF Torch/MPS | 相同原始权重与输入下，模型数学是否复写正确？ | 正确性参照；第一章无 cache 不是性能目标 |
| 我们的 miniengine | 手写引擎在本机怎样运行？ | 输入 token IDs 起，到交付 token IDs；含调度与实际 GPU 执行 |
| oMLX | 成熟 Apple Silicon 引擎的表现 | 当前 runner 测 HTTP/SSE 客户端，含服务与传输开销 |
| Lithos Metal | 另一种 Metal 引擎结构能否成为同机参照？ | 候选 baseline；先确认模型、硬件路径与本机运行资格 |

同一量化 checkpoint 可以减少一项重要差异，但仍不意味着不同测量边界的数字可以直接相除。

Lithos 值得参考其 layer-wise mixer megakernel 与 GPU 生成控制设计。官方要求 macOS 26+，但 M4 Max 32/40-core 当前列为未测量 native fallback，未测 backend 不自动启用 megakernel fusion/tensor accelerator；它使用的 DSpark 草稿也与我们的 DFlash2 不同。见 [官方硬件与模型说明](https://github.com/lithos-ai/lithos-metal#models-and-hardware)。

官方报告里的 M5 Max 168 tok/s 是假设接受 6 个草稿、验证 8 行的目标计算投影，排除了 draft、accept 与 rewind，不是本机端到端结果。见 [固定版本的验证报告](https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-27b-backend-verification.md)。本章将其登记为候选，不填未经本机测量的成绩。

## 2. 先统一哪些条件

一个可解释的实验至少记录：

- checkpoint 的实际目录、配置与量化格式，激活 dtype。
- 原始 prompt IDs、输入长度、输出上限与实际输出数。
- greedy/随机采样、EOS 与 stop 规则、并发、是否投机及匹配 drafter。
- 预热次数、重复次数、版本与机器，内存测量的含义。
- 计时边界、cache 命中与 native kernel 的实际执行证据。

固定 `max_new_tokens=128` 只是上限。模型遇到 EOS 可以提前停，分母必须使用实际生成数。为了固定步数而忽略 EOS，是另一种实验，应单独标注，不与正常停止的 API 结果混用。

终端 EOS 的计数也要说明。我们的 `output_tokens` 包含交付的 EOS ID，`logical_completion_tokens` 在启用 EOS 停止时扣掉它；核心速度字段按交付的全部 IDs 计算。oMLX 的 `completion_tokens` 排除终端 EOS，并记录 `finish_reason`。即使两边相同设置下都提前结束，两个原始 count 字段也未必相等。

```text
加载模型 → 完整预热请求 → 请求1 → 请求2 → 请求3
                     ↑ 从这里开始保存正式测量
每个正式请求：等待此前 GPU 工作完成 → 开始时钟 → step → 完成同步
```

MLX 有 lazy evaluation。只测 Python 创建计算图的时间，会把尚未执行的 GPU 工作遗漏。runner 读取输出前执行 `mx.eval`，benchmark 还在请求边界同步。

## 3. 读懂三个不同的时间

假设一个正常单 token decode 请求：首个 token 在 0.2 秒交付，第 5 个 token 在 0.6 秒交付。

```text
0s          0.2s          0.6s
开始 --------首 token------最后 token
   TTFT=0.2s   后续时间=0.4s
```

总输出吞吐 `5/0.6≈8.33 tok/s`；首 token 之后的吞吐 `(5−1)/(0.6−0.2)=10 tok/s`。它们回答不同的问题。

引擎 TTFT 是第一个输出 ID 交付的时间；oMLX 客户端 TTFT 是首个非空文本片段到达的时间。token 可能暂时解码不出文本，SSE 也可能合并片段，因此二者不是相同事件。

投机一次可能交付多个 token。首个交付块不只一个 token 时，计算“首个交付之后吞吐”需要扣掉整个首块，不能机械用 N−1。报告应同时保存每步交付数，避免把一块 token 假装成逐 token 到达。

## 4. 我们引擎的重复测量

```bash
python -m tutorial_code.benchmark_core \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 128 --repeat 3 --warmup-runs 1 \
  --out tutorial_code/examples/chapter11-core.json
```

脚本先加载模型，再用完整测试长度预热，计时里不包括加载、tokenizer 或 HTTP。每次都是新请求与自己的状态，不把上一轮 prompt 的结果当成跨请求 cache 命中。

重点读 [chapter11-core.json](../../tutorial_code/examples/chapter11-core.json) 的这些字段：

| 字段 | 含义 |
|---|---|
| `measurement_scope` | 计时包含与排除的边界 |
| `request.prompt_ids` | 真正输入的 IDs，不只一句文字 |
| `runs[].output_tokens` | 本轮实际输出数 |
| `runs[].logical_completion_tokens` | 排除终端 EOS 的完成 token 数；原始 IDs 与速度字段仍保留原计数 |
| `runs[].engine_elapsed_s` | 从请求进入到完成的时间 |
| `runs[].engine_ttft_s` | 首次交付输出 ID 的时间 |
| `runs[].step_delivery_times` | 每次交付发生在什么时候、交付几个 token |
| `runs[].first_delivery_tokens` | 首次交付的块中有几个 token |
| `runs[].post_first_delivery_s` | 从首块交付到结束的时间 |
| `runs[].mlx_peak_memory_bytes` | 本轮 MLX allocator 的峰值内存 |
| `median_*` | 重复请求的中位数，不能代替原始各轮记录 |

`mlx_active_memory_after_finish_bytes` 用于观察请求清理后仍保留的 MLX 活跃内存，其中包括模型权重。这些字段不是进程 RSS，也不是整个系统统一内存占用；不同工具的内存数字要先对齐含义再比较。

投机测量使用同一入口的 `--drafter-dir` 与 `--spec-block`，保存接受数与验证轮数。必须和同一个目标 checkpoint 的普通模式相比；第十章先核对输出，第十一章才解释速度。

## 5. oMLX 的重复测量

```bash
.artifacts/omlx-baseline/.venv/bin/python -m tutorial_code.baselines.run_omlx \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 128 --repeat 3 \
  --out tutorial_code/examples/chapter11-omlx.json
```

runner 在自己拥有的同一个临时服务器中预热、重复请求，结束后关闭该进程。它将第一章的 ChatML IDs 还原为保留特殊标记的文本，确认 roundtrip，再经 `/v1/completions` 请求，避免自动叠第二份 chat template。

对照 [chapter11-omlx.json](../../tutorial_code/examples/chapter11-omlx.json) 的每轮 usage、实际输出数、客户端首文本时间与总耗时。关闭跨请求 cache 与关闭模型内 decode 状态不是一回事：oMLX 仍使用自己的 KV/GDN 状态。

runner 对每个正式请求断言 `cached_tokens=0`，并保存 `stop` 或 `length` 的结束原因。预热使用相同输出上限，同时记录预热实际完成数；提前 EOS 不会被报告成生成满 128 个 token。

已有 [8-token smoke](../../tutorial_code/baselines/omlx-smoke.json) 只能证明安装与调用可用，不能作为持续吞吐或 27B 排名。当前 0.8B 的正式报告也不能代表 27B；换模型就重新记录相应实验。

## 6. 登记 Lithos 与实际报告资格

```bash
python -m tutorial_code.baselines.probe_baselines \
  --core-report tutorial_code/examples/chapter11-core.json \
  --omlx-report tutorial_code/examples/chapter11-omlx.json \
  --out tutorial_code/examples/chapter11-baselines.json
```

[baseline 清单](../../tutorial_code/examples/chapter11-baselines.json) 将可运行、系统要求满足和实际有测量报告分开记录。只有确实完成本机 Lithos 测量后，才传 `--lithos-report`；README 宣传值不是这份参数应读取的报告。

性能表先按边界分列：我们的核心引擎、oMLX HTTP 客户端、Lithos 未测候选。若要严谨比较 API 端到端速度，双方都用相同 API 驱动和请求条件；若要比较模型核心，就为双方都准备相同核心计时接口。

## 7. 回到每章究竟改了什么

```diff
- 一个短请求跑通，留下 elapsed
+ 全长度预热 + 重复请求 + 保存每轮原始结果
+ 同时记录输入IDs、checkpoint、dtype、实际输出数与计时边界
+ 正确性报告与性能报告分别解释
+ 候选baseline先核资格，再加入实测比较
```

```bash
git diff tutorial-10 tutorial-11 -- tutorial_code/benchmark_core.py tutorial_code/baselines
```

到这里，核心链路是 `IDs → 调度 → 自己的模型与 cache → 采样 → IDs`。tokenizer 留在调用边界；HTTP serving 可以以后另加一层，不是读懂或运行这套引擎的前提。

建议每次优化都沿用这里的顺序：先验证同输入计算和状态，再计时同工作负载，最后判断收益来自哪里。这样看到一个快的数字时，也能解释它是怎样得到的。

## 实测时发现的 EOS 元数据问题

第一次核心测量虽然设置 `ignore_eos=False`，仍然输出满 128 个 ID，并开始重复 ChatML 标记。问题是旧读取代码只看顶层 `config.json.eos_token_id`，而这个 checkpoint 把它放在 `text_config`；tokenizer 元数据还把 `<|im_end|>` 指定为结束标记。

本章增加 [eos.py](../../tutorial_code/miniengine/eos.py)，读取嵌套配置和 tokenizer JSON 中的特殊 token ID；这里不执行分词。最终 EOS 集合是 `{248044,248046}`，核心引擎与生成 CLI 共用这个规则。新增 [CPU 元数据检查](../../tutorial_code/examples/chapter11-eos-checks.json)，再重新完成预热与三次实际请求。[修复前记录](../../tutorial_code/examples/chapter11-core-before-eos-fix.json) 仅保留为诊断，不能当作正常停止基准。

```diff
- 只读 config 顶层的 eos_token_id
+ 读 generation/config 的 EOS IDs，包括 text_config
+ 从 tokenizer JSON 查询配置中的 EOS 特殊标记 ID
+ 报告实际 EOS IDs、结束与输出计数，然后重新测量
```

## 本机这次正式记录

同一 0.8B NVFP4 checkpoint、同一 19 个输入 IDs、输出上限 128、greedy、并发 1，预热后各测三次。

| 测量边界 | 实际输出 | 首次交付中位时间 | 本边界吞吐中位数 |
|---|---|---|---|
| 教学核心：IDs 输入到 IDs 交付 | 29 IDs，含 EOS；28 logical | 12.30 ms（首 ID） | 187.41 emitted IDs/s |
| oMLX：HTTP/SSE 客户端 | 28 completion tokens，stop | 16.14 ms（首非空文本） | 357.19 completion tokens/s |

两边的计时起止事件与输出计数不同，所以不把表中数字相除称为加速倍数。请求提前 EOS：这是一个短 prompt 的 0.8B 重复测量，没有生成满 128 个 token，不能代表持续吞吐、27B、批量或投机性能。完整每轮记录保留在各自 JSON。核心 MLX allocator 峰值约 511 MB；这是 MLX 管理内存，不能写成系统总内存或进程 RSS。

本机 [Lithos 资格记录](../../tutorial_code/examples/chapter11-lithos-qualification.json) 标出 M4 Max 40 核、48GB、macOS 27.0.1 和当时的安装/格式状态。此 tag 尚未包含 Lithos 实测，随后单独加入真实运行记录，保留本版本。

## 本机验证记录

完整结果见 [chapter11-checks.json](../../tutorial_code/examples/chapter11-checks.json)。这是本章代码实际运行的结果：

```json
{
  "model": "Qwen3.5-0.8B NVFP4 same local checkpoint",
  "prompt_tokens": 19,
  "output_cap": 128,
  "repeats": 3,
  "core": {
    "output_counts": [
      29,
      29,
      29
    ],
    "logical_counts": [
      28,
      28,
      28
    ],
    "median_ttft_id_s": 0.012300624992349185,
    "median_emitted_ids_per_engine_s": 187.41149492225355,
    "median_decode_ids_per_s": 196.57526157069236,
    "peak_memory_bytes": [
      511464387,
      510180251,
      511462723
    ],
    "scope": "token-ID synchronous engine; includes Python scheduling and GPU evaluation; excludes loading/tokenization/HTTP"
  },
  "omlx": {
    "completion_counts": [
      28,
      28,
      28
    ],
    "finish_reasons": [
      "stop",
      "stop",
      "stop"
    ],
    "median_ttft_text_s": 0.016139874991495162,
    "median_client_elapsed_s": 0.07838933299353812,
    "median_completion_tokens_per_client_s": 357.19145616799835,
    "scope": "HTTP SSE client timing includes server overhead; independent of pure engine timing"
  },
  "regressions": {
    "tutorial_final_raw_fp32_greedy": 48,
    "tutorial_final_cache": true,
    "original_pytest_passed": 22,
    "original_pytest_seconds": 9.38,
    "original_pytest_first_attempt": "18 passed, 1 failed, 3 setup errors: ignored local models folder missing; resolved by ignored symlink to existing model, then rerun all 22"
  },
  "lithos": "qualified but no run in tutorial-11; actual run preparation proceeds separately",
  "limitations": "one short 0.8B prompt, early EOS; no sustained 128-token, 27B throughput, concurrency or speculative performance claim; no cross-scope speedup ratio"
}
```
