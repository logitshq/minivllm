# 11：怎样证明实现正确，又怎样比较速度

我们已经有自己的模型、cache、调度、Metal kernel 与投机路径。本章把前十章的验证和计时整理成可复现实验，回答两个不同的问题：算得对吗，以及在指定工作负载上花了多少时间？第十二章起再逐步优化执行路径。

| 之前 | 本章之后 |
|---|---|
| 各章组件与生成检查 | 固定模型、输入与条件的实验记录 |
| 8-token oMLX 功能冒烟 | 全长度预热、重复测量、保存每轮实际输出数 |
| 只看一个 tok/s | 分清 TTFT、后续生成、引擎和 HTTP 边界 |

代码版本：`tutorial-11-v3`（保留原 `tutorial-11` / `tutorial-11-v2`）；上一个版本是 `tutorial-10`。入口 [benchmark_core.py](../../tutorial_code/benchmark_core.py)、[run_omlx.py](../../tutorial_code/baselines/run_omlx.py) 和 [probe_baselines.py](../../tutorial_code/baselines/probe_baselines.py)。现有 [baseline 说明](../../tutorial_code/baselines/README.md) 保留安装、版本固定和短请求记录。

以下是本章固定版本的历史记录，不是最新三家排名。后续同一 HTTP 边界的短负载比较见[第44章](../experiments/44-fresh-full-throughput-latency.md)和[第45章](../experiments/45-lithos-queued-concurrency.md)；它们不回填本章的核心引擎/API 表格。

## 1. 几个 baseline 各负责什么

| 对象 | 要回答的问题 | 这里的测量边界 |
|---|---|---|
| HF Torch/MPS | 相同原始权重与输入下，模型数学是否复写正确？ | 正确性参照；第一章无 cache 不是性能目标 |
| 我们的 miniengine | 手写引擎在本机怎样运行？ | 输入 token IDs 起，到交付 token IDs；含调度与实际 GPU 执行 |
| oMLX | 成熟 Apple Silicon 引擎的表现 | 当前 runner 测 HTTP/SSE 客户端，含服务与传输开销 |
| Lithos Metal | 另一种 Metal 引擎结构能否成为同机参照？ | 候选 baseline；先确认模型、硬件路径与本机运行资格 |

同一量化 checkpoint 可以减少一项重要差异，但仍不意味着不同测量边界的数字可以直接相除。

Lithos 值得参考其 layer-wise mixer megakernel 与 GPU 生成控制设计。官方要求 macOS 26+，但 M4 Max 32/40-core 当前列为未测量 native fallback，未测 backend 不自动启用 megakernel fusion/tensor accelerator；它使用的 DSpark 草稿也与我们的 DFlash2 不同。见 [官方硬件与模型说明](https://github.com/lithos-ai/lithos-metal#models-and-hardware)。

官方报告里的 M5 Max 168 tok/s 是假设接受 6 个草稿、验证 8 行的目标计算投影，排除了 draft、accept 与 rewind，不是本机端到端结果。见 [固定版本的验证报告](https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-27b-backend-verification.md)。原 `tutorial-11` 将其登记为候选；本章扩展版只填下方真实本机测量的成绩。

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

性能表按边界分列：我们的核心引擎、oMLX 与 Lithos 的 HTTP 客户端。若要严谨比较 API 端到端速度，双方都用相同 API 驱动和请求条件；若要比较模型核心，就为双方都准备相同核心计时接口。

## 7. 回到每章究竟改了什么

```diff
- 一个短请求跑通，留下 elapsed
+ 全长度预热 + 重复请求 + 保存每轮原始结果
+ 同时记录输入IDs、checkpoint、dtype、实际输出数与计时边界
+ 正确性报告与性能报告分别解释
+ 候选baseline先核资格，再加入实测比较
```

```bash
git diff tutorial-10 tutorial-11-v3 -- tutorial_code/benchmark_core.py tutorial_code/baselines
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

本机 [Lithos 资格记录](../../tutorial_code/examples/chapter11-lithos-qualification.json) 标出 M4 Max 40 核、48GB、macOS 27.0.1 和当时的安装/格式状态。原 `tutorial-11` 尚未包含 Lithos 实测；下文分别记录后来加入 `tutorial-11-v2` / `tutorial-11-v3` 的真实运行，旧 tag 保持原样。

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

## Lithos 安装与 CPU 准备

在保留 `tutorial-11` 之后，本机已安装第三条外部 baseline。官方 v0.1.2 wheel 缺少 M4 Max 登记与 MLX untied 输出头名称映射，因此使用完整官方源码 `5485ad6bac0a5649d66a347f3e01a655ea29d937` 构建 native runtime；没有本地模型或 kernel 补丁。归档 SHA 与实际构建文件 hash 见 [安装记录](../../tutorial_code/baselines/lithos-installation.json) 和 [构建 hash](../../tutorial_code/baselines/lithos-build-hashes.json)，重建命令见 [Lithos 说明](../../tutorial_code/baselines/lithos-README.md)。

CPU 准备已验证现有 [0.8B](../../tutorial_code/examples/chapter11-lithos-0.8b-cpu-prepared.json) 的 187 个和 [27B](../../tutorial_code/examples/chapter11-lithos-27b-cpu-prepared.json) 的 498 个 NVFP4 tensor binding；两者的聊天模板都逐个匹配第一章 19 个输入 IDs，EOS 集合均为 `{248044,248046}`。0.8B 缺少 generation_config，adapter 在隔离目录用符号链接构造只补 EOS 元数据的 checkpoint 视图，原 tensor/config/tokenizer 字节保持原样。

上面的 CPU 准备记录与随后的真实 GPU 测量分别保留。用户解锁后正常退出空闲应用，保留正在运行的远端 CUDA 工作与 Codex，然后串行完成 GPU 实测；原 qualification JSON 保留为安装前记录，不能当作当前性能成绩。运行时明确 `--no-draft`，DFlash2 不能代替 Lithos 的 DSpark。

## Lithos 实际运行：同一 0.8B checkpoint 的第三条记录

`tutorial-11-v2` 追加真实 Lithos GPU 记录，原章节 tag、安装前资格与 CPU 准备记录都保留。先通过 16-token 功能 smoke，再用输出上限 128 的同一工作负载预热、重复三次；全部使用 target-only `--no-draft`、greedy、正常 EOS、并发 1。报告对应实际 M4 Max 40 GPU 核、48GiB、macOS 27.0.1。

同一 0.8B NVFP4 checkpoint 与配置 SHA、同一 19 个输入 IDs，不代表不同实现逐 token 一致。本次 Lithos 的 BF16 normalization/projection fusion 默认 `commute_norm=True`，与其他路径的数值组织不同；实际输出长度也不同。

| 0.8B 测量边界 | 实际完成数 | 首次交付中位时间 | 本边界吞吐中位数 |
|---|---|---|---|
| 教学核心：IDs 输入到 IDs 交付 | 29 IDs 含 EOS；28 logical | 12.30 ms（首 ID） | 187.41 emitted IDs/s |
| oMLX：HTTP/SSE 客户端 | 28 completion，不含终端 EOS | 16.14 ms（首文本） | 357.19 completion tokens/s |
| Lithos：HTTP/SSE 客户端 | 34 committed IDs 含 EOS；33 logical | 1460.17 ms（首文本） | 20.51 committed IDs/s |

完整结果见 [Lithos 0.8B](../../tutorial_code/examples/chapter11-lithos-0.8b.json)、[0.8B smoke](../../tutorial_code/examples/chapter11-lithos-0.8b-smoke.json) 和已有核心/oMLX JSON。每次 Lithos 请求 `cached_prompt_tokens=0`、`verify_tokens=1`，确实走了目标模型；退出时仅终止自己的服务进程。

这些是各自边界下的真实数值，不能称为相同输出工作量的简单加速倍数。Lithos 的 `max_context=2048`，教学核心按 `prompt长度 + 输出上限 + 16` 设置 max_len，并使用随批容量增长的 cache；容量与分配策略也未统一。客户端 tok/s 包含 prefill 与服务，不是纯 decode tok/s；不同输出长度、EOS 计数与浮点路径都已保留。三轮提前 EOS，不能写成持续生成满 128 个 token。

Lithos 0.8B 日志中，三轮 prefill GPU command-buffer duration 中位数约 1449.79 ms，decode GPU duration 约 186.50 ms，测量轮 `setup_ms=0`。这些是 native GPUStart/End command-buffer 时间，不是 CPU 首次编译，也不是单个算子的纯数学耗时。记录的 `steps=[40, 40, 40]` 包含结束后的 no-op，不能当成实际输出 token 数去计算吞吐。

0.8B 本次服务进程到 API ready 为 1.443 s，显式完整预热请求为 2.940 s，均在正式重复请求之外。`--no-warmup` 把 GPU session 首次加载/编译留给显式预热；API ready 本身不证明模型已加载，pack 已存在时也不能把 ready 时间当成首次 pack 成本。

27B 本次服务进程到 API ready 为 1.445 s，显式完整预热请求为 49.859 s，均在正式重复请求之外。`--no-warmup` 把 GPU session 首次加载/编译留给显式预热；API ready 本身不证明模型已加载，pack 已存在时也不能把 ready 时间当成首次 pack 成本。

## 同一 27B checkpoint 的三条独立记录

下面记录固定缓存 checkpoint `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`。三条引擎复用这一相同 checkpoint、19 个输入 IDs、输出上限 128、greedy 与正常 EOS，预热后重复三次。核心和 API 仍分列；模型规模与上面的 0.8B 不同，不跨模型排速度。

| 27B 测量边界 | 实际完成数（三轮） | 首次交付中位时间 | 本边界吞吐中位数 |
|---|---|---|---|
| 教学核心：IDs 输入到 IDs 交付 | 37, 37, 37 emitted IDs；logical 36, 36, 36 | 189.57 ms（首 ID） | 18.11 emitted IDs/s |
| oMLX：HTTP/SSE 客户端 | 44, 44, 44 completion，不含终端 EOS | 243.56 ms（首文本） | 22.02 completion tokens/s |
| Lithos：HTTP/SSE 客户端，no-draft | 45, 45, 45 committed IDs；logical 44, 44, 44 | 42781.46 ms（首文本） | 0.97 committed IDs/s |

完整结果见 [教学核心 27B](../../tutorial_code/examples/chapter11-core-27b.json)、[oMLX 27B](../../tutorial_code/examples/chapter11-omlx-27b.json)、[Lithos 27B](../../tutorial_code/examples/chapter11-lithos-27b.json) 与 [27B smoke](../../tutorial_code/examples/chapter11-lithos-27b-smoke.json)。这是本机 registered M4 native fallback 的实际目标模型路径，上游 profile 仍标为 `unmeasured`，automatic mixer megakernel fusion 与 tensor accelerator 没有启用；本次测量不能被描述为整套 profile 校准，也没有测 DSpark 或 DFlash2 投机性能。

同一个测试 prompt 上，27B oMLX 的 44 个 logical completion 对应的完整可见文本与 Lithos 相同；教学核心产生 36 个 logical completion，措辞与长度不同。这是此案例的观察，文本相同也不等于证明完整内部 token IDs 或所有输入下的 greedy 路径相等。

27B 三轮日志的 prefill GPU command-buffer duration 为 41.78–42.89 s，测量轮 `setup_ms=0`。native `steps=[48, 48, 48]`（预热 64）包括排队后的 done/no-op，不是 45 个 committed 输出的计数。不能把约 42 秒解释成首次 CPU 编译，也不从这些 step 数推导每 token 吞吐。

报告的 `/api/ps size_vram` 是 unique session buffer allocations，不包括 CPU packs 或进程 RSS，也不是总系统内存/峰值。与教学核心 MLX allocator peak 的含义不同，分别保存，不混成同一内存排名。Source JSON 都保留 bytes；下表以十进制 GB（bytes / 1e9）展示，不称为 GiB。

| 内存记录 | 原始 bytes | 十进制 GB | 测量含义 |
|---|---|---|---|
| 0.8B 教学核心 | 511464387 | 0.511 | MLX allocator peak |
| 0.8B Lithos | 700525328 | 0.701 | unique session buffer current |
| 27B 教学核心 | 15493470551 | 15.493 | MLX allocator peak |
| 27B Lithos | 17001511056 | 17.002 | unique session buffer current |


[独立算术复核](../../tutorial_code/examples/chapter11-lithos-arithmetic.json) 、[同一 27B 三路复核](../../tutorial_code/examples/chapter11-paired-27b-arithmetic.json) 和 [集成时 CPU 复核](../../tutorial_code/examples/chapter11-lithos-integration-audit.json) 核对了实际输入、输出计数、EOS、native fallback、cache/verify、计时与中位数。这些复核都读取真实 GPU 运行后的报告，本身没有额外运行模型。

最终当前报告以带 startup 元数据的版本为准。较早 0.8B [测量](../../tutorial_code/examples/chapter11-lithos-0.8b-measurement-before-startup-metadata.json) 与 [smoke](../../tutorial_code/examples/chapter11-lithos-0.8b-smoke-before-startup-metadata.json) 单独保留，未用它们替换本表。各 owned 服务退出记录见 [清理检查](../../tutorial_code/examples/chapter11-owned-server-cleanup.json)。

## 从第十一章初版到扩展版

```diff
- Lithos 只有安装资格/CPU 格式准备，性能未测
+ 官方固定源码实际运行，保存 0.8B 与 27B GPU smoke/重复测量
+ 相同 27B checkpoint 补齐教学核心与 oMLX 的重复请求
+ 分列核心/API、EOS 计数、startup/预热/正式请求与内存边界
```

```bash
git diff tutorial-11 tutorial-11-v2 -- tutorial_code/baselines docs/tutorial-v2/11-benchmark.md
```

本次扩展没有重写模型数学或重新调整 M4 配方；它完成第三条实际 baseline 与同目标记录，并保留每次已运行结果的范围。

## 本机 M4 矩阵加速试验

这是在同一 M4 Max 40-core / Apple9 上增加一个明确的官方开关，不是把整份 M5 配置套到 M4。官方 `m5_max_40c` 要求 Apple M5 Max / Apple10，实际 `validate_device` 已拒绝本机；[拒绝证据](../../tutorial_code/examples/chapter11-m5-profile-rejection.json) 记录没有创建 native GPU device，也没有绕过检查。

官方 [M5 配置快照](../../tutorial_code/examples/chapter11-m5-profile-config.json) 写的是 M5 Max 40 GPU 核、`memory_gb=48`；GPU working-set 和单 buffer 容量分别为 `37.44` 与 `28.08`。这些是 profile 的容量字段，不是模型实际 RSS，也没有在本机应用。

实际保持 `m4_max` / Apple9、TG=1、attention=v1、commute_norm=True、原 NVFP4 权重与 EOS、context2048。只为自有服务设置官方 `MONOLITH_ACCELERATOR=on`；未修改已安装的模型、kernel、芯片 JSON、M5 cost/fusion recipe 或设备验证。复现入口与原始记录见 [M4 实验说明](../../tutorial_code/baselines/m4-accelerator-README.md)。默认 `--accelerator profile` 仍让本机 profile 选择 off，并清除继承的同名环境变量。

### 实际运行路径的证据

生成完成后，诊断 launcher 读取 live Engines 的 program。0.8B/27B prefill program 分别有 97/321 个 `gemm_tile` op，全部 `COOP_A9=1`、MSL4，T predicate 为 [1,128]，本次真实输入 T=19 在其中。这是完成生成关联的编译/编码 program 与有效 predicate 证据，不是每个 op 的独立 GPU timestamp。

Apple9 cooperative matrix tile 与 M5 神经加速器是不同硬件路径。T=1 decode program 没有 gemm_tile，但官方 accelerator 开关也影响部分单行 projection 与 GDN prepared 路径，所以不能把整个收益只归因于这些 prefill tile。

### 历史 off 与本次 on：分别保存事实

同一目标、19 输入 IDs、greedy、正常 EOS、cap128、并发1、预热后3次请求；cache0、verify1、setup0。以下 off 是已归档的默认配置历史运行，on 多了一层后置 program 读取/打印，既不同时段交替，也不完全同仪器。

| 目标与记录 | 完成 IDs（含 EOS） | 首文本中位数 | HTTP 全请求中位数 | prefill command-buffer 中位数 |
|---|---|---|---|---|
| 0.8b 历史 profile-off | [34, 34, 34] | 1460.169 ms | 1.657610 s | 1449.785 ms |
| 0.8b 本次 accelerator-on | [29, 29, 29] | 223.002 ms | 0.353966 s | 214.360 ms |
| 27b 历史 profile-off | [45, 45, 45] | 42781.459 ms | 46.192591 s | 42769.921 ms |
| 27b 本次 accelerator-on | [45, 45, 45] | 433.515 ms | 2.208017 s | 420.269 ms |

27B on/off 都是 45 committed IDs（44 logical completion），此 prompt 的完整可见文本相同，各自3轮稳定；on 的 decode GPU command-buffer 中位数约 1766.329 ms，历史 off 约 3341.699 ms。HTTP 没有返回完整 generated-ID 数组，文本相同不是完整 token/tensor 等价证明。

0.8B on 是 29 IDs /28 logical，历史 off 是 34/33，文本也改变；本次 on 的可见文本与该 prompt 的 oMLX 0.8B 记录相同。模型句子中的“减少显存”解释不准确，不能拿生成文本替代 KV cache 的数学定义或 kernel 数值验证。因此不把全请求 TPS 比例称为相同生成序列的无损加速。

后置诊断会给客户端总耗时增加少量 CPU 开销；已经结束的 native GPU command-buffer 时间不含这段读取/打印。prefill GPU 时间比较的输入是相同19行，但仍只代表这个参数和输入案例，不校准整份 profile，不宣称最优配置，也没有新增 batch、投机或 p99 实验。

本次 on unique session buffer 当前值：0.8B 为 703,742,032 bytes（十进制 0.704 GB），27B 为 17,045,438,688 bytes（十进制 17.045 GB）。这里不是 RSS、系统总内存或峰值；源 JSON 保留 bytes。全部自有服务已退出，见 [比较/cleanup](../../tutorial_code/examples/chapter11-m4-accelerator-comparison.json)。

[0.8B on 原始报告](../../tutorial_code/examples/chapter11-m4-accelerator-0.8b.json)、[27B on 原始报告](../../tutorial_code/examples/chapter11-m4-accelerator-27b.json) 与 [独立 CPU 算术审阅](../../tutorial_code/examples/chapter11-m4-accelerator-arithmetic.json) 保留 profile-off/live-session-on、真实路径、实际输出与所有测量边界。`tutorial-11-v2` / `baseline-trio-v1` 的默认配置数据保持原样；新增实验用 `tutorial-11-v3` / `baseline-m4-accelerator-ab-v1` 固定。

```bash
git diff tutorial-11-v2 tutorial-11-v3 -- tutorial_code/baselines docs/tutorial-v2/11-benchmark.md
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --accelerator on --max-new-tokens 128 --repeat 3 \
  --out .artifacts/lithos-baseline/m4-accelerator-on.json
```

[下一章：12 从同步到流水线，再到算子融合](12-从同步到流水线再到算子融合.md)
