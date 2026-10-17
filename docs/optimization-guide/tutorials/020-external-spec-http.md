# E020：把已资格通过的官方投机路线变成真实 HTTP 对照

状态：**已完成，15 个进程、90 个请求的 HTTP 合同及独立审计通过。** 以下保留预注册设计，实测见第 8 节；09568bc 冻结时的原教程也保存在证据归档。CPU runner 使用官方 oMLX/Lithos 服务入口，E015/E017 的 observer 时长不参与速度排名。

CPU prepare 已于2026-10-17通过，固定8658个source/runtime/model/pack pins；计划SHA256为
`e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc`。
计划、真实启动参数预览、native源码manifest和jobs副本在
`bench/m5_ultra_256gb/external_spec_http/prereg-v1/`。13项CPU fixtures由实现方和主线程独立通过；
修复了慢SSE不断续读可能越过服务暂停deadline的问题，现在有300秒绝对请求期限和有界回收。

## 1. 本轮只回答一个小问题

固定已有的 **19 输入 → 最多 32 输出、greedy、正常 EOS、C1** 文本请求，比较三个 fresh pair 内的：

* 官方 oMLX `none` 对 `VLM MTP`；
* 本机 M5 Ultra 适配 Lithos `none` 对 `DSpark`；
* 增加三个当前 native target-only fresh process，作为同一轮 HTTP 锚点。

所有引擎读取同一个已冻结 NVFP4 target snapshot，但数值路线不同，**Lithos qualified IDs 与 native/oMLX 不同**。每个框架都必须保持自己的 fresh control 输出合同；不能把这个实验称作跨框架严格等价、全质量 Pareto frontier 或全网最快。长 prompt，以及 256/1024 输出会改变投机成本回收点，属于后续新资格，不在本轮追加。

可以在统一输入、输出预算、EOS 与 C1 条件下比较三方 E2E latency/throughput，但每个点必须精确标注自己的输出合同：native/oMLX 与 Lithos 的实际文本不同。这里限制的是共享 exact oracle/质量等价的结论，并不禁止有明确限定的跨框架性能比较。

E015、E017 已分别通过真实 engine raw IDs 与实际 route 资格。本轮外部官方 HTTP 仅给文本，重新编码的 visible IDs 与此前同框架 qualified raw IDs 一致，仍然**不等于当次取得了实际 engine raw IDs**。native shim 在最终 SSE 附带当次 raw IDs，可做额外直接核对。不得混用 oracle。

## 2. 冻结模型、量化和真实入口

| 对象 | 固定 revision / 配置 |
|---|---|
| target `mlx-community/Qwen3.8-27B-nvfp4` | `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6` |
| oMLX MTP `mlx-community/Qwen3.8-27B-MTP-nvfp4` | `b9b764dada894b8c416f89cbc8cf99bc7e20b831`；checkpoint block size 3 |
| Lithos DSpark `LithosAI/Qwen3.8-27B-DSpark-NVFP4` | `b169bc4c8dc805a9e6178fa82f6a7b84b169a457`；7 proposals + anchor |
| oMLX | `0.7.1.dev1`；MLX/MLX Metal `0.32.3`；mlx-vlm `0.7.7`；mlx-lm `0.31.4.dev132+g94cdcae13` |
| Lithos checkout | `223658090ae549e59220bdec459957cbdad7926a`，本机 native `m5_ultra_80c`，无 Max recipe |
| native | prepare 时复制当前 core；compile_decode/fuse_decode ON，2 GiB，gates compile ON，norm reuse OFF，无 drafter、无 step observer |

native 保留已采用的 grouped prefill 默认逻辑，无 monkeypatch。C1 时 `len(reqs) < 2`，不会进入 grouped 路径；这不是 grouped prefill 的新性能实验。

四个共用环境变量沿用既有资格：`MLX_MAX_OPS_PER_BUFFER=50`、`MLX_MAX_MB_PER_BUFFER=50`、`MLX_BFS_MAX_WIDTH=20`、`MLX_METAL_FAST_SYNCH=0`。Lithos 另固定 `MONOLITH_ACCELERATOR=on`。不使用调频、后台 keep-warm、清 cache 调参或新量化 recipe。

Lithos **复用 E017 原 pack**：

| arm / payload | 现有目录（相对仓库） | capacity |
|---|---|---:|
| none target | `.artifacts/m5-ultra-256gb/lithos-spec-packs-v1/none/target-b6042c39f78b9df33a19ac1c` | 2048 |
| DSpark target | `.artifacts/m5-ultra-256gb/lithos-spec-packs-v1/dspark/target-61919dc70e5833f7b19fc670` | 2054 |
| DSpark draft | `.artifacts/m5-ultra-256gb/lithos-spec-packs-v1/dspark/draft-f7cb21710c30a1c544e31b99` | 2054 |

`--pack` / `--draft-pack` 指向 exact manifest 目录，官方 `ensure_pack` 验证后直接复用。2054 是 `2048 + gamma - 1`；不重建 pack。`--draft-quantization none` 保留 source NVFP4 及原 BF16 Markov embedding，不把已有 NVFP4 解读为 BF16。prepare 从 E017 读取 manifest/payload SHA，run 前后完整核对；日志还必须显示复用这些 exact pack。

oMLX 直接运行 `.artifacts/omlx-baseline/.venv/bin/omlx serve`，每进程使用全新 model symlink/base-path，只装固定 target。`state/model_settings.json` 复制 E015 的完整 per-model settings（version 1），仅依据 none/MTP 原资格选择对应配置；默认 block size 保持 checkpoint 的 3。设置 balanced、max-concurrent-requests 1、chunked prefill OFF、speed priority、decode fairness、`--no-cache --no-hf-cache --sse-keepalive-mode off`。原 CLI 调用同 PID 的 `uvicorn.Server.run`，不启动 fork wrapper。

Lithos 直接运行其 `.venv/bin/python -B -m monolith.serve`，固定 `--max-context 2048 --prefill-chunk-size 128-exact --keep-warm 0 --no-warmup --local-files-only`。none 显式 `--no-draft`；DSpark 显式同源 draft、`--draft-kind dspark --draft-block-size 7 --draft-sampling argmax --draft-quantization none`。没有 `--kernel-config`，官方 `uvicorn.run(..., workers=1)` 保持同 PID。

Lithos serve 没有关闭内存 prefix cache 的 CLI。19-token prompt 小于 128-token 资格下限，因此预期无命中，但必须逐请求检查官方日志 `cached_prompt_tokens == 0`。不能把机制存在说成 cache 被禁用；HTTP cache 字段缺失记为 unknown，不填 0。

## 3. API 和普通 EOS

workload 为 `.artifacts/m5-ultra-256gb/fullqual-build2/workload.json`，SHA256 `65be876cac2397bf9afa76105c8e3fcce23de5e6c8652a28cb1b5f39640c153c`。

* oMLX/native 使用 `/v1/completions`，prompt 是固定 19 IDs 经同 tokenizer 解码的文本；重新编码必须等于原 IDs。温度 0，repetition penalty 1，max_tokens 32。
* Lithos 使用 `/v1/chat/completions`，单条 user 文本 `用一句话解释什么是 KV cache。`，温度 0、top_p 1、max_completion_tokens 32。prepare 使用官方 tokenizer 的 chat-template 参数 `add_generation_prompt=True, enable_thinking=False` 纯 CPU 核对得到完全相同的 19 IDs。
* 全部 SSE 开启 `include_usage`，不传 ignore_eos 或自定义 stop。native 从 target 读取 EOS IDs，`ignore_eos=False`。固定样本资格中没有出现 EOS，但不由此宣称其他 EOS 分支已验证。
* 若出现提前 EOS、错误 finish、不同数量或不同 visible IDs，该请求原样保存并判合同失败，不改 prompt、不延长生成、不补请求。

oMLX MTP 目前只路由单请求，其他等待/运行/预填请求可使其走普通 BatchGenerator。Lithos 官方服务有单在途限制，并发会返回 429。所以本轮严格串行 C1，不从开关推断 C>1 的 MTP batching 能力。

## 4. 预先固定的矩阵、计时和结果

| fresh pair | 进程顺序（每个独立加载一次） |
|---:|---|
| 1 | native → oMLX none → oMLX MTP → Lithos none → Lithos DSpark |
| 2 | Lithos DSpark → Lithos none → native → oMLX MTP → oMLX none |
| 3 | oMLX none → oMLX MTP → Lithos none → Lithos DSpark → native |

总 **15 servers × (3 warm + 3 formal) = 90 requests**，其中 45 formal、合同通过时 1440 formal output tokens。所有 warm、formal、负收益与失败行保留。每请求固定完成后等待 2 秒；资源采样在请求前后，均在 HTTP timer 外。没有重复挑跑或丢弃最慢样本。

计时沿用 `lithos/client.py` 的 SSE 边界：client 创建后、POST 发出前开始，第一次非空 visible text 为 TTFT，[DONE] 收到为终点；stream close 的耗时另报。正式数据每次记录原 SSE payload/text event、完整文本、实际 usage、visible canonical count、实际 finish、first-chunk canonical prefix 条件。

每请求有固定 300 秒绝对 deadline（不仅是 HTTP 每次读取 timeout）。超时或 controller 收到取消信号也保留已收到的 SSE，再退出并 reap 实际 server，避免 trickle stream 越过暂停服务的 lease。

* **主 E2E latency**：POST 发出前→收到 [DONE]（秒），越低越好；完整请求耗时包含 prefilling、generation、传输及服务开销，stream close 另报。跨框架主图使用这个完整请求指标，不以 observer decode wall 或首 chunk 后局部时间替代。
* **TTFT**：首个非空文本时间减 POST 起点（秒），作为补充指标。不是 GPU 首 token 时间。
* **C1 aggregate throughput**：HTTP 实际 completion_tokens / POST→DONE 秒数，不使用硬编码 32 作失败样本分子。
* **client estimated interactivity**：剩余 canonical tokens / 首文本→DONE，仅首 chunk 是全输出 canonical prefix 且数量吻合时可计算。分块策略会影响它，不等同于 engine decode tokens/s。

固定 C1 且实际输出都为 32 时，aggregate throughput 就是 `32 / E2E latency`，两者互为重标度；不得把它们当作两个独立质量或服务能力维度来宣称完整 Pareto 优势。

每进程以自己的 3 个正式样本中位数描述；none/on 在各自 pair 内计算 throughput、wall、TTFT 比。全部三个独立 ratio 原点必须展示，三 pair 完整且合同均有效时才额外报 paired throughput geometric mean。没有注册采用门，不把正向 GM 当作所有 process 都变快。native 三个当前锚点完整展示；旧 native、E015/E017 observer 时长不混进这轮图。

## 5. 路由证据的强弱边界

oMLX 普通 INFO 日志必须有 MTP attach、六个不同 request 的 `vlm_mtp decode started` 和六条匹配的 stats；要求 block size 3、正常 length、emitted 32、rounds/accepted 为正，无 skipped/fallback/ignored 警告。日志读取 live `accept_lens`，但 `accepted=a/b` 的 b 是 `rounds*(block_size-1)` 的**名义容量**，不是末轮被截断后的实际 drafted count；原样记录，不能误报为新 raw acceptance trace。none 必须没有 MTP starts/stats/attach。

Lithos 普通日志必须指向 m5_ultra_80c、相同 target/draft、空 recipes、复用相同 pack，六条 Generation 的 configured verify width 为 1/8、cached_prompt_tokens 为 0，正式请求 setup_ms 为 0。`verify_tokens` 是配置宽度，`steps` 是诊断值；它们**不独立证明每轮实际接受数**。实际 raw route/acceptance 资格来自同源码、同 pack 的 E017；这里的无 observer HTTP 不擅自越过该证据边界。官方 log 的内置 GPU/wall 时间可以保存诊断，但不作排名。

## 6. owner、资源和恢复合同

`run.py` controller 直接 Popen 每个真正的官方 CLI/native server。每次准备后及请求后，`lsof` 的 LISTEN PID 必须**唯一等于 child.pid**；`ps`/resource exact-owned metadata 的 PPID 必须是 controller。不把 `guarded_v2` 的 child controller 虚称模型 owner。

每 server finally 执行 terminate → wait，必要时 kill → wait；保留 exit code、reaped、PID absence、listener absence、TCP connect failure，任意一项不满足即停止矩阵。资源沿用 `resources_v2`：已授权 PID 74384 只有处于 T 状态且 identity 不变时允许；同采样记录 AC、thermal、RSS、pressure、所有 swap/Pageouts/压缩计数。guard 不代表 Swapins/分页完全不变，未识别 GUI GPU 使用仍未知。

由现有 `suspended_service.py` 作为唯一外层 lease 暂停/恢复 74384，job **直接运行 E020 controller**。controller 核对 lease 父进程/child PID、已暂停 identity、deadline。启动前要求至少 1020 秒余量（readiness 上限 900 秒）；请求前资源采样之后再次检查至少 420 秒余量，覆盖绝对 300 秒请求与清理。资源采样自身有 30 秒 deadline；超时/取消的 raw 行写入后立即进入 reap，不再先做活 server 的额外采样；reap 阶段短暂屏蔽再次取消的信号，清理后恢复。最后 resume 发生在 controller 退出后，必须由主线程/后续 audit 读 lease `resume.json` 核对 T→S 和 identity；controller 不提前声称已恢复。

安全/transport/lifecycle 失败写 partial JSON 并停止；输出或 route 合同失败保留所有六个请求和后续计划进程，以 invalid 标注，绝不重新抽样。不满足合同的 process 仅保留诊断中位数，不进入 paired speed ratio。

## 7. 可复核运行入口（预注册计划）

源码均在 `bench/m5_ultra_256gb/external_spec_http/`。`prepare.py` 不导入 MLX、monolith runtime 或官方服务；只读 metadata/tokenizer、复制源代码、继承已资格 payload SHA。小 source 会即刻校验，大 weights/pack 完整 SHA 在 scheduled run 的矩阵前后核对。每 arm 还检查全体 size/mtime 与小 source SHA，不在请求 timer 中 hash。

CPU 准备（必须选择不存在的目录；此步会做较多 CPU source hashing，避免与 GPU 性能测量重叠）：

```sh
.venv/bin/python -B bench/m5_ultra_256gb/external_spec_http/prepare.py \
  --out .artifacts/m5-ultra-256gb/external-spec-http-v1-inputs
```

审查产生的 `plan.json`、`launch-preview.json`、`native-source.json`、frozen sources，记录 preparation 中的 plan SHA 并先 commit。随后创建外层 jobs JSON，唯一 argv 为：

```text
/Users/ryanli/code/minivllm/.venv/bin/python -B
/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/external_spec_http/run.py
--plan /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/plan.json
--plan-sha256 <reviewed plan SHA>
--out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1
--lease /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1-service-lease/lease.json
```

按上述四个共用 MLX 环境变量调用现有 lease；不要再套 guarded_v2：

```sh
.venv/bin/python -B bench/m5_ultra_256gb/suspended_service.py \
  --pid 74384 \
  --jobs .artifacts/m5-ultra-256gb/external-spec-http-v1-jobs.json \
  --out .artifacts/m5-ultra-256gb/external-spec-http-v1-service-lease
```

CPU fixtures：`python3 -B -m unittest discover -s bench/m5_ultra_256gb/external_spec_http -p 'test_*.py' -v`。它们仅模拟 SSE/log/cleanup 返回值，不创建 listener、不加载模型。预注册时尚未运行真实服务，因此最终 route log 和官方 HTTP 行为须经实际校验；parser 若暴露未知官方语义，原尝试冻结后另开修订计划，不能事后修改已测门。本轮实际结果如下。

## 8. 实测：正确性先过，再比较完整 HTTP

预注册 commit **09568bc**，plan SHA256 **e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc**。13 项纯 CPU fixtures 和双人源码审查通过后，按原 15-process 顺序串行执行；没有额外挑跑。15 臂全部有效，45 warm + 45 formal = **90 请求**，正式实际产出 **1440 tokens**。

每个下面的数字都是一个独立 fresh process 内三个正式请求的中位数；三值按 pair 1/2/3 排列，不能把三轮合并后只留下最好的一点。

| arm | E2E latency（ms） | aggregate throughput（tok/s） | TTFT（ms） |
|---|---|---|---|
| native target-only | 716.098 / 726.786 / 701.526 | 44.687 / 44.029 / 45.615 | 131.060 / 139.880 / 135.450 |
| oMLX none | 679.324 / 682.184 / 679.003 | 47.106 / 46.908 / 47.128 | 113.478 / 113.560 / 113.121 |
| oMLX MTP | 752.063 / 748.409 / 752.926 | 42.550 / 42.757 / 42.501 | 100.239 / 99.763 / 99.612 |
| Lithos none | 788.896 / 816.022 / 812.920 | 40.563 / 39.215 / 39.364 | 246.423 / 280.766 / 278.709 |
| Lithos DSpark | 731.923 / 741.955 / 752.397 | 43.720 / 43.129 / 42.531 | 241.591 / 251.650 / 263.404 |

| 同框架配对 | pair 1 | pair 2 | pair 3 | 配对 throughput GM |
|---|---:|---:|---:|---:|
| oMLX MTP / none | 0.903280× | 0.911512× | 0.901819× | **0.905527×** |
| Lithos DSpark / none | 1.077841× | 1.099826× | 1.080440× | **1.085991×** |

**MTP 在本工作负载三次都降低完整请求吞吐**，同时 TTFT 更低。DSpark 三次都提高自身 none 基线的吞吐。当前 native target-only 锚点尚未超过本轮 oMLX none 的 E2E/throughput；native DFlash2 不在这个 15-process 矩阵中。不能拿模块 speedup 加乘、或用更短 TTFT 代替完整请求耗时来改变这些结果。

质量合同分层保持原样：90 次 visible canonical 输出均与各自已有 qualified sequence 一致；其中 18 次 native 请求另有当次实际 raw IDs（576 IDs）核对。72 次官方外部 HTTP 请求提供的是文本及重新编码后的 2304 IDs，不能把它们升级为当次 engine raw-ID 采集。oMLX 18 个 MTP 请求的 stock started/stats 证据完整；Lithos 每个进程的 6 条 stock Generation 记录满足原 verify-width、cache-hit=0 和正式 setup_ms=0 合同。不同框架的 continuation 仍不完全相同。

独立审计从每个原始 SSE payload/timestamp 重建文本、usage、first text、DONE、实际数量、E2E、TTFT、吞吐和三对 ratio，全部与 runner 记录吻合。15 个实际 listener PID 都是 controller 的直接子进程，全部 wait/reap、PID 消失、端口消失；74384 的外层 lease identity 不变，T→S 恢复成功。

资源共 **225 个快照**。各进程区间 Swapins/Swapouts 增量均为 0；但首个 native 整个 owner 区间 **Pageouts +1、Compressions +813047**，Decompressions 也有变化，不能写成“分页和内存状态完全不变”。样本 RSS 不是统一内存/GPU allocation 的峰值，不据此比较模型真实显存占用。

[完整独立审计与归档](../../experiments/m5-ultra-256gb/E020-external-spec-http-v1/README.md) 保存 **9131 个成员**，压缩 **29,444,554 bytes**；全部 tar member size/SHA 独立复核通过，archive SHA256 **ba9c668ce0d98760f5413307af86400e4129d43974ab7d3d64cdfcfe0bdfddec**。8590 份冻结源码/config 独立重算 SHA，另 68 项权重/pack/runtime 大 payload 保留运行时矩阵前后完整 SHA receipts；CPU 审计没有重新读取或归档大权重与可运行 venv。

旁置的六个负向挑战全部被拒绝：raw SSE 文本被改、时间戳倒退、伪造 MTP acceptance、伪造 listener PID、伪造父 PID、错误恢复 identity。四个原始正向基线先通过，再只修改内存副本；不改原始数据或已验证的 tar。这说明审计确实读取原证据，而不是仅信任 `valid: true`。
