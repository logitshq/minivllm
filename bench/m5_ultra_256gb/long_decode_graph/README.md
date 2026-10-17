# E024：固定 1536 的 long-context graph-only 候选

2026-10-17 更新：**已在真机首点 B1/used257 失败，默认 OFF，不可采用，没有性能资格。** [失败证据](../../../docs/experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md)已归档；后续[四组隔离](../../../docs/experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md)确认，在该点改变 attention bound 足以产生差异。以下保留原预注册设计与 CPU 阶段记录，不能将其中待执行事项读成当前结果。旧 candidate/qualifier 保持冻结，新两档方案另目录实现。

本目录不修改 `tutorial_code/miniengine`，不安装到生产 runner，不修改 E023 的五份冻结 Python 源，不改变 D fixtures。原 E023 长任务 fallback baseline 必须保留。

## 假设与最小 API

当前 `CompiledDecode` 在 used length 超过 256 时回到 eager。8 个 D prompt 为 886–978 tokens、输出 cap512，因此从首个 decode 就超过此范围。假设：在这个长生成范围内，共享固定 attention L1536 的 graph 可以减少每步构图/提交成本，收益大于额外 masked attention 工作和编译成本。

```python
from bench.m5_ultra_256gb.long_decode_graph.candidate import LongContextDecode

candidate = LongContextDecode(engine.model)                 # 默认 OFF，直接原 model
candidate = LongContextDecode(engine.model, enabled=True)   # 只在 bench 显式 opt-in
logits = candidate(tokens, context)
```

未来资格 runner 可在隔离的 candidate 臂把 `engine.runner.decode_model` 指向该 callable；本原型不自行接入、不 monkeypatch core、不监听网络。没有 `fuse_decode`/`bound`/bucket 参数。`enabled`、`model`、`bound` 是只读属性；改变设置构造新对象。

严格范围为 **B1…8，257 ≤ `ctx.max_len` ≤ 1536**、normal target-only decode、原 Qwen hidden5120/64层、BF16 NVFP4/group16、pinned M5 Ultra + MLX backend。长域以外直接调用原 model。它不复用原 256 graph 的短域，未来若组合两种 callable，需单独登记路由和资格。

`ctx.max_len` 是本次实际使用长度，图内 attention bound 是 1536，两者不能混淆。K/V 两个 backing 都必须至少 1536，且 shape/dtype/行数一致；GDN conv/S 也校验本模型的 shape/dtype。调用者仍负责 `positions`/`lens` 数值对应实际 cache history，和 `lens=positions+1`；热路径只验 metadata，不调用 `.item()` 读 GPU 值。真实资格会核对这些 producer 不变量。

## 为什么没有继承或文本改写原 class

`CompiledDecode.__call__` 在 eligibility、cache backing 和 compiled closure 中分别写死 256，fused/packed 初始化又夹在同一函数中。仅继承、改一个常量无法安全改变三个位置；修改 AST 或字符串很容易留下旧 closure。隔离的小 class 只复用其 `CompiledDecode._states`，独立表达 graph-only 路线。

与 pinned 原实现的精确差异：

| 项 | 原 `CompiledDecode` | E024 |
|---|---|---|
| 范围 | B1…8，used length1…256 | B1…8，used length257…1536 |
| 三处长度 | guard ≤256、K backing≥256、closure max_len256 | guard ≤1536、K/V backing≥1536、closure max_len1536，来自同一构造时 bound |
| 编译内容 | 可选择 packed/fused projections | 普通 model call，完全没有 `_decode_projections`、PackedDecode 或 fused kernel |
| cache tree | 显式输入/输出；KV 建新 `[:]` wrapper | 保留，调用原 `_states`；返回结构不足时在修改 caller cache 前失败 |
| 参数 | `mx.compile(..., inputs=model)`，fused 时另带 packs | `inputs=model`，没有权重冻结常量或派生 packs |
| Python graph key | fused + 每 GDN gates/norm 开关 | bound + 每 GDN gates/norm 开关；B/backing shape 仍由 MLX 内部 specialization 处理，不声称所有 B 共一次 trace |
| guard | ctx/cache 类型与精确字段、设备、模型、量化、K 长度 | 保留并收紧：token/position/lens 的整数 dtype/shape、K/V 两者长度/dtype、GDN 状态 shape/dtype、层对象身份 |
| 模型生命周期 | 可变公开 model 属性 | 捕获对象只读；层对象替换回 eager；其余结构/非数组配置变更要求重建 callable |
| 来源 | 由外层 owner freeze | 启用时额外核对 31 份实际模型 Python 源与 cache/context/gdn/compiled_decode import root |

启用前按实际 `type(model)` 的文件位置定位模型源，要求 pinned `model/qwen3_5.py`，允许 controller 的独立 frozen tree。`model-source-pins.json` 完整列出当前 31 个 core Python 文件；缺失、额外 Python 文件或 SHA 变化均失败。其 SHA256 为 **`7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5`**。这不替代 owner 的模型权重、config、MLX/runtime pins。

默认 OFF 不导入 MLX、不校验文件、不持有请求 cache，只 forward。ON 后 graph closure 只捕获 model、layout、runtime helpers 和固定 bound；请求 tokens/positions/lens/states 都是显式参数。KV 使用新 wrapper 防止 MLX `__setitem__` 修改原 caller handle；GDN 状态按 core 的整体替换方式返回。该内存/alias语义仍须真 GPU 检验。

原模型的 array 参数可替换：`inputs=model` 保留原 CompiledDecode 的语义，CPU mock 证明接口把原 model 作为 captured inputs、不会偷偷另存一份权重。**mock 不证明真实 mx.compile 在所有 shape 下的 replacement 正确。** 模型结构、非数组超参数、层内部 module 对象更换不支持 live change；重新构造。gates/norm 的公开 setter 切换由 graph key 区分，但本轮实验参数仍须冻结。

## CPU 合同结果

```sh
.venv/bin/python -B bench/m5_ultra_256gb/long_decode_graph/test_cpu.py -v
```

15 项 stdlib mock tests PASS，覆盖默认 OFF、未知参数、只读 model/bound、used256/257/1536/1537、B0/1/2/3/7/8/9/64、K 或 V backing 不足、extra ctx/cache fields、输入 metadata/dtype、backend/quant/model/层替换拒绝、graph key、`inputs=model`、请求 cache 分离、返回 state 不完整，以及 source pin corruption。单独 subprocess 在禁止 MLX/core/tokenizers/numpy import 与 socket 创建的条件下成功 import 本模块。

没有真实 Metal compile、没有实际 GPU arrays、没有 loaded model weights，也没有数值/速度结论。CPU 测试中的 fake compiler 直接执行 Python，不能测出 MLX fusion、attention kernel selection、FMA 或舍入变化。

## GPU 资格候选：先 exact，后性能

在 root 另行冻结计划前，以下只是待审核矩阵：

1. **实际 D 输入：** 复用 E023 同调度 paired harness 的完整 cap512 C1/C8 及短退休日志。reference 是实际 eager fallback，candidate 仅替换 `runner.decode_model` 为本类 ON。dtype/quant、gates、norm OFF、allocator、scheduler、EOS、prefill 与全部 fixture IDs 相同。candidate graph 内多出的 padded attention shape 是唯一模型计算差异。
2. **B 范围：** 至少 B1…8 各一次，从对应数量的真实 D prefills 初始化独立 cache，比较后续 32 个同调度 greedy steps 的全 vocabulary logits、所有64层/128数组、tokens。C8 逐步退休不保证自然经过每个 B，因此显式覆盖 B2…7，不能只看 B1/B8。
3. **边界与 mixed lengths：** 支持范围首位置256（used257）、真实 D 起点及 mixed positions、used1023/1024/1025、1279/1280/1281、1490、1536；used256 和1537 应走 fallback。保存参数 `max_len=max(positions)+1`、每行 lens 与 physical allocation。非自然边界用明确命名的 diagnostic state/input，不能把人为补历史称真实 D free generation，也不改 D 性能 fixtures。
4. **KV backing：** eligible 用1536和1792，额外非零 padding 检查 mask；用 used900/backing1024 检查 underallocated-for-graph 但 eager 合法的 fallback。不得把 used1536/backing1024 的非法 core input 发给 GPU 来冒充 fallback 测试。
5. **生命周期：** actual lookahead 的产生/消费/失效，late arrival、不同 cap/EOS退休、batch reorder、clear/reuse、独立新请求 seed/cache，全量 committed + ahead 状态不得串线。额外 ctx 字段、不同 dtype/设备/量化 metadata 用 CPU拒绝和必要的安全 GPU fallback 验证。
6. **参数捕获：** 在真实模型上先编译 B1，再 B8，再回 B1；更换并恢复一份原参数数组（至少 norm weight 与一份 quantized weight/scales），每次与同参数 eager 对照。全部原参数最终恢复、签名不变；没有 packed projection 权重可代替检查。

每项使用原 dtype unsigned views逐位比较，先核 dtype/shape/数组数量并检查 finite；全部 logits/cache exact，不能只比 argmax或allclose。任何失败先写 partial/inputs/首个差异再 STOP，不做性能；固定1536的 masked SDPA 即使数学上等价，也可能改变 reduction/kernel 路线而不逐位相等。

保存每个真实 compiled function 的触发/shape 证据（资格观察器），实际 eligible/fallback 次数与理由；不能仅从 CLI ON 推断图在运行。首 compile、资格后首调用、warm 调用分开。多 shape specialization 的编译开销保留，不把warm micro当可采用资格。

## 长任务配对性能门候选

数值与生命周期全部通过，才另开无 observer 的 E023 D HTTP 测量。两臂各3个 fresh process，C1和C8，完整已冻结 D/cap512、正常EOS、相同 client jobs/arrival政策；3对顺序 OFF→ON / ON→OFF / OFF→ON。baseline仍是原长任务fallback，candidate只多这一项 graph。所有 raw requests、失败、cold/warm与输出长度分布保留；不修改任务让 graph 更容易赢。

主轴保持 `Y=实际 committed target tokens / 全请求窗口` 与 `L=p50(POST→SSE DONE)`，p95、TTFT和输出长度为配套信息，正确性是硬门。

供 root 审阅的采用门：C1+C8 六对 throughput ratio 的 GM≥1.10，每个 C 的3对 GM≥1.00，每个 raw pair Y ratio≥0.97；每个 C 的 process-median p50 E2E latency 不高于 baseline的1.03倍；并且至少一个 candidate配置中位点在 throughput–E2E latency 上严格支配一个实际 baseline配置点。p95/TTFT/长度及每一负点全部报告，不据辅助项事后挑配置。此候选门尚未冻结，GPU开始前必须写入 machine-readable plan。

先测 graph-only 的整请求收益，不以微基准替代。L1536 每步比真实887等长度读更多masked列；双阶段prefill不受本模块加速；C>8还会回 eager。这些都可能让完整D吞吐不增反降。未通过就保留证据，默认 core保持原样；不追加调小bound/挑prompt后称本候选通过。

一个长域 bound 无请求 cache owner、无allocator/prefix cache策略、无bucket调度器。只有 graph-only 经完整资格和实际长任务门通过，才值得讨论融合或第二个bound；任何扩展另登记。当前不存在“已实现1536生产支持”或“已超过官方引擎”的结论。


## 2026-10-17: cold-service execution adapter ready

`cold_owner.py` owns exactly one direct `e024-diagnostic` child. It invokes the
unchanged original E024 owner plan, qualifier and candidate: gates/norm/fuse/core
compile OFF, 1800-second numerical budget, 1830-second outer work budget, bounded
TERM/KILL and postflight. This does not inherit the gates-ON HTTP qualification.
The original earliest graph point is B1/used257/bound1536 after independently
loaded weights and real-prefix cache checks; the first numerical difference is
retained and stops the diagnostic before any performance claim.

`controller_v5.py` accepts a pinned, independently audited v4 RESTORED predecessor
from the immutable experiment archive. It keeps the original stop/physical/log/
HTTP-worker recovery operations and represents this actual single-child phase.
`audit_restore_v5.py` independently checks retained restoration and child-absence
bytes; numerical failure may still have a successful restoration. It never
awards numerical or performance eligibility.

Root 15 owner/controller +10 recovery-auditor CPU tests PASS. The two initial
owner-test fixture errors and narrow fixes are retained in
[cold CPU receipts](cold-cpu-v1/receipt.json); [root checks](cold-cpu-v1/root-review.json)
use the actual repository Python runtime. These are preparation results only.
The [cold preregistration](prereg-cold-v1/README.md) now binds the completed D512
cycle's immutable restoration journal and audit. Device execution remains a
separate step after this preparation commit.
