# GLM Flash 固定 checkpoint 的加载预算

2026-10-17，CPU 静态审查。没有运行 MLX、加载模型、下载或操作服务。对象是实际安装的 oMLX `0.7.1.dev1` / mlx-vlm `0.7.7` / MLX `0.32.3`，以及 `mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e`。下文 `P/` 指 `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/`。

**目前不能承诺能加载。** 后置 gate/up regroup 已经逐层求值、移除旧投影并清池，没有在 Python 层同时保留整模型的新旧 gate/up。更早的 **113,446 个原始 tensor → 逐 expert stack → 一次整模型 eval** 尚无本机峰值证据；后面的 bounded materialization 不能倒过来限制这一步。256 GiB 减去 checkpoint 文件约189.982 GiB、外部暂停服务此前 RSS 34.64 GiB，只剩粗略31.378 GiB，尚未计 OS、其他进程、临时数组与 driver retention。RSS、文件大小与物理 footprint 不能作为可直接相加的同口径预算。

## 实际生命周期

| 阶段与实际源码 | 保留什么 / 释放什么 | 对峰值的结论 |
|---|---|---|
| `omlx/engine/vlm.py:2045–2181`：pre-load compat，然后 global MLX executor 的 `_load_vlm_sync()` | 普通 GLM 设置没有 `lazy=True`；仅 Qwen4 或 expert offload 分支设置它。固定 affine mixed-bit 预期走普通 `mlx_vlm.utils.load`，首次须证实实际分支。 | 官方普通引擎会先 eager materialize；不能把裸 `load(lazy=True)` 当同一个 baseline。 |
| `vlm.py:1322` 的 `_force_qwen4_exp_sanitize_on_load` 同时覆盖 `glm5_next` | 暂时隐藏目标 shard 的 `format=mlx`，保证 sanitize 在量化模块选择前执行。PLE 资源上下文负责失败清理及所有权转移。 | 不是 streaming loader，也不是内存上限；不能跳过 sanitize 节省内存。 |
| `mlx_vlm/utils.py:941–954` | `_load_safetensors()` 对全部62个 shard 调 `mx.load`，存入一个 `weights` 字典；之后才构建 model。 | 所有原始数组对象被注册。仅看 Python/stub 无法断言全部 payload 此时已驻留；也不能把此处算作确定190 GiB的立即物理分配。 |
| vendor `glm5_next/glm5_next.py:129–166` | `weights`、`remapped`、`lang`/`vis` 是多个 dict，值引用同一数组，并非 Python 层复制 payload。进入 language sanitize 时外层 `remapped` 仍引用旧数组。 | 新旧图都可存活，但本阶段无显式 `eval`；不能由多份dict推导整模型物理翻倍。 |
| `glm_moe_dsa/deepseek_v32.py:844–1010` | 先删除编号≥45的 MTP 层，逐 layer/projection/field 从新字典 `pop` 288个 expert，`mx.stack(to_join)` 写回。没有每层 eval/clear。 | 固定 index 是逐 expert 格式，这一分支必需。42个 backbone MoE 的 stack 图会一起保留到后续求值；被 pop 的数组仍是图依赖。 |
| vendor `glm5_next/language.py:2020–2087` | conv q/k/v 合并、moveaxis，HC/router/gates 转 FP32；最后返回 remapped。 | 外层 sanitize 返回后，临时dict消失，原 expert 输入可只由 stack 图引用。无证据表明后续永久保留整模型的原 tensor 字典。 |
| `mlx_vlm/utils.py:1136–1192` | `nn.quantize` 建正确 packed module；`model.load_weights(list(weights.items()), strict=True)` 将相同数组引用写入模型；随后单次 `mx.eval(model.parameters())`。 | 初始化的 dense/random 数组是图，未发现这里先 eval 再替换的调用，不能声称另有320B BF16常驻副本。真正初始 stack 求值的释放次序、command-buffer/allocator保留量仍须测。 |
| `omlx/utils/model_loading.py:1390` | `materialize_lazy_state` 遍历其他/frozen数组，按 `_MATERIALIZE_EVAL_CHUNK` 分块 eval。 | 解决剩余 lazy state/thread stream 和长 command buffer；**普通 eager load 已发生**，这个分块无法限制前一步整模型 eval。 |
| `omlx/patches/moe_gate_up_fusion.py:96–137` | 每次只合并一层 packed weight/scales/biases；`mx.eval(fused)` 后写入 gate、删除原 gate/up 属性；函数返回后清池。targets列表只保存 SwitchGLU模块，不另存旧projection。 | 该段是逐层布局重排，不会主动把42层旧 gate/up 都保留到最后。当前层仍须同时容纳旧投影和新 fused payload。 |

融合清池用 `omlx/utils/metal_sync.py:56` 的锁、生成stream及default stream synchronize，再 `mx.clear_cache()`。同文件说明 driver 对释放后的 graphics footprint 可延迟回收；因此“一层额外 payload”不是物理峰值的严格上界。不要从其它CPU线程直接清活跃GPU池。

这份 checkpoint 没有 `weight_scale_inv` 键，也没有 router `mlp.gate.scales` 键，故不能把通用 FP8/quantized-router 反量化分支算作当前确定开销。index 有12个 `kv_b_proj.weight`、无对应 scales；其中第45层会删除，11个backbone DSA层做 split/transpose/contiguous，仍有真实临时量，但不走 quantized kv_b 的 dequantize/requantize。`_dequant_mla_proj_mode()` 固定为`"0"`。`_use_glm_moe_fused_gate_up()` 只认 `glm_moe_dsa`，当前 GLM5 Next 不在 sanitize 阶段提前 fuse；后置官方 regroup 才生效。

`Module.load_weights`（`mlx/nn/layers/base.py:123–210`）检查keys/shapes后 `update(tree_unflatten(...))`；它本身没有复制所有 tensor payload。`weights` 本地字典与模型在该次 eval 期间共同引用结果数组，不能因此算成两份结果。

## 当前有证据的数字与仍缺的数字

| 项目 | 数字 / 性质 |
|---|---|
| 固定62 shard文件总长 | 203,992,076,296 bytes =189.9824257 GiB；来自固定LFS manifest。index的 `total_size` 恰与文件总长相同，不能据此认定它已经扣除header。 |
| 固定 index | 113,446 tensor keys，2631个属于第45层。原始 MTP payload 的精确可释放量须读header；普通target仍先注册这些输入。 |
| 每MoE层gate+up新布局 | **估算** `2*288*4096*2048*(4/8+4/64)` =2.53125 GiB，前提scale和bias均BF16。42层总gate+up为106.3125 GiB；逐层融合不应同时额外分配这个总量。 |
| 每MoE层down堆叠payload | **估算** `288*4096*2048*(5/8+4/64)` =1.546875 GiB；gate/up各1.265625 GiB。stack也要考虑旧输入、输出、allocator缓存。 |
| C1 KDA持久状态 | FP32 S约136 MiB，BF16 conv约4.78 MiB；不是主要权重预算，prefill/vision/runtime还有额外量。 |
| 外部服务74384 | root此前实测RSS约34.64 GiB；暂停不会主动释放模型内存。不是本次新测量，也不是固定物理footprint。 |
| 缺项 | 实际tensor dtype/shape字节、MTP/vision占比、初次stack/eval峰值、driver延迟、进程phys/graphics footprint、OS/compressor/swap、第一次请求的编译与activation峰值。 |

新增 `bench/m5_ultra_256gb/glm_flash/runtime_probe.py` 是 **CPU header inventory**，名称不代表已实现或运行模型probe。完整下载的 `complete.json` 出现后才运行：

```sh
python3 bench/m5_ultra_256gb/glm_flash/runtime_probe.py \
  --download-receipt /absolute/path/to/download/complete.json \
  --out /absolute/new/path/glm-header-inventory.json
```

它验证 receipt 固定revision/62 shard SHA记录、当前file size、固定metadata SHA、index与全部header键/shape/dtype/连续offset、288 expert group完整性，汇总MTP/vision/backbone与逐层投影字节并保存header SHA。不读取payload、不重复190 GiB hash、不导入MLX、不下载、不signal。**输出始终 `approval_to_load=false`**：依赖已完成的下载SHA审计，stat/header检查不能证明此后payload未被同长度修改。源码和真实snapshot路径要随receipt冻结。此审查只测CLI及小型合成header，未对尚在下载的模型运行inventory。

## 开关与可保持数学不变的选择

官方现有 gate/up 分层释放已是正确的第一步；无需另外写全模型concat。`moe_gate_up_fusion_enabled` 只有engine内 `getattr` 使用及补丁说明，整个实际包没有对应 `ModelSettings` dataclass字段或公开CLI设置。给对象动态加属性可以改变这个内部分支，但那是显式实验adapter，**不能冒充受支持的官方配置，更不能悄悄改普通baseline**。关闭后也不会解决更早的 expert-stack peak。

`moe_expert_offload_enabled` 是真实设置：它会让load lazy，在materialization前包装experts，且跳过gate/up融合。即便使用相同量化数据，它也改变驻留/执行路径，只能作为单独命名的offload baseline。当前任务不启用，不降precision、不改变权重。

如果普通加载因预算失败，下一候选是 **逐层加载/求值后释放原expert输入**：保持相同288顺序、packed字节、每路径bits/group-size、FP32保留规则和strict keys；每层stack、eval、同步清缓存，再下一层。不是仅在当前sanitize函数循环中加eval：外层 `glm5_next.sanitize` 的旧dict仍引用全部原输入，可能反而更糟。需要明确转移/移除所有上游引用，且保持默认baseline代码不变，先做隔离加载adapter。候选资格逐层比较packed bytes/shape及全部raw IDs；不许通过全体反量化或重新量化省事。这个候选尚未实现、没有峰值或速度结果。

## 最小真实 baseline 资格和停止条件

1. 下载结束并完成全部LFS SHA审计后，先跑CPU inventory。记录当时 `hw.memsize`、各非实验进程phys footprint、可回收/压缩内存、swap与memory-pressure基线；RSS只作辅助。root统一lease管理外部74384，子进程不对它发signal。若预算需释放该服务权重，单纯SIGSTOP不够；是否停服另由root按授权决定。
2. 首次为**有观察器的加载资格**，不产HTTP速度。一个child模型owner，独立state/model root只含固定target symlink，复用 `EnginePool.discover_models` → `await pool.get_engine(... runtime_settings=...)` → `VLMBatchedEngine`，详见 `glm-flash-first-principles.md`。C1、chunked prefill关、prompt/prefix缓存关；MTP/DFlash/expert offload/TurboQuant/SpecPrefill关；不修改thinking或template默认语义，不创建错误的fusion setting。记录实际vendor `__file__`/SHA、custom-loader分支、strict/lazy状态和实际42层融合数量；融合异常在官方路径可能被debug吞掉，资格须单独拒绝混合/缺失布局，不能仅看engine.start成功。
3. 在 shard注册前后、sanitize返回、load_weights前后、首次eval前后、materialize前后、每层fusion前后及engine ready记录 **标量**资源日志：`mx.get_active_memory/get_cache_memory/get_peak_memory`，子进程phys/graphics footprint、系统pressure/compressor/swap。包装调用仅资格观察，不能持有model/old arrays、调用额外eval或把参数numpy化；记录hook source与时间开销。loader线程长eval时，parent采样物理指标；MLX计数只在owner合适的调用边界读取。
4. 用实测已占用物理footprint与保留量预注册child允许预算，建议保留至少16 GiB系统/恢复余量作为**候选操作门，不是硬件峰值定理**。目前不能从31.378 GiB粗余额证明满足该门。`mx.set_memory_limit` 的安装版文档明确它是guideline、可能使用swap，不是硬隔离；不能把它当作OOM保护保证。官方CLI还会把cache limit设为整机内存，而直接EnginePool不会自动复刻这条启动策略；必须冻结实际allocator策略，后续HTTP同策略再测。不要无记录地降cache/改lazy来把失败转成“官方默认成功”。
5. 独立watchdog只管理实验child，预注册持续物理预算越限、系统pressure升高或新增swap即失败/停止；保留峰值与原因。停止不能保证瞬时Metal分配绝不越界，所以**先完成预算证据，再决定是否排GPU**；不能以“有watchdog”为由直接赌加载。发生任一allocation error、GPU timeout、swap、非计划offload、部分fusion或外来owner都不产合格点。root保留cleanup/reap/外部服务恢复receipt。
6. 加载稳定后先一个≤128输入token、cap8的普通greedy文本请求，normal EOS。若资源稳定，再同一固定GLM tokenizer/template prompt运行3个独立request、cap32（允许EOS提前结束）。记录累计raw token IDs、实际prompt IDs、EOS/finish reason、cached_tokens=0、无speculation/offload、每请求状态reset；只证明该范围加载/确定性/路由，不冒充跨实现质量或长上下文资格。后续无观察器、真实HTTP throughput–实际E2E latency须另外配对，首轮加载时间不混入稳态decode点。

本次结论是：**原有官方gate/up regroup无需为内存重新实现；先用完整header和实际物理预算收窄初始stack/eval风险。当前没有加载成功、峰值安全或性能结论。**
