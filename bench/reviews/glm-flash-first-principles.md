# GLM-5.3-Flash：本机执行路径与最小实现边界

2026-10-17，CPU 静态审查。没有导入 MLX、加载模型、下载权重或运行 GPU；以下性能判断均为待测假设。本机参考环境是 `.artifacts/omlx-baseline/.venv`：oMLX 0.7.1.dev1、MLX/MLX-Metal 0.32.3、mlx-vlm 0.7.7。模型固定为 `mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e`，metadata 在 `bench/m5_ultra_256gb/glm_flash/metadata-v1/`。

结论：先建立真实 oMLX 普通解码 baseline，再从 **mixed-bit routed MoE 的 5-bit down projection** 开始独立测量。当前 oMLX 已有 C1 MoE、KDA 和 mHC 的大量融合；重新写一份朴素 GLM 并不自动更快。第一项候选只替换 expert 数学模块，不同时改 router、attention、缓存或调度。

## 实际加载路径：必须区分 upstream 和 vendor

以下 `P/` 均指 `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/`。

1. `omlx.engine_pool.EnginePool.discover_models()` → `await pool.get_engine(model_id, runtime_settings=settings)` → `omlx.engine.vlm.VLMBatchedEngine.start()`（`P/omlx/engine/vlm.py:2045`）。普通 baseline 要断言返回的是 `VLMBatchedEngine`，不接受变成 LLM 或 speculative engine 的静默回退。
2. `omlx.utils.model_loading.maybe_apply_pre_load_patches(..., for_vlm=True)` 在 `model_loading.py:846` 调用 `apply_mlx_vlm_glm5_next_compat_patch()`。该补丁把 vendor 路径插入 `mlx_vlm.models.__path__` 首位，删除已导入的 upstream GLM 模块并重新导入；随后验证 `language.__file__` 确实位于 vendor。它还安装 PoolingCache 支持和 GLM 消息格式。这意味着同样的 `mlx_vlm.models.glm5_next.language` 模块名不能证明加载了同一源码。
3. `VLMBatchedEngine.start()` 在全局 MLX executor 上执行加载。`_force_qwen4_exp_sanitize_on_load()` 名字虽然含 Qwen4，实际也覆盖 `glm5_next`（`vlm.py:1322`）：对本模型 shard 暂时隐藏 `format=mlx`，让 sanitize 在选择量化 module 之前执行。上下文退出即恢复。不能直接删除这个步骤。
4. 先尝试 `maybe_load_custom_quantization(path, is_vlm=True)`；当前 metadata 是普通 affine mixed-bit 描述，预期走 `mlx_vlm.utils.load(path, lazy=False, strict=True, trust_remote_code=False)` 的普通分支（`utils.py:1274`）。首次运行必须记录实际分支，而不是只记预期。
5. sanitize 保留 HC/router/gate 参数的 FP32，重命名权重、合并 Q/K/V conv，并通过 `DSV32Model.sanitize` 去掉编号 ≥45 的 next-token 层（`deepseek_v32.py:844`）。当前 checkpoint 有第45层 MTP 权重；普通解码用45层 backbone，不应把它误算成46层主模型。
6. 加载后在同一 executor 上 `materialize_lazy_state(model)`（`vlm.py:2272`），再做 gate/up lossless regroup（`vlm.py:2306`、`moe_gate_up_fusion.py:94`）。该转换每层 eval 并清 allocator cache，避免积累整模型副本。源码通过 `getattr(settings, "moe_gate_up_fusion_enabled", True)` 读取此项，但当前 `ModelSettings` 没有同名 dataclass 字段：**不要把它写成未经验证的 constructor keyword**。资格时记录实际 `gate_up_proj` 布局/转换层数。
7. `VLMModelAdapter` 交给官方 `AsyncEngineCore`/Scheduler；`make_cache()` 委托 GLM language model，不能替换成 Qwen cache（`P/omlx/models/vlm.py:258`）。

另一个明确可定位的加载 API 是 `omlx.patches.dflash_glm5.load_glm5_target_bundle(model_ref, lazy=False, quantize_kv_cache=False, verify_config=None, trust_remote_code=False)`（`dflash_glm5.py:706`）。它也使用 vendor/pre-sanitize/strict VLM load，并安装 speculative hooks。**普通 baseline 不走它**；它适合后续明确的 DFlash 资格，不代表官方 HTTP server。源码明确拒绝 GLM target KV quantization。

## 可运行 baseline 资格所需 API

复用 E015 的隔离 EnginePool child 结构，而非复用其 Qwen token IDs 或32-token length断言。最小主干如下；这是待唯一 GPU owner 执行的 API 示意，本审查未执行。

```python
from pathlib import Path
from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig

# model_root 是新建的隔离目录，仅包含一个指向完整固定 snapshot 的 symlink。
settings = ModelSettings(
    mtp_enabled=False, vlm_mtp_enabled=False, dflash_enabled=False,
    moe_expert_offload_enabled=False, turboquant_kv_enabled=False,
    specprefill_enabled=False, max_context_window=2048,
)
config = SchedulerConfig(
    max_num_seqs=1, completion_batch_size=1, chunked_prefill=False,
    prefill_speed_priority=True, decode_fairness=True,
    paged_ssd_cache_dir=None, hot_cache_max_size=0,
)
pool = EnginePool(scheduler_config=config)
try:
    pool.discover_models(str(model_root))
    engine = await pool.get_engine(model_id, runtime_settings=settings)
    assert type(engine).__name__ == "VLMBatchedEngine"
    assert engine.model_type == "glm5_next"
    assert engine.vlm_mtp_drafter is None
    result = await engine.generate(
        prompt=list(glm_prompt_ids), max_tokens=32,
        temperature=0, repetition_penalty=1.0,
    )
finally:
    await pool.shutdown()
```

`model_root`、`model_id`、`glm_prompt_ids` 由独立资格 harness 提供；`await` 位于 async 函数中。同一 prompt 文本要用 GLM 的真实 processor/tokenizer/template 生成并冻结 ID，记录 thinking 模式和 `add_generation_prompt`，不要复用 Qwen 的19个ID。保留 normal EOS：metadata 的 EOS 是 `[154820,154827,154829]`，同时记录 scheduler 的实际 stop set；输出可正常提前结束，不能要求一定32个token。

E015 的 `Scheduler._process_batch_responses` 观察 seam 可以收最终 `RequestOutput.output_token_ids`，这是累计 ID；不能把多次部分输出相加。3个请求必须是独立 request/cache，记录 raw IDs、实际 prompt IDs、finish reason、cached tokens=0、异常及每层 cache shape/offset。重复一致只是确定性证据；自研候选还须逐请求对这份官方 raw ID oracle 与状态比较。观察器仅用于资格，正式 throughput–E2E-latency 配对必须移除。

加载后冻结实际 `__file__` 和 SHA，至少覆盖 compat `__init__`、vendor `language.py`/`linear.py`/`gated_delta.py`/`hc_prefill.py`、`decode_kernels.py`、`glm53_kda_*`、`deepseek_v4/switch_layers.py`、`m5_gather_qmm*.py`、`moe_gate_up_fusion.py`、GLM DSA kernels、实际 native extension 与包版本。只记录 wheel 版本或 upstream 文件 hash 不够。设备资格还应记录 `_DECODE_FUSION`、实际 fused/fallback 次数、NAX canary 结果及是否发生 expert offload。

190 GiB 是磁盘权重大小，不能当作加载峰值。主线程观察到被暂停服务仍有约34.64 GiB RSS，SIGSTOP不释放这些页面；两者相加仅是粗预算，不是物理内存精确加法。loader sanitize、gate/up regroup、vision及allocator均有额外峰值，必须在加载前后实测。先C1短文本，不依据1M配置直接预分配长上下文；一旦发生offload，应另标baseline，不能当全驻留结果。

## 模块划分：固定一个 checkpoint，只保留必要语义

| 独立模块 | 最小契约 | 必须保留的内容 |
|---|---|---|
| `weights` | path → 明确的 per-projection packed weight/scales/biases/bits/group-size | group64，默认4-bit、down5-bit、shared6-bit；router/HC/gates保留FP32；严格 key/shape 检查，处理原MTP层与视觉权重 |
| `moe` | `(x, route_ids, route_weights, packed_experts) → y` | 288选8、FP32 sigmoid/correction/top-k/归一化/2.5缩放；先固定router输出，只替换expert；clamped SwiGLU limit10；shared expert与加和次序 |
| `kda` | `(x, conv_state, S, offset) → (y, new_states)` | 34层，64 heads×128，conv4，向量forget gate，BF16/FP32舍入位置、mask与每请求状态；全prefix回滚另设接口 |
| `dsa` | `(x, latent_kv, pool_state, positions) → (y, new_states)` | 11层，NoPE MLA latent512、64 heads；index32×128、pool4、top2048+尾部；选中索引/顺序、pool remainder与KV offset同时提交 |
| `mhc` | `(four_stream_hidden, branch) → four_stream_hidden` | residual `[B,T,4,4096]`，FP32 mix/Sinkhorn20轮；attention/FFN各一套，最后沿HC轴mean再norm |
| `runner` | 固定文本输入、显式request states、每步一个提交点 | 先C1和普通greedy；生命周期、停止、取消、cleanup先通；任何开关不隐式修改其他模块 |

首版只需把这些模块的 reference 实现与候选函数通过显式 config 选择；不用建立插件注册、自动选择模型、量化重写或跨设备offload系统。默认全reference，单项候选默认关闭。共享现有采样/HTTP/审计逻辑可以，但模型和cache契约必须独立。

## 首个大方向：同一 expert 的 packed 权重跨行复用，先补5-bit down缺口

**源码事实：**45层中前3层dense，后42层MoE。每个routed expert有两个 `4096→2048` 和一个 `2048→4096` 投影。当前量化描述给routed gate/up 4-bit、down 5-bit，group64；shared三个投影6-bit。只计算这42层每token选8个expert的packed权重，在scale/bias均BF16且每组各一个的前提下，理论权重字节为

`42 * 8 * 4096 * 2048 * (2 * (4/8 + 4/64) + (5/8 + 4/64)) = 5,108,662,272 bytes ≈ 4.758 GiB/token`。

这是布局推算，未测内存流量；不含shared/attention/embedding/HC，不代表一次token必从DRAM重读这些字节。精确scale/bias dtype、实际layout和cache命中需加载后确认。

**现有快路已做什么：**vendor `language.py:1641` 的 `_decode_select` 在C1把router选择并进gate/up kernel，然后down/combine；`_decode_experts` 覆盖B=1短块；`decode_kernels.py:774` 的SIMD qmv支持3/4/5/6/8-bit。不能声称C1的5-bit没有快路。

**具体缺口：**`P/omlx/patches/m5_gather_qmm_nax.py:1322` 的 `supports()` 对affine只接受4/8-bit；该oMLX segmented NAX快路不覆盖5-bit down。`deepseek_v4/switch_layers.py:45` 对本格式在routes≥64才排序；B>1不能进入GLM专用B=1融合。现有general路径有sort/map/fusion，不能假设每条route都在主存重读全部权重。这里只确认分支缺口，尚未确认端到端瓶颈，也未证明所有底层MLX 5-bit路径都不使用tensor units。

**最小候选：**独立 `expert_down5(x_routes, ids, W5, scales, biases)`，固定E288/N4096/K2048/group64，消费已有router结果和row map；仅在同expert有足够多行时复用packed weight tile，原route顺序还原后按原次序加权。第一版只改变down，保持gate/up/shared/router原实现；不得先反量化整个190 GiB模型、降低bit数或改INT8 activation。是否使用NAX取决于严格数值门：tensor-unit的归约顺序不能凭代数等价宣称逐字节一致；无法保持当前oracle字节则本候选失败，不能放宽门后仍声称同质量。

最小实验先从真实第3/中间/第44层捕获post-SwiGLU、route IDs/权重、5-bit down和scale/bias，记录重复expert占用率。覆盖C1/2/4/8/16普通decode，以及T8/64/256 prefill或verify形状；刻意加入无重复、全部同expert、route64阈值两侧和ragged尾块。冻结输入/output逐字节，先比较每route输出、原顺序加权结果和shared加和；然后独立fresh进程ABBA微计时，预注册长点GM≥1.25、其他每点≥.97，并同时保留路由统计与资源峰值。若真实decode路由几乎不重合，拒绝“权重复用能改善低并发”的假设，而不是只报prefill收益。最后才能做完整层、整模型raw IDs/state与无观察器HTTP throughput–E2E-latency资格。

## KDA / DSA / mHC 为什么暂不一起改

vendor `language.py:553` 已把T≤8的conv、SiLU、L2、gate、KDA recurrence、norm-gate融合；`KdaStepCapture`保存entry state并重放任意prefix（`:353`）。KDA的forget gate是每head每channel128维，不能拿Qwen GDN的标量head decay直接代换。34层FP32 recurrent state约136 MiB/请求，conv另约4.78 MiB/请求；decode读写S可能重要，但需要对比上述MoE成本后再排序。

DSA短块已走latent-space attention、短prefix绕过无意义top-k、长段走indexer/NAX；NoPE cache只存一份512维latent与零宽value（`:1267`），不能用“删重复K/V”重复立项。N≤2048不会充分覆盖稀疏选择，后续必须加入2047/2048/2049、pool4边界及≥4096点，核top-k tie/尾部顺序。mHC已有 `_HCDeferred` 跨半层消除expand独立dispatch，并每8层async_eval减轻Python构图停顿（`:166`、`:1866`）；再做whole-step编译会影响状态和调度，不应与第一项MoE实验混在一起。

## Qwen engine 不能直接复用的假设

- `tutorial_code/miniengine/config.py`要求Qwen的 `linear_num_key_heads` 等字段；GLM是 `linear_attn_config`，`head_dim=0`不能按普通attention解释。当前decoder只识别 `full_attention`，会把 `deepseek_sparse_attention` 错归到GDN。
- `model/qwen3_5.py`所有层是dense普通SwiGLU、两次 `x + branch`；GLM后42层是MoE+clamp，并有4路HC。最终mean和capture语义也不同。
- `model/loader.py`对已转换Qwen RMSNorm减1、统一转换大部分浮点权重、忽略vision/MTP；这些不能移植为GLM规则。GLM HC/router必须保FP32，普通RMSNorm不套Qwen zero-centered修正，per-path mixed quant也不能简化为全局4-bit。
- `cache.py`只分 `AttnCache` 与 `GDNCache`；GLM DSA必须同时维护latent KV和PoolingCache，KDA含不同forget参数与replay输入。只回滚KV offset会留下pool尾部/索引状态错误；C1→batch→C1与取消都要重新资格。
- 现有DFlash feature宽度25600和Qwen capture层列表不能复用。GLM target adapter按HC均值收缩hidden，且注明不支持prefix snapshot codec（`dflash_glm5.py:269`）。存在adapter不等于我们已有匹配GLM drafter或whole-verify精确资格。

官方模型身份与大小见 [GLM模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)、[固定MLX导出](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit/tree/fa04cbc3622b16aa7475404060b05e506d81bd4e)。本次仅依据已固定metadata和上述本机源码审读，没有把源码注释中的性能数字当作本机实测结果。
