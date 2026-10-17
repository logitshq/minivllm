# GLM-5.3-Flash：最小就绪状态与下一步

2026-10-17，CPU/source-only。只读保存的 config、manifest、header inventory 和当前支持源码；未打开 shard payload、导入 MLX、联网、重做大 hash、运行测试或访问服务/GPU。

**当前结论：下载完整；mini engine 与本机固定版 Lithos 不支持该架构；oMLX 的 vendored mlx-vlm 有真实加载路径，但这个190 GiB checkpoint 尚无本机加载成功、峰值或短请求资格。** 下载完成不等于能跑，更不等于有 GLM 吞吐—延迟点。

## 固定身份与支持矩阵

固定模型为 `mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e`，实际是 group64 mixed **4/5/6-bit affine**，并非 Qwen NVFP4。已保存 [config](../m5_ultra_256gb/glm_flash/metadata-v1/config.json) 指定 `Glm5NextForConditionalGeneration` / `glm5_next`：45个 backbone 层（34 KDA + 11 DSA）、前3层 dense、42层 MoE、288专家选8、4路 mHC；另外有第45层 MTP 权重和视觉塔。

| 路线 | 当前是否可用 | 明确源码依据 / 最小处理 |
| --- | --- | --- |
| 自己的 mini engine | **不支持，不能直接传 model_dir** | `tutorial_code/miniengine/config.py:43–50`直接读取 Qwen `linear_num_*` 字段，本配置没有。`model/qwen3_5.py:35–42`只把 `full_attention` 识别为 attention，会把 GLM `deepseek_sparse_attention` 错分到 GDN；MLP是dense普通SwiGLU，没有MoE/clamp/mHC。现cache也没有DSA pooling生命周期。改配置名不是适配。 |
| 当前 Lithos | **不支持，官方当前入口明确拒绝** | `/Users/ryanli/code/lithos-metal/monolith/models/__init__.py`只注册 Llama、Qwen3、Qwen3-MoE、Qwen3.5、Qwen3.5-MoE；无 `Glm5NextForConditionalGeneration`。`serving/setup.py:83–86`与`generate.py:748–751`在未注册架构直接抛错。此结论针对本机固定源码，不宣称未来版本永不支持；不能改chip配置、`--no-draft`或生成一个Qwen pack绕过。 |
| oMLX实验wheel | **有实现与加载路径，实机未验证** | `.artifacts/omlx-baseline/.venv`：oMLX0.7.1.dev1、MLX0.32.3、mlx-vlm0.7.7。官方 EnginePool → `VLMBatchedEngine.start` → pre-load compat → vendor GLM → strict VLM load/materialize →逐层gate/up regroup →官方Scheduler。首先走这条target-only路线。 |
| 裸 mlx-vlm / mlx-lm | **不能当同一官方 baseline** | 模型卡明确 mlx-lm 无 `glm5_next`支持；upstream mlx-vlm有该模型，但oMLX会替换为vendor并修复sanitize/cache/message format。直接 `mlx_vlm.load`没有证明与官方服务相同；不以裸load成功冒充oMLX成功。 |

oMLX关键源码位于其实际 site-packages 下：

- `omlx/utils/model_loading.py:846`调用 `apply_mlx_vlm_glm5_next_compat_patch`；compat `__init__.py:29–73`把vendor目录放入模块搜索路径并检查实际 `language.__file__`，同时补 GLM message format。
- `omlx/engine/vlm.py:2045–2181`在官方MLX executor加载；`_force_qwen4_exp_sanitize_on_load:1322`也处理GLM，不能因函数名误删。普通加载不默认lazy；expert offload才改变此路径。
- `mlx_vlm/utils.py:1180–1192` strict `load_weights`后对整个参数树eval。随后 `omlx/patches/moe_gate_up_fusion.py:96–137`每层合并、eval、释放原gate/up并清池。已有这个分层释放，不需再写一份。
- vendor `glm5_next/language.py:2108–2119`：KDA用`ArraysCache(2)`；DSA用`CacheList(KVCache, PoolingCache)`，不是Qwen的二选一cache。

需冻结真实vendor及依赖的`__file__`/SHA、实际引擎类/model_type、配置/量化/融合数和fallback。仅记wheel版本或upstream同名模块不足。已有 `dflash_glm5.load_glm5_target_bundle`是投机target adapter，不作为首个普通官方baseline。

## 已知字节与256 GiB边界

证据来自已关闭的本地回执：

- `.artifacts/m5-ultra-256gb/glm-flash-download-v1/complete.json`：`PASS_PINNED_DOWNLOAD_AND_SHA_ONLY`，62 shards此前逐SHA通过；回执 SHA `f42666fd0b18adb07a25c3e07c4e32f5bb374529059cf38520e52618311f8363`。
- 同目录 `header-inventory.json`：`PASS_HEADER_INVENTORY_ONLY`，明确`runtime_load_executed=false`、`peak_memory_measured=false`、`approval_to_load=false`；回执 SHA `6bcfc482687d61746d7e6fac5768c4bba27b8d006631dace17992950637497f3`。本轮只读这些小回执，未复核payload当前内容。

| 项目 | 已保存的精确逻辑字节 / 解释 |
| --- | --- |
| shard文件 | 203,992,076,296 B =189.982426 GiB |
| tensor payload | 203,976,457,080 B，113,446 tensors；header与padding不算payload |
| 45层target sanitize移除的MTP层 | 4,573,936,256 B =4.259810 GiB；初始加载仍会注册这些输入，不能视作加载前就不存在 |
| backbone routed experts | 183,911,841,792 B =171.28125 GiB |
| 官方VLM保留视觉部分 | 1,127,254,016 B =1.049837 GiB；首轮不擅自删模块变成另一个baseline |
| 仅扣MTP后的checkpoint payload | 199,402,520,824 B =185.708069 GiB；不是实际模型active/physical footprint |
| 单层gate/up新布局 | 2,717,908,992 B =2.53125 GiB；单projection stack最大1,660,944,384 B =1.546875 GiB |

`256−189.982426≈66.017574 GiB`只是文件大小减法，**不是可用工作区或安全余量**。sanitize会把113,446个原tensor按专家stack；原输入、lazy依赖与输出可能在首次整模型eval同时存活，峰值释放次序、allocator池和driver retention均未测。后置逐层gate/up融合无法限制更早的整模型eval峰值。普通load和官方CLI的allocator策略也须记录，不能偷偷改lazy、cache或precision后继续称原baseline。

配置/源码给出的持久cache数量级：C1 KDA FP32 S为 `34×64×128×128×4=142,606,336 B`（136 MiB），BF16 conv约4.78125 MiB。DSA已有单份512维latent与零宽V；若latent为BF16，11层KV逻辑项为 `11×T×512×2`，T2048仅22 MiB，但**不包括PoolingCache/indexer/top-k、prefill激活、padding/扩容和临时workspace**。C并发乘数与长期pool状态必须实际观察；配置的1,048,576 context不是256 GiB可运行承诺。

过去常驻oMLX的实测physical footprint曾约142.82 GiB；仅SIGSTOP不释放它。该数字是历史记录，本轮没有查询当前进程。下一GLM排队必须先按现有冷循环真正释放驻留模型，再测当时系统余量；不能拿旧RSS约34 GiB作预算，也不能预设当前恢复PID或余量。

## 最少实机前置，按失败即停顺序

| 阶段 | 最小动作 | 必须得到的证据 / 才能进入下一步 |
| --- | --- | --- |
| 1. 冷加载准入 | 复用已审cold owner的资源/独占/直属child/reap/recover机制；隔离目录只含固定snapshot symlink，C1/context2048；MTP、DFlash、SpecPrefill、TurboQuant和expert offload OFF | 最新已审RESTORED前序、释放后实际系统physical/free/pressure/swap、固定runtime/model/config/template身份；先注册预算及停止条件。现有1830s/资源阈值不能自动证明190 GiB加载安全；必要的不同预算须明确登记，而不是放宽失败后重命名成功。 |
| 2. 官方普通加载 | 单child复用E015 EnginePool API：`discover_models` → `await get_engine(...runtime_settings=...)` → `VLMBatchedEngine`，最后`await pool.shutdown()` | 真实vendor、loader分支、strict/sanitize、45层backbone/42层融合、无offload或静默fallback；在load/sanitize/eval/materialize/fusion边界仅记录标量MLX allocator及外部physical/swap。异常/pressure/新增swap依预注册停止；保存所有partial。加载通过不计性能点。 |
| 3. 最短请求资格 | GLM自己tokenizer/template冻结≤128-token文本，先cap8，再3个独立normal-EOS cap32请求，greedy、cache0 | 实际完整输入IDs、raw/committed IDs、EOS `[154820,154827,154829]`及实际stop set、finish/usage/text、request reset、KDA/DSA/pooling shapes/offset。不能借Qwen prompt IDs、EOS、64层状态计数或DFlash25600宽度。重复一致只证明本范围确定性/路由，不证明新实现逐状态等价。 |
| 4. 长负载与正式计时 | 短资格/资源稳定后才做GLM版本的D8原messages→冻结完整输入、cap512；先C1，再有资源证据的C8。另测DSA pool4与2048阈值边界，避免短请求代表长上下文 | 完整normal-EOS/count、实际batch/queue、每请求cache生命周期与内存边界。之后无observer warm8/formal64，x为POST→DONE p50/p95，y为committed tokens/共同窗口。GLM独立图，不把不同模型/模板token数混进Qwen前沿。 |

可复用的小模块是现有 HTTP原wire客户端、采样/计数审计形式、资源/回收控制和metadata/header工具。模型数学只能先借已存在的官方GLM reference；若后续自研，逐个拆出mixed-bit expert、KDA、DSA+pool、mHC，并保留各自逐位/state资格。当前无需设计通用架构注册或同时重写全部模块；首个交付应是**真实官方冷加载与短请求是否能通过的明确答案**。

更完整旧审查见 [加载预算](glm-flash-load-budget.md) 与 [执行路径/模块边界](glm-flash-first-principles.md)。其中“下载尚未结束”和旧RSS估计是历史状态；本页以已完成download/header回执更新准备状态，但不更新成尚不存在的模型资格。
