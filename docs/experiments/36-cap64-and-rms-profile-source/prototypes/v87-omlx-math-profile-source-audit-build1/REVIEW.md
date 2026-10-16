# V87：旧字节级 gold 与实际 oMLX Q35 是不同数值 profile，尚不能归因速度差距

`S=/Users/ryanli/code/minivllm-tutorial/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages`；`W=/private/tmp/minivllm-engine-optimization-20261014/candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine`。精确路径/旧环境身份SHA见 `source-map.json`；官方GPU源码仅保存tag/excerpt，不伪称wheel源码或实际PSO证明。

**先纠正实际路径。** 历史服务 `m4-final-native15-official-attempt2/output/C1-block1-omlx/server.log:8,31,36,48,58,60` 确认 VLM、MTP sanitize patch、Q35 q/k normalization patch及B1/T1 fused prework engaged。不是仅凭函数存在推断；Qwen4 L2/整步融合代码不属此路径。实际state叶子是 `S/mlx_vlm/models/qwen3_5/gated_delta.py:298–355`：T=1禁用packed分支，generic grid=(32,128,48)、TG=(32,4,1)由源码选择；没有保存该官方运行的PSO反射/dispatch记录，故几何属于source-selected而非GPU实测。

| 运算 | candidate44 / 旧 gold | 实际安装 Q35 路径与可能影响 |
|---|---|---|
| gamma加载 | `W/model/loader.py:37–44`：转换权重FP32减1后存BF16，norm再加1转BF16 | checkpoint48个conv都是[10240,4,1]；已启用sanitize `S/omlx/patches/mlx_vlm_mtp/qwen35_vlm_model.py:52–60,100–110` 不再次+1。**CPU逐161数组/668672元素回环差异0，最大偏差0**，原gamma范围[-0.1484375,3.21875]；48个GDN plain norm不减/加1。不能拿假设的gamma损失解释本checkpoint。|
| 普通/attention RMS | `W/kernels/norm_metal.py:20–46,58,62`：显式MLX-row tree、sum×FP32(1/D)、precise rsqrt、FP32 normalized×gamma后一次BF16；contract(off) | `S/mlx/nn/layers/normalization.py:163–164` 调用fast RMS；**官方同版本Metal源码**显示normalized先BF16、再乘gamma，且同样用precise rsqrt；不是只由CPU fallback推断。但wheel不附该实现，实际PSO/舍入尚未捕获，不能宣称已GPU证实。D256线程数来源不同：ours32/native64；D5120 native用PSO最大threads，当前官方实际max未记录。[MLX fast API](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/fast.cpp#L75)、[native RMS](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/rms_norm.metal#L24)、[launch](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/normalization.cpp#L47)。|
| GDN norm-gate | `W/model/layers.py:18–21`及norm_metal:46–53：weighted先BF16，SiLU(FP32 gate)也先BF16，再乘 | `S/mlx_vlm/models/qwen3_5/language.py:48–67`：fast RMS先落BF16；gate SiLU及其与norm结果乘法保持FP32，末尾BF16。两处舍入规则不同，会影响旧严格gold。|
| conv / qk prework | `W/.../gdn_activated_norm_reuse_cached_metal.py:62–77`：已激活BF16→FP32，单lane串行128项平方和，FP32保留q/k及scale；旧conv保留BF16乘法/加法边界 | `S/omlx/patches/qwen35_gdn_prework.py:105–128,155–168,780–801` 的 **L2=false**：FP32四tap累加→BF16 conv/SiLU；每lane四项+simd_sum，eps=1e-6/128，RMS及scale分别落BF16；grid(32,1,80)/TG32。数学等价的L2/RMS实数式不保证这些中间位相同。|
| gates / recurrence | ours a/b/dt_bias和门函数FP32；state[Dk,Dv]，四段serial32再ascending4，先dot(S,k)再乘exp(g)，o FP32后cast | `S/mlx_vlm/.../gated_delta.py:68–101,407–414`：a/dt_bias/b的原BF16链，g最终FP32、beta保留输入dtype；state[Dv,Dk]，每lane4项+simd_sum；先S×g再dot，再update/readout；y直接InT(BF16)。分配/归约/乘法位置均不同。不能将mlx-lm的packed默认误搬到实际VLM T1。|

**gamma独立实证范围。** `gamma_audit.py` 只用stdlib逐个读取detached小范围，共1,349,632B（含48个plain gated norm），不读整shard/不load模型。精确模拟FP32减1→BF16 RNE→FP32加1→BF16 RNE；每数组的header、absolute byte range、raw/effective SHA、min/max、差异数和最大偏差均保存于 `norm-gamma-checkpoint-roundtrip.json`。所有161数组有效gamma的SHA与原始SHA一致。这是CPU按源码顺序的位模型，不是GPU参数回读；不授权改变loader或旧gold。

**能建立独立 profile，但不能称已找到瓶颈。** 旧规范明确匹配HF式FP32显式表达式在MLX中的运算顺序，精确tree是实现选择，并非HF要求某种硬件tree。官方Q35的算子调用规范指向另一BF16规约；预期差异不等于精度错误，仍需实际算子数值对照。ours norm强制contract(off)，vendor对应源码没有这一pragma；是否实际FMA/fast-math及其收益未捕获，不能臆断。仅源码不能证明规约是34.6ms主因；当前自己的norm本来也已单kernel，替换fast RMS并不自动省launch。

**最小可验证实验建议（本轮不实现/运行）：** 先独立冻结 `OMLX_Q35_MLX0323` 算子参照，从实际安装的fast RMS、`_precise_swiglu`及L2=false prework定义输出；只做同一真实X/原gamma的普通D5120、qk D256、gated D128三类小片对照，报告旧/新差异、finite/readonly，禁止用FP32权重冒充donor BF16。先隔离norm，避免把conv/gates/state一起改后无法归因。候选对新donor参照严格比较，旧gold不匹配仍原样记录；进入完整模型前另预声明logits/cache误差及全32 greedy ID门槛，再做完整路径验证，不能失败后放宽。GDN另需FP32 state逻辑transpose、BF16 q/k/beta和完整gate链，不能只换SIMD树就称已实现donor profile。任何速度结论仍需独立端到端/服务配对，旧FAIL/STOP不改标。
