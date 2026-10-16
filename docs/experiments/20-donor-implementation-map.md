# 20：沿实际实现读三条执行路径

这是源码阅读记录，不是已集成的新能力，也不重写第18/19章的数值或性能结果。范围是本机 M4 Max40 / Apple9 / 48GiB、MLX0.32.3、Qwen3.8-27B NVFP4 的普通 C1 decode；本轮仅CPU源码与既有证据阅读。

## 先固定“读的是哪份代码”

| 路线 | 实际版本与证据 | 普通生成调用链 | 当前能下的结论 |
|---|---|---|---|
| oMLX | 安装wheel版本 `0.7.1.dev1`；实际路径前缀 `O=/Users/ryanli/code/minivllm-tutorial/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages`，逐文件SHA见[map](20-donor-source/omlx/source-map.json) | `O/omlx/scheduler.py:14075–14086` → `O/mlx_lm/generate.py:1872–1884` → `GenerationBatch.next:1439–1449` → `_step:1355–1414` | `:1367`调用普通model，`:1405`异步提交下一token，`:1410–1411`取当前结果；已具备ahead。 |
| 手搓引擎 | 被测 accepted15＋gates 文件与固定commit `f976d6a547a9dc1946b52b1de6db376d799b34d7`逐byte核对；当前26文件另含默认OFF实验guard，不能当成全部已启用 | [`ModelRunner.run:54–98`](https://github.com/logitshq/minivllm/blob/f976d6a547a9dc1946b52b1de6db376d799b34d7/tutorial_code/miniengine/engine/model_runner.py#L54-L98) → model/sample → `mx.async_eval` →保留下一拍cache句柄 | 已具备设备token一拍ahead；`:90–94`仍逐token构造context/model/sample，不能再把ahead本身列为新增方向。 |
| Lithos | 固定 `5485ad6bac0a5649d66a347f3e01a655ea29d937`；本机配置 accelerator=on、无draft、steps_per_cb=8、in_flight=3 | [`generate.py:632–645`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/generate.py#L632-L645) → [`Engine:58–101`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/runtime/engine.py#L58-L101) → [`Runner::run:327–375`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/runtime/src/metal_core.mm#L327-L375) | 固定buffer、预绑定dispatch、预编码ICB，设备StepState推进多个token；是运行时差异。 |

Lithos与手搓代码有固定commit permalink；oMLX被测wheel没有可证VCS commit，不能拿当前main或release的target_commitish冒充它的精确源码。oMLX出处按实际安装文件＋行号＋SHA追溯；安装来源见[baseline说明](../../tutorial_code/baselines/README.md)，不补造历史commit。

## 真正尚未具备的近端机制：卷积预处理单核

实际oMLX patch `O/omlx/patches/qwen35_gdn_prework.py:1864–1897`仅允许exact Qwen3.5、B1/T1、GPU FP16/BF16、无mask/lengths/padding/spec的普通ArraysCache、conv4、Dk=Dv128。既有日志确实记过“Qwen B1/T1 fused GDN prework engaged”；证据只覆盖那次C1运行，不外推B2/B4或所有后续launch。

该路线 `:2053–2099` 是分开投影 → fused prework → recurrence → gated norm → out projection → 一起提交两个state。`:108–120`在读旧tail/raw projection时做conv4＋SiLU，`:177–194`同dispatch写原始投影tail；Qwen3.5这支不融合gates，不能把Qwen4另一分支当成实测路径。

我们的[`GDN decode:159–176`](https://github.com/logitshq/minivllm/blob/f976d6a547a9dc1946b52b1de6db376d799b34d7/tutorial_code/miniengine/model/gdn.py#L159-L176)仍显式concat旧tail、BF16乘积/reduction、SiLU、head repeat。此前`v18`后卷积融合实验只处理卷积之后的SiLU/映射，未消除卷积链；其不足1%结果不代表已经测过真正conv预处理单核。这个实验代号与第18章的投影拼接不同。

下一项只准备[未接入CPU原型](20-donor-source/v23/README.md)及输入输出契约：读取不可变旧tail/raw projection/conv权重，输出独立SiLU结果和new_tail；现有四投影、head split/repeat、递推、归一化先保留。12项纯CPU结构检查与独立源码审查通过，未编译或执行GPU；原型没有数值PASS、实际engagement或收益结论。是否把既有FP32 gates放入同核，须另做语义与命中核对。

不能照抄oMLX的FP32卷积累加或BF16 gates。当原引擎product/reduction的实际layout、flags与M/Y分支确认后，才按原BF16乘积/求和拓扑固定实现；也保留MLX typed BF16 SiLU及FP32 gates中间边界。先逐byte中间量与tail，再检查连续递推、logits/IDs/cache和服务生命周期。

## Lithos的ICB是长程运行时工程

[`Engine:58–96`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/runtime/engine.py#L58-L96)预分配/共享weights、state、ring、arena，params不共享；pipeline缓存的键含source/macros/device/编译选项（`:31–43`）。[`ICB:224–251`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/runtime/src/metal_core.mm#L224-L251)一次绑定pipeline/buffer/grid/barrier；参数必须放buffer，offset受32bit限制。

[`barriers.py:38–74`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/compiler/barriers.py#L38-L74)按整个base buffer的读写冲突保留栅栏，第一条命令总等待上一步。设备[`advance:15–36`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/kernels/common/advance.metal#L15-L36)更新pending token/position/step并处理EOS、长度和ring overflow；CPU按每槽64bit序号读ring，不能直接信任尚在写入的head（[`runtime:310–324`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/runtime/src/metal_core.mm#L310-L324)）。

普通streaming每次`run(min(8,need))`，steps_per_cb=8时通常只提交一个CB，且[`run:373–375`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/runtime/src/metal_core.mm#L373-L375)退出前drain。因此in_flight=3在此路径不表示24token队列；长的非streaming run才可能填满多个CB。每CB内多次重放同一ICB仍减少逐token编码工作。

我们的局部`mx.compile`只在norm/gates等子图启用；没有证据说明它等同Lithos式固定buffer/ICB。迁移还需native Metal层、固定arena与state槽、设备采样/StepState/ring、完成栅栏，以及取消、abort、batch变化、cache回收复用协议，不能靠增加一个装饰器宣称完成。

M4走共享Metal运行时并不需要Apple10指令；但其[profile:5–22](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/backends/metal/m4_max/config-40c.json#L5-L22)仍标native fallback/unmeasured，默认accelerator=off。M5的[自动GDN mega fusion:12–18](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/backends/metal/m5_max_40c/scheduling.py#L12-L18)明确要求Apple10、40核、静态T8及测得recipe，不适用于本M4/T1。

NVFP4 [`gemv_nvfp4_rows:100–157`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/kernels/common/gemv_nvfp4_rows.metal#L100-L157)确有SiLU×up、residual/stat epilogue；[`emit:350–411`](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/compiler/emit.py#L350-L411)的输入norm融合由tuner choice决定，未融合时siblings共享norm scratch。当前具体epilogue **实际engagement UNKNOWN**，不能从accelerator=on推断。默认float SiLU与我们typed BF16路径也不能标为无损等价。

## allocator：上限差异已证，速度原因未知

最终`m4-final-native15-official-attempt2`的`client.py:52`启动`child.py`，其`:35`未传cache_limit_gb，实际采用[`LLMEngine:15–18`](https://github.com/logitshq/minivllm/blob/f976d6a547a9dc1946b52b1de6db376d799b34d7/tutorial_code/miniengine/engine/llm_engine.py#L15-L18)默认2GiB。此前旧core_http的显式0参数不适用于这次原始结果。

oMLX该次入口经`client.py:24–35` → `official_entry.py:13–14` → 安装CLI，`O/omlx/cli.py:452–463`在total_mem>0时设allocator cache limit为总RAM。prefix cache关闭与MLX allocator cache是两回事；本次已证策略是2GiB对总RAM上限。

[allocator观察记录](20-donor-source/omlx/allocator-source-observations.json)与[source-map](20-donor-source/omlx/source-map.json)保存launch及文件身份。原raw没有池内实际cache bytes/复用率字段：这些量 **UNKNOWN**，不能当成0，也不能据上限差异解释速度。下一步先记录limit/active/cache，再决定是否做单变量实验；旧raw数字不变。

## 阅读收据

- [oMLX原始只读报告](20-donor-source/omlx/REVIEW.md)及[source-map](20-donor-source/omlx/source-map.json)：实际C1 prework命中、own/v18差别与allocator追溯。
- [Lithos原始只读报告](20-donor-source/lithos/REVIEW.md)及[source-map](20-donor-source/lithos/source-map.json)：Metal replay、buffer/barrier寿命与epilogue未知项。
- [SHA256SUMS](20-donor-source/SHA256SUMS)固定报告、源码映射与原型参考；不复制checkpoint或原始日志整包。原报告按产生时原样保存；卷积规约是否可固定顺序，以oMLX报告中后续layout/M/Y资格要求为准。

本记录不改变26个引擎源码、V19 default OFF或第18/19章结论。新增conv方向仅CPU准备，ICB为架构参照，本轮零新增GPU结果、零新增性能结论。
