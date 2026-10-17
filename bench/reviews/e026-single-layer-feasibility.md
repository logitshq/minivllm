# E026 单个真实 GDN 层：CPU 可行性与独立观察预算

2026-10-17。只做源码、JSON 和 safetensors 文件头检查；未导入 MLX、未读权重 payload、未执行 GPU、未实现新 harness。这里建议的是**真实权重 + 合成有限 BF16 输入的单层数值诊断**，不称为真实 D prompt、整模型资格或性能实验。它不替代 E023 原 37 请求的完整同调度资格。

单层加载在结构上可行：固定目标的第 0 层是 `linear_attention`，其 GDN 只有 14 个权重 tensor，原始 payload 合计 **65,239,488 B = 62.217224 MiB**。按原 loader 将 `A_log` 和 `dt_bias` 从 BF16 转 FP32 后，逻辑参数为 **65,239,680 B = 62.217407 MiB**。1 GiB 可作为一个新的、需要实际验证的**观察预算**；源码不足以证明 allocator、临时图和 command buffer 的真实峰值必低于它。

## 输入证据与读取边界

模型目录固定为 `/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`。本次读取并重算以下小文件/文件头 SHA256：

| 证据 | SHA256 |
| --- | --- |
| `config.json` | `7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706` |
| `model.safetensors.index.json` | `8a3f70b51e57843efcb417829dc8be081e414517995234e71fc7cf508ae01f90` |
| `model-00001-of-00003.safetensors` 的 JSON header，文件 bytes `[8,86458)` | `5ee4fe36fedff37be5707fae8d414da5f4fbf11f23c74dd0c3a97d7384f497e7` |
| 同 shard 的长度前缀 + header，文件 bytes `[0,86458)` | `00f14d525d9db3745509ed68f8b824aa8bca9e8a670fb4b2c4e2712f584b87c3` |
| 已冻结 [E023 model manifest](../../.artifacts/m5-ultra-256gb/realistic-qualification-v1-inputs/model-manifest.json) | `de8ae0d26b5db015b9dc690bcaaec86cbbd5f1435c439baca22c589c7441fa1d` |

header 长度为 86,450 B；文件 stat 大小为 5,343,250,458 B。冻结 manifest 记录该完整 shard 的 SHA256 为 `a9de796bfb40fbe757ff3e567d83d5d100906d6dd231c8b11c431cd97d3dc288`，**本次没有重新读取/验证完整 shard**。未来真实执行必须独立验证所使用的权重身份；header 匹配本身不能证明 payload 未变。

配置确认：hidden 5120；Hk/Hv=16/48；Dk/Dv=128/128；卷积 K=4；64 层中每第 4 层为 full attention，因此第 0 层是 GDN。原量化 recipe 为 `group_size=16,bits=4,mode=nvfp4`，不改变。

## 固定 14-tensor allowlist

所有键共同前缀为 `language_model.model.layers.0.linear_attn.`，全部来自上述第一 shard。表中 offset 是 header `data_offsets`，相对 payload 起点 86,458；end 不包含在内。

| 键后缀 | 磁盘 dtype / shape | bytes | data offsets [begin,end) |
| --- | --- | ---: | --- |
| `A_log` | BF16 `[48]` | 96 | `[3164064448,3164064544)` |
| `conv1d.weight` | BF16 `[10240,4,1]` | 81,920 | `[2268569984,2268651904)` |
| `dt_bias` | BF16 `[48]` | 96 | `[2216994336,2216994432)` |
| `in_proj_a.scales` | U8 `[48,320]` | 15,360 | `[3665864576,3665879936)` |
| `in_proj_a.weight` | U32 `[48,640]` | 122,880 | `[2216994432,2217117312)` |
| `in_proj_b.scales` | U8 `[48,320]` | 15,360 | `[2032936640,2032952000)` |
| `in_proj_b.weight` | U32 `[48,640]` | 122,880 | `[5276869472,5276992352)` |
| `in_proj_qkv.scales` | U8 `[10240,320]` | 3,276,800 | `[5329961568,5333238368)` |
| `in_proj_qkv.weight` | U32 `[10240,640]` | 26,214,400 | `[2242355584,2268569984)` |
| `in_proj_z.scales` | U8 `[6144,320]` | 1,966,080 | `[2217117312,2219083392)` |
| `in_proj_z.weight` | U32 `[6144,640]` | 15,728,640 | `[2219083392,2234812032)` |
| `norm.weight` | BF16 `[128]` | 256 | `[2211423520,2211423776)` |
| `out_proj.scales` | U8 `[5120,384]` | 1,966,080 | `[2193728800,2195694880)` |
| `out_proj.weight` | U32 `[5120,768]` | 15,728,640 | `[2195694880,2211423520)` |

最小新 loader 只接受这 14 个已固定键及对应 dtype/shape/offset/bytes；检查区间长度与文件边界，用 `seek(86458+begin)` + 精确长度读取，只构造选中数组。U32/U8 保留原 bit；BF16 可由 little-endian uint16 host view 上传后再 view BF16。按原 loader 将 `conv1d.weight` 映射 `conv1d_weight` 并 reshape `[10240,4]`，仅 `A_log/dt_bias` 转 FP32；GDN norm 不做零中心减一。使用 GDN 实例的严格权重加载，核对所有参数键及加载前后 bit/shape/dtype，释放 host bytes/NumPy 暂存。该建议未实现或运行。

不要调用现有 `load_model`（它构造整模型、遍历所有 shard）；也不要先 `mx.load(5.34GB shard)` 再筛选，以免载入未使用 tensor。不能因为 MLX lazy 就声称这些临时对象不占实际资源。

## 最小模块依赖

只需 `ModelConfig.from_dir`、`GatedDeltaNet(cfg)`、`GDNCache()`，加独立 scoped integration/helper。GDN 的依赖是 `mlx.core/nn`、`model.layers`（Linear/GatedRMSNorm 与 norm Metal helper）、`kernels.gdn`（T=1 recurrent，T>1 原 chunk 算法）、按需 `kernels.gdn_metal` 的原 triangular solve，以及 `context/cache`。`model.layers` 的 compiled norm fallback 和 Metal kernel 对象仍是原实现；不另编译整个 prefill 或 scope 入口。

无需构造 `Qwen3_5ForCausalLM`、LLMEngine、scheduler、attention、MLP、tokenizer 或 KV cache。T=1 必须通过显式单层 **prefill** 入口，不能偷换成 `_decode`。实例临时 `_prefill` hook 的 private globals 只替换卷积尾状态返回；不改变 qkv/GDN 数学。

## 已知数组量级，不当作实测峰值

以下单位 MiB=2^20 B，均为逻辑张量字节。每个 case 顺序执行，不能把多个 T、多个 arm 或多个观察器历史同时保留。

| T | 输入 x BF16 `[T,5120]` | raw qkv BF16 `[T,10240]` | 卷积 xp BF16 `[T+3,10240]` | z BF16 `[T,6144]` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.009766 | 0.019531 | 0.078125 | 0.011719 |
| 19 | 0.185547 | 0.371094 | 0.429688 | 0.222656 |
| 935 | 9.130859 | 18.261719 | 18.320313 | 10.957031 |
| 978 | 9.550781 | 19.101563 | 19.160156 | 11.460938 |

每层持久 FP32 S `[48,128,128]` 为 **3 MiB**；逻辑 BF16 conv tail `[3,10240]` 为 **0.058594 MiB**。完整 GDN 输出 `[T,5120]` 与上表 x 同逻辑大小。旧 tail 可能持有整个 xp；已测 tiny 的 compact backing 数字不能直接当作真实层总内存。

| T | 原 chunk / padded T | 每个 FP32 q/k/v | 每个 FP32 pair/A/intra | solve 输入 / 输出各自 |
| ---: | ---: | ---: | ---: | ---: |
| 19 | 16 / 32 | 0.75 | 0.09375 | 1.5 |
| 935 | 64 / 960 | 22.5 | 11.25 | 45 |
| 978 | 64 / 1024 | 24 | 12 | 48 |

这只是重要数组清单，不能简单相加冒充实际 live peak。还有 `v_beta/k_beta`、q/k 衰减结果、padding/normalization/conv/matmul 临时值、chunk 输出 list、S 更新链、allocator 对齐、host 读取暂存与 runtime/编译内存。T978 有 16 个 chunk，若把初始及每次更新的 S 都命名计数是 17×3=51 MiB；实际共享、求值与回收由 runtime 决定。原 GDN 的 chunk64、solve、精度和 reduction 一律保持。

## 独立 ≤1 GiB 观察预算的可行条件

可以预注册一个新的数值诊断观察预算，不能沿用 tiny 的 64 MiB 合同：三 arm 各自 fresh process，单实例、一 case 一次；cache limit 固定 0；记录各边界 `active/peak/cache/get_array_buffer_size` 和 OS RSS/physical footprint。若 `peak` 或 `active+cache` 超过 1 GiB，记录 FAIL/partial 并停止，不调整预算补样本。运行预算、deadline、资源门和终止/reap 必须另行注册。该预算目前**未验证可满足**。

本地 MLX stub `set_memory_limit` 明确它是 graph evaluation 的 guideline，只有超额且系统没有 RAM/swap 才会分配失败。因此 `set_memory_limit(1GiB)` **不是硬 cap**；边界读取 peak 也不能保证在发现之前阻止瞬时超额。MLX active/cache/unique-buffer 不包括全部 graph、host 或 driver 内存，不能把它写成“进程总内存 ≤1 GiB”。若合同要求 OS/driver 总物理使用从未超过 1 GiB，当前方案尚无这种证明。

baseline 原 dtype bits 可逐项落盘，再释放 device oracle，后续 arm 与磁盘证据比较；避免同驻两套层/状态和无界 snapshots。每 arm 固定同一 deterministic finite BF16 合成输入和 fresh/continuing cache 种子。数值门包含 output、conv y、conv_state、S、offset、参数不变及 finite/shape/dtype；BF16 用 uint16 view、FP32 用 uint32 view。异常只保存序列化字段，不能长期保存带模型 frame 的 traceback 对象。

内存阶段至少区分 hook 后、原 prefill 产品 eval 后、共同 tail 验证 eval 后、丢弃 oracle/观察器引用后。y 的图也引用 xp；ON-materialize 并不保证 hook 当场释放全部旧 backing。ON-lazy 的复制可能直到后续验证或 batch 合并才执行，不能把额外验证 eval 造成的释放宣称为原执行路径收益。

本笔记不赋予 pressure2 运行许可。新数值诊断的资源门须独立预注册；整模型资格及 performance 的 **pressure1 门保持不变**。此处没有 HTTP throughput/latency、实际 D 正确性、整模型采用或 E023 压力根因结论。

若下一轮要真实 D 第 0 层输入，需要另行固定 tokenizer/fixture 并生成实际 embedding+input norm。整张量化 embedding 是 **682.03125 MiB**，不应顺手整表加载；按固定 token IDs 只读取所需 packed/scales 行，T978 未去重上限约 **2.686157 MiB**，再复用原 dequantize/norm 算术和 input norm 的 loader 转换。这仍是额外实现/资格，当前提案仅含合成输入。

## 已重算的小源码 SHA256

路径基准为 [E020 frozen native core](../../.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/)，不是可能随后变化的生产工作树。

| 文件 | SHA256 |
| --- | --- |
| `config.py` | `cfcc6f08c1063c59097886c89759bdfbe662be853432e4b3286169b8ea35f1c0` |
| `context.py` | `134614bd19cedae748b0db415cf1feac70cebd9826a641a843936db8327d6ba7` |
| `cache.py` | `056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791` |
| `model/gdn.py` | `c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030` |
| `model/layers.py` | `c2c9936df2a34877edb5e1702152583e54549f69b37d7c93e9996af1e7c20908` |
| `model/loader.py` | `e5170e97f1fdc747ae1d8485b7f1bdf086e4aa55930c0cbedfed90e343a13a2a` |
| `kernels/gdn.py` | `6be56a7d7ef41d3d422973a5385f598182b56f2cfd7495f6559d964460b6b0ab` |
| `kernels/gdn_metal.py` | `51757224b18a424eaee5ebb7346075837bf768c289b8721f6279b6d844de13a9` |
| `kernels/norm_metal.py` | `404a4481604adeb136190b76e83637882e7d1ac2890495dee4167f74fc84b884` |
| 已安装 `.venv/lib/python3.13/site-packages/mlx/core/__init__.pyi`，内存 API 823–916 行 | `bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce` |

CPU 接入文件见 [integration.py](../m5_ultra_256gb/conv_tail_owned_copy/integration.py) 与 [test_integration_cpu.py](../m5_ultra_256gb/conv_tail_owned_copy/test_integration_cpu.py)。12 项 CPU tests PASS；独立审阅另验证了临时 bound method 不进入 MLX Module 参数/state 字典。这些检查不是 GPU 数值资格。
