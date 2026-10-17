# V131：新论文与 Metal 实现带来的三个小实验

检索截止 **2026-10-17**；本次只做 web primary-source 阅读和本机源码对照，没有导入 MLX、运行 GPU、模型或服务，也未改动引擎。建议优先 **A：decoded-W 复用**，其次 **B：X tile 暂存**；C 是可快速否决的码本探针。shared-W 本身、ICB、GPU feedback、conv/norm 融合已具备，不作为新发现；M5 已研究的 DynaCore、batch-1 dispatch、MacPaw ANE、SiliconBench也不重复包装。

## 核过日期与实现的来源

| Primary source | 发表/版本日期 | 本次读到的可用信息及限制 |
| --- | --- | --- |
| [Profiling LLM Inference on Apple Silicon](https://arxiv.org/abs/2508.08531) / [全文§5–6](https://arxiv.org/html/2508.08531v1#S5) | 2025-08-12 v1 | M2 Ultra/Max、M4 Pro 的低bit分析；dequant/int/occupancy 与带宽一起限制执行，低bit不能自动推导速度。不是本机NVFP4加速证据。 |
| [Rigel](https://arxiv.org/abs/2606.12765) / [全文§6–8](https://arxiv.org/html/2606.12765v1#S8) / [作者说明](https://ramchandk.com/blog/rigel-apple-m4-metal-tensor) | 2026-06-11 v1；作者文06-12 | M4 Max、macOS27 beta 的张量路径实测；fp8不等于硬件加速，cheap epilogue与更大fusion不可混同。其宣称的MIT实验代码本次未定位到可核的具体repo，不冒称已读。 |
| [BaseRT](https://arxiv.org/abs/2607.00501) / [作者仓库](https://github.com/basecompute/baseRT#whats-in-this-repo) | 2026-07-01 v1 | 论文有native Metal/格式特化/fusion；**当前repo明说核心engine是另发的proprietary prebuilt**，开源的是CLI/格式/API/bindings，不能据此声称读到其kernel。 |
| [Lily作者的09-02报告](https://github.com/perplexityai/pplx-garden/blob/main/lily/docs/2026-09-02-performance-mlx-0.32.2.md) / [README](https://github.com/perplexityai/pplx-garden/tree/main/lily#requirements) / [skinny.metal](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/metal/skinny.metal) / [skinny.rs](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/skinny.rs) | 日期报告2026-09-02 | 真正读到staged-A及register-A两种small-M实现。整个项目要求Apple family10/M5+、MoE affineQ4/group64；不是M4/NVFP4可直接运行的donor。报告非HTTP计时。 |
| [GANQ](https://arxiv.org/abs/2501.12956)；[T-MAN](https://arxiv.org/abs/2511.11248) / [作者实现](https://github.com/microsoft/T-MAC/tree/main/t-man) | GANQ 2025-01-22 v1 /06-09 v3；T-MAN 2025-11-14 v1 | LUT可替代低bit解码是线索；前者实测RTX4090，后者作者README为Qualcomm NPU，均不是Metal NVFP4实现。只借机制，不搬数字/重量化方案。 |

Apple官方另有 [M5 MLX说明（2025-11-19）](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) 与 [最新Mac Studio说明（2026-10-15）](https://machinelearning.apple.com/research/mlx-distributed-inference)：M5 TensorOps/Neural Accelerators 与M4 shader路径应分开；后篇Qwen3.8示例是BF16、8192/65536context，不是我们的19→32 NVFP4 HTTP对照。以上日期均不晚于截止日；Lily当前main不是历史冻结snapshot，原型前仍需锁commit，GitHub API本次未能取得as-of SHA。

## A．缓存 decoded-W，而不只缓存 packed-W（首选增量）

**实际差异**：本机 [V124 shared_impl.metal:31](../../code-versions/44-shared-w-m1/kernel124/shared_impl.metal) 缓存4个uint16 packed字，scale在token loop外，但 `v124_qdot_cached:7–10` 的16次 `Dequantize<4>` 仍在每个token表达式中；是否已被编译器CSE提到loop外 **尚未知**。Lily [skinny.metal:128–165](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/metal/skinny.metal#L128-L165) 则先构造decoded `wq[]`，再遍历M行。这是packed-W复用后的另一层ALU复用，不把M5 E027或V124数值PASS当新idea/整模收益。

**最小变更**：仅在每row/K512块将16个E2M1值用原 `Dequantize<4>` 解码一次、保留为thread float[16]（或逐元素已证可无损的half[16]），T=2/4/5共享；保留原四项表达式括号、四次accum顺序、scale在qdot末尾相乘、R4 `simd_sum` 与末尾BF16 cast。不能照抄Lily的 `dot(float4)`、BF16-rounded affine weights或其不同归约。

**M4/M5与失败门**：普通Metal寄存器可用于二者，不需NAX。额外16个float的live set可能降低occupancy/溢出；half逻辑32B也不代表实际只有16bit寄存器。先单L0 gate、真实既存bank、T2/4/5，A=冻结V124、B=decoded-cache，另保留per-row native M1/control严格输出与input readonly。若编译后早已hoist、出现spill、任一byte不同，或受控native-ICB micro未过原5%门，即STOP；不先扩整模。当前没有新增GPU证据。

## B．只改变 X 的载入位置：每K块一次BF16 threadgroup tile

**实际差异**：V124 `shared_impl.metal:34–40` 为约束register压力，在四个output-row中反复从device读小X。Lily [skinny.metal:7–44,78–99](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/metal/skinny.metal#L7-L99) 有staged-A路线；其host [skinny.rs:105–129](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/skinny.rs#L105-L129) 按shape选择，不能把register路线的减速或加速泛化。

**最小变更**：固定现R4/SG2/K512，协作加载 `threadgroup T x_tile[TOKENS][512]`（T沿用原BF16类型），每K块所有线程统一barrier后读；进入下一K块前再barrier。T2/4/5分别2/4/5KiB逻辑TG scratch，只移动原BF16字节，不预归约、不pad实际T、不改qdot。先独立于A测，不同时改两个因素。

**风险与止损**：M4/M5普通threadgroup memory均可研究，但Lily全项目的M5门不能移除后直接宣称兼容。X原本可能已命中L1；K5120新增20个TG barrier、K17408新增68个，可能比节省load更贵。任一非uniform barrier/越界/字节漂移立即STOP；local compile/native micro若无稳态受控收益，不扩大LMhead或整模。此项测试memory-placement，区别于已STOP的rowblock、R2/R8、GDN shuffle和scale fold。

## C．精确16项E2M1码本：用SIMD shuffle替换位解码（低优先级探针）

由GANQ/T-MAN的LUT思想推导，**不是其作者已实现的Metal技巧**。每lane准备其编号对应的原E2M1值，`simd_shuffle(lane_value, nibble & 15)`获得FP32解码值；保留原packed layout、scale位置与全部qdot归约。表必须含正确正/负零，由冻结decoder生成，不能替换成NF4/learned codebook或重校准weights。

M4/M5都有普通SIMD shuffle；减少bit/half转换未必抵得过每权重一次shuffle。它与旧V29的GDN跨lane归约重排不同，也与V31移scale不同。第一门是16 nibble × 所有有限scale编码/零/符号的位结果；再单shape对native M1严格BF16及readonly。新增跨lane依赖、NaN/Inf/signed-zero不符、或micro不达门即停。不要将GANQ的CUDA倍率、T-MAN的NPU倍率写成M4预期。

## 不应据论文重复投入的路线

- 当前native101源码已经使用 `MTLDispatchTypeConcurrent`，且已有ICB/两步异步/GPU反馈；Lily的这些概念不是新delta，不能重新“加并发encoder”冒充优化。
- Rigel的MPP epilogue并不证明旧V28 gate+up+SwiGLU可以重跑，更不允许改变BF16中间舍入；它的M4/macOS27beta结果也不证明本机工具链可用MPP。
- [SpQt（2025-11-06）](https://arxiv.org/abs/2511.04477) 的阈值激活稀疏/新quant分组、VBQ训练期改bit、KV压缩均会改变当前严格baseline；不作为这轮无损kernel候选。

三项都只是可证伪的实验假设，没有收益承诺。后续即便local numeric/micro通过，仍须原32ID/全logits+128cache/readonly及reset/changed-request资格，再做同电源fresh HTTP吞吐—延迟比较；不能凭某shape微测或M5结果直接采用M4。
