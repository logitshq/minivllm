# M5 NAX：已有路径、实际分派与最小下一实验

2026-10-17；只读论文/官方源码、本机小型头文件与配置。没有导入 MLX、运行 GPU、读取模型 tensor、扫描二进制或修改冻结代码。源文件身份和静态形状计算保存在 [sources receipt](m5-nax-prefill-opportunities.sources.json)。

**结论：本机 MLX 0.32.3 已有 NVFP4 NAX 实现。机会是找出仍被 QMV/split-K 分支截走且值得加速的具体投影，不能把“接入 NAX”当成尚未做过的整体优化。** 对长 prefill，主要大投影已经符合 NAX 路径条件；需要先测剩余算子的时间占比。本文没有设备级 kernel 命中或加速结论。

## 论文确实说明了什么

阅读 [BaseRT v1](https://arxiv.org/html/2607.19438v1) §2、§3.1–3.2、§4.1。其策略按 M5 芯片、计算强度与形状选择 tensor 路径：prefill GEMM、MoE GEMM、prefill attention 使用 Metal 4 cooperative tensor；单行带宽受限 decode 保留原 GEMV。论文提到形状回退，但**没有公布数值阈值、完整 shape predicate 或 NVFP4 支持合同**。它的实验设备是 M5 Pro，MLX 对照是 0.32.0，不是我们的 M5 Ultra/0.32.3/固定 NVFP4 checkpoint。

[BaseRT 官方仓库](https://github.com/basecompute/baseRT#whats-in-this-repo)说明公开部分是 CLI、格式、头文件、bindings 等，engine 作为另行许可的预编译二进制分发。因此本次没有取得可审查的论文版 engine dispatch 源码，不声称复现其未公开阈值或使用同款 kernel。NAX 指 GPU 内矩阵单元，不能与独立 ANE 混同。

## 本机与官方版本核对

主 `.venv` 和官方 baseline 环境的 dist-info 均为 MLX **0.32.3**。主环境随 wheel 安装的 `fp_quantized_nax.h`、`steel/gemm/nax.h` 与官方 **v0.32.3** 同路径文件逐字节相同；此前本地保存的 `quantized.cpp` 也与本次取回的官方 tag 文件相同。只取了五个小源码文件，合计约 191 KiB；没有拉取整个框架或模型。

真实 kernel 链条是 `Linear.__call__` → `mx.quantized_matmul` → host dispatcher → `fp_qmm_t_nax` → `tile_matmad_nax` → `mpp::tensor_ops::matmul2d`。本机头文件显示 NVFP4 packed 权重解码到 threadgroup tile，以 FP32 accumulator 完成矩阵乘，再写回输出 dtype；这是实际的 MPP 路径，不只是名称包含 NAX。JIT 对非 affine 模式选 `fp_quantized_nax`，并复用 `gemm_nax`。[官方 JIT](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/jit_kernels.cpp#L1263)、[官方 FP kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/fp_quantized_nax.h)、[MPP 调用](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/gemm/nax.h#L401)。

NAX 可用性还要求没有编译宏 `MLX_METAL_NO_NAX`、系统版本至少 macOS 26.2，以及 architecture generation 满足 `gen >= (suffix=='p' ? 18 : 17)`。当前只读 `sw_vers` 是 **27.0.1**；既有资格使用 `applegpu_g17d`。安装 CMake 配置声明 Metal 构建开启、Metal language 400，但仅凭头文件/CMake metadata不能证明当前二进制内部宏值或某个请求实际命中 NAX。设备 trace 仍是后续资格项。这个宏是构建开关，不是已核实可随请求使用的环境变量。[官方可用性判定](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp#L952)。

## 分派顺序与回退

下面是固定 v0.32.3 host 代码的静态摘要，实际形状/连续性必须由运行时 receipt 补足：[quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L1868)。

1. 二维 weight + row-contiguous activation 会折叠前导维，`M=x.size/K`；否则保留 matrix 的 M 和独立 batch。
2. 转置矩阵先比较 QMV 阈值。对 `g17d` 且这里的 `K>4096`，阈值为 **12**；M1 为 QMV，M2–11 可走复用权重的 QMV-wide。不是一切 decode 都固定 GEMV：较大实际 batch 可进入 QMM。
3. 达到 QMM 阈值后，单矩阵转置路径先计算 split-K：`s=max(1,floor(512/(ceil(N/32)*ceil(M/32))))`，按 K/group 对齐收缩。**s>1 使用 SIMD partial + BF16 intermediate reduction**；s=1 才继续普通 QMM。
4. QMM 再要求 NAX available、`K%64==0`、转置或 affine；非转置另需 `N%64==0`。FP32 在 TF32 未允许时回退。转置 N 不整除 64 有安全尾部路径；不必为了 N48 padding 改数学。
5. 进入 NAX 后转置 `M<=32` 使用 BM32，其他用 BM64；BN/BK64。未命中这些条件会走 SIMD QMM。非连续输入可能先复制，而不是必然禁用 NAX。

这些判断不等于宣称任意 quant mode/bits 都有效；量化 API 本身另有合法格式约束。本项目的固定 `BF16 activation + transpose=True + NVFP4/group16/bits4` 已属于所读实现范围。不要通过打开 TF32、改量化或改变 M 来绕过正确性合同。

## 固定 Qwen3.8-27B 的静态投影矩阵

输入为原 checkpoint `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6` 的 config；K/N 由本仓库 `model/{layers,gdn,attention,qwen3_5}.py` 推导。下表假定单请求二维权重、连续 BF16 激活和 NAX availability。`NAX*` 只表示 host 条件满足；`Sx` 是计算出的 split-K 分区数，均非实测。

| 投影 | K | N | T19 | T128 | T512 | T1024 |
| --- | ---: | ---: | --- | --- | --- | --- |
| MLP gate/up | 5120 | 17408 | NAX* | NAX* | NAX* | NAX* |
| Attention q（含 gate） | 5120 | 12288 | NAX* | NAX* | NAX* | NAX* |
| GDN qkv | 5120 | 10240 | NAX* | NAX* | NAX* | NAX* |
| GDN z | 5120 | 6144 | S2 | NAX* | NAX* | NAX* |
| MLP down | 17408 | 5120 | S2 | NAX* | NAX* | NAX* |
| Attention/GDN out | 6144 | 5120 | S3 | NAX* | NAX* | NAX* |
| Attention k/v | 5120 | 1024 | S16 | S4 | NAX* | NAX* |
| GDN a/b | 5120 | 48 | S160 | S40 | S16 | S8 |

`last_only` lm_head 通常 M1，不能把 prompt 长度 T 当作它的矩阵 M。E010 已采用的成组 prefill 用广播权重保留 M=T，并只共享原先 splitK1 的投影；该路径不是本次新想法。既有 [prefill 设计](../../docs/optimization-guide/prefill-design.md) 已解释改变 M/分区为何可能破坏逐位一致。上述 N48 细投影虽然尚未 NAX，也不能仅因缺 NAX 就认定它是总时间瓶颈。

## 最小可关闭实验：保留 split-K 合同，只换 partial kernel

**候选，未实现/未运行。** 一个隔离模块开关 `prefill_nax_same_split=OFF|ON`，默认 OFF，仅针对真实 prefill 的 NVFP4/BF16 转置投影。首屏固定 T19 的 GDN z、MLP down、Attention/GDN out；先记录真实输入/packed weight/scale，不接 HTTP 服务。OFF 完整使用当前 MLX；ON 保留相同 split 数、K 分区、BF16 intermediate 形状及最终 reduction，只把各 partial 的矩阵乘换成固定 MLX NAX loader/MMA。分区 K 不整除 64、M 不在预注册集、dtype/quant 不符立即回原路径；N48/T19 的 K 分区只有 32，不纳入这个候选快路。

选择此切点是为了减少同时变化的因素：直接跳过 split-K 会删除原 BF16 partial 舍入点，不能只以数学等价或最终 token 相同放行。即便保留 partition，SIMD 与 NAX 的内部累加顺序仍可能不同；**这是待检验假设，不是 exactness 证明**。最小实现可放独立 bench extension/primitive，复用已安装头文件；不要改全局框架、核心 Linear 或冻结 baseline。

资格先于计时：每个 partial、最终输出逐字节相等，原 inputs/weight/scale bytes 不变；额外覆盖 M12/19/32/33/128 的边界和退回路径，明确 observer 确认真实 kernel 名。任何一点失败就拒绝该 exact 优化，不改成容差、换 prompt 或删除坏点。成功后才在无 observer 的同输入独立进程测该模块（各臂初调另记，2 warm、3 ABBA、2 GiB 池），预声明 micro GM≥1.20 且每点≥0.97 的筛选门。随后还必须过完整逐层 logits/cache、连续 decode、normal-EOS 输出与真实调度资格；最终收益只能进入同轮 **实际 output throughput vs POST→DONE latency** 的并发曲线。

若 partial 逐位门失败，本次结果应保留为“归约路径不兼容”，不继续靠重分区追速度。若微模块过门却整体收益很小，则说明这部分的占比不足；下一方向应由长 prefill 的实际 trace 决定，不能把论文里别的模型/量化/设备的倍数移植到我们的主图。

## SHA-256 与未决项

| 文件 | SHA-256 |
| --- | --- |
| v0.32.3 `quantized.cpp`，与原本地副本一致 | `36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982` |
| v0.32.3 `device.cpp` | `d2254ea5cb2ac282f65c2869c6ef56decc97669e2a5fc26fba9d831b60569c9f` |
| v0.32.3 `jit_kernels.cpp` | `f757ea0a975a0966be61a64cef6535c62ef4f2b5571abe7b8f7547ad36b38fb2` |
| installed/upstream `fp_quantized_nax.h` | `8a712870dc17493ddf05daa706481d52867b29e9141d2ecb76dccf47f04c8b42` |
| installed/upstream `steel/gemm/nax.h` | `fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec` |

仍未知：当前 wheel 的编译宏与每次请求的实际 kernel、各投影时间份额、partial 换核是否逐位兼容、实际收益。这里没有硬件 counter、Apple NAX utilization、性能或网络前沿新结论；GLM 的 MoE gather dispatch 也不由这份 Qwen dense 分析代替。
