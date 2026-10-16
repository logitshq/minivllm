# C1 GDN conv reduction contract：已确定 BF16 路径，仅剩一个 pipeline 数值条件
2026-10-15；CPU-only；未 import MLX、运行模型/GPU/resource/browserUI/app 或改 repo。
结论：指定 row-contiguous BF16 product `[1,4,10240]`、axes1 的源路径确定为 `col_reduce_small`，**accumulator 是 BF16，不是 FP32**。
精确左折叠还条件于该 kernel 的 `maxTotalThreadsPerThreadgroup >=128`；本轮禁止 GPU 查询，不能将这个 runtime pipeline 属性写成已实测。
## 固定来源与推导（均为官方 v0.32.3 或本机同版本 headers）
H=`/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx`，version.h 明示0.32.3；header SHA 与两份日志 SHA 在 `c1-conv-reduction-contract-evidence.json`。
目标输入 row-contiguous、strides `[40960,10240,1]`，只 reduce axis1；[get_reduction_plan](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/common/reduce.cpp#L25-L57) 得 `ContiguousStridedReduce(shape=[4],strides=[10240])`。
[metal/reduce.cpp:99–127,875–903](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/reduce.cpp#L875-L903) 得 reduction_size4/stride10240、non_col_reductions1；total4<32，选 `strided_reduce_small`。
[同文件:547–607](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/reduce.cpp#L547-L607) 参数 NDIMS1/int、n_reads4、threadgroup_x32、y=`min(4,M/32)`（整数除法），grid=(80,1,1)；M 为实际 pipeline max threads。
[同文件:227–256](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/reduce.cpp#L227-L256) sum 的 BF16 输入/输出均不升型。
[JIT:501–528](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/jit_kernels.cpp#L501-L528) 实例化 `col_reduce_small<bfloat16_t,bfloat16_t,Sum<bfloat16_t>,int,1>`，没有隐藏 `Sum<float>`。
`H/backend/metal/kernels/reduction/ops.h:132–165` 的 `Sum<U>::operator()(U,U)` 返回U；`bf16.h:9` 是原生 Metal bfloat。
## 精确算术顺序与唯一剩余 dispatch 条件
令 `R` 是目标 BF16 rounding；`p_j=R(xp[0,j,c]*w[c,j])`，原产品数组必须保留这个 R，不能 FMA 跨过它。
`H/backend/metal/kernels/reduction/reduce_col.h:25–58` 中每个 y-thread 先 `t=+0_BF16`，按 j=y,y+Y,... 做 `t=BF16_add(p_j,t)`。
`:61–86` 把各 y 的 BF16 t 写 shared memory；y0 按 y=1,2,... 做 `t=BF16_add(shared[y],t)`，无 simd tree/FP32 reduction。
若 M≥128，Y4，每线程仅一个 tap；结果是 `t0=BF16_add(p0,+0)`，`c=BF16_add(p3,BF16_add(p2,BF16_add(p1,t0)))`。
**还缺的仅是 M≥128 的 actual pipeline 证明**，不是未知 reduce selector；不能用 device max threads 或别的 kernel 的 M 替代。
若 M=64..95，Y2，顺序为 `fold(p0,p2)` 与 `fold(p1,p3)` 后再 BF16 合并；若 M=96..127，Y3，先 fold(p0,p3)，再p1、p2；若 M=32..63，Y1，顺序0→1→2→3。
因此后续 micro 可同时比对这些预声明分支；首次实际 GPU owner 记录原 kernel pipeline M 或 launch Y 后选已知分支，禁止按结果挑树。
若原 product 实际不是 row-contiguous，也必须记录 flags/strides 再重推；本结论只覆盖请求明确给定的布局，不外推 B2/B4/T>1。
## compiled SiLU 的中间 cast 不可省略
本机 `mlx/nn/layers/activations.py:138–146`：shapeless compiled `x*mx.sigmoid(x)`；这是 actual VLM 与我们共同的 helper，不是两次裸 Python elementwise。
`H/backend/metal/kernels/unary_ops.h:308–314`：Sigmoid<T> 返回T，内部表达式 `y=1/(1+metal::precise::exp(metal::abs(x)))`，结果 `(x<0)?y:1-y`。
[compiled Metal codegen:191–207](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/compiled.cpp#L191-L207) 给 tape 的每个 node 显式按其 dtype 声明局部值。
因此等效边界是 `BF16 s=Sigmoid()(c); BF16 activated=Multiply()(c,s)`；`H/.../binary_ops.h:208–213` Multiply 返回T，必须保留 sigmoid BF16 round。
不能替换成单个 FP32 `c/(1+exp(-c))` 后一次 BF16 cast；内部特殊值/denormal/exp 的精确行为仍以原 Metal helper 与真实 byte micro 为准。
## CPU 反例（source-contract 演算，不是 GPU 数值认证）
`c1-conv-contract-cpu-counterexamples.json` 用纯 stdlib struct 的 FP32+RNE BF16 emulator；无 numpy/MLX/模型。
有限 cancellation：products=[256,1,-256,1]，逐次 BF16 左折叠=1，FP32 taps 或先0+2/1+3的树=2。
product round：xp=[1.0078125,1,0,0]、w=[1.0078125,-1.015625,0,0]，本路径conv=0，FP32 products再cast=2^-14。
极小数边界：两项 xp=2^-133/w=.5，逐product RNE→0，FP32 products先和再cast→2^-133；这是IEEE/RNE理论反例，未认证 Metal denormal 是否 flush。
这些反例证明不能删中间 R；不证明任何新 kernel 已合格，finite/字节/状态/32IDs/生命周期门均保持。
## 指定两份 oMLX 日志：实际 VLM 已进入融合，不只是 installed
P=`/private/tmp/minivllm-engine-optimization-20261014`；两份指定日志均存在，各89行。
`P/freshscreens-ac-frontier-attempt1/omlx-balanced-c1/server.log:8,58` 记录 type/engine=vlm 与 VLMBatchedEngine loaded；`:48` patch applied；`:60` **Qwen B1/T1 fused GDN prework engaged**。
`P/metal-buffer-exploration-AC-attempt1/omlx-balanced-B/server.log:8,58,48,60` 同样，engaged 时间2026-10-15 01:37:07.626；freshscreen 对应2026-10-14 23:37:00.917。
对应已读 patch 源在成功 fused 调用后只 log once，故至少一次真实 B1/T1 调用进入；不能当每token/每layer/全部计分请求都走融合的计数证明。
此前 LM-only compile 研究所读 `mlx_lm.models.qwen3_5` 不能描述这些 actual VLM 路径；此更正只改源码解释，不改已测 official 配置/分数。
## 后续最窄行动
可进入默认off conv+SiLU算术 prototype 的准备：先固定 actual M/Y/原布局，再显式复制逐product/BF16 sum/typed sigmoid/typed output；不动gates/norm/S/QMM/量化。
本研究已消除 accumulator/selector unknown；保留上述一个精确 pipeline条件，禁止继续泛泛 NO-GO 或省略真实 byte micro。
