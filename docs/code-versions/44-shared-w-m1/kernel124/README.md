# V124：共享原始权重的多行 M1 QMV 源码原型

**结论：READY_FOR_ROOT_STRICT_LOCAL_NUMERIC（仅源码/CPU；GPU UNRUN）。** 这是 V117 数学 profile 下的新优化假设。没有改旧门槛、候选/engine、W 或 Git；没有新数值、ID、性能或采用资格。

## 真实路径与差异

- 固定 MLX0.32.3 commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`，`fp_quantized.h:339–406` 的 M1 fast：TG64=SG2，每SG四个 N 输出；lane 各处理16个K值，K外循环步512；每16值内四组原括号四项和→一次scale乘→FP32 result +=；最后原 `simd_sum`→BF16。`quantized.cpp:463–539` 给M4 R4（R2只gen17s）。本地固定头SHA `359a1ea9…d719f`，完整路径/SHA/行号见 source-map。
- 同版 `quantized.cpp:1825–1865` 的 M≥2 会走 `qmv_wide`；其 `fp_quantized.h:570–690` 使用16个K lanes、`dot(float4)`、shuffle梯，**不是相同M1规约**。V112在同一实际T5轨迹351/497投影不同；V113把M固定1/B=T后497/497严格相等。singleton W/scales是view，不是T份权重副本；不同TG仍各自发出权重读取，物理DRAM复读量未测。
- V117已独立审计的整链PASS：A/B各旧96IDs、49事务/211 verify行、49×128 committed cache与49 last-computed logits严格；不是旧失败重判。T分布1:6/2:2/3:1/4:2/5:38。该证据**不自动资格新shader**，也没有V117性能改善结论。
- 当前实际安装 oMLX `speculative/vlm_mtp.py:44–60` 接 mlx-vlm MTP；`speculative/mtp.py:673–683` 将anchor+draft交target verifier。Q35 `models/qwen3_5/speculative_verifier.py:48–58,215–235` 接 `speculative/ops/linear.py:33–132`：affine优化不合NVFP4后，投影用 `quantized_verifier.py:391–420` 的singleton `gather_qmm`；固定MLX gather fast最终仍调同一M1 impl（头1642–1699）。这是当前源码选择推断；109/111实际MTP参与证据没有独立观测各投影PSO。
- donor另有共享T的custom qmv，但 `quantized_verifier.py:1645–1690` 要求 **affine + bias + BF16/F16 scales**，不能直接给本目标NVFP4。头部另可走optimized NVFP4 argmax（同文件1844–1935，8值/lane、K步256）；不把它冒充本方16值/lane M1 reference，也不宣称已观测该叶子参与。
- Lithos `compiler/emit.py:312–405` 为verify生成T variants；`kernels/common/gemv_T.metal:385–509` 在T循环外读取/解码W，T各自FP32acc，但用lane-row packed slab、不同K分工/显式FMA/最终row-scale。可借共享机制，不能拷数学或重绑权重。此kernel为当前checkout源码；旧handoff text未包含该文件，不能冒称旧M4 none arm实际spec参与/二进制证明。

## 最小实现

`shared_m1.py` 是注入backend的standalone helper，默认不执行、不注册NN数组；只支持已声明9种N/K、BF16 `[T,K]`、T=2/4/5、原U32 `[N,K/8]` +U8 `[N,K/16]`、g16/b4/no bias/global scale。prepare强持有原对象，project核当前同一W/scales对象及元数据；独立观测的canonical strides必需。`ensure_row_contiguous=False`，没有concat/slab/repack/dequantize/隐式contiguous。

`shared_impl.metal` 每 K块/输出row只读取4个packed ushort+1scale一次，随后T个独立FP32 accumulator重复**原qdot四项表达式**；原K次序、scale位置、32lane simd_sum与BF16 cast不变。grid=(64,N/8,1)，TG=(64,1,1)；只有W/scales/X/Y四个显式指针，无shape/stride隐参。`kernel_header.metal`/MIT license原字节复用冻结V66；control_body调用原R4实现，不用V79 half2或factor-fold。

为避免同时常驻T×16个X float，本版逐输出row重读小X：源级逻辑X loads为原M1的4倍；result为4T floats/lane，另有16X floats与4packed ushort。共享packed loads不保证编译器只读一次/不spill，也不保证物理带宽改善。V117旧T分布下T2/4/5覆盖42/49事务、202/211行；仅按源码矩阵读取计51/211 passes，**不是实测75.8%带宽或E2E收益**。本原型不支持T1/3/6..8；将来若集成应在受限verify scope显式回到已资格V113，当前没有集成。

## exact 反例与下一数值门

禁止改为qmv_wide、让不同token共享/分割32lane reduction、scale折进每项、FP16/BF16预解码权重、跨K分块归约或改FMA策略。这些可数学等价却改变舍入。缓存uint16只搬位、不改原解码；即使源表达式相同，编译器优化仍可能改变实际结果，首个raw差异必须STOP，不放宽。

先比较同一原始W/scales/X：builtin concat-M1、冻结V113 broadcast-M1、原custom R4 bridge、V124；所有输出shape/dtype/finite/full raw bytes（含signed zero）与输入/参数readonly硬门。T2/4/5×9种N/K，8组预声明独立rows：零/带符号零、单点跨lane/512边界、重复/置换rows、正负抵消、小/大有限BF16；有限checkpoint scales。再重捕V112的497实际同input（保存的是hash，未保留全497原始输入，不能伪装离线bank）。历史MLP input若复用只能标generic，不冒GDN来源。

局部严格通过后，另建defaultOFF fork116 verify-only：保V117旧96IDs、独立ordinary-M1 A自己推进、same-frontier128与last-computed logits、literal/history/parameter readonly全部硬门；B不能读取A的未来IDs/状态。未过此门不得接服务/性能/采用。local速度也不能代替正式HTTP吞吐-延迟Pareto。

## CPU复验与使用接口

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v124-shared-weight-m1-source-build1/check_cpu.py
```

CPU结果：48几何/拒绝边界、54首末tile地址/唯一output owner检查；qdot四项表达式对固定头逐字token一致、header/license原字节相同；无浮点数值模拟、MLX/NumPy/GPU。本机 `xcrun --sdk macosx --find metal` 返回72：没有command-line metal compiler，故不声称Metal已编译。root后续显式GPU数值入口可注入mx调用 `prepare(mx, op, weight_strides=…, scale_strides=…, runtime_version='0.32.3', device_name='Apple M4 Max', architecture='applegpu_g16s').project(x, observed_x_strides=(K,1))`；本包没有启动GPU的CLI/计时器，不能误运行为性能资格。
