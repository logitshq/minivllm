# 24：恢复数值顺序，或把量化缩放移出循环，能否提速？

上一轮 SIMD GDN 在完整模型里失败。这一轮先恢复原求和顺序，再测试 Lithos 的另一处真实实现：NVFP4 解码时把二次幂因子移到分组 scale。两条路线目前都没有带来可采用的加速，原运行时代码保持原状。

## GDN：数学相同，还要看运算顺序和实际速度

原 kernel 把128维 K 分为四段，每段按顺序累加32项，最后按顺序相加四个部分和。V27 改成 SIMD 归约后，局部误差很小，但完整模型第一步 logits 最大差异0.125，超过预先确定的1/16。

V29 保留 compact Q/K、寄存器 state 和转置存储，但用 shuffle 和串行累加恢复原来的归约顺序。15组输入、两种 state 布局及16步链检查中，输出和新 state 的字节实际全部相同。这证明本次改写能恢复这些测试的数值，不能推出它更快。

局部计时恰好相反：原版约144.92µs，新版约292.67µs，描述性慢101.96%，六对均未获胜；控制组也有一项超过5%漂移。结果保留为 `CONTROL_DRIFT_STOP`，不进入完整模型或服务测试。不能用这个计时宣称稳定的慢速倍率，但它没有支持继续采用该实现。

另一个独立原型 V30 保持原 threadgroup、四段归约和 state 布局，只把每线程读到的32个 state 元素保存在局部数组，更新时复用。30项真实/合成输出检查严格通过；尚未运行性能实验，不能称为寄存器优化成功，编译器是否溢出到内存也没有实测证明。

## NVFP4：移动16384这个因子

在固定版本的 Lithos `monolith/formats/nvfp4.py` 中，默认 `NVFP4_DECODE=3` 把 native FP4 解码的二次幂因子移到分组 scale。简化后的改动是：

```text
原来：value = float(decode_half(code) * half(16384))
      scale = float(decode_fp8(group_scale))

实验：value = float(decode_half(code))
      scale = float(decode_fp8(group_scale)) * 16384
```

保留原 R4/K512 累加、权重布局和 BF16 输出，避免重排或复制权重。移动因子可能减少循环内操作，但极端 activation 的中间溢出/下溢并不普遍等价；CPU 反例和实际测试范围均保留，不能只由代数相等宣称所有浮点输入相同。

| 实验边界 | 严格数值检查 | 局部计时 | 结论 |
| --- | --- | --- | --- |
| V31：原 eager API 对自定义 kernel | 27项通过，输入只读 | A约205.25µs，B约207.42µs；仅1/6对获胜，控制漂移 | `CONTROL_DRIFT_STOP` |
| V32：A/B 都放进 `mx.compile`，activation 显式传入 | 新的27项通过，输入只读；编译在计时外，无重 tracing | A约205.27µs，B约211.55µs；控制组通过，B慢3.0617% | `STOP-MICRO-GATE` |

V32 是为了单独检查 Python 调用边界的影响而预先新建的协议，不是把 V31 的旧结果改成通过；它也没有支持该改法。这里列出的都是局部 host-wall 边界，不是独占 shader 时间，没有完整模型、HTTP 或新 Pareto 点。

下一条实现路线转向 Lithos 的固定 buffer 和原生执行序列。先捕获真实绑定及同步依赖，再验证重放的 logits、全部缓存和原输入。是否能提高连续生成速度，必须由后续运行回答。

[原型源码与范围说明](24-numerics-and-qmv-source/README.md)、[文本证据索引](24-numerics-and-qmv-source/evidence-index.json) 和 [原始收据压缩包](24-numerics-and-qmv-source/evidence.tar.gz) 保留本轮结果；所有实际进程均已回收并有外部不存在证明。
