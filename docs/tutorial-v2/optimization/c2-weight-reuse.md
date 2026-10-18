# C2：先分清复用的是输入，还是权重

[六篇目录](README.md) · [上一节 C1](c1-norm-reuse.md) · [下一节 C3](c3-state-buffers.md)

## 问题：合并矩阵乘，为什么可能没有收益？

先读[第七章量化](../07-quantization.md)和[MLP 基础](../02-从ID到向量再到分数.md)。单行 decode 中，多个投影读取同一个 x，但权重各不相同；多 token 验证则是同一个 W 乘多行 x。两个方向不能用“合成一个核”混为一谈。

## 沿两个轴看代码变化

以下是伪代码，省略 guard 和完整数值表达式：

```python
# 同一 x、不同 W：输出行拼接，或一核接收多组原始指针
ys = [qmm(x, W, scale) for W, scale in projections]
ys = multi_projection(x, original_weight_and_scale_pointers)

# 同一 W、多个 x：每个 K tile 的 packed W 供多行复用
words, scale = load_original_weight_tile()
for t in range(T):
    acc[t] += original_qdot(words, x_tile[t], scale)
```

输出行拼接减少调用，却不减少需要读取的总权重；改变输出维度 N，还可能触发另一个后端调度。直接传多组原指针避免了拼接副本，但新累加器会占寄存器。类似地，每 SIMD 处理更多输出行能复用 x，却也可能降低占用率。

## 不变量：存储相同不等于输出相同

NVFP4 的4-bit值、每16值的 scale、原 qdot 求和顺序、最终 BF16 cast 都是合同。原字节的 rowblock 重排也要验证逆变换和实际读取；把解码因子移到 scale 虽有实数恒等式，极端中间值仍有溢出或下溢风险。

shared-W 的数值基线尤其要明确：每个 token 行保留 native M1 数学，而非把普通多行 QMM 当作严格 gold。已经通过的共享核缓存 packed words 和 scale，没有缓存整份解量化矩阵，也还没有实现跨 T 保存 decoded16 的后续设想。

## 为什么保留这些停止结果

原输出行拼接、R2几何没有越过收益门；rowblock、真 gate/up 融合等遇到控制漂移；factor-fold 在两臂都 compile 的新协议里仍更慢。恢复归约顺序的 GDN shuffle 通过数值却没支持采用。这些路线不能用局部正确性代替速度证据。

后来的四投影整模资格通过，仍未证明服务收益。共享 W 则已过合成输入和固定真实算子输入比较，但输出没有推进后续层；它尚不能继承整条投机路径资格，也没有性能点。应先保证同输入、同数学，再问真实宽度下是否摊薄了读取成本。

**检查理解：** 输出行拼接和跨 T 复用，哪一种可能减少同一 W 的重复读取？为什么少一次调用不能证明总内存流量减少？

## 回到完整记录

[旧18](../18-先确认优化真的更快.md)、[旧19](../19-把量化矩阵乘拆开看.md)、[旧23](../../experiments/23-kernel-tests-and-gpu-diagnostics.md)、[旧24](../../experiments/24-numerics-and-qmv.md)、[旧32](../../experiments/32-native-residual-projection.md)保留各路线完整正负结果，残差 carry 与 profiler 的独立分支也仍在原文。[拼接源码/证据](../../experiments/18-projection-packing/README.md)、[R2固定源码映射](../../experiments/19-nvfp4-qmv/source-map.json)、[四投影接入 diff](../../experiments/32-native-residual-projection-source/prototypes/v68-native-fullqual-build1/candidate.diff)、[shared-W 原源码与资格](../../code-versions/44-shared-w-m1/LOCAL-NUMERIC.zh.md)可继续核对；[共享核](../../code-versions/44-shared-w-m1/kernel124/shared_impl.metal)保留实际循环。
