# C1：相同的 Q/K 范数，只计算一次

[六篇目录](README.md) · [上一节 B3](b3-gate-fusion.md) · [下一节 C2](c2-weight-reuse.md)

## 问题：哪些结果真的可以共享？

先读[第四章 GDN](../04-gdn.md)与[后卷积融合](b1-postconv.md)。本模型16个 Q/K 头服务48个 value 头；一个 Q/K 向量的范数还会被不同输出列、不同 key 分段重复计算。它只依赖 Q/K，不依赖这些消费者自己的 S，所以可以先算一次再共享。

## 从每个消费者重算，到两阶段生产

下面是对应原实现的伪代码：

```python
# baseline：各个 head / value列 / key分段重复做相同工作
qs, ks = serial_qk_norm(q, k)
out, new_S = state_update(q, k, v, qs, ks, S)

# 候选：仅16个原 Q/K 头计算 scales
q_ready, k_ready, v_ready, scales = prepare(raw_postconv_u)  # 内部做 typed SiLU
# 第 h 个 value 头读取 scales[h//3]，但仍更新自己的 S[h]
out, new_S = state_with_shared_scales(q_ready, k_ready, v_ready, scales, S)
```

例如激活后 q=(3,4)，平方和为25；同一 q 的所有消费者都可读取同一个缩放标量。代码把原 d=0..127 的串行求和移到每个 Q/K 头的一个线程上，没有改成并行树归约。

## 不变量：共享标量，不提前改写向量

这是 L2 normalization：`qs=rsqrt(sum(q*q)+eps)*rsqrt(Dk)`，`ks=rsqrt(sum(k*k)+eps)`。它不先除 Dk 取均值，因此不是 RMSNorm；递推后另有一个 gated RMSNorm，也不能混称。

ready 向量保存 typed BF16 SiLU 后转成的 FP32 值，未提前乘 scale。epsilon 的位置、两个 rsqrt 的乘法顺序、状态表达式与四段归约都保留。把两个 rsqrt 合成一个，或把缩放提前乘进 q/k，即使实数等价，也会移动浮点舍入边界。

## 复用也需要付出空间和分派

按源码逻辑，重复 norm 对数由24,576降为16；这不是硬件指令或时间比例。方案从一个状态核变成 prework 加状态核，并产生约41KB临时结果。原路径本来也有 activation、repeat 和 cast，不能只凭新核数推断整个 forward 净多一次调用。

该阶段局部 host-wall 降低约13.33%，后续限定整模资格通过；正式端到端尝试却在末尾资源守卫处停止。完整请求已产生，也不等于正式 suite 通过，更没有由此获得默认启用或服务收益资格。早期加载后的资源 STOP 同样不是数学 FAIL。

**检查理解：** 三个 value 头共享范数后，哪些量仍不能共享？若只把串行平方和换成 `simd_sum`，为什么需要重新做数值资格？

## 回到完整记录

[旧17讲解](../../experiments/17-gdn-norm-reuse-walkthrough.md)、[全部阶段结果](../../experiments/17-gdn-norm-reuse-status.md)、[结果索引](../../experiments/17-gdn-norm-reuse-result.json)保留原计数、fallback 与资源历史；[固定 prework/state 源码](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py)可对照阅读。

```bash
# 只读比较接入位置；不会启用新路径
git diff tutorial-15-v1 tutorial-17-static-reading-v1 -- tutorial_code/miniengine
```
