# B3：把小投影和门计算放进同一个 prework

[六篇目录](README.md) · [上一节 B2](b2-conv-fusion.md) · [下一节 C1](c1-norm-reuse.md)

## 问题：门已经编译了，为什么还会有小算子？

先读[第十五章门计算](../15-把GDN的小门计算一起编译.md)与[卷积融合](b2-conv-fusion.md)。第十五章把门公式放进局部编译函数，产生 a、b 的两次量化投影仍独立。这里的 prework 接收卷积后的激活，输出供递推使用的 Q/K/V 与缩放标量；本篇把两个小投影也带进去，状态递推核继续独立。[C1](c1-norm-reuse.md)随后展开这些缩放标量的范数原理。

## 一起生产递推需要的六组输入

下面是教学伪代码，接口名与中间变量为便于阅读作了简化：

```python
# baseline：两路 QMM 先各自产生 BF16 结果
b, a = b_proj(x), a_proj(x)
beta, g = original_fp32_gates(b, a, A_log, dt_bias)
q, k, v, scales = norm_prework(activated)

# 候选：使用同一份 x、activated 和原始权重指针
q, k, v, scales, g, beta = ba_gates_prework(
    activated, x, Wb, Sb, Wa, Sa, A_log, dt_bias)
```

每路 BA 只输出48行；已有48个 V 组的部分线程承担这些行的投影，随后准备门。权重和 scale 仍是原对象，不拼接、不重新量化。未知布局、参数对象变化或不满足正常 B1 路径的输入，在状态改写前回退，不能拿旧 prepared 指针继续算。

## 不变量：先 BF16 投影，再 FP32 gates

beta 仍为 `sigmoid(float32(b))`；g 仍为 `-exp(A_log)*softplus(float32(a)+dt_bias)`。新核保留原 R4、K512、qdot、归约和投影末尾的 BF16 cast，再进入原 FP32 functor。softplus 沿用 `logaddexp(x,0)`，不另换看似等价的 exp/log 公式。

这也是不能直接复制另一模型分支的原因：所读 Qwen4 分支的 gates 舍入不同。融合边界相似，不代表数学相同。Q/K 串行范数、状态核和输出 norm 也不因本次融合而更换。

## 数值门与速度门分开

局部720项严格检查之后，真实模型还要验证48层的输入布局、原参数身份与只读范围，并确认观察 hook 在 replay 前移除。三请求逐步 logits、128缓存和反馈标量共12,276项对旧参考通过，才取得这条限定路径的整模资格。

随后 raw-native screen 记录约28.69 token/s，但没有公平配对 A 或 HTTP 客户端，不能称为速度提升。更少 dispatch 只说明图的边界变了，不能从数量直接推算耗时。源码的 buffer 参数数目也不等于实际反射已观察到的数目。

**检查理解：** 为什么把 a、b 直接留在 FP32 再计算 gates，仍可能改变模型输出？为什么局部通过后还要观察真实48层布局？

## 回到完整记录

[旧34完整章](../../experiments/34-native-ba-gates-prework.md)保留局部输入来源、布局、只读及计时边界；[固定 prework](../../experiments/34-native-ba-gates-prework-source/prototypes/v76-gdn-gates-prework-build1/gdn_gates_prework_metal.py)、[接入与 guard 差异](../../experiments/34-native-ba-gates-prework-source/prototypes/v82-native-fullqual-build1/gdn.diff)、[证据索引](../../experiments/34-native-ba-gates-prework-source/evidence-index.json)保留完整源码与实际记录。
