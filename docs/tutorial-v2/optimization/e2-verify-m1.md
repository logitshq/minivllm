# E2：一次 verify，保留逐行 M=1 的投影数学

[六篇目录](README.md) · [上一节 E1](e1-spec-transactions.md) · [下一节 E3](e3-draft-loading.md)

事务规则正确，为什么 greedy 输出仍会改变？普通解码的 Linear 输入是 `[1,K]`，verify 则是 `[T,K]`。各行在实数数学上互相独立，但底层可按形状选择不同分块和归约树。有限精度加法不满足结合律，同一份量化权重、同一个输入，也不保证批量与逐行结果逐字节相同。

baseline 的定位方法是先完成原 verify，再对捕获的**同一输入**建立替代图。hook 继续返回原输出，替代结果不流入下一层，避免把两条已分叉的轨迹误当作同输入比较。497个真实 Linear 中，原批量与逐行参考只有146个相同、351个不同；这仍只是局部诊断，不能据此认定其他层永远无差异。

逐行参考调用 T 次 `[1,K]`。关键改动是在一次调用中表达 T 个独立的 M=1 矩阵：

```python
# 主线节选：x[T,K]，原 packed W[N,Kpack]
y = mx.quantized_matmul(
    x[:, None, :], W[None, ...], scales[None, ...], None,
    transpose=True, group_size=16, bits=4, mode="nvfp4",
).reshape(T, N)
```

权重前导维度为1，所有批次共享原权重；只增加 singleton view，不复制 T 份权重、不整份解量化。官方源码中的形状分派支持这一设计，但该轮没有观测实际 PSO，不能把静态推断写成实际 dispatch 证据。广播 M1 在497个捕获输入上全部匹配逐行参考后，才进入完整路径资格。

接入默认关闭，仅覆盖实际 target verify 的497个 Linear，包括 lm_head；普通 prefill、drafter 及其借用的 head 保持原路径，退出时用 `finally` 恢复实例标记。完整检查另建普通 A：A 自己 prefill、自己 argmax 推进，不给 B 喂未来答案。首次整模尝试在加载阶段因资源压力停止，尚未运行 B；随后使用新加载器的检查才通过三请求、每请求32个（合计96个）旧 IDs，以及49次事务的提交缓存和最后已计算 logits。局部相等、资源中止、整模通过是三个不同结论，也都不证明提速。

**理解题：** 为什么把输入改成 `[1,T,K]`，仍不能表达这里需要的逐行 M=1？

原始记录：[旧39章](../../experiments/39-verify-broadcast-m1.md)、[旧40章](../../experiments/40-m1-verify-and-bounded-loader.md)；固定实现：[广播 helper](../../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/spec/broadcast_m1_113.py)、[verify 作用域](../../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/spec/verify_broadcast_m1.py)；[接入 diff](../../code-versions/39-broadcast-m1/108-to-114-verify.diff)；证据：[完整检查摘要](../../experiments/40-source/v117-actual-summary.json)。
