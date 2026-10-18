# B2：融合卷积时，别忘了下一步要读的 tail

[六篇目录](README.md) · [上一节 B1](b1-postconv.md) · [下一节 B3](b3-gate-fusion.md)

## 问题：怎样一起算当前输出和未来状态？

先读[第六章 cache](../06-cache.md)与[后卷积融合](b1-postconv.md)。上一种融合仍留下 concat、逐项乘法与 sum。本篇把四个 tap 的真卷积、SiLU 和新 tail 放到同一个核；目标是减少真实中间结果，而不改变递推输入。

## 找到完整的生产者边界

以下是保留形状含义的伪代码：

```python
# baseline：tail 是前三个原始投影，raw 是本步投影
xp = concatenate([tail, raw[:, None]], axis=1)
convolved = (xp * weight.T[None]).sum(axis=1)
act = silu(convolved)
new_tail = xp[:, 1:]

# 候选：同一通道拥有自己的输出和新 tail
act, new_tail = conv_silu_raw_tail(raw, tail, weight, sum_y=4)
```

四个输入依次为 `old0/old1/old2/raw`，新 tail 是 `old1/old2/raw`。它保存原始投影，不是卷积结果，更不是 SiLU 后的值；否则下一步会重复卷积或激活。原来的 tail 切片本就是 view，不能把它虚报为省掉一次独立 copy kernel。

## 不变量：四项和不只是一个加号

原计算在每个 product、每次合并后都有 BF16 舍入。这里的 `sum_y=4` 来自受测原生 reduction pipeline 和实际 product 布局，不是挑到输出相等才选出的参数。该拓扑还有各 partial 与正零相加的步骤；新核保留其升序合并和 typed BF16 SiLU。改成 FP32 连加后只落一次 BF16，不能沿用这份证明。

新核返回独立的 act 和 tail，不原地改 raw、旧 tail 或权重。投影、拆头、门、状态递推与输出 norm 继续用原路径，因此数值检查既要看激活，也要看新 tail 和随后产生的 S。

## 从局部结果走到组合验证

早期84项严格比较通过，但 raw/tail 来自合成输入；使用真实权重不让它自动成为真实模型激活。局部 host-wall 减少约12.21%后，才补真实整模与生命周期检查。后续 HTTP 实验还同时加入函数图复用，吞吐改善约0.76%；不能把组合结果归因于卷积单项，也未据此宣布外部前沿获胜。首次生命周期比较误用 B1 固定流作 B2 gold 的失败仍保留。

**检查理解：** 若 act 正确但 new_tail 保存成 act，为何单步测试可能通过，而第二步就出错？

## 回到完整记录

[旧21：规约、局部资格与 micro](../../experiments/21-gdn-conv-fusion.md)、[旧22：组合、生命周期与 HTTP](../../experiments/22-combined-decode-screen.md)保留全部阶段边界。[固定 Python/Metal 原型](../../experiments/21-gdn-conv-source/prototype/conv_silu_raw_tail.py)、[Metal 运算顺序](../../experiments/21-gdn-conv-source/prototype/conv_silu_raw_tail.metal)、[原规约契约](../../experiments/21-gdn-conv-source/reference/conv-reduction-contract.md)、[组合源码与证据索引](../../experiments/22-combined-decode-source/README.md)供逐项核对；原文内含 baseline 与候选的完整对应关系。
