# B1：把卷积后的准备工作移进状态核

[六篇目录](README.md) · [上一节 A2](a2-controlled-comparison.md) · [下一节 B2](b2-conv-fusion.md)

## 问题：少一个中间数组，就一定少做计算吗？

先读[第四章 GDN](../04-gdn.md)、[第九章 Metal](../09-metal.md)和[第十五章门计算](../15-把GDN的小门计算一起编译.md)。本篇只移动卷积之后的 SiLU、拆头与 Q/K 复制；卷积、门、输出 norm 都不在替换范围内。

## 从准备好 Q/K/V，改成按需读取

下面是省略 guard 的教学伪代码，不是可直接执行的入口：

```python
# baseline：u 是卷积结果，BF16[1,10240]
q, k, v = split_heads(silu(u))   # Q/K 各重复三次
out, new_S = state_kernel(q, k, v, g, beta, S)

# 候选：输入原 u，在状态核内完成激活和头号映射
out, new_S = postconv_state_kernel(u, g, beta, S)
```

这个模型有16个 Q/K 头、48个 value 头。原来把每个 Q/K 头连续复制三份；现在 value 头 h 直接读取 Q/K 头 `h//3`。例如 h=7 对应头2。改变的是寻址，不是把三个头的状态合并：各头仍有自己的 S、V 和门。

## 不变量：激活的中间舍入也是接口

原路径先得到 BF16 sigmoid，再做 BF16 乘法，之后转 FP32 参与递推。新核必须保留这两个 BF16 边界，不能换成 FP32 SiLU 算完才 cast。衰减后预测、四段归约和状态更新顺序不变；conv tail 仍保存卷积前输入。

融合也有代价。每个 value 头拆成四个列 tile，多个 tile 会重新激活同一 Q/K。按源码逻辑计数，一层激活元素从10,240增至55,296；省掉外部数组与调用，并不等于每项工作都减少。它不是硬件指令计数。

## 结果教会了什么

该阶段局部 host-wall 降低约13.96%，完整模型受测输出一致；但 HTTP 吞吐只改善约0.632%，未达到当时预先声明的1%采用门，实验未采纳。局部比例不能外推为整模型比例。新示例入口当时仅做 CPU 检查，也不能借已有模型资格宣称示例已经跑通。

**检查理解：** 为什么 h=6、7、8 能读同一份 Q/K，却不能共用同一份 S？如果把卷积也移入这个核，为什么需要新的数值资格？

## 回到完整记录

[旧16完整章（固定 tag）](https://github.com/logitshq/minivllm/blob/tutorial-16-experiment-v1/docs/tutorial-v2/16-把GDN的SiLU和拆头并入状态更新.md)、[现有阅读入口](../16-实验篇从哪里开始.md)保留全部讲解、开关、fallback、计数与结果。[固定核源码](https://github.com/logitshq/minivllm/blob/tutorial-16-experiment-code/tutorial_code/miniengine/kernels/gdn_postconv_metal.py)和[原结果索引](https://github.com/logitshq/minivllm/blob/tutorial-16-experiment-v1/tutorial_code/examples/chapter16-gdn-fusion-results.json)可逐项核对。

```bash
# 只读查看真实差异，不切换当前目录
git diff tutorial-15-v1 tutorial-16-experiment-code -- tutorial_code/miniengine
```
