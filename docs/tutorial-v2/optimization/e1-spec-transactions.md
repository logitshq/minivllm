# E1：验证一块 token，怎样只提交接受的前缀

[六篇目录](README.md) · [上一节 D3](d3-service-ownership.md) · [下一节 E2](e2-verify-m1.md)

投机解码先提出草稿，再让目标模型验证一块输入。难点不只在“接受几个”，还在于请求的缓存必须停在正确位置。本节只讨论固定模型的 greedy 路径；有限精度下的实际结果仍须单独检查，不能由实数公式推导逐字节一致。

普通解码有一个容易忽略的边界：刚交付的最后一个 token，通常还没有作为输入计算。它是下一步的 **anchor**。假设缓存停在位置 `start`，本次输入为旧 anchor 和三个草稿；目标接受前两个草稿，再给出一个纠正 token：

```text
verify 输入：A（旧 anchor）、B、C、D
交付输出：  B、C、Z
提交输入：  A、B、C       keep=3
下一 anchor：Z            新 frontier=start+3
```

这里交付了三个 token，也提交了三个输入，但二者不是同一组 token。预算或 EOS 还可能缩短交付列表，因此必须先确定真正交付的长度，再选择缓存前缀，不能直接使用草稿接受数或整块长度。

Attention 的 K/V 有 token 轴；GDN 的递归状态却没有可切片的历史轴。baseline 只留下块末状态时，无法取出“刚算完 C”的状态。关键改动是在同一次 verify 内逐 token 调用原 GDN helper，保留每次真实返回的 conv tail、S 和 producer 所有者；提交前先检查所有层的见证，再统一应用 `keep`。`keep=0` 恢复起点，attention 的未来尾部也要恢复或清零。不能另算较短前缀来冒充本次调用的见证。

原实验的块长 1/3/5/8、全部 `keep=0..T` 共21个 GPU 小例通过；只覆盖了 producer 之前触发的异常，未覆盖部分 producer 执行后的异常回滚。整模普通参考通过96个旧 IDs，投机路径仍在第5个输出失败。小例证明了有限范围的事务行为，没有证明整条投机路径正确；下一节才追查形状变化带来的数值差异。

**理解题：** 上例若输出预算只剩两个，应该提交哪些输入？为什么不能提交 Z？

原始记录：[旧38章](../../experiments/38-spec-transaction-and-qmm.md)；固定实现：[事务与真实历史](../../experiments/38-spec-transaction-and-qmm-source/candidate108/tutorial_code/miniengine/spec/transaction.py)、[调用与交付](../../experiments/38-spec-transaction-and-qmm-source/candidate108/tutorial_code/miniengine/engine/spec_runner.py)；前后差异：[runner diff](../../experiments/38-spec-transaction-and-qmm-source/prototypes/v108-dflash2-transaction-build1/engine-spec_runner.py.diff)；证据：[保存结果复核](../../experiments/38-spec-transaction-and-qmm-source/reviews/v108-independent-results-review-build1/REVIEW.md)。
