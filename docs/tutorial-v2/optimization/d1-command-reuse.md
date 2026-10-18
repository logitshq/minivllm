# D1：复用命令，省下每步重新编码

[六篇目录](README.md) · [上一节 C3](c3-state-buffers.md) · [下一节 D2](d2-device-feedback.md)

前置：[C3 状态与双缓冲](c3-state-buffers.md)。输入、输出与两套状态区域已固定，CPU却仍在每个token重新设置pipeline、buffer和dispatch。问题是：**计算不变，能不能只写一次这串命令？**

## 从逐次编码到执行命令表

下面是流程伪代码，不是可直接运行的接口：

```text
之前：每步 encode(所有dispatch及同步) → commit → wait
之后：构建时 encode(偶数bank ICB、奇数bank ICB)
      每步 declare_resources → execute(ICB[bank]) → commit → wait
```

ICB是Metal的间接命令缓冲，不是把Python循环换个名字。固定的两套表分别绑定两组状态；每步选对表，仍执行原来的kernel。实际调用节选见[原生源码](../../experiments/27-native-icb-pingpong-source/prototypes/v38-native-icb-build1/replay.mm)：

```objective-c
[e executeCommandsInBuffer:I().banks[PP().bank]
                 withRange:NSMakeRange(0,I().dispatches)];
```

这行之前还须声明读写资源。权重和常量必须存活，原函数、线程组与绑定范围不能换；不支持ICB的pipeline要按原descriptor重建并重新验证。原来的短常量不能继续用`setBytes`，因此保存进固定常量buffer。同步标记落到下一条命令的`setBarrier`，尾部依赖也不能漏。

## 组合算子，是另一层变化

命令复用减少编码，算子融合才可能减少命令。卷积融合已输出SiLU后的值，接norm预处理时就不能再激活一次；[实际差异](../../experiments/28-native-gdn-composition-source/prototypes/v44-gdn-activated-normreuse-build1/prework.diff)删掉的是这次重复激活。图变了，应重新捕获、核对状态绑定，不能继续套旧表；Metal参数index也不能按Python输入个数猜。

整步`mx.compile`又是第三层：先编译纯计算，再捕获命令。[纯函数源码](../../experiments/33-native-range-compiled-step-source/prototypes/v74-compiled-step-icb-qual-build1/functional_step.py)显式接收token和缓存；cold求值在capture外。它不是被测oMLX的整步机制。

## 结果告诉了我们什么

历史记录中，graph编码中位从3.195降至0.070ms，GPU包络仍约35.6ms；两者重叠，不能相加。独立buffer克隆分支没有收益；实际捕获没有资源级barrier，不能把公共arena当作已证实的慢因。组合算子及整步编译各通过有限整模资格，但单臂计时不能证明HTTP获胜。另一次ICB范围微测把每SIMD行数加倍，控制通过仍略慢：命令可复用，不代表任意kernel替换都更快。

**理解题：** 命令数变少，为什么请求延迟可能不降？请分别考虑GPU计算、依赖等待和主机编码是否处在关键路径。

原始推导与详细证据：[旧27章](../../experiments/27-native-icb-pingpong.md)、[旧28章](../../experiments/28-native-gdn-composition.md)、[旧33章](../../experiments/33-native-range-compiled-step.md)。下一篇：[设备反馈](d2-device-feedback.md)。
