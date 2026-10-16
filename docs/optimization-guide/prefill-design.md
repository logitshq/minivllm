# E010设计笔记：先守住矩阵的归约，再扩大batch

2026-10-16；MLX0.32.3、Apple M5 Ultra、Qwen3.8-27B NVFP4。
正确性和已通过的HTTP资格见[实验账本](experiments.md)。本文解释实现；后文的
混合长度分桶仍是待验证设计，没有计入已获得的性能。

## 为什么不能直接把B和T乘起来

单请求的量化线性层输入是`[T,K]`。MLX遇到连续`[B,T,K]`和二维权重时，可能
把它当成`[B*T,K]`；Python形状有三维不代表GPU仍然把每个请求视为独立矩阵。

固定版本的`quantized.cpp`有两个相关选择：

- 单矩阵转置QMM先计算split-K，部分和先写到BF16中间buffer，再求和。
- NAX转置QMM在`M<=32`时用BM32，超过32用BM64。

因此从T19改成B×19可能同时改变分区、部分和舍入与tile。数学上相同的矩阵乘法
不自动等于同一浮点执行过程。不能只用最终32个token相同证明无损。

## 只改变独立batch维

成本假设也来自执行顺序：原先先为请求A跑完64层，再为B跑完64层；成组路径
先在同一层处理A/B，再进入下一层。这样减少重复的主机构图与提交，并给同层权重
的缓存复用创造机会。源代码能证明调用顺序变化，不能替代带宽counter对收益来源的归因。

候选把packed weight和scale广播到`[B,N,Kpacked]`和`[B,N,K/group]`，激活
保持`[B,T,K]`。权重增加逻辑batch维后，固定MLX路径保留`M=T`。

| 检查 | 目的 |
|---|---|
| 原`ceil(N/32)*ceil(T/32)>256` | 证明`floor(512/threadgroups)<=1`，原调用无需split-K |
| K>4096、T≥14 | 避开本机已固定的QMV切换区间 |
| K%64=0 | 满足固定版本NAX分派条件 |
| BF16、NVFP4 group16/bits4、已知packed元数据 | 限定数值路径 |
| 保留M=T | 批处理前后使用同一M相关tile |

不满足的投影按原来的每请求T调用，包括小输出gate和last-only lm_head。
MLX的广播视图与matrix-contiguous检查不会要求本实现主动复制B份权重；这不等于
已经用硬件counter证明权重只从DRAM读一次。这里测量的是完整prefill和端到端请求。

核心没有新kernel、通用算子注册表或缓存类型；复用已有成组GDN/attention路径。
GDN只合并独立请求/头维，不把不同请求的token串成同一时间轴。runner仍然只收
fresh、完整、同长度、未记录状态的prefill；spec、capture和继续分块不进入该路径。
超出B2–32、T1–512或B×T4096以及未知硬件/版本，继续原路径。

## 从实际安装的MLX-LM继续学什么

本机被测oMLX环境中的MLX-LM `generate.py:1171–1211`显式收集长度，右侧padding，
在缓存上执行`prepare(lengths, right_padding)`，最后`finalize()`。
`models/cache.py:674–694`的ArraysCache保留lengths并产生有效位置mask。
`models/qwen3_5.py:157–197`会屏蔽padding投影、按真实末位置提取卷积tail，
把mask交给递推更新。这些是实际安装源码的能力；这里没有增加一次oMLX服务跑分，
也不由类的存在推断所有配置都走该路线。

值得保留的设计是**长度、状态末位置和padding必须一起管理**。泛化到不同长度时，
只给attention加mask不够：GDN状态和卷积tail也必须停在每条请求的真实末尾。
我们的下一步应先做有限长度桶的原型，而不是复制整套extend/filter/split/cache抽象。

在保持逐位一致的路线中，长度桶还必须保留QMV/QMM选择、split-K、NAX tile和GDN
chunk边界。例如17–32长度桶是可以研究的范围，但padding后的最终状态、signed-zero、
卷积tail以及真实最后一个logit都必须实测；这里尚未实现或证明该桶。

当前同长度优化不会等待更多请求来凑batch，也没有prefix复用。相同prompt的基准
有较高命中率，不能把其收益直接套到随机长度或真实到达流。下一项性能实验应固定
混合长度/到达trace，同时报告命中率、padding工作量、TTFT和尾延迟。

安装文件SHA256（本机`.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/mlx_lm/`）：

```text
generate.py        4a3bf57f5679dac73791f069268866ab5b9336e58582d436b31133984e4b6971
models/cache.py    c96cb169fd5e0ec4584a162dd61dc62ab08ccdca03a9850f8a491ce6809145e6
models/qwen3_5.py  10c59a71d8d739b8cdff38c7e9c7345deb9a9688e821ac37cb774d3086cac42e
```

MLX分派依据：[v0.32.3 quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp)。
这些源码推论由逐层、完整cache、连续decode和真实调度验证补足；耗时必须另测。
