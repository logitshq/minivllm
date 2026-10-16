# 37：SIMD4、异步提交与下一步投机解码

截至 2026-10-16，**尚未超越 oMLX/Lithos 的完整 Pareto 前沿，没有把实验默认打开或合并 main**。本章记录第36章之后的源码和实际测试，供另一台 Mac 继续工作。

## 这次具体替换了什么

1. **GDN 的四段归约。** V44 每个线程保留32个 state 元素，再用 threadgroup 共享内存和 barrier 合并四段。V95/V99 将 SIMD lane 映射为4个分段×8个 value 列，使用四次 shuffle，仍按原顺序逐段相加；FP32表达式、每段串行归约及物理 state 布局保留。参考了 oMLX 的寄存器持有 state、SIMD 归约方向，但没有照搬会改变浮点结果的归约树。
2. **真实 CAP64 与直接异步反馈。** V101/V102 将已验证的直接异步提交路径扩展到物理 KV 容量64，并在位置加1前检查边界。CAP64 在构造引擎和任何 token 之前设置。生产模式使用两个4字节预测槽、FIFO等待，允许最多两个提交；诊断模式另存状态供检查。数学仍为 V44，没有混入 V92 的新 norm profile 或 V95 的新 kernel。
3. **下一步减少完整 target forward 次数。** V106 检查了本地真实 MTP/DFlash2 checkpoint 和 oMLX/Lithos 的调用链。V107 发现旧 GDN verify 只给出块末 S，partial rollback 却用不同 chunk 大小重算，无法提供同次 verify 的中间状态见证，因此入口保持 HOLD。新的小块递推历史事务实现还在进行中，本归档不包含或声称它已通过。

前后关系如下；三个分支不能只因分别通过就直接组合：

```text
V44 旧数学 ── V95 SIMD4局部 ── V99完整资格 ── V104原生计时
     │
     └── 直接异步提交 + CAP64：V101/V102 ── V103完整资格 ── V105 HTTP
     │
     └── DFlash2旧路径审计：V106 ── V107 HOLD ── 新事务实现进行中
```

## 实际运行与边界

| 实际运行 | 结果 | 能证明什么 |
| --- | --- | --- |
| V95 局部数值 | PASS | 128项GPU lane探针；30项o/S严格字节比较，15组输入只读 |
| V99 完整27B | PASS | 三场景96个旧token IDs；93步×132项=12,276个输出严格对齐；真实GPU CAP64，CPU规范化为256只用于离线比较 |
| V104 原生 B-only 计时 | 完成 | 三次测量中位34.397462 ms/步，29.071912 decode tok/s；单臂探索值，不是配对胜出或HTTP点 |
| V101 原生小规模探针诊断/生产 | 两次PASS | 诊断75项逐步严格检查；生产15项预测及15项终态检查 |
| V103 服务链路诊断 | PASS | 161个已交付native步、21,252项逐步输出hash对照 |
| V103 服务链路生产 | PASS | 同样161步的IDs及660项终态hash对照；不声称生产模式逐步复制全部状态 |
| V105 三引擎C1 HTTP | thermal STOP | 8/9 arms完成；第9个oMLX首次warm完成后thermal=1，违反预设nominal=0守卫，未形成完整三轮排名 |

V103 两种模式都覆盖 reset、changed prompt、fallback、提前终止及丢弃一个已提交的未来步：162次提交/完成、161次交付、1次discard、6次reset、48项S与83条直接反馈链接。目标1,349个参数身份保持，原始所有者只读检查通过。保存的是hash和metadata，不包含可重新数值计算的原始tensor。

V104 与旧 V90 的绝对速度很接近，不能把微小差值当有效收益。V105 完整保留所有已运行请求与下降点：49次HTTP请求/1,568个输出token，其中24次测量/768token，25次warm/800token；另有三次native启动warm，共52次请求/1,664token。计划本为54次HTTP加三次启动warm，没有完成计划。

第三轮native下降时，保存的17份资源样本thermal仍为0；第9臂oMLX之后才观察到1。没有频率/功率遥测，**不能把下降直接归因于热降频**。本次没有将partial点补入旧官方曲线，也没有删除慢点。

## 为什么转向投机解码

本地 target 固定为 `mlx-community/Qwen3.8-27B-nvfp4` revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，架构是 Qwen3.5。config 虽声明 MTP 层，实际1,682个tensor键中没有内嵌 MTP 权重；不能只开一个配置就声称启用。

本地已有外部 MTP adapter（23 tensors，238,933,307B）和 DFlash2（81 tensors，原始文件3,848,817,896B）；header、索引、文件长度和配对契约已核，未把这些metadata检查叫权重内容或数值验证。MTP较小，DFlash2有现成教学路径；实际常驻内存还取决于量化、激活、KV和state history。

接下来的新实现保留 batched 权重投影，对小块 GDN 使用逐token递推并保存同次调用产生的状态，接受前缀时直接提交对应状态。要先验证partial/all/zero接受、预算/EOS、reset及96个旧IDs，再测draft+verify+commit完整开销。失败仍保留，不能喂入gold未来token或放松旧ID门槛。

公平比较也要允许 oMLX 使用其实际支持的外部 MTP/DFlash2。旧 balanced .03、draft=None只是**最好已测非spec配置**，不是全局最优。Lithos的实际draft接口是DSpark；当前没有观察到配对DSpark本地权重，不能把MTP/DFlash2假装成它的兼容插件。

## 源码、证据与另一台机器接手

[归档说明](37-simd4-async-and-spec-handoff-source/README.md)、[源码映射](37-simd4-async-and-spec-handoff-source/source-map.json)、[实际结果索引](37-simd4-async-and-spec-handoff-source/evidence-index.json)、[依赖映射](37-simd4-async-and-spec-handoff-source/dependency-map.json)保存原始字节身份；以前章节已有的相同文本通过相对路径引用，新增证据压缩保存一次。

[最新256GB Mac交接补记](../handoffs/256gb-mac/PROGRESS-20261016.md)给出接手顺序。源码和harness是参考文本，旧绝对路径、原生库、GO、PID与结果不构成目标机可直接运行的资格；需要重新适配、编译、冻结和验证。M4与另一台Mac结果分开画曲线。
