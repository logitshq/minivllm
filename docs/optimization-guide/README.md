# 从瓶颈到实现：M5 Ultra 推理优化攻略

更新：2026-10-18。目标机器是 M5 Ultra / 80 GPU cores / 256 GiB；目标模型
主线为 `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef1`，第二重点为 GLM-5.3-Flash；后者尚未完成本机资格。代码在
`tutorial_code/miniengine`。这是可复查的实验手册，新的结果按编号追加。

入口：[论文与设计笔记](papers.md) · [当前对手兼容性矩阵](research/2026-10-17-frontier-candidates.md) · [新近状态与运行时论文](research/2026-10-17-state-and-runtime-papers.md) · [逐步实验账本](experiments.md) ·
[Qwen3.8 / GLM-5.3-Flash 模型路线](model-roadmap.md) ·
[D512 三引擎吞吐—延迟对比（v15，2026-10-18）](progress/2026-10-17-v15/README.md) ·
[历史 cap32 分面图](progress/2026-10-17-v5/README.md) ·
[并发 throughput–latency Pareto](progress/2026-10-17-v3/README.md) ·
[最新官方 MTP / DSpark 同轮 HTTP 对照](progress/2026-10-17-v4/README.md) ·
[NAX prefill设计](prefill-design.md) ·
[E011逐次实验教程](tutorials/011-large-batch-graphs.md) ·
[E012设备端草稿选路](tutorials/012-device-draft-selector.md) ·
[E013草稿成本分解](tutorials/013-draft-cost-profile.md) ·
[E014有界草稿KV](tutorials/014-bounded-draft-kv.md) ·
[E015官方MTP资格](tutorials/015-official-mtp.md) ·
[E016草稿图复用](tutorials/016-draft-backbone-graph.md) ·
[E017 Lithos DSpark资格](tutorials/017-lithos-dspark.md) ·
[E018短块GDN状态驻留](tutorials/018-persistent-gdn-verify.md) ·
[E019只提取新增hidden history](tutorials/019-drafter-history-tail.md) ·
[E020官方投机模式 HTTP 对照](tutorials/020-external-spec-http.md) ·
[E022 HTTP 发送延迟预注册](tutorials/022-native-transport.md) ·
[E023 长生成吞吐—延迟实测](tutorials/023-realistic-throughput-latency.md) ·
[E024 图编译与attention长度隔离](tutorials/024-decode-graph-dispatch.md) ·
[E025 相同 split-K NAX 筛选与拒绝](tutorials/025-same-split-nax.md) ·
[E029 GDN 长路径预处理融合与37请求验证](tutorials/029-unbound-gdn-prework.md) ·
[C1差距定位与下一项可证伪实验](tutorials/030-c1-gap-diagnosis.md) ·
[版本化进展图与逐点数据](progress/README.md) ·
[M4 Max 交接](../handoffs/m5-ultra-to-m4-max.md) ·
[最新分组prefill与Pareto结果](../experiments/m5-ultra-256gb/prefill-nax-v2/README.md) ·
[上一轮完整 Pareto 结果](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-report.txt)。

最新D512图包含38个进程观察、14组独立配置/并发汇总。按各配置进程指标的中位数比较，GDN ON的C1为47.86tok/s、完整POST→DONE p50为10.69s，吞吐仍低于oMLX约5.75%；Native的C8前沿保留E024 ON的170.62tok/s。**E024与GDN是分开的开关配置，组合未验证**，也不合并两轮OFF或历史N=1。p50/p95联合前沿仍为oMLX C1与E024 C8；C2–7未知。实际更新日期为10月18日，v15目录保留历史命名。

## 1. 先算每个 token 为哪些字节和等待付钱

一个 decode step 的近似成本是

```
T_step ≳ max(实际读取字节 / 有效带宽, 实际运算 / 有效算力)
         + 无法重叠的提交、同步、分配、状态整理时间
吞吐 ≈ 本步实际输出 token 数 / T_step
```

这只是定位工具；缓存命中、不同 kernel 的串行依赖和重叠会改变实际时间。
不要把厂商峰值带宽直接当作模型有效带宽。先量活跃张量、缓存池、请求
长度与 batch，再分别测 prefill、稳态 decode、请求完成和新请求进入。

对本模型，48 个 GDN 层的 FP32 状态共有
`48 × B × 48 × 128 × 128 × 4` 字节：每条序列 **144 MiB**，B64 为
**9 GiB**。若每步完整读取并写回一次，就有约18 GiB的状态流量，尚未计入
模型权重和 attention KV。batch 可摊薄权重读取，却不能摊薄每条序列的状态。
因此“batch越大越快”和“decode永远只受权重带宽限制”都不成立。

正确性是候选进入性能比较的硬前提，不是可与速度交换的优化轴。主 Pareto 固定为
**横轴完整请求延迟（ms，越低越好）、纵轴整机输出吞吐（tok/s，越高越好）**。
同模型、同量化、同负载分别画图；不把局部 kernel 倍数或 GLM 与 Qwen 的点混成一条前沿。
延迟直接从 HTTP request start 到 SSE DONE 重算，不能用旧交付速率取倒数代替。
用户看到的成本还包括排队和prefill。HTTP实验同时保存：

- TTFT、完整请求耗时、完整并发组窗口和实际输出数。
- 每用户首文本之后的交付速率，以及整机输出吞吐。这两个指标组成旧图的坐标。
- 峰值活跃内存、allocator缓存池、系统压力、热状态、CPU/GPU流水情况。

单 kernel、整模型前向、同步引擎、HTTP 是四层不同证据；分别标注，不能混画。

## 2. 每个大方向先回答一个可证伪的问题

| 方向 | 要移除的成本 | 最小实现 | 什么时候值得做 |
|---|---|---|---|
| 固定执行计划 | 每token建图/提交/同步 | 显式输入输出状态的有限图；随后才考虑原生Metal计划 | 主机空隙占比足够高 |
| 状态生命周期 | 重复gather、复制、分配 | 同一步退出只压缩一次；测allocator工作集 | 高并发或频繁加入/退出 |
| 有效融合 | 中间数组、重复归一化、kernel启动 | 共享输入的有限投影与GDN预处理 | 完整前向收益超过噪声 |
| 矩阵路线 | GEMV/GEMM与硬件不匹配 | 按M/N/K、dtype、量化格式选路径 | 先确认实际调用和归约 |
| prefill | 长prompt矩阵计算、串行小prefill | 分块预算、兼容长度批处理、NAX GEMM | TTFT或prefill占主要时间 |
| 投机解码 | 每个接受token重读目标权重 | 先自适应现有draft长度，再考虑批量verify | 草稿+验证成本低于节省 |
| 长上下文 | 无效KV扫描/容量搬运 | 有界长度分桶、split-KV、再评估paged布局 | 长上下文实测由KV主导 |
| 近似压缩（主线之外） | 实际权重/状态读取量 | 单独的剪枝或量化实验线 | 不进入本轮同权重无损 Pareto |

一次只改一个主因。微基准快而模型不快，先用 Amdahl 定律解释：
若优化部分占比为 `f`、局部加速 `s`，整体上限约 `1/(1-f+f/s)`。
不要因为局部漂亮就扩充维护成本。

### 模块与实验开关

先按实际成本和状态所有权划边界，再写代码。每个实验只有一个独立开关和明确输入输出；
对照路径随时可用，启用状态写入运行identity。不要用隐藏环境变量自动叠加多个实验。

| 模块 | 独立控制 | 组合约束 |
|---|---|---|
| decode图复用 | `compile_decode` | 数学、shape与显式状态分别验证 |
| decode投影/预处理融合 | `fuse_decode` | 依赖graph，但不能用graph收益替代融合收益 |
| 大batch graph原型E011 | 隔离candidate类 | 已拒绝，生产B1–8边界不扩大 |
| draft选路E012 | `draft_device_selection=False/True`，默认OFF | 三进程模块资格通过；只换选路，不动draft长度、verify与commit |
| 有界draft KV E014 | `draft_bounded_kv=False/True`，默认OFF | 实际核心2944严格检查通过，但一个16 GiB短点未过收益门，保留实验模块 |
| 投机验证/提交 | 保留独立数值资格 | 组合前检查接受、拒绝、EOS、预算和cache owner |

先测模块，再测完整请求；实验独立通过不自动意味着组合通过。

## 3. 从 oMLX、Lithos 拿走什么

**oMLX**：学设备上一拍ahead、减少事件循环交接、GDN真卷积预处理和
allocator复用。我们的同步worker已经有ahead；再移植executor不会自动变快。
需核对被测wheel的实际文件，源码checkout的SHA不能冒充wheel身份。
前缀KV缓存和MLX allocator缓存池是两件事；关闭前缀缓存不等于关闭buffer复用。
详见[安装源码审读](../experiments/20-donor-source/omlx/REVIEW.md)。

**Lithos**：学固定buffer生命周期、arena复用、预编码ICB和设备端推进。
这些机制必须与EOS、取消、ring、barrier和完成栅栏一起设计。现有`mx.compile`
只复用图，不能宣称实现了Metal ICB。先做一个固定模型、固定batch的原生
decode原型，再决定是否替换热路径；不要先写跨模型compiler或插件注册体系。
详见[执行路径审读](../experiments/20-donor-source/lithos/REVIEW.md)。

“简单”以**生产路径和不变量数量**衡量：每个存活请求有唯一状态owner；
提交给GPU的旧状态保持有效；停止token不会穿过输出边界；不支持的形状有明确
fallback。benchmark可以详细，热路径应短。借用代码时固定来源版本并保留许可证。

## 4. 正确性不是只比一句输出

按改动性质选择检查：

1. 保持算术的融合：检查中间dtype、舍入、规约顺序；逐位比较logits、全部KV、
   GDN状态和原始卷积tail；加非连续输入、不同尺度和连续多步。
2. 调度/内存优化：检查实际原始token IDs、逐步存活状态、交错结束、新请求加入、
   EOS/stop、ahead失效、空batch和请求ID复用；检查不可变模型参数。
3. 改变归约顺序：单独定义误差与质量门，不在看完差异后放宽“exact”标准。
4. 投机解码：保留目标分布的接受/拒绝及回滚协议；混合GDN状态也要回滚，不能
   只移动attention offset。吞吐分母包括draft、verify和失败工作。
5. 剪枝/状态低精度：改变了模型，独立报告困惑度、代码/数学/中文与长生成质量。
   不把它算进同权重的无损加速曲线。

## 5. 一次实验怎样落盘

先写 `plan.json`：假设、来源、源文件SHA、模型revision、环境、变量、对照、
顺序、预热、输出长度、正确性门、收益门和停止条件。再运行，避免按速度挑样本。

先用小范围筛选排除错误；有价值的候选再做至少三对独立进程，交替先后顺序。
预热与正式结果分别保留；冷启动/编译也记录。发布HTTP结论时保留所有请求和
失败重跑，不删慢样本，报告波动及同并发比较。资源异常的失败留档，释放自己的
进程；不动其他工作线程的服务。

每一步记录在[实验账本](experiments.md)，一项实验至少包含：

```
ID / 日期 / 基线commit / 论文或源码来源
假设与可预见上限 / 唯一变量 / 预注册门槛
实现diff与源文件SHA / 精确命令 / 原始数据位置
正确性 / 性能 / 内存与热状态 / 失败与重跑
采用、拒绝或继续资格验证 / 理由 / 下一个可证伪问题
```

代码和可复核汇总进Git；模型、venv、大量原始事件留在本机并给出哈希清单。
禁止把本机绝对路径写成跨机器可直接执行的承诺。M5推
`experiment/m5-ultra-256gb`，同步M4分支后再合并；跨机器收益必须独立测。

### 持续保存进展图

按2026-10-17的用户要求，每轮形成新的性能结论、通过接入资格或否决重要方向时，
刷新并保存静态进展图。图、机器可读数据、生成脚本、来源hash和对应commit一并归档；
旧版本保留。优先画同条件 throughput–latency Pareto、各fresh process的实际点，以及采用/拒绝/待验证时间线。
模块加速与HTTP收益分开，不能乘加为虚构的整体加速；数值资格没有速度时显示“未计时”。
点的范围是观察到的波动，不能冒充置信区间。新图必须渲染检查后再展示。
图集保存在`progress/`，新增结果从这里链接到对应版本。

## 6. 扩大目标范围的次序

已完成的Pareto实验是19输入/32输出；下一阶段按下表扩展，而不是只优化这一点。
这些是**待测矩阵**，不是已获得的性能结论。

| 场景 | 输入/输出token | 并发 | 主指标 |
|---|---|---|---|
| 旧基准回归 | 19/32 | 1、8、32、64 | 双坐标、TTFT、内存 |
| 真实短对话 | 256/128 | 1、8、32 | TTFT、p50/p95间隔、整请求耗时 |
| 代码/长输出 | 1024/512 | 1、8 | 接受长度、完整耗时、质量 |
| 长上下文 | 8192/128，先确认内存 | 1、4、8 | KV流量、TTFT、峰值内存 |
| 混合到达 | 不同prompt/输出长度，固定到达trace | 逐级增压 | p95排队、尾延迟、公平性 |

allocator配对未过收益门，保持2 GiB。E010的M5成组prefill/NAX已过采用门：
C8/C32/C64配对吞吐约+14.7%/+12.3%/+14.5%，只改两个核心文件。
下一项先测真实混合长度/到达trace与长度分桶；其余固定buffer/设备执行计划、
自适应投机按实际成本占比排序。
对可能带来数量级变化的方案大胆建小原型；对
不足以影响端到端结果的小kernel变化及时止损。

2026-10-17：[E024四组诊断教程](tutorials/024-decode-graph-dispatch.md)确认在B1/used257同长度编译全部逐位一致，改变补齐长度重现失败。无新速度点；下一步保持SDPA归约路径再验证。

2026-10-18：[E030组合教程](tutorials/031-gdn-graph-composition.md)的42次参数诊断/37请求整模型检查、独立资源与恢复审计全部通过。实际C1/C8 HTTP捕获/回放已启动；无新增速度点，当前Pareto仍见v15。
