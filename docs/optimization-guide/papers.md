# 新论文 → 机制 → 我们的实验

检索和核对日期：2026-10-16。下列日期是arXiv首次公开或修订日期，不等于
同行评审发表日期。只用作者论文/官方实现作技术依据；论文自报速度不视为
本机复现。以下“落地点”是我们的推论与计划。

| 论文 | 日期/阅读范围 | 核心机制 | 对M5/本模型的取舍 |
|---|---|---|---|
| [SparseDecoding](https://arxiv.org/html/2610.12327v1) | 2026-10-08 v1；摘要、方法与kernel设计 | 用自生成decode轨迹校准剪枝；bitmask和固定步数的N:M SpMV降低索引开销 | 学“针对真实生成校准”和“减少实际字节”；A100 kernel不能直接移植。剪枝改变权重，应独立做质量—速度曲线，不能混入exact优化 |
| [JetSpec](https://arxiv.org/abs/2606.18394v5) | 首次2026-06-16；v5为2026-10-03；摘要/方法入口 | 一次并行draft产生有分支因果条件的候选树，避免draft深度线性增加成本 | 本模型需要匹配的训练头；先测现有DFlash2的draft/verify耗时与接受长度，不能把别的Qwen/H100结果外推 |
| [ASPIRE](https://arxiv.org/html/2609.17943v1) | 2026-09-16；摘要、动机、调度结构；作者标注COLM 2026接收 | 请求独立选择draft/verify时机，在混合forward中复用batch，估计接受率和验证成本 | 学每请求成本模型；本模型含GDN，稀疏attention草稿与回滚不能直接套用。先做单请求自适应block，再考虑混合batch |
| [BaseRT：M5 Neural Accelerators](https://arxiv.org/html/2607.19438v1) | 2026-07-21；§2–4 | compute-bound GEMM/prefill走Metal4矩阵单元，单行带宽受限decode保留专用路径 | 对固定27B测实际M/N/K及分界。NAX是GPU内矩阵单元，别与独立ANE混为一谈；大batch decode也可能变成计算受限 |
| [BaseRT：Native Metal](https://arxiv.org/html/2607.00501v1) | 2026-07-01；§3、限制 | 预分配scratch、专用融合、降低dispatch与CPU/GPU往返 | 借鉴固定生命周期和执行计划。其该版限制包含无continuous batching；我们必须保留动态请求正确性，不能直接采用单流结论 |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) | 2026-03-05；摘要筛选 | 针对Blackwell矩阵/非矩阵单元不对称扩展重设计流水 | 这是较早的2026背景文献。可学瓶颈迁移和减少非matmul工作；不移植CUDA专属机制，也不假设短上下文GDN模型主要耗时在attention |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | 2025基础文献，非新论文；摘要/官方论文入口 | 面向serving的attention内核与运行时调度 | 长KV优化阶段再深入；先确认当前KV扫描占比，不为通用性引入整个框架 |

## 把阅读变成可执行问题

**优先级P0：状态与内存复用。** BaseRT的固定buffer思想与oMLX allocator源码
共同提示要把“分配/释放”从算术中拆开测。已有一次批量retirement是成功实例。
本轮单变量配对2/16 GiB池，记录active/cache/peak；即使池收益大，也不等于
证明原生arena或ICB已实现。原生固定状态布局是后续更大的架构实验。

**优先级P1：设备端执行计划。** 先只支持固定batch和greedy，明确双槽GDN、
KV写入、设备token、EOS/取消输出边界。测“CPU提交间隙”可消除的上限；
若贡献低，不为模仿Lithos而重写全部runtime。若贡献高，逐步加入动态batch。

**优先级P1：prefill与decode分开选矩阵路线。** 大M走适合它的矩阵单元，小M
优化真实权重读取和解量化。已有投影打包在某些M触发不同归约，是必须保存的
反例。需要先量kernel选择与数值契约，再扩大形状范围。

**优先级P2：自适应投机。** 每个接受token的成本约为
`(draft耗时 + verify耗时 + 回滚/调度耗时) / 实际产出token数`。
用真实接受率调block长度；采集长输出和不同任务，避免固定32token样例偏置。
先验证已有drafter，不因论文有大倍数就先下载不兼容模型。

**独立研究线：压缩。** SparseDecoding提示decode轨迹比静态teacher forcing
更贴合实际。若做剪枝/量化，固定校准集、留出任务和权重版本，记录质量下降；
同权重无损前沿与近似模型前沿各自标清。此轮没有改权重，也未复现其性能。

## 阅读日志约定

每次新增：精确URL/版本/日期、实际读过的章节、硬件/模型前提、作者自报结论、
本机能借用的机制、不能直接套用的部分、关联实验ID。未读全文就标注摘要筛选。
复现失败、数值不等和无收益都进入账本，不把“读过”写成“已实现”。

## 2026-10-17阅读追加

- [TreeWY v1](https://arxiv.org/html/2608.20961v1)，2026-08-21；读§1–5。
  用三角求解验证GDN草稿树，接受后重建状态，避免每节点完整快照。其B200实验主要
  在内存限制并发时获益，C1表格有回退；§4明确token流不是bit-identical。
  推论：我们先量draft/verify/commit和状态存储，建立同次verify的前缀状态合同，
  再考虑重建。M4正在做literal-prefix-history，不能直接把本论文改写混入该资格。
- [MSInfer GDN技术报告 v1](https://arxiv.org/html/2607.16831v1)，2026-07-18；
  读引言、§2–3.2。把decode分派/运行时和prefill chunk结构区别处理；报告还给出
  测量环境改变后相对强基线结论反转的实例。推论：E011只拆出大batch图复用，
  不能把局部核或比赛相对简单参考的加速当本机端到端收益。未移植CUDA代码。

第一轮检索覆盖2026年10月的GDN投机和Apple Silicon关键词，未找到可确认直接适用的新条目；
随后扩大到并行草稿，新增下方10月7日EDR与9月23日DPara笔记。检索日期不替代发布日期。

- [AdaFlash v1](https://arxiv.org/html/2607.19223v1)，2026-07-21；读摘要、§2及§3.2–3.3。
  根据预测接受长度减少无效verify，训练出的长度头与在线蒸馏是其实现前提。
  本机没有该头，不能只改block长度就称复现。E012先拆draft中的主机选路成本；
  后续以实测draft/verify/commit与接受数建立简单成本模型，独立评估自适应长度。

- [DPara v1](https://arxiv.org/html/2609.27396v1)，2026-09-23；读§1–3和附录D。
  把可提前计算的多接受边界草稿主干与等待bonus的轻量头分离；需要M-DFlash微调权重。
  主要结果用了额外draft GPU；单GPU在batch8/16时相对DSpark的优势消失。
  本机推论：先量关键路径与资源竞争，不把两个stream当免费重叠；现有DFlash2不能直接
  假装支持无特征多anchor。本轮未实现DPara，E012只减少原有选择器同步。
- [Expected Decoding Rounds v1](https://arxiv.org/html/2610.10411v1)，2026-10-07；
  摘要与§1筛选。将跨轮接受依赖建成Markov reward过程，以期望轮数训练/评估并行草稿，
  提供基于共享目标rollout的离线比较。推论：后续不要只看单block接受率，也记录完整生成
  的轮数；本机不把轮数减少等同wall time减少。未读证明，未训练或复现作者结果。
- [DSpark v1](https://arxiv.org/html/2607.05147v1)，2026-07-06；读§3.1–3.2。
  小型顺序头补并行草稿依赖，校准的前缀生存概率与本机SPS成本表决定验证长度。
  §3.2说明提前停止的因果要求，以及全局最优依赖平滑/单峰成本曲线。
  推论：不在看到未来candidate后回选长度，也不把未校准confidence直接当概率。
  本机先保留fixed block成本与真实接受trace；暂无该checkpoint的校准置信头。
