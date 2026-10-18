# 47：再读五篇近期论文，把想法缩成一次可验证的改动

阅读日期：2026-10-17。root 与三个 agent 分读后对照原文和本方代码。本章是研究记录；尚未修改引擎、运行 GPU 或新增 Pareto 点。

第46章确定了权重复用、混合请求调度和投机成本三个方向。这次补读9月底至10月的工作，最有用的增量是：**prefill 的频率和大小可以分别控制；检索候选可以减少 draft 模型工作，但仍须支付验证成本。** 以下实验都是本方提出的待测方案，论文成绩不代表我们的收益。

## 1. 本次读了什么

| 论文与版本 | 本次读到的机制 | 作者实现的核对范围 |
| --- | --- | --- |
| [Apple Silicon 优化评估](https://research.macpaw.com/publications/llm-optimizations-apple-silicon)，作者页09-30 | 小矩阵权重复用；显式缓存和投机的反例 | 已读全文附录和实现节选；未核到完整公开实验仓库 |
| [JustFit v3](https://arxiv.org/html/2609.17475v3)，09-30修订，09-15首发 | prefill cadence、组件owner、执行模式切换 | 已读作者 `justfit-repro-v4`，固定提交 `3cc6e0cf41f1d0ed939a9702926bd62642a3fef8` |
| [AgSpec v2](https://arxiv.org/html/2610.01108v2)，10-06修订，10-01首发 | 会话/工作区检索，以及适配实际输出格式的索引 | 已读原文；未确认作者代码发布 |
| [TLAR v1](https://arxiv.org/html/2610.07350v1)，10-05 | 从已生成轨迹找候选，用反馈控制检索 | 已读原文；[作者仓库链接](https://github.com/YUYANG0915/tlr)本次读取失败，未完成代码核对 |
| [QBA v1](https://arxiv.org/html/2610.17795v1)，10-13 | 长请求晋升的限额与欠账 | 已读作者模拟器，固定提交 `e51873c85cb62a0095567577fc11bf4e1b7c7cde`；不是GPU服务实测 |

JustFit 和 AgSpec 也是 M5 已提过的线索，本次补独立阅读。原文、代码和本方推论分开记录；读不到实现的论文不标成“已审作者代码”。

## 2. 先理解一个更小的调度改动：多久允许做一次 prefill

我们的[现有 scheduler](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/scheduler.py)已经有 continuous batching 和分块 prefill：先安排 decode，再用剩余 token 配额处理 prompt。新问题是：即使每次只处理一小块，**是否每个 decode 步都需要插入它？**

JustFit v3 §3.3 将两个参数分开：N 控制活跃 decode forward 之间的 prefill 机会，P 控制每次 prefill 的大小。作者[启动脚本](https://github.com/YuhuaBillChen/justfit-mlx/blob/3cc6e0cf41f1d0ed939a9702926bd62642a3fef8/examples/justfit/run_server.sh#L53)采用 N=4/P=64；库默认则是 N=1/P=0。[实际执行代码](https://github.com/YuhuaBillChen/justfit-mlx/blob/3cc6e0cf41f1d0ed939a9702926bd62642a3fef8/mlx_vlm/generate/ar.py#L4234)还检查活跃请求和容量，不能把脚本配置说成任何场景下的最优值。

**本方现有行为与拟替换行为，简化示意：** D 表示一次 decode forward；P 表示一次符合预算和容量约束的 prefill 机会，二者宽度不代表耗时。

```text
原来 N=1：D P  D P  D P  D P
候选 N=4：D    D    D    D P
                       ↑ 关键：改变插入频率

另一个独立实验：保持 N=1，只减小每个 P 的 token 上限。
```

这样可能减少正在生成的请求被 prefill 打断的次数，但新请求会等更久才能拿到首 token。**它不保证两轴同时改善。** 我们先比较 N=1/2/4、保持原 chunk 预算，再单独比较 chunk 大小。频率不是并发数，也不是每个请求生成多少 token。

适配时只给实际执行的 decode 计数；没有活跃 decoder 时立即推进 prefill，不能等待一个永远不会出现的第四步。prefill 获得机会不等于一定能接纳请求，槽位和容量检查继续保留。首次实验走 ordinary batch 路径，避免一次加入投机/普通模式切换。

**如何判断值不值得保留：** 冻结一条“长 prompt 到达正在 decode 的短请求”的 arrival trace，先核输出、有效 KV 前缀、GDN 状态和卷积状态，再测全部请求的 HTTP 吞吐与完整延迟。同时看 TTFT、token 间隔和最大等待，不能只展示正在 decode 的请求变顺畅。当前 CAP64 短请求结果不授予长负载资格。

JustFit 的另一个可拆机制是组件 owner：仍有请求使用 head，就不能因为另一个请求进入 prefill 而卸载它。其[驻留管理器](https://github.com/YuhuaBillChen/justfit-mlx/blob/3cc6e0cf41f1d0ed939a9702926bd62642a3fef8/mlx_vlm/server/component_residency.py#L82)值得参考；本方原生图仍持有 buffer，单把 Python 字段设为 None 不够。该论文的 TQ4 KV 是有损压缩，容量收益不能归给调度，也不进入当前严格数值对照。M5 E026 已研究部分生命周期问题，这里不重复作为新实验。

## 3. Metal 方向：已有想法获得了支持，也多了反例

MacPaw 的[全文附录 C.3.4–C.3.5](https://research.macpaw.com/publications/llm-optimizations-apple-silicon/Evaluating_the_Applicability_of_LLM_Inference_Optimizations_on_Apple_Silicon__NeurIPS_ODI_2026.pdf#page=19)展示了解码后的权重跨少量行复用。Table 6 有窄正例：Qwen3-8B、256输出、k=7时，移植内核的投机 decode 为59.0 tok/s，普通 decode 为52.4，原投机为51.9；这是两次中位数，不能概括为投机普遍有效或无效。C.5.4 的 repeated-expert staging 反而只有0.76–0.93倍，说明显式搬进共享存储也可能更慢。

**本方拟改动仍是第46章那一个：** [V124](../code-versions/44-shared-w-m1/kernel124/shared_impl.metal)已经复用压缩 W；进一步把原 NVFP4 解码移到 token 循环外，让几个 token 共用 decoded16。保留 E2M1/E4M3 解码、scale 位置、乘加分组及 BF16 舍入。论文的 affine W4A16 数学不能直接替换我们的 NVFP4。

```text
已有：读压缩W一次 → 每个token分别解码W、做乘加
候选：读压缩W一次 → 解码W一次 → 每个token做原顺序乘加
                       ↑ 关键：多占寄存器，换少做重复解码
```

先核编译器是否已自动完成这个移动；额外寄存器若导致 spill、变慢或任何数值差异，就保留失败结果。暂不同时加 threadgroup staging。M5 原生 `fp_qmv_wide` 已有类似复用，本项针对本方自定义 kernel，不能称作给所有 backend 新增能力。V127 的局部数值通过也不等于整模提速。

## 4. 检索式投机：先试能否省掉 draft 模型，再谈复杂索引

AgSpec 的增量不只是“从 prompt 抄几个 token”，而是组织会话和工作区来源，并按 diff、转义等真实输出形式建立索引。论文 generation throughput 不含 prefill，global corpus 还有数 GB 的内存成本。[TLAR](https://arxiv.org/html/2610.07350v1)则从已生成轨迹提取候选，配合小 draft 模型和反馈；主性能实验不能解释成“无 draft 模型也已提速”。其在线策略在历史超过512 token后才启用，不能直接证明本方32输出负载的收益。

**本方最小方案是更早已有的 request-local lookup，尚未实现：** 只从当前请求的 prompt 和已提交输出找候选，不加载 DFlash。先用它测算检索成本与 verify 成本，再决定是否实现 AgSpec 的会话索引。检索无命中时回到普通路径；目标模型仍验证候选，才允许交付。

| 替换位置 | 当前代码依赖 | 拟变化 |
| --- | --- | --- |
| [llm_engine.py](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/llm_engine.py) | 根据 drafter_dir 加载 DFlash | 独立、默认关闭的 lookup 模式；不加载 draft 权重 |
| [spec_runner.py](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/spec_runner.py) | drafter.cfg、hidden features、draft调用 | 独立 LookupSpecRunner；复用 target verify/commit，不伪造 drafter 对象 |
| [model_runner.py](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/model_runner.py) | feature history、drafter reset | 单独处理 lookup 生命周期；无 capture 时注意模型只返回 logits |
| scheduler.update 之后 | 提交实际交付的 token | 只把已接受、未被EOS/预算截掉的输出加入索引 |

K 个候选还要加一个 anchor，验证长度是 **T=K+1**。K=1/2/4需要T=2/3/5；现 shared-W 仅覆盖T=2/4/5，因此必须补T3，不能沿用V127的全部资格。lookup自身的 reject/partial/EOS/reset 和 GDN 状态也要独立核对。

cold 对照每个 warm/measure 请求都新建索引，不能留下上一轮同题的答案。真实会话复用另设负载，只允许当时已经可见的文本。接受率只是解释量，最终仍算从 POST 到终止的全部时间和真正交付的 token，包含索引维护及验证成本。

## 5. QBA：暂留候选，因为我们还缺一个前提

[QBA](https://arxiv.org/html/2610.17795v1)把“等得久就晋升”细化为限额与欠账：[作者代码](https://github.com/Julianjiuxi/qba-llm-serving/blob/e51873c85cb62a0095567577fc11bf4e1b7c7cde/src/llm_in_queuing/schedulers/qba.py#L94)在短请求完成 prefill 后累计额度，达到限额产生晋升义务；容量不足时保留欠账，真正接纳一个符合条件的长请求才清除。它的主要证据来自事件模拟，noiseless 分类使用真实输出长度。

我们线上看不到未来输出，短 prompt 也不等于短输出。因此先保留这个状态机思路；在没有可验证的因果预测器前，不采用其完整策略，更不把知道答案长度的模拟成绩画进 GPU 前沿。未来可用冻结 trace 单独检查取消、预算不足与重复计数，再研究预测误差。

## 6. 读完后，下一步具体缩小到哪里

1. **服务路线：先试 prefill 频率 N，后试 chunk 上限 P。** 相比直接引入时延预测器，先回答一个更小的问题；原 FCFS 和数学路径保持为对照。
2. **算子路线：继续原 decoded-W 小实验。** 新论文提供支持与反例，没有理由因此扩大成一次重写。
3. **投机路线：先计算 request-local lookup 的成本。** T3和状态资格补齐后才有完整比较；会话索引、TLAR长轨迹和QBA留在后续。

每个候选独立提交，先过正确性再看完整请求的吞吐—延迟。图中本方、oMLX、Lithos的当前点仍来自[第45章](45-lithos-queued-concurrency.md)，本章不增加未测点。上述资料随Git交接给M5；发出与对方确认阅读是两个状态。
