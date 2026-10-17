# 新论文 → 机制 → 我们的实验

检索和核对日期：2026-10-16至2026-10-17。下列日期是arXiv首次公开或修订日期，不等于
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

- [Adaptive DHiSparse v1](https://arxiv.org/html/2610.13708v1)，2026-10-09；读摘要、§1、§3.4。
  草稿KV按可见窗口保存，验证暂存按活跃层复用；逻辑位置与物理驻留分离，覆盖前等待旧reader。
  作者在长上下文稀疏模型上报告收益，不能外推到M5统一内存或本Qwen稠密attention。
  本机可借用的独立机制是bounded draft KV：当前DFlash2只读2048窗口却拼接全历史。
  E014先保留原投影、可见KV和位置数学，检验减少无效复制；不是复现HBM/host分层系统。

- [ReaDy v1](https://arxiv.org/html/2610.20503v1)，2026-10-14；读摘要、§3、§4.1，
  同时核对[官方实现说明](https://github.com/StartLuxLabs/ReaDy)。在DSpark中加入经训练的
  低秩reader，让不同draft位置读取已提交GDN状态；固定大小状态仍有gather与准备成本，
  实现将这些搬运融合。作者的B200/BF16、C1、thinking-on结果不能外推本机NVFP4。
  该方法需要reader训练与drafter适配，当前DFlash2权重不能加一个读取开关就复现。
  我们的推论：下一轮保持committed-state ownership明确，并优先消除已有状态的重复搬运；
  E018只研究原始递归数学的短T驻留kernel，不是ReaDy复现，也不先移植其训练头。

- [Tree Exit Verification v1](https://arxiv.org/html/2610.11750v1)，2026-10-08；
  读§2、§3.2、§4、§6.1。固定草稿树下，达到上界的精确验证器有相同输出块长度分布；
  TEV用一次exit-node及bonus采样组织并行计算，ExitTrain则训练更好的树覆盖。
  §6.1交叉实验中，固定drafter只换TEV的端到端收益为0.4%/0.7%，不能把组合系统约14%
  归给验证器替换。本机推论：优先减少实测主要成本；E012选路、E018状态核和E019历史提取
  分别量完整draft/verify影响。当前是greedy线性块，尚无本文的树与训练头，不引入树框架。
- [MacPaw Apple Silicon优化评估](https://research.macpaw.com/publications/llm-optimizations-apple-silicon)，
  作者页日期2026-09-30；读[论文](https://research.macpaw.com/publications/llm-optimizations-apple-silicon/Evaluating_the_Applicability_of_LLM_Inference_Optimizations_on_Apple_Silicon__NeurIPS_ODI_2026.pdf)
  摘要、附录B.3–B.6及C.1。主要投机比较是M2 Pro/M3 Ultra的4B/8B、C1聊天负载，
  不能代表本机27B/M5；附录也保留特定配置正收益，不能概括为Apple投机必慢。
  本机推论：E020要测无观察器的完整HTTP，分别记TTFT、实际产出与总时长；
  19→32只作首个可比较点，后续长输出/不同任务须新资格，不能据此宣称完整前沿。
  接受长度增大只有在draft+verify+commit总成本低于被替换普通步骤时才有意义。

- [Edge0 v1](https://arxiv.org/html/2609.18063v1)，2026-09-16；读§3.1–3.3、§4、§5.1。
  prerouter的预测直接替换原路由，配套训练recovery LoRA；不能作为同权重、同路由的
  无损预取优化。可独立借鉴fixed-slot/incremental stack，只重写发生变化的expert槽位。
  本机推论：先保留GLM原路由与packed字节，明确buffer所有权，再量布局准备成本；
  不引入训练头。作者active-memory数字不含OS page cache，不当成本机物理峰值预算。
- [BaseRT当前官方仓库](https://github.com/basecompute/baseRT)，2026-10-17查阅README，
  与前述7月论文分开：当前宣称已有continuous batching、paged KV和prefix cache；
  不能继续把旧论文的功能边界当最新版本。公开的是转换器、格式、绑定等，实际推理
  核心为单独分发的专有预编译binary，未读取其Metal实现。README列affine Q2–Q8；
  加入本机基线前仍须核对NVFP4是否可原样表示，不能换量化后称同checkpoint比较。
  本轮未安装、未跑分，不把其自报速度放入本机Pareto。

- [FusionML v1](https://arxiv.org/html/2607.22785v1)，2026-07-24；2026-10-17读§3–6。
  已materialize的输入可让CPU/GPU行拆分并发；lazy GPU依赖可能使其串行。统一内存共享带宽，
  增加计算单元不等于增加decode带宽。正文§5.4说明六组生成中有一组near-tie token分歧，
  §5.1用误差容限验数值，因此不能从摘要的token-identical概括为本任务的逐位无误差资格。
  本机推论：暂不引入CPU分流主线；可借鉴按实际瓶颈选择单独开关，以及避免在线探测
  挤出当前working set。该论文不是10月新发表，也没有在本机复现它的速度。
- [BaseRT M5 v1](https://arxiv.org/html/2607.19438v1)，2026-07-21；2026-10-17读§2–4。
  M5 GPU内部矩阵单元由Metal4 cooperative tensor操作使用；论文将高算术强度prefill与
  memory-bound decode分别调度，并保留shape不满足时的SIMD路径。§3.2未公布完整shape
  阈值；其Q4/Q8格式及M5 Pro结果不能直接证明本机NVFP4受益。推论：先核查MLX现有
  NVFP4/NAX dispatch，再做独立prefill形状实验；不重新量化checkpoint，不将作者速度
  放进本机Pareto。当前E024只研究长decode图复用，与论文tensor kernel加速分开验证。

- [Orthrus精度复现研究 v1](https://arxiv.org/html/2609.15504v1)，2026-09-14；2026-10-17读§3–5。
  RTX3090/Qwen3-1.7B、1190 prompts、greedy cap128的BF16完整轨迹一致率为45%/43%，
  FP32在这组实验中全部一致；下游任务分数近似不证明逐token等价。作者未定位具体偏差算子，
  因此不能把该失败率外推到本机或某个kernel。推论：继续保留固定数值合同、完整logits/
  状态与轨迹验证，正确性失败不能用速度或平均任务分数抵消。本机未训练或移植Orthrus。

2026-10-17追加检索10月Apple Silicon/GDN/persistent inference/continuous batching，未筛出
比已记录ReaDy等更直接适用的10月系统方案；EAGLE-ASR是语音模型专用训练头，未纳入本轮。
上述7月和9月条目是本轮新阅读，不称为本月新发表。


2026-10-17后续检索补入确切的10月官方资料：
- [Apple MLX/RDMA on Mac Studio](https://machinelearning.apple.com/research/mlx-distributed-inference)，2026-10-15。
  单机Qwen3.8为BF16/512GB/8192或65536输入，并非本机NVFP4/256GB HTTP合同；不直接排速度名次。
  可借prefill算力与decode带宽分开分析；多机RDMA/pipeline收益不归给单机，实验性FAST_SYNCH
  没有独立单机消融，暂不默认启用。GLM5.3未覆盖。
- [TreeWY v1](https://arxiv.org/html/2608.20961v1)，2026-08-21。生产实现按BF16容差验证，
  token轨迹不逐位相同，因此不直接接入其状态重建算术。本仓已有单checkpoint+tape而非
  store_all，不能套其状态节省倍数。可借鉴缩短已核对snapshot的存活期，并区分准入收益与每步成本。
  [独立阅读笔记与候选切点](../../bench/reviews/2026-10-17-apple-mlx-research.md)保留来源和适用边界。

### 2026-10-17: GDN ownership, structural prefill, and fused-scale audits

Read the July [MSInfer technical report](https://arxiv.org/html/2607.16831v1) and September [NVFP4 GDN study](https://arxiv.org/html/2609.04098v1). Selected lessons: distinguish dispatch costs from chunked prefill structure, retain strong same-environment baselines, and verify original per-projection scale semantics before fusion. No numerical relaxation or NVIDIA speed factor is imported. Local oMLX/Lithos source checks independently support explicit small-state ownership. [Detailed bounded review and source hashes](../../bench/reviews/2026-10-17-gdn-ownership-and-fusion-research.md).

### 2026-10-17: workload-conditioned Pareto interpretation

[Scaling LLM Inference via Parallelism v1](https://arxiv.org/html/2610.19405v1),
2026-10-14; read introduction and §§2–2.2. The paper defines workload-conditioned
frontiers and separates prefill/decode costs; its experiment is Qwen2.5-14B BF16
on eight H200 GPUs. This supports keeping workload and hardware fixed while
comparing configurations. It does not establish a single-Mac kernel speedup.
Our inference: retain throughput vs **complete-request latency** as requested,
and use TTFT/ITL only for diagnosis. Do not copy the paper's TTFT–ITL axes or
multi-GPU TP/PP recommendation into our M5 plot. E027 tests weight reuse with
unchanged per-row arithmetic; it is our local candidate, not a reproduction of
this paper or a new Pareto measurement.


### 2026-10-17: recent MLX-LM proposals checked against our actual path

This bounded search did not identify another directly applicable October paper
that preserves this experiment's weights and arithmetic. Two newly read items
were proposals in the upstream MLX-LM repository, not accepted research results:

- [PR #1950](https://github.com/ml-explore/mlx-lm/pull/1950), opened October 6,
  still open when read. Read its description, reported tests/results and server
  and test diffs. It avoids a device-to-host scalar log-probability read when a
  response does not request log probabilities. Our native runner has no matching
  unused log-probability read; its remaining token reads serve delivery and EOS.
  No local speed gain is inferred from the proposal's measurements.
- [PR #1970](https://github.com/ml-explore/mlx-lm/pull/1970), opened October 13
  and withdrawn by its author October 16. Read the discussion/reproduction and
  `_LeftPaddingMixin` plus prepare/finalize changes. Removing a mask can change
  SDPA dispatch and rounding. B1 alone does not establish absence of padding,
  especially with a compiled fixed attention bound. It is not adopted here.

A narrower local hypothesis is to reuse identical boolean mask arrays within
one forward context while keeping their values, shapes, K/V bound and array-mask
SDPA call unchanged. First establish that eager long decode actually repeats
this work; compiled execution may already eliminate it. This is a hypothesis,
not implemented or qualified, and is lower priority than the already prepared
E025 native-QMM-first NAX screen. Neither PR supplies a new Pareto point.


### 2026-10-17: October scheduling papers and local measurement decisions

[Lightweight Scheduling Policies v1](https://arxiv.org/html/2610.15899v1),
October 11; read §§2, 5–8. This PyTorch/Hugging Face study uses L40S/A100/H100,
small models and trace-defined output lengths, not normal-EOS generation.
Aging shortest-prompt-first is useful in some settings, not universally;
remaining-length scheduling uses unavailable oracle information. Our inference:
a small waiting-admission policy may merit a separate experiment once the
baseline exists. Preserve active-row order and test starvation; do not import
its speed ratios or mix model/workload averages into our frontier. The current
D prompts have a narrow length range, so this paper does not predict a benefit.

[PhaseGate v1](https://arxiv.org/html/2610.04537v1), October 3; read §§2–5 and
Appendix A.1. M4/M2 experiments coordinate independent CPU HNSW retrieval with
GPU inference. Their synthetic loop uses fixed decode inputs without sampling
or output feedback; reported retrieval throughput gains are not LLM speedups.
Our inference: formal M5 performance windows should also exclude heavy CPU
hashing, archive compression and raw-array audits, since CPU and GPU share
memory service. Finish those jobs before timing. We do not copy the calibrated
4→1 retrieval limits or add a retrieval scheduler to this engine. Numerical
qualification and uninstrumented HTTP performance remain separate runs.


### 2026-10-17: Admission pacing and limits of lookahead batching

Read [Stable Token-level Scheduling](https://arxiv.org/html/2610.17689v1) and
[Online Batching with Lookahead](https://arxiv.org/html/2610.16013v1). The first
motivates limiting prefill interference; the second assumes a fixed batch service
time that does not model this autoregressive workload. Our separate hypothesis
is a default-OFF limit of one new prefill in a step with existing decode work.
No speed ratio is projected. Changed batch history still needs full numerical
checks and its own HTTP trace; complete request latency includes added queueing.
[Source scope, local mechanism and proposed checks](../../bench/reviews/2026-10-17-prefill-admission-research.md).


### 2026-10-17: recent ANE research and the shared-memory constraint

Read the October 10 primary [MacPaw technical note](https://research.macpaw.com/publications/llm-inference-apple-neural-engine),
Methods through Heterogeneous Execution. It studies LFM2.5-2.6B on M2/M3/M4 Pro,
not our Qwen3.8 NVFP4 on M5 Ultra. Its ANE path changes layout, precision and
quantization; its quality class is not arithmetic equivalence. Concurrent ANE/GPU
weight reads did not provide additive memory bandwidth in its microbenchmark.
Tree verification benefits depended on the measured width-cost curve.

Our inference: leave the current target and numerical contract intact; prioritize
GPU graph reuse and measure verification width on this machine before considering
another execution backend. Do not assume using another compute unit increases
memory bandwidth. This read adds no local performance point or ANE implementation;
HTTP capture/replay remains the next experiment.


### 2026-10-17: shape adaptation is not permission to change arithmetic

Read October 5 [DynaCore v1](https://arxiv.org/html/2610.07443v1), §§7–8. Its
batch-dependent mapping uses offline profiling and a small runtime lookup. The
reported platform is synthesized/custom hardware with cycle-accurate simulation,
not an Apple GPU; it also changes prefill/decode quantization. Those speed ratios
and accuracy results do not establish an optimization for our NVFP4 contract.

Our inference: profile actual M/N/K and select among independently qualified
implementations outside the timed loop. Preserve reduction/scale semantics; the
prior E025/E027 numerical failures remain failures. Reusing a bounded long-context
graph (E024) is still a more direct local next test than importing a Split-K or
phase-quantization change. No new kernel, quality relaxation, or performance point
is introduced by this reading. Full-request latency remains the Pareto x-axis.
