# V131：近期投机解码 / hybrid state 研究筛选

2026-10-17 检索；只据论文、作者仓库与原始 RFC。仅研究；未运行 GPU、模型或服务，也未改动引擎。建议新方向：**成本感知策略 → 长上下文异步混合调度**；hybrid 少快照仅列已有机制核验/另 fork 观察。本文是新实验建议，不是数值资格或预期加速承诺；旧短 19→32 负结果、V127 局部 1491 PASS 均保持原范围。

| 原始论文、首次日期 | 作者代码实际状态 | 对当前项目的增量 / 限制 |
|---|---|---|
| [Learning to Draft](https://arxiv.org/abs/2603.01639)，2026-03-02，ICLR 2026，Zhang 等 | [官方实现](https://github.com/zhzihao/Learning-to-Draft)，depth/size 两策略训练与评估已发布 | 直接优化每轮 draft+verify 吞吐，不只优化接受长度；现成 EAGLE/CUDA 策略不是 M4/27B 可用权重。 |
| [CaDDTree](https://arxiv.org/abs/2606.01813)，2026-06-01，Zhang 等 | [官方实现](https://github.com/ZhangShuai1230/CaDDTree)，含 verify/draft profiling 与成本拟合 | 无额外策略训练、按时间成本选树预算；发布成本模型针对 Qwen3-4B/8B、8×A800，不能搬到 Apple GPU。 |
| [LibraSpec](https://arxiv.org/abs/2608.08721)，2026-08-09，Lin 等 | [全文](https://arxiv.org/html/2608.08721v1)未给作者实现链接；本次未找到可确认的官方 repo | 扩长仅在边际接受收益超过验证成本时有益。§3.2 Eq2 忽略/固定 diffusion draft 成本，后续推导近似固定每位置 verify 增量；本方 bounded DFlash 的实际 draft 成本不能忽略；oMLX MTP 是另一条路径，须分别计量。 |
| [ReplaySSM](https://arxiv.org/abs/2610.13778)，论文 2026-10-09；[作者技术文](https://dao-lab.ai/blog/2026/replayssm/)标记 2026-06-15，Liou 等 | [作者 vLLM 研究仓库](https://github.com/Johnny-Liou/ReplaySSM)含 [GDN spec kernel](https://github.com/Johnny-Liou/ReplaySSM/blob/main/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py)；[RFC #49232](https://github.com/vllm-project/vllm/issues/49232)取代旧 #47572 | checkpoint+近期输入，延迟状态写回/指针回滚；GDN 缓存修正项 u，不能只存原 v。论文明确允许浮点差异；不是我方逐字节资格。 |
| [TreeWY](https://arxiv.org/abs/2608.20961)，2026-08-21，Ghantasala | 原文只描述未上游化 fork；未找到作者实现发布链接 | 三角解 pseudo-values、只重建接受状态；[§4](https://arxiv.org/html/2608.20961v1#S4)明确 BF16 tolerance、token streams 非 bit-identical。因此只作研究/另 fork，不能放宽 strict 门。 |
| [ASPIRE](https://arxiv.org/abs/2609.17943)，2026-09-16，COLM 2026，Ziashahabi 等 | [官方代码](https://github.com/Amir-zsh/ASPIRE)与[项目页](https://amir-zsh.github.io/ASPIRE/)已发布；CUDA/H100、patched FlashInfer，支持 Qwen3-1.7B/8B、DS-Llama8B | 同一 GPU batch 混合 sparse-attn selfdraft 与 full-attn verify，逐请求选验证时点；不是已实现的 native 2-slot ahead，也未提供 GDN/本27B/Metal 支持。 |
| [DLoop](https://arxiv.org/abs/2610.07659)，2026-10-06，Gu 等 | [NAVER repo](https://github.com/naver-ai/DLoop)目前只有 README，声明代码 internal review | 多轮并行 drafting 后统一 verify，需要 loop-aware drafter training；不能把现有 drafter hidden states 直接循环多轮当作兼容实现。暂列观察项。 |

## 方向一：成本感知 OFF/spec 与验证长度

**新点。** 现有 `spec_runner.py:51,100` 已有 q≥0.45 选5、否则3的接受率启发式；这里不把动态长度本身当新意。借 LTD/CaDDTree/LibraSpec 的真实收益/时间目标，加入 draft/verify/rollback 成本与 OFF 动作，保留现有目标接受器；首先只做 request 级 `{ordinary, 已整模 qualified 的固定 spec}` 选择，再考虑逐轮控制。逐轮切换必须另证 committed state 交接，不能假定安全。不同于重跑旧固定短块，也不先投入 RL 训练。

**最小实验。** 先用已有长输出交易做成本表/可盈利上界，缺少时只加一次分阶段诊断；按剩余输出、过去接受统计分桶，拟合后用未参与拟合的提示验证。新的 shared-W T2/4/5 只有 V127 同输入局部资格，必须先完成真实 speculative 事务整模资格，再比较验证长度。512/4K→128/512 属新 context/容量域，不能借短 CAP64 PASS 自动开放。

**成本与计数。** 记录 proposed、accepted proposals、bonus、实际 delivered（截断/EOS 后），另记 draft、verify、accept/commit、rollback、host sync。接受率与 delivered/verify-call 分开；目标 `E[实际提交数]/E[draft+verify+commit/rollback+host]`。修改 LibraSpec 时必须保留真实 draft 成本与实测非线性 verify 曲线，不能沿用“draft 免费”。重叠阶段不机械相加 GPU 包络；HTTP queue/prefill/冷启动另计。相同 workload 的 POST→DONE p50 与总吞吐才是端到端判据，oracle 回放不是实际策略收益。

## 纠正：accepted-prefix replay 已存在，不作为新 idea

已只读实际 candidate116：`cache.py:43–78` 的 GDNCache 已有 snapshot+tape+rollback，部分接受时用 `gated_delta_rule_chunk` 重放接受前缀并重建 conv tail；`spec_runner.py:53–62` 的 legacy 路径明确调用它。此前把这一机制建议为新方向不准确，撤回；尤其该 legacy chunk replay 不能冒充后来严格资格通过的事务路径。

**当前 qualified 事务的真实区别。** `spec_runner.py:39–40,87–139` 在 transaction 开启时改走 `_step_transaction108`；`model/gdn.py:105–108,131–142` 转入 `fused_recurrent_history`。`spec/transaction.py:173–178` 每个 token 调原44 conv/state 核，`:45–50` 把每个 prefix 的 conv/S 和 producer owner 保留，`:126–130` 显式 eval 全部 history/producer，`:138–145` commit 直接选 literal `history[keep]`，**不重放**。因此此分支确实要求每个 draft 位置产出、保留 S；不是根据 vLLM 论文猜测本项目也存全状态。单层每 S 为 `[1,48,128,128]` FP32（3 MiB）；这仅是源码要求的逻辑输出量，未新增测量实际 allocator/RSS 或性能占比，owner/view 重复引用不能算两份。

这些 prefix 正是原同 invocation/literal128cache 资格合同的 witness，不能删掉后仍宣称原合同未变。现阶段只建议读已有 timing/内存证据，分清 verify 与 commit 的成本；**不重新实现已有 replay、不切回 legacy、不先写减少历史的引擎**。若以后证明物化是实质瓶颈，才由 root 明确新事务合同/独立资格后另 fork，仍保持当前严格参考。

ReplaySSM/TreeWY 只作此处研究参照：其闭式/三角解重结合算术，不能替代原核 strict PASS。若另 fork，先 FP32 checkpoint/u、短 window，禁用低精度变体，raw o/S 任一字节不等即停止；[ReplaySSM正文](https://arxiv.org/html/2610.13778v1)明确仅在浮点差异内相等。论文大 batch/NVIDIA 的收益不作为 B1/M4 预期。

## 方向二：ASPIRE 的逐请求 mixed draft/verify（优先 M5 长上下文）

**新点。** 已有 native ahead 解决 host/GPU 衔接；ASPIRE 解决的是 batch 内不同请求无需同一 draft/verify 节拍。可借“逐请求接受统计 + batch 成本”调度，但原作依赖 sparse-attention selfdraft 和 mixed-attention GPU kernel，不能把 HTTP FIFO 串行排队称作该机制，也不能把 Qwen3 dense 支持等同 GDN 支持。

**最小实验。** 先用 M5 长输出 traces 做 mixed-vs-lockstep 的离线调度成本上界；C1 没有该批内调度收益。若值得，再独立 prototype 两请求：一条 draft、一条 verify，shared-W MLP/linear 保留各自输入行与独立 cache；先分角色输出逐字节、每请求接受/回滚/reset，再考虑运行时接入。需新增 GDN 事务处理与实际 mixed forward，不能只改客户端并发。成本分列本轮 batch 工作量、accepted/delivered、verify/rollback、scheduler 与队列等待；图中保留 full POST→DONE，不仅 decode tok/s。

**暂不优先 DLoop。** 先看长输出 full-accept 频率和“多一次 draft 换掉一次 verify”的乐观上界；连上界也不盈利就不训练。兼容代码/权重未发布前不假装有可跑整合。M5 新 E027/E028/E024 是 root 转交的独立进展，本文未复审、未当作 HTTP 或本机性能证据。

上述两个新方向及另 fork 研究都需新的实际资格/计时；没有把论文的 speedup、数学等价或作者 lossless 声明直接转成我方采用结论。
