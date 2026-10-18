# 46：读新论文，找下一处值得替换的代码

研究日期：2026-10-17。**本章完成源码与论文筛选，尚未实现候选或产生新的性能点。** 第45章的三家九点图保持原样。

后续状态更新：[M5已确认阅读本章并反馈](../progress/2026-10-17-m5-reply-update.md)。其原生MLX已有decoded-W复用；本方自定义shared-W仍可独立研究。成本表需补T=3。下文保留本章写作时点，不能用当时的“待回执”或早期局部资格描述最新状态。

上一章回答的是“19 输入、32 输出、C1/C2/C4 时谁更快”。接下来有两个不同问题：同样的计算还能不能少做一点；长短请求混在一起时，能不能更合理地安排计算。它们需要不同实验。

## 1. 先和另一台机器对齐

本次先读取 M5 `86757b8` 的交接，再发送研究请求 `ce068d8`。随后实际 fetch 到 `3078d7b`：对方已合入 M4 的 `bea0a02` 与 `ce068d8`，明确回复同意交换 T=1/2/4/5 成本，并由 M5 优先完成长输出服务验证、M4 推进混合请求调度。**这轮请求已有回执；后发的详细研究包 `7ffe424` 尚未收到阅读确认。** 通信记录在双向 Git handoff。

| M5 已有进展 | 现在能得出的结论 |
| --- | --- |
| E024 图复用：B1 forward 速率约提高 6.68%–7.01%，B8 约 5.68%–5.92% | 局部计时；默认关闭，尚待完整服务验证 |
| E028 identity-refill：固定调度的整模检查通过 | 状态与补位路径取得数值证据，尚无 HTTP 提速结论 |
| Lithos D512 的 C1/C8 请求校验通过 | 可以进入正式测量；C8 是官方 FIFO 排队，不能称 GPU batch=8 |
| E027 shared-W M1 局部数值通过 | 可以交流投影成本，尚未证明整模收益 |

M5 回复会先完成 Lithos 正式基线，再验证 E024 图复用和 E028 identity-refill 的完整 HTTP 收益；它指出 prefill、调度、采样、输出与图外缓存更新都可能抵消局部收益，尚需测量。M4 先研究下面的小幅算子改动与混合请求调度。两台机器各自标定成本，分别画图。M5 上已读的 SiliconBench、PhaseGate、DynaCore 等只作共同背景，不重新包装成新发现。

## 2. 首选小实验：权重解码一次，供多个 token 使用

我们已有的 [shared-W 内核](../code-versions/44-shared-w-m1/kernel124/shared_impl.metal) 会将压缩权重读一次，再给 T 个 token 使用。但在源码表达式中，每个 token 仍然重新把 4-bit 编码转换成浮点值。

**已有代码，简化节选：**

```cpp
load_packed_words(cached);              // 权重读取已复用
for (int token = 0; token < VERIFY_T; ++token) {
    load_x(token, x_thread);
    result[token] += qdot(cached, x_thread, scale);
    // qdot 内部仍调用 Dequantize<4>
}
```

**拟验证的替换；这是伪代码，尚未接入：**

```cpp
load_packed_words(cached);
decode_original_fp4(cached, decoded);   // ← 关键①：移到 token 循环外
for (int token = 0; token < VERIFY_T; ++token) {
    load_x(token, x_thread);
    result[token] += same_order_qdot(decoded, x_thread, scale);
    // ← 关键②：保留原乘加分组、scale 位置与最终舍入
}
```

直观上，原来“读一次压缩 W，解码 T 次”，候选变为“读一次，解码一次”。新增代价是把 16 个解码值留在寄存器里。编译器也可能已经自动完成这个移动，因此源码少写几次不等于 GPU 少执行几次。

这一数据复用结构参考了作者公开的 [Lily skinny.metal](https://github.com/perplexityai/pplx-garden/blob/main/lily/src/kernels/metal/skinny.metal)。Lily 项目针对 M5+、affine Q4；我们只借循环组织思路，保留本方 NVFP4 解码和累加顺序。新论文 [Rigel（2026-06-11）](https://arxiv.org/abs/2606.12765)对 M4 Max 张量路径的实测也提示：格式更小、接口更新，都不能直接推导成速度更快。

**最小实验：** 固定原 shared-W 为 A，decoded-W 为 B，先测已有真实投影的 T=2/4/5；核对完整输出与输入只读，再看受控局部计时。若编译器已做同样优化、出现寄存器溢出或无收益，就停止。shared-W 本身还没有整模采用资格，此项也不能直接计入现有前沿。

## 3. 服务实验：从“每轮多少 token”走向“每轮花多少时间”

先看 [现有 scheduler](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/scheduler.py)：它已经安排 decode，再把剩余 token 预算给未完成的 prefill 和新请求。[runner](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/engine/model_runner.py)也已按 computed cursor 继续处理 prompt。因此 continuous batching 和 chunked prefill 都已经存在。

现在缺少的是处理长短请求竞争的策略。相关论文 [Fairness-Aware and Latency-Controllable Scheduling（2026-06-08）](https://arxiv.org/abs/2606.09061)研究等待时间、剩余 prefill 工作和时延预算；[作者调度器代码](https://github.com/Charmstok/Chunked_Prefill_Serving/blob/main/sarathi/core/scheduler/opt_sarathi_scheduler.py)已公开。本方拟拆成两个独立改动：

| 实验 | 之前 | 拟替换为 | 保持固定的部分 |
| --- | --- | --- | --- |
| A：排序 | prefill 按原先后顺序 | 综合已等待时间与剩余工作量排序，同分稳定排序 | token 预算、decode 优先级、kernel |
| B：预算 | 固定 `max_batched_tokens` | 用本机实测时延表选择当前可承担的 chunk | 原 FCFS 顺序，先不加额外 admission 限制 |

```text
已有：先安排 decode → 剩余 token 配额 → prefill
候选：先安排 decode → 剩余时间预算 → 选择 prefill chunk
```

不是任何 512 个 token 都花同样的时间：decode batch、已有上下文长度和 prefill chunk 都会影响耗时。不能直接搬论文在其他硬件上训练的预测器。

**最小实验：** 在 ordinary batch 路径上冻结混合 arrival trace，例如 24 个短 prompt＋8 个长 prompt、Cmax=4、输出上限512。先取得新长度和调度形状的正确性资格，再分别比较排序、预算；不要把二者一次全改。记录所有请求的完整延迟、总吞吐、首 token 时间，以及长短两类的最大等待和尾延迟。短请求变快但长请求持续饿死不算成功。

M4 目前 CAP64 的 native C1 不能直接运行这些长请求；新实验需要长容量普通路径的独立验证。当前静态启动 profile 也没有运行中自动迁移缓存的能力。

## 4. 投机解码：先算清楚成本，再决定猜多少

我们在第42章已经测到短请求投机更慢。现有 `spec_runner.py` 也已有按接受率选择长度3或5的启发式；新增方向是加入真实时间成本与关闭投机的选项，不能把动态长度本身称作新能力。

[LibraSpec（2026-08-09）](https://arxiv.org/abs/2608.08721)用额外接受收益与额外验证成本决定扩展长度；但其推导简化了 diffusion drafting 成本。我们需要实测本方 bounded DFlash 的成本，oMLX MTP 则是另一条路径，须分别计量。我们自己的待测目标是：

```text
每轮收益 = EOS／长度截断后真正交付的 token 数
每轮成本 = draft + verify + commit／rollback + 必要的主机同步
选择目标 = 长期实际交付数 ÷ 实际耗时
```

重叠阶段按真实时间包络计，不能重复相加。请求排队、prefill 和启动成本还要计入最终 HTTP 指标。

**第一步不训练策略网络。** 请 M5 交换长输出的阶段成本，与两机 T=1/2/4/5 的投影成本对照。先计算盈利上界；若连理想选择都无法省时，就停止。若有空间，先做请求级“普通路径／已通过整模资格的固定投机路径”选择，再考虑逐轮切换。动态切换需要验证状态交接；shared-W 的局部 PASS 不代表所有验证长度已取得整模资格。

## 5. 最新论文里，哪些值得读但暂不直接接入

| 来源 | 有用的机制 | 本项目下一步判断 |
| --- | --- | --- |
| [ReplaySSM，2026-10-09](https://arxiv.org/abs/2610.13778)，[作者代码](https://github.com/Johnny-Liou/ReplaySSM) | checkpoint＋近期输入，减少 recurrent state 写回，支持 GDN | 先测当前事务状态保留成本；原算法允许浮点差异，不能直接继承我们的严格资格 |
| [ASPIRE，2026-09-16](https://arxiv.org/abs/2609.17943)，[作者代码](https://github.com/Amir-zsh/ASPIRE) | 一个 batch 内各请求可处于不同 draft／verify 阶段 | M5 长上下文的后续方向；现有 CUDA 实现未提供本方 GDN／Metal 路径 |
| [Marconi，MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf)，[作者代码](https://github.com/ruipeterpan/marconi) | 联合缓存 attention KV 与 recurrent state，控制前缀收录 | 可做小规模精确前缀缓存；这是较早基础论文，首版2024-11-28，不当成今年新论文 |

这里有个需要读代码才会发现的区别：[旧 GDNCache](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/cache.py)本来已有 snapshot＋tape＋接受前缀重放；而当前取得严格资格的 [VerifyTransaction](../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/spec/transaction.py)保留每个前缀的真实 S，并显式求值以支持同一次调用的检查。**不能把已有重放再称作新发明，也不能直接切回旧路径。** 若研究减少状态保留，须独立核对每个可提交前缀、conv tail、KV 与 logits。

精确前缀缓存则是另一个问题：跨请求复用同一个 prompt。它必须保存匹配位置的 KV＋S＋conv，以及必要的最后 logits；不能只恢复 KV。重复完整 prompt 时还需避免把最后 token 再算一次。先测1–2个条目的命中、分叉隔离和淘汰，cached workload 单独画图。

TreeWY 的浮点重结合、Tail-Replay 的近似状态恢复，以及需要新训练的 DLoop 暂不进入当前严格路线。完整来源、代码可用性与实验限制见三个研究笔记：[Metal](46-paper-notes/metal-kernels.md)、[调度与缓存](46-paper-notes/scheduling-cache.md)、[投机与 hybrid state](46-paper-notes/speculative-hybrid.md)。

## 6. 下一步按什么顺序验证

```mermaid
flowchart LR
    A[论文与实际源码对照] --> B[一个明确改动]
    B --> C[输出和状态正确性]
    C --> D[局部成本是否下降]
    D --> E[完整服务吞吐与延迟]
    E --> F[同负载三家 Pareto 图]
```

建议先做 decoded-W 小实验及 mixed-load 调度准备；M5 完成已有 D512 测量后交换成本，再决定投机策略和缓存路线。每一步独立提交，失败结果保留。

本章版本只新增研究与交接文档，**没有更换正在使用的引擎、增加性能点或放宽正确性要求**。主指标仍是相同负载下的总输出吞吐与完整 POST→协议终止延迟（SSE `[DONE]`／NDJSON `done:true`）；TTFT、尾延迟和内存帮助解释原因。所有通过正确性检查的配置都保留在图中，两台机器的曲线分开。
