# 2026-10-17：MLX 图复用与状态生命周期跟进

本轮未找到新的、可直接用于当前 Qwen3.8-27B NVFP4 并保持逐位合同的 2026 年 10 月论文。检索后选择三份官方或项目作者一手资料：一份版本固定的机制文档、两项近期故障修复记录。它们不是三个新算法，也不提供我们的 HTTP 加速结论。已记录的 SiliconBench、BaseRT、Apple 10 月 15 日分布式 MLX、StableTokenScheduling、DynaCore、PhaseGate 不作为本轮新发现。

## 来源与实际阅读范围

| 一手来源 | 日期与阅读范围 | 可确认的机制及边界 |
| --- | --- | --- |
| [MLX v0.32.3 compilation 文档](https://github.com/ml-explore/mlx/blob/v0.32.3/docs/src/usage/compile.rst) | 固定版本背景资料，并非 10 月新论文。读 Basics、Pure Functions、Shapeless Compilation。 | 首次编译有成本；shape/dtype/输入数量变化可能重新编译；反复创建 callable 会损失复用。动态状态须作为输入或捕获输入，更新状态须输出。`shapeless=True` 不会重新执行依赖形状的 Python 分支。文档只承诺数值精度范围内一致，不能替代逐位门。 |
| [MLX PR #4556](https://github.com/ml-explore/mlx/pull/4556)、[完整小 diff](https://github.com/ml-explore/mlx/pull/4556/files) | 9 月 24 日提出，10 月 1 日合并。读描述及 `mlx/transforms.cpp`、`python/tests/test_eval.py` 修改。 | 异常清理可能等待尚未提交的 GPU 工作；修复先 finalize GPU streams，再同步 streams。测试覆盖 GPU 工作之后的 CPU 算子错误与后续 eval 可用性。这是失败路径正确性，不是吞吐优化；未核当前 wheel 是否已包含该提交，也未证明我们的路径可触发它。 |
| [MTPLX 作者项目 issue #544](https://github.com/youssofal/MTPLX/issues/544) | 9 月 26 日报告；10 月 2 日维护者解释修复，10 月 12 日报告者复验。读问题描述、stream/nonstream 对照及上述回复，未审完整实现 diff。 | 维护者将事件耗尽归因于未被读取的 lazy 稀疏索引写入、draft-history 追加和 session 状态持有；修复在产生步骤求值。报告者确认事件错误不再出现，但内存仍增长。其 Flash-Next/MTP/长 nonstream 请求与我们的 27B target-only 工作负载不同，不能迁移为“我们也泄漏”或“所有内存问题已解决”。 |

## 第一优先：保留 dispatch 的图复用进入真实调度

**假设 H1：** 每个 runner 保留 E024 callable，并显式传入/返回状态，可将已观察到的 forward 收益转化为完整请求收益；不改变算术、量化或调度策略。

实际实现已具备必要结构：[candidate.py](../../../bench/m5_ultra_256gb/route_decode_graph/candidate.py) 100–127 行按 bound 与 GDN 开关缓存 callable，118 行用 `mx.compile(forward, inputs=model)`，状态作为显式实参和返回值。两个 bound 的 Python callable **不等于只有两次底层编译**：batch shape 改变仍可能产生专门化。需区分 callable 建立次数、实际可观察的编译事件、首次调用耗时与稳定调用耗时；不能用 Python 字典长度假冒编译计数。

保留现有 1023/1536 分界；不引入 shapeless 或统一 1536 上界。[attention.py](../../../tutorial_code/miniengine/model/attention.py) 79–93 行用 `ctx.max_len` 决定 KV slice 和 mask 形状，因此“有效 token 相同”不足以证明 dispatch/reduction 相同。既有 E024 失败及后续 route-preserving 结果已经说明这一点。

最小下一实验为默认 OFF 的单开关集成，保持目标 HTTP 既有 baseline 的 **gates ON、norm reuse OFF**，并固定同一原生模型/运行时。gates OFF 仅属于历史矩阵或独立诊断，不另造正式对照；已有 forward 表包含对应 ON 结果。这个 serving 设置仍需新的 changed-value、完整状态及 ahead 资格：

1. 在真实 scheduler 中复用现有 C1/C8 D512 测试，覆盖 B1→B8→B1、1023→1024、ahead 消费/丢弃、refill 与 retire。每步核完整 logits 加 128 状态的原 dtype 位、输入不变及参数值回执，保存实际候选调用证据。复用现有矩阵证据，不把它升级成尚未执行的 HTTP 资格。
2. 数值门通过后，OFF/ON 用相同 fresh-process、warm/formal 请求和计时合同配对。正式 observer OFF，主轴是实际 committed tokens / 共同请求窗口与 POST→DONE p50/p95；冷启动/首个形状专门化另记，不藏进 warm 排除项。
3. 任一逐位差异、错误状态复用或资格范围外 dispatch 即拒绝；若 forward 收益被调度、同步或编译成本抵消，则拒绝“提高 HTTP 前沿”的假设。

现有 [E024 timing 记录](../../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md) 的 5.44%–10.67% 只是八点单进程 forward 速率收益，不是 HTTP 或竞品比较。

## 第二优先：先验证 E028 的引用生命周期，再考虑额外求值

**条件假设 H2：** 去掉 identity gather 后，旧 batch/staging/ahead 的别名可能改变 backing 或 lazy graph 的存活时间；只有实际出现多余保留，才值得增加一个独立默认 OFF 的生命周期改动。当前没有这类泄漏证据。

[identity_refill/candidate.py](../../../bench/m5_ultra_256gb/identity_refill/candidate.py) 69–102 行只在 identity/nonempty/容量不变条件下移除 gather，仍保留原 concat。不能把逻辑数组字节、共享 backing 字节与进程物理占用相加当作互不重叠内存。

存在重要反证：[model_runner.py](../../../tutorial_code/miniengine/engine/model_runner.py) 88、101 行已经将当前 batch 与 ahead 全部状态加入 `async_eval`，158 行也求值当前 batch；174、195–204 行在换批/退场清理 ahead、staging 和 batch。因此 MTPLX 的“无人读取状态写入”机制尚未在我们这里成立，不建议直接添加同步 fence。

最小诊断在既有 OFF/ON refill 重演的原求值边界记录 old/staging/ahead 的持有关系、去重 backing 和 allocator 计数；观察器释放与生产对象释放分开。继续保留 offset/strided view、延迟求值、源/输出分别写入及 BF16/F32 全字节测试。若确认某个必要状态输出未被提交，可单独测试把它加入**现有** eval/async_eval 参数，或在最后读者之后释放引用；不另加 CPU 同步、不改变请求顺序。若状态已全部覆盖、重复 refill 后保留量稳定，或改动增加延迟，假设即被证伪，不实现优化。

PR #4556 仅提醒未来跨流异常测试要保留提交、deadline、reap 的真实证据；本轮不改运行时、不运行故障注入，也不把它包装成性能开关。

本轮只做网页与源码阅读；未运行模型、设备测试、性能测量或服务操作，未修改冻结实现。
