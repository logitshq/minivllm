# GLM 官方 C1 冷加载：最少剩余接线

当前只有已准备的 [load_probe.py](load_probe.py) 和 [physical_guard.py](physical_guard.py)，没有实际 GLM load PASS。本说明为 source-only 接线计划；没有执行测试、runtime import、模型加载、服务动作或大 payload hash。固定模型仍是原 mixed 4/5/6-bit affine snapshot，不改量化或开启 offload。

**不能直接把 probe 填进现有 cold single-child owner。** `cold_cycle_v2/owner.py:81–96` 仍固定 Qwen `4dff…` 准备且生成的 child argv 缺 `--run/--admission`；`cold_cycle_v3/controller.py:125–127` 要求 `qualifier_sha256/deadline_s`，与 GLM 的 `deadlines` 和外部 admission 不同。前者可能只 describe 后 exit0；Qwen 的 157 项来源检查也不验证 GLM 模型。

只需一个 GLM 专用单直属 child 适配，复用已审 journal lock、原服务 absence、durable intent、实际 Popen handle、TERM15/KILL15/wait 和恢复算法；不增加 HTTP/client 框架，也不更改旧 owner/source identity。

1. **真实准备。** 固定 probe 的 `GLM-official-load-plan-v1`、≤128-token plain UTF8 prompt、62-shard model inventory、实际 oMLX Python/runtime 与全部 `REQUIRED_SOURCES`。已审 oMLX runtime inventory 可以作为来源起点，必须补实际 GLM vendor/loader dependencies 并核一致；不能把旧下载 SHA+当前 stat 写成刚完成的全 payload 重哈希。完整 model/runtime 验证由新 owner 的明确回执承担。
2. **真实冷准入。** 绑定最新封存 RESTORED journal 与独立 audit，真正释放旧服务。继续原 pressure1/AC/thermal/foreign-process 门；调用 `physical_guard.measure` 对 owner 自身 PID 做两次系统 footprint 观测，交 `cold_admission` 验证。两次间隔4–60s，系统≤40GiB、变化≤1GiB、swap不增/Swapouts不增。
3. **冻结有限观察预算。** 明确 `child_limit_bytes≤240GiB`、`work_seconds≤1800`、采样间隔和单次 probe 总时限（各≤5s）；child 上限必须由计划显式给出，不能从权重文件大小反推安全值。运行时系统≤240GiB，child top/current/historical auxiliary peak 分别守界，pressure1、相对冷基线零 swap 增长。process 与 system footprint 不相加；这些门不是 Metal 分配硬 cap。
4. **签发后直接启动。** Popen 前写 `GLM-official-load-admission-v1`，绑定 owner PID、plan/probe SHA、snapshot/out、有效期限与 `cold_release/resources/model_runtime` 三份真实证据。由新 owner 构造唯一 child 命令：

   ```text
   <pinned oMLX Python> -I -B load_probe.py --run
     --plan ABS --plan-sha256 SHA
     --admission ABS --admission-sha256 SHA --out NEW
   ```

   `load_probe.prepare` 核直接 PPID 和固定环境。owner 先保留 Popen 返回对象/真实 PID，再查身份，持续 `measure→observe`；遇未知、超时、资源失败即终止并严格回收。不能用 wrapper PID 代替模型 PID，unknown Popen 不能发 NO_CHILD 证明。现恢复算法可以复用，但新计划/worker argv/source pin 必须真实接入。
5. **分开结束门。** child rc0、真实 `GLM-official-load-result-v1` PASS、caps `[8,32,32,32]` 四请求、`shutdown.complete`、原始 physical 命令/JSON/SHA 独立重解析、strict reap 全部成立才报告短生命周期诊断通过。`physical_guard.diagnostic_report` 只是已解析 observation 的复核，不能代替 raw audit；最后仍执行并独立审核原服务恢复。

Probe 已覆盖官方 EnginePool→VLMBatchedEngine/vendor 路线、45层/34 KDA/42 MoE融合、normal EOS/实际 raw与committed ID、cache0、请求队列 drain 及重复 greedy 请求一致；无需为首轮重写这些模块。取消的 executor/C++加载不会被 asyncio timeout 强制停止，外层硬 deadline/reap 必须始终有效。

仍未知首次 sanitize/stack/eval 的真实峰值。physical 采样与 TERM 可落后于分配，最后 live sample 后也没有独立 child peak 历史；不得把诊断 PASS 写成“190GiB确定安全装下”。本轮不是 HTTP、完整模型逐状态正确性或性能资格。后续 D512/C8 和 GLM throughput–latency 图需要分别建立自己的合同，不能沿用 Qwen prompt/EOS/state 数量或 Pareto 点。

依据：[最小就绪状态](../../reviews/glm53-flash-minimal-readiness.md)、[资源准入复核](../../reviews/glm-resource-admission-review.md)。
