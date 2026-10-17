# 固定冷启动生命周期与逐次不可变计划

这是 CPU-only 设计，不改变本轮 v7、不启动服务。v7 canary 继续按已冻结源执行。

后续 40 点应由一个数值子进程读取其固定矩阵；owner/controller 不需要理解 40 个点，也不应因此增加 40 种 phase 或复制 v8/v9。先只支持现有单 child 合同，暂不合并 HTTP 多 phase 控制器。

## 最小变化

保留一份固定 controller、一份固定单 child owner、一份固定 restoration auditor。只把当前源码中每轮变化的 phase、实验脚本/计划及其 source pins 移入不可变 run plan。OS 查询、停止/恢复、物理资源、SIGTERM/KILL/wait、未知 Popen、STOPPED 锁和完整原始证据格式保持原语义。

每次计划仍绑定：

| 字段 | 固定约束 |
|---|---|
| `lifecycle` | 已审阅源码清单的路径/SHA，内含 controller、owner、restoration auditor 的路径/SHA；本轮全程不变 |
| `predecessor` | 归档 RESTORED journal 与独立 restoration receipt 的路径/SHA；二者必须精确绑定 |
| `experiment` | 唯一 phase、qualifier 路径/SHA、实验 plan 路径/SHA、实际依赖 source pins |
| `inherited_owner_plan` | 现有 E024 `4dff…` 原计划，继续调用原 prepare，保留真实环境和模型来源验证 |
| `outputs` | 唯一新 journal/out 绝对目录；child 只能写约定的 `phases/<phase>/qualification` |
| `budgets` | 当前固定资源合同及 work 1830s / child ≤1800s；计划不能放宽资源阈值或回收上界 |

命令只由固定 owner 构造：原 preflight 返回的固定 Python 和环境，`-I -B <qualifier> --plan <experiment-plan> --plan-sha256 <sha> --out <qualification>`。不支持任意 shell 命令、动态插件或自定义 cleanup 回调。实际矩阵属于实验 plan；若 40 点超过现有预算，分成多个按同一协议独立关闭的 run，不扩大已批准预算。

## 同版本连续运行与源码指纹

固定 controller 可以接受同一 lifecycle 源清单下的前一次 RESTORED journal；要求 predecessor audit 的源 SHA、journal SHA、owner plan SHA、恢复 PID/identity、raw log 与 worker returncode 都闭合。新 original identity 来自该经过验证的 replacement，并在真实 prepare 再核对。实验 FAIL 不阻止一致的恢复证据作为前序；缺 release 或未知 Popen 仍拒绝恢复。

首次迁移只允许明确固定的 v7 journal/audit 对，之后要求相同 lifecycle 清单版本。首次迁移不是一个宽泛的旧版本 allowlist。

**避免互相嵌入 SHA 的循环。** 固定 controller 若嵌入同版本 auditor SHA，而 auditor 又嵌入 controller SHA，会产生无法冻结的相互依赖。改由已审阅、逐次计划固定 SHA 的 lifecycle 清单同时绑定两者；两份代码独立核对本身和对应实际源文件与清单相符。清单本身不嵌入运行 journal/receipt SHA。保留对原底层 OS/parser/support 源的现有固定 pin。采用前必须单独审阅此信任入口，不能仅凭任意 JSON 声明接受未知 auditor。

## 证据与资格边界

每次仍保存 attempt、STOPPED 原字节、spawn intent、已知 child identity（探测失败可为 null）、实际 wait/reap 与 raw ps absence、source copies、原 prepare 证据、资源原件、恢复原件。NO_CHILD 只允许 intent 之前的失败；未知 Popen 绝不转换为 NO_CHILD。所有 receipt 绑定当前 plan/owner/journal/phase，不能只验证文件存在或用离线 `process(None)` 代替原始 absence 证据。

owner 只报告执行与回收结果；child PASS 不能直接变成完整模型或性能资格。数值 auditor 按实验固定合同验证 logits/state/raw outputs，restoration auditor 只验证恢复。后续正式 throughput–latency 点依然单独要求数值、资源和无 observer 计时门全部通过。

## 一次性迁移验证

在改变固定实现前，先用 CPU fake 连续跑两个同版本 cycle：第二轮只能接受第一轮精确归档恢复；修改 phase、plan、源清单、receipt SHA、旧 PID 或恢复时间窗应拒绝。复用已有 pre-intent failure、未知 Popen、identity 探测失败、重复取消、TERM→KILL、存活 PID、物理单位/时间窗与 experiment FAIL/restoration PASS 测试。再对 v7→固定版本的函数差异做一次审查，后续只审实验候选和不可变 run plan。

本轮已冻结实现：route owner `4d850e3b…63f0c96`，controller v7 `c30aaffc…e70a1ab`，restoration auditor `0ab9fd49…6149ad1`。它们不因这份设计发生变化。
