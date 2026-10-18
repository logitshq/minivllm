# E024 route graph：实际 HTTP capture → replay 资格周期

**当前是已准备的源码，尚未执行本 HTTP 周期。** 前置的 [E024 serving 资格](../../../docs/experiments/m5-ultra-256gb/E024-serving-v1/README.md) 已实际通过独立数值、资源及恢复审计：原 37 请求轨迹，加独立的 42 次 idle 参数替换/恢复调用。它没有授予实际 HTTP、未来 arrival history 或性能资格；未发生的 ahead invalidation 仍不在覆盖范围。

本目录使用同一 Qwen 目标、D512 fixtures、normal EOS、target-only 路线，route graph ON，E028 identity refill OFF。原始 157 项 runtime/source 准备及模型前后验证保留；新适配源码按 repository-relative 路径冻结，避免同 basename 覆盖。前置 37/42 结果被精确 SHA 引用，不在这里冒称重新执行。

| 固定阶段 | 实际工作 | 关闭条件 |
| --- | --- | --- |
| `c1-capture` | C1，8 个真实 HTTP 请求，保存 SSE/group、scheduler trace 和实际 route events | owner 直接持有 server Popen；wait/reap、PID 和端口消失 |
| `c1-replay` | 用实际 arrivals/prompt/cap 在独立 reference/candidate 引擎重放该 8 请求历史 | 使用各臂自己的 greedy token，完整 logits/state/ahead 门；直属 replay child 回收 |
| `c8-capture` | C8，16 个真实 HTTP 请求，保留同样原始证据 | server 严格回收后才允许 replay |
| `c8-replay` | 重放该 16 请求实际历史 | 完整数值、route 事件与状态清理；直属 child 回收 |

这是 **24 个 HTTP 请求、两个 capture 进程和两个 replay 进程**。Replay 不增加 HTTP 请求数，不把记录的 token 或 scheduler 输出喂给 candidate。HTTP outstanding 并发与 GPU 实际 batch 分别审计。任何阶段失败立即结束后续矩阵，保留已发生的阶段前缀和失败证据；未知 Popen 结果不能被空查询变成“没有 child”。

## 封存与执行顺序

`controller.py`、`audit_restore.py`、`audit_cycle.py` 的最终 SHA 链仍待上一轮 oMLX formal-v2 **真实 RESTORED、独立恢复 PASS、归档完成**后封存。不得填预测 PID/摘要，也不得绕过 `UNSEALED`。随后由根 owner 统一运行聚焦 CPU 检查并审核；本说明不代替该回执。

在仓库根目录按以下顺序操作，各命令均须在相应门通过后单独运行：

1. 封存上述源码和不可变前序恢复 journal/audit 后，运行实际准备：

   ```sh
   .venv/bin/python -I -B bench/m5_ultra_256gb/route_http_cycle/prepare.py --run
   ```

   Builder 写入新的 `prereg-http-v1/`，实际调用 owner 的 `--prepare-only` 和 controller 的只读 `prepare`，保存 capture specs、命令、源码副本及 PREPARED journal SHA。此步不停止服务、不加载实验模型；默认无 `--run` 仅描述。源码数量从实际 manifest 读取，不写死。

2. 审核真实准备回执，**先提交预注册源码、计划、receipt，再启动**：

   ```sh
   .venv/bin/python -I -B bench/m5_ultra_256gb/route_http_cycle/execute.py --run
   ```

   Driver 执行 cold stop → 四阶段 owner → `finally` 恢复。使用全新的 `.artifacts/m5-ultra-256gb/route-http-v1-{owner,cold-journal,execution,cpu-prepare}`，不覆盖旧 journal。默认无 `--run` 不执行。

3. 等 driver 实际关闭，`final.json` 的 recovery rc0 且 journal 为 RESTORED，再运行已审离线脚本：

   ```sh
   .venv/bin/python -I -B /tmp/audit-closed-route-http-v1.py --run
   ```

   顺序为独立恢复 → `http_audit_v2.py` 的 C1/C8 实际 replay-plan SHA 审计 → `audit_cycle.py`。后者消费两份实际审计的路径/SHA，并核原始资源、全 source/runtime 绑定及四阶段回收。不会提前读运行中的 steps ledger。尚未产生 replay plan 的阶段记 `NOT_RUN`，不制造数值回执。

4. 独立恢复 PASS 后保存实际成功或失败周期：

   ```sh
   .venv/bin/python -I -B /tmp/archive-route-http-v1.py --run
   ```

   不把第 3、4 步用 `&&` 串起来：实验失败可令审计 runner 返回 1，但独立恢复 PASS 的失败证据仍应归档。输出为 `docs/experiments/m5-ultra-256gb/E024-route-http-v1/`。脚本保留五类 raw 树、预注册/源码、所有独立审计尝试及实际首差异 tensor，逐个验证 tar 成员 SHA/字节；不打包模型/runtime payload。重复审计使用新的 `--analysis .../route-http-v1-analysis-repeat-N`，归档时显式选择同一路径，全部旧尝试保留。

`/tmp` 脚本是本机待审核操作入口，运行时分别把自身源码存入分析或归档；默认仅描述。具体命令/退出码/stdout/stderr 与实际结果一起保存。

## 结果边界

Producer 的 `replay_numerical_checks_passed` 不是独立资格。仅 C1/C8 数值审计、cycle 资源/来源门及恢复门全部 PASS，才能授予这两条已捕获 HTTP 历史的资格。成功 tensor payload 不重放离线 GPU 算术；离线核对原 dtype 比较回执、哈希、完整 schedule/route/ahead 语义。所有带 observer 的时长只作执行账目，**不产生 speedup 或 throughput–complete-request-latency Pareto 点**。下一性能阶段须另行冻结 observer-OFF 负载与测量协议。

恢复范围是已授权的 persisted-visible 服务配置和模型状态；不声称 warm KV 或隐藏 runtime 状态逐字恢复。失败诊断、未来 arrival、B>8、取消/key-invalidation、E028 组合均不自动获得新资格。

## Actual registration, 2026-10-17 18:20 PDT

The original formal-v2 service restoration independently passes; its first formal/resource audit failure is preserved separately. The new controller/restoration/cycle source pins are now sealed to that actual archived restoration. Source-only metadata and read-only service preparation PASS;135 source pins, owner plan `ba7556ffddd28323ce0a7541cc1c6a8557378a687fbdace981c610bfc7b2129d`, PREPARED journal `2ecfb7d0ad0da2dd1441352e0850f3ef376a7d076f352d781fcbfbb694368c59`. Original service PID19019 remains running at registration. No actual route HTTP phase has yet run.

The initial173-test CPU run had one fixture failure; only that fixture changed and the entire12-test restoration module then passed. Production sources are unchanged across those runs. [Both CPU attempts](../../../docs/experiments/m5-ultra-256gb/next-modules-prepared-cpu-v1/README.md) are retained.
