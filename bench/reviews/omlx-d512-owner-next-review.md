# oMLX D512：最小外层接线复核

2026-10-17，源码只读结论；未运行测试、runtime imports、HTTP、服务或 payload hash。目标仅 C1×8、C8×16 的 D512 observed qualification；不是正式性能点。

**推荐：新窄 oMLX 两阶段 owner 同进程直接 `await trial.run(spec, folder/'trial', checkpoint, backend=Tracked())`，复用 Lithos retry 的所有权/回收算法。** 阶段固定 `omlx-c1 → omlx-c8`。不要用 `cold_cycle_v2.owner` Popen 一个 trial wrapper：它的 `run` 仅对自己的直接 child 调 `resources.snapshot(label, child.pid)`、TERM/KILL/wait；wrapper 再生 server 会形成孙进程，现协议不能授予其资源或回收闭合。

## 可复用接口与必须保留的接线

| 来源 | 最少复用方式 |
| --- | --- |
| `omlx_d512/trial.py:prepare/validate/run` | 已有官方命令、真实 cwd、私有 settings/model symlink、Bearer readiness、frozen SSE client、closed raw join。直接调用，不再写 client 或模型 runner。 |
| `lithos_d512_retry/owner.py:System`、`prior_replacement`、`work_seconds` 的算法 | OS backend/独立 postflight deadline、不可变 predecessor 解析、阶段 budget 算法可沿用；`work_seconds` 的 `trial_names` 绑定需改成真实 oMLX 阶段，不能直接调用后伪造 labels。 |
| 同文件 `run` 内 `spawn_impl/spawn/release/trial_one/checkpoint` | 保留其算法的窄 vendor 适配；这些是闭包，不能作为现成独立函数 import。`Tracked.spawn(command, environment, stdout, stderr, *, cwd)` 调实际 backend，立即保存 Popen 和 PID，再验 `ppid==owner PID`。资源 checkpoint 的 pid 来自 trial 当前 server，不能填 owner/wrapper PID。 |
| `pareto_realistic/qualify_absent_owner.py` 的 `absolute/durable/journal_lock/absence/load_resources/System` | 可按原真实 module 路径和 SHA 直接复用。`load_resources` 可使用固定 E023 plan 的资源源码 pins；这只复用资源策略，**不是**验证 oMLX runtime/model。 |
| `resources_v2.snapshot/guard/deltas` | 保持 pressure1、AC、thermal、swap 原门；`suspended_pids=()`，每个已知 inference process 必须为该次真实 server PID。按不可变 predecessor 的 original PID 核 service/job/8001 absence。 |
| `lithos_d512_retry/owner.py:full_prepare` | 稳定 fd、每块 deadline、stat/SHA 前后验证和 frozen-source 写盘算法原样保留；函数依赖硬编码 `GROUPS`，不能直接复用再猴补其 globals。只给新 owner 的真实组定义做窄复制/适配。 |

`checkpoint` 保持 `asyncio.to_thread(probe)` + shield/settle：取消先等待正在进行的有界资源 probe 结束，再传播取消，避免回收后仍有后台写证据。outer `live` 保留实际 Popen，即使 trial 内层失败，finally 仍能补做 TERM15→KILL15→wait、严格 PID 和 listener absence。canonical `phases/{phase}/{intent,child,release}.json` 与内层 `trial/{spawn-intent,child,release}.json` 绑定同 PID；known identity:null 仍须真实 wait/reap；未知 Popen 结果不能写 NO_CHILD。首阶段及下一阶段 pre-Popen 拒绝使用已有明确 no-popen 证据规则。

## 真正 vendor-specific 的最小差异

- `validate_plan/stopped` 的 schema、真实两个 phase、实际新 owner/controller/source pins；不改 `HERE/__file__`，不伪装旧 journal。controller 必须理解两个 direct-server phases 和内层 trial receipts；可窄复用已审多阶段恢复协议，不能直接用 stable single-child reaper。user cold-stop/restore 授权延续，predecessor 在执行前绑定届时最新已审 RESTORED journal+audit。
- trial 成功门改为 `E023-omlx-D512-trial-result-v1`，route/count/text、cleanup/client task resolved、small pins、真实 raw-join 和 request-seal 完整；不要求 Lithos `generation-log-check.json`、pack、FIFO 或 `eval_count−EOS`。oMLX usage 本身是 committed count；正常 EOS 的 raw terminal 由 observer 另存。
- 官方 uvicorn 可能在正常 shutdown 后重发本进程收到的 SIGTERM。仅保留 trial 已实现的 `0` 或 **自身发 TERM且返回−15** 的解释；−9 或未解释退出不可授资格。seal 只证明实际请求/raw/SSE DONE 绑定，不是 EOF、进程关闭或全部 ASGI/lifespan 任务成功；完整 stdout/stderr 与外层真实 exit/reap 必须保留并独立检查已知异常路径。
- source evidence 使用 `serve_probe_v2.py/join_v2.py`，不能退回旧 probe：新版修了真实 Pydantic `temperature=0.0`/不支持的 `n`、loaded VLM/source identity、cache disabled 和 seal 后异常。`/v1/models` readiness 只证可见模型，不能替代首实际请求的 loaded-class/model receipt。

## Pin 清单已有范围与缺项

**模型已有完整注册清单：** E023 `pareto_realistic/prereg-qualification-v1/plan.json` SHA `21ba52…47fb` 的 model_manifest 列同一 Qwen3.8 NVFP4 snapshot 全13文件；E020 `external_spec_http/prereg-v1/plan.json` SHA `e3f0c3…e2bdc` 有这些 payload SHA/stat 和完整 `qualified_model_settings['omlx-none']`。本次只 target-only，不需要 drafter 或 Lithos pack。

**oMLX 已有可继承的明确 inventory，但不能称全环境封装：** E020 `pins/pin_stats` 合并 E015 的 `omlx/mlx/mlx-metal/mlx-vlm/mlx-lm` distributions，以及 HTTP 的 `fastapi/starlette/uvicorn/pydantic/transformers/tokenizers`，另 pin resolved Python binary 和 `bin/omlx`。`official_mtp/qualify.py:freeze_distributions`、`external_spec_http/prepare.py:runtime_files` 都只枚举指定 distributions，没有递归 dependency closure；不能因这份清单存在就宣称 pydantic-core/anyio 等传递依赖全覆盖。新 trial 仅 hash 小 source/tokenizer，不能代替这些 runtime 前后检查。

**最小新 verification 用四个真实组：** `omlx_runtime`（该专用 venv 的已继承 binaries/packages + 新确认的依赖）、`omlx_source`（实际官方 probe seams/CLI/uvicorn 等源码，可与 runtime 重叠但 identity 必须一致）、`target_model`（13文件）、`outer_runtime`（实际外层 Python + HTTP/tokenizers 依赖）。本地 adapter/controller/client pins 仍在 execution.pins；无需为凑 Lithos 五组捏造 pack。在注册准备阶段通过 distribution metadata 扩齐实际专用 venv HTTP/tokenizer 依赖，保留当前版本及 inherited SHA 冲突检查；不能静默更新包或改量化。现 Lithos formal 的 outer_runtime inventory 可作为当前外层候选输入，只有 sys.executable.resolve、版本/metadata、所有实际 stat/identity仍匹配才复用。

执行前 owner `full_prepare` 与执行后 postflight 真正重哈希所注册 runtime/model；此说明未做那些验证。OS shared libraries/stdlib 继续明确未独立全封装。关闭后独立 join 审全部24请求及 C8真实 batch≥2、资源审真实 server 生命周期、恢复审真实两阶段回收。任一缺失就不授资格，更不输出 Pareto 点。
