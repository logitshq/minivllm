# Reviewable temporary oMLX cold restart

2026-10-17. **Proposal only; no service mutation was executed and no executable mutator was created.** The exact command data and restoration contract are in [plan.json](omlx-cold-restart-plan-v1/plan.json); source and redacted launchd evidence are in [evidence.json](omlx-cold-restart-plan-v1/evidence.json). This replaces neither the existing SIGSTOP/SIGCONT lease nor its receipts.

The concrete no-auth-change option is to remove **one** launchd job, run an independently admitted experiment, bootstrap the unchanged original plist, and explicitly reload the original physical model. It changes the server PID and loses in-memory requests/KV. The available evidence supports restoring persisted configuration plus the visible VLM/MTP/resident-PLE variant; it does **not** prove that an unexposed transient runtime signature can be restored exactly. Existing user approval for pausing/resuming PID 74384 is not treated here as approval for terminating/restarting it.

## Decision ready for the user

Recommended option to approve: **allow this cold restart with the explicit persisted-visible recovery contract below**. No API key, password, new login, authentication setting, model file, service plist, quantization setting or model setting needs to be changed. No credential should be pasted into chat.

Suggested exact approval question:

> 可以临时停止并移除 `gui/501/local.omlx.server`，释放其约 143 GiB 内存，实验清理后用原 plist 重启并加载原 `Qwen3.8-Flash-Next-oQ6e-mtp` 吗？这会换 PID、丢失暖 KV/在途请求，退出最多等 5 秒，因此不保证所有 SSD 缓存写入完成。恢复将核对原配置、VLM、Lightning MTP 和 resident PLE；隐藏的临时运行参数无法证明逐项一致。若恢复模式不符，会明确报告未恢复完成，不改配置强行通过。

If the user requires graceful model unload or refuses cold restart, the minimum additional capability is an already working **official admin unload session/interface** for this service, operated through its normal UI/session. Anonymous admin GET currently returns 401, the main API key is not configured, and normal login cannot manufacture access from that state. We will not set an API key, read the signing secret, forge a cookie, or ask for the secret in chat. A legitimate admin session would permit the admission-locked unload path and retain the process; it still does not expose the old hidden variant signature.

If exact hidden-runtime restoration is required, the missing information is the original exposed profile and transient engine settings (including any custom MTP depth/offload settings), or an official runtime export that actually exposes that signature. Persisted settings and a model name cannot reconstruct unknown values. Neither restart nor admin unload can promise warm KV continuity.

## Known restoration identity and limits

| Item | Observed value / contract |
|---|---|
| Original process | PID 74384, `Fri Oct 16 08:01:58 2026`, `omlx-server`, parent launchd |
| Job | `gui/501/local.omlx.server`, running in saved 10:05 UTC receipt |
| Original plist | `/Users/ryanli/Library/LaunchAgents/local.omlx.server.plist` |
| Plist SHA-256, read this review | `6cfa472c4ae61ac8290f39f447d00d7d1e025c4dd7d35b8faf63b13d1fc76bdb` |
| Persistence | KeepAlive=true, RunAtLoad=true, ThrottleInterval=30; on-disk definition is unchanged |
| Listener/version | `127.0.0.1:8001`, runtime oMLX 0.7.0 |
| Actual runtime installation | `~/.local/share/uv/tools/omlx`, not the experiment baseline venv |
| Physical loaded set | Exactly `Qwen3.8-Flash-Next-oQ6e-mtp` |
| Path | `/Users/ryanli/code/models/mlx/Qwen3.8-Flash-Next-oQ6e-mtp` |
| Persisted target settings | `mtp_enabled=true`, `max_context_window=32768`, `ttl_seconds=null`; no recorded active profile; profiles file absent |
| Model-settings SHA | `b474a171fe3022027842af02ee021206289a7cd9306720841da375029cf42acb` |
| Visible original load | VLM / qwen4_exp, Lightning MTP enabled, resident PLE; load transaction Oct 16 08:02:37–08:02:59 |
| Unobservable | Full `EngineEntry.runtime_settings_signature`, historical transient overrides, warm request/KV state |

These saved status observations are not a fresh lock on the service. The controller must refresh the idle/identity/physical-loaded-set checks immediately before mutation and refuse if they differ. It must not infer that every currently installed source file is identical to Python bytes loaded by the old process yesterday. Source receipt hashes identify on-disk files for the planned restart. The current review never read credential values or retained process argv/environment values.

**Exit-time conflict:** the actual loaded job's saved `launchctl print` record says `exit timeout=5`. The local Apple `launchd.plist(5)` manual defines this as the interval between SIGTERM and SIGKILL when stopping a job. oMLX's teardown can use 60 seconds (`utils/fatal.py`); its pool shutdown waits for pending unload/lease cleanup before engine teardown. Consequently bootout can end the process before graceful SSD writer completion. Checkpoints and configuration are not deletion targets, but warm/pending SSD cache persistence cannot be guaranteed. This consequence is part of the requested cold-restart approval, not silently hidden behind the word “pause.”

## Exact transaction to review — do not execute before approval

All actions target the current user session. Never run `bootout gui/501` without the service name: that would target the domain. Never use `disable`/`enable` (persistent launchd state), `kickstart -k`, plist edits, or the unsafe public-unload fallback to implement this proposal.

1. **Write a durable restore record outside `~/.omlx`.** Include the plan SHA, approved contract, current UID=501, PID/start/parent, launchd path/state/PID, listener owner, runtime version, exact physical loaded set/path, selected settings and model-settings hash, target config/index identities, current file stats for global settings, and starting log offsets/inodes. Do not copy or hash global credential values. Require active/waiting/loading=0 and no existing experiment GPU owner. Refuse config/source/plist drift or any additional resident model. Reserve an independent recovery owner before the first mutation, with no unconditional reload while an experiment process is alive.

2. **Remove only the service definition and stop its process.** The approved mutation is exactly:

   ```text
   /bin/launchctl bootout gui/501/local.omlx.server
   ```

   Save command status and sanitized errors. Then independently require the job to be absent, original PID/start identity gone, and no listener on port 8001. A failed command or timeout is an unknown outcome until these checks reconcile it. Do not send additional kill signals as an unreviewed fallback. This definition removal prevents KeepAlive from racing a replacement into the experiment. The plist remains on disk.

3. **Verify reclamation before any experiment.** Mark the record `service_removed_not_restored`. Obtain system physical-footprint/pressure/swap samples and prove no replacement or original owner survived. Do not equate “PID gone” with instantaneous complete driver reclamation. Pressure is currently 2; wait for an independently specified stable admission window. The released ~143 GiB does not prove the ~190 GiB GLM loader peak fits. If GLM's separate gate cannot pass, skip GLM and enter recovery immediately. No automatic lower precision, expert offload or different checkpoint substitution.

4. **Run only the already reviewed bounded child, if admitted.** Keep the job absent throughout. A foreign port owner, replacement job/PID, pressure violation or timeout stops the experiment. Reap the child and verify its physical allocations reclaimed before recovery. A roughly 190 GiB experiment and roughly 143 GiB original service must never be loaded concurrently by cleanup.

5. **Restore the unchanged job after reaping and memory checks.** Require the exact plist SHA and model/settings identities still match. The approved mutation is exactly:

   ```text
   /bin/launchctl bootstrap gui/501 /Users/ryanli/Library/LaunchAgents/local.omlx.server.plist
   ```

   Expect a **new** PID; bind it to the job, start time and port owner. Preserve original RunAtLoad/KeepAlive behavior. Poll bounded read-only `/health`, `/api/status` and `/v1/models/status`. The original model is unpinned; startup preloads only pinned models, so a healthy server alone is not successful restoration. A bootstrap error is reconciled with job/listener state before any retry. Do not use another command that starts duplicate servers.

6. **Explicitly load the original physical model once.** After management readiness and budget admission, use the approved HTTP mutation with no settings body:

   ```text
   POST http://127.0.0.1:8001/v1/models/Qwen3.8-Flash-Next-oQ6e-mtp/load
   ```

   Existing public API access was anonymous; fresh 401 means stop and report an authentication mismatch, not read/configure credentials. This official route invokes `get_engine` with persisted settings. Use a bounded load deadline (proposed 600 seconds) and reconcile loading/loaded status after any network timeout: never start overlapping retries. Before a single reconcile-driven retry, require no active loader and a documented retry cause. Do not force resident PLE or modify settings following 507/timeout.

7. **Verify the complete persisted-visible contract.** Require exact physical loaded set/path, new PID owned by the original launchd label, runtime 0.7.0, VLM qwen4_exp engine, matching default/pinned/model-settings state, no loading/active/waiting work, and healthy management responses. Read only new log bytes after the saved offset (handle inode rotation explicitly). Match a new target load transaction containing **Lightning MTP enabled**, **PLE mode resident**, resident table fusion, and final successful VLM/model load. The old log lines must not satisfy this check. Require stable physical memory and no experiment owner. An optional functional generation probe is a separate approved/qualified action, not invented here.

8. **Close only with a truthful status.** Success is `restored_persisted_visible_variant`, never `same_pid_restored` or `exact_runtime_signature_restored`. Missing MTP evidence, SSD/mmap PLE, unexpected residency/config, unavailable management or unresolved timeout means `recovery_incomplete`; retain the journal and notify the user. A partial bootstrap is not model recovery. Avoid repeated restart/load loops or “always reload” cleanup while memory is unreclaimed.

The recovery commands should also be displayed to the supervising user before the transaction. A crash-safe journal must distinguish `prepared`, `service_removed`, `experiment_running`, `experiment_reaped`, `job_restored`, `model_loading`, `restored_persisted_visible_variant`, and `recovery_incomplete`. Operator recovery always begins by inspecting the recorded experiment PID/start identity and current memory, not by blindly executing the load command.

## Resident PLE cannot be guaranteed in advance

Actual `engine_pool.py:661-755` can select SSD PLE from the live admission ceiling without changing persisted settings. `server.py:3479` public load does not provide an exact historical transient-variant argument. Therefore this proposal cannot guarantee resident-mode recovery merely from unchanged files. It minimizes the risk by reaping the experiment first and demanding the **new** resident-PLE load evidence. If resident recovery remains impossible, it reports the failure instead of claiming success or silently accepting a slower SSD variant.

This is why the requested choice is approval of the bounded cold-restart/recovery contract, or a legitimate official admin-unload interface if graceful unload is required. An admin interface reduces shutdown risk but does not turn hidden runtime state into observable evidence or eliminate the independent GLM peak-budget gate.

No full-model experiment is authorized by this document. No script that stops, authenticates, unloads, loads or restarts a service was run or created during this preparation.
