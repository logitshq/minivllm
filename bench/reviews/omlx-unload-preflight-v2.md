# oMLX unload/reload preflight v2

Read-only observations, 2026-10-17 10:05–10:11 UTC. This continues [the actual-service unload review](omlx-temporary-unload.md). No unload, load, login, authentication change, signal, restart, inference, model import or download was performed. Small redacted receipts are in [omlx-unload-preflight-v2/](omlx-unload-preflight-v2/); their exact SHA-256 values and reviewed on-disk source identities are in `manifest.json` there.

**The launchd job is confirmed running. The current physical model and its VLM/MTP/resident-PLE load are identifiable. The preferred admin-unload route is not currently anonymously usable, and the running service has no configured main API key with which to perform its normal login.** A valid pre-existing admin session could change this, but none was examined or acquired. This preflight does not authorize an authentication/configuration change or claim exact recovery of an unexposed transient runtime signature.

## Service and authentication evidence

| Observation | Result | Receipt |
| --- | --- | --- |
| `launchctl print gui/501/local.omlx.server` | Exit 0; top-level `state=running`, `pid=74384`, `runs=4`, `keepalive \| runatload`, minimum runtime 30 seconds | `service-state.json` |
| `ps` identity | PID 74384, parent 1, start `Fri Oct 16 08:01:58 2026`, comm `omlx-server`, state S | `service-state.json` |
| Anonymous `GET /api/status` | 200; runtime 0.7.0; exactly one physical loaded ID; zero loading/active/waiting requests | `management-get.json` |
| Anonymous `GET /v1/models/status` | 200; matching physical path, VLM engine, `qwen4_exp`, loaded, not pinned | `management-get.json` |
| Anonymous `GET /admin/api/models` | 401; response body not read | `management-get.json` |
| Global settings auth metadata | `auth.api_key` exists with JSON type null; `auth.secret_key` is a nonempty string; only names/types/presence retained | `auth-field-metadata.json` |
| Anonymous `GET /admin/` | 200; exactly one `loginForm(false)` in the login template; runtime main API key is not configured | `login-page-metadata.json` |

All GETs used loopback `127.0.0.1:8001`, no credentials and a two-second timeout. The login-page probe did not follow redirects, use a cookie jar, or retain HTML. The installed login-page handler computes that boolean from live `global_settings.auth.api_key`; it is stronger evidence than the disk file alone. No running process arguments/environment values, browser sessions or credential stores were inspected.

The lexical global-settings inspection decoded object keys only, skipped scalar string values, and emitted direct auth field names/types plus nonempty status. It did not decode, copy into a credential field, hash or output credential values. The settings file itself was not archived or hashed. `target-config.json` predates this additional check and truthfully records that its own earlier operation did not read the global file.

The actual installed `omlx/admin/routes.py:1906` login route is `POST /admin/api/login`, accepting `api_key` and optional `remember`. It uses the **main** key only; missing main key returns 400 before authentication. `admin/auth.py:113,157` compare supplied and configured UTF-8 strings directly with `secrets.compare_digest`; this is not a stored password-hash login. `auth.secret_key` signs sessions and is not the login credential. Creating a session locally from that signing secret would bypass the official login and is not proposed. Setup would modify authentication configuration, outside this preflight and the preserve-configuration restoration contract.

`require_admin` accepts a genuine session cookie or configured loopback skip-auth; a Bearer key alone does not satisfy admin unload. The observed anonymous 401 rules out using its skip-auth branch now. Public inference/status being anonymously available does not establish admin authorization. No evidence establishes whether some existing user-owned browser session is still valid, and this review did not search for one.

The launchd record is now evidence of the actual loaded job, beyond the earlier on-disk plist. Repeated resource/jetsam `type/state` fields in the filtered receipt are nested metadata, not additional services. KeepAlive means a fatal engine-teardown exit may create a replacement PID; the controller must abort experimental admission on an identity change. A replacement process does not itself restore the original unpinned residency.

## What is known about the physical resident variant

The exact physical ID is `Qwen3.8-Flash-Next-oQ6e-mtp`, path `/Users/ryanli/code/models/mlx/Qwen3.8-Flash-Next-oQ6e-mtp`. Its reported `actual_size` is 150844290256 bytes and tracked/resident estimate is 158077204528 bytes. These model accounting fields are not process physical footprint. Root's prior physical measurement was 153346982168 bytes, about 142.82 GiB; do not substitute RSS when budgeting the approximately 190 GiB GLM checkpoint.

Only the exact configured service stdout/stderr and its `server.log` were inspected, each bounded to the last 1 MiB. The current files fit inside that bound, stdout is empty, and no unrelated logs were traversed. `target-load-logs.json` stores target lifecycle lines, offsets and source-file identities; `load-window.json` stores only relevant selected lines from the target's load transaction. On 2026-10-16, local service log time:

- 08:02:37.366: begins loading this physical model.
- 08:02:37.382: `Qwen4-Exp PLE mode ... resident`; Lightning MTP enabled, checkpoint layout `mtp.`.
- 08:02:43.016–.017: exact hybrid projection pairs skipped because Lightning MTP verification is enabled; 100 compiled decode paths; one resident PLE table fused into a packed device embedding.
- 08:02:59.714–.715: VLM engine and physical model loaded.

No later target unload or runtime-variant-reload message appears in the bounded files. This supports continuity of that observed load, but absence of a log line is not a proof that every runtime setting was captured.

The target's persisted record is present in `~/.omlx/model_settings.json`: `mtp_enabled=true`, `max_context_window=32768`, `ttl_seconds=null`; other target field names are `is_default`, `is_favorite`, `is_pinned`. There is no `active_profile_name` field and `~/.omlx/model_profiles.json` does not exist. The per-model file SHA-256 is `b474a171fe3022027842af02ee021206289a7cd9306720841da375029cf42acb`; the canonical target-record SHA-256 is `30bbdb34ee71855cdef122ee6d3bd0501b441376ec75751750bd3a68112ddc16`. The file modification time predates this model load. These facts make a persisted-settings reload plausible; they do not prove a transient profile was never applied in memory.

**Minimum evidence-backed recovery signature:** same physical ID/path, actual service version and launchd/PID identity, unchanged persisted model configuration, VLM `qwen4_exp`, Lightning MTP enabled and resident PLE. Exact `EngineEntry.runtime_settings_signature`, MTP depth and all other dynamically resolved settings are not exposed by the public status receipts. There is no claim to reconstruct that entire hidden signature, a historical transient profile or warm in-memory KV/request state.

Resident PLE needs an explicit post-reload check. `engine_pool.py:661–755` can force SSD PLE through effective settings without changing the persisted file. Normal resolution uses a stable memory ceiling, but pre-load admission can supply its live ceiling. Consequently an unchanged config hash alone does not prove resident-mode restoration. Reap the experimental owner and reclaim its memory before reload, then verify the new load's resident-PLE and Lightning-MTP log messages. Do not change precision or force residency to make a failed memory admission appear successful.

## Minimal proposed controller transaction

This is an interface proposal, not an executed transaction. The existing SIGCONT-only lease is insufficient.

1. **`prepare_restore_record()`**: durably save original PID/start/launchd/listener identity, physical loaded set and exact paths, service version, model config hashes, and the observed variant signature above. Recheck idle/loading state. Keep the record outside the server's configuration directories. Require a valid official admin session before proceeding; currently this is an unresolved capability, not a request to set up a new key.
2. **`unload_original_set(admin_session)`**: sequential `POST /admin/api/models/{url_encoded_physical_id}/unload`, no settings body. It calls `request_unload`, which locks/blocks new admission during unload; 202 means pending. Record status and poll public status until the exact original engines are absent and loading/active/waiting counts are zero. Never silently replace this with public `/v1/models/{id}/unload`: that route directly calls `_unload_engine` without the same admission contract. An idle GET does not eliminate the race.
3. **`verify_reclaimed_then_pause_original()`**: recheck process identity and measure physical process/system memory until stable; tracked model memory zero is insufficient. Root may then use its existing authorized suspension of the same empty process. Verify it is stopped and memory remains within the independent GLM admission budget. Empty-status-to-pause is not atomic, so a reload/race, new identity or unexpected footprint fails admission. Persist `unloaded_not_restored` before experimental work.
4. **`run_guarded_experiment()`**: independent owner/pressure/memory controls and durable cleanup. Removing the old model does not prove GLM's sanitize/materialization peak fits. Do not admit GLM on checkpoint size alone.
5. **`reap_experiment_then_restore()`**: prove the experimental child is gone and its physical allocations reclaimed before resuming/loading the original roughly 143 GiB service. Resume only the original identity. Sequential public `POST /v1/models/{url_encoded_physical_id}/load` constructs persisted settings; retain failures/timeouts and reconcile status after unknown outcomes. No blind overlapping retry, restart or automatic “always reload” while GLM survives.
6. **`verify_restored()`**: exact original physical loaded set/path, same process and runtime version, unchanged model config, no loading, original default/pinned state, and new load evidence of VLM + Lightning MTP + resident PLE. Record stable physical memory and healthy management responses. Only then mark `restored_persisted_variant`; do not name it `exact_runtime_signature_restored`. Retain an unresolved recovery record on failure.

Before an actual transaction, two material gaps remain: official admin authentication is unavailable from the observed current configuration/session, and the full prior runtime signature is unobservable. A valid existing session would address the first without changing settings; accepting the explicit minimum recovery signature addresses the second only at that stated scope. Neither is resolved by this read-only review. A root controller may continue other Qwen experiments while deciding these prerequisites.

The source hashes identify the actual service installation's on-disk files, not a proof of Python bytes imported before a possible on-disk update. No baseline-environment source was substituted. Logs are operational evidence, not model-output correctness evidence or a GPU performance result.
