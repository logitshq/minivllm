# Temporary resident-model unload: actual PID 74384

Static review, 2026-10-17. No HTTP mutation, inference, credential read, model import, signal, restart, or download was performed by this review. This is a proposed controller transaction, not a completed unload or permission to claim GLM fits.

The official API can unload the current engine while retaining model files and configuration. It requires a stronger recovery transaction than the existing SIGSTOP/SIGCONT lease: record the original physical loaded set first, verify physical memory reclamation, and reload that set only after the experiment process and its memory are gone. SIGSTOP alone preserves the current 142.82 GiB physical footprint.

## Actual service and observed resident set

Root's read-only receipts are in `.artifacts/m5-ultra-256gb/glm-flash-physical-preflight-v1/`:

- `service-location.json`: PID 74384, original start identity `Fri Oct 16 08:01:58`, listener `127.0.0.1:8001`, cwd `/Users/ryanli/.omlx`; loaded extensions belong to `/Users/ryanli/.local/share/uv/tools/omlx/lib/python3.13/site-packages/`. Interpreter is `/Users/ryanli/.local/share/uv/python/cpython-3.13.16-macos-aarch64-none/bin/python3.13`.
- `status.json`, from `GET /api/status`, reports runtime **0.7.0**, one physical loaded ID `Qwen3.8-Flash-Next-oQ6e-mtp`, zero loading/active/waiting requests, and that same default model.
- `model-status.json`, from `GET /v1/models/status`, reports that physical ID at `/Users/ryanli/code/models/mlx/Qwen3.8-Flash-Next-oQ6e-mtp`, `engine_type=vlm`, `pinned=false`, `loaded=true`, `is_loading=false`, `estimated_size=158077204528`, `actual_size=150844290256` bytes.
- `footprint.json` and `raw.json`: service physical footprint **153346982168 bytes = 142.82 GiB**, despite RSS about 33.88 GiB. System footprint is 187547942912 bytes; normal pressure and about 268.81 MiB swap are snapshots, not a load-peak bound. Do not add process footprint to system footprint: the latter already includes the former.

The pinned GLM checkpoint is about 189.98 GiB before loader temporaries/cache. Existing service footprint plus those raw weights exceeds 256 GiB; RSS was not an adequate admission measure. Even after unload, the GLM expert stacking/materialization peak remains unqualified; see `glm-flash-load-budget.md`.

`/v1/models` is discovery, not residency. Use `/api/status.loaded_models` as the physical ID set and join those exact IDs to `/v1/models/status.models`. The status route also creates exposed-profile rows by copying base `loaded` flags; do not treat all such rows as independent loaded engines. These are point-in-time readings, not a reservation against a new request.

This review reads the service's actual installed 0.7.0 package, **not** `.artifacts/omlx-baseline` 0.7.1.dev1. Runtime `/api/status.version` agrees. The lsof environment and runtime version establish provenance but do not prove every Python file on disk is byte-identical to code imported before a possible package update. No debugger or live source introspection was used.

## Existing launchd recovery configuration

The on-disk `/Users/ryanli/Library/LaunchAgents/local.omlx.server.plist` has `Label=local.omlx.server`, `RunAtLoad=true`, `KeepAlive=true`, and `ThrottleInterval=30`. Its executable is `/Users/ryanli/.local/bin/omlx`, working directory `/Users/ryanli/.omlx`, and stdout/stderr paths are `~/.omlx/logs/service.stdout.log` and `service.stderr.log`. This review emitted no argument values or environment values; only the environment key `PATH` is present. The job has four program arguments. A plist existing on disk does not prove this PID currently belongs to the loaded launchd job; a controller can confirm that separately with a narrowly filtered read-only job status.

With this job actually loaded, an exit can trigger launchd replacement under its restart throttle. Thirty seconds is a throttle, not a promised recovery deadline. SIGSTOP does not exit the process. Therefore a teardown fatal-exit can turn into a new live service while a PID-only watchdog is looking for 74384: abort experiment admission on identity change and reconcile the launchd job/listener before any recovery decision. Do not disable KeepAlive, bootout/bootstrap, kickstart, or rewrite the plist for this experiment. Existing startup code preloads pinned models; the observed resident model is unpinned, so a launchd restart alone must not be counted as restoration of its previous residency. Even a healthy replacement needs the explicit original-set reload checks and is a changed-identity recovery, not success of the original-PID lease.

## Exact routes and authentication

Paths below are relative to `http://127.0.0.1:8001`; encode the physical model ID as one URL path component. Load/unload routes have no model-settings request body.

| Request | Actual 0.7.0 behavior | Authentication and use |
| --- | --- | --- |
| `GET /health` | Health and pool totals; no precise resident set/version | Unauthenticated. Insufficient alone. |
| `GET /api/status` | Runtime version, physical `loaded_models`, loading/active/waiting counts | `verify_api_key`; root already obtained 200 without credentials. |
| `GET /v1/models/status` | Physical model path, sizes, type, loaded/loading, pinned; adds virtual/profile rows | Same API authentication. Root already obtained 200 without credentials. |
| `GET /admin/api/models` | Per-model settings and exposed profiles as well as model metadata | `require_admin`: existing admin cookie or configured loopback skip-auth. If used, retain only needed model fields; do not dump unrelated settings. |
| `POST /admin/api/models/{id}/unload` | `request_unload`: pool lock, busy/loading checks, pending-unload admission block, abort/drain then unload | **Preferred controller route.** Cookie or configured loopback skip-auth; a Bearer key alone does not satisfy this route. 200 means completed; 202 means pending, so poll until absent. Busy unload defaults to aborting active requests; the route exposes no graceful-only flag. |
| `POST /v1/models/{id}/unload` | Direct private `_unload_engine`, without the `request_unload` lock/admission sequence | API auth. **Not interchangeable with the admin route.** A preceding zero-active snapshot cannot close its request-admission race. Do not silently fall back to it if admin authentication fails. |
| `POST /v1/models/{id}/load` | `await pool.get_engine(physical_id)`; blocks until engine starts, or returns already loaded | API auth; suitable persisted-settings reload. 404 missing model, 409 unavailable/loading/busy, 507 memory admission, 500 other failure. |
| `POST /admin/api/models/{id}/load` | Same physical engine construction | Admin cookie, valid main/sub Bearer, or configured loopback skip-auth. Unlike admin unload, Bearer is accepted. |

API no-auth success does not prove admin no-auth: a loopback service with no API key and a service with explicit skip-auth are distinct branches. Establish admin access with a read-only admin GET if necessary. Do not read credential values, change auth settings, or include cookies/Authorization in evidence. Do not use model deletion, `/admin/api/reload`, cache clearing, settings writes, or service restart as substitutes.

Source anchors under the actual site-packages root: `omlx/server.py:323,2945,3107,3413,3463,3479`; `omlx/admin/auth.py:257`; `omlx/admin/routes.py:2311,2577,2611,2645`; `omlx/engine_pool.py:1413,1953,2061`.

## What unload releases, and what it cannot promise

The VLM wrapper stops its engine and drops the model, processor, adapter, tokenizer and MTP/draft references. EngineCore closes the scheduler on its owning executor, flushes and closes the SSD cache manager, clears collectors and model ownership, then synchronizes and reclaims MLX memory on that executor. `scheduler.shutdown` explicitly preserves paged SSD files for reload; `deep_reset` discards in-memory model/cache references. This does not delete checkpoint files or rewrite model settings. In-flight request and in-memory KV state are not resumable after unload; SSD cache persistence is best effort within the teardown budget.

`engine_pool._stop_and_unload_engine` also drops pool references, runs GC, synchronizes/clears the global MLX cache, and compares active-memory and physical-footprint deltas. Its nominal settle loop has ten 0.5-second retry intervals and three additional one-second emergency rounds. **A failed settle barrier logs warnings but still clears the pool's tracked model charge.** Consequently HTTP 200, `loaded_models=[]`, and tracked memory zero do not establish reclaimed physical memory. Measure actual service/system footprint again after completion, require stable observations and no new loading/requests, and re-run the GLM physical admission calculation before any experiment load.

Teardown is not guaranteed to preserve the process. `EngineCore.close` calls `fatal_exit` on MLX teardown timeout/failure or a surviving SSD writer; this is `os._exit(70)`, with a 60-second fatal-teardown constant and progress-aware budgeting. If PID identity changes/disappears, stop the transaction before GLM; a supervisor-created replacement must not be treated as the original PID or signalled under the old lease. A network timeout also has an unknown outcome: `_unload_engine` shields its teardown task from caller cancellation. Reconcile read-only state before retrying.

Source anchors: `omlx/engine/vlm.py:2884`, `omlx/engine_core.py:1236`, `omlx/scheduler.py:13757,13814`, `omlx/engine_pool.py:2936,2954`, `omlx/utils/fatal.py:12`.

## Recovery transaction for the root controller

1. **Write the restore record before the first mutation.** Store exact PID/start identity, port owner, version, physical loaded IDs and paths, model engine types, default/pinned state, and non-secret configuration identity. Preserve existing settings/model files. If configuration hashes are captured by the controller, persist only hashes, never raw credentials. The current observed set contains one model above. Confirm its effective engine uses its persisted settings before claiming exact variant restoration.
2. **Unload through the admin admission path while the server runs.** Prefer an idle window, but still use the route with `request_unload`. Poll both status endpoints after 200/202 and verify the original engine is absent, no engine is loading, and no active/waiting work remains. The status counts inspect async engine collectors and are not a universal global admission lock. No global maintenance/reservation route was found in the reviewed server/admin routes.
3. **Measure reclaim, then suspend the same empty service.** Recheck PID identity throughout. Incoming inference can reload an unloaded model before suspension, so a refreshed empty snapshot is necessary but not atomic. After root applies its existing authorized suspension, obtain another physical-footprint sample; any unexpected footprint, loading evidence, identity mismatch or residual experiment process fails admission. Keep the durable record marked `unloaded_not_restored` until reload verification finishes.
4. **Run only after an independent GLM load-budget gate.** Removing the current service's residency creates headroom but does not bound GLM sanitize/stack/eval peak. Retain the existing experiment owner, pressure/swap/footprint monitoring and fail-closed cleanup. Do not lower precision or alter official baseline settings to force admission.
5. **Reap the experiment before attempting reload.** Finally/recovery must first verify the experiment child is gone and its allocated memory has been reclaimed. Only then resume the same original PID, wait for management readiness, and load each original physical ID sequentially through `/v1/models/{id}/load`. Do not load the original roughly 140 GiB engine while a roughly 190 GiB GLM owner survives. Reload itself also needs measured headroom and a bounded long timeout; timeout/507 is not a successful restoration.
6. **Verify the complete result.** Reconcile exact `/api/status.loaded_models` set, model paths, original version/PID identity, `is_loading=false`, unchanged default/pinned/settings identity, and physical memory stabilization. Check the whole set after all loads because the pool may evict another engine during admission. Keep response/status timestamps and all failed attempts. Mark restored only after these checks; otherwise retain the recovery record and report the failed restoration, without claiming success or automatically restarting/reconfiguring the service.

The existing SIGCONT-only lease does not implement steps 1, 5 or 6. A watchdog must understand the experiment owner and the durable unload/reload state; a generic “always reload on exit” hook could create the very memory collision this procedure avoids.

**Exact-variant limitation:** inference can call `get_engine` with exposed-profile transient `runtime_settings` or an LM-only variant; ordinary load resolves neither profile nor transient settings. `EngineEntry.runtime_settings_signature` is not returned by the reviewed status endpoints, and `engine_type` alone does not prove the loaded settings variant. Admin model settings describe persisted state, not necessarily the already-loaded variant. If a transient variant was in use, original physical ID + persisted configuration is insufficient evidence of exact restoration; resolve that provenance before mutation. This review does not propose inference to reconstruct unknown settings or claim preservation of warm KV state.

## Reviewed on-disk source SHA-256

These identify the small files reviewed, not runtime-loaded Python bytes. No weights or configuration secrets were hashed/read by this review.

```text
45a00d19b46ee5efaf7a1c17235f907b68203cf7ecd5341a663d3bd8975b2e4e  omlx/_version.py
a79a0493b06e1f69326aab95f26b81b4653f9b6317f2e50d58efe483a9451d9d  omlx/server.py
669f28228d1dc54a9affc687d06f9487190dab1d5d499c485843cb9a818aef42  omlx/admin/routes.py
e67308293625b2f606530284d03a95ab4c1b7cd37aa267c8f5d076b32421f314  omlx/admin/auth.py
06e9186c4a5dc640c8ef0d83918ee9dadc7fdbb81782415dae361a6b2e7e5b92  omlx/engine_pool.py
dd0fa65f3f77b28f0300e054a5d55e1983e680f529239da0931805b24809791f  omlx/engine_core.py
c3cbf147d8f30965a5d503b8cee5cfd43050570c1876eff455155660a4ac928e  omlx/scheduler.py
8e92d9c8e16df6ab02947f1f15232dd179d8c255fb8b6f52590c0378191c29fd  omlx/utils/fatal.py
```
