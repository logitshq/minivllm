# Cold-restart controller: independent interface and owner review

2026-10-17. Controller reviewed at SHA `aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3`. No implementation changed. No `System` backend method, controller phase, HTTP mutation, live HTTP request, signal, model import or GPU operation was executed. Only bounded read-only `ps`, `launchctl print`, and `lsof` commands were run; existing HTTP/footprint payloads were checked offline.

## Verified interfaces

At 11:01:04 UTC, the exact controller-shaped `ps` query returned PID 74384, PPID 1, start `Fri Oct 16 08:01:58 2026`, comm `omlx-server`. Top-level `launchctl print gui/501/local.omlx.server` fields have exactly one leading tab and match the parser; nested state fields have two tabs. A deliberately nonexistent, unrelated label returned **113** with empty stdout, validating the absence-code assumption without removing any service. The exact `lsof -Fpn` query returned `p74384`, an extra ignored `f4`, and `n127.0.0.1:8001`; the controller's selected-field parser accepts this.

Evidence: [read-only-command-evidence.json](omlx-controller-interface-audit-v1/read-only-command-evidence.json), SHA `9d06c863c196beb18d8d78430c4d7392ec3198940fc87afbbe95e3b4353a10ac`. No raw launchd argv/environment/stderr was retained.

The saved complete `/api/status` and `/v1/models/status` responses were passed through `service_status()` using an offline provider (no `System` object). They pass the exact physical target, default, version, idle counters, paths, engine/config type, loaded/pinned fields and absence of other pinned models. The actual source builds those counters and physical IDs in `server.py:2945-3015`; `engine_pool.py:3930-3978` supplies row metadata even before a model is loaded. Exposed-profile duplicate rows do not substitute for the physical loaded set.

The existing real footprint JSON is byte-valued (`unit="byte"`, `bytes per unit=1`), with integer `auxiliary.sys_footprint=187547942912`, empty errors/warnings. That is the correct field, not `total footprint`, which describes only the selected process. The current implementation's field access matches this schema. Evidence: [saved-payload-evidence.json](omlx-controller-interface-audit-v1/saved-payload-evidence.json), SHA `dd04f630a3521ad94bee64bff0ada9c16d076928784789eca53f2eb022bff25b`.

Remaining actual-interface checks are narrow: the controller's self-PID footprint invocation and its two-sample gate have not run; the original real sample targeted PID 74384. A read-only `prepare` has not run, so its combined current pins/global-stat/model-metadata path is unvalidated as a whole. Empty-service startup, in-progress loading, replacement-PID binding and new log-window matching remain static/mock-tested, not real restart-tested. User authorization is still pending; these gaps are not permission to run stop/recover/load.

## Blocking integration gap: the 37-request owner assumes a suspended service

The frozen E023 `pareto_realistic/qualify_owner.py` cannot be used as-is after cold stop:

- Lines 99 and 106–108 pass `suspended_pids=(args.suspended_pid,)` and require exactly one real matching process observed with `authorized_suspended=true`.
- CLI lines 183–187 require a positive `--suspended-pid`; the old jobs bind 74384.
- The completion receipt labels restoration as `outer suspended_service sole owner` (line 174), which would be false under cold restart.
- Its independently frozen auditor, `audit_qualification.py:422-425`, also requires 74384 in T state, allows that exact service in process inventory, and demands `authorized_suspended_pids == [74384]`.

A new service-absent owner/plan/jobs/auditor is required; retain the old frozen run unchanged. Minimal adaptation: keep the 37-request numerical workload and strict output/state checks; set `suspended_pids=()`; at every resource phase require old PID absent, the exact launchd job absent, port 8001 unowned, and no foreign recognized inference process. Abort the owned child if the service reappears. Pin the cold-restart journal/contract and the new owner/auditor source, and make restoration ownership explicit. Do not fabricate a suspended-process entry, pass a dummy PID, or loosen the original auditor to accept both contradictory protocols silently.

The existing E023 owner's `attempt.json`, `child.json`, and successful cleanup `release.json` field shapes **are compatible** with `controller.reaped()`: the actual failed run recorded owner PID 24388, child PID 24398, returncode -15, `waited_reaped=true`, `strict_absence=true`. Recovery correctly need not require numerical PASS. New receipts must belong to the predeclared new directory and postdate service removal; old E023 receipts cannot be reused.

## Second blocking recovery edge: pre-spawn owner failure

`qualify_owner.py:114` performs validation/hash preparation before writing `attempt.json` at line 118; preflight occurs before `Popen`. A validation/pressure failure can therefore leave an owner directory but no child or release receipt. The restart controller then refuses both normal recovery (missing reap receipt) and `--no-experiment` (directory already exists), even when no model process ever started.

The new owner needs a source-pinned, durable **no-child-started terminal receipt** for failures strictly before its spawn intent, with owner identity and terminal phase. Recovery can accept that separate contract only after the owner is absent and its no-child proof is independently validated. A crash after spawn intent but before child identity cannot be called no-child; that remains manual cleanup/verification. Do not synthesize a fake `release.json`, use a null/dummy PID, or delete the owner directory to make `--no-experiment` pass. This is an availability gap in the current fail-closed controller, not permission to bypass safety evidence.

## Recovery boundaries that must remain explicit

1. The controller is an admission/reap waiter, **not a continuous physical-memory watchdog during loading**. It samples before bootstrap, before POST, and after completed load. Its one POST may block up to 600 seconds. Root still needs independent physical monitoring during reload; the 40 GiB baseline gate and historical 184.8 GiB service peak do not prove the next peak cannot exceed RAM.
2. The predeclared owner directory is a cooperative sole-owner contract, not an OS-wide launch lock. The producer must be cancelled before `--no-experiment`; no producer may start after recovery begins. Old owner/auditor binding is insufficient to enforce this by itself.
3. Resource evidence currently keeps only `latest_resources` in the journal and deletes each temporary raw footprint JSON; later gates overwrite earlier pairs. It records timestamps/events, but an independent audit cannot reconstruct all earlier raw physical samples or verify their byte units from the journal alone. Similarly, final log proof stores fixed matched markers and the starting offset, without an archived new-window SHA. These are evidence-retention gaps if recovery is to meet the repository's full raw audit standard. No false restoration follows directly from them, but the limitation must not be hidden.
4. A replacement that loads SSD/mmap PLE, loses MTP evidence, changes config/stat identity, rotates the log, or changes PID remains `recovery_incomplete`. Current code deliberately does not repeatedly bootstrap/POST after an unknown outcome. It cannot preserve hidden runtime signatures or warm KV; the 5-second launchd exit versus possible 60-second teardown remains part of the pending cold-restart consent.

The directly checked OS/parser interfaces and the saved loaded-service HTTP/footprint schemas are consistent. The two concrete blockers before a 37-request cold-service run are the service-absent owner/auditor protocol and an explicit pre-spawn-failure recovery proof. Neither requires changing the frozen numerical qualifier or treating the pending authorization as approval.
