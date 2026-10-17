# E024 HTTP adapter: minimum next change

Read-only source review, 2026-10-17. No model, MLX import, test, server, timing,
or device operation was performed. This is a proposed adapter, not qualification
or a speed result. Repository: `/Users/ryanli/code/minivllm`.

## Keep the current serving comparison intact

Use the existing frozen Qwen3.8-27B NVFP4 source/model/fixture/runtime identities,
D8 normal-EOS cap512, C1/C8, and the existing candidate settings:
short compile ON, fuse ON, gates ON, norm reuse OFF. E028 stays OFF. Add only the
E024 route installation flag, default OFF; do not make gates OFF the new serving
baseline. Existing eager/gates-OFF numerical reference remains the independent
correctness oracle, not the performance comparator.

The old 52-point matrix and forward timing are useful prerequisites. They do
not qualify the new instance installation, changed parameters, ahead/refill
paths, or real HTTP histories. The newly prepared `serving_qualification.py`
37-request plus 42-call diagnostic must pass its independent audit first;
current source preparation is not a device result.

## Actual insertion point

`route_decode_graph/integration.py:85–119` is already the required instance API:
`install(engine.runner, enabled=flag)`. OFF returns before touching the runner.
ON verifies the actual loaded ModelRunner/cache/short-compiler sources and
requires fresh empty target-only state. It replaces only `runner.decode_model`.
`_Dispatch.__call__` at 50–54 retains the old short/out-of-domain path; the
candidate itself guards B1..8, used257..1536, original model/backend and cache
layout, with bound1023 below1024 and bound1536 thereafter. Do not widen it.

The new HTTP entry should insert installation after the existing layer settings
and target-only check (`native_server.py:369–374`), before identity construction,
`observe_runner` (393), synchronization and `worker.initialize` (396).
Both current decode and ahead use this same decoder entry point. Installing
after requests or after a full-state Observer's CallProxy violates the contract.

The frozen server's `main()` owns its loader as a nested function and offers no
factory argument. The clean small adapter is a new server entry containing only
the argument/main/load glue, importing the frozen `BatchWorker`, `handler_class`,
`observe_runner`, request parser and completion functions by their real pinned
source. Do not patch the frozen module's classes/globals or spoof `__file__`.
The new OFF and ON entry share this glue; OFF leaves `decode_model` unchanged.
All request admission, queueing, token delivery, EOS handling, SSE and sockets
continue through the existing classes/functions.

Retain the installation handle for the worker/engine lifetime. If Python cleanup
runs, restore owned observer wrappers before `Installation.restore()` (73–81),
after the worker stops or during final engine teardown. Never restore while the
worker is still submitting work. SIGTERM need not execute Python cleanup;
the external direct-child TERM/KILL/wait/strict-absence proof still owns that
case. Do not add a service controller inside the HTTP entry.

## Explicit identity; no new trace protocol is needed

Keep all existing identity fields and configuration values. Add one versioned
`route_decode_graph` object containing at least requested `enabled`, actual
`installed`, integration and candidate path/SHA, original supporting candidate
and model-source-pin refs, bounds `[1023,1536]`, B domain `[1,8]`, and used domain
`[257,1536]`. Record E028 disabled. Set `server_source_sha256` to the NEW entry's
actual bytes and add a separate `frozen_server_source_sha256` for its donor.
Do not claim the old source manifest contains these external bench adapters.
Do not include process-local object IDs in an expected deterministic identity.

`http_qualification.expected_identity` (103–124) needs a new versioned wrapper
that calls the frozen function, then adds exactly those adapter fields and the
actual new entry SHA. Preserve float `io_timeout_seconds` and exact boolean/
string types. The frozen `http_join.join` accepts extra identity fields and
compares the full startup identity exactly (177), so it can be reused without
changing its SSE, queue, scheduling or raw-ID rules.

Installation is not proof of an actual compiled invocation. Qualification must
also witness execution of the two graph families rather than count a guard
decision. The existing `serving_qualification.RouteWitness` (77–105) can provide
that narrow observer around the installed long callable. If used in the HTTP
capture, write it to a separate route-events file; do not inject new events into
the frozen four-event scheduling trace that `http_join` parses. Set its phase/
step around the actual runner call before wrapping with `observe_runner`.
Qualification-only counters do not belong in formal timing. Explicitly record
the route-observer flag and turn it OFF together with the step observer for
formal runs. Preserve the separate raw route-event SHA in the capture plan.

## Replay the candidate's actual arrivals with independent engines

The frozen `http_replay.run_recorded_case` (19–151) already supplies the needed
loop: only recorded arrivals/prompt IDs/caps are inputs; both engines really
schedule and compute, and the recorded schedule, raw IDs and committed IDs are
comparison targets. Retain its before/after state checks, full logits/cache
comparison through `pair_calls`, semantic ahead matching, normal EOS/cap,
retirement, final drain and C8 actual-multirow gate. No new scheduler is needed.

The new per-C qualification wrapper must install E024 ONLY on the candidate
engine before the frozen `Observer` constructor wraps `model`/`decode_model`
(`qualify_native.py:197–209,549–570`). Reference remains independently loaded
eager and uses the existing grouping-disabled/previous-ahead-discard rules.
The same fresh installation and source settings must be used by HTTP capture
and replay; the old baseline-only HTTP audit receipt cannot grant this route.

Reuse the install/witness/restore behavior in `serving_qualification.adapt`
(179–256) on a privately loaded qualifier. Do not blindly nest its original
`run_case` wrapper with `http_qualification.run_adapter` (147–164): the latter
replaces `base.run_case`, so the former's parameter prelude and phase assignment
would be bypassed. A tiny replay facade can set `observer.case`/case-step and
`route_witness.phase` to HTTP-C1/C8, then call the unchanged replay loop while
the private RouteObserver is installed. The 37/42 diagnostic is a separate
prerequisite; do not accidentally rerun it in every HTTP replay or call its
`require_complete` against the 8/16-request HTTP workload. Ensure observer
restoration precedes install restoration even on constructor/run failure.

Retain actual route hit counts and observed ahead/retire/refill coverage for
this capture. Do not require an unobserved discard count or claim cancellation/
sampling-key invalidation was tested. A successful replay covers these captured
histories, not all possible future arrivals or full raw tensor recomputation
where only source-pinned device comparison receipts were retained.

## Minimum new file/version set

| Piece | Required delta; frozen donor retained |
|---|---|
| New E024 HTTP server entry | New main/load glue, explicit OFF/ON install and actual source identity; reuse original transport/worker. |
| New E024 HTTP qualification wrapper | Additional exact plan/source/adapter refs and expected identity; candidate-only Observer install; unchanged join/replay core. |
| Capture plan/launcher adapter | New command/source/identity factory with optional route-event path. `http_capture_v2.prepare/verify_sources` hardcode the old server and qualification modules, so altering its returned command would fail validation and is not an adapter. Reuse its actual OS/readiness/cleanup helpers with explicit new spec validation. |
| Numerical audit adapter | Bind the new wrapper/recording/route receipts and reuse existing strict metadata/array/ahead checks; retain the HTTP float0.0 sampling correction from audit v2. Do not label the result original baseline qualification or the 37-request result. |
| Later formal binder/launcher | Bind the new independent captured-history audit and same adapter settings, with both observers OFF. The frozen `native_formal` binder pins the old six-source baseline; give the new route explicit versioned bindings rather than editing old receipts. Reuse its raw SSE checks, client/cleanup and point arithmetic. |

Pin actual new source files plus integration, candidate, original long-decode
helper/model-source-pins, numerical/HTTP donors and the frozen native model
package separately. Keep model/runtime full pre/post checks with the existing
outer owner. The old baseline evidence and archives stay unchanged. No new
general lifecycle or parser framework is necessary.

## Measurement boundary

Capture server must be fully reaped, including PID and port absence, before
starting the independently owned dual-model replay. Formal trials start fresh
with qualification observers OFF, retain all 8 warm and 64 formal requests and
the registered C1/C8 request policy, and use the same model/config/resources.
Compute committed throughput over first POST to last received DONE and each
request's POST→DONE p50/p95; retain queue residence, EOS-shortened actual counts,
raw wire bytes, usage, prompt/fixture map and text checks. Evaluate each fresh
trial independently before aggregating. No local forward percentage is an HTTP
speed prediction. E024 alone is the ON arm; E028 stays OFF so attribution is
preserved.
