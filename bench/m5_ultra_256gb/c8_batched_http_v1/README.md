# Candidate C8 HTTP capture, warm8 + screen8

Source prepared; **10/10 CPU contracts passed** in `.artifacts/m5-ultra-256gb/c8-batched-http-cpu-v1` with unchanged sources and no failures. The guard denied backend imports, processes, signals and network operations; only six local AF_UNIX sockets for asyncio self-pipes were created. No plan has been prepared and no HTTP or GPU execution has occurred. The prior full E051 recorded-history result must actually pass before preparation. That result covers the old graph-only E047 arrival history; it does not qualify the new candidate server's observed schedules.

`server.py` reuses the original combined server `main` code object in a private globals dictionary. The frozen native worker, model loader, SSE handler, tokenization, scheduler, request completion, and cleanup remain intact. The actual installation module is `gdn_batched_graph/integration.py`, installed after native flags and before observation or tracing. Both `qualification` and `formal` phases use this same source and integration. Qualification requires both switches ON and separate scheduling/route sidecars. Formal forbids those observers; formal switch ablations retain the original argument rules. The entry is inert without explicit `--run`. C8/context2048/budget2048/cache2GiB and ops50/MB50/BFS20/sync0 remain fixed.

One necessary trace-only adaptation is explicit: the old combined trace helper asserted that B>1 must use native fallback. That predicate is incompatible with the new batched candidate. The new server reuses the original `TraceGraph.__call__` code with a private `batched_delta` helper: all48 installed layers must have equal nonnegative Python counter deltas; B1–8 fallback delta must be zero. Candidate delta may be zero on compiled reuse. The original helper still requires an observed same-bound/B trace before classifying compiled reuse. Raw before/after counters, route calls, compiled invocation counts and failures remain retained. These are Python tracing/submission counts, not GPU kernel counts or proof of completed arithmetic.

Startup identity names the actual new server/integration/source pins, phase, actual installation and observer state,48GDN handles, and the new trace-policy source. Numerical/full-state/performance qualification flags remain false. The route sidecar retains its original structural `GDN-graph-combined-http-events-v1` schema, with the actual new identity distinguishing the branch policy and implementation.

`trial.py` derives the exact original E047 two-wave jobs from E033's frozen C8/off/r1 base spec, then changes only the server path, explicit `--run`, GDN ON, identity/source closure, and predecessor binding. Each wave has eight unchanged request bodies/cap512; the first group drains before the second. E047 `System`, group cleanup and derived-view join are reused verbatim. Original direct-child capture code owns bounded readiness/requests, TERM/KILL/wait/strict absence and retained failures; only its private validate and route-startup schema bindings change. The external owner alone handles the user service and physical resource policy.

Preparation checks E051's actual top/plan/base receipts,16completed/1030steps,1028graph calls,48layer cleanup, emitted sidecars and unchanged full source/input contract. Root additionally binds its independent audit and restored lifecycle. All new capture performance/qualification flags remain false, even when transport capture completes. Observed route shapes, steps and token IDs are actual outputs; no assumption that the new scheduling trace matches E047 is made.

Interfaces:

```text
trial.prepare(base_spec_ref, predecessor_ref)
trial.run(spec, output, checkpoint=owner_callback)
trial.join_child(child, actual_spec_sha256)
```

Both refs are explicit absolute path/SHA dictionaries. `base_spec_ref` is the original E033 `r1-c8-off.json`; `predecessor_ref` is E051's actual numerical `complete.json`, whose adjacent `plan.json` is bound by its SHA. Spec schema `c8-batched-http-spec-v1`, label `c8-batched-mb50-two-waves`, result `c8-batched-http-result-v1`, joined-input `c8-batched-http-joined-input-v1`. Original raw SSE and trace inputs remain authoritative. After closed capture, this new history still needs complete candidate/eager replay and independent audit before observer-OFF formal performance comparisons.
