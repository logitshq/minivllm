# E024 next HTTP launch: minimal adapter review

Read-only review, 2026-10-17. Repository: `/Users/ryanli/code/minivllm`. No tests, module imports, device work, service operations, or frozen-source changes were performed. The current serving numerical run is not treated as passed here.

The new `route_decode_graph/http_server.py` is a suitable prepared entry, but is not yet an HTTP-qualified candidate. It installs one instance before the donor scheduling observer and retains the real donor handler/worker. Its identity explicitly says `route_observer=False` and `actual_compiled_invocations_observed=False`. Installation alone cannot prove that HTTP requests used the new graph.

## Smallest source changes for the next actual qualification

All paths below are under `bench/m5_ultra_256gb/`. These are proposed new versioned adapters; preserve the existing frozen factories and evidence.

| Component | Minimal new contract | Reuse without changing behavior |
|---|---|---|
| Versioned `route_decode_graph/http_server.py` entry | Qualification-only, explicit route-events file; bind its actual path/SHA and the witness dependency. Attach the witness to the already installed instance before `donor.observe_runner`. Formal rejects both step and route observers. | Existing `native_server.BatchWorker`, `handler_class`, `completion`, `observe_runner`; current queue, SSE, raw IDs, full prompt, normal EOS and load settings. |
| New route HTTP capture factory | Same `prepare/validate/capture` interface as `http_capture_v2`, with its own source set and exact argv. Command explicitly includes `--phase qualification --route-decode-graph on`, step trace and separate route-events path. Expected readiness/startup identity describes the new entry. | Argument-based listener/readiness/process helpers and cancellation-safe owned-child cleanup. `native_cleanup.run_group` keeps original POST/DONE timestamps and raw SSE. |
| New route HTTP qualification wrapper | Source-bound plan includes group, scheduling trace and route-events references, plus the independent serving qualification dependency. Build the new exact identity, join the original HTTP data, install the route in the replay candidate before the original Observer/CallProxy, retain reference unchanged. | Existing `jobs_for` (C1: 8 requests; C8: 16), tokenizer/full prompt verification, `http_join.join`, `http_replay.run_recorded_case`, original model loading and numerical comparisons. |
| Narrow independent route HTTP audit | New entry/plan/source bindings; original per-step logits/cache/ahead gates; independently match actual route events and reject missing/extra/duplicate events. Bind the exact independent serving 37-request/42-direct-call receipt once it exists and passes. | Existing replay ledger numerical audit logic. Do not pass a new schema/source through the old fixed-source auditor by changing globals or paths. |
| Later observer-free formal adapter | Accept only the new independent C1/C8 HTTP qualification receipt and same model/runtime/config/candidate sources. Use `--phase formal --route-decode-graph on`, no trace or route observer. | Existing warm8/formal64 jobs, raw response/count/text checks, common-window throughput, POST→DONE latency and strict reap. |

The outer execution owner also needs explicit new factory/source/argv bindings, rather than pretending this is the old native capture. Its resource checks, direct-child ownership, capture-reap-before-replay boundary and stop/recover implementation can remain the reviewed logic. No new general lifecycle system is needed.

## Why the frozen factory cannot simply change its command

* `pareto_realistic/http_capture_v2.py:56–120` fixes an exact five-source set, constructs `native_server.py` argv, and reconstructs the whole specification in `validate`. Replacing the command after preparation must fail validation. Its capture function adds only `--qualification-trace` (`:223`); the new entry defaults to formal and correctly rejects a trace unless `--phase qualification` is explicit.
* `http_qualification.py:62–147` fixes the old six-source set and builds an identity with the donor server SHA. The new server reports its real source, donor source, benchmark phase and route identity. Exact identity comparison must change deliberately, including the existing float `io_timeout_seconds` contract.
* `http_qualification.run_adapter` (`:147`) does not install E024. Reusing it unchanged would replay the recorded tokens against the old candidate stack, without proving the new route's state correctness.
* `http_join.join` (`:163`) accepts a supplied identity and is reusable, but requires exactly the original four events per step. Adding route events into that JSONL would break the valid existing contract. Keep a separate sidecar.
* `native_formal.prepare/bind_qualification` (`:76–158`) bind the old HTTP auditor and factory. A new executable and a new numerical receipt require a narrow new binder, not mutation of an old prepared spec.

## Actual-route witness: use the existing small mechanism

`route_decode_graph/serving_qualification.py:77` already has `RouteWitness`: it counts calls through actual compiled functions, records batch/used length/KV capacities and distinguishes fallback. The candidate is attached through `installation.decoder.long`; there is no need for a global patch or another installed decoder.

In the HTTP entry, the single worker's trace callback can advance a local step counter on the unchanged donor `before_step` event. Set the witness's phase/step there before `engine.step`; emit and flush route-call records separately. Install this wrapper before `donor.observe_runner`, and restore it only after the worker has stopped, together with the existing installation cleanup. Pin the actual witness source. Do not call the graph a second time merely to observe eligibility or evaluate extra model arrays in the witness.

Bind the sidecar to PID, C, source identity and the original trace. Its exact ordered calls must be reconciled with the captured/replayed scheduling and ahead state: consuming a prior ahead result may produce no current forward call, while producing the next ahead result does call the model. One graph call per HTTP request or one per step is not a valid assumed count. A `SUBMITTED` event alone does not prove completion; the corresponding completed step and full response, followed by independent replay, supply that evidence. Retain closed raw files after reap; a SIGTERM exit need not invent a successful observer seal.

Require actual eligible invocation evidence for each claimed C point. Also validate observed fallbacks and their shape/capacity causes. Keep context **2048**, short compile/fuse/gates **ON**, norm reuse **OFF**, E028 **OFF**, the original D fixtures and normal EOS512. Do not pad requests, lower the service context, or force a larger cache to manufacture hits. Physical KV capacity may make an otherwise short request legitimately fall back. Report the actual bound/shape coverage; do not claim both 1023 and 1536 routes were hit unless the evidence shows it. The separate full serving numerical gate covers its own registered domains, not unseen HTTP histories.

## Qualification and timing boundary

For both C1/C8: real new-entry HTTP capture → response cleanup and strict server reap → fresh independent replay → independent numerical/route audit. Reuse full original-dtype logits, all cache state and ahead checks; raw-ID/text agreement or greedy argmax agreement alone is insufficient. `http_replay.run_recorded_case` already regenerates the scheduler from actual arrivals and checks full state, so only candidate installation/route binding needs adaptation.

The instrumented capture remains diagnostic. Only after the actual entry, complete prompt/IDs/usage, route evidence and replay pass may the observer-free formal adapter produce new candidate points. It retains the common window `max(DONE)-min(POST)`, actual committed-token numerator, per-request POST→DONE p50/p95 and separate resource/restoration qualification. Passing a captured history does not establish numerical equivalence for all possible future arrival schedules, and this review predicts no speedup.
