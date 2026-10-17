# Native HTTP baseline readiness — 2026-10-17

CPU/source-only independent review. No new blocker was found in the existing capture → join → replay modules. This review did not run a process, HTTP request, model, GPU, large ledger, or new tests. Actual HTTP qualification is still pending; the outer owner/controller must finish before execution.

## What the current path establishes

- `pareto_realistic/native_server.py:53,86,163`: the server assigns real UUIDs, admits pending requests at actual step boundaries, and records the scheduler output passed to the model runner and its actual decode rows.
- `http_qualification.py:127`: the frozen tokenizer verifies complete prompt IDs. The join binds original request bodies, fixture IDs, response UUIDs, all retained raw SSE bytes, normal EOS/cap, usage, complete raw/committed IDs and exact decoded text to the ordered server trace. Derived client fields are not token oracles.
- `http_qualification.py:147` and `http_replay.py:19`: each C loads a fresh independent eager/current-stack pair. Only recorded arrivals, prompt IDs and caps are inputs. Recorded schedules and tokens are comparison targets. Both actual schedules must match before candidate model work; original full-vocabulary logits, all cache/state and ahead checks remain. Final raw/committed IDs must equal the received HTTP results, every request must drain, and C8 must exercise actual multirow decode.
- `http_capture.py:293–374`: client cleanup cannot skip owned-server TERM/KILL/wait. Successful release requires strict PID and port absence. Repeated outer cancellation is recorded while the independent bounded cleanup task continues. Unresolved cleanup is FAIL and retains the exact child PID for outer reconciliation.

A successful replay covers the **captured discrete HTTP schedule**. Full state comparisons occur in replay; the HTTP server itself records schedule/row metadata and output IDs, not every internal tensor. This does not establish every future formal arrival pattern, additional concurrency values, or E028 deployment. Capture has an observer and must not become a performance point.

## Minimum execution path

Finish the single outer HTTP owner/controller: verify runtime/weights/source pins and the authorized service/resource contract, run fresh C1 capture, require capture/cleanup/reap success, then freeze the actual group/trace SHA-bound replay plan and run its replay. Repeat independently for C8. The fixed qualification workload is eight D512 requests at C1 and sixteen at C8, using normal EOS. No warm inference may precede the captured trace on that server.

The replay plan's six source pins do not cover client cleanup; the outer capture contract separately pins `http_capture.py` and `native_cleanup.py`. Preserve both bindings. Use `native_cleanup.run_group` and its own `GroupCancelled` class. Any incomplete client or server cleanup prevents replay and restoration until reconciled by the owner.

After qualification, formal measurements require separate fresh runs with `N=max(64,8*C)`, observers/trace disabled, and the fixed model/config/fixture/count contract. A recorded-schedule PASS does not by itself supply an oracle for those new schedules. Qualification and throughput-versus-complete-latency evidence remain separate; no new generic validator is needed to start this baseline experiment.

## Narrow later E028 adaptation

The UUID restriction is in `identity_refill/model_qualification.py:78–81`: its observation helper assigns cases only from `c1-`, `c8-`, or `probe-` prefixes. The candidate and `integration.install` have no such prefix restriction. `require_complete:165–175` separately assumes 37 requests and two model pairs. Neither restriction belongs in the algorithm.

For an independent HTTP E028 qualifier, privately reuse the frozen refill observer and change only its transition case assignment to explicit `HTTP-C1`/`HTTP-C8`, requiring every real UUID to belong to the joined recording. Keep all 128 predecode arrays, counter increments, full original numerical checks and restoration. Use a one-C completion gate: 8/16 requests, one install/two observer restorations, C1 zero hits, C8 at least one actual paired refill, and final counters equal observed hits. Do not rename UUIDs or change the original 37-request artifacts.

For a separate server version, the clean installation point is after fresh `LLMEngine` construction at `native_server.py:365` and before identity/trace/ready at `378/391/396`. Add an explicit default-OFF `--identity-refill` flag, call the existing installer there, and report the actual flag plus adapter/candidate source identities. The frozen main has no factory hook and imports the engine internally: a versioned source copy with these small explicit changes is simpler than AST rewriting or global patches. Qualification can observe refills; paired formal OFF/ON runs must not install observation hooks. E028 branch counters alone do not establish numerical qualification or a speedup.

## Reviewed small-source identities

All paths below are relative to `bench/m5_ultra_256gb/`.

| Source | SHA-256 |
|---|---|
| `pareto_realistic/native_server.py` | `9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f` |
| `pareto_realistic/http_capture.py` | `8f43c71b8f3fd99d655ba760aa344eb3026eef35450be9c5ed1f06171109a0fb` |
| `pareto_realistic/http_qualification.py` | `471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2` |
| `pareto_realistic/http_join.py` | `a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e` |
| `pareto_realistic/http_replay.py` | `1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65` |
| `pareto_realistic/native_cleanup.py` | `6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6` |
| `identity_refill/candidate.py` | `bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5` |
| `identity_refill/integration.py` | `ea0d6cb6af7c80d7580b4ab1c8319a4b76110fd1d73cf660c048c8dd45cf0c52` |
| `identity_refill/model_qualification.py` | `b8b84a57b63c6f46ea1449ec9eeed8310d1b4dbe074a81082e5a665371061042` |
