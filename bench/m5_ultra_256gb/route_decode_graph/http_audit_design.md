# Minimal independent E024 recorded-HTTP route audit

Source-only design, 2026-10-17. No tests, runtime/model imports, device work or service actions were performed. This document neither qualifies the new HTTP entry nor adds a performance point.

## Interface and producer contract

Use one new `audit_http.py` with `audit(child_dir, expected_plan_sha256) -> dict` and CLI `--child --plan-sha256 --out NEW_JSON`. Do not call the old fixed-source auditor's top-level `audit()` or make a new result look like an old E023 result.

The pending `http_qualification_v2.py` producer and this audit share this exact proposed plan:

```text
schema: E024-recorded-route-http-plan-v1
qualifier_sha256: actual new qualifier SHA
qualification: {path, sha256}                 # unchanged E023 plan21ba…47fb
serving: {plan: {path, sha256}, audit: {path, sha256}}
concurrency: 1 | 8
capture: {spec, complete, child, release, absence, port}  # each {path, sha256}
inputs: {group, trace, route_events}            # each {path, sha256}
sources: exact actual helper/entry/self map
```

`capture` refers to `spec.json`, `complete.json`, `child.json`, `release.json`, `child-absence.json`, `port-after.json` in the same closed capture directory. `inputs` refers to that directory's `group.json`, `trace.jsonl`, `route-events.jsonl`. The new producer's `E024-recorded-route-http-result-v1` retains these references, `base_result`, recording SHA, route comparison and coverage. Producer PASS sets `replay_numerical_checks_passed=true`, but leaves numerical/HTTP/performance qualification false and `independent_audit_required=true`.

The audit first checks terminal files, exact source/plan identities and the real capture/reap chain. Missing terminal evidence is INCOMPLETE before opening a large ledger; any producer FAIL remains FAIL. Rebuild the new capture spec through its pinned factory and independently rejoin raw SSE plus the four-event scheduling trace using `http_join.join`, the pinned CPU tokenizer and the new exact startup identity. Preserve full prompt IDs, body/rid/UUID binding, normal EOS, raw/committed usage, exact text, 8/16 drained jobs and C8 multi-row checks. Do not normalize raw evidence to fit the old schema.

Bind the serving prerequisite to the actual archived `E024-serving-v1/independent-numerical.json`, its exact file SHA, auditor `064634ac…573ba`, child plan `e85f92aa…49c56` and original plan21ba. Require its PASS and matching candidate/integration/source/model/config identities. That receipt covers the separate original37 requests and42 direct parameter calls; it must not become evidence that HTTP has repeated the parameter diagnostic. Current short compile/fuse/gates stay ON, norm reuse and E028 stay OFF. Replay retains the original1800s numerical deadline; the outer owner supplies its separately registered hard cleanup budget.

One genuinely new HTTP check is required: the frozen join does not sweep client request intervals. Independently require finite `start_perf_counter_s < done_perf_counter_s` for every completed raw request, then sweep half-open POST→DONE intervals with DONE before POST at equal times. Require no negative count, final outstanding0 and peak exactly the declared C. Reuse the small interval calculation in `pareto_realistic/audit_native_formal.py:raw_metrics`, while retaining original request/group deadline and cleanup-clock checks. This proves observed HTTP caller concurrency; the independently simulated decode histogram proves GPU batch coverage separately. A serial history labeled C8 must fail even if its declared configuration is8.

## Reuse without changing old evidence contracts

| Existing code | Reuse | New work |
| --- | --- | --- |
| `pareto_realistic/http_join.py` (`a362c321…7b4e`) | Raw SSE/trace reconstruction with caller-supplied identity | Build that identity for the real v2 entry, not the old native source |
| `pareto_realistic/audit_http_replay_v2.py` (`7cdcf5a2…f031`) `case_type` / `stream_ledger` | Actual arrivals, independent FCFS/slot simulation, HTTP float-temperature expectations, all full-logit/state/ahead checks, four load/release events and final IDs | A narrow base `CaseAudit` subclass adds route reconciliation in `check_arrays`; the old HTTP class inherits it |
| `pareto_realistic/audit_qualification.py` (pinned by the HTTP auditor) | Full64-layer/128-state names, shape/dtype/finite/bit gates and original semantic call pairing | Preserve all checks; no new array groups are needed |
| `route_decode_graph/audit_serving.py` `ServingCase.check_arrays` / `RouteEvents.route` | Existing source-backed KV-capacity derivation and expected route-row rules | Two streaming cursors, one for capture and one for replay; no parameter-prelude parser |

Load the base audit module privately, keep its original `CaseAudit`, then assign only that private module's `CaseAudit` to the small subclass before passing it into the unchanged HTTP `stream_ledger`. Its inherited `check_arrays` runs first. This uses the old numerical primitives under their real paths/SHAs; it does not rewrite their `HERE`, filenames, global source constants, old plans or rows. The new receipt names the new auditor and all reused source SHAs. No use of the old top-level binding/acceptance result is implied.

Each HTTP replay loads a fresh independent pair for one C. Initialize free slots normally; do not carry the original37 workload's C8→lifecycle free-stack correction across separate HTTP processes.

## Independently derive the actual route-call stream

The original HTTP audit already reconstructs ordered semantic calls as current decode, possible next-ahead decode, then prefill. Consume route events from that validated sequence, not from a declared count:

1. Keep the prior post-retirement batch IDs and physical KV capacity. On a batch rebuild, compute new rows and surviving rows from independently reconstructed requests. The required capacity is the maximum of `prompt length + cap` for new rows and the old capacity if survivors remain, rounded up to256. Preserve capacity when the batch does not rebuild; reset it when no decode rows survive. Check each relevant K/V descriptor against that derived capacity as in `audit_serving.py`.
2. For each semantic decode call, skip the current call if it equals the candidate's independently checked `consumed_prior_future`: that graph ran in a previous step. Prefill never goes through the route decoder. For an actually executed decode call, derive `B=len(rids)` and `used=max(positions)+1` from the semantic key already validated against scheduler/input/output history.
3. `_Dispatch` calls the long candidate only for `257 <= used <= 1536`; other lengths produce no route event. Within that domain, derive expected graph eligibility from B and reconstructed physical capacity. For the supported serving scope, a graph hit requires B1..8 and physical1536; smaller capacities record fallback. Larger capacity remains unqualified, matching the frozen witness's explicit gate.
4. Read exactly one next row from **each** sidecar for each derived long-route call. Require exact keys, types and values: `event=route-call`, `phase=HTTP-C{C}`, CPU step, `direct_label=null`, `status=SUBMITTED`, B, used, all16 `[K,V]` capacities, `expected_graph`, actual invocation0/1 and prefix callable-family count. The family count is the number of seen bound1023/1536 keys with fixed gates/norm flags, not the number of Metal machine graphs. Both sidecars must independently match the same expected row; mutual equality alone is insufficient.
5. Require EOF after the last completed step, no truncated rows, no RUNNING/extra/duplicate events, and at least one actual graph invocation in each C being qualified. Report the observed families/ranges/fallbacks; do not manufacture coverage by requiring both bounds if real HTTP requests do not traverse both. Preserve C8 actual multi-row decode and caller concurrency as separate facts.

There can be zero, one or two decode submissions in a step; consuming ahead may remove one. Partial retirement, refill or a new arrival can change the next key. A `SUBMITTED` row alone is not proof of numerical completion: it must match a successful full-state ledger step and the joined original completed trace/SSE. Event order is bound through the worker's `before_step` counter and independently inferred within-step order. Current route events do not contain an explicit token key or unique call ordinal; do not claim stronger identity than these fields establish.

Capture startup must be exactly the server's `route-startup` with schema `E024-route-http-events-v1`, actual server PID, HTTP-C phase and full startup identity. Replay starts with `replay-route-startup`, schema `E024-route-http-replay-events-v1`, its distinct process PID/phase and exact captured-sidecar reference. Verify the header fields and retained byte hashes; PID ownership of the replay belongs to the outer owner. Neither server emits a successful end seal on SIGTERM, so do not invent one: strict closed-server reap, complete original trace/requests and exact route EOF are the real boundary.

## Result and focused checks

A proposed `E024-recorded-route-http-independent-audit-v1` PASS grants only numerical/route qualification of this one recorded HTTP history. Return plan/source/input/terminal hashes, streamed ledger and both sidecar byte hashes/counts, actual route domains, array-check totals, ahead/retirement/EOS observations and the exact serving prerequisite. Keep `performance_qualified=false`, `owner_resources_restore_qualified=false` and `offline_tensor_bytes_rechecked=false`. The successful device tensor payloads are not available for offline arithmetic recomputation. Formal timing and resource/restoration acceptance remain separate.

Only focused mechanism tests are needed next: actual fake producer→wire→join→replay; an ahead-consumed step with no repeated current route call; current+future double submission; early smaller-capacity fallback; changed B/used/capacity; identically corrupted capture and replay sidecars; extra/missing/reordered rows; wrong PID/source/prerequisite or truncated terminal; C8 mislabeled serial history; original full-state corruption still rejected. Preserve original failure/partial evidence, and do not loosen numerical comparison or source binding to pass a fixture.

At this design point the new capture uses source `b331a029…4151f`, the real server v2 uses `8ed952b4…5621b7`, and its witness is the frozen serving qualifier `75de7205…4a7264` with counter `c24219f4…3faf7`. The pending qualifier/auditor need final actual SHAs before registration. No additional lifecycle framework or model execution is part of this design.
