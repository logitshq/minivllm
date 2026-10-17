# E023 actual HTTP baseline v2: qualification PASS

2026-10-17 · M5 Ultra 256 GiB · Qwen3.8-27B NVFP4. The fixed D512 C1/C8 HTTP
histories passed independent numerical, provenance, observed-resource and recovery
audits. This is qualification evidence; it does not create a performance point.

| Captured concurrency | Requests | Replay steps | Committed tokens | Original-dtype bit comparisons |
|---|---:|---:|---:|---:|
| C1 | 8 | 4096 | 4096 | 5,251,008 |
| C8 | 16 | 1028 | 8192 | 1,321,965 |
| Total | 24 | 5124 | 12288 | 6,572,973 |

Original raw SSE and actual UUID/arrival/schedule traces were joined to full fixture
prompts, committed IDs/text and usage. Two independently loaded engines replayed
those actual arrivals with full vocabulary logits and all cache/ahead state checked.
The offline auditor reconstructs FCFS scheduling and accounts for every recorded
comparison; it does not rerun GPU arithmetic. C8 exercised actual B2/4/6/8 decode
rows (B8 on 1016 steps). All requests hit cap 512; normal EOS was enabled, but observed
early EOS was zero. This evidence covers these captured histories, not every future
free-arrival workload or cross-framework output equivalence. E028 and E024 remain OFF.

## Failures retained and narrow fixes

The original numerical auditor failed at C1 step1 because its independent expected
sampling temperature was integer 0, while the frozen HTTP producer explicitly uses
float 0.0. The same representation appears in ahead metadata. Separate v2 changes
only those two expected values; raw records and strict comparisons stay intact.
Actual first-three-step bytes reproduce the failure and exercise ahead production
and consumption. AST/negative tests retain every other check and reject integer,
boolean and nonzero HTTP temperatures. There was no model rerun for this audit fix.

The original owner auditor rejected three legitimate empty `__init__.py` files.
Separate v2 accepts nonnegative exact-integer sizes while retaining pinned SHA and
actual file-size checks. The original source, FAIL receipts and both fixes are all
archived. Root 21 numerical/formal CPU tests, 8 original-owner tests and 12 owner-v2
regression tests passed; the completed C1/C8 and owner-v2 actual audits also passed.

## Resources and restoration

All four direct children were strictly waited/reaped: C1 capture 83220 / replay 83874,
C8 capture 87147 / replay 87529; owner 83156 exited. Across 249 paired samples pressure
stayed 1; swap usage, Swapouts and Pageouts did not grow. Swapins increased 3. Retained
process inventory only excludes the recognized runtimes, not unknown GPU activity.

The approved cold cycle replaced original oMLX PID 81337 with PID 90204. Recovery
finished at 2026-10-17T15:44:24.977406Z. The original Qwen3.8-Flash-Next-oQ6e-mtp model
was loaded and fresh logs confirm Lightning MTP and resident PLE. Independent audit
checks 19 raw physical samples (maximum 175,635,365,888 system-footprint bytes), raw
logs and child cleanup. Final status/owner absence are source-pinned controller
receipts. Warm KV and exact hidden runtime state are not restored.

## Reproduce and inspect

`evidence.tar.xz` retains the entire owner, cold journal, execution, analysis,
preregistration and relevant audit source trees. `manifest.json` lists every member,
SHA, byte count and original path. `archive-verification.json` verifies all extracted
members, original hashes and exact tree membership. No model weight payload is
embedded. The separate exact `restored-journal.json` and `restoration-audit.json`
sidecars bind the next authorized cycle to this verified replacement identity.

To inspect, extract into a new directory. Analysis includes `summary.json`, both
`c*-independent-audit-v2.json`, `owner-independent-audit-v2.json`, all original failed
audits, source-pinned root CPU receipts and CPU-prepared formal specs. Runtime and
model snapshot paths are machine-specific; archived frozen source copies and full
model manifests/producer hash receipts preserve provenance. The independent owner
auditor rehashes runtime/source files and verifies weight manifest/hash/stat receipts;
it does not independently rehash model shard payloads.

The next performance cycle uses fresh observer-OFF native servers at C1/C8, each
with 8 warm and 64 formal requests, identical D512 prompts, normal EOS and no cache
hits. Throughput is actual committed tokens divided by first POST→last DONE; latency
is each full request POST→DONE p50/p95. First-pair n = 1 points will be labeled before
three alternating repeats. Official oMLX and Lithos need their own D512 qualification
and measurements. Historic cap 32 points cannot be joined to this workload's frontier.
