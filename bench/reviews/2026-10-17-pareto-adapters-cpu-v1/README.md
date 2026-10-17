# Pareto qualification adapters: CPU preparation, 2026-10-17

The objective is actual committed output tokens per common HTTP window versus
complete-request POST-to-DONE latency. Correctness is a gate, not an axis or a
tradeoff. This preparation produced **zero new performance points**.

## E028 independent numerical and owner audits

`audit_model.py` reuses the byte-pinned original 37-request checker. It derives
eligible refill transitions and rebuild counters from independently checked
schedules, checks all 64 layers/128 full arrays, and consumes exactly one matching
refill event. The original logits, staging, ahead, retirement and release gates
remain intact. Recorded comparison flags are audited; this CPU audit does not
rerun GPU arithmetic or manufacture a model PASS.

`audit_model_owner.py` checks the real new nested directory and CLI, 157 original
preflight entries plus five copied adapter sources, direct-child exit/reap,
original pressure/resource gates and actual restoration evidence. It does not
rearrange the files into a fictitious legacy run.

Root rehashed all actual prepared sources and runtime binaries (241,872,766
binary bytes), without mocking: [preflight result](actual-cpu-preflight-audit.json).
Weight shards were not rehashed; that remains work for the model child.

The actual new cold-controller journal was prepared **read-only**. Its only event
is `prepared_v2_read_only`, state `PREPARED`; the experiment owner directory does
not exist. The new audit correctly returns `INCOMPLETE`, `qualified=false`,
`ledger_opened=false`: [result](actual-prepared-audit.json),
[unaltered journal snapshot](prepared-journal-snapshot.json). The snapshot is
historical evidence, not a substitute for the live journal when operating the
controller. No stop, restart, unload, model load or GPU work occurred. The
previously asked cold-removal/restoration permission is still pending; the
existing pause/resume authorization does not authorize cold removal.

## Lithos official concurrent HTTP client

The adapter preserves the frozen finite closed-loop scheduler while using real
`/api/chat` request bodies and NDJSON records. Raw response bytes, original
request bytes/body hashes, usage counts and receive timestamps remain available.
A worker refills at the actual unique `done:true` record, while a separate task
drains and closes the response. Cleanup shares one deadline and closes the HTTP
client only after responses. Extra, duplicate or truncated terminal data rejects
the point. C is actual HTTP outstanding, including official FIFO queueing; it is
not a claim of concurrent GPU generations.

The count contract requires a separate raw-ID qualification: Lithos reports
`len(tokens)` including the terminal EOS. Only the qualified normal-EOS stop
subtracts one; output-cap length is unchanged. Text is never retokenized to invent
output counts. This narrow client supports the fixed-D uncached contract; the
external controller must still verify every warm/formal Generation log has zero
cached prompt tokens. Client count eligibility alone is not an admitted Pareto
point or a cache-qualification receipt.

## Review history and reproducibility

[Initial CPU run](cpu-tests.json) passed 60 tests: numerical audit 11, owner audit
14, Lithos client 13 and frozen base client 22. Independent source review then
found a missing cancellation case: after a valid EOF, a cancelled response closer
could escape cleanup's error list and incorrectly leave the group eligible.
That initial run is retained as evidence of the test gap, not final acceptance.
The final correction/review and focused rerun are recorded beside this file.

These are machine-pinned M5 harness tests. The owner tests require the existing
local CPU-prepared source fixture; they are not portable general CI checks.
Run the commands recorded in the receipts from this M5 checkout. No benchmark
server or socket listener is created by these tests; HTTP uses MockTransport.

Full-model E028 execution, actual same-route HTTP/raw-ID qualification and fresh
uninstrumented OFF/ON timing remain pending. Current admitted chart remains
[v5](../../../docs/optimization-guide/progress/2026-10-17-v5/README.md).

Final acceptance: [correction tests](final-correction-tests.json) record 15 repaired
Lithos tests PASS. Both new negative tests reject the exact prior source, whose
hash-verified bytes are retained in [pre-fix](pre-fix/manifest.json). The unchanged
11 numerical-audit, 14 owner-audit and 22 frozen-client tests remain passing from
the initial run: **62 tests in the final selected set**. The focused rerun did not
repeat unaffected suites. [Root review](root-review.json) binds final sources,
independent reviews, fetched M4 ancestry and the unchanged running service.
These results qualify the CPU tooling only; no model or performance PASS follows.
