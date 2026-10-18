# C1 diagnostic HTTP entry

`http_server.py` is a separate, default-OFF diagnostic entry. The existing
`instrumentation.py`, `delivery.py`, combined server and native transport remain
frozen. No model/service run has been performed for this entry. It never grants
numerical qualification or a formal throughput–latency point.

Without arguments it prints the contract and loads no runtime. To run under a
future external owner, use the same explicit model, package, source-manifest,
capacity, allocator, TCP, timeout and candidate-switch arguments as the combined
server, plus:

```text
--profile off|runner|full
--diagnostic-out NEW_DIRECTORY
--drain-token OWNER_GENERATED_NONCE_AT_LEAST_32_CHARACTERS
--warm-requests 0|8
--worker-join-seconds 10
```

The entry requires C1/context 2048, short decode/fusion/gates ON, norm reuse OFF.
`--route-decode-graph` and `--gdn-prework` retain their independent default-OFF
semantics; an OFF/runner/full observer comparison must hold both flags and all
other configuration fixed. The underlying entry argument `--phase formal` means
its qualification observers are absent, **not** that this diagnostic is formal:
the actual top-level identity says `benchmark_phase=diagnostic` and includes the
diagnostic contract. No qualification trace or route sidecar is accepted.

There are exactly eight diagnostic requests. If `--warm-requests 8` is explicit,
the first eight submissions are separately labeled warm, followed by eight
diagnostic submissions. The caller must use the same original D512 normal-EOS
request fixtures in each mode and retain raw client groups and count checks.
This server does not substitute a token/count or numerical auditor.

After the fixed requests reach normal DONE and all responses are closed, the
client explicitly sends a local POST to `/_diagnostic/drain`, with zero-length
body and `X-Diagnostic-Drain-Token` equal to the nonce. Readiness connections are
closed after their original GET handler returns. A drain with incomplete
submissions, unsuccessful DONE flushes, active/pending engine jobs, live
completion/readiness handlers, a wrong nonce, or a runtime error is rejected.
An already-flushed preceding completion may still be leaving its handler when
the next sequential C1 POST arrives; that is allowed, but that handler still
blocks drain until it exits. Unexpected overlapping pre-DONE requests fail the
diagnostic contract.

The original SSE handler performs every response write and flush. A transparent
wrapper records normal DONE only after the original exact DONE bytes were fully
written and the original flush returned successfully. This is a local transport
fact; it does not prove remote receipt. Drain acceptance stops new admissions.
`ThreadingHTTPServer.server_close()` joins all non-daemon handlers. The worker
then stops through a separate Event and a `None` wakeup for the original `tick`;
normal stop never calls `worker.fail`. The only worker-loop difference is the
additional stop predicate. Genuine load/compute/delivery failures retain their
original error and prevent a successful profile receipt.

After actual handler and worker exit, hooks are restored in this order: common
submission tap, full delivery tap, runner tap, combined candidate installation.
Only then is `Recorder.write_once()` called. OFF emits identity and closure
receipts without a profile file. Runner/full emit `profile.json` only after clean
closure; overflow or recorder errors yield `INCOMPLETE`. There is no SIGTERM
flush handler. A blocked connection/model call still needs an external hard
deadline and strict PID reap; this entry adds no resource owner.

`closure.json` records fixed request/warm counts, successful terminal flushes,
per-request phase and wall-clock boundaries, actual joined worker/handlers,
cleanup errors and source pins. Its per-request `handler_exit_ns` is the end of
the completion `do_POST` scope; final handler-thread termination is separately
proved by the server join before closure. `drained` means no further model work
can be submitted by these worker/handler threads. It does **not** assert that
every already-submitted ahead GPU operation has completed. No end-of-run or
per-step `mx.synchronize` was added.

All three modes include common admission/DONE/drain bookkeeping overhead.
Runner and full add their separately reported observation overhead. The spans
are overlapping Python wall intervals, never additive exclusive GPU times.

CPU evidence retains the first 15-test PASS in
`.artifacts/m5-ultra-256gb/c1-diagnostic-http-cpu-v1`. A later source-only
correction admits the legitimate DONE-before-handler-finally interleaving and
adds a focused test. Only that new test was run in the fresh
`.artifacts/m5-ultra-256gb/c1-diagnostic-http-cpu-v2` attempt: **1/1 PASS**,
no denied operation, source unchanged. The earlier 15-test suite was not rerun
on the revised source, and both original source snapshots and receipts remain.
These are fake CPU control/transport tests, not device or service qualification.
