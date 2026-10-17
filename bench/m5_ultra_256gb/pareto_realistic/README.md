# E023: realistic throughput–latency inputs and HTTP harness

CPU preparation only; no model was loaded, no listener was started and no E023
performance or numerical qualification has run. Correctness is a prerequisite
to the throughput–latency frontier. The candidate execution protocol is in
[`pareto-next-workloads.md`](../../reviews/pareto-next-workloads.md).

`fixtures/tasks-v1.json` defines eight self-contained Python coding tasks before
seeing any generated answer or speed result. `prepare_fixtures.py` freezes their
messages, exact chat-template text and IDs with the pinned Qwen3.8-27B tokenizer.
`fixtures/prepared-D-v1/` preserves the actual CPU output and receipt. The input
lengths are **978, 886, 926, 921, 954, 935, 903, 958 tokens**, with a 512-token output
cap, thinking disabled and normal EOS. These are naturally sized prompts, not
the exact 1024±32 synthetic bucket suggested in the earlier candidate plan. No
padding, truncation, answer selection or replacement of early EOS is allowed.

The fixture SHA256 is
`6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`.
The preparation checks both tokenizer APIs and decode round trips; this proves
inputs only, not correct execution by an engine or semantic correctness of the
model's generated program.

`native_server.py` keeps one engine worker, drains arrivals before each step and
uses the existing scheduler to limit actual admission. Requests have separate
delivery queues and retire independently. Final SSE retains raw token IDs,
committed IDs and a separately identified terminal EOS; usage excludes only that
terminal token. Qualification tracing is opt-in and must be disabled for formal
timings. The current core has no cancellation API: a cancelled admitted request
invalidates the entire benchmark server for controller cleanup. This is a small
measurement adapter, not a general serving cancellation implementation.

`client.py` uses an asynchronous finite work queue. At concurrency C, formal
N=max(64,8C); a finished request immediately releases its slot for the next frozen
job. The complete group window spans the first POST to the last DONE. Latencies
include queueing; throughput uses actual committed tokens, never N×output cap.
Absolute request/group deadlines retain partial failures, without retries.
An independently checked qualification receipt is required before usage is
trusted. It binds every fixture to the full canonical request-body SHA and path,
and the exact request-ID to fixture mapping. Unqualified stop/minimum-output/EOS
overrides are rejected. The client does not grant model, numerical or scheduling
qualification; the future controller must verify receipt provenance, sources and
actual loaded engine identity.

The endpoint configuration, qualification replay/controller, resource lifecycle,
external-engine adapters and complete C1–64 runtime plan are still pending. No
prepared fixture or mock result is eligible for the Pareto chart. E020 and E022
remain unchanged and their historical measurements are not pooled with E023.

CPU-only checks (no socket or MLX):

```sh
.venv/bin/python -B bench/m5_ultra_256gb/pareto_realistic/test_native_cpu.py -v
.venv/bin/python -B bench/m5_ultra_256gb/pareto_realistic/test_client_cpu.py -v
```

The author's runs and root's independent run passed 18 server and 22 client
tests. `cpu-v1/receipt.json` saves root's complete outputs, five Python source
SHAs and seven verified input pins. The independent review's missing body binding
and raw-EOS cap boundary are fixed and covered by negative tests. This receipt is
CPU-only and does not authorize a numerical or performance claim.
