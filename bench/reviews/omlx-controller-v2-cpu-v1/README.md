# Cold-service recovery v2: CPU and read-only progress

The new controller and service-absent experiment owner have passed root CPU
review. This is preparation, not a successful cold restart or complete model
qualification. No stop, bootstrap or load POST was executed. The existing
pending cold-restart approval is not supplied by these tests.

Root ran 16 controller tests, 30 owner tests and 9 separate diagnostic-bridge
tests: 55 PASS. Every test uses CPU fakes or static source checks; output and
exact source identities are retained in [receipt.json](receipt.json) and the
three stdout/stderr pairs. The old controller, numerical plan, qualifier and
old numerical auditor remain unchanged. The new independent review is stored
alongside this receipt; its scope must not be mistaken for a real restart.

The v2 protocol covers a cooperative journal lock, durable pre-spawn attempt,
separate fsynced spawn intent, verified child reap, and an explicit
NO_CHILD_STARTED terminal for caught failures strictly before spawn intent.
The owner verifies original PID, launchd job and port absence at each resource
phase and immediately before spawn. Its absence records use an independent
index and exclusive writes, so a failed preflight cannot be overwritten by
postflight. The original 37-request numerical workload is retained.

Recovery retains every raw physical sample and every complete bounded log
window with hashes. During its single load POST, the parent samples resources
while directly owning a bounded HTTP client child. A resource/monitor failure
reaps only that client; it does not cancel or kill the service-side loader.
The journal retains LOAD_INTENT/post_monitor_failed and refuses an automatic
success or repeated POST. These are observation bounds, not OS hard caps.

## Actual read-only checks

The frozen v1 `prepare` succeeded against the live original service. Its
[original journal](read-only-v1-prepare-journal.json) remains PREPARED: original
PID 74384/start/launchd/listener, oMLX 0.7.0, and the exact existing loaded
Qwen3.8-Flash-Next-oQ6e-mtp model were observed. Global settings were statted,
not read for credentials. This is a v1 diagnostic journal, not an executable
v2 recovery journal; never reuse or migrate it into the new lifecycle.

Root separately called v2 `System.resources()` twice, five seconds apart,
without running any controller phase, HTTP request, signal, model or GPU work.
Both actual self-PID `footprint --sysFootprint` results parsed successfully.
Complete commands, raw JSON, parsed samples and service identity before/after
are retained under [read-only-physical](read-only-physical/receipt.json).

| Read-only observation | First | Second |
|---|---:|---:|
| System physical footprint (GiB) | 164.595901 | 164.691055 |
| Pressure | 2 | 2 |
| Reported swap (MiB) | 445.06 | 445.06 |
| Installed RAM (GiB) | 256 | 256 |

The recorded sample interval is 5.051133 seconds. The original
service remained the same identity in S state. These observations do **not**
pass the pressure-1 / at-most-40-GiB cold admission gate; the real gate itself
was not invoked. Interface validation does not grant experiment admission.

## Remaining execution work

1. Freeze a new wrapper plan with the final controller/owner source hashes and
   new journal/owner paths. Perform the actual v2 read-only prepare and resource
   checks under that plan; the already tested v1 prepare is not a substitute.
2. Add the separate service-absent independent numerical/lifecycle auditor.
   Reuse frozen `CaseAudit`, `stream_ledger` and `audit_pins`; adapt only the
   lifecycle/resource evidence and use the new `launch.json`. The old auditor
   requires a suspended T-state service and T-to-S lease and cannot validate a
   service-absent run. Never fabricate that evidence.
3. Obtain the still-pending explicit cold-restart approval before stop/load.
   Execute and independently audit the complete 37-request numerical workload
   and restoration before considering throughput/latency measurements.

A SIGKILL or power failure after attempt creation but before terminal creation
still requires manual reconciliation. Unknown spawn outcomes also remain
unresolved until independently checked. The controller cannot infer a no-child
proof, fabricate a release receipt or erase the owner directory. The current
protocol handles caught pre-intent failures; it does not promise automatic
recovery from arbitrary crashes. Actual empty startup, loading, replacement
identity and reload log matching have not been exercised by a real restart.
