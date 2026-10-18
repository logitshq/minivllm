# Independent physical and metadata monitoring

`live_monitor.py` is a new stdlib helper. The frozen `physical_guard.py` and
`glm_flash_cycle/owner.py` are unchanged. It is not yet registered or integrated
into a real GLM run; CPU results establish mechanism behavior only.

The original serial loop could spend 0.244 s sampling physical memory and then
1.598 s running metadata commands, violating its 1 s physical gap. This helper
uses two threads in the actual owner process, preserving the owner PID seen by
the frozen physical sampler. Each lane runs one callback at a time.

| Registered quantity | Physical | Metadata |
| --- | ---: | ---: |
| Maximum previous-end to next-start gap | 1 s | 5 s |
| Scheduling target for that gap | 0.5 s | 2.5 s |
| Maximum callback execution time | 5 s | 5 s |

The target leaves scheduling margin. These are observed bounds, not real-time
guarantees; any measured violation rejects. Maximum start-to-start spacing can
include both the callback duration and its following gap. The metadata lane is
an explicitly different contract from the old checkpoint-after-every-sample
contract. A new plan and corresponding offline verifier must state this change.

The API is `monitor(child, out=..., physical=..., metadata=...,
work_deadline=...)`. A callback receives `(index, commands, absolute_deadline)`.
Use the frozen `guard.measure(..., run=commands.run)` and `guard.observe(...)`
in the physical callback. Use frozen `owner.checkpoint(commands, ...,
deadline=absolute_deadline, record=...)` in the metadata callback. The callback
must save its original raw receipts, retain the original full identity checks,
and pass its remaining deadline to every command. An instance-local `System`
whose `read` is `commands.read` can retain the frozen `System.identity` method;
no shared module or subprocess globals need modification.

`Commands` retains an intent before Popen, the returned probe PID, command
start/end/elapsed, original output and actual wait outcome. It uses stdlib
`communicate(timeout)` for bounded reads. On failure/cancellation it attempts
TERM then KILL of that returned read-command handle, each with bounded reads and
waits. The unrelated-process `ps ... args=` output remains hash/size-only on
disk, just as in the frozen checkpoint. The actual raw text is returned only
to that checkpoint's parser. These are read-command receipts, not model-child
release evidence.

Model child exit does not turn a failed physical or metadata callback into a
success. An in-flight callback finishes under its existing deadline; failure
remains `INCOMPLETE`. Any exception, callback overrun, unknown Popen outcome,
unreaped read command, missing lane sample, or unjoined worker raises
`MonitorError`, with a retained `monitor.json`. Joining both threads has one
aggregate 7 s bound. Python cannot forcibly stop an uncooperative callback;
the report identifies any surviving thread/command, and the outer owner must
refuse to certify complete cleanup. The helper never signals the model child.
Its `poll()` may reap a child internally, but the outer owner still owes the
original final wait, exact PID absence and cold-service recovery evidence.

Integration must also pin this source, retain the original power/thermal,
service-absence, physical/swap and model identity gates, and independently audit
the two receipt sequences. A monitor `COMPLETE` is only a scheduling/command
closure result. It does not establish model fit, full physical safety, numerical
quality or performance.

The focused CPU suite uses fake processes and synthetic physical output, with a
guard against real subprocess/network/signal/backend operations. It covers slow
metadata concurrent with continuing physical samples, successful-exit races,
callback deadlines, unknown launch, failed cleanup, cancellation, per-command
timing/redaction, and both frozen callback APIs. Full first-attempt outputs and
source snapshots are retained under `.artifacts/m5-ultra-256gb/glm-dual-monitor-cpu-v1`.
The only follow-up source change replaces an ambiguous "model reaped" flag
with `model_child_release_proven=false`; `Popen.poll()` can itself reap, while
the helper cannot prove the owner's full release protocol.
