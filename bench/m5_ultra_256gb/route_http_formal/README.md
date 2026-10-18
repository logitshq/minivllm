# E024 observer-OFF paired HTTP timing — source prepared, unrun

This is the next measurement adapter, not a result or executable cold cycle.
The original 37-request/42-direct-call serving gate passed. The new C1/C8 route
capture/replay cycle is still running at preparation time; `prepare` refuses any
missing, incomplete, failed, mismatched-source or unrestored cycle receipt.
No tests or device calls were run while preparing these files.

`formal.prepare(base_plan, C, repeat, *, arm, baseline_qualification,
route_qualification, environment, transport, deadlines)` constructs a fresh spec.
`validate(spec)` reconstructs it. `await run(spec, out, checkpoint=..., backend=None)`
requires an outer owner in the **same process**, with the original server Popen
and real PID. There is no wrapper child or new cold controller here. Default CLI
prints the plan; `--spec PATH --spec-sha256 SHA --out NEW` only validates the spec.

The baseline binding retains the native formal exact three fields
`audit`, `replay_plan`, `child_directory`. The route binding adds `cycle_audit`:
all file references are absolute `{path, sha256}`. The new numerical audit must
be `E024-recorded-route-http-independent-audit-v2`; the whole-cycle audit must be
`E024-route-http-cycle-independent-audit-v1`, independently passing both numerical
histories, source/runtime/resource checks and restoration. The binder checks the
actual terminal files, both audit references, owner files and RESTORED journal.
It does not reread the large steps ledger or claim to repeat its arithmetic.

OFF command and identity stay exactly those of frozen `native_formal.prepare`:
original `pareto_realistic/native_server.py`. ON changes only the entry to frozen
`route_decode_graph/http_server_v2.py` plus `--phase formal --route-decode-graph on`.
Neither accepts `--qualification-trace` or `--route-events`. E028 identity-refill
stays OFF. The per-instance graph guard and out-of-domain fallback stay frozen.
Formal readiness proves installation and identity, not an observed graph hit.

The runner delegates `System`, native cleanup/transport, count contract and raw
SSE/text/EOS metrics to the exact original native implementation. A narrow copy
of its `run` uses the new validator/result schema and checks both observer flags;
cleanup, cancellation settlement, direct-child TERM/KILL/wait and checkpoint
ordering are retained. Imported module globals, `HERE`, and `__file__` are never
patched. Source maps explicitly retain both original and new files. Weight and
runtime payload verification still belongs to the future outer owner.

## Fixed measurement order

| Pair | Concurrency | First process | Second process |
| --- | --- | --- | --- |
| r1 | C1 | OFF | ON |
| r1 | C8 | ON | OFF |
| r2 | C8 | OFF | ON |
| r2 | C1 | ON | OFF |
| r3 | C1 | OFF | ON |
| r3 | C8 | ON | OFF |

Each of 12 fresh processes sends exactly eight warm requests, then 64 formal
requests cycling the unchanged eight D512 fixtures. Thus 96 warm + 768 formal =
864 requests. Warmup is excluded, fully closed before formal admission. There is
no extra graph warmup, retry, replacement sample or exclusion of formal compile
cost. Normal EOS may shorten a request; original raw/committed IDs and usage
rules determine token counts. HTTP concurrency is not claimed to equal GPU batch.

Each point uses committed output tokens divided by the common first-POST to
last-DONE window, with p50/p95 of complete request POST-to-DONE latencies. All
12 process points are retained. Pair comparisons share repeat/concurrency and
adjacent execution, not identical realized arrival schedules. Report six paired
ratios and three-process medians per arm/C; no confidence interval or paired
claim against historical native N1 or external framework runs.

Deadlines are explicit keys `ready, request, warm, formal, close, term, kill,
overall, resource_interval`. The original gate requires overall at least
ready+warm+formal, TERM/KILL 15 seconds each and resource probes at most five
seconds apart. Freeze actual values, 12-phase owner budget, port, source/runtime
pins, immutable predecessor, and all qualification references before execution.

Remaining work is a small 12-phase direct-server owner/recovery adapter and
independent raw timing/resource/restoration audit, then centralized CPU review.
Every trial and point remains `performance_qualified=false` until that audit;
representative schedule qualification is not a proof of every future schedule.
