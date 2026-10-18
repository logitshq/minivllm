# Combined graph + GDN: proposed C2/C4 qualification domain

Source preparation only. No C2/C4 server, model, numerical replay, HTTP capture or performance run has occurred. The six fake CPU tests are written but not run. The current C1/C8 formal sources and their registered closures are unchanged.

The domain is exactly **C2 and C4**, with **N=max(8,2C)=8** requests per C. Each original D8 fixture appears once, with the unchanged `/v1/completions` body, full prompt text/IDs, normal EOS and cap512. HTTP requests must actually overlap at the declared C; the independent half-open POST→DONE interval sweep must have peak C. This is separate from requiring actual decode batch B=C in the recorded scheduler trace. If that capacity is not exercised, report incomplete domain coverage rather than infer it from `max_seqs` or from the C8 result.

`domain.py` is the first small reusable piece: strict new-domain settings, frozen request bodies (only client IDs change), and a one-C independent model-pair constructor. Its default CLI only describes the scope. `load_pair_into` takes explicit runtime dependencies and appends every returned engine before further checks, preserving ownership if later construction fails. It imports no backend and has no release authority; the future qualifier must own the original 1800s alarm, source/model checks and `finally` observer/model cleanup. The original base-plan SHA stays `21ba52…47fb`; a distinct C2/C4 run plan records the real capacity. We will not rewrite the original plan and call it an old qualified domain.

## Why changing an enum is insufficient

| Frozen boundary | Actual constraint | Minimal new-domain adaptation |
|---|---|---|
| `gdn_graph_combined/http_server.py:108` | Candidate ON accepts max_seqs only 1/8 | New entry with C2/C4 argument and actual source identity; reuse frozen `main`, `BatchWorker`, handler, installation and observers in an explicit dependency namespace. No `__file__`/`HERE` spoof or donor-global mutation. |
| `pareto_realistic/http_qualification.py:89` | Jobs and expected identity assume C1/C8 and the old capacity map | New request IDs and explicit max_seqs=C; body values remain the frozen helper's eight templates. |
| `pareto_realistic/qualify_native.py:533,577` | Exact old capacity map; pair loading triggered only for D-C1/D-C8 | New single-C orchestration using `domain.load_pair_into`. Preserve the original pre/post model/source checks and all reference/candidate flags; no repeated generic workload loop. |
| `pareto_realistic/http_replay.py:25` | Recorded-domain admission only 1/8 | Small separate replay version, with AST parity for the full loop except the exact C2/C4 admission and B=C coverage checks. No token/schedule injection. |
| `pareto_realistic/audit_qualification.py:77` plus `audit_http_replay_v2.py:122` | Ancestor constructs eight free slots for any name other than D-C1; HTTP subclass changes C afterwards | New independent HTTPCase initialization must set `free=list(range(C-1,-1,-1))` before accepting any steps. Reuse the frozen FCFS/retirement/full-state/ahead methods. Merely changing `self.C` is incorrect. |
| `gdn_graph_combined/http_audit.py:56,191` | Phase allowlist and 8/16 count binding | New C2/C4 binding/cursor with N=8; exact actual source/runtime/capture/replay refs, unchanged mathematical checks. |

## Minimum remaining files and direct reuse

1. **`http_server.py`**: the new small entry above. Reuse frozen combined `main` through explicit `arguments/source_pins/identity_fields` dependencies. Identity records the real new entry and the unchanged donor. Qualification remains both candidates ON with distinct schedule trace and route-event files; formal use remains unavailable until this domain is qualified.
2. **`http_capture.py`**: construct C2/C4 spec/jobs/identity and inject only its validators/startup contract into the frozen capture coroutine. Reuse actual native transport, initial barrier, immediate refill, raw SSE chunks/timestamps, deadline/cancel/close/reap logic. No new HTTP client.
3. **`qualify.py` + small replay-domain function**: one fresh reference/candidate pair at C, source/model verification before and after, native Comparator/Observer/state/pair_calls, and frozen combined `adapted`/`CapturedRoutes`. The runner installs the candidate before Observer construction. Use the joined captured *arrivals and prompt/cap inputs* only; actual engines compute the schedule and next tokens. Recorded schedules and IDs remain comparison evidence.
4. **`audit.py`**: reuse native HTTP full-state numerical methods and combined route/TraceGraph reconstruction. Fix initial free-slot capacity explicitly, then require all eight UUIDs admitted/completed, original128 state arrays/full-vocabulary logits/finite/bit checks, before/after retirement and ahead reuse. Each sidecar independently follows semantic calls; no one-call-per-step or 48 Python hooks per compiled submission assumption. Reuse original raw SSE/full prompt/EOS/text/count and actual client interval checks.

The original `http_join.join` already admits positive C and reconstructs FCFS from true events; it needs no modification. Likewise the original native client, cleanup, native engine, candidate mathematics, graph function and cache layouts remain unchanged. The existing combined four-phase owner's resource, direct-Popen, cancellation, intent/reap and restore functions can be reused in a small later lifecycle binding for **c2-capture → c2-replay → c4-capture → c4-replay**. Its first no-Popen path must use `c2-capture`, not a forged `c1-capture`. No new 500-line owner or new resource algorithm is proposed. The predecessor must be the latest actual closed/restored cycle at registration time; no future hash or current service identity is invented here.

## Fixed candidate scope and evidence

- Route graph guard remains **B1..8**, used257..1536, with its existing 1023/1536 route-dependent bound and physical backing checks. We do not extend its shape or context domain; max_len stays2048.
- GDN prework remains **B1 only**. B2/B4 execute the original native GDN fallback; a tail B1 may hit prework. These are not qualifications of the separate batched-GDN candidate.
- Original native short/fuse/gates settings remain ON, norm reuse OFF, E028 OFF, target-only. Reference remains eager/fuse/gates OFF with compared prior ahead discarded. No parameter replacement, buffer reuse across independent models or warm compile cache migration.
- Require at least one actual compiled long-route call and actual B=C decode coverage per C. Record all observed B/bound/used domains; do not require an unvisited bound or claim that every future arrival schedule has been proven. The actual state/logit comparisons, not argmax alone, decide numerical equality.
- Keep the original two-model and owner resource/time budgets. A timeout/resource refusal remains a failed or incomplete qualification; it is not permission to relax limits.
- C16/C32 are deliberately absent. New C2/C4 numerical + HTTP + resource/restoration PASS must precede any observer-OFF performance points at these C values.

## First focused CPU evidence, before a device plan

The six current source-only tests cover exact body preservation, rejection of C1/C8/C16/bool as the new domain, unchanged old-plan bytes, actual constructor C and eager/candidate flags, independent weight handles, and retained partial ownership on Observer failure. Once the remainder is built, add one real NumPy-Scheduler pipeline per C (same existing `test_qualify_cpu.Engine`), admissions/reuse/retire/ahead, wrong initial free-slot and captured-token negative cases; AST parity protects the copied replay loop. These fake checks would validate mechanisms only and cannot grant model/HTTP/performance qualification.

## Source-only extension,2026-10-18

`http_server.py`, `http_capture.py`, `pair.py`, `replay.py`, `qualify.py` and
`audit.py` now exist but have **not** had their new entry/control-flow suite run.
Only the earlier six domain tests have actual CPU results. No C2/C4 server,
model pair, HTTP request, numerical result or timing point has run.

The independent audit reuses the original raw-wire, logits/state receipt,
FCFS/scheduling, graph tracing and process-closure algorithms. It separately
fixes the inherited non-C1 eight-slot initialization to actual C2/C4, requires
eight original request bodies, and reconstructs both a full B=C step and a
continuing multirow-to-B1 tail. Producer `domain_coverage` must equal those
independent facts. Actual C2/C4 capture identity is used; no source path or
recorded phase is relabeled C1/C8. The extra capture-adapter layer is traversed
explicitly to the original readiness check.

`audit.PRODUCER_SHA` is UNSEALED. Next: review and focused fake CPU tests for
actual argv/capture construction, real capacity initialization, positive and
negative tail coverage, then source sealing and an externally owned device
cycle. Current ongoing formal C1/C8 timing sources are unchanged.

## Independent source review and prepared regression module

The source review found and fixed one inherited-domain rejection in the new
`audit.py`: the original HTTP interval checker required 16 requests for every
non-C1 value. The new explicit function variant requires `domain.request_count(C)`
(eight at C2/C4). Its clock, cleanup, origin, actual half-open POST→DONE sweep,
and peak-concurrency checks are unchanged. No frozen C1/C8 file was edited.

`test_http_cpu.py` adds **12 prepared test methods, not run**. They cover the
actual server argument parser; real spec construction with eight unchanged
bodies; fake capture through the real cleanup coroutine and independent
readiness/receipt binding; C-slot initialization; full128-state descriptor
checks, actual full B=C and continuing-row B1 coverage; wrong-count/serial
HTTP intervals; frozen replay AST parity outside the explicit domain/coverage
delta; two independent toy NumPy-engine replays; recorded tokens as comparison
targets; and single-pair cleanup after construction, step, or restore failure.
The model/OS/alarm boundaries in the cleanup tests are explicit fakes. The
descriptor fixtures provide no model values or device qualification.

Run the whole new module only after the active formal benchmark closes, with
the existing guarded CPU process. Its imports include the prior NumPy CPU
Scheduler fixtures and existing fake capture helpers. Do not select only
passing methods. The earlier six domain tests remain separate prior evidence;
these new 12 tests are prepared source only. Producer sealing, a real device
run, full independent audit, resource closure and restoration are still pending.
