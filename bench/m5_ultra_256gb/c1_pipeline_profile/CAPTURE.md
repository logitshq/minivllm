# C1 pipeline diagnostic capture

This remains device-unexecuted preparation. On 2026-10-19, all fifteen focused
CPU capture tests passed in next-candidate-cpu-v2 (the combined suite retained
one unrelated GLM exception-expectation error). The existing server/instrumentation/delivery
sources are unchanged. No diagnostic result, model qualification or performance
point is supplied by this work.

## Minimal integration

The external cold/resource owner imports `capture.py` from its registered source
path and supplies one previously prepared, independently qualified **combined C1
formal specification** as `{path, sha256}`. That specification fixes the native
source/model/runtime, graph ON, the chosen GDN OFF/ON arm, D8 fixture bodies,
normal EOS, cap512, original socket settings and count qualification. Choose one
GDN arm and keep it identical for all three diagnostic modes.

```python
spec = capture.prepare(
    formal_spec_ref, mode,  # mode is "off", "runner", or "full"
    output_directory=fresh_absolute_trial_directory,
    drain_token=owner_generated_nonce,  # printable ASCII, at least 32 characters
    server_sha256=registered_diagnostic_server_sha256,
    drain_seconds=30,
    worker_join_seconds=10,
)
capture.validate(spec)
result = await capture.run(
    spec, fresh_absolute_trial_directory,
    checkpoint=owner_checkpoint,
    backend=owner_tracked_backend,
)
```

The optional backend delegates the frozen native `System` methods, implements
`group(spec, stage, output)` and `drain(port, nonce, seconds, output)`, and tracks
the actual direct server `Popen`. It must retain a returned handle before any
identity/receipt operation can fail. An unknown spawn outcome is unresolved;
absence of a child receipt is not proof that no child started. This adapter does
not stop or restore launchd services. The owner must bind full source/runtime/
weights inventories and exclusive physical admission, own its hard deadline and
TERM/KILL/wait fallback, then independently reconcile exact PID and port absence
before restoration or the next fresh process.

`capture.py` with no arguments only describes the API. It has no standalone
device-start CLI because an external lifecycle checkpoint is mandatory.

## Fixed request and closure protocol

Each fresh server receives exactly eight warm requests and eight diagnostic
requests, C1 sequentially. The original queue, D512 bodies, request IDs, raw SSE
bytes/chunk clocks, full prompt IDs, UUIDs, committed IDs, EOS, usage and text
decoder remain those of the frozen native client/count gate. The saved second
group is `admission_probe`, with `formal_point_eligible=false`.

The original count checker has a fixed eight-request warm gate. Its reuse here
is explicit: a read-only validation view selects that gate for both groups;
the saved diagnostic phase, original bodies, bytes and timing records are never
rewritten. Each raw-check receipt names both the actual phase and the validation
view. Reported metrics are diagnostic observations, not Pareto points.

Only after both groups have received their unique normal DONE and completed
client cleanup does the adapter send the nonce-protected empty POST to
`/_diagnostic/drain`. A rejected 409 control request may be retried within one
absolute drain deadline, because DONE receive can precede a handler's finally.
Inference requests are never retried. Each control response/error is retained.

Success requires the accepted drain, server natural exit with return code zero,
actual wait and PID/port absence, and a complete server `closure.json`. Its 16
ordered UUIDs and normal-DONE/handler-exit clocks must match the retained client
responses. Runner/full also require a closed, non-overflowed, error-free profile;
OFF must have no profile. TERM/KILL, incomplete control-client cleanup, missing
closure, nonzero exit or late callback cancellation makes the result FAIL.

The callback labels are `pre-spawn`, `spawned`, `starting`, `ready`, `warm`,
`after-warm`, `diagnostic`, `after-diagnostic`, `draining`, and finally
`after-reap` or `cleanup-incomplete`. `starting`, active-group and drain polling
use the supplied resource cadence. The callback must settle any in-flight
resource read when cancellation arrives; this capture cannot safely reconcile
an external probe that keeps running after its callback returns.

All modes retain shared admission/DONE/drain bookkeeping. Only profile mode,
fresh output directory and nonce differ. `numerical_qualified` and
`performance_qualified` remain false even on diagnostic PASS. Python spans
overlap submission, asynchronous work, waits and delivery; they are not exclusive
GPU time. The drain proves no further worker/handler submissions, not that every
speculative ahead GPU operation completed at the local drain timestamp.

## Focused CPU source coverage

The new test source covers mode/config parity, invalid bindings, default inert
CLI, real frozen raw-client/count-checker use through `httpx.MockTransport`,
natural closure without signals, the 409 handler-finally race, bad raw stopping
before drain, incomplete/nonzero closure, UUID/profile/flush corruption, repeated
cancellation with TERM/KILL/wait, and strict surviving-PID rejection. Synthetic
server closure receipts do not establish actual device behavior. Run only under
the root's guarded CPU runner after the active formal timing window closes.
