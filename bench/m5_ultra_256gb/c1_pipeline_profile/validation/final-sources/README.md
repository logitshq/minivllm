# C1 pipeline diagnostic: CPU prepared, device unrun

The current native C1 gap is mostly after first output. This module measures
existing Python boundaries while preserving ahead generation. It does **not**
measure exclusive GPU kernel time, grant numerical qualification, or produce a
formal throughput–latency point. All instrumentation is default OFF.

`instrumentation.install(runner, enabled=True)` must run on a fresh target-only
runner after any E024 installation, before requests and other observers. OFF
returns `None` without inspecting the runner. Supported real sources are pinned
native eager/short decode and the existing E024 long decoder. The returned
session reuses the exact original `ModelRunner.run` code object with a **private**
globals dictionary. Only that dictionary's `mx.eval`, `mx.async_eval`, and
`sample` bindings are tapped; the shared MLX module is unchanged.

The in-memory `C1-pipeline-profile-v1` events contain monotonic `start_ns/end_ns`,
thread ID and scalar metadata:

| Event | What the interval actually measures |
| --- | --- |
| `runner_step` | Original runner wall time, prior ahead key match, actual current/future calls, retained successor |
| `decode_graph_submission` | Existing decoder Python call, B/used and requested length domain; domain alone does not prove guard eligibility |
| `compiled_callable` | Actual callable invocation; first use may include tracing/compilation, later use can still encounter a new specialization |
| `callable_family_created` | Python callable dictionary insertion, not a machine graph/kernel count |
| `sample_graph` | Existing sampling graph construction |
| `existing_async_eval` | Existing async submission call, not device completion |
| `existing_eval` | Existing eval call wall time, including any host/driver work and wait |

The original current-versus-successor call sequence, device token dependencies,
state updates and final host token conversion stay intact. No extra array
access, `eval`, `synchronize`, token `.tolist()`, graph execution or kernel is
introduced. Original ahead arrays are not retained for observation. A bounded
200,000-event log stops recording on overflow while preserving the program;
the resulting receipt is `INCOMPLETE`. Exceptions retain their type and propagate.

`delivery.py` is an optional independent layer. Before constructing the server,
call `delivery.install(None, worker, donor, tokenizer, recorder, enabled=True)`
and use its `handler_class(...)` in place of the donor factory. In the existing
load-worker callback, after engine creation, call `delivery_scope.attach_engine(engine)`
before `worker.initialize`. The runner installation can share that same recorder
through `instrumentation.install(..., recorder=recorder)`. This fits the existing
threaded server initialization order without moving model loading to another
thread. The original worker submit code constructs its original Job, replacing
only that Job's queue with a transparent tap **before** pending admission.

Optional events are `engine_step`, `delivery_queue_put/get`, `text_decode`,
`http_write`, `http_flush`, and `http_handler`. Writes/flushes are forwarded once;
SSE bytes, tokenizer arguments/results, queue items and exceptions are unchanged.
Queue `get` includes waiting for production. Write completion means the original
write returned, not that the remote client received the token. The log contains
request IDs/ordinals but no tensor references or response text.

After the worker and handlers have actually stopped/drained, restore the delivery
scope, then the runner scope; call `recorder.write_once(NEW_PATH, sources=..., drained=True)`
once. Supply the complete wrapper/adapter/native source map and retain the external
owner's source/runtime, process and HTTP receipts. `drained=True` is a caller
assertion, not a process-reap certificate. No server entry, shutdown protocol,
resource owner or live execution is added here. SIGTERM cannot be assumed to flush
this memory log; a future diagnostic wrapper must arrange explicit drained closure.

Do not add nested spans together: compiled calls lie inside decode submissions,
which lie inside runner/engine steps, and delivery runs on another thread while
ahead GPU work may already be executing. Compare the construction/submission and
existing-eval intervals with delivery/queue timelines; a shorter submission that
leaves end-to-end timing unchanged may have been hidden by overlap, but this alone
does not prove the amount of overlap. Required first device experiment: identical
workload/configuration with profile OFF versus runner-only ON (and separately full
delivery taps) to quantify perturbation. Keep all of those diagnostic runs outside
formal Pareto evidence.

CPU validation: first whole-module run **11/11 PASS**, no failures/skips, 0.056 s
(tool chunk `c842ec`). It AST-loads the frozen native runner with explicitly fake
arrays/backend and compares OFF/ON outputs plus exact model/eval/async/tolist call
order. Other tests cover ahead invalidation, actual callable counts, exception and
reference cleanup, overflow, private E024 module binding, and actual native worker
submit with fake engine/transport. The command used import/effect guards rejecting
MLX/NumPy/vendor/native package imports, subprocess, network and signals. This is
CPU routing/observation evidence only. After small source-identity hardening, the
same guarded whole module passed again: **11/11**, 0.053 s, zero errors/skips.
No frozen source is modified. Run the fake suite with:

```sh
python3 -B -m unittest bench.m5_ultra_256gb.c1_pipeline_profile.test_instrumentation_cpu
```
