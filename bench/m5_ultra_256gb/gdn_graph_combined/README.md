# GDN prework + E024 graph: construction adapter

This is a new **unqualified combination**, default OFF. The two frozen candidates
are reused without changes. No GPU/model/HTTP/performance result is supplied by
the adapter. Separate successful GDN and E024 qualifications do not qualify the
composition, and their measured speedups must not be multiplied.

```python
scope = integration.install(engine, gdn_prework=False, route_graph=False)
```

The four Boolean settings are independent. OFF/OFF returns without inspecting
the engine or reading dependency sources. Any enabled setting requires a fresh
target-only context2048 runner, original short compile/fuse ON, gates ON, norm
OFF, distinct 64/48 layer layout, original E028-OFF batch, no requests/staging/
ahead, no observers, and unused short/gates caches. This is an API entry; running
the file prints an inert description and cannot start a device or server.

## Installation and lifetime

Install the 48 instance GDN `_decode` hooks first, the E024 decode dispatcher
second, and qualification observers last. The graph key does not contain the
Python `_decode` identity, so installing GDN into a previously traced graph is
unsafe. This adapter refuses existing caches; it does not try to repair them.

Stop the worker and remove observers before `scope.restore()`. Restore checks
all observable ownership before changing anything, clears this scope's long
graphs, removes the dispatcher, clears short graphs/packed metadata constructed
inside the scope, then restores GDN hooks in reverse order. Layer restore
failures retain a retryable handle. No array evaluation, state migration or
global class patch occurs. The caller must guarantee no concurrent or future
model call, no retained graph/callable, and no queued request; Python object
inspection cannot prove these facts. Use a fresh runner for another experiment.

## What changes and what does not

| Path | Composition |
| --- | --- |
| Prefill / grouped prefill | Original `runner.model`; original GDN `__call__` chooses `_prefill` / `_prefill_batch`, so the instance `_decode` hook is not invoked. |
| Used length <=256 | Original `CompiledDecode`; eligible fused projection path uses `_decode_fused` and bypasses this GDN hook. Native fallback rules remain intact. |
| Used length 257..1536 | E024's original explicit-state graph; bounds 1023 / 1536 and physical KV guard are unchanged. Graph tracing calls the installed GDN hook. |
| Ordinary B1 decode | GDN's existing five native projections and gate branch plus existing fused prework/state kernel, subject to its original guards. |
| B2..8 decode | GDN falls back to its original `_decode`; E024 may still compile that path. A C8 workload can later shrink to B1. |
| Beyond E024's domain | Original short decoder's fallback, with GDN still independently eligible for ordinary B1. |

No prefill, attention, projection, sampling, scheduler, ahead, E028, or kernel
source is edited. That is a source/routing statement, **not** proof that combined
graph compilation preserves every numerical result. Nested gate compilation,
fused Metal primitives and changed graph boundaries still need an exact gate.

GDN Python counters count tracing/eager calls. A reused compiled graph need not
increment them. The old GDN HTTP observer's `48 increments per model call` rule
therefore cannot be copied into combined qualification. Keep E024's actual
compiled-call witness and record per-layer GDN branch selection while each
shape is traced; do not call tracing counts executed GPU-call counts.

Array parameters remain original model attributes and E024 uses `inputs=model`;
there are no new weight packs. This does not establish parameter replacement
correctness for the combined graph. Keep weights, layer structure and flags
fixed during serving. Test changed-value/restore on an idle diagnostic scope;
recreate the model/scope after non-array structural changes. Shape eligibility
remains in the frozen candidates, including B1 GDN versus B1..8 E024. No claim
that a service context of 2048 implies a physical KV capacity of 1536 is made.

## Smallest real qualification path

1. Add a separate thin qualifier around the existing E023 independent eager
   oracle and `route_decode_graph/serving_qualification.py` helpers. Install this
   scope before `Observer`, reuse its 42 changed/restored direct calls and the
   original 37 requests/full logits/128 state arrays/schedule/ahead checks. In
   particular compare B1 and B8 at both E024 bounds, and changed GDN
   weights/gate/conv parameters with already traced graphs. Keep native gates
   ON/norm OFF for the candidate, with the original eager reference contract.
   Reuse `cold_cycle_v3`'s single-child lifecycle only with a genuinely accepted
   current predecessor; a new small predecessor bridge may be needed after GLM.
2. Only after independent numerical PASS, add a thin HTTP server entry using
   the frozen native handler/BatchWorker and this instance installer, with both
   switches explicit in identity. Reuse the GDN/route capture and recorded
   replay math, full prompt/raw IDs/POST-to-DONE transport and actual scheduling.
   Capture → strict server reap → independent replay for C1, then C8. Reuse
   `gdn_prework_http_cycle`'s four-phase owner/resource/recovery methods with new
   source bindings, not a new lifecycle framework. The current frozen factories
   hard-bind old identities and observers; they cannot be passed this source
   unchanged. Captured output tokens never become candidate replay inputs.
3. After that new cycle PASS, observer-OFF formal uses fresh instances and the
   existing native 8-warm/64-formal protocol. Keep OFF, GDN-only, E024-only and
   combined configurations distinct; do not infer formal graph hits or speedup
   from installation, tracing counts, or earlier individual measurements.

The CPU tests cover source identity, four switches, routing, install order,
foreign-wrapper refusal, rollback, reverse cleanup and cache disposal. Their
fake graph is explicitly a Python cache, not an MLX compilation or numeric
oracle. No fullstate/HTTP/performance qualification is granted.
